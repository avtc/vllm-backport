# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Group-size alignment probe for TP-sharded quantized shared experts.

The Qwen3-Next/qwen4_exp shared expert must keep whole quant groups per TP
partition. Upstream detected misalignment only for Quark OCP MX checkpoints;
compressed-tensors pack-quantized checkpoints (AutoRound/INC int4/int6/int8
group weights) impose the same constraint via
``verify_group_size_divides_partition`` but were probed as "no group",
crashing at TP4/TP8 instead of replicating the shared expert.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.utils.config_utils import (
    get_compressed_tensors_group_size,
    get_quantized_linear_group_size,
)
from vllm.model_executor.models.qwen3_next import (
    _should_replicate_misaligned_shared_expert,
)

LAYER = "language_model.model.layers.0.mlp.shared_expert.down_proj"

# Mirrors the AutoRound INT4-Mixed Qwen3.8-Flash-Next checkpoint: dense
# projections int6/g64, routed experts int4/g128, HC int8/g64, misc int8/g128.
CT_CONFIG_DICT = {
    "config_groups": {
        "group_0": {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": [
                "model.language_model.layers.0.mlp.shared_expert.down_proj",
                "model.language_model.layers.0.linear_attn.out_proj",
            ],
            "weights": {
                "block_structure": None,
                "dynamic": False,
                "group_size": 64,
                "num_bits": 6,
                "observer": "memoryless_minmax",
                "scale_dtype": None,
                "strategy": "group",
                "symmetric": True,
                "type": "int",
            },
        },
        "group_1": {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["re:(model|language_model)\\..*\\.mlp\\.experts\\..*"],
            "weights": {
                "block_structure": None,
                "dynamic": False,
                "group_size": 128,
                "num_bits": 4,
                "observer": "memoryless_minmax",
                "scale_dtype": None,
                "strategy": "group",
                "symmetric": True,
                "type": "int",
            },
        },
        # Non-GROUP strategies must never yield an alignment constraint: fp8
        # channelwise (tensor) and 128x128 block scales shard along different
        # axes and are handled by their own schemes.
        "group_2_fp8_channel": {
            "format": "float-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["model.language_model.layers.0.linear_attn.dt_proj"],
            "weights": {
                "block_structure": None,
                "dynamic": False,
                "group_size": None,
                "num_bits": 8,
                "observer": "memoryless_minmax",
                "scale_dtype": None,
                "strategy": "tensor",
                "symmetric": True,
                "type": "float",
            },
        },
        "group_3_fp8_block": {
            "format": "float-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["model.language_model.layers.0.self_attn.o_proj"],
            "weights": {
                "block_structure": [128, 128],
                "dynamic": False,
                "group_size": None,
                "num_bits": 8,
                "observer": "memoryless_minmax",
                "scale_dtype": None,
                "strategy": "block",
                "symmetric": True,
                "type": "float",
            },
        },
    },
    "format": "pack-quantized",
    "quant_method": "compressed-tensors",
}


@pytest.fixture()
def ct_config() -> CompressedTensorsConfig:
    return CompressedTensorsConfig.from_config(dict(CT_CONFIG_DICT))


def test_probe_reads_group_size_for_exact_target(ct_config):
    name = "model.language_model.layers.0.mlp.shared_expert.down_proj"
    assert get_compressed_tensors_group_size(ct_config, name) == 64
    assert get_quantized_linear_group_size(ct_config, name) == 64


def test_probe_reads_group_size_for_regex_target(ct_config):
    name = "model.language_model.layers.3.mlp.experts.down_proj"
    assert get_compressed_tensors_group_size(ct_config, name) == 128


def test_probe_ignores_unmatched_layer(ct_config):
    name = "model.language_model.layers.0.input_layernorm.weight"
    assert get_compressed_tensors_group_size(ct_config, name) is None


def test_probe_ignores_non_group_strategies(ct_config):
    """fp8 tensor/channelwise and block strategies impose no group-size
    alignment constraint on TP partitions (they shard along other axes or
    use their own block checks)."""
    channel = "model.language_model.layers.0.linear_attn.dt_proj"
    block = "model.language_model.layers.0.self_attn.o_proj"
    assert get_compressed_tensors_group_size(ct_config, channel) is None
    assert get_compressed_tensors_group_size(ct_config, block) is None


def test_probe_matches_module_name_targets(ct_config, monkeypatch):
    """Module-name targets resolve through the caller-supplied module's
    class name, mirroring how the real linear construction resolves them."""
    dict_with_module_target = dict(CT_CONFIG_DICT)
    dict_with_module_target["config_groups"] = {
        **dict_with_module_target["config_groups"],
        "group_4_module_name": {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["Qwen3NextMLP"],
            "weights": {
                "block_structure": None,
                "dynamic": False,
                "group_size": 32,
                "num_bits": 8,
                "observer": "memoryless_minmax",
                "scale_dtype": None,
                "strategy": "group",
                "symmetric": True,
                "type": "int",
            },
        },
    }
    cfg = CompressedTensorsConfig.from_config(dict_with_module_target)

    class Qwen3NextMLP(torch.nn.Module):
        pass

    name = "model.language_model.layers.5.mlp.some_proj"
    # The default dummy nn.Linear does not match a Qwen3NextMLP target...
    assert get_compressed_tensors_group_size(cfg, name) is None
    # ...but the caller's module does.
    assert get_compressed_tensors_group_size(cfg, name, Qwen3NextMLP()) == 32


def test_probe_handles_no_quant_config():
    assert get_quantized_linear_group_size(None, LAYER) is None
    assert get_compressed_tensors_group_size(None, LAYER) is None


def test_probe_env_kill_switch(ct_config):
    """VLLM_CT_SHARED_EXPERT_TP_REPLICATE=0 restores quark-only detection."""
    name = "model.language_model.layers.0.mlp.shared_expert.down_proj"
    with patch.dict(os.environ, {"VLLM_CT_SHARED_EXPERT_TP_REPLICATE": "0"}):
        _clear_envs_cache()
        assert get_quantized_linear_group_size(ct_config, name) is None


def _clear_envs_cache() -> None:
    import vllm.envs as envs

    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()


@pytest.fixture(autouse=True)
def _reset_envs_cache():
    """Leave the envs lru-cache clean so later tests see real env values."""
    yield
    _clear_envs_cache()


def test_probe_kill_switch_default_on(ct_config):
    name = "model.language_model.layers.0.mlp.shared_expert.down_proj"
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VLLM_CT_SHARED_EXPERT_TP_REPLICATE", None)
        _clear_envs_cache()
        assert get_quantized_linear_group_size(ct_config, name) == 64


@pytest.mark.parametrize(
    ("tp_size", "ep", "expected"),
    [
        (1, False, False),
        (2, False, False),  # 640/2 = 320, whole g64 groups: shard
        (4, True, True),  # 160 not a whole group: replicate under EP
        (8, True, True),  # 80 not a whole group: replicate under EP
    ],
)
def test_replicate_decision_for_int6_shared_expert(tp_size, ep, expected):
    assert (
        _should_replicate_misaligned_shared_expert(
            intermediate_size=640,
            tp_size=tp_size,
            group_size=64,
            enable_expert_parallel=ep,
            is_sequence_parallel=False,
        )
        is expected
    )


@pytest.mark.parametrize("tp_size", [4, 8])
def test_replicate_decision_without_ep_raises(tp_size):
    with pytest.raises(ValueError, match="expert parallelism"):
        _should_replicate_misaligned_shared_expert(
            intermediate_size=640,
            tp_size=tp_size,
            group_size=64,
            enable_expert_parallel=False,
            is_sequence_parallel=False,
        )


def test_group_divisible_sizes_never_replicate():
    assert (
        _should_replicate_misaligned_shared_expert(
            intermediate_size=512,
            tp_size=4,
            group_size=64,
            enable_expert_parallel=False,
            is_sequence_parallel=False,
        )
        is False
    )
