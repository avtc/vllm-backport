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

from vllm.model_executor.layers.quantization.compressed_tensors import (
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


def test_dummy_module_is_linear():
    """The probe passes a plain nn.Linear for module-name target matching."""
    layer = torch.nn.Linear(8, 8)
    assert get_compressed_tensors_group_size(None, "x", layer) is None
