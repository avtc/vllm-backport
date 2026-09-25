# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split hyper-connection projections for quantized checkpoints.

AutoRound/INC checkpoints (e.g. Qwen3.8-Flash-Next INT4-Mixed) quantize
``input_mix_weight_down`` and ``input_mix_weight_up`` independently (int8
group-64). Two separately-quantized tensors cannot be stacked into the merged
``input_mix_weight_down_block_inject`` packed weight on load, so the module
must keep the projections split, mirroring the checkpoint layout.
"""

import pytest

from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig
from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual

HC_COUNT = 4
HIDDEN = 64
LOWRANK = 16


def _config() -> HyperConnectionConfig:
    return HyperConnectionConfig(
        hc_count=HC_COUNT,
        hidden_size=HIDDEN,
        hc_lowrank=LOWRANK,
        rms_norm_eps=1e-6,
        hc_per_branch_norm=True,
    )


@pytest.mark.parametrize("use_combine", [True, False])
@pytest.mark.parametrize("quantized", [True, False], ids=["int8", "bf16"])
def test_gated_residual_split_projections(use_combine: bool, quantized: bool):
    """The down/inject projection must exist as separate modules, never as
    the merged ``input_mix_weight_down_block_inject`` linear."""
    quant_config = _fake_quant_config() if quantized else None
    hc = GatedResidual(_config(), use_combine=use_combine, quant_config=quant_config)

    down = hc.input_mix_weight_down
    assert down.output_size == LOWRANK
    assert down.input_size == HC_COUNT * HIDDEN
    assert down.quant_config is quant_config

    up = hc.input_mix_weight_up
    assert up.output_size == HC_COUNT * HIDDEN
    assert up.input_size == LOWRANK
    assert up.quant_config is quant_config

    if use_combine:
        inject = hc.block_inject_weight
        # block_inject stays unquantized: quantized checkpoints leave it bf16.
        assert inject.output_size == HC_COUNT
        assert inject.quant_config is None
    else:
        assert not hasattr(hc, "block_inject_weight")

    assert not hasattr(hc, "input_mix_weight_down_block_inject")
    assert not hasattr(hc, "pad_size")


def _fake_quant_config():
    """Stand-in quant config whose get_quant_method yields the unquantized
    method, so construction runs on CPU while attribute plumbing is still
    observable."""
    from unittest.mock import SimpleNamespace

    from vllm.model_executor.layers.quantization.unquantized import (
        UnquantizedLinearMethod,
    )

    return SimpleNamespace(
        get_quant_method=lambda layer, prefix="": UnquantizedLinearMethod()
    )


def test_no_merged_hc_name_in_weight_mappers():
    """No qwen4_exp weight mapper may emit the merged HC name; the PLE kv
    stacking must survive (both shards share one quant scheme)."""
    from vllm.models.qwen4_exp.nvidia import model as qwen4_model
    from vllm.models.qwen4_exp.nvidia import mtp as qwen4_mtp

    merged = "input_mix_weight_down_block_inject"
    for mapping in (
        qwen4_model._EXTRA_WEIGHTS_MAPPER.orig_to_new_stacked,
        qwen4_model.Qwen4ExpForCausalLM.packed_modules_mapping,
        qwen4_model.Qwen4ExpForConditionalGeneration.packed_modules_mapping,
        qwen4_mtp.Qwen4ExpMTP.packed_modules_mapping,
    ):
        assert merged not in mapping
        assert merged not in str(mapping.values())

    for mapping in (
        qwen4_model._EXTRA_WEIGHTS_MAPPER.orig_to_new_stacked,
        qwen4_model.Qwen4ExpForCausalLM.packed_modules_mapping,
        qwen4_mtp.Qwen4ExpMTP.packed_modules_mapping,
    ):
        assert mapping.get("ple.kv_proj", mapping.get("kv_proj")) is not None


def test_checkpoint_split_names_load_directly():
    """Split checkpoint names must match module attribute names 1:1, i.e.
    the mapper no longer redirects them anywhere."""
    from vllm.models.qwen4_exp.nvidia import model as qwen4_model

    stacked = qwen4_model._EXTRA_WEIGHTS_MAPPER.orig_to_new_stacked
    for name in (
        "hyper_connection.input_mix_weight_down.weight",
        "hyper_connection.block_inject_weight.weight",
        "hyper_connection.input_mix_weight_up.weight",
    ):
        assert name not in stacked
