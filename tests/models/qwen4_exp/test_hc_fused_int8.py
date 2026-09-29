# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused INT8 hyper-connection projection scheme selection.

VLLM_HC_FUSED_INT8 swaps the compressed-tensors WNA16 scheme of qualifying
INT8 HC projections for _HCInt8Scheme, which keeps the pack-quantized weight
as loaded (uint8 view q = value + 128) and runs fused Triton GEMVs at decode
batch sizes. The selection must reject anything but symmetric 8-bit group
quantization whose group size divides the input.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import nn


@pytest.fixture(autouse=True)
def _clean_envs_cache():
    yield
    import vllm.envs as envs

    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()


def _FakeWNA16(num_bits=8, symmetric=True, group_size=64):
    # The eligibility check matches the scheme by class NAME.
    return type(
        "CompressedTensorsWNA16",
        (),
        {"num_bits": num_bits, "symmetric": symmetric, "group_size": group_size},
    )()


class _FakeLinear(nn.Module):
    def __init__(self, input_size, scheme):
        super().__init__()
        self.scheme = scheme
        self.input_size_per_partition = input_size
        self.weight_packed = nn.Parameter(
            torch.zeros(4, input_size // 8, dtype=torch.int32), requires_grad=False
        )
        self.weight_scale = nn.Parameter(torch.zeros(4, input_size // 64))


def test_use_hc_int8_scheme_accepts_symmetric_int8_groups():
    from vllm.models.qwen4_exp.nvidia.hyperconnection import (
        _HCInt8Scheme,
        _use_hc_int8_scheme,
    )

    linear = _FakeLinear(1024, _FakeWNA16())
    assert _use_hc_int8_scheme(linear) is True
    assert isinstance(linear.scheme, _HCInt8Scheme)
    assert linear.scheme.group_size == 64


def test_use_hc_int8_scheme_rejects_other_schemes():
    from vllm.models.qwen4_exp.nvidia.hyperconnection import _use_hc_int8_scheme

    # wrong bit width
    assert _use_hc_int8_scheme(_FakeLinear(1024, _FakeWNA16(num_bits=4))) is False
    # asymmetric
    assert _use_hc_int8_scheme(_FakeLinear(1024, _FakeWNA16(symmetric=False))) is False
    # group size that does not divide the input
    assert _use_hc_int8_scheme(_FakeLinear(1000, _FakeWNA16(group_size=64))) is False
    # unsupported group size
    assert _use_hc_int8_scheme(_FakeLinear(1024, _FakeWNA16(group_size=48))) is False
    # not a WNA16 scheme at all
    assert _use_hc_int8_scheme(_FakeLinear(1024, None)) is False


def test_int8_byte_layout_q_is_value_plus_128():
    """The kernels read weight_packed as uint8 with q = value + 128, LSB
    first in the int32 words - the compressed-tensors uint8b128 zero-point
    format (ScalarType.uint(8, 128)), not two's-complement bytes."""
    import numpy as np

    values = np.array([-128, -1, 0, 1, 127, -5, 42, 7], dtype=np.int64)
    q_bytes = ((values + 128) % 256).astype(np.uint8)
    packed = torch.from_numpy(q_bytes.copy().view(np.int32))
    q = packed.view(torch.uint8).numpy()
    assert np.array_equal(q.astype(np.int64) - 128, values)


def test_hc_fused_env_default_on():
    from vllm import envs

    with patch.dict("os.environ", {}, clear=False):
        import os

        os.environ.pop("VLLM_HC_FUSED_INT8", None)
        if hasattr(envs.__getattr__, "cache_clear"):
            envs.__getattr__.cache_clear()
        assert envs.VLLM_HC_FUSED_INT8 is True


def _quantize_int8(w: torch.Tensor, gs: int):
    """Symmetric group quantization in the uint8b128 layout the kernels read:
    q = value + 128 per group-scaled int8, plus the [N, K // gs] scales."""
    n, k = w.shape
    wg = w.float().reshape(n, k // gs, gs)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(wg / scale).clamp(-128, 127) + 128
    dequant = (q - 128) * scale
    return (
        q.reshape(n, k).to(torch.uint8).contiguous(),
        scale.reshape(n, k // gs).float().contiguous(),
        dequant.reshape(n, k).to(w.dtype),
    )


@pytest.mark.usefixtures("dist_init")
def test_fused_projection_matches_quantized_reference():
    """_project_fused (Triton INT8 GEMVs) must match the unfused path on the
    same dequantized weights for M = 1..4 tokens."""
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("needs CUDA to launch the Triton kernels")
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig
    from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual

    torch.manual_seed(0)
    dev = "cuda"
    hidden, hc_count, rank = 512, 4, 128  # hyper = 2048 (kernel tile width)
    config = HyperConnectionConfig(
        hc_count=hc_count,
        hidden_size=hidden,
        hc_lowrank=rank,
        params_dtype=torch.bfloat16,
    )
    mod = GatedResidual(
        config, use_combine=True, quant_config=None, prefix="test_hc"
    ).to(dev)
    mod._hc_fused = False

    for gs in (64, 128):
        for name in ("input_mix_weight_down", "input_mix_weight_up"):
            linear = getattr(mod, name)
            hc_q, hc_scale, dequant = _quantize_int8(linear.weight.data, gs)
            linear.hc_q = hc_q
            linear.hc_scale = hc_scale
            linear.scheme = SimpleNamespace(group_size=gs)
            with torch.no_grad():
                linear.weight.copy_(dequant)

        for m in (1, 2, 3, 4):
            xn = torch.randn(m, hc_count * hidden, device=dev, dtype=torch.bfloat16)
            with torch.no_grad():
                block_ref, inj_ref = mod._project(xn)
                mod._hc_fused = True
                block_fused, inj_fused = mod._project_fused(xn)
                mod._hc_fused = False
            # The injection is a dot over K=2048 (magnitude ~45, one bf16
            # ulp 0.25) and the block input passes through a sigmoid of a
            # K=128 dot: allow a couple of ulps at each magnitude.
            torch.testing.assert_close(
                inj_fused.float(), inj_ref.float(), rtol=2e-2, atol=5e-1
            )
            torch.testing.assert_close(
                block_fused.float(), block_ref.float(), rtol=2e-2, atol=5e-2
            )


@pytest.mark.parametrize("m", [5, 17, 33])
def test_w8a16_gemm_matches_dequantized_reference(m):
    """The prefill-path Triton W8A16 GEMM (batch > 4) matches a reference
    GEMM on identically dequantized (BF16-rounded) weights."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to launch the kernel")
    from vllm.models.qwen4_exp.nvidia.hyperconnection import _w8a16_gemm

    torch.manual_seed(0)
    n, k, gs = 320, 2048, 64
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    hc_q, hc_scale, dequant = _quantize_int8(w, gs)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    got = _w8a16_gemm(x, hc_q, hc_scale, gs)
    ref = x @ dequant.T
    # Both round fp32 accumulations of the same products to bf16 in
    # different orders; dots over K=2048 reach ~45 where one bf16 ulp is
    # 0.25 - allow a couple of ulps.
    torch.testing.assert_close(got, ref, atol=5e-1, rtol=2e-2)
