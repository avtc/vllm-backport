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

import torch
from torch import nn


class _FakeWNA16:
    def __init__(self, num_bits=8, symmetric=True, group_size=64):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.group_size = group_size


class _FakeLinear(nn.Module):
    def __init__(self, input_size, scheme):
        super().__init__()
        self.scheme = scheme
        self.input_size_per_partition = input_size
        self.weight_packed = nn.Parameter(
            torch.zeros(4, input_size // 8, dtype=torch.int32)
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
    """The kernels read weight_packed as uint8 with q = value + 128 (LSB
    first in the int32 words); the reference dequant used by the test below
    relies on that layout."""
    import numpy as np

    values = np.array([-128, -1, 0, 1, 127, -5, 42, 7], dtype=np.int8)  # one int32 word
    packed = np.asarray([values.view(np.int32).item()], dtype=np.int32)
    q = torch.from_numpy(packed.copy()).view(torch.uint8).numpy()
    assert np.array_equal(q.astype(np.int32) - 128, values.astype(np.int32))


def test_hc_fused_env_default_on():
    from vllm import envs

    with patch.dict("os.environ", {}, clear=False):
        import os

        os.environ.pop("VLLM_HC_FUSED_INT8", None)
        if hasattr(envs.__getattr__, "cache_clear"):
            envs.__getattr__.cache_clear()
        assert envs.VLLM_HC_FUSED_INT8 is True
