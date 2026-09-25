# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA fp8 main-KV cache mode selection (VLLM_QSA_FP8_KV).

The QSA sparse GQA kernel reads the paged K/V cache directly, so an fp8
cache needs a read path: "decode" decodes e4m3 in-register inside the
kernel (SM86 gets uint8 pointers + ALU decode); "gather" materializes the
selected rows into a bf16 workspace and runs the kernel unchanged. Both
are A/B-comparable; "" keeps the historical bf16-only guards.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch


def _reload_qsa():
    import importlib

    import vllm.models.qwen4_exp.nvidia.qsa as qsa

    return importlib.reload(qsa)


def test_fp8_mode_env_parsing():
    from vllm.models.qwen4_exp.nvidia.qsa import _qsa_fp8_kv_mode

    for raw, expected in (
        ("", ""),
        ("decode", "decode"),
        ("GATHER", "gather"),
        (" gather ", "gather"),
    ):
        with patch.dict(os.environ, {"VLLM_QSA_FP8_KV": raw}):
            from vllm import envs

            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()
            assert _qsa_fp8_kv_mode() == expected, raw

    with patch.dict(os.environ, {"VLLM_QSA_FP8_KV": "fast"}):
        from vllm import envs

        if hasattr(envs.__getattr__, "cache_clear"):
            envs.__getattr__.cache_clear()
        with pytest.raises(ValueError, match="not one of"):
            _qsa_fp8_kv_mode()


def test_supported_kv_cache_dtypes_includes_fp8():
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend

    assert "fp8" in Qwen4ExpQSAFlashAttentionBackend.supported_kv_cache_dtypes


def test_workspace_layout_is_kernel_addressable():
    """The synthetic gather layout satisfies the sparse GQA kernel's
    addressing: logical token j -> workspace page `row`, offset j."""
    rows, topk, heads, dim = 3, 8, 2, 16
    k_ws = torch.arange(rows * topk * heads * dim, dtype=torch.float32).reshape(
        rows, topk, heads, dim
    )
    # Kernel addressing: page * stride(0) + offset * stride(1)
    #                    + kv_head * stride(2) + d
    page, offset, head, d = 2, 5, 1, 7
    flat = page * k_ws.stride(0) + offset * k_ws.stride(1) + head * k_ws.stride(2) + d
    assert k_ws.flatten()[flat].item() == k_ws[page, offset, head, d].item()


def test_uint8_view_preserves_strides():
    """fp8 caches are passed as uint8 views; the reinterpretration must
    keep shapes and strides so the kernel's address math is unchanged."""
    # vLLM cache layout [blocks, kv_heads, page, 2*D]; transpose+split
    # yields the kernel's [blocks, page, heads, D] key/value views.
    cache = torch.zeros(4, 2, 16, 2 * 32, dtype=torch.float8_e4m3fn)
    key_cache, value_cache = cache.transpose(1, 2).split(32, dim=-1)
    assert key_cache.shape == (4, 16, 2, 32)
    assert key_cache.stride(3) == 1
    k8 = key_cache.view(torch.uint8)
    assert k8.shape == key_cache.shape
    assert k8.stride() == key_cache.stride()
