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


@pytest.fixture(autouse=True)
def _clean_envs_cache():
    yield
    import vllm.envs as envs

    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()


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


def test_gather_workspace_pads_invalid_columns():
    """The synthetic selection buffer must -1-pad columns past each row's
    valid count (the expand kernel's contract): the sparse GQA kernel only
    masks columns via logical_token >= 0, so identity indices everywhere
    would make the last tile read the untouched workspace memory."""
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("needs CUDA to build the workspace")
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_gather_dequant_workspace

    rows, topk, heads, dim = 4, 8, 2, 32
    device = "cuda"
    logical = torch.full((rows, topk + 1), -1, dtype=torch.int32, device=device)
    logical[:, :3] = torch.arange(3, dtype=torch.int32, device=device)  # count 3
    logical[:, topk] = 3
    k_cache = torch.randn(  # PAGE_SIZE=topk
        rows * 2, topk, heads, dim, device=device, dtype=torch.bfloat16
    )
    v_cache = torch.randn_like(k_cache)
    k_scale = torch.ones(1, device=device)
    v_scale = torch.ones(1, device=device)
    block_table = torch.arange(rows, dtype=torch.int32, device=device)[:, None]
    token_to_req = torch.arange(rows, dtype=torch.int32, device=device)

    k_ws, v_ws, packed, _, _ = qsa_gather_dequant_workspace(
        k_cache, v_cache, logical, block_table, token_to_req, k_scale, v_scale
    )
    # identity within valid count, -1 beyond, trailing count preserved
    assert packed[:, :3].tolist() == [[0, 1, 2]] * rows
    assert (packed[:, 3:topk] == -1).all().item()
    assert (packed[:, topk] == 3).all().item()
    # gathered rows match the cache selection for the valid columns
    torch.testing.assert_close(k_ws[:, :3], k_cache[:rows, :3])
    torch.testing.assert_close(v_ws[:, :3], v_cache[:rows, :3])


def test_gather_workspace_matches_dense_attention():
    """End-to-end gather + sparse kernel equals a dense softmax reference."""
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("needs CUDA to launch the kernels")

    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        qsa_gather_dequant_workspace,
        qsa_sparse_paged_attention,
    )

    rows, topk, heads, dim = 3, 8, 2, 32
    device = "cuda"
    torch.manual_seed(0)
    logical = torch.full((rows, topk + 1), -1, dtype=torch.int32, device=device)
    counts = [5, 8, 2]
    for r, c in enumerate(counts):
        logical[r, :c] = torch.arange(c, dtype=torch.int32, device=device)
    logical[:, topk] = torch.tensor(counts, dtype=torch.int32, device=device)

    # One cache page per row; the row's selections are tokens [0, count).
    k_cache = torch.randn(rows, topk, heads, dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(rows, topk, heads, dim, device=device, dtype=torch.bfloat16)
    k_scale = torch.ones(1, device=device)
    v_scale = torch.ones(1, device=device)
    block_table = torch.arange(rows, dtype=torch.int32, device=device)[:, None]
    token_to_req = torch.arange(rows, dtype=torch.int32, device=device)

    k_ws, v_ws, packed, synth_bt, synth_req = qsa_gather_dequant_workspace(
        k_cache, v_cache, logical, block_table, token_to_req, k_scale, v_scale
    )
    q = torch.randn(rows, heads * 4, dim, device=device, dtype=torch.bfloat16)
    out = qsa_sparse_paged_attention(
        q,
        k_ws,
        v_ws,
        packed,
        synth_bt,
        synth_req,
        use_prefill_config=False,
        kv_fp8=False,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    # Dense reference: softmax over each row's valid selections only.
    # Query head h attends kv head h // GROUP_SIZE (GROUP_SIZE=4).
    q_ref = q.reshape(rows, heads * 4, 1, dim).float()
    ref = torch.empty_like(out, dtype=torch.float32)
    for r, c in enumerate(counts):
        for h in range(heads * 4):
            kv = h // 4
            k = k_cache[r, :c, kv].float()  # [c, dim]
            scores = (q_ref[r, h, 0] @ k.T) / (dim**0.5)
            p = torch.softmax(scores, dim=-1)
            ref[r, h] = p @ v_cache[r, :c, kv].float()
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_sparse_attention_fp8_decode_matches_dequantized_reference():
    """The in-kernel e4m3 decode branch (VLLM_QSA_FP8_KV=decode) equals a
    dense softmax over the dequantized cache."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to launch the kernel")
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    rows, topk, kv_heads, group, dim = 3, 8, 2, 4, 32
    device = "cuda"
    torch.manual_seed(0)
    logical = (
        torch.arange(topk, dtype=torch.int32, device=device)
        .expand(rows, topk)
        .contiguous()
    )
    counts = torch.full((rows, 1), topk, dtype=torch.int32, device=device)
    packed = torch.cat([logical, counts], dim=1)
    block_table = torch.arange(rows, dtype=torch.int32, device=device)[:, None]
    token_to_req = torch.arange(rows, dtype=torch.int32, device=device)

    bf16 = torch.randn(rows, topk, kv_heads, dim, device=device, dtype=torch.bfloat16)
    k8 = bf16.to(torch.float8_e4m3fn)
    v8 = bf16.to(torch.float8_e4m3fn)
    k_scale = torch.ones((), dtype=torch.float32, device=device)
    v_scale = torch.ones((), dtype=torch.float32, device=device)

    q = torch.randn(rows, kv_heads * group, dim, device=device, dtype=torch.bfloat16)
    out = qsa_sparse_paged_attention(
        q,
        k8,
        v8,
        packed,
        block_table,
        token_to_req,
        use_prefill_config=False,
        kv_fp8=True,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    k_ref = k8.to(torch.bfloat16).float()
    v_ref = v8.to(torch.bfloat16).float()
    q_ref = q.reshape(rows, kv_heads * group, 1, dim).float()
    ref = torch.empty_like(out, dtype=torch.float32)
    for r in range(rows):
        for h in range(kv_heads * group):
            kv = h // group
            scores = (q_ref[r, h, 0] @ k_ref[r, :, kv].T) / (dim**0.5)
            p = torch.softmax(scores, dim=-1)
            ref[r, h] = p @ v_ref[r, :, kv]
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
