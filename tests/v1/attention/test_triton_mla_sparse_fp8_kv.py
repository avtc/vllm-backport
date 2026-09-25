# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 KV cache path for the Triton sparse-MLA backend.

Covers the SM86 (Ampere) plan: the KV cache is stored as plain fp8 e4m3 with
a per-tensor scale (written by the shared concat_and_cache_mla op), and the
read path gather-dequantizes only the top-k selected rows to bf16 before the
existing bf16 sparse-MLA kernel. Works for any head size divisible by 64
(512 NoPE GLM-5.3-Flash, 576 NoPE+RoPE DeepSeek-V3.2 style).
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        "Triton sparse MLA fp8 KV tests require CUDA.",
        allow_module_level=True,
    )

from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseBackend,
    dequant_gather_fp8_rows,
    remap_to_gather_rows,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    triton_mla_sparse_attention,
)

DEVICE = torch.device(current_platform.device_type)

HEAD_DIM = 512
BLOCK_SIZE = 64
NUM_HEADS = 4
SM_SCALE = HEAD_DIM**-0.5


def test_supports_kv_cache_dtype():
    """--kv-cache-dtype fp8 is the opt-in: fp8/fp8_e4m3 are accepted,
    other quantized formats stay rejected, bf16 always passes."""
    assert TritonMLASparseBackend.supports_kv_cache_dtype("bfloat16")
    assert TritonMLASparseBackend.supports_kv_cache_dtype(None)
    assert TritonMLASparseBackend.supports_kv_cache_dtype("fp8")
    assert TritonMLASparseBackend.supports_kv_cache_dtype("fp8_e4m3")
    assert not TritonMLASparseBackend.supports_kv_cache_dtype("fp8_e5m2")


def _make_fp8_cache(rows: torch.Tensor, block_size: int) -> torch.Tensor:
    """Quantize bf16 rows [N, head_dim] into a flat fp8 cache, row i -> slot i."""
    n, head_dim = rows.shape
    num_blocks = (n + block_size - 1) // block_size
    cache = torch.zeros(
        (num_blocks, block_size, head_dim), dtype=torch.float8_e4m3fn, device=rows.device
    ).view(-1, head_dim)
    cache[:n] = rows.to(torch.float8_e4m3fn)
    return cache.view(num_blocks, block_size, head_dim)


@pytest.mark.parametrize("head_dim", [512, 576])
@pytest.mark.parametrize("pad", [0, 37])
def test_dequant_gather_fp8_rows_matches_cast(head_dim: int, pad: int):
    """Rows gathered + dequantized from the fp8 cache must equal the plain
    torch cast roundtrip (same e4m3 quantization error, nothing more)."""
    torch.manual_seed(0)
    n = 256
    rows = torch.randn(n, head_dim, dtype=torch.bfloat16, device=DEVICE)
    cache = _make_fp8_cache(rows, BLOCK_SIZE)
    # Scatter row ids with -1 padding in between to exercise the pad path.
    ids = torch.arange(n, dtype=torch.int32, device=DEVICE)
    if pad:
        pad_ids = torch.full((pad,), -1, dtype=torch.int32, device=DEVICE)
        ids = torch.cat([ids[:64], pad_ids, ids[64:]])
    out = torch.empty((ids.numel(), head_dim), dtype=torch.bfloat16, device=DEVICE)
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    dequant_gather_fp8_rows(out, cache, ids, k_scale)

    ref = rows.to(torch.float8_e4m3fn).to(torch.bfloat16)
    valid = ids >= 0
    # Map gathered slot j back to source row: ids[j] (rows were stored 1:1).
    assert torch.equal(out[valid], ref[ids[valid].long()])
    # Padded slots must be zero-filled (they are masked by -1 at attention
    # time; zeros keep NaN-free arithmetic).
    assert torch.count_nonzero(out[~valid]) == 0


def test_remap_to_gather_rows_preserves_padding():
    """Gathered row ids are token-major; -1 padding must survive the remap or
    the bf16 kernel would attend to bogus zero rows."""
    torch.manual_seed(0)
    num_tokens, topk = 5, 8
    topk_global = torch.randint(0, 1000, (num_tokens, 1, topk), dtype=torch.int32)
    topk_global[2, 0, 3:] = -1
    topk_global[4, 0, :] = -1
    flat = topk_global.reshape(-1)
    remapped = remap_to_gather_rows(flat)
    assert remapped.dtype == torch.int32
    for i in range(num_tokens * topk):
        if flat[i] >= 0:
            assert remapped[i].item() == i
        else:
            assert remapped[i].item() == -1


@pytest.mark.parametrize("head_dim", [512, 576])
def test_fp8_sparse_attention_parity_vs_bf16(head_dim: int):
    """End-to-end parity of the fp8 read path against the bf16 path on the
    same underlying rows: gather-dequant + remap + bf16 kernel vs. bf16 cache
    + original indices. Tolerance is the e4m3 roundtrip error propagated
    through a 512/576-dim dot product and softmax."""
    torch.manual_seed(0)
    num_tokens = 3
    topk = 128
    total_rows = num_tokens * topk
    rows = torch.randn(total_rows, head_dim, dtype=torch.bfloat16, device=DEVICE)
    topk_global = torch.arange(total_rows, dtype=torch.int32, device=DEVICE).reshape(
        num_tokens, 1, topk
    )
    q = torch.randn(num_tokens, NUM_HEADS, head_dim, dtype=torch.bfloat16, device=DEVICE)

    # bf16 reference path (existing kernel, flat contiguous rows).
    out_bf16 = triton_mla_sparse_attention(
        q,
        rows.view(-1, 1, head_dim),
        topk_global,
        sm_scale=SM_SCALE,
    )

    # fp8 path: quantized cache + gather-dequant + remapped indices.
    cache = _make_fp8_cache(rows, BLOCK_SIZE)
    flat = topk_global.reshape(-1)
    gathered = torch.empty((total_rows, head_dim), dtype=torch.bfloat16, device=DEVICE)
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    dequant_gather_fp8_rows(gathered, cache, flat, k_scale)
    remapped = remap_to_gather_rows(flat).reshape(num_tokens, 1, topk)
    out_fp8 = triton_mla_sparse_attention(
        q,
        gathered.view(-1, 1, head_dim),
        remapped,
        sm_scale=SM_SCALE,
    )

    torch.testing.assert_close(out_fp8, out_bf16, rtol=0.05, atol=0.02)
