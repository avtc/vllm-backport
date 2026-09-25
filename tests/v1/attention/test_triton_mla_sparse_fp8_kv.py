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
    """Quantize bf16 rows into a flat fp8 cache, row i living at slot i."""
    n, head_dim = rows.shape
    num_blocks = (n + block_size - 1) // block_size
    cache = torch.zeros(
        (num_blocks, block_size, head_dim),
        dtype=torch.float8_e4m3fn,
        device=rows.device,
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


def test_dequant_gather_preserves_nan_byte():
    """The e4m3fn NaN encoding (0x7F/0xFF) must decode to NaN (matching
    torch's cast), not to a large finite value."""
    head_dim = 512
    cache = torch.zeros(1, BLOCK_SIZE, head_dim, dtype=torch.uint8, device=DEVICE)
    cache[0, 0] = 0x7F  # +NaN in e4m3fn
    cache[0, 1] = 0xFF  # -NaN in e4m3fn
    cache = cache.view(torch.float8_e4m3fn)
    ids = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    out = torch.empty((2, head_dim), dtype=torch.bfloat16, device=DEVICE)
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    dequant_gather_fp8_rows(out, cache, ids, k_scale)
    assert torch.isnan(out.float()).all()


def test_remap_to_gather_rows_preserves_padding():
    """Gathered row ids are token-major; -1 padding must survive the remap or
    the bf16 kernel would attend to bogus zero rows."""
    torch.manual_seed(0)
    num_tokens, topk = 5, 8
    topk_global = torch.randint(0, 1000, (num_tokens, 1, topk), dtype=torch.int32)
    topk_global[2, 0, 3:] = -1
    topk_global[4, 0, :] = -1
    # One deliberately out-of-range id: must be masked to -1 like padding.
    topk_global[1, 0, 0] = 5000
    flat = topk_global.reshape(-1)
    remapped = remap_to_gather_rows(flat, num_rows=1000)
    assert remapped.dtype == torch.int32
    for i in range(num_tokens * topk):
        if flat[i] >= 0 and flat[i] < 1000:
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
    q = torch.randn(
        num_tokens, NUM_HEADS, head_dim, dtype=torch.bfloat16, device=DEVICE
    )

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
    remapped = remap_to_gather_rows(flat, num_rows=total_rows).reshape(
        num_tokens, 1, topk
    )
    out_fp8 = triton_mla_sparse_attention(
        q,
        gathered.view(-1, 1, head_dim),
        remapped,
        sm_scale=SM_SCALE,
    )

    torch.testing.assert_close(out_fp8, out_bf16, rtol=0.05, atol=0.02)


@pytest.mark.parametrize("head_dim", [512, 576])
def test_fp8_subbatched_parity(head_dim: int, monkeypatch: pytest.MonkeyPatch):
    """The byte-budgeted sub-batch path (the real shared function) must be
    bit-identical to a single-shot gather-dequant + attention call when the
    budget forces one sub-batch per token (prefill-shaped batches rely on
    it to bound the workspace)."""
    from vllm import envs
    from vllm.v1.attention.backends.mla.triton_mla_sparse import (
        _fp8_kv_subbatched_forward,
    )

    torch.manual_seed(0)
    num_tokens, topk = 20, 128
    total_rows = num_tokens * topk
    rows = torch.randn(total_rows, head_dim, dtype=torch.bfloat16, device=DEVICE)
    topk_global = torch.arange(total_rows, dtype=torch.int32, device=DEVICE).reshape(
        num_tokens, 1, topk
    )
    q = torch.randn(
        num_tokens, NUM_HEADS, head_dim, dtype=torch.bfloat16, device=DEVICE
    )
    cache = _make_fp8_cache(rows, BLOCK_SIZE)
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

    def _forward_pinned() -> torch.Tensor:
        if hasattr(envs.__getattr__, "cache_clear"):
            envs.__getattr__.cache_clear()
        return _fp8_kv_subbatched_forward(
            q, cache, topk_global, k_scale, SM_SCALE, None, num_kv_splits=2
        )

    # Huge budget: one sub-batch. 1 MiB -> budget_rows = 8 -> 3 sub-batches
    # (8/8/4) for 20 tokens. Pin num_kv_splits so the split heuristic (which
    # depends on num_tokens) cannot pick different split counts for the two
    # calls and mask a real difference behind a different reduction order.
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_FP8_GATHER_MB", "1024")
    single = _forward_pinned()
    monkeypatch.setenv("VLLM_TRITON_MLA_SPARSE_FP8_GATHER_MB", "1")
    subbatched = _forward_pinned()
    torch.testing.assert_close(subbatched, single, rtol=0.0, atol=0.0)


def test_forward_mqa_rejects_e5m2():
    """Quantized dtypes outside the fp8 e4m3 whitelist must raise a clear
    NotImplementedError from the Triton impl's dispatch (config mistakes
    land here, not in a kernel crash)."""
    from vllm.v1.attention.backends.mla.triton_mla_sparse import (
        TritonMLASparseImpl,
    )

    impl = TritonMLASparseImpl.__new__(TritonMLASparseImpl)
    impl.kv_cache_dtype = "fp8_e5m2"
    q = torch.empty(1, 1, 512, dtype=torch.bfloat16, device=DEVICE)
    kv = torch.empty(1, 1, 512, dtype=torch.float8_e4m3fn, device=DEVICE)
    with pytest.raises(NotImplementedError, match="fp8/fp8_e4m3 only"):
        impl.forward_mqa(q, kv, None, None)
