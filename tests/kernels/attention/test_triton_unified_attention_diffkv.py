# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the Triton DiffKV unified-attention kernel.
"""

import pytest
import torch

import vllm.v1.attention.ops.triton_unified_attention_diffkv as _diffkv_module
from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    set_random_seed,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)

pytestmark = pytest.mark.skip_global_cleanup

DEVICE_TYPE = current_platform.device_type

# (num_query_heads, num_kv_heads): MHA, GQA, and the num_kv_heads==1
# (degenerate-stride) case.
NUM_HEADS = [(4, 4), (8, 2), (5, 1)]
# (head_size_qk, head_size_v).  (192, 128) is the canonical asymmetric
# DiffKV shape; FA4 on Blackwell only supports head_size>128 when it is
# 192, and FA3 on Hopper supports it too -- so this pair is runnable on
# both.  (128, 128) keeps the equal-dim path covered through the DiffKV
# kernel.
HEAD_SIZES = [(128, 128), (192, 128)]
BLOCK_SIZES = [16]
DTYPES = [torch.bfloat16]

NUM_BLOCKS = 2048

# 0: 2D decode kernel; 8: 3D (split-KV) decode kernel.
SEQ_THRESHOLD_3D_VALUES = [0, 8]

NUM_PAR_SOFTMAX_SEGMENTS = 16


def _alloc_segm_buffers(seq_threshold_3D: int, num_query_heads: int, head_size_v: int):
    """Allocate the split-KV softmax scratch (last dim == head_size_v)."""
    head_size_v_padded = next_power_of_2(head_size_v)
    segm_output = torch.empty(
        (
            seq_threshold_3D,
            num_query_heads,
            NUM_PAR_SOFTMAX_SEGMENTS,
            head_size_v_padded,
        ),
        dtype=torch.float32,
    )
    segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, NUM_PAR_SOFTMAX_SEGMENTS),
        dtype=torch.float32,
    )
    segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, NUM_PAR_SOFTMAX_SEGMENTS),
        dtype=torch.float32,
    )
    return segm_output, segm_max, segm_expsum


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328), (5, 18), (129, 463)],  # mixed prefill + decode
        [(1, 523), (1, 37), (1, 2011)],  # decode-only (exercises 3D path)
    ],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_sizes", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 128])
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_diffkv_vs_fa(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_sizes: tuple[int, int],
    sliding_window: int | None,
    soft_cap: float | None,
    dtype: torch.dtype,
    block_size: int,
    seq_threshold_3D: int,
) -> None:
    head_size_qk, head_size_v = head_sizes

    # DiffKV requires FA3 (Hopper) / FA4 (Blackwell) as the reference.
    fa_version = get_flash_attn_version(head_size=head_size_qk, head_size_v=head_size_v)
    if not is_flash_attn_varlen_func_available() or fa_version not in (3, 4):
        pytest.skip(f"FA DiffKV needs FA3/FA4 (got version {fa_version}).")

    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size_qk**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size_qk, dtype=dtype)
    # Packed KV cache: [num_blocks, block_size, num_kv_heads, hqk + hv].
    kv_cache = torch.randn(
        NUM_BLOCKS,
        block_size,
        num_kv_heads,
        head_size_qk + head_size_v,
        dtype=dtype,
    )
    key_cache = kv_cache[..., :head_size_qk]
    value_cache = kv_cache[..., head_size_qk:]

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    # ---- FlashAttention DiffKV (ground truth) ---------------------------
    # Mirror the backend: fix degenerate strides on size-1 dims so FA's
    # TMA path sees ≥16-byte-aligned strides (matters for num_kv_heads==1).
    fa_k = canonicalize_singleton_dim_strides(key_cache)
    fa_v = canonicalize_singleton_dim_strides(value_cache)
    fa_out = torch.empty(sum(query_lens), num_query_heads, head_size_v, dtype=dtype)
    flash_attn_varlen_func(
        q=query,
        k=fa_k,
        v=fa_v,
        out=fa_out,
        cu_seqlens_q=cu_query_lens,
        max_seqlen_q=max_query_len,
        seqused_k=kv_lens_t,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=list(window_size),
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        fa_version=fa_version,
    )

    # ---- Triton DiffKV --------------------------------------------------
    segm_output, segm_max, segm_expsum = _alloc_segm_buffers(
        seq_threshold_3D, num_query_heads, head_size_v
    )
    triton_out = torch.empty(sum(query_lens), num_query_heads, head_size_v, dtype=dtype)
    unified_attention_diffkv(
        q=query,
        k=key_cache,
        v=value_cache,
        out=triton_out,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        max_seqlen_q=max_query_len,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
    )

    (
        torch.testing.assert_close(triton_out, fa_out, atol=2e-2, rtol=2e-2),
        f"triton vs FA max abs diff: {torch.max(torch.abs(triton_out - fa_out))}",
    )


# ---- split-KV spec-verify + wide-prefill knobs (diffbot recipe port) -------
#
# Pure-PyTorch fp32 references so these run on any CUDA device (sm86 has no
# FA3/FA4 DiffKV reference).  They exercise the env-gated split-KV verify
# launch (VLLM_DIFFKV_SPEC_3D_MAX_Q / VLLM_DIFFKV_SPEC_3D_BLOCK_M), the wide
# prefill tiles (VLLM_DIFFKV_PREFILL_BLOCK_M), and raised segment counts
# (VLLM_DIFFKV_FULL_ATTN_SEGMENTS semantics via the launcher parameter).

# (num_query_heads, num_kv_heads): GQA and the TP8 degenerate-stride case
# (MiMo-V2.6 global layer at TP8: 64 q heads / 1 kv head).
SPEC3D_NUM_HEADS = [(8, 2), (64, 1)]
SPEC3D_QLENS = [4, 8]  # MTP-3 verify (k+1) and DFlash k=7 verify
SPEC3D_SEQ_LENS = [
    [9000, 300],
    [65, 9],  # segments > seq len: exercises the all-masked-segment M=-inf guard
]


def _diffkv_fp32_ref(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    query: torch.Tensor,
    kv_lens: list[int],
    qlen: int,
    num_query_heads: int,
    scale: float,
    block_size: int,
    window_size: tuple[int, int],
    sinks: torch.Tensor | None,
) -> torch.Tensor:
    """Per-sequence fp32 attention: causal (+ optional window), packed cache."""
    num_kv_heads = kv_cache.shape[2]
    head_size_qk = query.shape[2]
    outs = []
    for i, seq_len in enumerate(kv_lens):
        num_blocks = (seq_len + block_size - 1) // block_size
        blocks = (
            kv_cache[block_table[i, :num_blocks].long()]
            .reshape(num_blocks * block_size, num_kv_heads, -1)[:seq_len]
            .float()
        )
        K, V = blocks[..., :head_size_qk], blocks[..., head_size_qk:]
        Q = query[i * qlen : (i + 1) * qlen].float()
        K = K.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
        V = V.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
        S = torch.einsum("qhd,khd->hqk", Q, K) * scale
        qpos = torch.arange(seq_len - qlen, seq_len, device=Q.device)[:, None]
        kpos = torch.arange(seq_len, device=Q.device)[None, :]
        mask = kpos <= qpos
        if window_size[0] >= 0:
            mask &= kpos > qpos - (window_size[0] + 1)
        S = S.masked_fill(~mask[None], float("-inf"))
        if sinks is not None:
            # Attention sinks join the softmax as a zero-V virtual key.
            S = torch.cat(
                [S, sinks.float()[:, None, None].expand(num_query_heads, qlen, 1)],
                dim=-1,
            )
            P = torch.softmax(S, -1)[..., :-1]
        else:
            P = torch.softmax(S, -1)
        outs.append(torch.einsum("hqk,khd->qhd", P, V))
    return torch.cat(outs)


def _run_diffkv(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_lens: list[int],
    qlen: int,
    block_table: torch.Tensor,
    num_query_heads: int,
    head_size_qk: int,
    head_size_v: int,
    block_size: int,
    window_size: tuple[int, int],
    sinks: torch.Tensor | None,
    seq_threshold_3d: int,
    segments: int,
    out_dtype: torch.dtype,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> torch.Tensor:
    cu_query_lens = torch.tensor([0] + [qlen] * len(kv_lens), dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    segm_output = torch.empty(
        (seq_threshold_3d, num_query_heads, segments, next_power_of_2(head_size_v)),
        dtype=torch.float32,
    )
    segm_max = torch.empty(
        (seq_threshold_3d, num_query_heads, segments), dtype=torch.float32
    )
    segm_expsum = torch.empty(
        (seq_threshold_3d, num_query_heads, segments), dtype=torch.float32
    )
    out = torch.empty(query.shape[0], num_query_heads, head_size_v, dtype=out_dtype)
    unified_attention_diffkv(
        q=query,
        k=kv_cache[..., :head_size_qk],
        v=kv_cache[..., head_size_qk:],
        out=out,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        softmax_scale=head_size_qk**-0.5,
        causal=True,
        window_size=window_size,
        block_table=block_table,
        softcap=0,
        max_seqlen_q=qlen,
        seq_threshold_3D=seq_threshold_3d,
        num_par_softmax_segments=segments,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
        sinks=sinks,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    return out


@pytest.mark.parametrize("num_heads", SPEC3D_NUM_HEADS)
@pytest.mark.parametrize("qlen", SPEC3D_QLENS)
@pytest.mark.parametrize("kv_lens", SPEC3D_SEQ_LENS)
@pytest.mark.parametrize("segments", [16, 64])
@pytest.mark.parametrize("sliding_window", [None, 128])
@pytest.mark.parametrize("sinks", [False, True])
@torch.inference_mode()
def test_diffkv_spec3d_and_segments_vs_ref(
    num_heads: tuple[int, int],
    qlen: int,
    kv_lens: list[int],
    segments: int,
    sliding_window: int | None,
    sinks: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not current_platform.is_cuda():
        pytest.skip("CUDA-only kernel")
    if sliding_window is not None and segments != 16:
        pytest.skip("windowed 2D path is segment-count independent")

    # Force the split-KV verify knobs on (server controls these via env vars).
    monkeypatch.setattr(_diffkv_module, "_SPEC_3D_MAX_Q", 16)
    monkeypatch.setattr(_diffkv_module, "_SPEC_3D_BLOCK_M", 128)
    monkeypatch.setattr(_diffkv_module, "_SPEC_3D_NUM_WARPS", 8)
    monkeypatch.setattr(_diffkv_module, "_SPEC_3D_TILE", 16)

    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_query_heads, num_kv_heads = num_heads
    head_size_qk, head_size_v, block_size = 192, 128, 16
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    sink_t = torch.randn(num_query_heads, dtype=torch.float32) if sinks else None

    num_seqs = len(kv_lens)
    max_blocks = (max(kv_lens) + block_size - 1) // block_size
    kv_cache = torch.randn(
        4096, block_size, num_kv_heads, head_size_qk + head_size_v, dtype=torch.bfloat16
    )
    block_table = torch.randint(0, 4096, (num_seqs, max_blocks), dtype=torch.int32)
    query = torch.randn(
        num_seqs * qlen, num_query_heads, head_size_qk, dtype=torch.bfloat16
    )

    out = _run_diffkv(
        query,
        kv_cache,
        kv_lens,
        qlen,
        block_table,
        num_query_heads,
        head_size_qk,
        head_size_v,
        block_size,
        window_size,
        sink_t,
        seq_threshold_3d=64,
        segments=segments,
        out_dtype=torch.bfloat16,
    )
    ref = _diffkv_fp32_ref(
        kv_cache,
        block_table,
        query,
        kv_lens,
        qlen,
        num_query_heads,
        head_size_qk**-0.5,
        block_size,
        window_size,
        sink_t,
    )
    assert not torch.isnan(out.float()).any(), "NaN in split-KV verify output"
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_heads", SPEC3D_NUM_HEADS)
@pytest.mark.parametrize("prefill_block_m", [16, 128])
@pytest.mark.parametrize("sliding_window", [None, 128])
@torch.inference_mode()
def test_diffkv_prefill_tiles_vs_ref(
    num_heads: tuple[int, int],
    prefill_block_m: int,
    sliding_window: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not current_platform.is_cuda():
        pytest.skip("CUDA-only kernel")

    # Wide-prefill knobs (VLLM_DIFFKV_PREFILL_BLOCK_M et al. on the server).
    monkeypatch.setattr(_diffkv_module, "_PREFILL_BLOCK_M", prefill_block_m)
    monkeypatch.setattr(_diffkv_module, "_PREFILL_NUM_WARPS", 8)
    monkeypatch.setattr(_diffkv_module, "_PREFILL_NUM_STAGES", 2)
    monkeypatch.setattr(_diffkv_module, "_PREFILL_TILE", 32)

    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_query_heads, num_kv_heads = num_heads
    head_size_qk, head_size_v, block_size = 192, 128, 16
    qlen, seq_len = 256, 3000  # chunked-prefill shape: max_seqlen_q >= _PREFILL_MIN_Q
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)

    kv_cache = torch.randn(
        4096, block_size, num_kv_heads, head_size_qk + head_size_v, dtype=torch.bfloat16
    )
    max_blocks = (seq_len + block_size - 1) // block_size
    block_table = torch.randint(0, 4096, (1, max_blocks), dtype=torch.int32)
    query = torch.randn(qlen, num_query_heads, head_size_qk, dtype=torch.bfloat16)

    out = _run_diffkv(
        query,
        kv_cache,
        [seq_len],
        qlen,
        block_table,
        num_query_heads,
        head_size_qk,
        head_size_v,
        block_size,
        window_size,
        None,
        seq_threshold_3d=0,
        segments=16,
        out_dtype=torch.bfloat16,
    )
    ref = _diffkv_fp32_ref(
        kv_cache,
        block_table,
        query,
        [seq_len],
        qlen,
        num_query_heads,
        head_size_qk**-0.5,
        block_size,
        window_size,
        None,
    )
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


# ---- fp8 E4M3 KV cache (per-tensor scales) ----------------------------------
#
# Contract (diffbot recipe): the kernel output on a quantized cache must match
# a fp32 reference computed on the DEQUANTIZED cache (kernel correctness,
# budget = max(2x the bf16-cache kernel error, 1e-2)); the quantization error
# vs the original bf16 cache is informational.  Both dequant strategies are
# covered: the sm86-safe LUT gather and the sm89+ native conversion.

F8_E4M3 = torch.float8_e4m3fn
FP8_QUANT_MAX = 448.0


def _quant_kv(
    kv_cache: torch.Tensor,
    head_size_qk: int,
    k_scale: float,
    v_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a bf16 packed cache to E4M3 with per-tensor scales.

    Returns the fp8 cache and its bf16 dequantization.
    """
    k8 = (kv_cache[..., :head_size_qk].float() / k_scale).to(F8_E4M3)
    v8 = (kv_cache[..., head_size_qk:].float() / v_scale).to(F8_E4M3)
    kv8 = torch.cat([k8, v8], dim=-1)
    deq = torch.cat([k8.float() * k_scale, v8.float() * v_scale], dim=-1).to(
        kv_cache.dtype
    )
    return kv8, deq


FP8_CASES = [
    # (qlen, sliding_window, sinks, seq_threshold_3d)
    (1, None, False, 64),  # decode (stock 3D split-KV)
    (8, None, False, 64),  # spec verify (patched spec-3D launch)
    (256, None, False, 0),  # prefill chunk (2D + wide-prefill tiles)
    (8, 128, True, 64),  # SWA + sinks verify
    (128, 128, True, 0),  # SWA prefill + sinks
]


@pytest.mark.parametrize("num_heads", SPEC3D_NUM_HEADS)
@pytest.mark.parametrize("dequant_mode", ["lut", "native"])
@pytest.mark.parametrize("qlen,sliding_window,sinks,seq_threshold_3d", FP8_CASES)
@pytest.mark.parametrize("scale_mode", ["1.0", "amax"])
@torch.inference_mode()
def test_diffkv_fp8_kv_vs_ref(
    num_heads: tuple[int, int],
    dequant_mode: str,
    qlen: int,
    sliding_window: int | None,
    sinks: bool,
    seq_threshold_3d: int,
    scale_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not current_platform.is_cuda():
        pytest.skip("CUDA-only kernel")
    major, minor = torch.cuda.get_device_capability()
    if dequant_mode == "native" and (major, minor) < (8, 9):
        pytest.skip("native E4M3 conversion needs sm89+")

    monkeypatch.setattr(_diffkv_module, "_FP8_DEQUANT_MODE", dequant_mode)
    # Exercise the same launch shapes the split-KV knobs produce on the server.
    monkeypatch.setattr(_diffkv_module, "_SPEC_3D_MAX_Q", 16)
    monkeypatch.setattr(_diffkv_module, "_SPEC_3D_BLOCK_M", 128)
    monkeypatch.setattr(_diffkv_module, "_SPEC_3D_NUM_WARPS", 8)
    monkeypatch.setattr(_diffkv_module, "_SPEC_3D_TILE", 16)
    monkeypatch.setattr(_diffkv_module, "_PREFILL_BLOCK_M", 128)
    monkeypatch.setattr(_diffkv_module, "_PREFILL_NUM_WARPS", 8)
    monkeypatch.setattr(_diffkv_module, "_PREFILL_NUM_STAGES", 2)
    monkeypatch.setattr(_diffkv_module, "_PREFILL_TILE", 32)

    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_query_heads, num_kv_heads = num_heads
    head_size_qk, head_size_v, block_size = 192, 128, 16
    kv_lens = [9000, 300, qlen]
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    sink_t = torch.randn(num_query_heads, dtype=torch.float32) if sinks else None

    num_seqs = len(kv_lens)
    max_blocks = (max(kv_lens) + block_size - 1) // block_size
    kv_cache = (
        torch.randn(
            4096,
            block_size,
            num_kv_heads,
            head_size_qk + head_size_v,
            dtype=torch.bfloat16,
        )
        * 1.5
    )
    block_table = torch.randint(0, 4096, (num_seqs, max_blocks), dtype=torch.int32)
    query = torch.randn(
        num_seqs * qlen, num_query_heads, head_size_qk, dtype=torch.bfloat16
    )

    if scale_mode == "amax":
        k_scale = kv_cache[..., :head_size_qk].abs().max().item() / FP8_QUANT_MAX
        v_scale = kv_cache[..., head_size_qk:].abs().max().item() / FP8_QUANT_MAX
    else:
        k_scale = v_scale = 1.0
    kv8, deq = _quant_kv(kv_cache, head_size_qk, k_scale, v_scale)
    k_descale = torch.tensor([k_scale], dtype=torch.float32)
    v_descale = torch.tensor([v_scale], dtype=torch.float32)

    common = dict(
        kv_lens=kv_lens,
        qlen=qlen,
        block_table=block_table,
        num_query_heads=num_query_heads,
        head_size_qk=head_size_qk,
        head_size_v=head_size_v,
        block_size=block_size,
        window_size=window_size,
        sinks=sink_t,
        seq_threshold_3d=seq_threshold_3d,
        segments=16,
        out_dtype=torch.bfloat16,
    )
    out8 = _run_diffkv(query, kv8, k_descale=k_descale, v_descale=v_descale, **common)
    base = _run_diffkv(query, kv_cache, **common)
    ref_deq = _diffkv_fp32_ref(
        deq,
        block_table,
        query,
        kv_lens,
        qlen,
        num_query_heads,
        head_size_qk**-0.5,
        block_size,
        window_size,
        sink_t,
    )
    ref_bf16 = _diffkv_fp32_ref(
        kv_cache,
        block_table,
        query,
        kv_lens,
        qlen,
        num_query_heads,
        head_size_qk**-0.5,
        block_size,
        window_size,
        sink_t,
    )

    assert not torch.isnan(out8.float()).any(), "NaN in fp8-KV output"
    base_err = (base.float() - ref_bf16).abs().max().item()
    kernel_err = (out8.float() - ref_deq).abs().max().item()
    # The sm<89 LUT path clamps the wide tiles (BLOCK_M 128 -> 64), which
    # splits the reduction into more partial tiles and adds LSE combine
    # steps, so allow a little more deviation than the wide-tile configs.
    budget = max(3 * base_err, 1.5e-2)
    assert kernel_err <= budget, (
        f"fp8 kernel error {kernel_err:.4f} vs dequant ref exceeds budget "
        f"max(2*{base_err:.4f}, 1e-2)"
    )


@torch.inference_mode()
def test_diffkv_fp8_store_matches_reference() -> None:
    """fp8 KV store: bytes in the cache must equal torch's E4M3 cast of the
    scaled inputs, on both the sm89+ implicit-cast path and the sm<89
    torch-software-cast path (fp8e4nv cannot be lowered on Ampere)."""
    if not current_platform.is_cuda():
        pytest.skip("CUDA-only kernel")

    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_diffkv,
    )

    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)
    num_tokens, num_heads, head_size_k, head_size_v, block_size = 7, 2, 192, 128, 16
    key = torch.randn(num_tokens, num_heads, head_size_k, dtype=torch.bfloat16)
    value = torch.randn(num_tokens, num_heads, head_size_v, dtype=torch.bfloat16)
    k_scale = torch.tensor(1.5, dtype=torch.float32)
    v_scale = torch.tensor(0.75, dtype=torch.float32)
    num_slots = 32
    cache = torch.zeros(
        num_slots,
        block_size,
        num_heads,
        head_size_k + head_size_v,
        dtype=torch.uint8,
    )
    slots = torch.arange(num_tokens, dtype=torch.int64) % num_slots

    triton_reshape_and_cache_flash_diffkv(
        key, value, cache, slots, "fp8", k_scale, v_scale
    )

    ref_k = (key / k_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    ref_v = (value / v_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    mismatches = [
        i
        for i, slot in enumerate(slots.tolist())
        if not torch.equal(
            cache[slot, i % block_size, :, :head_size_k], ref_k[i]
        )
    ]
    if mismatches:
        i = mismatches[0]
        needle = ref_k[i].reshape(-1)
        # Where did token i's K bytes actually land in the pool?
        flat = cache.reshape(-1)
        hits = (flat == needle[0]).nonzero().flatten().tolist()[:8]
        detail = {
            "mismatched_tokens": mismatches,
            "first_bad_token": i,
            "slot": int(slots[i]),
            "expected_head0_first6": needle[:6].tolist(),
            "got_at_expected_pos": cache[int(slots[i]), i % block_size, 0, :6]
            .tolist(),
            "first_byte_found_at_flat": hits,
        }
        raise AssertionError(f"fp8 store mismatch: {detail}")
    for i, slot in enumerate(slots.tolist()):
        got_v = cache[slot, i % block_size, :, head_size_k:]
        assert torch.equal(got_v, ref_v[i]), f"V bytes mismatch at token {i}"


@torch.inference_mode()
def test_diffkv_fp8_prefill_wide_tile_lut_fits_sm86() -> None:
    """Regression: BLOCK_M=128 prefill + LUT dequant needs ~112KB shared
    memory and fails to launch on sm80/sm86 (99KB ceiling). The launcher
    must clamp the wide tile so real prefills (qlen >= _PREFILL_MIN_Q) run."""
    if not current_platform.is_cuda():
        pytest.skip("CUDA-only kernel")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(_diffkv_module, "_FP8_DEQUANT_MODE", "auto")
    monkeypatch.setattr(_diffkv_module, "_PREFILL_BLOCK_M", 128)
    monkeypatch.setattr(_diffkv_module, "_PREFILL_NUM_WARPS", 8)
    monkeypatch.setattr(_diffkv_module, "_PREFILL_NUM_STAGES", 2)
    monkeypatch.setattr(_diffkv_module, "_PREFILL_TILE", 32)

    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)
    num_query_heads, num_kv_heads, head_size_qk, head_size_v, block_size = (
        8,
        2,
        192,
        128,
        16,
    )
    qlen, seq_len = 256, 3000
    kv_cache = torch.randn(
        4096,
        block_size,
        num_kv_heads,
        head_size_qk + head_size_v,
        dtype=torch.bfloat16,
    )
    k_descale = torch.tensor(1.0, dtype=torch.float32)
    v_descale = torch.tensor(1.0, dtype=torch.float32)
    kv_cache_fp8 = torch.cat(
        [
            kv_cache[..., :head_size_qk].to(torch.float8_e4m3fn),
            kv_cache[..., head_size_qk:].to(torch.float8_e4m3fn),
        ],
        dim=-1,
    )
    block_table = torch.randint(0, 4096, (1, seq_len // block_size), dtype=torch.int32)
    query = torch.randn(qlen, num_query_heads, head_size_qk, dtype=torch.bfloat16)

    out = _run_diffkv(
        query,
        kv_cache_fp8,
        [seq_len],
        qlen,
        block_table,
        num_query_heads,
        head_size_qk,
        head_size_v,
        block_size,
        (-1, -1),
        None,
        seq_threshold_3d=0,
        segments=16,
        out_dtype=torch.bfloat16,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    assert out.shape == (qlen, num_query_heads, head_size_v)
    assert not torch.isnan(out.float()).any(), "NaN in fp8-KV prefill output"
