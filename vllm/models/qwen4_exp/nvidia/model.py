# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4Exp model."""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import (
    is_model_fused_shared_expert_compatible,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateCopyFuncsByType,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextMLP,
    Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    StageMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from ..config import Qwen4ExpConfig
from .hyperconnection import GatedResidual, HyperConnectionConfig
from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
from .ple_layer import Qwen4ExpPLELayer
from .qsa import Qwen4ExpQSAAttention


def without_modelopt_fp4(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Return ``None`` for weights excluded from Qwen4Exp ModelOpt-FP4."""

    if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
        return None
    return quant_config


def _remap_qsa_cache_scale_name(
    name: str,
    qsa_layer_ids: frozenset[int],
) -> str:
    """Map serialized main-cache scales onto the merged QSA owner.

    Regular attention keeps cache scales below its ``attn`` child. QSA owns
    that cache directly, so only QSA layers need the final path component
    moved to the owner's persistent ``_k_scale``/``_v_scale`` buffers.
    """

    scale_suffixes = {
        "k_proj.k_scale": "_k_scale",
        "k_proj.output_scale": "_k_scale",
        "attn.k_scale": "_k_scale",
        "attn._k_scale": "_k_scale",
        "k_scale": "_k_scale",
        "_k_scale": "_k_scale",
        "v_proj.v_scale": "_v_scale",
        "v_proj.output_scale": "_v_scale",
        "attn.v_scale": "_v_scale",
        "attn._v_scale": "_v_scale",
        "v_scale": "_v_scale",
        "_v_scale": "_v_scale",
    }
    for layer_id in qsa_layer_ids:
        marker = f"layers.{layer_id}.self_attn."
        marker_start = name.find(marker)
        if marker_start < 0 or (marker_start > 0 and name[marker_start - 1] != "."):
            continue
        suffix = name[marker_start + len(marker) :]
        mapped_suffix = scale_suffixes.get(suffix)
        if mapped_suffix is not None:
            return f"{name[: marker_start + len(marker)]}{mapped_suffix}"
    return name


_QWEN4_EXP_IGNORED_MISSING_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]

# The checkpoint stores these projections separately; runtime packs each group
# into adjacent logical shards of a MergedColumnParallelLinear. The
# hyper-connection projections intentionally stay out: quantized checkpoints
# (AutoRound/INC int8 group-64) quantize each piece independently, and
# separately-quantized tensors cannot be stacked into one packed weight.
_EXTRA_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_stacked={
        "ple.key_proj": ("ple.kv_proj", 0),
        "ple.value_proj": ("ple.kv_proj", 1),
    }
)


logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Decode-time MoE block (up to _MOE_DECODE_MAX_TOKENS tokens)
# ---------------------------------------------------------------------------
#
# Ported from Minachist's Apache-2.0 decode-05 patch. At one token the
# unfused MoE block spends more time in small kernels than in the expert
# GEMMs: the BF16 router GEMV (cuBLAS, ~12 us), topk_softmax (one CTA,
# ~7 us), moe_align_block_size (two kernels, ~7 us), the shared-expert gate
# (dot + reduce + sigmoid + mul, ~7 us), moe_sum and the shared+routed add.
# The expert weights themselves are ~25 MB and stream in ~42 us.
#
# _moe_router_kernel computes the router logits and the shared-expert gate
# logit in one GEMV; the program that finishes last (atomic ticket) picks the
# top-k, renormalizes and writes the block alignment the Marlin MoE GEMM
# reads (moe_align_block_size's contract with expert_map=None). The Marlin
# GEMMs and the activation are the unchanged vLLM ones (_fused_marlin_moe),
# and _moe_combine_kernel replaces moe_sum + sigmoid gate + the final add.
# Values are rounded to BF16 at the same points as the unfused path.
# VLLM_MOE_FUSED_DECODE=0 keeps the unfused path.
#
# Only the single-GPU MoE layout is covered: the fused alignment and combine
# skip FusedMoE's dispatch/combine reductions, so anything other than
# MoE tp_size=1 without EP falls back (checked in _setup_fused_decode).
_MOE_DECODE_MAX_TOKENS = 4


def _moe_fused_decode_enabled() -> bool:
    return bool(envs.VLLM_MOE_FUSED_DECODE)


@triton.jit
def _moe_router_kernel(
    x_ptr,
    w_ptr,
    wsg_ptr,
    logits_ptr,
    sgate_ptr,
    ticket_ptr,
    topk_w_ptr,
    topk_ids_ptr,
    sorted_ptr,
    expert_ids_ptr,
    ntpp_ptr,
    stride_x,
    M,
    K: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MPAD: tl.constexpr,
    P: tl.constexpr,
    RENORM: tl.constexpr,
):
    pid = tl.program_id(0)
    NPROG: tl.constexpr = E // ROWS + 1
    ms = tl.arange(0, MPAD)
    mmask = ms < M
    ks = tl.arange(0, BLOCK_K)
    if pid < E // ROWS:
        rows = pid * ROWS + tl.arange(0, ROWS)
        acc = tl.zeros((MPAD, ROWS), tl.float32)
        for k0 in tl.range(0, K, BLOCK_K):
            kmask = (k0 + ks) < K
            w = tl.load(
                w_ptr + rows[:, None] * K + k0 + ks[None, :],
                mask=kmask[None, :],
                other=0.0,
            ).to(tl.float32)
            x = tl.load(
                x_ptr + ms[:, None] * stride_x + k0 + ks[None, :],
                mask=mmask[:, None] & kmask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
        # Router logits are BF16 in the unfused path.
        tl.store(
            logits_ptr + ms[:, None] * E + rows[None, :],
            acc.to(tl.bfloat16).to(tl.float32),
            mask=mmask[:, None],
        )
    else:
        acc1 = tl.zeros((MPAD,), tl.float32)
        for k0 in tl.range(0, K, BLOCK_K):
            kmask2 = (k0 + ks) < K
            w = tl.load(wsg_ptr + k0 + ks, mask=kmask2, other=0.0).to(tl.float32)
            x = tl.load(
                x_ptr + ms[:, None] * stride_x + k0 + ks[None, :],
                mask=mmask[:, None] & kmask2[None, :],
                other=0.0,
            ).to(tl.float32)
            acc1 += tl.sum(x * w[None, :], axis=1)
        tl.store(sgate_ptr + ms, acc1.to(tl.bfloat16).to(tl.float32), mask=mmask)

    # Last-program-done: every thread's stores are ordered before the ticket
    # (barrier, then the releasing atomic), and the winner's reads after it.
    tl.debug_barrier()
    ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel", scope="gpu")
    if ticket == NPROG - 1:
        tl.debug_barrier()
        tl.store(ticket_ptr, 0)
        es = tl.arange(0, E)
        logits = tl.load(
            logits_ptr + ms[:, None] * E + es[None, :],
            mask=mmask[:, None],
            other=0.0,
            cache_modifier=".cg",
        )
        # Top-k on the logits (softmax is monotonic). The logits are BF16
        # values, so a 32-bit key holds the whole order: the order-preserving
        # 16-bit form of the value above the inverted expert id (equal values
        # prefer the lower id, like the unfused kernel). One max per pick.
        bits = logits.to(tl.int32, bitcast=True)
        ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
        key = ((ordered >> 16) << 16) | (65535 - es[None, :])
        ks_t = tl.arange(0, P)
        kmask = ks_t[None, :] < TOPK
        top = tl.zeros((MPAD, P), tl.int32)
        for j in tl.static_range(TOPK):
            best = tl.max(key, axis=1)
            top = tl.where(ks_t[None, :] == j, best[:, None], top)
            key = tl.where(key == best[:, None], -2147483648, key)
        sel_i = 65535 - (top & 0xFFFF)
        hi = (top >> 16) << 16
        tlog = tl.where(hi < 0, (hi | 0xFFFF) ^ 0x7FFFFFFF, hi).to(
            tl.float32, bitcast=True
        )
        # With renormalization the full-softmax denominator cancels:
        # w_i = exp(l_i) / sum over the selected exp(l_j).
        if RENORM:
            e_top = tl.where(kmask, tl.exp(tlog - tl.max(tlog, axis=1)[:, None]), 0.0)
            sel_w = e_top / tl.sum(e_top, axis=1)[:, None]
        else:
            mx = tl.max(logits, axis=1)
            den = tl.sum(tl.exp(logits - mx[:, None]), axis=1)
            sel_w = tl.exp(tlog - mx[:, None]) / den[:, None]
        tl.store(
            topk_w_ptr + ms[:, None] * TOPK + ks_t[None, :],
            sel_w,
            mask=mmask[:, None] & kmask,
        )
        tl.store(
            topk_ids_ptr + ms[:, None] * TOPK + ks_t[None, :],
            sel_i,
            mask=mmask[:, None] & kmask,
        )
        # Block alignment over the flat entries j = t * TOPK + k.
        NE: tl.constexpr = MPAD * P
        flat_i = tl.reshape(sel_i, (NE,))
        jj = tl.arange(0, NE)
        tok = jj // P
        kk = jj % P
        valid = (tok < M) & (kk < TOPK)
        flat = tok * TOPK + kk
        eid = tl.where(valid, flat_i, -1 - jj)  # invalid entries never match
        same = eid[:, None] == eid[None, :]
        before = jj[None, :] < jj[:, None]  # [j, i]: i precedes j
        rank = tl.sum((same & before).to(tl.int32), axis=1)
        count = tl.sum(same.to(tl.int32), axis=1)
        nblk = tl.where(valid & (rank == 0), (count + BS - 1) // BS, 0)
        # Block offset of each expert = blocks of the experts first seen
        # before it.
        first = valid & (rank == 0)
        blk_before = tl.sum(tl.where(before & first[None, :], nblk[None, :], 0), axis=1)
        # Every entry takes the offset of its expert's first occurrence.
        first_of = tl.sum(
            tl.where(same & first[None, :], blk_before[None, :], 0), axis=1
        )
        total_blk = tl.sum(nblk, axis=0)
        L: tl.constexpr = NE * BS
        pos_all = tl.arange(0, L)
        numel = M * TOPK
        tl.store(
            sorted_ptr + pos_all,
            numel + tl.zeros((L,), tl.int32),
            mask=pos_all < total_blk * BS,
        )
        tl.debug_barrier()
        tl.store(sorted_ptr + first_of * BS + rank, flat, mask=valid)
        # An expert holds at most M * TOPK entries; the gating caps that at
        # _MOE_DECODE_MAX_TOKENS * top_k <= 32 flat entries, i.e. ceil(32/BS)
        # <= 4 blocks at BS >= 8 (the marlin block sizes). The loop covers
        # exactly those blocks.
        for q in tl.static_range(4):
            tl.store(
                expert_ids_ptr + blk_before + q,
                flat_i,
                mask=first & (q < nblk),
            )
        tl.store(ntpp_ptr, total_blk * BS)


@triton.jit
def _moe_combine_kernel(
    routed_ptr,
    shared_ptr,
    sgate_ptr,
    out_ptr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for k in tl.static_range(TOPK):
        acc += tl.load(routed_ptr + (t * TOPK + k) * K + offs).to(tl.float32)
    g = tl.sigmoid(tl.load(sgate_ptr + t))
    sh = tl.load(shared_ptr + t * K + offs).to(tl.float32)
    # Unfused: moe_sum (bf16 out) + (sigmoid(g) * shared, bf16) summed in bf16.
    routed = acc.to(tl.bfloat16).to(tl.float32)
    shg = (sh * g.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + t * K + offs, (routed + shg).to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Shared expert: INT6 weights in a decode-friendly layout
# ---------------------------------------------------------------------------
#
# The shared expert's gate/up (1280 x 2560) and down (2560 x 640) projections
# are INT6 group-64 (compressed-tensors pack-quantized: a tight little-endian
# bit stream, 32 values in six int32 words, value + 32). Humming runs these
# small shapes at ~250 GB/s at one token. _Int6PlanesScheme splits the same
# 6 bits per weight into a 4-bit and a 2-bit plane once at load time (same
# memory), which the kernels below decode with shifts:
#
#   _se_gate_up_act_kernel   gate/up GEMV with SiluAndMul fused into the store
#   _se_down_combine_kernel  down GEMV fused with the MoE output combine:
#                            sum of the routed rows + sigmoid(shared gate) *
#                            shared output (replaces _moe_combine_kernel)
#
# Larger batches use _w6a16_gemm_kernel on the same planes.


def _int6_to_planes(
    packed: torch.Tensor, K: int, rows_per_chunk: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tight pack-quantized INT6 [N, K * 6 / 32] int32 -> lo/hi planes.

    lo [N, K / 8] int32 holds bits 0..3 of eight values per word, hi
    [N, K / 16] int32 bits 4..5 of sixteen. Done in row chunks so the
    load-time temporaries stay a few MB (they would otherwise stay reserved
    by the caching allocator next to a full GPU).
    """
    N = packed.shape[0]
    # The planes take exactly the packed bytes (K / 8 + K / 16 = 6 K / 32 words
    # per row). They go into one allocation of the packed size, and the caller
    # drops the packed tensor: the same allocate-new / free-old pattern as the
    # Humming repack this replaces. Two separately sized plane tensors left
    # ~100-190 MiB of stranded allocator cache per GPU.
    assert packed.is_contiguous() and packed.shape[1] * 32 == 6 * K
    storage = torch.empty_like(packed).view(-1)
    lo = storage[: N * (K // 8)].view(N, K // 8)
    hi = storage[N * (K // 8) :].view(N, K // 16)
    for r0 in range(0, N, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, N)
        words = packed[r0:r1].reshape(r1 - r0, K // 32, 6)
        vals = torch.empty(
            (r1 - r0, K // 32, 32), dtype=torch.int32, device=packed.device
        )
        for i in range(32):
            w, o = divmod(6 * i, 32)
            v = torch.bitwise_right_shift(words[:, :, w], o) & ((1 << (32 - o)) - 1)
            if o + 6 > 32:
                v = v | torch.bitwise_left_shift(words[:, :, w + 1], 32 - o)
            vals[:, :, i] = v & 63
        vals = vals.reshape(r1 - r0, K)
        lo_c = (vals & 15).reshape(r1 - r0, K // 8, 8)
        hi_c = torch.bitwise_right_shift(vals, 4).reshape(r1 - r0, K // 16, 16)
        acc = torch.zeros((r1 - r0, K // 8), dtype=torch.int32, device=packed.device)
        for j in range(8):
            acc |= torch.bitwise_left_shift(lo_c[:, :, j], 4 * j)
        lo[r0:r1] = acc
        acc = torch.zeros((r1 - r0, K // 16), dtype=torch.int32, device=packed.device)
        for j in range(16):
            acc |= torch.bitwise_left_shift(hi_c[:, :, j], 2 * j)
        hi[r0:r1] = acc
    return lo, hi


@triton.jit
def _i6_tile(
    lo_ptr, hi_ptr, rows, k0, K: tl.constexpr, R: tl.constexpr, BLOCK_K: tl.constexpr
):
    """Unscaled INT6 tile: [R, BLOCK_K] fp32 of (value - 32)."""
    lw = tl.load(
        lo_ptr
        + rows[:, None] * (K // 8)
        + k0 // 8
        + tl.arange(0, BLOCK_K // 8)[None, :]
    )
    hw = tl.load(
        hi_ptr
        + rows[:, None] * (K // 16)
        + k0 // 16
        + tl.arange(0, BLOCK_K // 16)[None, :]
    )
    lo = (lw[:, :, None] >> (4 * tl.arange(0, 8))[None, None, :]) & 15
    hi = (hw[:, :, None] >> (2 * tl.arange(0, 16))[None, None, :]) & 3
    v = tl.reshape(lo, (R, BLOCK_K)) | (tl.reshape(hi, (R, BLOCK_K)) << 4)
    return (v - 32).to(tl.float32)


@triton.jit
def _row_dot(
    w3, sc, x_ptr, k0, NG: tl.constexpr, GS: tl.constexpr, BLOCK_K: tl.constexpr
):
    x = tl.load(x_ptr + k0 + tl.arange(0, BLOCK_K)).to(tl.float32)
    return tl.sum(tl.sum(w3 * tl.reshape(x, (1, NG, GS)), axis=2) * sc, axis=1)


@triton.jit
def _se_gate_up_act_kernel(
    x_ptr,
    lo_ptr,
    hi_ptr,
    s_ptr,
    act_ptr,
    stride_x,
    stride_act,
    M: tl.constexpr,
    K: tl.constexpr,
    INTER: tl.constexpr,
    GS: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr = 1,
):
    # Rows: gate [i0, i0 + BLOCK_I) and up [INTER + i0, ...), so silu(g) * u
    # is formed in registers. Tokens (M <= 4) are unrolled.
    pid = tl.program_id(0)
    R: tl.constexpr = 2 * BLOCK_I
    NG: tl.constexpr = BLOCK_K // GS
    r = tl.arange(0, R)
    rows = (r // BLOCK_I) * INTER + pid * BLOCK_I + (r % BLOCK_I)
    a0 = tl.zeros((R,), tl.float32)
    a1 = tl.zeros((R,), tl.float32)
    a2 = tl.zeros((R,), tl.float32)
    a3 = tl.zeros((R,), tl.float32)
    for k0 in tl.range(0, K, BLOCK_K, num_stages=STAGES):
        w3 = tl.reshape(_i6_tile(lo_ptr, hi_ptr, rows, k0, K, R, BLOCK_K), (R, NG, GS))
        sc = tl.load(
            s_ptr + rows[:, None] * (K // GS) + k0 // GS + tl.arange(0, NG)[None, :]
        ).to(tl.float32)
        a0 += _row_dot(w3, sc, x_ptr, k0, NG, GS, BLOCK_K)
        if M > 1:
            a1 += _row_dot(w3, sc, x_ptr + stride_x, k0, NG, GS, BLOCK_K)
        if M > 2:
            a2 += _row_dot(w3, sc, x_ptr + 2 * stride_x, k0, NG, GS, BLOCK_K)
        if M > 3:
            a3 += _row_dot(w3, sc, x_ptr + 3 * stride_x, k0, NG, GS, BLOCK_K)
    offs = pid * BLOCK_I + tl.arange(0, BLOCK_I)
    _silu_mul_store(a0, act_ptr, offs, BLOCK_I)
    if M > 1:
        _silu_mul_store(a1, act_ptr + stride_act, offs, BLOCK_I)
    if M > 2:
        _silu_mul_store(a2, act_ptr + 2 * stride_act, offs, BLOCK_I)
    if M > 3:
        _silu_mul_store(a3, act_ptr + 3 * stride_act, offs, BLOCK_I)


@triton.jit
def _silu_mul_store(acc, act_ptr, offs, BLOCK_I: tl.constexpr):
    # gate_up is BF16 in the unfused path; SiluAndMul computes in fp32.
    gu = tl.reshape(acc.to(tl.bfloat16).to(tl.float32), (2, BLOCK_I))
    sel = tl.arange(0, 2)[:, None]
    g = tl.sum(tl.where(sel == 0, gu, 0.0), axis=0)
    u = tl.sum(tl.where(sel == 1, gu, 0.0), axis=0)
    tl.store(act_ptr + offs, (g * tl.sigmoid(g) * u).to(tl.bfloat16))


@triton.jit
def _se_down_combine_kernel(
    act_ptr,
    lo_ptr,
    hi_ptr,
    s_ptr,
    routed_ptr,
    sgate_ptr,
    out_ptr,
    stride_act,
    stride_out,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    GS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr = 1,
):
    # out[t, n] = sum_k routed[t * TOPK + k, n] + sigmoid(g_t) * down(act_t)[n],
    # rounded like moe_sum (BF16) + sigmoid-gated shared output (BF16) + add.
    pid = tl.program_id(0)
    NG: tl.constexpr = BLOCK_K // GS
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    a0 = tl.zeros((BLOCK_N,), tl.float32)
    a1 = tl.zeros((BLOCK_N,), tl.float32)
    a2 = tl.zeros((BLOCK_N,), tl.float32)
    a3 = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in tl.range(0, K, BLOCK_K, num_stages=STAGES):
        w3 = tl.reshape(
            _i6_tile(lo_ptr, hi_ptr, rows, k0, K, BLOCK_N, BLOCK_K),
            (BLOCK_N, NG, GS),
        )
        sc = tl.load(
            s_ptr + rows[:, None] * (K // GS) + k0 // GS + tl.arange(0, NG)[None, :]
        ).to(tl.float32)
        a0 += _row_dot(w3, sc, act_ptr, k0, NG, GS, BLOCK_K)
        if M > 1:
            a1 += _row_dot(w3, sc, act_ptr + stride_act, k0, NG, GS, BLOCK_K)
        if M > 2:
            a2 += _row_dot(w3, sc, act_ptr + 2 * stride_act, k0, NG, GS, BLOCK_K)
        if M > 3:
            a3 += _row_dot(w3, sc, act_ptr + 3 * stride_act, k0, NG, GS, BLOCK_K)
    _combine_store(a0, 0, routed_ptr, sgate_ptr, out_ptr, rows, N, TOPK)
    if M > 1:
        _combine_store(
            a1, 1, routed_ptr, sgate_ptr, out_ptr + stride_out, rows, N, TOPK
        )
    if M > 2:
        _combine_store(
            a2, 2, routed_ptr, sgate_ptr, out_ptr + 2 * stride_out, rows, N, TOPK
        )
    if M > 3:
        _combine_store(
            a3, 3, routed_ptr, sgate_ptr, out_ptr + 3 * stride_out, rows, N, TOPK
        )


@triton.jit
def _combine_store(
    acc,
    t,
    routed_ptr,
    sgate_ptr,
    out_ptr,
    rows,
    N: tl.constexpr,
    TOPK: tl.constexpr,
):
    shared = acc.to(tl.bfloat16).to(tl.float32)
    g = tl.sigmoid(tl.load(sgate_ptr + t)).to(tl.bfloat16).to(tl.float32)
    shg = (shared * g).to(tl.bfloat16).to(tl.float32)
    r = tl.zeros_like(acc)
    for k in tl.static_range(TOPK):
        r += tl.load(routed_ptr + (t * TOPK + k) * N + rows).to(tl.float32)
    r = r.to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + rows, (r + shg).to(tl.bfloat16))


@triton.jit
def _w6a16_gemm_kernel(
    x_ptr,
    lo_ptr,
    hi_ptr,
    s_ptr,
    y_ptr,
    M,
    N,
    stride_x,
    stride_y,
    K: tl.constexpr,
    GS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.minimum(offs_n, N - 1)
    mmask = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for g in tl.range(0, K // GS):
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x + g * GS + tl.arange(0, GS)[None, :],
            mask=mmask[:, None],
            other=0.0,
        )
        w = _i6_tile(lo_ptr, hi_ptr, rows, g * GS, K, BLOCK_N, GS)
        sc = tl.load(s_ptr + rows * (K // GS) + g).to(tl.float32)
        acc = tl.dot(x, tl.trans((w * sc[:, None]).to(tl.bfloat16)), acc)
    tl.store(
        y_ptr + offs_m[:, None] * stride_y + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=mmask[:, None] & (offs_n < N)[None, :],
    )


def _w6a16_gemm(
    x: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    x = x.contiguous()
    M, K = x.shape
    N = lo.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    # Measured on RTX 3090 at M = 512 (a prefill chunk).
    if M <= 16:
        block_m, block_n = 16, 64
    elif M <= 64:
        block_m, block_n = 32, 64
    else:
        block_m, block_n = 128, (32 if N <= 1280 else 64)
    _w6a16_gemm_kernel[(triton.cdiv(M, block_m), triton.cdiv(N, block_n))](
        x,
        lo,
        hi,
        scale,
        y,
        M,
        N,
        x.stride(0),
        y.stride(0),
        K=K,
        GS=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return y


class _Int6PlanesScheme:
    """Replaces the compressed-tensors WNA16 scheme of a shared-expert INT6
    projection: the weight is kept as lo/hi planes instead of a Humming
    repack."""

    def __init__(self, group_size: int) -> None:
        self.group_size = group_size

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if getattr(layer, "weight_packed", None) is None and hasattr(layer, "i6_lo"):
            # Already relaid out (weight reload); the registered buffers keep
            # their storage, so captured decode graphs stay valid.
            return
        K = layer.input_size_per_partition
        lo, hi = _int6_to_planes(layer.weight_packed.data, K)
        # Registered as non-persistent buffers so weight-reload paths that
        # copy parameters and buffers in place (sleep-mode wake, layerwise
        # update) preserve the storage the decode CUDA graphs captured.
        layer.register_buffer("i6_lo", lo, persistent=False)
        layer.register_buffer("i6_hi", hi, persistent=False)
        layer.i6_scale = layer.weight_scale.data
        # The planes replace the packed weight (same bytes).
        layer.weight_packed = None

    def apply_weights(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert bias is None
        shape = x.shape
        y = _w6a16_gemm(
            x.reshape(-1, shape[-1]),
            layer.i6_lo,
            layer.i6_hi,
            layer.i6_scale,
            self.group_size,
        )
        return y.view(*shape[:-1], y.shape[-1])


def _int6_planes_group_size(linear: nn.Module) -> int | None:
    """Group size if the projection qualifies for the INT6 plane layout.

    Checked for every projection before any scheme swap (both or neither).
    """
    scheme = getattr(linear, "scheme", None)
    if scheme is None or type(scheme).__name__ != "CompressedTensorsWNA16":
        return None
    gs = getattr(scheme, "group_size", -1)
    K = linear.input_size_per_partition
    if (
        scheme.num_bits != 6
        or not scheme.symmetric
        or gs != 64
        or K % 128
        or linear.output_size_per_partition % 16
    ):
        return None
    return gs


def _use_int6_planes(linear: nn.Module) -> bool:
    """Swap in _Int6PlanesScheme for a qualifying projection.

    Decide eligibility for every projection first (via
    _int6_planes_group_size); this mutates ``linear.scheme`` as a side
    effect.
    """
    gs = _int6_planes_group_size(linear)
    if gs is None:
        return False
    linear.scheme = _Int6PlanesScheme(gs)
    return True


class Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """Qwen3Next MoE with Qwen4Exp HC validation."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp HC does not support sequence-parallel MoE"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)

        self._has_lora = vllm_config.lora_config is not None
        # Decided on the first forward, once the weights are loaded.
        self._decode_state: dict | None = None
        self._shared_int6_decode = False
        self._decode_checked = not _moe_fused_decode_enabled()
        # The INT6 plane layout has to be chosen before the weights load.
        self._shared_int6 = False
        if (
            _moe_fused_decode_enabled()
            and self.shared_expert is not None
            and not self.replicate_shared_expert
        ):
            gu = self.shared_expert.gate_up_proj
            dn = self.shared_expert.down_proj
            if (
                _int6_planes_group_size(gu) is not None
                and _int6_planes_group_size(dn) is not None
            ):
                self._shared_int6 = _use_int6_planes(gu) and _use_int6_planes(dn)

    def _setup_fused_decode(self) -> None:
        """Enable the decode path only for the exact configuration it covers."""
        self._decode_checked = True
        from vllm.scalar_type import scalar_types

        experts = self.experts
        routed = getattr(experts, "routed_experts", None)
        qm = getattr(routed, "quant_method", None)
        shared = self.shared_expert
        reasons = []
        if routed is None or qm is None:
            reasons.append("no routed experts")
        else:
            if type(qm).__name__ != "CompressedTensorsWNA16MoEMethod" or not getattr(
                qm, "is_marlin", False
            ):
                reasons.append(f"quant method {type(qm).__name__}")
            if getattr(qm, "num_bits", None) != 4 or not getattr(
                qm, "symmetric", False
            ):
                reasons.append("not symmetric INT4")
            if routed.expert_map is not None or self.enable_eplb:
                reasons.append("expert map / EPLB")
            if routed.local_num_experts != routed.global_num_experts:
                reasons.append("expert parallel")
            # The decode path skips FusedMoE's dispatch / combine and its
            # reductions, so it only covers the single-GPU MoE layout.
            pc = routed.moe_config.moe_parallel_config
            if pc.tp_size != 1 or pc.dp_size != 1 or pc.pcp_size != 1 or pc.use_ep:
                reasons.append("TP / DP / EP MoE")
            if (
                routed.scoring_func != "softmax"
                or routed.use_grouped_topk
                or routed.custom_routing_function is not None
                or routed.e_score_correction_bias is not None
                or routed.routed_scaling_factor != 1.0
                or routed.apply_router_weight_on_input
            ):
                reasons.append("routing variant")
            if routed.activation.name != "SILU":
                reasons.append(f"activation {routed.activation}")
            # RoutedExperts leaves the swiglu knobs at None by default; only
            # a set limit or a non-default alpha/beta (or the situ betas,
            # which live on the moe_config) rules the fused path out.
            if (
                getattr(routed, "swiglu_limit", None) is not None
                or getattr(routed, "swiglu_alpha", None) not in (None, 1.0)
                or getattr(routed, "swiglu_beta", None) not in (None, 0.0)
                or getattr(routed.moe_config, "activation_situ_beta", None) is not None
                or getattr(routed.moe_config, "activation_situ_linear_beta", None)
                is not None
            ):
                reasons.append("swiglu activation config")
            if getattr(routed, "w13_bias", None) is not None or hasattr(
                routed, "w13_weight_zero_point"
            ):
                reasons.append("bias / zero points")
            if getattr(qm, "input_quant", None) is not None:
                reasons.append("static input quantization")
            if routed.global_num_experts & (routed.global_num_experts - 1):
                reasons.append("expert count not a power of two")
            # The alignment loop above writes at most four expert_ids blocks
            # per expert and the router grid needs E // ROWS >= 1 programs.
            if (
                routed.top_k * _MOE_DECODE_MAX_TOKENS > 32
                or routed.global_num_experts < 2
            ):
                reasons.append(f"top-k {routed.top_k} / expert count too small")
        if shared is None or shared.expert_gate is None or self.replicate_shared_expert:
            reasons.append("shared expert layout")
        if type(self.gate.quant_method).__name__ != "UnquantizedLinearMethod":
            reasons.append("quantized router")
        if self._has_lora:
            reasons.append("LoRA")
        hidden = self.gate.weight.shape[1]
        if hidden % 512:
            reasons.append(f"hidden size {hidden}")
        # The fused kernels round and store at BF16 points.
        if self.gate.weight.dtype != torch.bfloat16:
            reasons.append(f"params dtype {self.gate.weight.dtype}")
        if reasons:
            logger.info_once(
                "Qwen4Exp fused MoE decode disabled: %s", ", ".join(reasons)
            )
            return
        # The INT6 decode kernels run unmasked tiles: 512 columns of the
        # hidden size, 128 of the intermediate size. Otherwise the planes are
        # only used through the GEMM path.
        inter = shared.down_proj.input_size_per_partition
        self._shared_int6_decode = (
            self._shared_int6 and hidden % 512 == 0 and inter % 128 == 0
        )
        device = self.gate.weight.device
        self._decode_state = dict(
            routed=routed,
            quant_type=scalar_types.uint4b8,
            top_k=routed.top_k,
            renormalize=bool(routed.renormalize),
            num_experts=routed.global_num_experts,
            gate_weight=self.gate.weight,
            shared_gate_weight=shared.expert_gate.weight.reshape(-1),
            ticket=torch.zeros(1, dtype=torch.int32, device=device),
        )
        logger.info_once(
            "Qwen4Exp fused MoE decode enabled (up to %d tokens)",
            _MOE_DECODE_MAX_TOKENS,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        if not self._decode_checked:
            self._setup_fused_decode()
        num_tokens = hidden_states.shape[0]
        if (
            self._decode_state is not None
            and hidden_states.dim() == 2
            and 0 < num_tokens <= _MOE_DECODE_MAX_TOKENS
        ):
            return self._forward_decode(hidden_states)
        return super().forward(hidden_states, already_sequence_parallel)

    def _forward_decode(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
            _fused_marlin_moe,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            get_marlin_input_dtype,
        )

        st = self._decode_state
        routed = st["routed"]
        x = x.contiguous()
        M, K = x.shape
        E, topk = st["num_experts"], st["top_k"]
        # fused_marlin_moe's block size choice.
        for block_size_m in (8, 16, 32, 48, 64):
            if M * topk / E / block_size_m < 0.9:
                break
        input_dtype = get_marlin_input_dtype()
        if input_dtype is not None and input_dtype.itemsize == 1:
            block_size_m = max(block_size_m, 16)
        dev = x.device
        numel = M * topk
        max_pad = min(numel * block_size_m, numel + E * (block_size_m - 1))
        logits = torch.empty((M, E), device=dev, dtype=torch.float32)
        sgate = torch.empty((M,), device=dev, dtype=torch.float32)
        topk_w = torch.empty((M, topk), device=dev, dtype=torch.float32)
        topk_ids = torch.empty((M, topk), device=dev, dtype=torch.int32)
        sorted_ids = torch.empty((max_pad,), device=dev, dtype=torch.int32)
        expert_ids = torch.empty(
            (triton.cdiv(max_pad, block_size_m),), device=dev, dtype=torch.int32
        )
        ntpp = torch.empty((1,), device=dev, dtype=torch.int32)
        rows = 2
        _moe_router_kernel[(E // rows + 1,)](
            x,
            st["gate_weight"],
            st["shared_gate_weight"],
            logits,
            sgate,
            st["ticket"],
            topk_w,
            topk_ids,
            sorted_ids,
            expert_ids,
            ntpp,
            x.stride(0),
            M,
            K=K,
            E=E,
            TOPK=topk,
            BS=block_size_m,
            ROWS=rows,
            BLOCK_K=1024,
            MPAD=triton.next_power_of_2(M),
            P=triton.next_power_of_2(topk),
            RENORM=st["renormalize"],
            num_warps=4,
        )
        shared = self.shared_expert
        if self._shared_int6_decode:
            gu = shared.gate_up_proj
            inter = gu.i6_lo.shape[0] // 2
            act = torch.empty((M, inter), device=dev, dtype=torch.bfloat16)
            block_i = 4
            _se_gate_up_act_kernel[(inter // block_i,)](
                x,
                gu.i6_lo,
                gu.i6_hi,
                gu.i6_scale,
                act,
                x.stride(0),
                act.stride(0),
                M=M,
                K=K,
                INTER=inter,
                GS=gu.scheme.group_size,
                BLOCK_I=block_i,
                BLOCK_K=512,
                STAGES=3,
                num_warps=2,
            )
        else:
            gate_up, _ = shared.gate_up_proj(x)
            shared_out, _ = shared.down_proj(shared.act_fn(gate_up))
        routed_out = _fused_marlin_moe(
            hidden_states=x,
            w1=routed.w13_weight,
            w2=routed.w2_weight,
            bias1=None,
            bias2=None,
            w1_scale=routed.w13_weight_scale,
            w2_scale=routed.w2_weight_scale,
            topk_weights=topk_w,
            num_topk=topk,
            quant_type=st["quant_type"],
            apply_router_weight_on_input=False,
            expert_map=None,
            block_size_m=block_size_m,
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=ntpp,
            activation=routed.activation,
            topk_ids=topk_ids,
            workspace=getattr(routed, "workspace", None),
            input_dtype=input_dtype,
        )
        out = torch.empty_like(x)
        if self._shared_int6_decode:
            dn = shared.down_proj
            block_n = 8
            _se_down_combine_kernel[(K // block_n,)](
                act,
                dn.i6_lo,
                dn.i6_hi,
                dn.i6_scale,
                routed_out,
                sgate,
                out,
                act.stride(0),
                out.stride(0),
                M=M,
                K=act.shape[1],
                N=K,
                GS=dn.scheme.group_size,
                TOPK=topk,
                BLOCK_N=block_n,
                BLOCK_K=128,
                STAGES=3,
                num_warps=1,
            )
            return out
        block = 512
        _moe_combine_kernel[(M, triton.cdiv(K, block))](
            routed_out,
            shared_out,
            sgate,
            out,
            K=K,
            TOPK=topk,
            BLOCK=block,
            num_warps=4,
        )
        return out


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp HC does not support sequence-parallel MoE"
            )
        self.ple: Qwen4ExpPLELayer | None = None
        ple_layer_ids = config.ple_layer_ids
        if (self.layer_idx + 1) in ple_layer_ids:
            ple_layer_ids_sorted = sorted(set(ple_layer_ids))
            ple_dense_layer_id_map = {
                abs_id: idx for idx, abs_id in enumerate(ple_layer_ids_sorted)
            }
            ple_dense_layer_id = ple_dense_layer_id_map[self.layer_idx + 1]
            self.ple = Qwen4ExpPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=ple_dense_layer_id,
                prefix=f"{prefix}.ple",
            )

        if layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
        elif layer_type == "full_attention":
            use_qsa = getattr(config, "indexer_n_heads", None) is not None
            if not use_qsa:
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                self.self_attn = Qwen4ExpQSAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        num_experts = getattr(config, "num_experts", 0) or 0
        absolute_layer_id = self.layer_idx + 1
        is_moe_layer = self.layer_idx not in mlp_only_layers and (
            num_experts > 0 and absolute_layer_id % config.decoder_sparse_step == 0
        )
        if is_moe_layer:
            self.mlp = Qwen4ExpSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidual(
            hc_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "attn_hyper_connection"),
        )
        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor | None,
        prev_injection: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prev_block_output is None:
            assert prev_injection is None
        attn_hc = self.attn_hyper_connection
        if self.ple is not None:
            # PLE adds directly to the multi-stream state, so pending HC state
            # must be materialized before the addition.
            if prev_block_output is not None:
                hidden_states = attn_hc.combine(
                    hidden_states, prev_block_output, prev_injection
                )
                prev_block_output = prev_injection = None

            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            hidden_states = hidden_states + self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )

        # Fuse a pending combine with this HC module's mix when possible.
        if prev_block_output is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(
                hidden_states, prev_block_output, prev_injection
            )
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)

        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states=block_input)
        elif self.layer_type == "full_attention":
            attn_out = self.self_attn(
                hidden_states=block_input,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")

        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(
            hidden_states, attn_out, injection
        )
        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection


class Qwen4ExpMixtureOfExperts(MixtureOfExperts):
    """Expose Qwen4Exp routed experts through vLLM's EPLB protocol."""

    def set_moe_parameters(self, layers: Iterable[nn.Module]) -> None:
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in layers:
            if isinstance(layer, Qwen4ExpDecoderLayer) and isinstance(
                layer.mlp, Qwen4ExpSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.num_moe_layers = len(self.moe_layers)
        if example_moe is None:
            self.num_expert_groups = 0
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        self.num_expert_groups = 1
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_physical_experts = num_physical_experts
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class Qwen4ExpModel(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self.vocab_size = config.vocab_size
        self._qsa_layer_ids = frozenset(
            layer_idx
            for layer_idx, layer_type in enumerate(config.layer_types)
            if layer_type == "full_attention"
            and getattr(config, "indexer_n_heads", None) is not None
        )
        # Only the first PP rank embeds tokens; later ranks would hold an
        # unused ~0.6 GiB copy of the (quantized) table.
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "embed_tokens"),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str) -> Qwen4ExpDecoderLayer:
            layer_idx = extract_layer_index(prefix)
            return Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[layer_idx],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.is_fused_shared_expert_enabled = is_model_fused_shared_expert_compatible(
            self.layers,
            Qwen4ExpSparseMoeBlock,
            "mlp",
        )
        intermediate_size = config.hidden_size * config.hc_count
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], intermediate_size
        )

        self.hyper_connection_mixer: GatedResidual | None
        if get_pp_group().is_last_rank:
            hc_config = HyperConnectionConfig(
                hc_count=config.hc_count,
                hidden_size=config.hidden_size,
                params_dtype=torch.bfloat16,
                hc_lowrank=config.hc_lowrank,
                rms_norm_eps=config.rms_norm_eps,
                hc_per_branch_norm=True,
            )
            self.hyper_connection_mixer = GatedResidual(
                hc_config,
                use_combine=False,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
            )
        else:
            self.hyper_connection_mixer = None

        spec_config = vllm_config.speculative_config
        # MTP HC multi-stream outputs: when speculative method=="mtp" and the
        # model uses HC with hc_count>1, retain the pre-final-mixer multi-stream
        # hidden state [T, hc_count*H] so the MTP drafter can feed a real
        # multi-stream backbone hidden on its first step (scheme A). Derived
        # purely from config (NOT node identity) so P/D nodes stay consistent.
        needs_mtp_hidden = (
            spec_config is not None
            and getattr(spec_config, "method", None) == "mtp"
            and get_pp_group().is_last_rank
        )
        if needs_mtp_hidden:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.hc_count * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    @staticmethod
    def _start_layer_ple_prefetch(
        layer: nn.Module,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
    ) -> None:
        """Start a layer's PLE prefetch when the required inputs exist."""
        ple: Qwen4ExpPLELayer | None = getattr(layer, "ple", None)
        if ple is None:
            return
        if input_ids is None or query_start_loc is None or ngram_context is None:
            raise RuntimeError("PLE inputs were not prepared")
        ple.start_prefetch(
            hidden_states,
            input_ids,
            query_start_loc,
            ngram_context,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        block_output = None
        injection = None
        last_layer = None
        if self.start_layer < self.end_layer:
            self._start_layer_ple_prefetch(
                self.layers[self.start_layer],
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )
        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            last_layer = layer
            if layer_idx + 1 < self.end_layer:
                self._start_layer_ple_prefetch(
                    self.layers[layer_idx + 1],
                    hidden_states,
                    input_ids,
                    query_start_loc,
                    ngram_context,
                )
            hidden_states, block_output, injection = layer(
                hidden_states=hidden_states,
                prev_block_output=block_output,
                prev_injection=injection,
                positions=positions,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                deepstack_embed = (
                    deepstack_embed.unsqueeze(-2)
                    .expand(
                        *deepstack_embed.shape[:-1],
                        self.config.hc_count,
                        self.config.hidden_size,
                    )
                    .flatten(-2)
                )
                # Deepstack is an external addition to the materialized
                # multi-stream state and therefore terminates delayed combine.
                hidden_states = layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
                block_output = None
                injection = None
                hidden_states = hidden_states + deepstack_embed

        if not get_pp_group().is_last_rank:
            # PP transports one tensor, not the delayed HC tuple. Materialize
            # with the HC module that produced the pending injection.
            if last_layer is not None and block_output is not None:
                hidden_states = last_layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
            return IntermediateTensors({"hidden_states": hidden_states})

        # The final mixer consumes the last pending combine and returns both
        # the sampled single stream and the materialized multi-stream state.
        final_mixer = self.hyper_connection_mixer
        assert final_mixer is not None
        multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(
            hidden_states, block_output, injection
        )
        if self._mtp_hidden_buffer is not None:
            # Capture the pre-final-mixer multi-stream hidden state
            # [T, hc_count*H] for the MTP drafter (zero extra compute:
            # this tensor is needed by the final mixer regardless).
            num_tokens = multi_hidden.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        return sample_hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = (
            (
                _remap_qsa_cache_scale_name(name, self._qsa_layer_ids),
                weight,
            )
            for name, weight in weights
        )
        weights = maybe_fuse_shared_experts(
            weights,
            enabled=self.is_fused_shared_expert_enabled,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        # Non-persistent PLE state rebuilt in __init__; skip any ckpt
        # column for them.
        skip_substrs = (
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        )
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={substr: None for substr in skip_substrs}
        )
        # The final HC mixer only exists on the last PP rank; earlier ranks
        # must drop its checkpoint weights instead of failing to place them.
        ignore_prefixes = (
            None
            if self.hyper_connection_mixer is not None
            else ["hyper_connection_mixer."]
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=ignore_prefixes,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(
            weights,
            mapper=mapper,
        )
        return loaded


class Qwen4ExpForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    Qwen4ExpMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "kv_proj": ["key_proj", "value_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."}
    )
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen4Exp currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.model = Qwen4ExpModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Only the last PP rank computes logits.
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)
        enable_qwen4_exp_low_latency_gemm(self, self.model_config.dtype)

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen4ExpModelState

        return Qwen4ExpModelState

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Forward kwargs unchanged so the runner's _maybe_add_ngram_kwargs
        # path (query_start_loc / ngram_context) reaches Qwen4ExpModel.
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @classmethod
    def get_ple_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_ple_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]:
        hf_config = vllm_config.model_config.hf_text_config
        conv_kernel_size = hf_config.ple_conv_kernel_size
        short_conv_dilation = hf_config.ngram_size
        conv_state_len = (conv_kernel_size - 1) * short_conv_dilation
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        hc_count = hf_config.hc_count
        hc_hidden_size = hf_config.hidden_size * hc_count
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=hc_hidden_size,
            conv_kernel=conv_state_len + 1,
            num_spec=num_spec,
        )

    @classmethod
    def get_gdn_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_gdn_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return cls.get_gdn_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return cls.get_gdn_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        copy_funcs_by_type = {
            MambaAttentionBackendEnum.GDN_ATTN: cls.get_mamba_state_copy_func(),
            MambaAttentionBackendEnum.SHORT_CONV: (
                MambaStateCopyFuncCalculator.short_conv_state_copy_func()
            ),
        }
        missing_types = mamba_types - copy_funcs_by_type.keys()
        assert not missing_types, f"missing state copy funcs for {missing_types}"
        return {
            mamba_type: copy_funcs_by_type[mamba_type] for mamba_type in mamba_types
        }

    @classmethod
    def get_mamba_specs_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[MambaSpec, ...]:
        """Return all MambaSpecs for this model (GDN layers + PLE layer).

        The PLE layer uses a separate short_conv MambaSpec whose page_size_bytes
        may exceed the GDN spec; callers should take the maximum.
        """
        return (
            MambaSpec(
                shapes=cls.get_gdn_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_gdn_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
            ),
            MambaSpec(
                shapes=cls.get_ple_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_ple_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
                tp_replicated=True,
            ),
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.model._mtp_hidden_buffer

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"mtp.": None}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)


class Qwen4ExpProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self) -> Qwen4ExpConfig:
        return self.ctx.get_hf_config(Qwen4ExpConfig)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen4ExpProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen4ExpForConditionalGeneration(
    Qwen3_5ForConditionalGeneration,
    HasInnerState,
    Qwen4ExpMixtureOfExperts,
):
    """Qwen3-VL vision tower backed by the Qwen4Exp language model."""

    requires_raw_input_tokens = True

    packed_modules_mapping = Qwen3_5ForConditionalGeneration.packed_modules_mapping | {
        "kv_proj": ["key_proj", "value_proj"],
    }

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen4ExpModelState

        return Qwen4ExpModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config: Qwen4ExpConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError(
                "Qwen4ExpForConditionalGeneration requires multimodal_config"
            )

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.language_model_only = multimodal_config.language_model_only
        if self.language_model_only:
            self.use_data_parallel = False
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            self._tokenizer = None
            self.visual = StageMissingLayer("vision_tower")
            self._tower_model_names = []
        else:
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            self._init_video_pruning(multimodal_config)
            self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=config.text_config.rms_norm_eps,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )

        self.use_deepstack = (
            not self.language_model_only
            and bool(config.vision_config.deepstack_visual_indexes)
            and not isinstance(self.visual, StageMissingLayer)
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
            self.deepstack_input_embeds_num_tokens = 0

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen4ExpForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        if not get_pp_group().is_first_rank and self.use_deepstack:
            assert self.language_model.model.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), (
                "start_layer should be greater than or equal to "
                "len(deepstack_visual_indexes)"
            )
        self.set_moe_parameters(self.language_model.model.layers)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if self.language_model_only:
            raise ValueError(
                "Qwen4Exp language_model_only does not accept multimodal embeddings"
            )

        is_multimodal = _require_is_multimodal(is_multimodal)
        if self.use_deepstack:
            deepstack_input_embeds, multimodal_embeddings = (
                self._compute_deepstack_embeds(
                    inputs_embeds=inputs_embeds,
                    multimodal_embeddings=multimodal_embeddings,
                    is_multimodal=is_multimodal,
                )
            )
        else:
            deepstack_input_embeds = None

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)
        return inputs_embeds

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.language_model.get_mtp_target_hidden_states()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            query_start_loc=kwargs.get("query_start_loc"),
            ngram_context=kwargs.get("ngram_context"),
            deepstack_input_embeds=deepstack_input_embeds,
        )
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"mtp.": None},
            orig_to_new_prefix={"visual.": None} if self.language_model_only else {},
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen4ExpForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen4ExpForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return Qwen4ExpForCausalLM.get_mamba_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        return Qwen4ExpForCausalLM.get_mamba_state_copy_funcs(mamba_types)

    @classmethod
    def get_mamba_specs_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[MambaSpec, ...]:
        return Qwen4ExpForCausalLM.get_mamba_specs_from_config(vllm_config)


__all__ = [
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMixtureOfExperts",
    "Qwen4ExpModel",
    "Qwen4ExpSparseMoeBlock",
]
