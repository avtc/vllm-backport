# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Triton sparse MLA backend for SM80 (A100) / SM121 (GB10)."""

from typing import TYPE_CHECKING, ClassVar

import torch

from vllm import envs
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseBackend,
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
    XPUMLASparseMetadataBuilder,
)
from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    KV_SPLITS_CANDIDATES,
    triton_mla_sparse_attention,
)

if TYPE_CHECKING:
    from vllm.config.cache import CacheDType

# KV cache dtypes this backend can serve beyond the XPU base's bf16/fp16:
# fp8 e4m3 with a per-tensor scale (written by the shared
# concat_and_cache_mla op, scale = layer._k_scale). e5m2 and friends stay
# unsupported: the SM8x-safe byte decoder below assumes e4m3fn.
_FP8_KV_DTYPES = ("fp8", "fp8_e4m3")


@triton.jit
def _dequant_gather_fp8_rows_kernel(
    out_ptr,  # [total_rows, HEAD_DIM] bf16
    cache_ptr,  # fp8 e4m3 cache, flat rows at slot * HEAD_DIM bytes
    indices_ptr,  # [total_rows] int32 flat cache row ids, -1 = padding
    k_scale_ptr,  # scalar fp32 dequant scale
    num_rows,  # flat cache rows, for the bounds mask
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,  # elements per chunk, divides HEAD_DIM
    N_CHUNKS: tl.constexpr,  # HEAD_DIM // BLOCK
):
    """Gather-dequantize scattered fp8 rows into a flat bf16 workspace.

    SM8x Triton cannot load or bitcast fp8e4nv element types, so bytes are
    decoded via the shared fp8_sm80 helper (NaN-preserving, SM89+ fast
    path). Global row ids address the cache as a flat contiguous row array
    (the hybrid allocator never selects inter-block gap rows), so the byte
    offset is simply slot * HEAD_DIM. Indices outside [0, num_rows) are
    treated like -1 padding: zero-filled, then masked by the attention
    kernel's remapped -1.
    """
    pid = tl.program_id(0)
    slot = tl.load(indices_ptr + pid).to(tl.int64)

    ks = tl.load(k_scale_ptr)

    out_row = out_ptr + pid.to(tl.int64) * HEAD_DIM

    if (slot < 0) | (slot >= num_rows):
        for c in tl.static_range(N_CHUNKS):
            offs = c * BLOCK + tl.arange(0, BLOCK)
            zero = tl.zeros([BLOCK], dtype=tl.bfloat16)
            tl.store(out_row + offs, zero)
        return

    base = cache_ptr + slot * HEAD_DIM
    for c in tl.static_range(N_CHUNKS):
        offs = c * BLOCK + tl.arange(0, BLOCK)
        x_uint8 = tl.load(base + offs)
        x_float = _decode_fp8_f32(x_uint8, False)
        tl.store(out_row + offs, (x_float * ks).to(tl.bfloat16))


def dequant_gather_fp8_rows(
    out: torch.Tensor,  # [total_rows, head_dim] bf16
    cache: torch.Tensor,  # [num_blocks, block_size, head_dim] fp8 e4m3
    indices: torch.Tensor,  # [total_rows] int32 flat row ids (-1 = pad)
    k_scale: torch.Tensor,  # scalar fp32
) -> None:
    """Gather-dequantize fp8 cache rows at ``indices`` into bf16 ``out``."""
    total_rows = indices.shape[0]
    if total_rows == 0:
        return
    head_dim = cache.shape[-1]
    assert head_dim % 64 == 0, f"head_dim must be 64-aligned, got {head_dim}"
    assert cache.dtype == torch.float8_e4m3fn, cache.dtype
    # Triton on SM8x cannot handle fp8e4nv pointer types at all; view the
    # storage as raw bytes and decode e4m3fn in the kernel.
    cache_u8 = cache.view(torch.uint8)
    num_rows = cache_u8.numel() // head_dim
    _dequant_gather_fp8_rows_kernel[(total_rows,)](
        out,
        cache_u8,
        indices,
        k_scale,
        num_rows,
        HEAD_DIM=head_dim,
        BLOCK=64,
        N_CHUNKS=head_dim // 64,
    )


def _fp8_kv_subbatched_forward(
    q: torch.Tensor,  # [num_tokens, heads, head_dim] bf16
    kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, block_size, head_dim] fp8
    topk_indices_global: torch.Tensor,  # [num_tokens, topk] int32 row ids
    k_scale: torch.Tensor,
    softmax_scale: float,
    sm_count: int | None,
) -> torch.Tensor:
    """fp8 gather-dequant + bf16 sparse attention, sub-batched by budget.

    The workspace is [tokens, topk, head_dim] bf16: ~34 MB for decode-sized
    batches, but ~1.1 GiB per layer at prefill shape (512 tokens x 2176 topk
    x 512). Sub-batch tokens under a byte budget
    (VLLM_TRITON_MLA_SPARSE_FP8_GATHER_MB, default 64 MiB) and reuse one
    workspace allocation; decode batches always fit a single sub-batch.
    """
    flat_indices = topk_indices_global.reshape(-1)
    num_tokens = q.shape[0]
    head_dim = kv_c_and_k_pe_cache.shape[-1]
    topk_width = topk_indices_global.shape[-1]
    budget_rows = max(
        1,
        envs.VLLM_TRITON_MLA_SPARSE_FP8_GATHER_MB * 1024 * 1024
        // (topk_width * head_dim * 2),
    )
    rows_per_subbatch = min(num_tokens, budget_rows) * topk_width
    gathered = torch.empty(
        (rows_per_subbatch, head_dim), dtype=torch.bfloat16, device=q.device
    )
    outs = []
    for t0 in range(0, num_tokens, budget_rows):
        t1 = min(t0 + budget_rows, num_tokens)
        sub = flat_indices[t0 * topk_width : t1 * topk_width]
        dequant_gather_fp8_rows(
            gathered[: sub.numel()], kv_c_and_k_pe_cache, sub, k_scale
        )
        remapped = remap_to_gather_rows(sub).view(t1 - t0, 1, -1)
        outs.append(
            triton_mla_sparse_attention(
                q[t0:t1],
                gathered[: sub.numel()].view(-1, 1, head_dim),
                remapped,
                sm_scale=softmax_scale,
                sm_count=sm_count,
            )
        )
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)


def remap_to_gather_rows(flat_indices: torch.Tensor) -> torch.Tensor:
    """Remap flat global row ids to ids into the gather-dequant workspace.

    Rows are gathered token-major, so gathered row i came from flat slot i;
    -1 padding must survive so the bf16 kernel keeps masking those columns.
    """
    device = flat_indices.device
    return torch.where(
        flat_indices >= 0,
        torch.arange(flat_indices.numel(), dtype=torch.int32, device=device),
        flat_indices,
    )


class TritonMLASparseMetadataBuilder(XPUMLASparseMetadataBuilder):
    # XPU base keeps NEVER (not validated under cudagraph); this subclass
    # claims UNIFORM_BATCH for the CUDA/Triton path.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class TritonMLASparseImpl(XPUMLASparseImpl):
    """Triton sparse-MLA impl with split-KV decode (3-7× faster than the
    single-pass XPU base for single-query decode on SM80 / SM121)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sm_count: int | None = None
        if self.topk_indices_buffer is not None:
            self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)
        self._warmup_autotune()

    def _warmup_autotune(self) -> None:
        """Prime `@triton.autotune` caches at init so the first request
        doesn't pay the inline config-sweep cost."""
        if self.topk_indices_buffer is None:
            return
        device = self.topk_indices_buffer.device
        topk = self.topk_indices_buffer.shape[-1]
        dim_qk = self.head_size
        q = torch.empty(1, self.num_heads, dim_qk, dtype=torch.bfloat16, device=device)
        kv = torch.empty(64, 1, dim_qk, dtype=torch.bfloat16, device=device)
        indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
        for splits in KV_SPLITS_CANDIDATES:
            triton_mla_sparse_attention(
                q,
                kv,
                indices,
                sm_scale=self.softmax_scale,
                num_kv_splits=splits,
                sm_count=self._sm_count,
            )

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]
        topk_indices: torch.Tensor,  # [sq, topk]
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = triton_mla_sparse_attention(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            sm_scale=self.softmax_scale,
            sm_count=self._sm_count,
        )
        return output

    def _forward_fp8_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk] bf16
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, block_size, d_qk] fp8
        topk_indices_global: torch.Tensor,  # [sq, topk] int32 flat row ids
        layer,  # AttentionLayer, provides the per-tensor _k_scale
    ) -> torch.Tensor:
        """fp8 KV read path: gather-dequant the selected rows, reuse the
        bf16 kernel. Sparse attention touches only the ~topk selected rows
        per token, so the dequant cost stays bounded regardless of context
        length."""
        return _fp8_kv_subbatched_forward(
            q,
            kv_c_and_k_pe_cache,
            topk_indices_global,
            layer._k_scale,
            self.softmax_scale,
            self._sm_count,
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: XPUMLASparseMetadata,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if is_quantized_kv_cache(self.kv_cache_dtype):
            if self.kv_cache_dtype not in _FP8_KV_DTYPES:
                raise NotImplementedError(
                    f"{self.kv_cache_dtype} KV cache is not supported by the "
                    "Triton sparse-MLA backend (fp8/fp8_e4m3 only)"
                )
            if isinstance(q, tuple):
                q = torch.cat(q, dim=-1)
            _, block_stride_rows = flat_kv_row_view(
                kv_c_and_k_pe_cache, attn_metadata.block_size
            )
            topk_indices_global = self._topk_global_indices(
                q.shape[0], block_stride_rows, attn_metadata
            )
            return (
                self._forward_fp8_kv(
                    q, kv_c_and_k_pe_cache, topk_indices_global, layer
                ),
                None,
            )
        return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)


class TritonMLASparseBackend(XPUMLASparseBackend):
    """Same bf16 sparse-MLA contract as the XPU backend, CUDA Triton kernels."""

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The DSA indexer backend requires block size 64 on CUDA and shares
        # the KV cache group with this backend; the base-class MultipleOf(1)
        # default lets auto-selection settle on 16, which then fails
        # select_common_block_size ("No common block size for 16").
        # MultipleOf(64) (rather than [64]) keeps larger user-specified
        # sizes like 128 usable, which measurably lowers profile-time peak
        # memory for very long contexts.
        return [MultipleOf(64)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 576 = 512 latent + 64 RoPE (DeepSeek-V3.2 / GLM-5).
        # 512 = NoPE MLA (GLM-5.3-Flash, qk_rope_head_dim = 0).
        return [512, 576]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: "CacheDType | None") -> bool:
        # fp8 e4m3 + per-tensor scale: the write path is the shared
        # concat_and_cache_mla op; the read path gather-dequantizes the
        # selected top-k rows to bf16 (see TritonMLASparseImpl.forward_mqa),
        # which needs no native fp8 Triton support and so works on SM8x.
        # ``--kv-cache-dtype fp8`` itself is the opt-in / A-B switch.
        if kv_cache_dtype in _FP8_KV_DTYPES:
            return True
        return super().supports_kv_cache_dtype(kv_cache_dtype)

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseMetadataBuilder"]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["TritonMLASparseImpl"]:
        return TritonMLASparseImpl
