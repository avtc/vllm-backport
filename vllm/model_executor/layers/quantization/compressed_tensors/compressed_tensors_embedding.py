# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantized embedding method for compressed-tensors.

Adds dequant-on-lookup support for a pack-quantized ``VocabParallelEmbedding``
(2-8 bit INT, channel- or group-quantized). Only the gathered token rows are
unpacked and dequantized, so the packed weight is never densified.
"""

import os

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from vllm import envs
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.triton_utils import tl, triton
from vllm.utils import init_logger

__all__ = ["CompressedTensorsEmbeddingWNA16Int"]

logger = init_logger(__name__)


@triton.jit
def _dequant_gather_kernel(
    ids_ptr,
    packed_ptr,
    scale_ptr,
    out_ptr,
    hidden,
    packed_cols,
    num_groups,
    num_rows,
    oob_ptr,
    NUM_BITS: tl.constexpr,
    PACK_FACTOR: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gather embedding rows by token id, unpack int32-packed INT weights, and
    dequantize to ``out`` dtype in one pass (no int8 intermediate).

    Out-of-range token ids are clamped to row 0 and recorded (count + first
    raw value) instead of faulting the gather: an id that escapes the vocab
    turns an illegal memory access into a wrong-but-visible token, and the
    recorded raw value fingerprints the producer (deferred-state races on
    pipeline ranks can reconstruct input ids from not-yet-written state).
    """
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    col_mask = col < hidden
    raw = tl.load(ids_ptr + row).to(tl.int64)
    if (raw < 0) | (raw >= num_rows):
        tl.atomic_add(oob_ptr, 1)
        # int32-saturated fingerprint: int64 atomic_min is not universally
        # supported, and any out-of-range id's origin is identifiable from a
        # saturated value (-2**31 = large negative, 2**31-1 = large positive).
        tl.atomic_min(
            oob_ptr + 1, tl.minimum(tl.maximum(raw, -(2**31)), 2**31 - 1).to(tl.int32)
        )
    tid = tl.maximum(tl.minimum(raw, num_rows - 1), 0)

    packed_idx = col // PACK_FACTOR
    shift = (col % PACK_FACTOR) * NUM_BITS
    packed = tl.load(
        packed_ptr + tid * packed_cols + packed_idx, mask=col_mask, other=0
    )
    q = ((packed >> shift) & ((1 << NUM_BITS) - 1)) - (1 << (NUM_BITS - 1))

    if GROUP_SIZE == 0:  # channel: one scale per row
        scale = tl.load(scale_ptr + tid)
    else:  # group: one scale per (row, group)
        grp = col // GROUP_SIZE
        scale = tl.load(scale_ptr + tid * num_groups + grp, mask=col_mask, other=0.0)

    out = q.to(tl.float32) * scale.to(tl.float32)
    tl.store(
        out_ptr + row * hidden + col, out.to(out_ptr.dtype.element_ty), mask=col_mask
    )


_OOB_BUFFERS: dict[tuple[int, torch.device], torch.Tensor] = {}
_OOB_WARNED = False


def _dequant_gather_triton(
    ids: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    hidden: int,
    num_bits: int,
) -> torch.Tensor:
    n = ids.numel()
    out = torch.empty(n, hidden, dtype=weight_scale.dtype, device=weight_packed.device)
    num_groups = weight_scale.shape[1]
    group_size = 0 if num_groups == 1 else hidden // num_groups
    block = min(triton.next_power_of_2(hidden), 1024)
    grid = (n, triton.cdiv(hidden, block))
    key = (os.getpid(), weight_packed.device)
    oob = _OOB_BUFFERS.get(key)
    if oob is None:
        oob = torch.zeros(2, dtype=torch.int64, device=weight_packed.device)
        # min-slot starts at the sentinel maximum so atomic_min records real ids.
        oob[1].fill_(torch.iinfo(torch.int64).max)
        _OOB_BUFFERS[key] = oob
    _dequant_gather_kernel[grid](
        ids,
        weight_packed,
        weight_scale,
        out,
        hidden,
        weight_packed.shape[1],
        num_groups,
        weight_packed.shape[0],
        oob,
        NUM_BITS=num_bits,
        PACK_FACTOR=32 // num_bits,
        GROUP_SIZE=group_size,
        BLOCK=block,
    )
    global _OOB_WARNED
    if envs.VLLM_EMBED_GATHER_DEBUG and not _OOB_WARNED:
        count, first = oob.tolist()
        if count:
            _OOB_WARNED = True
            logger.warning(
                "quantized embedding gather clamped %d out-of-range token id(s) "
                "(first raw id: %d); the producer of that id is reconstructing "
                "input ids from unwritten/stale state",
                count,
                first,
            )
    return out


class CompressedTensorsEmbeddingWNA16Int(QuantizeMethodBase):
    def __init__(self, weight_quant: QuantizationArgs):
        self.num_bits = weight_quant.num_bits
        self.pack_factor = 32 // self.num_bits
        self.strategy = weight_quant.strategy
        self.group_size = weight_quant.group_size
        self.is_group = (
            self.strategy == QuantizationStrategy.GROUP.value
            and self.group_size is not None
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs["weight_loader"]
        # Embedding weight is [num_embeddings(vocab), embedding_dim(hidden)];
        # vocab is the output (partitioned) dim, hidden is the input dim.
        vocab_pp = sum(output_partition_sizes)
        hidden = input_size_per_partition
        layer.hidden_size = hidden

        weight_packed = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
            data=torch.empty(vocab_pp, hidden // self.pack_factor, dtype=torch.int32),
        )

        if self.is_group:
            assert hidden % self.group_size == 0
            weight_scale = GroupQuantScaleParameter(
                output_dim=0,
                input_dim=1,
                weight_loader=weight_loader,
                data=torch.empty(
                    vocab_pp, hidden // self.group_size, dtype=params_dtype
                ),
            )
        else:
            weight_scale = ChannelQuantScaleParameter(
                output_dim=0,
                weight_loader=weight_loader,
                data=torch.empty(vocab_pp, 1, dtype=params_dtype),
            )

        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
        )

        layer.register_parameter("weight_packed", weight_packed)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        ids = input_.reshape(-1).contiguous()
        hidden = layer.hidden_size
        deq = _dequant_gather_triton(
            ids, layer.weight_packed, layer.weight_scale, hidden, self.num_bits
        )
        return deq.reshape(*input_.shape, hidden)

    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "CompressedTensorsEmbeddingWNA16Int supports embedding lookup only"
        )
