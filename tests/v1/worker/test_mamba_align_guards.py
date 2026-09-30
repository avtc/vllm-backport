"""Align state-copy guards: impossible inputs skip instead of faulting.

The deferred PP postprocess can feed the fused mamba align copy collapsed
counts (dst_col -1), fresh -1 state columns, or freed table rows holding -1
block ids; without validation the table read yields an arbitrary block id
and the copy dereferences a wild address (the trapped pp4 IMA at
mamba_utils.py:377). Legitimate copies are unaffected and the diagnostic
register records the first offender.
"""

import pytest
import torch

from vllm.v1.worker.mamba_utils import postprocess_mamba_fused_kernel

try:
    from vllm.triton_utils import triton
except ImportError:
    triton = None

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="align guards drive Triton kernels"
)

pytestmark = pytest.mark.skipif(triton is None, reason="triton unavailable on this box")

NUM_STATES = 2  # state 0: conv, state 1: temporal
NUM_BLOCKS = 8
BLOCK_SIZE = 4
CONV_WIDTH = 4
TEMPORAL_INNER = 16  # int32 elements


class _AlignCtx:
    def __init__(self, block_ids, device):
        self.block_table = torch.tensor([block_ids], dtype=torch.int32, device=device)
        self.conv_state = torch.zeros(
            NUM_BLOCKS, CONV_WIDTH * 8, dtype=torch.int32, device=device
        )
        self.temporal_state = torch.zeros(
            NUM_BLOCKS, TEMPORAL_INNER, dtype=torch.int32, device=device
        )
        self.align_debug = torch.zeros(8, dtype=torch.int64, device=device)
        self.num_accepted = torch.zeros(1, dtype=torch.int32, device=device)
        self.state_idx = torch.zeros(1, dtype=torch.int32, device=device)
        self.num_computed = torch.zeros(1, dtype=torch.int32, device=device)
        self.num_accepted_out = torch.zeros(1, dtype=torch.int32, device=device)

    def run(self, num_accepted, state_idx, new_num_computed):
        self.num_accepted.fill_(num_accepted)
        self.state_idx.fill_(state_idx)
        self.num_computed.fill_(new_num_computed)
        postprocess_mamba_fused_kernel[(1, NUM_STATES, 1)](
            self.num_accepted,
            self.state_idx,
            None,  # num_scheduled: unused under PRECOMPUTED_NEW_COMPUTED
            self.num_computed,
            None,  # num_draft: unused under PRECOMPUTED_NEW_COMPUTED
            torch.tensor([self.block_table.data_ptr()], dtype=torch.int64).to(
                self.block_table.device
            ),
            self.block_table.shape[1],
            torch.tensor(
                [self.conv_state.data_ptr(), self.temporal_state.data_ptr()],
                dtype=torch.int64,
            ).to(self.block_table.device),
            torch.tensor(
                [self.conv_state[0].numel() * 4, TEMPORAL_INNER * 4],
                dtype=torch.int64,
            ).to(self.block_table.device),
            torch.tensor([4, 4], dtype=torch.int32).to(self.block_table.device),
            torch.tensor([CONV_WIDTH * 8, TEMPORAL_INNER], dtype=torch.int64).to(
                self.block_table.device
            ),
            torch.tensor([CONV_WIDTH, 0], dtype=torch.int32).to(
                self.block_table.device
            ),
            torch.tensor([0, 0], dtype=torch.int32).to(self.block_table.device),
            torch.tensor([0, 0], dtype=torch.int32).to(self.block_table.device),
            torch.tensor([0, 0], dtype=torch.int64).to(self.block_table.device),
            self.align_debug,
            self.num_accepted_out,
            None,  # idx_mapping: batch order == req order
            1,
            block_size=BLOCK_SIZE,
            COPY_BLOCK_SIZE=1024,
            CONV_STATE_DIM_FIRST=False,
            HAS_IDX_MAPPING=False,
            PRECOMPUTED_NEW_COMPUTED=True,
            TEMPORAL_TILES=1,
        )


@requires_cuda
def test_align_copy_valid_untouched():
    ctx = _AlignCtx([5, 7], "cuda")
    ctx.conv_state[5].fill_(0x1234)
    ctx.temporal_state[5].fill_(0xABCD)
    # num_accepted=1 -> bias 0; new_computed=8 -> dst col 1; src col 0+0=0.
    ctx.run(num_accepted=1, state_idx=0, new_num_computed=8)
    torch.cuda.synchronize()
    assert torch.equal(ctx.temporal_state[7], ctx.temporal_state[5])
    assert torch.equal(ctx.conv_state[7], ctx.conv_state[5])
    assert ctx.align_debug[0].item() == 0, "guard must not trip on valid copy"


@requires_cuda
def test_align_copy_negative_dst_col_skips():
    ctx = _AlignCtx([5, 7], "cuda")
    # new_computed=0 -> dst_col = -1: freed-slot count collapse.
    ctx.run(num_accepted=1, state_idx=0, new_num_computed=0)
    torch.cuda.synchronize()
    assert ctx.align_debug[0].item() >= NUM_STATES
    assert ctx.align_debug[1].item() >= 1  # dst_col guard kind


@requires_cuda
def test_align_copy_fresh_state_skips():
    ctx = _AlignCtx([5, 7], "cuda")
    # state_idx -1 (request never ran a forward) must not read the table.
    ctx.run(num_accepted=1, state_idx=-1, new_num_computed=8)
    torch.cuda.synchronize()
    assert ctx.align_debug[0].item() >= NUM_STATES
    assert ctx.align_debug[2].item() >= 1  # src_col guard kind


@requires_cuda
def test_align_copy_freed_block_id_skips():
    ctx = _AlignCtx([-1, 7], "cuda")
    # dst col 1 valid, but the row holds a freed -1 id.
    ctx.run(num_accepted=1, state_idx=0, new_num_computed=8)
    torch.cuda.synchronize()
    assert ctx.align_debug[0].item() >= NUM_STATES
    assert ctx.align_debug[5].item() >= 1  # src-id guard kind (temporal+conv)
