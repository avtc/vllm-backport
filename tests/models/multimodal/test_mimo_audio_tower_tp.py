# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_MIMO_AUDIO_TOWER_TP gating, tokenizer weight slicing, and the
parallel-branch forward equivalence of the audio tower's sharded linears."""

import pytest
import torch
import torch.nn as nn

from tests.utils import ensure_current_vllm_config
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.mimo_audio import (
    AudioEncoderAttention,
    AudioProjection,
    _audio_tower_tp_for_dims,
    _shard_state_dict_for_rank,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


@pytest.fixture(scope="module")
def tp1_env():
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        backend="nccl",
    )
    with ensure_current_vllm_config():
        initialize_model_parallel(tensor_model_parallel_size=1)
        yield


def test_tp_for_dims_gate_and_divisibility():
    # Gate off: always replicate.
    assert _audio_tower_tp_for_dims(8, (16, 1024, 4096), enabled=False) == 1
    # Gate on, all dims divisible: full TP.
    assert _audio_tower_tp_for_dims(8, (16, 1024, 4096), enabled=True) == 8
    # Gate on, heads not divisible (e.g. 20-head whisper at TP8): replicate
    # rather than shard a subset (row-parallel reduce is global-TP).
    assert _audio_tower_tp_for_dims(8, (20, 1024, 4096), enabled=True) == 1
    assert _audio_tower_tp_for_dims(8, (16, 1000, 4096), enabled=True) == 1
    # Single GPU: nothing to shard.
    assert _audio_tower_tp_for_dims(1, (16, 1024, 4096), enabled=True) == 1


def test_tp1_builds_plain_linear_by_default(tp1_env):
    attn = AudioEncoderAttention(embed_dim=64, num_heads=8)
    assert isinstance(attn.q_proj, nn.Linear)
    assert isinstance(attn.out_proj, nn.Linear)
    proj = AudioProjection(64, 256, 64)
    assert all(isinstance(m, nn.Linear) for m in proj.mlp if isinstance(m, nn.Linear))


def _tiny_sharded_module():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            # Column-parallel: rows sharded, bias sharded with it.
            self.col = nn.Linear(8, 8)
            # Row-parallel: columns sharded, bias replicated.
            self.row = nn.Linear(8, 8)
            # Replicated: full shapes both sides.
            self.rep = nn.Linear(8, 8)

    m = M()
    with torch.no_grad():
        m.col.weight = nn.Parameter(torch.empty(4, 8))
        m.col.bias = nn.Parameter(torch.empty(4))
        m.row.weight = nn.Parameter(torch.empty(8, 4))
    return m


def test_shard_state_dict_for_rank():
    m = _tiny_sharded_module()
    ckpt = {
        "col.weight": torch.arange(64, dtype=torch.float32).view(8, 8),
        "col.bias": torch.arange(8, dtype=torch.float32),
        "row.weight": torch.arange(32, dtype=torch.float32).view(8, 4) * 10,
        "row.bias": torch.arange(8, dtype=torch.float32),
        "rep.weight": torch.zeros(8, 8),
        "decoder.unused.weight": torch.zeros(8, 8),  # dropped by strict=False
    }

    for rank in (0, 1):
        out = _shard_state_dict_for_rank(m, ckpt, tp_rank=rank)
        assert torch.equal(
            out["col.weight"], ckpt["col.weight"][rank * 4 : (rank + 1) * 4]
        )
        assert torch.equal(out["col.bias"], ckpt["col.bias"][rank * 4 : (rank + 1) * 4])
        assert torch.equal(
            out["row.weight"], ckpt["row.weight"][:, rank * 2 : (rank + 1) * 2]
        )
        assert torch.equal(out["row.bias"], ckpt["row.bias"])
        assert torch.equal(out["rep.weight"], ckpt["rep.weight"])
        assert torch.equal(out["decoder.unused.weight"], ckpt["decoder.unused.weight"])

    # Rank wraps modulo the per-tensor factor (subgroup-style replication).
    out2 = _shard_state_dict_for_rank(m, ckpt, tp_rank=5)
    assert torch.equal(out2["col.weight"], ckpt["col.weight"][4:8])

    # A shape that is neither a clean row nor column shard must raise.
    bad = dict(ckpt)
    bad["col.weight"] = torch.zeros(5, 8)
    with pytest.raises(RuntimeError, match="col.weight"):
        _shard_state_dict_for_rank(m, bad, tp_rank=0)


def test_parallel_projection_forward_matches_reference(tp1_env):
    """At world size 1 the parallel branch must be numerically identical to
    the plain nn.Sequential path with the same weights (GELU mid-activation)."""
    torch.manual_seed(0)
    parallel = AudioProjection(64, 256, 64, tp_size=2)
    reference = nn.Sequential(
        nn.Linear(64, 256, bias=False), nn.GELU(), nn.Linear(256, 64, bias=False)
    )
    assert isinstance(parallel.mlp[0], ColumnParallelLinear)
    assert isinstance(parallel.mlp[2], RowParallelLinear)
    with torch.no_grad():
        reference[0].weight.copy_(parallel.mlp[0].weight)
        reference[2].weight.copy_(parallel.mlp[2].weight)
    x = torch.randn(17, 64)
    torch.testing.assert_close(parallel(x), reference(x))


def test_parallel_attention_projections_match_at_tp1(tp1_env):
    """Head-sharded construction at world size 1 keeps full-size projections
    whose outputs match a plain nn.Linear with the same weights."""
    torch.manual_seed(1)
    attn = AudioEncoderAttention(embed_dim=64, num_heads=8, tp_size=2)
    assert isinstance(attn.q_proj, ColumnParallelLinear)
    assert isinstance(attn.out_proj, RowParallelLinear)
    plain = nn.Linear(64, 64, bias=True)
    with torch.no_grad():
        plain.weight.copy_(attn.q_proj.weight)
        plain.bias.copy_(attn.q_proj.bias)
    x = torch.randn(9, 64)
    torch.testing.assert_close(attn.q_proj(x), plain(x))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_parallel_attention_full_forward_shapes(tp1_env):
    """End-to-end attention forward through the parallel branch (flash-attn
    varlen path) returns the full embed dim after the row-parallel reduce."""
    attn = AudioEncoderAttention(
        embed_dim=64, num_heads=8, tp_size=2, window_size=(-1, -1), causal=False
    ).to("cuda")
    cu = torch.tensor([0, 5, 9], dtype=torch.int32, device="cuda")
    x = torch.randn(9, 64, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", torch.bfloat16):
        out = attn(x, cu, 5)
    assert out.shape == (9, 64)
