# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the fp8 fused-QKV sharding in ``mimo_v2``.

Covers the two fp8 block-scale layouts found in MiMo-V2.6-Flash-RL
(per-group padded on the full-attention layers, globally packed on the
SWA layers) and KV-head replication for ``tp_size > num_kv_heads``.
``scaled_quantize`` is stubbed to an identity so the sharding/reorder
logic is compared exactly against a reference dequantization.
"""

import pytest
import torch

import vllm.model_executor.models.mimo_v2 as mimo_v2
from vllm.model_executor.models.mimo_v2 import _shard_fp8_qkv_proj

BLOCK = 128
COLS = 4096

# (label, num_heads, num_kv_heads, head_dim, v_head_dim, scale layout)
LAYER_CASES = [
    ("full_attn_padded", 64, 4, 192, 128, "padded"),
    ("swa_packed", 64, 8, 192, 128, "packed"),
]

TP_SIZES = [1, 2, 4, 8]


def _identity_scaled_quantize(w, group, dtype, compute_dtype=torch.float32):
    rows = -(-w.shape[0] // group[0])
    cols = -(-w.shape[1] // group[1])
    return w.to(compute_dtype), torch.ones(rows, cols, dtype=torch.float32)


@pytest.fixture
def stubbed_quantize(monkeypatch):
    monkeypatch.setattr(mimo_v2, "scaled_quantize", _identity_scaled_quantize)


def _make(num_heads, num_kv_heads, head_dim, v_head_dim, layout, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q_rows = (num_heads // num_kv_heads) * head_dim
    rpg = q_rows + head_dim + v_head_dim
    total = num_kv_heads * rpg
    w = (torch.randn(total, COLS, generator=gen) * 0.05).to(torch.float8_e4m3fn)
    if layout == "padded":
        s_rows = num_kv_heads * -(-rpg // BLOCK)
    else:
        s_rows = -(-total // BLOCK)
    s = torch.rand(s_rows, COLS // BLOCK, generator=gen) + 0.5
    return w, s


def _reference_groups(w, s, num_heads, num_kv_heads, head_dim, v_head_dim, layout):
    """Dequantize into [num_kv_heads, rows_per_group, cols]."""
    q_rows = (num_heads // num_kv_heads) * head_dim
    rpg = q_rows + head_dim + v_head_dim
    w32 = w.to(torch.float32)
    if layout == "padded":
        per_group = -(-rpg // BLOCK)
        parts = []
        for g in range(num_kv_heads):
            s_g = s[g * per_group : (g + 1) * per_group]
            exp = s_g.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[
                :rpg, :COLS
            ]
            parts.append(w32[g * rpg : (g + 1) * rpg] * exp)
        return torch.stack(parts)
    exp = s.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[: w.shape[0], :COLS]
    return (w32 * exp).reshape(num_kv_heads, rpg, COLS)


def _expected_rank_rows(ref, rank, tp, num_heads, num_kv_heads, head_dim, v_head_dim):
    q_rows = (num_heads // num_kv_heads) * head_dim
    rpg = q_rows + head_dim + v_head_dim
    if tp <= num_kv_heads:
        g0 = rank * (num_kv_heads // tp)
        g1 = g0 + num_kv_heads // tp
        q_off, q_rows_rank = 0, q_rows
    else:
        reps = tp // num_kv_heads
        g0 = rank // reps
        g1 = g0 + 1
        q_off = (rank % reps) * (q_rows // reps)
        q_rows_rank = q_rows // reps
    qs, ks, vs = [], [], []
    for g in range(g0, g1):
        qs.append(ref[g, q_off : q_off + q_rows_rank])
        ks.append(ref[g, q_rows : q_rows + head_dim])
        vs.append(ref[g, q_rows + head_dim : rpg])
    return torch.cat([torch.cat(qs), torch.cat(ks), torch.cat(vs)])


@pytest.mark.parametrize("label,nh,nk,hd,vd,layout", LAYER_CASES)
@pytest.mark.parametrize("tp", TP_SIZES)
def test_shard_fp8_qkv_proj_matches_reference(
    stubbed_quantize, label, nh, nk, hd, vd, layout, tp
):
    w, s = _make(nh, nk, hd, vd, layout)
    ref = _reference_groups(w, s, nh, nk, hd, vd, layout)
    q_rows = (nh // nk) * hd
    if tp <= nk:
        groups = nk // tp
        exp_rows = groups * (q_rows + hd + vd)
    else:
        exp_rows = q_rows // (tp // nk) + hd + vd

    for rank in range(tp):
        w_r, s_r = _shard_fp8_qkv_proj(
            w, s, nh, nk, hd, vd, tp_rank=rank, tp_size=tp
        )
        assert tuple(w_r.shape) == (exp_rows, COLS)
        assert s_r.shape[0] == -(-exp_rows // BLOCK)
        # Dequantize through the returned scales: exact for both paths (the
        # stubbed slow path returns all-ones scales).
        s_exp = s_r.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
        deq = w_r.to(torch.float32) * s_exp[:exp_rows, :COLS]
        exp = _expected_rank_rows(ref, rank, tp, nh, nk, hd, vd)
        assert torch.allclose(deq, exp, atol=1e-6, rtol=1e-5)


def test_shard_fp8_qkv_proj_rejects_unknown_scale_layout(stubbed_quantize):
    w, s = _make(64, 8, 192, 128, "packed")
    with pytest.raises(ValueError, match="scale has 115 rows"):
        _shard_fp8_qkv_proj(w, s[:-1], 64, 8, 192, 128, tp_rank=0, tp_size=4)


def test_shard_fp8_qkv_proj_rejects_indivisible_tp(stubbed_quantize):
    w, s = _make(64, 8, 192, 128, "packed")
    with pytest.raises(ValueError, match="evenly divide"):
        _shard_fp8_qkv_proj(w, s, 64, 8, 192, 128, tp_rank=0, tp_size=3)
