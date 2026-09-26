# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused decode-time MoE block (VLLM_MOE_FUSED_DECODE, decode-05).

Covers the INT6 plane relayout (bit-exact round-trip through the
compressed-tensors tight pack format), the both-or-neither scheme swap, the
fused-path gating reasons, and - on CUDA - the router kernel's top-k /
renormalization / block alignment against reference torch computations.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn


def _pack_int6(vals: torch.Tensor) -> torch.Tensor:
    """Inverse of the compressed-tensors int6 unpack: biased 6-bit values in
    a 192-bit LSB-first stream per 32 values, six int32 words."""
    N, K = vals.shape
    bits = vals.reshape(N, K // 32, 32).to(torch.int64) + 32
    stream = torch.zeros(N, K // 32, 192, dtype=torch.int64)
    for i in range(32):
        for b in range(6):
            stream[:, :, 6 * i + b] = (bits[:, :, i] >> b) & 1
    words = torch.zeros(N, K // 32, 6, dtype=torch.int32)
    for w in range(6):
        for b in range(32):
            words[:, :, w] |= stream[:, :, 32 * w + b].to(torch.int32) << b
    return words.reshape(N, K * 6 // 32)


def _decode_planes(lo: torch.Tensor, hi: torch.Tensor, K: int) -> torch.Tensor:
    """CPU equivalent of the kernels' _i6_tile decode."""
    N = lo.shape[0]
    vals = torch.empty(N, K, dtype=torch.int32)
    for k in range(K):
        lw = lo[:, k // 8]
        hw = hi[:, k // 16]
        vals[:, k] = ((lw >> (4 * (k % 8))) & 15) | (((hw >> (2 * (k % 16))) & 3) << 4)
    return vals - 32


def test_int6_to_planes_round_trip():
    from vllm.models.qwen4_exp.nvidia.model import _int6_to_planes

    torch.manual_seed(0)
    for n, k in ((5, 256), (300, 128)):  # includes the multi-chunk path
        vals = torch.randint(-32, 32, (n, k), dtype=torch.int32)
        packed = _pack_int6(vals)
        lo, hi = _int6_to_planes(packed, k)
        assert lo.shape == (n, k // 8) and hi.shape == (n, k // 16)
        assert torch.equal(_decode_planes(lo, hi, k), vals)


def test_int6_planes_both_or_neither():
    """A projection whose partner does not qualify must not be swapped."""
    from vllm.models.qwen4_exp.nvidia.model import (
        _int6_planes_group_size,
        _Int6PlanesScheme,
        _use_int6_planes,
    )

    class _WNA16:
        def __init__(self, num_bits=6, symmetric=True, group_size=64):
            self.num_bits = num_bits
            self.symmetric = symmetric
            self.group_size = group_size

    class _Lin(nn.Module):
        def __init__(self, k, n, scheme):
            super().__init__()
            self.scheme = scheme
            self.input_size_per_partition = k
            self.output_size_per_partition = n

    good = _Lin(2560, 1280, _WNA16())
    assert _int6_planes_group_size(good) == 64
    assert _use_int6_planes(good) is True
    assert isinstance(good.scheme, _Int6PlanesScheme)

    # int8 not int6
    assert _int6_planes_group_size(_Lin(2560, 1280, _WNA16(num_bits=8))) is None
    # group size != 64
    assert _int6_planes_group_size(_Lin(2560, 1280, _WNA16(group_size=128))) is None
    # K % 128, N % 16
    assert _int6_planes_group_size(_Lin(2000, 1280, _WNA16())) is None
    assert _int6_planes_group_size(_Lin(2560, 1284, _WNA16())) is None


def _fake_block(**over):
    """A Qwen4ExpSparseMoeBlock shell whose dependencies are fakes, driving
    _setup_fused_decode's reason list."""
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpSparseMoeBlock

    block = object.__new__(Qwen4ExpSparseMoeBlock)
    pc = SimpleNamespace(tp_size=1, dp_size=1, pcp_size=1, use_ep=False)
    _qm_cls = type(
        "CompressedTensorsWNA16MoEMethod",
        (),
        {"is_marlin": True, "num_bits": 4, "symmetric": True},
    )
    routed = SimpleNamespace(
        quant_method=_qm_cls(),
        expert_map=None,
        local_num_experts=512,
        global_num_experts=512,
        moe_config=SimpleNamespace(moe_parallel_config=pc),
        scoring_func="softmax",
        use_grouped_topk=False,
        custom_routing_function=None,
        e_score_correction_bias=None,
        routed_scaling_factor=1.0,
        apply_router_weight_on_input=False,
        activation=SimpleNamespace(name="SILU"),
        top_k=8,
        renormalize=True,
    )
    block.experts = SimpleNamespace(routed_experts=routed)
    block.shared_expert = SimpleNamespace(
        expert_gate=SimpleNamespace(weight=torch.zeros(1, 512)),
        gate_up_proj=SimpleNamespace(input_size_per_partition=512),
        down_proj=SimpleNamespace(input_size_per_partition=640),
    )
    block.replicate_shared_expert = False
    block.enable_eplb = False
    block.gate = SimpleNamespace(
        quant_method=type("UnquantizedLinearMethod", (), {})(),
        weight=torch.zeros(512, 512),
    )
    block._has_lora = False
    block._shared_int6 = False
    for k, v in over.items():
        setattr(block, k, v)
    return block


def test_setup_fused_decode_accepts_single_gpu_layout():
    block = _fake_block()
    block._setup_fused_decode()
    assert block._decode_state is not None
    assert block._decode_state["num_experts"] == 512
    assert block._decode_state["top_k"] == 8


def test_setup_fused_decode_rejects_ep_and_moe_tp():
    # EP: local experts != global
    routed = _fake_block().experts.routed_experts
    routed.local_num_experts = 64
    block = _fake_block()
    block.experts = type(block.experts)(routed_experts=routed)
    block._setup_fused_decode()
    assert block._decode_state is None

    # MoE TP: tp_size > 1
    routed2 = _fake_block().experts.routed_experts
    routed2.local_num_experts = routed2.global_num_experts
    routed2.moe_config.moe_parallel_config.tp_size = 8
    block2 = _fake_block()
    block2.experts = type(block2.experts)(routed_experts=routed2)
    block2._setup_fused_decode()
    assert block2._decode_state is None


def test_moe_router_topk_matches_reference():
    """The router kernel's winner-program math (order-preserving bitcast
    top-k + renormalized weights + block alignment) matches torch."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to launch the kernel")
    import triton

    from vllm.models.qwen4_exp.nvidia.model import _moe_router_kernel

    torch.manual_seed(0)
    dev = "cuda"
    shapes = ((1, 512, 8, 16), (4, 512, 8, 8), (3, 64, 8, 16), (2, 8, 4, 8))
    for m, e, topk, bs in shapes:
        k = 512
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        w = torch.randn(e, k, device=dev, dtype=torch.bfloat16)
        wsg = torch.randn(k, device=dev, dtype=torch.bfloat16)
        logits = torch.empty((m, e), device=dev, dtype=torch.float32)
        sgate = torch.empty((m,), device=dev, dtype=torch.float32)
        topk_w = torch.empty((m, topk), device=dev, dtype=torch.float32)
        topk_ids = torch.empty((m, topk), device=dev, dtype=torch.int32)
        numel = m * topk
        max_pad = min(numel * bs, numel + e * (bs - 1))
        sorted_ids = torch.empty((max_pad,), device=dev, dtype=torch.int32)
        expert_ids = torch.empty(
            (triton.cdiv(max_pad, bs),), device=dev, dtype=torch.int32
        )
        ntpp = torch.empty((1,), device=dev, dtype=torch.int32)
        ticket = torch.zeros(1, dtype=torch.int32, device=dev)
        _moe_router_kernel[(e // 2 + 1,)](
            x,
            w,
            wsg,
            logits,
            sgate,
            ticket,
            topk_w,
            topk_ids,
            sorted_ids,
            expert_ids,
            ntpp,
            x.stride(0),
            m,
            K=k,
            E=e,
            TOPK=topk,
            BS=bs,
            ROWS=2,
            BLOCK_K=1024,
            MPAD=triton.next_power_of_2(m),
            P=triton.next_power_of_2(topk),
            RENORM=True,
            num_warps=4,
        )

        # reference: bf16 logits, top-k, softmax over the selection
        ref_logits = (x.float() @ w.float().T).to(torch.bfloat16).float()
        ref_sg = (x.float() @ wsg.float()).to(torch.bfloat16).float()
        sel_w, sel_i = torch.topk(ref_logits, topk, dim=-1)
        ref_w = torch.softmax(sel_w, dim=-1)
        torch.testing.assert_close(logits, ref_logits, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(sgate, ref_sg, atol=1e-6, rtol=1e-5)
        # Exact ties (possible in bf16) may order differently; compare the
        # selections as id-sorted (weights, ids) pairs.
        got = sorted(zip(topk_ids.tolist(), topk_w.tolist()))
        ref = sorted(zip(sel_i.tolist(), ref_w.tolist()))
        assert [i for i, _ in got] == [i for i, _ in ref], (m, e, got, ref)
        torch.testing.assert_close(
            torch.tensor([w for _, w in got]),
            torch.tensor([w for _, w in ref]),
            atol=1e-5,
            rtol=1e-4,
        )

        # alignment contract: every valid flat entry appears once under its
        # expert's blocks, padded entries hold >= numel sentinel-ish values
        assert ntpp.item() % bs == 0
        valid_sorted = sorted_ids[: ntpp.item()]
        for j in range(numel):
            assert (valid_sorted == j).sum().item() == 1, (m, e, j)
        assert int(valid_sorted.min()) >= 0
        assert ticket.item() == 0  # reset for the next launch
