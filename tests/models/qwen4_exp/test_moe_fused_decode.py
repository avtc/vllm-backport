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
    """Padded int6 packing matching this fork's checkpoint format: five
    values per int32 word at bits [6j, 6j+6), value + 32 (the inverse of
    vLLM's unpack_quantized_values_into_int32). Cross-checked against the
    installed compressed-tensors packer when importable."""
    N, K = vals.shape
    words = (K + 4) // 5
    padded = torch.zeros((N, words * 5), dtype=torch.int32, device=vals.device)
    padded[:, :K] = vals.to(torch.int32) + 32
    packed = torch.zeros((N, words), dtype=torch.int32, device=vals.device)
    for j in range(5):
        packed |= torch.bitwise_left_shift(padded[:, j::5], 6 * j)
    try:
        from compressed_tensors.compressors.pack_quantized.helpers import (
            pack_to_int32,
        )

        real = pack_to_int32(vals.to(torch.int8), num_bits=6, packed_dim=1)
        assert torch.equal(real, packed), (
            "compressed-tensors int6 packing changed; vLLM's unpacker and "
            "_int6_to_planes must be re-verified"
        )
    except ImportError:
        pass
    return packed


def _decode_planes(lo: torch.Tensor, hi: torch.Tensor, K: int) -> torch.Tensor:
    """CPU equivalent of the kernels' _i6_tile decode."""
    N = lo.shape[0]
    vals = torch.empty(N, K, dtype=torch.int32)
    for k in range(K):
        lw = lo[:, k // 8]
        hw = hi[:, k // 16]
        vals[:, k] = ((lw >> (4 * (k % 8))) & 15) | (((hw >> (2 * (k % 16))) & 3) << 4)
    return vals - 32


def _pack_int6_dense(vals: torch.Tensor) -> torch.Tensor:
    """Dense int6 packing (the AutoRound export format): value i at bit 6i of
    the row's little-endian bit stream, split across int32 boundaries."""
    N, K = vals.shape
    assert K % 32 == 0
    device = vals.device
    bits = vals.reshape(N, K // 32, 32).to(torch.int64) + 32
    stream = torch.zeros(N, K // 32, 192, dtype=torch.int64, device=device)
    for i in range(32):
        for b in range(6):
            stream[:, :, 6 * i + b] = (bits[:, :, i] >> b) & 1
    words = torch.zeros(N, K // 32, 6, dtype=torch.int32, device=device)
    for w in range(6):
        for b in range(32):
            words[:, :, w] |= stream[:, :, 32 * w + b].to(torch.int32) << b
    return words.reshape(N, 6 * K // 32)


def test_int6_to_planes_round_trip():
    """Both layouts round-trip: dense (the export this fork runs - the pp4
    boot crashed on its 480-word rows at K=2560) and padded (the 0.17.0
    packer, cross-checked when importable)."""
    from vllm.models.qwen4_exp.nvidia.model import _int6_to_planes

    torch.manual_seed(0)
    for n, k in ((5, 2560), (300, 640), (2, 128)):  # includes multi-chunk
        vals = torch.randint(-32, 32, (n, k), dtype=torch.int32)
        for packed in (_pack_int6_dense(vals), _pack_int6(vals)):
            lo, hi = _int6_to_planes(packed, k)
            assert lo.shape == (n, k // 8) and hi.shape == (n, k // 16)
            assert torch.equal(_decode_planes(lo, hi, k), vals)


def test_int6_to_planes_rejects_unknown_width():
    from vllm.models.qwen4_exp.nvidia.model import _int6_to_planes

    with pytest.raises(ValueError, match="neither the dense"):
        _int6_to_planes(torch.zeros(2, 100, dtype=torch.int32), 2560)


def test_int6_planes_both_or_neither():
    """A projection whose partner does not qualify must not be swapped."""
    from vllm.models.qwen4_exp.nvidia.model import (
        _int6_planes_group_size,
        _Int6PlanesScheme,
        _use_int6_planes,
    )

    # The eligibility check matches the scheme by class NAME.
    def _WNA16(num_bits=6, symmetric=True, group_size=64):
        return type(
            "CompressedTensorsWNA16",
            (),
            {"num_bits": num_bits, "symmetric": symmetric, "group_size": group_size},
        )()

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
    # unsupported group size
    assert _int6_planes_group_size(_Lin(2560, 1280, _WNA16(group_size=48))) is None
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
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
    )
    routed.moe_config.activation_situ_beta = None
    routed.moe_config.activation_situ_linear_beta = None
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
        weight=torch.zeros(512, 512, dtype=torch.bfloat16),
    )
    block._has_lora = False
    block._shared_int6 = False
    block._decode_state = None
    for k, v in over.items():
        setattr(block, k, v)
    return block


def test_setup_fused_decode_accepts_single_gpu_layout():
    block = _fake_block()
    block._setup_fused_decode()
    assert block._decode_state is not None
    assert block._decode_state["num_experts"] == 512
    assert block._decode_state["top_k"] == 8


def test_setup_fused_decode_rejects_ep():
    routed = _fake_block().experts.routed_experts
    routed.local_num_experts = 64
    block = _fake_block()
    block.experts = type(block.experts)(routed_experts=routed)
    block._setup_fused_decode()
    assert block._decode_state is None

    # EP flag even with full local experts
    routed_ep = _fake_block().experts.routed_experts
    routed_ep.moe_config.moe_parallel_config.use_ep = True
    block_ep = _fake_block()
    block_ep.experts = type(block_ep.experts)(routed_experts=routed_ep)
    block_ep._setup_fused_decode()
    assert block_ep._decode_state is None


def test_setup_fused_decode_accepts_moe_tp_without_ep():
    """MoE-TP without EP is supported: router/top-k/alignment run
    identically on every rank and _forward_decode adds the output
    all-reduce after the combine."""
    routed = _fake_block().experts.routed_experts
    routed.moe_config.moe_parallel_config.tp_size = 4
    block = _fake_block()
    block.experts = type(block.experts)(routed_experts=routed)
    block._setup_fused_decode()
    assert block._decode_state is not None
    assert block._decode_state["moe_tp"] == 4


def test_setup_fused_decode_rejects_replicated_shared_at_moe_tp():
    """A replicated shared expert's full-sum output would be counted
    tp_size times by the pre-reduce combine."""
    routed = _fake_block().experts.routed_experts
    routed.moe_config.moe_parallel_config.tp_size = 4
    block = _fake_block()
    block.experts = type(block.experts)(routed_experts=routed)
    block.replicate_shared_expert = True
    block._setup_fused_decode()
    assert block._decode_state is None


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
        # The router GEMV accumulates in fp32 tree order, the reference in
        # cublas order: pre-bf16-rounding values differ slightly, so compare
        # the rounded logits loosely (1 ulp at magnitude <= 64 is 0.5).
        torch.testing.assert_close(logits, ref_logits, atol=0.6, rtol=1e-2)
        torch.testing.assert_close(sgate, ref_sg, atol=0.6, rtol=1e-2)
        # Selection and weights are checked against the KERNEL'S OWN stored
        # logits, so accumulation order cannot flip the comparison: the
        # kernel's top-k uses the order-preserving int form of the bf16
        # value with ties to the lower expert id, and the renormalized
        # weights are a softmax over exactly those selected logits.
        bits = logits.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        ordered = torch.where(bits < 0, bits ^ 0x7FFF, bits).to(torch.int64)
        ids = torch.arange(e, device=logits.device)
        key = ordered * 65536 + (65535 - ids)[None, :]
        sel = torch.argsort(key, dim=-1, descending=True)[:, :topk]
        assert topk_ids.tolist() == sel.tolist(), (m, e)
        sel_w = torch.softmax(torch.gather(logits, 1, sel), dim=-1)
        torch.testing.assert_close(topk_w, sel_w, atol=1e-5, rtol=1e-4)

        # alignment contract: every valid flat entry appears once under its
        # expert's blocks, padded entries hold >= numel sentinel-ish values
        assert ntpp.item() % bs == 0
        valid_sorted = sorted_ids[: ntpp.item()]
        for j in range(numel):
            assert (valid_sorted == j).sum().item() == 1, (m, e, j)
        assert int(valid_sorted.min()) >= 0
        assert ticket.item() == 0  # reset for the next launch


def _quantize_int6(w: torch.Tensor, gs: int = 64):
    """Symmetric group int6 in the pack format, plus scales and dequant."""
    n, k = w.shape
    wg = w.float().reshape(n, k // gs, gs)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 31.0
    q = torch.round(wg / scale).clamp(-32, 31)
    return (
        q.reshape(n, k).to(torch.int32),
        scale.reshape(n, k // gs),
        ((q * scale).reshape(n, k).to(w.dtype)),
    )


def test_w6a16_gemm_matches_dequantized_reference():
    """The prefill-path Triton W6A16 GEMM matches a reference GEMM on
    identically dequantized (BF16-rounded) weights."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to launch the kernel")
    from vllm.models.qwen4_exp.nvidia.model import (
        _int6_to_planes,
        _w6a16_gemm,
    )

    torch.manual_seed(0)
    n, k = 1280, 512
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    q, scale, dequant = _quantize_int6(w)
    packed = _pack_int6(q)
    lo, hi = _int6_to_planes(packed.cpu(), k)
    lo, hi, scale_d = lo.cuda(), hi.cuda(), scale.cuda()

    for m in (5, 17, 33):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        got = _w6a16_gemm(x, lo, hi, scale_d, 64)
        ref = x @ dequant.T
        # bf16-rounded outputs from differently ordered fp32 accumulations:
        # allow a couple of bf16 ulps at the output magnitudes.
        torch.testing.assert_close(got, ref, atol=5e-1, rtol=2e-2)


def test_se_gate_up_act_matches_reference():
    """_se_gate_up_act_kernel: INT6 GEMV pair with SiluAndMul fused, bf16
    rounding at the unfused path's points."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to launch the kernel")

    from vllm.models.qwen4_exp.nvidia.model import (
        _int6_to_planes,
        _se_gate_up_act_kernel,
    )

    torch.manual_seed(0)
    k, inter, gs = 512, 128, 64
    w = torch.randn(2 * inter, k, device="cuda", dtype=torch.bfloat16)
    q, scale, dequant = _quantize_int6(w)
    packed = _pack_int6(q)
    lo, hi = _int6_to_planes(packed.cpu(), k)
    lo, hi, scale_d = lo.cuda(), hi.cuda(), scale.float().cuda()

    for m in (1, 2, 4):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        act = torch.empty(m, inter, device="cuda", dtype=torch.bfloat16)
        _se_gate_up_act_kernel[(inter // 4,)](
            x,
            lo,
            hi,
            scale_d,
            act,
            x.stride(0),
            act.stride(0),
            M=m,
            K=k,
            INTER=inter,
            GS=gs,
            BLOCK_I=4,
            BLOCK_K=512,
            STAGES=3,
            num_warps=2,
        )
        gu = (x.float() @ dequant.float().T).to(torch.bfloat16).float()
        g, u = gu[:, :inter], gu[:, inter:]
        ref = (g * torch.sigmoid(g) * u).to(torch.bfloat16)
        # silu(g) * u reaches magnitude ~100 where one bf16 ulp is 0.5-1.0.
        torch.testing.assert_close(act, ref, atol=2.0, rtol=5e-2)


def test_se_down_combine_matches_reference():
    """_se_down_combine_kernel: down GEMV fused with the routed sum +
    sigmoid-gated shared combine, bf16 rounding like the unfused path."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to launch the kernel")

    from vllm.models.qwen4_exp.nvidia.model import (
        _int6_to_planes,
        _se_down_combine_kernel,
    )

    torch.manual_seed(0)
    inter, n, gs, topk = 128, 512, 64, 8
    w = torch.randn(n, inter, device="cuda", dtype=torch.bfloat16)
    q, scale, dequant = _quantize_int6(w)
    packed = _pack_int6(q)
    lo, hi = _int6_to_planes(packed.cpu(), inter)
    lo, hi, scale_d = lo.cuda(), hi.cuda(), scale.float().cuda()

    for m in (1, 3, 4):
        act = torch.randn(m, inter, device="cuda", dtype=torch.bfloat16)
        routed = torch.randn(m, topk, n, device="cuda", dtype=torch.bfloat16)
        sgate = torch.randn(m, device="cuda", dtype=torch.float32)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        _se_down_combine_kernel[(n // 8,)](
            act,
            lo,
            hi,
            scale_d,
            routed,
            sgate,
            out,
            act.stride(0),
            out.stride(0),
            M=m,
            K=inter,
            N=n,
            GS=gs,
            TOPK=topk,
            BLOCK_N=8,
            BLOCK_K=128,
            STAGES=3,
            num_warps=1,
        )
        shared = (act.float() @ dequant.float().T).to(torch.bfloat16).float()
        g = torch.sigmoid(sgate).to(torch.bfloat16).float()
        shg = (shared * g[:, None]).to(torch.bfloat16).float()
        rsum = routed.float().sum(dim=1).to(torch.bfloat16).float()
        ref = (rsum + shg).to(torch.bfloat16)
        torch.testing.assert_close(out, ref, atol=5e-1, rtol=3e-2)


def test_moe_combine_kernel_matches_reference():
    """_moe_combine_kernel: moe_sum (bf16) + sigmoid-gated shared (bf16) +
    add, matching the unfused combine exactly."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to launch the kernel")
    import triton

    from vllm.models.qwen4_exp.nvidia.model import _moe_combine_kernel

    torch.manual_seed(0)
    m, k, topk = 4, 512, 8
    routed = torch.randn(m, topk, k, device="cuda", dtype=torch.bfloat16)
    shared = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    sgate = torch.randn(m, device="cuda", dtype=torch.float32)
    out = torch.empty(m, k, device="cuda", dtype=torch.bfloat16)
    _moe_combine_kernel[(m, triton.cdiv(k, 512))](
        routed, shared, sgate, out, K=k, TOPK=topk, BLOCK=512, num_warps=4
    )
    rsum = routed.float().sum(dim=1).to(torch.bfloat16).float()
    g = torch.sigmoid(sgate).to(torch.bfloat16).float()
    shg = (shared.float() * g[:, None]).to(torch.bfloat16).float()
    ref = (rsum + shg).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_setup_fused_decode_rejects_large_topk():
    """The alignment loop covers at most four expert_ids blocks per expert:
    top_k * _MOE_DECODE_MAX_TOKENS must stay <= 32 flat entries."""
    block = _fake_block()
    routed = _fake_block().experts.routed_experts
    routed.top_k = 16
    block.experts = type(block.experts)(routed_experts=routed)
    block._setup_fused_decode()
    assert block._decode_state is None


def test_setup_fused_decode_accepts_default_swiglu_and_rejects_nondefault():
    """RoutedExperts leaves swiglu_alpha/beta at None; the gate must accept
    None (and 1.0/0.0) and only reject actually-set knobs."""
    routed = _fake_block().experts.routed_experts
    routed.swiglu_limit = None
    routed.swiglu_alpha = None
    routed.swiglu_beta = None
    routed.moe_config.activation_situ_beta = None
    routed.moe_config.activation_situ_linear_beta = None
    block = _fake_block()
    block.experts = type(block.experts)(routed_experts=routed)
    block._setup_fused_decode()
    assert block._decode_state is not None

    routed2 = _fake_block().experts.routed_experts
    routed2.swiglu_alpha = 1.5
    block2 = _fake_block()
    block2.experts = type(block2.experts)(routed_experts=routed2)
    block2._setup_fused_decode()
    assert block2._decode_state is None

    routed3 = _fake_block().experts.routed_experts
    routed3.moe_config.activation_situ_linear_beta = 0.25
    block3 = _fake_block()
    block3.experts = type(block3.experts)(routed_experts=routed3)
    block3._setup_fused_decode()
    assert block3._decode_state is None


def test_setup_fused_decode_rejects_non_bf16_params():
    from types import SimpleNamespace

    routed = _fake_block().experts.routed_experts
    block = _fake_block()
    block.experts = type(block.experts)(routed_experts=routed)
    block.gate = SimpleNamespace(
        quant_method=type("UnquantizedLinearMethod", (), {})(),
        weight=torch.zeros(512, 512, dtype=torch.float16),
    )
    block._setup_fused_decode()
    assert block._decode_state is None


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8, 16, 64])
@pytest.mark.parametrize("n,k,gs", [(1280, 2560, 64), (2560, 640, 64)])
def test_w6a16_gemm_real_shapes(m, n, k, gs):
    """The standalone int6-planes GEMM at the checkpoint's real shapes.

    PP tp1 boots the shared expert through _Int6PlanesScheme on the real
    weights for the first time (tp8+EP replicates it, so the path never ran
    before). Exercises every M-driven block config including the M=1 decode
    shape that JIT-compiles separately, against a dequant-then-matmul
    reference."""
    cuda = pytest.importorskip("torch").cuda
    if not cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.models.qwen4_exp.nvidia.model import (
        _int6_to_planes,
        _w6a16_gemm,
    )

    torch.manual_seed(0)
    vals = torch.randint(-32, 32, (n, k), dtype=torch.int32, device="cuda")
    lo, hi = _int6_to_planes(_pack_int6_dense(vals), k)
    scale = torch.rand(n, k // gs, dtype=torch.bfloat16, device="cuda") * 0.01
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

    y = _w6a16_gemm(x, lo, hi, scale, gs)

    w = vals.to(torch.float32) * scale.repeat_interleave(gs, dim=1).to(torch.float32)
    ref = (x.to(torch.float32) @ w.T).to(torch.bfloat16)
    torch.testing.assert_close(y, ref, rtol=2e-2, atol=2e-1)
