# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities — NVIDIA model variant.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606). This NVIDIA variant
delays each HC combine to the following HC mix boundary. HC glue kernels,
including fused combine+RMSNorm, live in ``ops/hc.py``; projections remain
standard vLLM Linear modules.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout).

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidual(hc_config)

    hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
    attention_output = attention(block_input)
    hidden_states, block_input, injection = self.mlp_hc.combine_and_mix(
        hidden_states, attention_output, injection
    )
"""

import torch
from torch import nn

from vllm import envs
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import maybe_prefix
from vllm.triton_utils import tl, triton

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from .ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)

# ---------------------------------------------------------------------------
# Fused INT8 low-rank projections without a repack
# ---------------------------------------------------------------------------
#
# Ported from Minachist's Apache-2.0 decode-04 patch (credited). At decode
# batch sizes (up to 4 tokens) each GatedResidual runs two Triton kernels on
# the INT8 pack-quantized weights as loaded (no Marlin repack, no extra
# memory): down projection + BF16 injection + silu, and up projection + gate
# mix over the HC streams. That replaces six kernels, ~32 us -> ~13 us per
# module at one token, 96 modules per token. Larger batches use a Triton
# W8A16 GEMM on the same weights (faster than Marlin on these shapes at 512
# tokens). Outputs round to BF16 at the same points as the unfused path;
# the in-register weight dequant stays fp32 (slightly more precise than
# the prefill GEMM, which rounds dequantized tiles to BF16 before the MMA).
#
# The pack-quantized INT8 layout is plain bytes: the int32 words hold four
# values, least significant byte first, so weight_packed viewed as uint8 is
# q[N, K] with q = value + 128. VLLM_HC_FUSED_INT8=0 keeps the Marlin path.
_HC_FUSED_MAX_TOKENS = 4


def _hc_fused_int8_enabled() -> bool:
    return bool(envs.VLLM_HC_FUSED_INT8)


@triton.jit
def _hc_down_inject_silu_kernel(
    x_ptr,
    wq_ptr,
    ws_ptr,
    winj_ptr,
    lora_ptr,
    inj_ptr,
    stride_x,
    stride_lora,
    stride_inj,
    M: tl.constexpr,
    K: tl.constexpr,
    N_DOWN: tl.constexpr,
    GS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MPAD: tl.constexpr,
):
    # One output row per program: programs [0, N_DOWN) take the INT8 down
    # rows, the remaining ones the BF16 injection rows.
    pid = tl.program_id(0)
    ms = tl.arange(0, MPAD)
    mmask = ms < M
    ks = tl.arange(0, BLOCK_K)
    NG: tl.constexpr = BLOCK_K // GS
    acc = tl.zeros((MPAD,), tl.float32)
    if pid < N_DOWN:
        for k0 in tl.range(0, K, BLOCK_K):
            w = tl.load(wq_ptr + pid * K + k0 + ks).to(tl.int32) - 128
            sc = tl.load(ws_ptr + pid * (K // GS) + k0 // GS + tl.arange(0, NG)).to(
                tl.float32
            )
            x = tl.load(
                x_ptr + ms[:, None] * stride_x + k0 + ks[None, :],
                mask=mmask[:, None],
                other=0.0,
            ).to(tl.float32)
            p = tl.reshape(x * w.to(tl.float32)[None, :], (MPAD, NG, GS))
            acc += tl.sum(tl.sum(p, axis=2) * sc[None, :], axis=1)
        # hc_silu, on the projection rounded to BF16 like the unfused path.
        v = acc.to(tl.bfloat16).to(tl.float32) / HC
        tl.store(
            lora_ptr + ms * stride_lora + pid,
            (v * tl.sigmoid(v)).to(tl.bfloat16),
            mask=mmask,
        )
    else:
        row = pid - N_DOWN
        for k0 in tl.range(0, K, BLOCK_K):
            w = tl.load(winj_ptr + row * K + k0 + ks).to(tl.float32)
            x = tl.load(
                x_ptr + ms[:, None] * stride_x + k0 + ks[None, :],
                mask=mmask[:, None],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(x * w[None, :], axis=1)
        tl.store(inj_ptr + ms * stride_inj + row, acc.to(tl.bfloat16), mask=mmask)


@triton.jit
def _hc_row_dot(w3, sc, x_ptr, k0, ks, kmask, NG: tl.constexpr, GS: tl.constexpr):
    """sum_k w[r, k] * x[k] * sc[r, k // GS] for one token. w3 is [R, NG, GS]."""
    x = tl.load(x_ptr + k0 + ks, mask=kmask, other=0.0).to(tl.float32)
    return tl.sum(tl.sum(w3 * tl.reshape(x, (1, NG, GS)), axis=2) * sc, axis=1)


@triton.jit
def _hc_gate_mix_store(
    acc, rows, xn_ptr, out_ptr, pid, HC: tl.constexpr, BLOCK_H: tl.constexpr
):
    gate = acc.to(tl.bfloat16).to(tl.float32)
    xv = tl.load(xn_ptr + rows).to(tl.float32)
    mixed = tl.reshape(tl.sigmoid(gate) * xv, (HC, BLOCK_H))
    out = tl.sum(mixed, axis=0) / HC
    tl.store(out_ptr + pid * BLOCK_H + tl.arange(0, BLOCK_H), out.to(tl.bfloat16))


@triton.jit
def _hc_up_gate_mix_kernel(
    lora_ptr,
    wq_ptr,
    ws_ptr,
    xn_ptr,
    out_ptr,
    stride_lora,
    stride_xn,
    stride_out,
    M: tl.constexpr,
    K: tl.constexpr,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    GS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # The program's rows are stream-major [HC, BLOCK_H] of the up projection,
    # so the gate mix over the HC streams happens in registers. Tokens are
    # unrolled into separate accumulators (M <= 4).
    pid = tl.program_id(0)
    ROWS: tl.constexpr = HC * BLOCK_H
    NG: tl.constexpr = BLOCK_K // GS
    r = tl.arange(0, ROWS)
    rows = (r // BLOCK_H) * HC_DIM + pid * BLOCK_H + (r % BLOCK_H)
    ks = tl.arange(0, BLOCK_K)
    a0 = tl.zeros((ROWS,), tl.float32)
    a1 = tl.zeros((ROWS,), tl.float32)
    a2 = tl.zeros((ROWS,), tl.float32)
    a3 = tl.zeros((ROWS,), tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        kmask = (k0 + ks) < K
        w = tl.load(
            wq_ptr + rows[:, None] * K + k0 + ks[None, :],
            mask=kmask[None, :],
            other=128,
        )
        w3 = tl.reshape((w.to(tl.int32) - 128).to(tl.float32), (ROWS, NG, GS))
        g = k0 // GS + tl.arange(0, NG)
        sc = tl.load(
            ws_ptr + rows[:, None] * (K // GS) + g[None, :],
            mask=(g < K // GS)[None, :],
            other=0.0,
        ).to(tl.float32)
        a0 += _hc_row_dot(w3, sc, lora_ptr, k0, ks, kmask, NG, GS)
        if M > 1:
            a1 += _hc_row_dot(w3, sc, lora_ptr + stride_lora, k0, ks, kmask, NG, GS)
        if M > 2:
            a2 += _hc_row_dot(w3, sc, lora_ptr + 2 * stride_lora, k0, ks, kmask, NG, GS)
        if M > 3:
            a3 += _hc_row_dot(w3, sc, lora_ptr + 3 * stride_lora, k0, ks, kmask, NG, GS)
    _hc_gate_mix_store(a0, rows, xn_ptr, out_ptr, pid, HC, BLOCK_H)
    if M > 1:
        _hc_gate_mix_store(
            a1, rows, xn_ptr + stride_xn, out_ptr + stride_out, pid, HC, BLOCK_H
        )
    if M > 2:
        _hc_gate_mix_store(
            a2,
            rows,
            xn_ptr + 2 * stride_xn,
            out_ptr + 2 * stride_out,
            pid,
            HC,
            BLOCK_H,
        )
    if M > 3:
        _hc_gate_mix_store(
            a3,
            rows,
            xn_ptr + 3 * stride_xn,
            out_ptr + 3 * stride_out,
            pid,
            HC,
            BLOCK_H,
        )


@triton.jit
def _w8a16_gemm_kernel(
    x_ptr,
    wq_ptr,
    ws_ptr,
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
    # y = x @ dequant(q)^T for larger batches (prefill). One quantization group
    # per K step; the dequantized tile is rounded to BF16 before the MMA.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, GS)
    mmask = offs_m < M
    nmask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for g in tl.range(0, K // GS):
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x + g * GS + ks[None, :],
            mask=mmask[:, None],
            other=0.0,
        )
        w = tl.load(
            wq_ptr + offs_n[:, None] * K + g * GS + ks[None, :],
            mask=nmask[:, None],
            other=128,
        )
        sc = tl.load(ws_ptr + offs_n * (K // GS) + g, mask=nmask, other=0.0)
        wd = (w.to(tl.int32) - 128).to(tl.float32) * sc.to(tl.float32)[:, None]
        acc = tl.dot(x, tl.trans(wd.to(tl.bfloat16)), acc)
    tl.store(
        y_ptr + offs_m[:, None] * stride_y + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=mmask[:, None] & nmask[None, :],
    )


def _w8a16_gemm(x: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, gs: int):
    x = x.contiguous()
    M, K = x.shape
    N = wq.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    # Measured on RTX 3090 at M = 512 (a prefill chunk): 32 x 64 beats the
    # Marlin/Humming GEMMs on both HC shapes.
    BLOCK_M = 16 if M <= 16 else 32
    BLOCK_N = 64
    _w8a16_gemm_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
        x,
        wq,
        ws,
        y,
        M,
        N,
        x.stride(0),
        y.stride(0),
        K=K,
        GS=gs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return y


class _HCInt8Scheme:
    """Replaces the compressed-tensors WNA16 scheme of an INT8 HC projection.

    Keeps the loaded pack-quantized weight (no Marlin repack) and exposes it
    as q[N, K] uint8. ``apply_weights`` is the general path; GatedResidual
    calls the fused decode kernels directly.
    """

    def __init__(self, group_size: int) -> None:
        self.group_size = group_size

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # Copy weight and scales once, like the Marlin repack this replaces
        # (allocate new, free old): keeping the loaded tensors in place shifts
        # the free blocks the other layers' repacks leave behind, which
        # stranded ~160 MiB of allocator cache on one GPU of a 3-card setup.
        layer.weight_packed.data = layer.weight_packed.data.clone()
        layer.weight_scale.data = layer.weight_scale.data.clone()
        layer.hc_q = layer.weight_packed.data.view(torch.uint8)  # [N, K]
        layer.hc_scale = layer.weight_scale.data  # [N, K // GS]
        assert layer.hc_q.is_contiguous() and layer.hc_scale.is_contiguous()

    def apply_weights(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert bias is None
        shape = x.shape
        y = _w8a16_gemm(
            x.reshape(-1, shape[-1]), layer.hc_q, layer.hc_scale, self.group_size
        )
        return y.view(*shape[:-1], y.shape[-1])


def _hc_int8_group_size(linear: nn.Module) -> int | None:
    """Group size if the projection qualifies for the fused INT8 path."""
    scheme = getattr(linear, "scheme", None)
    if scheme is None or type(scheme).__name__ != "CompressedTensorsWNA16":
        return None
    group_size = getattr(scheme, "group_size", -1)
    if (
        scheme.num_bits != 8
        or not scheme.symmetric
        or group_size not in (32, 64, 128)
        or linear.input_size_per_partition % group_size
    ):
        return None
    return group_size


def _use_hc_int8_scheme(linear: nn.Module) -> bool:
    """Swap in _HCInt8Scheme when the projection is pack-quantized INT8.

    Check eligibility for every projection first (via _hc_int8_group_size)
    and only then call this: it mutates ``linear.scheme`` as a side effect,
    and a short-circuited ``and`` would leave the first projection swapped
    while the fused path stays disabled.
    """
    group_size = _hc_int8_group_size(linear)
    if group_size is None:
        return False
    linear.scheme = _HCInt8Scheme(group_size)
    return True


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> split
    low-rank down and inject GEMMs -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output, it fuses its
    residual combine with the RMSNorm. A missing injection selects unit-weight
    combine. Final mixers use ``use_combine=False`` and do not produce a new
    injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules, so GEMM dispatch (e.g. the low-latency skinny
    GEMM) applies through the standard quant_method mechanism. The down and
    inject projections stay split because quantized checkpoints (AutoRound /
    INC int8 group-64) quantize ``input_mix_weight_down`` and
    ``input_mix_weight_up`` independently -- separately-quantized tensors
    cannot be stacked into one packed merged weight on load. The merged
    variant's deliberate 16-row CuBLAS alignment padding is dropped with it;
    these skinny GEMMs are not dispatch-critical.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # Split projections, checkpoint-native. All are replicated: the skinny
        # shapes do not benefit from TP sharding and the checkpoint quantizes
        # down/up (but not block_inject) with group-64 scales.
        self.input_mix_weight_down = ReplicatedLinear(
            self.hyper_hidden_size,
            self.lora_rank,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "input_mix_weight_down"),
            return_bias=False,
            disable_tp=True,
        )
        if use_combine:
            self.block_inject_weight = ReplicatedLinear(
                self.hyper_hidden_size,
                self.hc_count,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "block_inject_weight"),
                return_bias=False,
                disable_tp=True,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
            disable_tp=True,
        )
        # Fused INT8 decode path (see _HC_FUSED_MAX_TOKENS). Both projections
        # must qualify, or neither is touched. The fused kernels run
        # unmasked tiles: 2048 columns of the HC state, 8 hidden positions
        # per program.
        self._hc_fused = False
        if (
            _hc_fused_int8_enabled()
            and self.hyper_hidden_size % 2048 == 0
            and self.hidden_size % 8 == 0
        ):
            down, up = self.input_mix_weight_down, self.input_mix_weight_up
            # Both projections must qualify, or neither is touched.
            if all(
                type(getattr(m, "scheme", None)).__name__ == "CompressedTensorsWNA16"
                and m.scheme.num_bits == 8
                and _hc_int8_group_size(m) is not None
                for m in (down, up)
            ):
                self._hc_fused = _use_hc_int8_scheme(down) and _use_hc_int8_scheme(up)

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = grouped_gemma_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        block_input, injection = self._project(xn)
        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is fused with this
        module's input RMSNorm. A missing injection applies the block output
        to every stream with unit weight.
        """
        hidden_states, xn = hc_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        block_input, injection = self._project(xn)
        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor | None,
    ) -> torch.Tensor:
        return hc_combine(hidden_states, block_output, injection, self.hc_count)

    def _project(self, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """injection, down -> silu -> up, and the gate mix over the streams."""
        num_tokens = xn.shape[0]
        if self._hc_fused and 0 < num_tokens <= _HC_FUSED_MAX_TOKENS:
            return self._project_fused(xn)
        injection = self.block_inject_weight(xn) if self.use_combine else None
        lora = self.input_mix_weight_down(xn)

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)
        return block_input, injection

    def _project_fused(
        self, xn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        down, up = self.input_mix_weight_down, self.input_mix_weight_up
        num_tokens, hyper = xn.shape
        assert xn.stride(1) == 1
        rank = down.hc_q.shape[0]
        lora = torch.empty((num_tokens, rank), device=xn.device, dtype=torch.bfloat16)
        if self.use_combine:
            winj = self.block_inject_weight.weight
            injection = torch.empty(
                (num_tokens, winj.shape[0]), device=xn.device, dtype=torch.bfloat16
            )
            n_inj = winj.shape[0]
        else:
            winj, injection, n_inj = down.hc_q, lora, 0
        _hc_down_inject_silu_kernel[(rank + n_inj,)](
            xn,
            down.hc_q,
            down.hc_scale,
            winj,
            lora,
            injection,
            xn.stride(0),
            lora.stride(0),
            injection.stride(0),
            M=num_tokens,
            K=hyper,
            N_DOWN=rank,
            GS=down.scheme.group_size,
            HC=self.hc_count,
            BLOCK_K=2048,
            MPAD=triton.next_power_of_2(num_tokens),
            num_warps=8,
        )
        block_h = 8
        block_input = torch.empty(
            (num_tokens, self.hidden_size), device=xn.device, dtype=torch.bfloat16
        )
        _hc_up_gate_mix_kernel[(self.hidden_size // block_h,)](
            lora,
            up.hc_q,
            up.hc_scale,
            xn,
            block_input,
            lora.stride(0),
            xn.stride(0),
            block_input.stride(0),
            M=num_tokens,
            K=rank,
            HC_DIM=self.hidden_size,
            HC=self.hc_count,
            GS=up.scheme.group_size,
            BLOCK_H=block_h,
            # The up kernel reshapes BLOCK_K into (NG, GS) groups; BLOCK_K
            # must cover at least one full group (GS can be 128).
            BLOCK_K=max(64, up.scheme.group_size),
            num_warps=4,
        )
        return block_input, (injection if self.use_combine else None)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]
