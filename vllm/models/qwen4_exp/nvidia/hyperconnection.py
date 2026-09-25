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

from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import maybe_prefix

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
    ``input_mix_weight_up`` independently — separately-quantized tensors
    cannot be stacked into one packed merged weight on load.
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

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = grouped_gemma_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        injection = self.block_inject_weight(xn) if self.use_combine else None
        lora = self.input_mix_weight_down(xn)

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

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

        injection = self.block_inject_weight(xn) if self.use_combine else None
        lora = self.input_mix_weight_down(xn)

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor | None,
    ) -> torch.Tensor:
        return hc_combine(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]
