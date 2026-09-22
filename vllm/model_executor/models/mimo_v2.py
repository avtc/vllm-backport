# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
    str_dtype_to_torch_dtype,
)
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_quantize,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

from .interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class MiMoV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiMoV2MoE(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.n_routed_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        dtype = getattr(config, "moe_router_dtype", "float32")
        self.gate_dtype = str_dtype_to_torch_dtype(dtype)
        self.gate = nn.Linear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            dtype=self.gate_dtype,
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=self.gate_dtype)
        )

        self.experts = FusedMoEFactory(
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func="sigmoid",
            router_logits_dtype=self.gate_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, "MiMoV2MoE only supports 1D or 2D inputs"
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.gate_dtype is not None:
            gate_input = hidden_states.to(self.gate_dtype)
        else:
            gate_input = hidden_states
        router_logits = self.gate(gate_input)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class MiMoV2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int | None = None,
        v_scale: float | None = None,
        sliding_window_size: int = -1,
        attention_bias: bool = False,
        add_swa_attention_sink_bias: bool = False,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        max_position_embeddings: int = 32768,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        partial_rotary_factor: float = 1.0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = head_dim

        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim

        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_head_dim

        self.v_scale = v_scale
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            v_head_size=self.v_head_dim,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config if "mtp.layers" not in prefix else None,
            reduce_results=True,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": rope_theta,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )

        self.attention_sink_bias = (
            torch.nn.Parameter(torch.empty(self.num_heads), requires_grad=False)
            if add_swa_attention_sink_bias
            else None
        )

        sliding_window = sliding_window_size if sliding_window_size > -1 else None

        # Use DiffKV backend when V has a different head dim than K.
        # sm<90 patch (validated on Ampere with MiMo-V2.5): FA3/4 DiffKV is
        # Hopper-only, and the TRITON_ATTN_DIFFKV backend computes the
        # asymmetric-V + SWA-sinks path incorrectly on Ampere, garbling the
        # output. Route sm<90 through the standard TritonAttentionBackend with
        # V zero-padded v_head_dim->head_dim before attention and the padded
        # tail sliced off the output (exact: discarded dims are provably
        # output-independent). Hopper+ keeps the native FA-DiffKV path; an
        # explicit `--attention-backend *_DIFFKV` is still honored.
        self._pad_v = False
        _cap = current_platform.get_device_capability()
        _is_hopper_plus = _cap is not None and _cap.major >= 9
        if self.v_head_dim != self.head_dim:
            requested = get_current_vllm_config().attention_config.backend
            if requested is not None and requested.name.endswith("_DIFFKV"):
                attn_backend = requested.get_class()
                attn_backend.set_head_size_v(self.v_head_dim)
                attn_head_size_v = self.v_head_dim
                logger.info_once("Using %s for attention.", attn_backend.get_name())
            else:
                fa_backend = AttentionBackendEnum.FLASH_ATTN_DIFFKV.get_class()
                assert hasattr(fa_backend, "is_supported_on_current_device")
                if _is_hopper_plus and fa_backend.is_supported_on_current_device(
                    head_size=self.head_dim,
                    head_size_v=self.v_head_dim,
                    has_sinks=self.attention_sink_bias is not None,
                ):
                    attn_backend = fa_backend
                    attn_backend.set_head_size_v(self.v_head_dim)
                    attn_head_size_v = self.v_head_dim
                    logger.info_once(
                        "Using %s for attention.", attn_backend.get_name()
                    )
                elif _is_hopper_plus:
                    attn_backend = AttentionBackendEnum.TRITON_ATTN_DIFFKV.get_class()
                    attn_backend.set_head_size_v(self.v_head_dim)
                    attn_head_size_v = self.v_head_dim
                    logger.info_once(
                        "Using %s for attention.", attn_backend.get_name()
                    )
                else:
                    self._pad_v = True
                    attn_backend = TritonAttentionBackend
                    attn_head_size_v = self.head_dim
                    logger.info_once(
                        "sm<90: routing asymmetric-V attention through "
                        "TritonAttentionBackend with V padded %d->%d.",
                        self.v_head_dim,
                        self.head_dim,
                    )
        else:
            attn_backend = None
            attn_head_size_v = self.v_head_dim

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.attention_sink_bias,
            attn_backend=attn_backend,
            head_size_v=attn_head_size_v,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        # Apply v_scale before attention
        if self.v_scale is not None:
            v = v * self.v_scale

        if self._pad_v:
            # Pad V to head_dim so K/V share a head size for the standard
            # Triton backend, then slice the zero-padded tail off the output
            # (exact).
            v = v.view(-1, self.num_kv_heads, self.v_head_dim)
            v = nn.functional.pad(v, (0, self.head_dim - self.v_head_dim))
            v = v.reshape(-1, self.num_kv_heads * self.head_dim)

        attn_output = self.attn(q, k, v)

        if self._pad_v:
            attn_output = attn_output.view(-1, self.num_heads, self.head_dim)
            attn_output = attn_output[..., : self.v_head_dim].reshape(
                -1, self.num_heads * self.v_head_dim
            )

        output, _ = self.o_proj(attn_output)
        return output


class MiMoV2FlashDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        layer_id = extract_layer_index(prefix)

        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_id = layer_id

        rope_theta = getattr(config, "rope_theta", 1000000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)

        v_scale = getattr(config, "attention_value_scale", None)

        if self.is_compressed_softmax_layer():
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=config.swa_num_attention_heads,
                num_kv_heads=config.swa_num_key_value_heads,
                head_dim=config.swa_head_dim,
                v_head_dim=getattr(config, "swa_v_head_dim", None),
                v_scale=v_scale,
                sliding_window_size=config.sliding_window_size,
                attention_bias=config.attention_bias,
                add_swa_attention_sink_bias=getattr(
                    config, "add_swa_attention_sink_bias", False
                ),
                layer_id=layer_id,
                rope_theta=getattr(config, "swa_rope_theta", rope_theta),
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = MiMoV2Attention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                v_head_dim=getattr(config, "v_head_dim", None),
                v_scale=v_scale,
                sliding_window_size=-1,  # normal attention
                attention_bias=config.attention_bias,
                layer_id=layer_id,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
                prefix=f"{prefix}.self_attn",
            )

        self.is_layer_sparse = self.is_moe_layer(layer_id)
        if self.is_layer_sparse:
            self.mlp = MiMoV2MoE(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = MiMoV2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def is_moe_layer(self, layer_idx: int) -> bool:
        return (
            hasattr(self.config, "moe_layer_freq")
            and layer_idx >= 0
            and not isinstance(self.config.moe_layer_freq, int)
            and self.config.moe_layer_freq[layer_idx]
        )

    def is_compressed_softmax_layer(self) -> bool:
        return self.config.hybrid_layer_pattern[self.layer_id] == 1


def _shard_fp8_qkv_proj(
    w_full: torch.Tensor,
    s_full: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    tp_rank: int,
    tp_size: int,
    block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shard the fp8 qkv_proj weights for ``tp_rank``.

    The checkpoint stores the fused QKV as ``num_kv_heads`` contiguous groups
    (one per KV head; ``n`` below), each ordered ``[Q | K | V]``:

        [Q_1 | K_1 | V_1 | Q_2 | K_2 | V_2 | ... | Q_n | K_n | V_n]

    Per group, Q has ``(num_heads / num_kv_heads) * head_dim`` rows, K has
    ``head_dim`` rows, and V has ``v_head_dim`` rows.

    Two fp8 block-scale layouts exist in the wild and both are supported,
    detected from ``s_full.shape[0]``:

    * per-group padded: ``num_kv_heads * ceil(rows_per_group / block)`` rows;
      every group's scales are padded to whole blocks (the full-attention
      layers of MiMo-V2.6-Flash: 4 groups x 3392 rows -> 4 x 27 = 108 rows).
    * globally packed: ``ceil(total_rows / block)`` rows with no per-group
      alignment, so a scale block may straddle a group boundary (the SWA
      layers of MiMo-V2.6-Flash: 8 groups x 1856 rows -> 116 rows; a padded
      layout would have 8 x 15 = 120).

    Rank assignment: with ``tp_size <= num_kv_heads`` each rank owns
    ``num_kv_heads / tp_size`` whole groups. With ``tp_size > num_kv_heads``
    (e.g. TP8 against the 4 KV heads of the full-attention layers) each KV
    group is replicated across ``tp_size / num_kv_heads`` ranks, and each of
    those ranks takes a distinct contiguous slice of the group's Q rows;
    ``QKVParallelLinear`` builds the matching per-rank parameter shapes via
    ``num_kv_head_replicas``.

    The forward expects the rank's rows de-interleaved into a single Q, K and
    V block:

        [Q_1 | Q_2 | ... | Q_g | K_1 | K_2 | ... | K_g | V_1 | V_2 | ... | V_g]

    A plain chunk reaches that layout only when exactly one whole padded
    group lands on one rank and there is no replication. Otherwise the rank's
    rows are dequantized to float (dropping the block constraint: K is 192
    rows = 1.5 blocks, so blocks straddle row boundaries under any
    permutation), reordered, and re-quantized to fp8.
    """
    q_rows_per_group = (num_heads // num_kv_heads) * head_dim
    k_rows_per_group = head_dim
    v_rows_per_group = v_head_dim
    rows_per_group = q_rows_per_group + k_rows_per_group + v_rows_per_group
    total_rows = num_kv_heads * rows_per_group
    w_cols = w_full.shape[1]
    if w_full.shape[0] != total_rows:
        raise ValueError(
            f"fp8 qkv_proj weight has {w_full.shape[0]} rows; expected "
            f"{total_rows} (num_kv_heads={num_kv_heads}, "
            f"rows_per_group={rows_per_group})."
        )

    # Detect the scale layout (the two agree when rows_per_group % block == 0).
    per_group_scale_rows = -(-rows_per_group // block)
    padded_scale_rows = num_kv_heads * per_group_scale_rows
    packed_scale_rows = -(-total_rows // block)
    if s_full.shape[0] == padded_scale_rows:
        scale_is_padded = True
    elif s_full.shape[0] == packed_scale_rows:
        scale_is_padded = False
    else:
        raise ValueError(
            f"fp8 qkv_proj scale has {s_full.shape[0]} rows; expected the "
            f"per-group padded layout ({padded_scale_rows} rows) or the "
            f"globally packed layout ({packed_scale_rows} rows)."
        )

    # Assign KV groups to this rank, replicating groups when TP exceeds the
    # number of KV heads.
    if tp_size <= num_kv_heads:
        if num_kv_heads % tp_size != 0:
            raise ValueError(
                f"TP size ({tp_size}) must evenly divide the number of KV "
                f"heads ({num_kv_heads})."
            )
        kv_replicas = 1
        group_start = tp_rank * (num_kv_heads // tp_size)
        group_end = group_start + num_kv_heads // tp_size
        q_offset_in_group = 0
    else:
        if tp_size % num_kv_heads != 0:
            raise ValueError(
                f"TP size ({tp_size}) must be a multiple of the number of "
                f"KV heads ({num_kv_heads}) for KV-head replication."
            )
        kv_replicas = tp_size // num_kv_heads
        group_start = tp_rank // kv_replicas
        group_end = group_start + 1
        q_offset_in_group = (tp_rank % kv_replicas) * (
            q_rows_per_group // kv_replicas
        )
    q_rows_per_rank = q_rows_per_group // kv_replicas

    # Fast path: one whole padded group per rank and no replication — the
    # rank's slice is already [Q | K | V] and block-aligned, so plain chunks
    # shard it without re-quantization.
    if group_end - group_start == 1 and kv_replicas == 1 and scale_is_padded:
        w = w_full.chunk(tp_size, dim=0)[tp_rank]
        s = s_full.chunk(tp_size, dim=0)[tp_rank]
        return w, s

    # Dequantize this rank's groups to float.
    if scale_is_padded:
        w_deq_groups = []
        for g_idx in range(group_start, group_end):
            row_start = g_idx * rows_per_group
            scale_row_start = g_idx * per_group_scale_rows
            s_g = s_full[scale_row_start : scale_row_start + per_group_scale_rows]
            s_g_expanded = s_g.repeat_interleave(block, dim=0).repeat_interleave(
                block, dim=1
            )[:rows_per_group, :w_cols]
            w_g = w_full[row_start : row_start + rows_per_group].to(torch.float32)
            w_deq_groups.append(w_g * s_g_expanded.to(torch.float32))
    else:
        # Globally packed scales: expand over the whole tensor (blocks may
        # straddle group boundaries), then slice groups out of the result.
        s_expanded = s_full.repeat_interleave(block, dim=0).repeat_interleave(
            block, dim=1
        )[:total_rows, :w_cols]
        w_deq = w_full.to(torch.float32) * s_expanded.to(torch.float32)
        w_deq_groups = [
            w_deq[g_idx * rows_per_group : (g_idx + 1) * rows_per_group]
            for g_idx in range(group_start, group_end)
        ]

    k_end = q_rows_per_group + k_rows_per_group
    qs, ks, vs = [], [], []
    for w_g_deq in w_deq_groups:
        qs.append(w_g_deq[q_offset_in_group : q_offset_in_group + q_rows_per_rank])
        ks.append(w_g_deq[q_rows_per_group:k_end])
        vs.append(w_g_deq[k_end : k_end + v_rows_per_group])

    # Combine into [Q_1, ..., Q_g, K_1, ..., K_g, V_1, ..., V_g].
    grouped = torch.cat([torch.cat(qs), torch.cat(ks), torch.cat(vs)], dim=0)
    # scaled_quantize requires whole ``block``-row groups (e.g. TP8 replicas
    # own 1856 rows = 14.5 blocks). Zero-pad the tail block; the caller
    # truncates the returned weight to the parameter size, and the zero rows
    # do not affect the real rows' block scales.
    pad_rows = -grouped.shape[0] % block
    if pad_rows:
        grouped = torch.nn.functional.pad(grouped, (0, 0, 0, pad_rows))
    return scaled_quantize(
        grouped, GroupShape(block, block), w_full.dtype, compute_dtype=torch.float32
    )


@support_torch_compile
class MiMoV2Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config
        eplb_config = vllm_config.parallel_config.eplb_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.num_redundant_experts = eplb_config.num_redundant_experts

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiMoV2FlashDecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping: list[tuple[str, str, str | int]] = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        # Pro-format fused qkv_proj arrives as two tensors (weight and
        # weight_scale_inv). Store them per-layer so that they can be
        # sharded together.
        pending_fp8_qkv_proj: dict[str, dict[str, torch.Tensor]] = {}
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if "mtp" in name:
                continue

            expert_matched = False
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue

                name_rewritten = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name_rewritten, self):
                    continue

                if (
                    name_rewritten.endswith(".bias") or name_rewritten.endswith("_bias")
                ) and name_rewritten not in params_dict:
                    continue

                if name_rewritten not in params_dict:
                    continue

                param = params_dict[name_rewritten]
                weight_loader = param.weight_loader

                weight_loader(
                    param,
                    loaded_weight,
                    name_rewritten,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                loaded_params.add(name_rewritten)
                expert_matched = True
                break

            if expert_matched:
                continue
            # Support fused qkv_proj checkpoint (Pro format)
            if self._try_load_fp8_qkv_proj(
                name,
                loaded_weight,
                pending_fp8_qkv_proj,
                params_dict,
                loaded_params,
                tp_rank,
                tp_size,
            ):
                continue
            stacked_matched = False
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name_rewritten = name.replace(weight_name, param_name)

                if (
                    name_rewritten.endswith(".bias")
                    and name_rewritten not in params_dict
                ):
                    continue

                if is_pp_missing_parameter(name_rewritten, self):
                    continue

                if name_rewritten not in params_dict:
                    continue

                param = params_dict[name_rewritten]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name_rewritten)

                stacked_matched = True
                break

            if stacked_matched:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            orig_name = name
            mapped_name = maybe_remap_kv_scale_name(name, params_dict)
            name = mapped_name if mapped_name is not None else orig_name

            if name not in params_dict:
                continue

            param = params_dict[name]

            if "attention_sink_bias" in name:
                total_heads = loaded_weight.shape[0]
                heads_per_rank = total_heads // tp_size
                head_start = tp_rank * heads_per_rank
                narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)

                param.data.copy_(narrow_weight)
                loaded_params.add(name)
            else:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params

    def _try_load_fp8_qkv_proj(
        self,
        name: str,
        tensor: torch.Tensor,
        fp8_qkv_proj_dict: dict[str, dict[str, torch.Tensor]],
        params_dict: dict[str, torch.nn.Parameter],
        loaded_params: set[str],
        tp_rank: int,
        tp_size: int,
    ) -> bool:
        """
        The fused fp8 QKV projection weights and scale are stored separately.
        Special care must be taken while sharding these tensors across TP ranks.
        See _shard_fp8_qkv_proj for more details.

        Returns:
            True if ``tensor`` was an fp8 qkv_proj weight/scale and was consumed
            (caller should skip it); False otherwise, so the caller falls
            through to its normal loading path.
        """
        is_weight = (
            name.endswith("qkv_proj.weight") and tensor.dtype == torch.float8_e4m3fn
        )
        is_scale = name.endswith("qkv_proj.weight_scale_inv")
        if not is_weight and not is_scale:
            # Weight is not in FP8 format. Ignore.
            return False

        if is_pp_missing_parameter(name, self):
            # This qkv_proj is for a layer not on this PP rank.
            return True

        prefix, qkv_kind = name.rsplit(".", 1)
        entry = fp8_qkv_proj_dict.setdefault(prefix, {})
        entry[qkv_kind] = tensor
        if "weight" not in entry or "weight_scale_inv" not in entry:
            # Still waiting for the other param.
            return True
        del fp8_qkv_proj_dict[prefix]

        # Get self_attn module, which is a parent of qkv_proj.
        attn = self.get_submodule(prefix.rsplit(".", 1)[0])

        # Shard the qkv_proj per-rank.
        w_rank, s_rank = _shard_fp8_qkv_proj(
            entry["weight"],
            entry["weight_scale_inv"],
            num_heads=attn.total_num_heads,
            num_kv_heads=attn.total_num_kv_heads,
            head_dim=attn.head_dim,
            v_head_dim=attn.v_head_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        sharded = {"weight": w_rank, "weight_scale_inv": s_rank}
        for kind, tensor in sharded.items():
            param_name = f"{prefix}.{kind}"
            param = params_dict[param_name]
            if tensor.shape[0] > param.shape[0]:
                tensor = tensor[: param.shape[0]]
            default_weight_loader(param, tensor)
            loaded_params.add(param_name)
        return True


class MiMoV2FlashForCausalLM(nn.Module, SupportsPP, MixtureOfExperts, SupportsEagle3):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.model = MiMoV2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class MiMoV2ForCausalLM(MiMoV2FlashForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
