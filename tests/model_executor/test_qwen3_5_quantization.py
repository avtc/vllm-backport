# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest


def test_qwen3_5_lm_head_receives_quant_config():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config
    mock_vllm_config.lora_config = None

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5.Qwen3_5Model") as MockModel,
        patch("vllm.model_executor.models.qwen3_5.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()

        Qwen3_5ForCausalLMBase(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def _mock_model_config(**overrides):
    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64
    mock_hf_config.rms_norm_eps = 1e-6
    mock_hf_config.layer_types = ["full_attention"]
    mock_hf_config.num_hidden_layers = 1
    mock_hf_config.model_type = "qwen3_5"
    for key, value in overrides.items():
        setattr(mock_hf_config, key, value)
    return mock_hf_config


def test_qwen3_5_embed_tokens_receives_quant_config_and_prefix(default_vllm_config):
    """Quantized checkpoints pack embed_tokens (weight_packed/weight_scale);
    the embedding must build through the quant config or loading fails."""
    import torch.nn as nn

    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

    mock_quant_config = Mock()

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = _mock_model_config()
    mock_vllm_config.parallel_config.eplb_config.num_redundant_experts = 0
    mock_vllm_config.quant_config = mock_quant_config

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5.VocabParallelEmbedding") as MockEmb,
        patch(
            "vllm.model_executor.models.qwen3_5.make_layers",
            return_value=(0, 1, nn.ModuleList()),
        ),
        patch(
            "vllm.model_executor.models.qwen3_5.get_pp_group",
            return_value=mock_pp_group,
        ),
        patch("vllm.model_executor.models.qwen3_5.Qwen3_5RMSNorm"),
    ):
        Qwen3_5Model(vllm_config=mock_vllm_config, prefix="model")

        MockEmb.assert_called_once()
        call_kwargs = MockEmb.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config
        assert call_kwargs["prefix"] == "model.embed_tokens"


def test_qwen3_next_embed_tokens_receives_quant_config_and_prefix(default_vllm_config):
    """Qwen3NextModel shares the Qwen3.5 embed_tokens wiring pattern."""
    import torch.nn as nn

    from vllm.model_executor.models.qwen3_next import Qwen3NextModel

    mock_quant_config = Mock()

    mock_hf_config = _mock_model_config(model_type="qwen3_next")

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.parallel_config.eplb_config.num_redundant_experts = 0
    mock_vllm_config.quant_config = mock_quant_config

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch(
            "vllm.model_executor.models.qwen3_next.VocabParallelEmbedding"
        ) as MockEmb,
        patch(
            "vllm.model_executor.models.qwen3_next.make_layers",
            return_value=(0, 1, nn.ModuleList()),
        ),
        patch(
            "vllm.model_executor.models.qwen3_next.get_pp_group",
            return_value=mock_pp_group,
        ),
        patch("vllm.model_executor.models.qwen3_next.Qwen3NextRMSNorm"),
    ):
        Qwen3NextModel(vllm_config=mock_vllm_config, prefix="model")

        MockEmb.assert_called_once()
        call_kwargs = MockEmb.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config
        assert call_kwargs["prefix"] == "model.embed_tokens"


def _compressed_tensors_config_for_vl_targets(target):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )

    return CompressedTensorsConfig.from_config(
        {
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": [target, "re:.*embed_tokens"],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "group",
                        "group_size": 128,
                        "symmetric": True,
                        "dynamic": False,
                    },
                }
            },
        }
    )


def _mock_causal_lm_vllm_config(quant_config):
    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = _mock_model_config()
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = quant_config
    mock_vllm_config.lora_config = None
    return mock_vllm_config


@pytest.mark.parametrize(
    ("prefix", "target_forms", "aliased_target"),
    [
        # text-only: loader renames "model.language_model.*" -> "model.*"
        (
            "",
            [
                "model.language_model.layers.0.self_attn.q_proj",
                "language_model.model.layers.0.self_attn.q_proj",
            ],
            "model.layers.0.self_attn.q_proj",
        ),
        # VL: loader renames to the wrapper-relative
        # "language_model.model.*" while the backbone sits at
        # "model.language_model.model.*"
        (
            "model.language_model",
            [
                "model.language_model.layers.0.self_attn.q_proj",
                "language_model.model.layers.0.self_attn.q_proj",
            ],
            "model.language_model.model.layers.0.self_attn.q_proj",
        ),
    ],
)
def test_qwen3_5_aliases_quant_targets_to_vllm_names(
    prefix, target_forms, aliased_target
):
    """Quant targets use HF checkpoint names; without aliasing onto this
    model's construction names the scheme lookup misses and quantized layers
    build unquantized, crashing on their packed weights."""
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    for target_form in target_forms:
        quant_config = _compressed_tensors_config_for_vl_targets(target_form)
        mock_vllm_config = _mock_causal_lm_vllm_config(quant_config)

        mock_pp_group = Mock()
        mock_pp_group.is_last_rank = True

        with (
            patch("vllm.model_executor.models.qwen3_5.Qwen3_5Model") as MockModel,
            patch("vllm.model_executor.models.qwen3_5.ParallelLMHead"),
            patch("vllm.model_executor.models.qwen3_5.LogitsProcessor"),
            patch(
                "vllm.model_executor.models.qwen3_5.get_pp_group",
                return_value=mock_pp_group,
            ),
        ):
            MockModel.return_value.make_empty_intermediate_tensors = Mock()

            Qwen3_5ForCausalLMBase(vllm_config=mock_vllm_config, prefix=prefix)

            assert aliased_target in quant_config.target_scheme_map
            assert (
                quant_config.target_scheme_map[aliased_target]
                is quant_config.target_scheme_map[target_form]
            )
            assert "re:.*embed_tokens" in quant_config.target_scheme_map


def test_qwen3_5_mtp_embed_tokens_receives_quant_config_and_prefix(
    default_vllm_config,
):
    """With mtp_use_dedicated_embeddings=false the drafter reloads the
    (packed, quantized) target embed_tokens, so its own embedding must build
    through the quant config."""
    import torch.nn as nn

    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor

    mock_quant_config = Mock()

    mock_hf_config = _mock_model_config(mtp_num_hidden_layers=1)

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.quant_config = mock_quant_config

    with (
        patch(
            "vllm.model_executor.models.qwen3_5_mtp.VocabParallelEmbedding"
        ) as MockEmb,
        patch("vllm.model_executor.models.qwen3_5_mtp.ColumnParallelLinear"),
        patch(
            "vllm.model_executor.models.qwen3_5_mtp.Qwen3_5DecoderLayer",
            return_value=nn.Module(),
        ),
        patch("vllm.model_executor.models.qwen3_5_mtp.Qwen3_5RMSNorm"),
    ):
        Qwen3_5MultiTokenPredictor(vllm_config=mock_vllm_config, prefix="mtp")

        MockEmb.assert_called_once()
        call_kwargs = MockEmb.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config
        assert call_kwargs["prefix"] == "mtp.embed_tokens"


def test_qwen3_5_mtp_lm_head_receives_quant_config():
    from vllm.config import CompilationMode
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.compilation_config.mode = CompilationMode.NONE
    mock_vllm_config.quant_config = mock_quant_config

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5_mtp.Qwen3_5MultiTokenPredictor"),
        patch("vllm.model_executor.models.qwen3_5_mtp.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5_mtp.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5_mtp.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        Qwen3_5MTP(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config
