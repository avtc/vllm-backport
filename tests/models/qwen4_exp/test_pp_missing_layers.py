# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP-conditional construction of embed_tokens / lm_head (mem-01).

Only the first pipeline rank embeds tokens and only the last computes
logits; the other ranks hold PPMissingLayer placeholders instead of an
unused ~0.6 GiB copy of each quantized vocab table. AutoWeightsLoader
skips StageMissingLayer/PPMissingLayer modules when loading.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = PPMissingLayer()
        self.used = nn.Linear(4, 4, bias=False)


def test_pp_missing_layer_load_skipped():
    """AutoWeightsLoader ignores weights targeting PPMissingLayer modules."""
    model = _Tiny()
    weights = [
        ("embed_tokens.weight", torch.randn(8, 4)),
        ("used.weight", torch.randn(4, 4)),
    ]
    loaded = AutoWeightsLoader(model).load_weights(weights)
    # only the live module's name is reported; nothing materializes on the
    # placeholder (it has no parameters at all)
    assert set(loaded) == {"used.weight"}
    assert list(model.embed_tokens.named_parameters()) == []
    assert model.used.weight.shape == (4, 4)
