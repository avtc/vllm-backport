# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DraftTokensHandler: latest-drafts tracking for grammar validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DraftTokensHandler needs a CUDA stream"
)


def _handler() -> DraftTokensHandler:
    return DraftTokensHandler.__new__(DraftTokensHandler)


def _batch(req_ids: list[str], structured: bool) -> MagicMock:
    batch = MagicMock(spec=InputBatch)
    batch.req_ids = req_ids
    batch.has_structured_output_reqs = structured
    return batch


def _drafts(rows: int, cols: int) -> torch.Tensor:
    return torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)


@pytest.fixture
def handler() -> DraftTokensHandler:
    h = _handler()
    h.device = torch.device("cuda")
    h.copy_stream = torch.cuda.Stream(device="cuda")
    h.copy_event = torch.cuda.Event(blocking=True)
    h.req_ids: list[str] = []
    h.draft_tokens_np: np.ndarray | None = None
    h.num_draft_tokens = 0
    h.latest_drafts: dict[str, list[int]] = {}
    h.copy_pending = False
    return h


def test_structured_then_plain_batch_does_not_crash(handler):
    """A structured batch followed by a batch without structured requests
    must not crash _collect (the drafts survive into latest_drafts)."""
    handler.set_draft_tokens(_batch(["r0"], True), _drafts(1, 2))
    assert handler.copy_pending is True
    handler.set_draft_tokens(_batch(["r0"], False), _drafts(1, 2))
    # Would previously assert: draft_tokens_np=None with copy_pending=True.
    result = handler.get_draft_tokens()
    assert result is not None
    assert result.token_ids["r0"] == [0, 1]


def test_plain_batch_before_any_structured_batch_is_noop(handler):
    handler.set_draft_tokens(_batch(["r0"], False), _drafts(1, 2))
    assert handler.copy_pending is False
    assert handler.get_draft_tokens() is None


def test_draft_tokens_handler_tracks_latest_drafts(handler):
    drafts = _drafts(2, 3)
    handler.set_draft_tokens(_batch(["a", "b"], True), drafts)
    got = handler.get_draft_tokens()
    assert got is not None
    assert got.token_ids == {"a": [0, 1, 2], "b": [3, 4, 5]}


def test_remove_request_evicts_drafts(handler):
    handler.set_draft_tokens(_batch(["a", "b"], True), _drafts(2, 2))
    handler.remove_request("a")
    got = handler.get_draft_tokens()
    assert got is not None
    assert "a" not in got.token_ids
    assert got.token_ids["b"] == [2, 3]


def test_remove_request_with_pending_copy(handler):
    handler.set_draft_tokens(_batch(["a"], True), _drafts(1, 2))
    handler.remove_request("a")
    assert handler.copy_pending is False
    assert handler.latest_drafts == {}
