# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred pipeline-parallel receive (VLLM_PP_DEFER_RECV, decode-03).

Non-first PP ranks receive intermediate tensors through
DeferredRecvIntermediateTensors: the actual irecv_tensor_dict runs when the
model runner first reads ``.tensors`` (right before the forward), not at the
top of execute_model, so the rank's input/attention-metadata preparation
overlaps the previous rank's GPU work.
"""

from __future__ import annotations

import torch

from vllm.v1.worker.gpu_worker import DeferredRecvIntermediateTensors


def test_deferred_recv_runs_only_on_first_read():
    calls = []

    def fake_recv():
        calls.append(1)
        return ({"hidden_states": torch.zeros(2)}, None, None)

    t = DeferredRecvIntermediateTensors(fake_recv)
    assert calls == []  # receive deferred, nothing touched yet
    first = t.tensors  # __getattribute__ gates .tensors on wait_for_comm
    assert calls == [1]
    assert first["hidden_states"].shape == (2,)
    _ = t.tensors  # second read does not receive again
    assert calls == [1]


def test_deferred_recv_wait_for_comm_is_idempotent():
    calls = []
    waits = []

    def fake_recv():
        calls.append(1)
        # Handles are objects with a wait() method.
        handle = lambda: None  # noqa: E731
        handle.wait = lambda: waits.append(1)
        return ({"hidden_states": torch.zeros(1)}, [handle], [lambda: None])

    t = DeferredRecvIntermediateTensors(fake_recv)
    t.wait_for_comm()
    t.wait_for_comm()
    assert calls == [1]
    assert len(t._comm_handles) == 1
    assert waits == [1]
    assert t._comm_waited is True


def test_deferred_recv_wait_for_comm_swallows_secondary_errors():
    """The trailing wait must not mask the original exception when the recv
    itself also fails (the caller only cares about the forward's error)."""
    calls = []

    def bad_recv():
        calls.append(1)
        raise RuntimeError("recv failed")

    t = DeferredRecvIntermediateTensors(bad_recv)
    try:
        _ = t.tensors
    except RuntimeError as e:
        assert "recv failed" in str(e)
    assert calls == [1]
    # A second wait does not re-run the broken recv.
    t.wait_for_comm()
    assert calls == [1]
