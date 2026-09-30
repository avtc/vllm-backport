"""PP step phase timer: no-op without the env, accumulates and dumps with it."""

import time

import pytest

import vllm.envs as envs
from vllm.v1.worker.gpu.pp_timing import PPPhaseTimer


def test_phase_timer_noop_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "VLLM_PP_TIMING_DEBUG", False, raising=False)
    t = PPPhaseTimer()
    with t.phase("recv_wait"):
        time.sleep(0.001)
    t.step_done()
    assert t.accum == {}


def test_phase_timer_accumulates_and_dumps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "VLLM_PP_TIMING_DEBUG", True, raising=False)
    t = PPPhaseTimer()
    t.DUMP_EVERY = 2
    with t.phase("recv_wait"):
        time.sleep(0.002)
    with t.phase("forward"):
        time.sleep(0.001)
    t.step_done()
    t.step_done()  # triggers the dump at count=2
    assert t.accum["recv_wait"] >= 0.002
    assert t.accum["forward"] >= 0.001
