"""Unit tests for the env-gated Qwen4Exp layer correctness probe."""

import logging

import torch

from vllm.models.qwen4_exp.nvidia.model import _q4e_probe


def test_probe_records_baseline_then_flags_divergence(caplog):
    state = {"step": 5, "logged": set()}
    with caplog.at_level(logging.INFO, logger="vllm.models.qwen4_exp.nvidia.model"):
        _q4e_probe("L0.attn_out", torch.ones(8) * 2.0, state)
        assert state["base"] == 2.0
        # 21x baseline -> diverged, logged once
        _q4e_probe("L0.attn_out", torch.ones(8) * 42.0, state)
        assert any("DIVERGED" in r.message for r in caplog.records)
        # flood control: a second divergence is not logged again
        caplog.clear()
        _q4e_probe("L0.attn_out", torch.ones(8) * 99.0, state)
        assert not any("DIVERGED" in r.message for r in caplog.records)


def test_probe_flags_nonfinite():
    state = {"step": 1, "logged": set()}
    _q4e_probe("L1.mlp_out", torch.ones(4), state)
    assert state["base"] == 1.0
    # NaN/Inf content diverges even though absmax stays near the baseline.
    bad = torch.tensor([1.0, float("nan"), float("inf"), 1.0])
    _q4e_probe("L1.mlp_out", bad, state)
    assert "L1.mlp_out" in state["logged"]


def test_probe_interval_logging():
    state = {"step": 0, "logged": set()}
    _q4e_probe("L2.residual", torch.ones(4), state)  # baseline
    state["step"] = 64

    records = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    lg = logging.getLogger("vllm.models.qwen4_exp.nvidia.model")
    lg.addHandler(handler)
    try:
        _q4e_probe("L2.residual", torch.ones(4) * 3.0, state)
    finally:
        lg.removeHandler(handler)
    assert any("step=64" in m for m in records)


def test_int6_planes_env_escape(monkeypatch):
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia import model as q4e_model

    class CompressedTensorsWNA16:
        pass

    linear = SimpleNamespace(
        scheme=SimpleNamespace(
            __class__=CompressedTensorsWNA16,
            num_bits=6,
            symmetric=True,
            group_size=64,
        ),
        input_size_per_partition=2560,
        output_size_per_partition=512,
    )
    linear.scheme.__class__ = CompressedTensorsWNA16
    monkeypatch.setenv("VLLM_QWEN4EXP_INT6_PLANES", "1")
    assert q4e_model._int6_planes_group_size(linear) == 64
    monkeypatch.setenv("VLLM_QWEN4EXP_INT6_PLANES", "0")
    assert q4e_model._int6_planes_group_size(linear) is None
