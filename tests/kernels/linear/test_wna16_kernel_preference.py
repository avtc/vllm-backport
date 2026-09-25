# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_WNA16_PREFER_KERNEL: prefer marlin or humming with fallback.

Reorders kernel-candidate lists (linear MP kernels and WNA16 MoE backends);
capability checks still skip incompatible candidates, so unsupported configs
fall back to the remaining ones unchanged.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from vllm.model_executor.kernels.linear.preference import apply_kernel_preference


class FakeKernel:
    def __init__(self, name: str) -> None:
        self.__name__ = name


CANDIDATES = [
    FakeKernel("CutlassW4A8LinearKernel"),
    FakeKernel("MarlinLinearKernel"),
    FakeKernel("ExllamaLinearKernel"),
    FakeKernel("HummingLinearKernel"),
]

NAMES = [k.__name__ for k in CANDIDATES]


def _clear_envs_cache() -> None:
    import vllm.envs as envs

    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()


@pytest.fixture(autouse=True)
def _clean_envs_cache():
    yield
    _clear_envs_cache()


def test_default_auto_is_identity():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VLLM_WNA16_PREFER_KERNEL", None)
        _clear_envs_cache()
        assert [k.__name__ for k in apply_kernel_preference(CANDIDATES)] == NAMES


def test_prefer_marlin_moves_marlin_first_and_keeps_rest_stable():
    with patch.dict(os.environ, {"VLLM_WNA16_PREFER_KERNEL": "marlin"}):
        _clear_envs_cache()
        got = [k.__name__ for k in apply_kernel_preference(CANDIDATES)]
    assert got == [
        "MarlinLinearKernel",
        "CutlassW4A8LinearKernel",
        "ExllamaLinearKernel",
        "HummingLinearKernel",
    ]


def test_prefer_humming_moves_humming_first():
    with patch.dict(os.environ, {"VLLM_WNA16_PREFER_KERNEL": "humming"}):
        _clear_envs_cache()
        got = [k.__name__ for k in apply_kernel_preference(CANDIDATES)]
    assert got[0] == "HummingLinearKernel"
    # everything else keeps the original relative order
    assert got[1:] == [
        "CutlassW4A8LinearKernel",
        "MarlinLinearKernel",
        "ExllamaLinearKernel",
    ]


def test_preference_is_case_insensitive():
    with patch.dict(os.environ, {"VLLM_WNA16_PREFER_KERNEL": "Marlin"}):
        _clear_envs_cache()
        assert apply_kernel_preference(CANDIDATES)[0].__name__ == "MarlinLinearKernel"


def test_unknown_family_is_noop():
    with patch.dict(os.environ, {"VLLM_WNA16_PREFER_KERNEL": "conch"}):
        _clear_envs_cache()
        assert [k.__name__ for k in apply_kernel_preference(CANDIDATES)] == NAMES


def test_envs_default_value():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VLLM_WNA16_PREFER_KERNEL", None)
        _clear_envs_cache()
        import vllm.envs as envs

        assert envs.VLLM_WNA16_PREFER_KERNEL == "auto"
