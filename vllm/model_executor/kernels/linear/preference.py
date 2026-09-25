# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_WNA16_PREFER_KERNEL reordering: marlin/humming preference with
automatic fallback to any other kernel that can implement the config."""

from __future__ import annotations

from typing import Protocol


class _Named(Protocol):
    __name__: str


def apply_kernel_preference(items: list[_Named]) -> list[_Named]:
    """Stable-reorder ``items`` moving the preferred WNA16 kernel family first.

    VLLM_WNA16_PREFER_KERNEL=marlin|humming moves that family to the front of
    the candidate list; incompatible members are still skipped by the caller's
    capability checks, so unsupported configs fall back to the remaining
    candidates unchanged (prefer-with-fallback, not force).
    """
    import vllm.envs as envs

    prefer = envs.VLLM_WNA16_PREFER_KERNEL
    if prefer in ("", "auto"):
        return items
    return sorted(
        items, key=lambda it: 0 if it.__name__.lower().startswith(prefer) else 1
    )
