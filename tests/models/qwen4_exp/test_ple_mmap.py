# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVMe-backed (mmap) PLE n-gram table: host-gather semantics + file lifecycle.

VLLM_PLE_MMAP_PATH backs the per-rank table shard with a file mapping instead
of GPU residency (~6.4 GiB/GPU freed) or pinned host RAM (no 51 GiB mlock).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
    _FP8_STORAGE_DTYPES,
    _gather_rows_from_table,
)


def test_gather_in_range_rows_only():
    table = torch.arange(0, 24, dtype=torch.bfloat16).reshape(8, 3)
    ids = torch.tensor([10, 12, 9])  # window [9, 12): 12 is EXCLUDED
    out = _gather_rows_from_table(table, ids, vocab_start=9, vocab_end=12)
    assert out.shape == (3, 3)
    assert torch.equal(out[0], table[1])  # 10 - 9 = 1
    assert torch.equal(out[1], torch.zeros(3))  # 12 is out of range
    assert torch.equal(out[2], table[0])  # 9 - 9 = 0


def test_gather_out_of_range_rows_are_zero():
    table = torch.randn(8, 4, dtype=torch.float32)
    ids = torch.tensor([0, 100, -5, 7])  # only 7 in [0, 8)
    out = _gather_rows_from_table(table, ids, vocab_start=0, vocab_end=8)
    assert torch.equal(out[1], torch.zeros(4))
    assert torch.equal(out[2], torch.zeros(4))
    assert torch.equal(out[3], table[7])


def test_gather_fp8_rows_bit_exact():
    table = torch.randn(16, 8).to(torch.float8_e4m3fn)
    assert table.dtype in _FP8_STORAGE_DTYPES
    ids = torch.tensor([3, 5])
    out = _gather_rows_from_table(table, ids, vocab_start=0, vocab_end=16)
    assert out.dtype == torch.float8_e4m3fn
    assert torch.equal(out.view(torch.uint8)[0], table.view(torch.uint8)[3])
    assert torch.equal(out.view(torch.uint8)[1], table.view(torch.uint8)[5])


def test_gather_empty_ids():
    table = torch.zeros(4, 3, dtype=torch.bfloat16)
    out = _gather_rows_from_table(table, torch.tensor([], dtype=torch.long), 0, 4)
    assert out.shape == (0, 3)


def test_mmap_roundtrip_persists_rows(tmp_path):
    """A tensor written through the mmap mapping is readable after re-open.

    Views keep the mapping exported (mmap.close raises BufferError while
    numpy/torch hold pointers), so each mapping lives in a function scope
    that drops the views before the mapping closes.
    """
    import mmap as _mmap

    import numpy as np

    path = tmp_path / "table.bin"
    rows, dim = 128, 16
    nbytes = rows * dim * 2  # bf16

    def write():
        f = open(path, "w+b")  # noqa: SIM115
        try:
            f.truncate(nbytes)
            mm = _mmap.mmap(f.fileno(), 0)
            try:
                arr = np.frombuffer(mm, dtype=np.uint8, count=nbytes)
                t = torch.frombuffer(arr, dtype=torch.uint8)
                t = t.view(torch.bfloat16).reshape(rows, dim)
                t[7] = torch.arange(dim, dtype=torch.bfloat16)
                mm.flush()
            finally:
                del t, arr
                mm.close()
        finally:
            f.close()

    def read() -> torch.Tensor:
        f = open(path, "rb")  # noqa: SIM115
        try:
            mm = _mmap.mmap(f.fileno(), 0)
            try:
                arr = np.frombuffer(mm, dtype=np.uint8, count=nbytes)
                t = torch.frombuffer(arr, dtype=torch.uint8)
                return t.view(torch.bfloat16).reshape(rows, dim).clone()
            finally:
                del t, arr
                mm.close()
        finally:
            f.close()

    write()
    t = read()
    assert torch.equal(t[7], torch.arange(dim, dtype=torch.bfloat16))
    # untouched rows read back zero (fresh file is zero-filled)
    assert torch.count_nonzero(t[0]) == 0


def test_mmap_table_path_is_per_rank(tmp_path):
    from vllm.models.qwen4_exp.nvidia import ngram_embedding as ne

    class _FakeGroup:
        def __init__(self, rank):
            self._rank = rank

        def size(self):
            return 8

    _FakeEtP = type("G", (), {"device_group": _FakeGroup(5)})
    fake_dist = type(
        "M",
        (),
        {"get_rank": staticmethod(lambda group: group._rank)},
    )
    with (
        patch.dict(os.environ, {"VLLM_PLE_MMAP_PATH": str(tmp_path / "ple")}),
        patch.object(torch, "distributed", fake_dist),
        patch.object(ne, "get_etp_group", lambda: _FakeEtP(5)),
    ):
        from vllm import envs

        if hasattr(envs.__getattr__, "cache_clear"):
            envs.__getattr__.cache_clear()
        assert ne._mmap_table_path().endswith(f"{os.sep}ple.rank5")


def test_envs_defaults():
    from vllm import envs

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VLLM_PLE_MMAP_PATH", None)
        os.environ.pop("VLLM_PLE_MMAP_REBUILD", None)
        if hasattr(envs.__getattr__, "cache_clear"):
            envs.__getattr__.cache_clear()
        os.environ.pop("VLLM_PLE_MMAP_PIN_STAGING", None)
        assert envs.VLLM_PLE_MMAP_PATH is None
        assert envs.VLLM_PLE_MMAP_REBUILD is False
        assert envs.VLLM_PLE_MMAP_PIN_STAGING is True


def test_tensor_from_mapping_roundtrip(tmp_path):
    """The allocate path (mmap -> np uint8 -> torch view) holds for bf16 and
    fp8 storage dtypes and writes through to the file."""
    import mmap as _mmap

    import numpy as np

    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        rows, dim = 32, 8
        nbytes = rows * dim * torch.empty((), dtype=dtype).element_size()
        path = tmp_path / f"t_{dtype}.bin"
        with open(path, "w+b") as f:
            f.truncate(nbytes)
            with _mmap.mmap(f.fileno(), 0) as mm:
                arr = np.frombuffer(mm, dtype=np.uint8, count=nbytes)
                t = (
                    torch.frombuffer(arr, dtype=torch.uint8)
                    .view(dtype)
                    .reshape(rows, dim)
                )
                t[5] = torch.full((dim,), 1.0).to(dtype)
                mm.flush()
        with open(path, "rb") as f, _mmap.mmap(f.fileno(), nbytes) as mm:
            arr = np.frombuffer(mm, dtype=np.uint8, count=nbytes)
            t = torch.frombuffer(arr, dtype=torch.uint8).view(dtype).reshape(rows, dim)
            assert t[5][0] == torch.tensor(1.0).to(dtype)
            assert torch.count_nonzero(t[0].to(torch.uint8)) == 0


def test_clamp_cudagraph_mode_for_host_gather():
    from vllm.config.compilation import CUDAGraphMode
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
        _clamp_cudagraph_mode_for_host_gather,
    )

    for mode in (
        CUDAGraphMode.NONE,
        CUDAGraphMode.PIECEWISE,
    ):
        assert _clamp_cudagraph_mode_for_host_gather(mode, True) == (mode, None)
        assert _clamp_cudagraph_mode_for_host_gather(mode, False) == (mode, None)

    for mode in (
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ):
        clamped, reason = _clamp_cudagraph_mode_for_host_gather(mode, True)
        assert clamped == CUDAGraphMode.PIECEWISE
        assert reason is not None
        clamped, reason = _clamp_cudagraph_mode_for_host_gather(mode, False)
        assert clamped == CUDAGraphMode.NONE
        assert reason is not None
        # With the gather moved to prepare_inputs, forward is pure GPU ops
        # and FULL graphs stay sound: every mode passes through untouched.
        assert _clamp_cudagraph_mode_for_host_gather(mode, False, True) == (
            mode,
            None,
        )
        assert _clamp_cudagraph_mode_for_host_gather(mode, True, True) == (
            mode,
            None,
        )


def test_mmap_marker_roundtrip(tmp_path):
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
        _read_mmap_marker,
        _write_mmap_marker,
    )

    table = tmp_path / "table.bin"
    assert _read_mmap_marker(str(table)) is None
    meta = {
        "num_embeddings": 1024,
        "embedding_dim": 256,
        "dtype": "torch.bfloat16",
        "model": "/models/qwen38",
    }
    _write_mmap_marker(str(table), meta)
    assert _read_mmap_marker(str(table)) == meta
    # corrupted marker json -> None, not a crash
    (tmp_path / "table.bin.meta.json").write_text("{not json")
    assert _read_mmap_marker(str(table)) is None


def test_e4m3fn_decode_lut_values():
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import _e4m3fn_decode_lut

    lut = _e4m3fn_decode_lut()
    assert lut[0x38] == 1.0
    assert lut[0x3C] == 1.5
    assert lut[0x40] == 2.0
    assert lut[0xB8] == -1.0
    assert abs(lut[0x01] - 2.0**-9) < 1e-12  # smallest subnormal
    assert lut[0x7E] == 448.0  # largest finite
    assert lut[0x00] == 0.0 and lut[0x80] == 0.0  # +-0
    assert lut[0x7F] == 0.0 and lut[0xFF] == 0.0  # NaN encodings sanitized


def test_gather_fp8_storage_dequantizes():
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
        _gather_rows_from_table,
    )

    base = torch.randn(32, 8, dtype=torch.bfloat16)
    absmax = base.abs().amax().float().item()
    scale = max(absmax / 448.0, 1.0e-12)
    table = (base.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    ids = torch.tensor([3, 5, 100, -1])  # last two out of range -> zeros
    rows = _gather_rows_from_table(
        table, ids, 0, 32, scale=scale, compute_dtype=torch.bfloat16
    )
    assert rows.dtype == torch.bfloat16
    decoded = table.to(torch.float32) * scale
    assert torch.allclose(rows[0].float(), decoded[3], rtol=0.05)
    assert torch.allclose(rows[1].float(), decoded[5], rtol=0.05)
    assert torch.equal(rows[2], torch.zeros(8, dtype=torch.bfloat16))
    assert torch.equal(rows[3], torch.zeros(8, dtype=torch.bfloat16))


def test_gather_scale_requires_fp8_table():
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import (
        _gather_rows_from_table,
    )

    table = torch.randn(8, 4, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="fp8 table storage"):
        _gather_rows_from_table(table, torch.tensor([1]), 0, 8, scale=0.5)
