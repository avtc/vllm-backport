def test_qsa_fp8_cache_guard_accepts_uint8_storage(monkeypatch):
    """--kv-cache-dtype fp8 resolves to uint8 storage; the guard must
    accept it with the read-path env set and reject it without."""
    import torch

    import vllm.envs as envs
    from vllm.models.qwen4_exp.nvidia.qsa import _qsa_fp8_cache_allowed

    monkeypatch.setattr(envs, "VLLM_QSA_FP8_KV", "decode", raising=False)
    assert _qsa_fp8_cache_allowed("fp8", torch.uint8)
    assert _qsa_fp8_cache_allowed("fp8_e4m3", torch.float8_e4m3fn)
    monkeypatch.setattr(envs, "VLLM_QSA_FP8_KV", "", raising=False)
    assert not _qsa_fp8_cache_allowed("fp8", torch.uint8)
    assert not _qsa_fp8_cache_allowed("fp8_e5m2", torch.uint8)
    assert _qsa_fp8_cache_allowed("auto", torch.bfloat16)


def test_qsa_impl_construction_masks_fp8_from_parent(monkeypatch):
    """The impl dtype handed to the FA parent is 'auto' under fp8+mode so
    the FA device-capability check cannot reject it; unchanged otherwise."""

    import vllm.envs as envs
    from vllm.models.qwen4_exp.nvidia.qsa import _qsa_impl_cache_dtype

    monkeypatch.setattr(envs, "VLLM_QSA_FP8_KV", "decode", raising=False)
    assert _qsa_impl_cache_dtype("fp8") == "auto"
    assert _qsa_impl_cache_dtype("fp8_e4m3") == "auto"
    monkeypatch.setattr(envs, "VLLM_QSA_FP8_KV", "", raising=False)
    assert _qsa_impl_cache_dtype("fp8") == "fp8"
    assert _qsa_impl_cache_dtype("auto") == "auto"
