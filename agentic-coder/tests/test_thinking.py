"""Model capability resolution: /api/show + the live thinking probe, its two
gates, and the worker's force-no-thinking architectural override
(llm/resolution.py). The redesign replaced model-name pattern matching
(THINK_* constants, detect_thinking_mode) with a live Ollama capability probe."""

from __future__ import annotations

import pytest

from config import ModelTierCfg, load_config
from llm.resolution import ModelResolutionError, resolve_tier


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or ""

    def json(self):
        return self._payload


def _tier_cfg(*, use_thinking: bool, model: str = "some-model:7b") -> ModelTierCfg:
    return ModelTierCfg(
        name="test", model=model, temperature=0.2, context_window_pct=0.6,
        max_tokens_override=None, use_thinking=use_thinking,
    )


def _cfg():
    c = load_config()
    c.ollama.max_num_ctx = 32768
    return c


def test_gate_one_use_thinking_false_never_probes(monkeypatch, bus):
    def _post(url, json=None, timeout=None):
        if url.endswith("/api/show"):
            return _Resp(200, {"model_info": {"family.context_length": 32768}})
        raise AssertionError(f"probe should not run when use_thinking=False, got POST {url}")

    monkeypatch.setattr("llm.resolution.httpx.post", _post)
    rmc = resolve_tier(_cfg(), _tier_cfg(use_thinking=False), bus)
    assert rmc.use_thinking is False
    assert rmc.thinking_enabled is False


def test_gate_two_probe_success_enables_thinking(monkeypatch, bus):
    calls = {"show": 0, "chat": 0}

    def _post(url, json=None, timeout=None):
        if url.endswith("/api/show"):
            calls["show"] += 1
            return _Resp(200, {"model_info": {"family.context_length": 32768}, "capabilities": ["thinking"]})
        if url.endswith("/api/chat"):
            calls["chat"] += 1
            return _Resp(200, {"message": {"role": "assistant", "content": "hello"}})
        raise AssertionError(url)

    monkeypatch.setattr("llm.resolution.httpx.post", _post)
    rmc = resolve_tier(_cfg(), _tier_cfg(use_thinking=True), bus)
    assert rmc.thinking_enabled is True
    assert rmc.supports_think_param is True
    assert calls["chat"] == 1


def test_probe_explicit_unsupported_error_disables_with_warning(monkeypatch, bus):
    def _post(url, json=None, timeout=None):
        if url.endswith("/api/show"):
            return _Resp(200, {"model_info": {"family.context_length": 32768}})
        return _Resp(400, {"error": "this model does not support thinking"})

    monkeypatch.setattr("llm.resolution.httpx.post", _post)
    rmc = resolve_tier(_cfg(), _tier_cfg(use_thinking=True), bus)
    assert rmc.thinking_enabled is False
    warnings = [e for e in bus.of_type("log") if e.data.get("level") == "warn"]
    assert warnings, "expected a warning log when thinking is unsupported"


def test_probe_unexpected_failure_is_a_hard_error(monkeypatch, bus):
    def _post(url, json=None, timeout=None):
        if url.endswith("/api/show"):
            return _Resp(200, {"model_info": {"family.context_length": 32768}})
        return _Resp(500, {"error": "internal server error"})

    monkeypatch.setattr("llm.resolution.httpx.post", _post)
    with pytest.raises(ModelResolutionError):
        resolve_tier(_cfg(), _tier_cfg(use_thinking=True), bus)


def test_worker_force_no_thinking_overrides_intent_and_skips_probe(monkeypatch, bus):
    calls = {"chat": 0}

    def _post(url, json=None, timeout=None):
        if url.endswith("/api/show"):
            return _Resp(200, {"model_info": {"family.context_length": 32768}})
        calls["chat"] += 1
        return _Resp(200, {"message": {}})

    monkeypatch.setattr("llm.resolution.httpx.post", _post)
    rmc = resolve_tier(_cfg(), _tier_cfg(use_thinking=True), bus, force_no_thinking=True)
    assert rmc.thinking_enabled is False
    assert calls["chat"] == 0  # the worker override skips the probe entirely
    warnings = [e for e in bus.of_type("log") if e.data.get("level") == "warn"]
    assert warnings, "expected a warning that use_thinking=true was ignored for the worker"


def test_num_ctx_clamped_to_hardware_ceiling(monkeypatch, bus):
    def _post(url, json=None, timeout=None):
        return _Resp(200, {"model_info": {"family.context_length": 262144}})

    monkeypatch.setattr("llm.resolution.httpx.post", _post)
    cfg = _cfg()
    cfg.ollama.max_num_ctx = 32768
    rmc = resolve_tier(cfg, _tier_cfg(use_thinking=False), bus)
    assert rmc.reported_num_ctx == 262144
    assert rmc.num_ctx == 32768  # clamped to the hardware ceiling
