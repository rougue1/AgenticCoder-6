"""Model-memory policy: never keep two DIFFERENT models resident (the OOM/EOF fix).

These tests stub Ollama's HTTP API and the streaming call, so NO real model is
loaded — they verify the eviction *decisions* the client makes.
"""

import types

from config import load_config
from llm.client import CompletionResult, LLMClient


class _FakePS:
    status_code = 200

    def json(self):  # both candidate models reported as loaded so unload proceeds
        return {"models": [{"name": "big-a:1"}, {"name": "big-b:1"}]}


def _patch_http(monkeypatch, unloaded: list):
    monkeypatch.setattr("llm.client.httpx.get", lambda url, **k: _FakePS())

    def _post(url, json=None, **k):
        unloaded.append(json["model"])
        return types.SimpleNamespace(status_code=200)

    monkeypatch.setattr("llm.client.httpx.post", _post)


def _client(bus):
    cfg = load_config()
    cfg.models = {"planner": "ollama/big-a:1", "implementer": "ollama/big-b:1"}
    c = LLMClient(cfg, bus)
    c._stream = lambda *a, **k: CompletionResult(text="ok", raw="ok")  # no litellm
    return c, cfg


def test_evicts_only_on_model_switch(monkeypatch, bus):
    unloaded: list = []
    _patch_http(monkeypatch, unloaded)
    c, _ = _client(bus)

    c.complete("planner", [{"role": "user", "content": "hi"}])  # first load, nothing to evict
    c.complete("planner", [{"role": "user", "content": "hi"}])  # SAME model -> stays warm
    assert unloaded == []

    c.complete("implementer", [{"role": "user", "content": "hi"}])  # switch -> evict the planner model
    assert unloaded == ["big-a:1"]

    c.unload_all()  # end of run -> free the last model
    assert unloaded == ["big-a:1", "big-b:1"]


def test_no_eviction_when_disabled(monkeypatch, bus):
    unloaded: list = []
    _patch_http(monkeypatch, unloaded)
    c, cfg = _client(bus)
    cfg.evict_on_model_switch = False

    c.complete("planner", [{"role": "user", "content": "hi"}])
    c.complete("implementer", [{"role": "user", "content": "hi"}])
    c.unload_all()
    assert unloaded == []  # big-box mode: model stacking allowed, nothing force-evicted
