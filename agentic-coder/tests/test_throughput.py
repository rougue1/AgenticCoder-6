"""tok/s throughput is computed and emitted on llm_complete."""

from config import load_config
from llm.client import CompletionResult, LLMClient


def test_llm_complete_carries_tokens_per_second(bus):
    cfg = load_config()
    cfg.evict_on_model_switch = False  # keep this a pure unit test (no Ollama HTTP)
    c = LLMClient(cfg, bus)
    c._stream = lambda *a, **k: CompletionResult(text="hello world", raw="hello world " * 20)

    c.complete("intake", [{"role": "user", "content": "hi"}])

    done = bus.of_type("llm_complete")
    assert done, "no llm_complete event emitted"
    tps = done[-1].data.get("tokens_per_second")
    assert isinstance(tps, (int, float)) and tps >= 0
