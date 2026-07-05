"""tok/s throughput is computed and emitted on llm_complete."""

from config import load_config
from llm.client import CompletionResult, LLMClient
from llm.resolution import RuntimeModelConfig


def test_llm_complete_carries_tokens_per_second(bus):
    cfg = load_config()
    cfg.ollama.evict_on_model_switch = False  # keep this a pure unit test (no Ollama HTTP)
    c = LLMClient(cfg, bus)
    c.set_runtime(
        {
            "manager": RuntimeModelConfig(
                tier="manager", model="test-model:7b", reported_num_ctx=8192, num_ctx=8192,
                max_tokens=4096, context_window_pct=0.5, temperature=0.2,
                use_thinking=False, thinking_enabled=False, supports_think_param=False,
            )
        }
    )
    c._stream = lambda *a, **k: CompletionResult(text="hello world", raw="hello world " * 20)

    c.complete("manager", "intake", [{"role": "user", "content": "hi"}])

    done = bus.of_type("llm_complete")
    assert done, "no llm_complete event emitted"
    tps = done[-1].data.get("tokens_per_second")
    assert isinstance(tps, (int, float)) and tps >= 0
