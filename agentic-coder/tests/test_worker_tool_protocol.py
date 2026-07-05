"""Worker tool-call tag protocol — regression coverage for the Ollama-native-
parser collision (see the WHY block in llm/tool_parser.py and tools/registry.py).

``<tool_call>...</tool_call>`` is the reserved native tool-calling delimiter
for the qwen model family (ornith is a qwen3.5 build). Ollama's own
renderer/parser recognizes that literal tag SERVER-SIDE and tries to lift it
into ``message.tool_calls`` using its own name/arguments schema before our
code ever sees the text. Our schema uses tool/args, so the native parser fails
mid-parse ("qwen tool call parsing failed error=EOF") and the whole
``/api/chat`` response comes back as ``{"error": "EOF"}``, which crashes
litellm with a raw ``KeyError: 'message'``. The fix: the Worker is prompted to
use a non-native ``<agentic_call>`` tag instead, so Ollama never recognizes
anything tool-call-shaped in the raw output and passes it through untouched.

The first two tests are fast regression guards (no LLM, no network) that fail
loudly if anyone reintroduces the collision by reverting the prompt text. The
last test is an opt-in integration test against a REAL running Ollama +
ornith:latest (the small worker model — never ornith:35b, which is too slow to
load for a test) that exercises the actual failure path end-to-end.
"""

from __future__ import annotations

import os

import pytest

from config import WORKER, load_config
from llm.client import LLMClient
from llm.resolution import resolve_tier
from llm.tool_parser import extract_tool_call
from stages.worker import _PROTOCOL_CORRECTION
from tools.registry import TOOL_INSTRUCTIONS


def test_worker_prompt_never_instructs_the_native_tag():
    """The prompted protocol must use <agentic_call>, never bare <tool_call>."""
    for label, text in (("TOOL_INSTRUCTIONS", TOOL_INSTRUCTIONS), ("_PROTOCOL_CORRECTION", _PROTOCOL_CORRECTION)):
        assert "<agentic_call>" in text, f"{label} must instruct the non-native <agentic_call> tag"
        assert "<tool_call>" not in text, (
            f"{label} must never tell the model to emit <tool_call> — that is the reserved "
            "native qwen tag Ollama's own parser intercepts server-side (see module docstrings)"
        )


def test_worker_prompt_examples_round_trip_through_the_parser():
    """Every worked example the Worker is shown must itself be parseable by our
    parser (skips the one illustrative "wrapped in <agentic_call>...</agentic_call>"
    mention, which has no JSON body)."""
    import re

    examples = [m for m in re.findall(r"<agentic_call>.*?</agentic_call>", TOOL_INSTRUCTIONS, re.DOTALL) if "{" in m]
    assert len(examples) >= 4  # read_file, write_file, patch_file, run (x2)
    for example in examples:
        call = extract_tool_call(example)
        assert call is not None and call.is_known, f"unparseable example in TOOL_INSTRUCTIONS: {example!r}"


# ── real-model integration test ─────────────────────────────────────────────
WORKER_MODEL = "ornith:latest"  # the small (9b) worker model — ornith:35b is too slow for a test


def _ollama_up() -> bool:
    import httpx

    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


def _model_pulled(model: str) -> bool:
    import httpx

    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
        names = {m.get("name") for m in resp.json().get("models", [])}
        return model in names or f"{model.split(':')[0]}:latest" in names
    except Exception:
        return False


@pytest.mark.e2e
@pytest.mark.skipif(not os.environ.get("AIFORGE_E2E"), reason="set AIFORGE_E2E=1 to run against a real Ollama")
def test_worker_tool_call_survives_a_real_ornith_call(bus):
    """Reproduces the reported failure path end-to-end against the real worker
    model: a call that asks ornith:latest to emit a tool call must come back
    as normal text (no LLMError / no litellm KeyError), and that text must
    parse cleanly via extract_tool_call — proving the <agentic_call> tag does
    not collide with Ollama's native qwen tool-call parser the way <tool_call>
    did.
    """
    if not _ollama_up():
        pytest.skip("ollama not reachable on :11434")
    if not _model_pulled(WORKER_MODEL):
        pytest.skip(f"{WORKER_MODEL} not pulled")

    cfg = load_config()
    cfg.dump_llm_calls = False
    client = LLMClient(cfg, bus)
    try:
        rmc = resolve_tier(cfg, cfg.tier(WORKER), bus, force_no_thinking=True)
        client.set_runtime({WORKER: rmc})

        messages = [
            {"role": "system", "content": TOOL_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    "Call the read_file tool to read the file at path 'app.py'. "
                    "Emit ONLY the tool call, nothing else."
                ),
            },
        ]
        result = client.complete(WORKER, "test", messages)

        assert result.text.strip(), "worker returned no text"
        call = extract_tool_call(result.text)
        assert call is not None, f"worker reply did not contain a parseable tool call: {result.text!r}"
        assert call.name == "read_file"
        assert call.args.get("path") == "app.py"
    finally:
        client.unload_all()
