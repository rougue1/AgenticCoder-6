"""End-to-end pipeline test with SMALL models only (opt-in).

Exercises the WHOLE machine — intake -> requirements -> stack -> architect -> sdd
-> task planning -> subtask loop -> review — including model eviction across the
phase/model switches, but using only small, fast models so it never approaches the
16GB-VRAM ceiling. The large 27B/30B models are intentionally NOT tested here.

Opt in with ``AIFORGE_E2E=1`` and a running ``ollama serve`` that has the small
models below pulled. It runs the orchestrator synchronously (no server thread).
"""

import os

import pytest

from config import load_config
from orchestrator.orchestrator import Orchestrator
from orchestrator.states import PipelineState
from server.events import EventBus

pytestmark = pytest.mark.e2e

# Small, fast models (all < ~5GB) — swapping between them still exercises eviction.
SMALL_MODELS = {
    "intake": "ollama/llama3.2:3b",
    "requirements": "ollama/qwen2.5:7b",
    "stack_decider": "ollama/qwen2.5:7b",
    "architect": "ollama/qwen2.5:7b",
    "sdd_generator": "ollama/qwen2.5:7b",
    "task_planner": "ollama/qwen2.5:7b",
    "planner": "ollama/qwen2.5:7b",
    "escalation": "ollama/qwen2.5:7b",
    "implementer": "ollama/qwen2.5-coder:7b",
    "reviewer": "ollama/qwen2.5-coder:7b",
    "tool_caller": "ollama/qwen2.5-coder:7b",
}

PROMPT = (
    "Create a Python module calc.py with a function add(a, b) that returns a + b, "
    "and a pytest test in tests/test_calc.py asserting add(2, 3) == 5."
)


def _ollama_up() -> bool:
    import httpx

    try:
        return httpx.get("http://localhost:11434/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not os.environ.get("AIFORGE_E2E"), reason="set AIFORGE_E2E=1 to run the small-model e2e")
def test_pipeline_end_to_end_small(tmp_path):
    if not _ollama_up():
        pytest.skip("ollama not reachable on :11434")

    cfg = load_config(project_dir_override=str(tmp_path / "e2e"))
    cfg.models = dict(SMALL_MODELS)
    cfg.num_ctx = 8192
    cfg.budgets = {k: 8192 for k in SMALL_MODELS}
    cfg.reserve_for_output = 2048
    cfg.dump_llm_calls = False
    cfg.limits.max_fix_retries = 1
    cfg.limits.max_escalations = 1

    bus = EventBus()  # no loop bound -> emit() just logs; fine for a synchronous run
    orch = Orchestrator(cfg, bus)
    orch.run(PROMPT)  # synchronous; whole pipeline runs in this thread

    ws = orch.workspace
    assert ws is not None, "workspace was never attached"
    # The upstream SDD suite + task plan must have been produced.
    assert ws.agent_doc_exists("requirements.md")
    assert ws.agent_doc_exists("tasks.json")
    # The pipeline must terminate cleanly (DONE even if a subtask blocked).
    errs = [e.data.get("message") for e in bus.events if e.type == "error"]
    assert orch.state == PipelineState.DONE, f"ended in {orch.state} (errors: {errs})"
    # And the coder must have written at least one real project file.
    files = [p for p in ws.root.rglob("*") if p.is_file() and ".agent" not in p.parts]
    assert files, "no project source files were written"
