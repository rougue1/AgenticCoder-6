"""End-to-end pipeline test with SMALL models only (opt-in).

Exercises the whole two-tier machine — model resolution -> pre-flight -> stack
determination + anchor -> environment -> requirements -> architecture -> task
planning -> subtask loop -> final review — including model eviction across the
manager/worker switches, but using only small, fast models so it never
approaches the 16GB-VRAM ceiling. The large ``ornith:35b`` manager model is
intentionally NOT used here.

Opt in with ``AIFORGE_E2E=1`` and a running ``ollama serve`` that has the small
models below pulled. It runs the orchestrator synchronously (no server thread).
"""

import os

import pytest

from config import load_config
from orchestrator.orchestrator import Orchestrator
from orchestrator.states import PipelineState

pytestmark = pytest.mark.e2e

# Small, fast models (all < ~5GB) for both tiers — swapping between them still
# exercises the one-model-resident eviction policy without approaching 16GB.
SMALL_MODELS = {
    "manager": "qwen2.5:7b",
    "worker": "qwen2.5-coder:7b",
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
def test_pipeline_end_to_end_small(tmp_path, bus):
    if not _ollama_up():
        pytest.skip("ollama not reachable on :11434")

    cfg = load_config(project_dir_override=str(tmp_path / "e2e"))
    cfg.tiers["manager"].model = SMALL_MODELS["manager"]
    cfg.tiers["manager"].use_thinking = False  # keep the small model fast/deterministic
    cfg.tiers["worker"].model = SMALL_MODELS["worker"]
    cfg.ollama.max_num_ctx = 8192
    cfg.dump_llm_calls = False
    cfg.pipeline.max_fix_retries = 1
    cfg.pipeline.max_escalations = 1

    orch = Orchestrator(cfg, bus)  # `bus` is the CaptureBus fixture: no loop bound, records events
    orch.run(PROMPT)  # synchronous; whole pipeline runs in this thread

    ws = orch.workspace
    assert ws is not None, "workspace was never attached"
    # The upstream doc suite + task plan must have been produced.
    assert ws.agent_doc_exists("requirements.md")
    assert ws.agent_doc_exists("tasks.json")
    # The pipeline must terminate cleanly (DONE even if a subtask blocked).
    errs = [e.data.get("message") for e in bus.of_type("error")]
    assert orch.state == PipelineState.DONE, f"ended in {orch.state} (errors: {errs})"
    # And the worker must have written at least one real project file.
    files = [p for p in ws.root.rglob("*") if p.is_file() and ".agent" not in p.parts]
    assert files, "no project source files were written"
