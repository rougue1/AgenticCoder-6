"""Opt-in end-to-end regression for the exit-126 permission-bit loop.

Reproduces the reported failure: a Worker asked to run a script that lacks the
executable bit gets exit 126 and, before the fix in tools/registry.py, had no
signal that the *reason* differs from every other kind of shell failure — so
it kept retrying trivial variations of the same broken command forever (see
the activity log in the task this fixes).

This test drives the REAL small worker model (``ornith:latest`` — never
``ornith:35b``, which is far too large/slow to load just for a test) through
one real conversation turn and checks that, once it has seen our exit-126
hint, its very next tool call is no longer the identical bare invocation.

The scenario deliberately used here is a plain shell script *outside*
node_modules/.bin (not `node_modules/.bin/tsc` itself), because that path is
now proactively repaired/rewritten before it ever reaches the model (see
test_sandbox.py) — this test instead covers the residual, non-Node case the
hint in tools/registry.py._render_run exists for: any script a model wrote
itself and forgot to chmod.

Opt in with ``AIFORGE_E2E=1`` and a running ``ollama serve`` with
``ornith:latest`` pulled.
"""

from __future__ import annotations

import os

import pytest

from config import WORKER, load_config
from context.manifest import Manifest
from llm.client import LLMClient
from llm.resolution import resolve_tier
from llm.tool_parser import ToolCall, extract_all_tool_calls
from tools.process_manager import ProcessManager
from tools.registry import TOOL_INSTRUCTIONS, ToolRegistry
from tools.sandbox import Sandbox

pytestmark = pytest.mark.e2e

WORKER_MODEL = "ornith:latest"  # the small (9b) worker model — never ornith:35b


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


@pytest.mark.skipif(not os.environ.get("AIFORGE_E2E"), reason="set AIFORGE_E2E=1 to run against a real Ollama")
def test_worker_self_corrects_after_a_real_126_with_the_new_hint(workspace, bus):
    if not _ollama_up():
        pytest.skip("ollama not reachable on :11434")
    if not _model_pulled(WORKER_MODEL):
        pytest.skip(f"{WORKER_MODEL} not pulled")

    # A real script that is missing its execute bit — the generic (non-Node)
    # case the tools/registry.py 126 hint exists for.
    script = workspace.root / "build.sh"
    script.write_text("#!/bin/sh\necho build-ok\n")
    script.chmod(0o644)

    sandbox = Sandbox(workspace, load_config().sandbox, bus)
    registry = ToolRegistry(workspace, sandbox, Manifest(workspace, bus), bus, ProcessManager(workspace, sandbox, bus))

    first_call = ToolCall(name="run", args={"cmd": "./build.sh"})
    first_result = registry.dispatch(first_call, "test")
    assert first_result.payload.get("exit_code") == 126
    assert "[hint]" in first_result.display  # sanity: the fix under test fired

    messages = [
        {"role": "system", "content": TOOL_INSTRUCTIONS},
        {"role": "user", "content": "Run ./build.sh to build the project."},
        {"role": "assistant", "content": "<agentic_call>{\"tool\": \"run\", \"args\": {\"cmd\": \"./build.sh\"}}</agentic_call>"},
        {"role": "user", "content": first_result.display},
    ]

    cfg = load_config()
    cfg.dump_llm_calls = False
    client = LLMClient(cfg, bus)
    try:
        rmc = resolve_tier(cfg, cfg.tier(WORKER), bus, force_no_thinking=True)
        client.set_runtime({WORKER: rmc})
        result = client.complete(WORKER, "test", messages)

        calls = extract_all_tool_calls(result.text)
        assert calls, f"worker reply had no parseable tool call: {result.text!r}"
        next_cmd = (calls[0].args or {}).get("cmd", "")
        assert next_cmd.strip() != "./build.sh", (
            "worker repeated the identical failing command instead of self-correcting off the "
            f"126 hint: {result.text!r}"
        )
    finally:
        client.unload_all()
