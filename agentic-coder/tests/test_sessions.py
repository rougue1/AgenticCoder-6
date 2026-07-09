"""Background sessions: start/check/stop lifecycle, parallel sessions, the
run/check_session/stop_session Worker tools, and the subtask-boundary cleanup
guarantee (no session survives the subtask that started it).

Everything here needs a FUNCTIONAL bwrap (see test_sandbox.bwrap_functional);
tests skip with the AppArmor pointer otherwise.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest

from config import SandboxCfg, load_config
from context.manifest import Manifest
from llm.client import LLMClient
from llm.tool_parser import ToolCall
from orchestrator.states import SubtaskOutcome
from orchestrator.subtask_loop import SubtaskLoop, VerifyResult
from services import Services
from taskstore import TaskStore
from tools.registry import ToolRegistry
from tools.sandbox import Sandbox

from test_sandbox import require_bwrap


@pytest.fixture
def sandbox(workspace, bus):
    # Short grace window: the semantics are identical, the suite just doesn't
    # sit out the full 2s default for every long-lived test session.
    sb = Sandbox(workspace, SandboxCfg(background_grace=0.4), bus)
    yield sb
    sb.terminate_sessions("test teardown")


def _wait_exit(sandbox: Sandbox, session_id: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and sandbox.check_session(session_id).running:
        time.sleep(0.05)


# ── session lifecycle ───────────────────────────────────────────────────────────
def test_background_start_returns_uuid_and_keeps_running(sandbox, workspace):
    require_bwrap()
    start = sandbox.run_background("echo booted-marker; sleep 30")
    assert start.ok and start.running
    uuid.UUID(start.session_id)  # session ids are real UUIDs
    assert "booted-marker" in start.output  # grace window captured early output
    log = workspace.agent_dir / "sessions" / f"{start.session_id}.log"
    assert log.is_file() and "booted-marker" in log.read_text()


def test_background_instant_crash_reports_exit_and_output(sandbox):
    require_bwrap()
    start = sandbox.run_background("echo port-in-use >&2; exit 7")
    assert start.session_id and not start.running and not start.ok
    assert start.exit_code == 7
    assert "port-in-use" in start.output  # the Worker can self-correct from this


def test_check_session_running_then_after_stop(sandbox):
    require_bwrap()
    start = sandbox.run_background("echo alive; sleep 30")
    assert start.running

    status = sandbox.check_session(start.session_id)
    assert status.exists and status.running and status.exit_code is None
    assert "alive" in status.output and status.uptime >= 0

    stopped = sandbox.stop_session(start.session_id)
    assert stopped.exists and not stopped.running

    again = sandbox.check_session(start.session_id)  # record kept until subtask end
    assert again.exists and not again.running


def test_check_session_returns_crash_output_after_death(workspace, bus):
    """A session that was RUNNING at start-time but crashed later: the crash
    exit code and output must come back through check_session."""
    require_bwrap()
    sb = Sandbox(workspace, SandboxCfg(background_grace=0.2), bus)
    try:
        start = sb.run_background("sleep 1.2; echo crash-trace >&2; exit 9")
        assert start.running  # outlived the 0.2s grace window
        _wait_exit(sb, start.session_id)
        status = sb.check_session(start.session_id)
        assert status.exists and not status.running
        assert status.exit_code == 9
        assert "crash-trace" in status.output
    finally:
        sb.terminate_sessions("test teardown")


def test_unknown_session_id_is_reported_helpfully(sandbox):
    require_bwrap()
    status = sandbox.check_session("no-such-session")
    assert not status.exists
    assert "unknown session" in status.detail
    assert not sandbox.stop_session("no-such-session").exists


def test_parallel_sessions_run_simultaneously_with_own_logs(sandbox):
    require_bwrap()
    a = sandbox.run_background("echo svc-A; sleep 30")
    b = sandbox.run_background("echo svc-B; sleep 30")
    assert a.running and b.running and a.session_id != b.session_id
    sa, sb_ = sandbox.check_session(a.session_id), sandbox.check_session(b.session_id)
    assert sa.running and sb_.running
    assert "svc-A" in sa.output and "svc-B" not in sa.output
    assert "svc-B" in sb_.output and "svc-A" not in sb_.output
    assert set(sandbox.active_sessions) == {a.session_id, b.session_id}


def test_terminate_sessions_kills_everything_and_forgets_ids(sandbox):
    require_bwrap()
    a = sandbox.run_background("sleep 30")
    b = sandbox.run_background("sleep 30")
    procs = [sandbox._get_session(s).proc for s in (a.session_id, b.session_id)]

    killed = sandbox.terminate_sessions("test")
    assert killed == 2
    for proc in procs:
        assert proc.poll() is not None  # actually dead, not just forgotten
    assert sandbox.active_sessions == []
    assert not sandbox.check_session(a.session_id).exists  # ids don't outlive cleanup


def test_background_server_is_reachable_from_foreground_commands(sandbox):
    """The whole point of sessions: start a server in background, hit it with
    a foreground command (shared network namespace)."""
    require_bwrap()
    port = 8931
    start = sandbox.run_background(f"python3 -m http.server {port} --bind 127.0.0.1")
    assert start.running, f"http.server did not stay up: {start.output}"
    try:
        deadline = time.monotonic() + 10
        res = None
        while time.monotonic() < deadline:
            res = sandbox.run(f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/")
            if res.ok and "200" in res.stdout:
                break
            time.sleep(0.3)
        assert res is not None and res.ok and "200" in res.stdout, (
            f"could not reach background server: exit={getattr(res, 'exit_code', '?')} "
            f"stdout={getattr(res, 'stdout', '')!r} stderr={getattr(res, 'stderr', '')!r}"
        )
        # And its request log is visible via check_session.
        status = sandbox.check_session(start.session_id)
        assert status.running
    finally:
        sandbox.stop_session(start.session_id)


# ── worker tool surface (registry-level) ────────────────────────────────────────
def _registry(workspace, bus):
    sb = Sandbox(workspace, SandboxCfg(), bus)
    return ToolRegistry(workspace, sb, Manifest(workspace, bus), bus), sb


def test_run_tool_rejects_foreground_server_with_guidance(workspace, bus):
    registry, _ = _registry(workspace, bus)  # rejection happens before bwrap runs
    result = registry.dispatch(ToolCall(name="run", args={"cmd": "uvicorn app:app"}), "test")
    assert not result.ok
    assert result.payload.get("rejected") is True
    assert "long-running process" in result.display
    assert "background" in result.display


def test_run_tool_background_then_check_then_stop(workspace, bus):
    require_bwrap()
    registry, sb = _registry(workspace, bus)
    try:
        started = registry.dispatch(
            ToolCall(name="run", args={"cmd": "echo tool-session; sleep 30", "background": True}), "test"
        )
        assert started.ok and started.payload["background"] is True
        sid = started.payload["session_id"]
        assert sid and started.payload["running"] is True
        assert "check_session" in started.display

        checked = registry.dispatch(ToolCall(name="check_session", args={"session_id": sid}), "test")
        assert checked.ok and checked.payload["running"] is True
        assert "tool-session" in checked.payload["output"]

        stopped = registry.dispatch(ToolCall(name="stop_session", args={"session_id": sid}), "test")
        assert stopped.ok and stopped.payload["running"] is False
    finally:
        sb.terminate_sessions("test teardown")


def test_run_tool_background_crash_surfaces_error_output(workspace, bus):
    require_bwrap()
    registry, sb = _registry(workspace, bus)
    try:
        result = registry.dispatch(
            ToolCall(name="run", args={"cmd": "echo EADDRINUSE >&2; exit 1", "background": True}), "test"
        )
        assert not result.ok
        assert result.payload["exit_code"] == 1
        assert "EXITED IMMEDIATELY" in result.display
        assert "EADDRINUSE" in result.display
    finally:
        sb.terminate_sessions("test teardown")


def test_run_tool_ignores_legacy_smoke_arg_with_note(workspace, bus):
    require_bwrap()
    registry, sb = _registry(workspace, bus)
    try:
        result = registry.dispatch(
            ToolCall(
                name="run",
                args={"cmd": "sleep 30", "background": True, "smoke": ["curl http://x"]},
            ),
            "test",
        )
        assert result.ok
        assert "no longer a run argument" in result.display
    finally:
        sb.terminate_sessions("test teardown")


def test_check_session_tool_unknown_id_is_an_error(workspace, bus):
    registry, _ = _registry(workspace, bus)
    result = registry.dispatch(ToolCall(name="check_session", args={"session_id": "bogus"}), "test")
    assert not result.ok and "unknown session" in result.display


# ── subtask-boundary cleanup (the lifecycle guarantee) ─────────────────────────
def test_subtask_loop_terminates_sessions_when_subtask_completes(workspace, bus, monkeypatch):
    require_bwrap()
    cfg = load_config()
    services = Services(config=cfg, bus=bus, client=LLMClient(cfg, bus), cancel_event=threading.Event())
    services.workspace = workspace
    services.sandbox = Sandbox(workspace, cfg.sandbox, bus)
    loop = SubtaskLoop(services)

    holder: dict = {}

    class FakeWorkerSession:
        """Stands in for the Worker: starts a dev-server-style session and
        leaves it running, exactly like a model that forgot stop_session."""

        def __init__(self, services_, sub, instructions):
            self.files_touched: set[str] = set()
            self.attempt = 0
            self.last_call_ok = True  # Feature 5 (completion gate) reads this

        def implement(self):
            start = services.sandbox.run_background("sleep 30")
            assert start.running
            holder["sid"] = start.session_id
            holder["proc"] = services.sandbox._get_session(start.session_id).proc
            return True

        def fix(self, **kwargs):
            return True

    import orchestrator.subtask_loop as sl
    from orchestrator.completion_gate import GateResult

    monkeypatch.setattr(sl, "WorkerSession", FakeWorkerSession)
    monkeypatch.setattr(sl.manager, "handoff", lambda *a, **k: "fake instructions")
    monkeypatch.setattr(sl.summarizer, "summarize_files", lambda *a, **k: 0)
    monkeypatch.setattr(loop, "_verify", lambda sub: VerifyResult("echo ok", True, "", "", 0))
    # This test is about session cleanup at the subtask boundary, not about
    # Feature 3/5 themselves (both covered in tests/test_subtask_loop_features.py);
    # the subtask's declared file ("a.txt") is never actually written by the
    # FakeWorkerSession above, so the real completion gate is stubbed to pass.
    monkeypatch.setattr(sl.completion_gate, "check", lambda *a, **k: GateResult(passed=True))

    store = TaskStore.from_data(
        workspace,
        {
            "project": "t",
            "tasks": [
                {
                    "id": "T1",
                    "title": "task",
                    "subtasks": [
                        {
                            "id": "T1.1",
                            "title": "sub",
                            "type": "config",
                            "files": ["a.txt"],
                            "dependencies": [],
                            "test_command": "",
                        }
                    ],
                }
            ],
        },
    )
    task = store.tasks[0]
    sub = task["subtasks"][0]

    outcome = loop._process_subtask(store, task, sub)

    assert outcome == SubtaskOutcome.DONE
    assert holder["proc"].poll() is not None, "background session survived the subtask"
    assert services.sandbox.active_sessions == []
    assert not services.sandbox.check_session(holder["sid"]).exists
