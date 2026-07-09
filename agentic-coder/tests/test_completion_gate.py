"""Feature 5 — the completion gate: declared-file existence/non-emptiness,
conservative placeholder-marker detection (exact strings only — a generic
"# TODO: review later" must never flag), the last-tool-call-ok check, and the
lingering-background-session check with its exact required message."""

from __future__ import annotations

from orchestrator import completion_gate


class _FakeSandboxNoSessions:
    active_sessions: list[str] = []


class _FakeSandboxWithSessions:
    def __init__(self, ids):
        self.active_sessions = ids


class _FakeServices:
    def __init__(self, workspace, sandbox=None):
        self.workspace = workspace
        self.sandbox = sandbox if sandbox is not None else _FakeSandboxNoSessions()


class _FakeSession:
    def __init__(self, files_touched, last_call_ok=True):
        self.files_touched = set(files_touched)
        self.last_call_ok = last_call_ok


def _sub(files=None):
    return {"id": "T1.1", "title": "s", "type": "implement", "files": files or []}


# ── condition 2: declared files exist and are non-empty ─────────────────────
def test_missing_declared_file_names_it(workspace):
    services = _FakeServices(workspace)
    result = completion_gate.check(services, _sub(files=["missing.py"]), _FakeSession([]))
    assert not result.passed
    assert any("missing.py" in c for c in result.failed_conditions)


def test_empty_declared_file_is_flagged(workspace):
    workspace.write_file("empty.py", "")
    services = _FakeServices(workspace)
    result = completion_gate.check(services, _sub(files=["empty.py"]), _FakeSession([]))
    assert not result.passed
    assert any("empty.py" in c for c in result.failed_conditions)


def test_present_non_empty_declared_files_pass(workspace):
    workspace.write_file("real.py", "x = 1\n")
    services = _FakeServices(workspace)
    result = completion_gate.check(services, _sub(files=["real.py"]), _FakeSession([]))
    assert result.passed and result.failed_conditions == []


# ── condition 3: placeholder markers ─────────────────────────────────────────
def test_raise_not_implemented_error_is_flagged(workspace):
    workspace.write_file("app.py", "def handler():\n    raise NotImplementedError\n")
    services = _FakeServices(workspace)
    result = completion_gate.check(services, _sub(), _FakeSession(["app.py"]))
    assert not result.passed
    assert any("app.py" in c for c in result.failed_conditions)


def test_js_not_implemented_throw_is_flagged(workspace):
    workspace.write_file(
        "handler.js", "function handler() {\n  throw new Error('not implemented');\n}\n"
    )
    services = _FakeServices(workspace)
    result = completion_gate.check(services, _sub(), _FakeSession(["handler.js"]))
    assert not result.passed


def test_pass_placeholder_comment_is_flagged(workspace):
    workspace.write_file("stub.py", "def handler():\n    pass  # placeholder\n")
    services = _FakeServices(workspace)
    result = completion_gate.check(services, _sub(), _FakeSession(["stub.py"]))
    assert not result.passed


def test_todo_implement_is_flagged(workspace):
    workspace.write_file("stub.py", "# TODO: implement this function\ndef f(): pass\n")
    services = _FakeServices(workspace)
    result = completion_gate.check(services, _sub(), _FakeSession(["stub.py"]))
    assert not result.passed


def test_generic_todo_review_later_is_not_flagged(workspace):
    """Conservative matching: a note like this is NOT incomplete-code logic."""
    workspace.write_file("app.py", "def f():\n    return 1  # TODO: review later\n")
    services = _FakeServices(workspace)
    result = completion_gate.check(services, _sub(), _FakeSession(["app.py"]))
    assert result.passed


def test_generic_fixme_needs_cleanup_is_not_flagged(workspace):
    workspace.write_file("app.py", "def f():\n    return 1  # FIXME: needs cleanup\n")
    services = _FakeServices(workspace)
    result = completion_gate.check(services, _sub(), _FakeSession(["app.py"]))
    assert result.passed


def test_complete_implementation_is_not_flagged(workspace):
    workspace.write_file("app.py", "def add(a, b):\n    return a + b\n")
    services = _FakeServices(workspace)
    result = completion_gate.check(services, _sub(), _FakeSession(["app.py"]))
    assert result.passed


# ── condition 4: last tool call must not be an error ─────────────────────────
def test_last_call_not_ok_is_flagged(workspace):
    services = _FakeServices(workspace)
    result = completion_gate.check(services, _sub(), _FakeSession([], last_call_ok=False))
    assert not result.passed
    assert any("last tool call" in c for c in result.failed_conditions)


# ── condition 5: no lingering background sessions ────────────────────────────
def test_active_background_session_is_flagged_with_exact_message(workspace):
    services = _FakeServices(workspace, sandbox=_FakeSandboxWithSessions(["abc-123"]))
    result = completion_gate.check(services, _sub(), _FakeSession([]))
    assert not result.passed
    assert any(
        "Background session abc-123 is still running. Stop it before marking this subtask "
        "complete." == c
        for c in result.failed_conditions
    )


def test_no_active_sessions_passes(workspace):
    services = _FakeServices(workspace)
    result = completion_gate.check(services, _sub(), _FakeSession([]))
    assert result.passed


# ── all conditions clear together ────────────────────────────────────────────
def test_all_conditions_clear_yields_passed_gate(workspace):
    workspace.write_file("app.py", "def add(a, b):\n    return a + b\n")
    services = _FakeServices(workspace)
    result = completion_gate.check(
        services, _sub(files=["app.py"]), _FakeSession(["app.py"], last_call_ok=True)
    )
    assert result.passed and result.failed_conditions == []
