"""Integration coverage for the restructured post-pass branch of
``SubtaskLoop._process_subtask_inner``: findings.md on failure, one code-review
cycle before the completion gate, and a gate/review failure spending the same
fix-retry budget a test failure would (escalating once it's exhausted, never
looping forever). Follows the ``FakeWorkerSession`` monkeypatch pattern
established in ``tests/test_sessions.py``: only ``WorkerSession``,
``manager.handoff/escalate``, ``summarizer.summarize_files``, and
``SubtaskLoop._verify`` are mocked — the real ladder/escalation control flow
in ``orchestrator/subtask_loop.py`` runs unmodified."""

from __future__ import annotations

import threading

import orchestrator.subtask_loop as sl
from config import load_config
from llm.client import LLMClient
from orchestrator.states import SubtaskOutcome
from orchestrator.subtask_loop import SubtaskLoop, VerifyResult
from server import events
from services import Services
from stages import code_review
from orchestrator import completion_gate
from taskstore import TaskStore


def _services(workspace, bus, *, max_fix_retries=None, max_escalations=None):
    cfg = load_config()
    if max_fix_retries is not None:
        cfg.pipeline.max_fix_retries = max_fix_retries
    if max_escalations is not None:
        cfg.pipeline.max_escalations = max_escalations
    services = Services(config=cfg, bus=bus, client=LLMClient(cfg, bus), cancel_event=threading.Event())
    services.workspace = workspace
    return services


def _store_with_one_subtask(workspace):
    data = {
        "project": "p",
        "tasks": [
            {
                "id": "T1",
                "title": "task",
                "subtasks": [
                    {
                        "id": "T1.1",
                        "title": "sub",
                        "type": "implement",
                        "role": "backend",
                        "files": ["app.py"],
                        "dependencies": [],
                        "test_command": "pytest",
                    }
                ],
            }
        ],
    }
    store = TaskStore.from_data(workspace, data)
    task = store.tasks[0]
    sub = task["subtasks"][0]
    return store, task, sub


class _FakeWorkerSession:
    def __init__(self, last_call_ok: bool = True):
        self.files_touched = {"app.py"}
        self.attempt = 0
        self.last_call_ok = last_call_ok
        self.fix_calls = 0
        self.address_review_calls = 0
        self.address_gate_calls = 0

    def implement(self) -> bool:
        return True

    def fix(self, **kwargs) -> bool:
        self.fix_calls += 1
        return True

    def address_review(self, issues) -> bool:
        self.address_review_calls += 1
        return True

    def address_completion_gate(self, failed_conditions) -> bool:
        self.address_gate_calls += 1
        return True

    def summarize_failure(self) -> str:
        return "the test command failed for a fake reason."


class _FakeWorkerSessionFactory:
    def __init__(self, last_call_ok: bool = True):
        self.sessions: list[_FakeWorkerSession] = []
        self._last_call_ok = last_call_ok

    def __call__(self, services, sub, instructions) -> _FakeWorkerSession:
        session = _FakeWorkerSession(last_call_ok=self._last_call_ok)
        self.sessions.append(session)
        return session


def _repeating(results):
    """A callable returning *results* in order, repeating the last one if
    called more times than provided (defensive — never raises StopIteration)."""
    results = list(results)
    calls = {"n": 0}

    def _next(*_args, **_kwargs):
        idx = min(calls["n"], len(results) - 1)
        calls["n"] += 1
        return results[idx]

    return _next


def _patch_common(monkeypatch, factory=None, *, escalate_reply="re-plan"):
    monkeypatch.setattr(sl, "WorkerSession", factory or _FakeWorkerSessionFactory())
    monkeypatch.setattr(sl.manager, "handoff", lambda *a, **k: "instructions")
    monkeypatch.setattr(sl.manager, "escalate", lambda *a, **k: escalate_reply)
    monkeypatch.setattr(sl.summarizer, "summarize_files", lambda *a, **k: 1)


# ── happy path: review once, gate once, done, no findings ────────────────────
def test_happy_path_reviews_once_gates_once_then_done(workspace, bus, monkeypatch):
    services = _services(workspace, bus)
    factory = _FakeWorkerSessionFactory()
    _patch_common(monkeypatch, factory)
    monkeypatch.setattr(sl.code_review, "run", _repeating([code_review.ReviewResult(approved=True)]))
    monkeypatch.setattr(sl.completion_gate, "check", _repeating([completion_gate.GateResult(passed=True)]))

    loop = SubtaskLoop(services)
    monkeypatch.setattr(loop, "_verify", _repeating([VerifyResult("pytest", True, "", "", 0)]))

    store, task, sub = _store_with_one_subtask(workspace)
    outcome = loop._process_subtask(store, task, sub)

    assert outcome == SubtaskOutcome.DONE
    assert len(bus.of_type(events.TASK_REVIEW_START)) == 1
    review_complete = bus.of_type(events.TASK_REVIEW_COMPLETE)
    assert len(review_complete) == 1 and review_complete[0].data["status"] == "approved"
    gate_checks = bus.of_type(events.TASK_COMPLETION_CHECK)
    assert len(gate_checks) == 1 and gate_checks[0].data["status"] == "passed"
    assert not workspace.agent_doc_exists("findings.md")
    assert factory.sessions[0].fix_calls == 0


# ── review issues -> one fix cycle -> approved -> gate -> done ───────────────
def test_review_issues_trigger_one_address_review_cycle_then_done(workspace, bus, monkeypatch):
    services = _services(workspace, bus)
    factory = _FakeWorkerSessionFactory()
    _patch_common(monkeypatch, factory)
    monkeypatch.setattr(
        sl.code_review,
        "run",
        _repeating([code_review.ReviewResult(approved=False, issues=["fix the bug"])]),
    )
    monkeypatch.setattr(sl.completion_gate, "check", _repeating([completion_gate.GateResult(passed=True)]))

    loop = SubtaskLoop(services)
    monkeypatch.setattr(
        loop,
        "_verify",
        _repeating([VerifyResult("pytest", True, "", "", 0), VerifyResult("pytest", True, "", "", 0)]),
    )

    store, task, sub = _store_with_one_subtask(workspace)
    outcome = loop._process_subtask(store, task, sub)

    assert outcome == SubtaskOutcome.DONE
    # Review only ever RUNS once even though verify() is called twice — the
    # second pass just re-checks the gate.
    assert len(bus.of_type(events.TASK_REVIEW_START)) == 1
    assert bus.of_type(events.TASK_REVIEW_COMPLETE)[0].data["status"] == "issues_found"
    assert factory.sessions[0].address_review_calls == 1
    assert factory.sessions[0].address_gate_calls == 0


# ── gate failure with fix budget remaining: loop back, no second review ─────
def test_gate_failure_with_fix_budget_remaining_loops_back_without_rerunning_review(workspace, bus, monkeypatch):
    services = _services(workspace, bus)
    factory = _FakeWorkerSessionFactory()
    _patch_common(monkeypatch, factory)
    monkeypatch.setattr(sl.code_review, "run", _repeating([code_review.ReviewResult(approved=True)]))
    monkeypatch.setattr(
        sl.completion_gate,
        "check",
        _repeating(
            [
                completion_gate.GateResult(passed=False, failed_conditions=["declared file missing: app.py"]),
                completion_gate.GateResult(passed=True),
            ]
        ),
    )

    loop = SubtaskLoop(services)
    monkeypatch.setattr(
        loop,
        "_verify",
        _repeating([VerifyResult("pytest", True, "", "", 0), VerifyResult("pytest", True, "", "", 0)]),
    )

    store, task, sub = _store_with_one_subtask(workspace)
    outcome = loop._process_subtask(store, task, sub)

    assert outcome == SubtaskOutcome.DONE
    assert len(bus.of_type(events.TASK_REVIEW_START)) == 1  # exactly once, whole subtask
    assert len(bus.of_type(events.TASK_COMPLETION_CHECK)) == 2  # checked, failed, re-checked, passed
    assert factory.sessions[0].address_gate_calls == 1
    assert factory.sessions[0].address_review_calls == 0
    assert len(factory.sessions) == 1  # no escalation needed


# ── gate failure with fix budget exhausted: escalates, doesn't loop forever ──
def test_gate_failure_with_fix_budget_exhausted_escalates(workspace, bus, monkeypatch):
    services = _services(workspace, bus, max_fix_retries=1, max_escalations=1)
    factory = _FakeWorkerSessionFactory()
    _patch_common(monkeypatch, factory)
    monkeypatch.setattr(sl.code_review, "run", _repeating([code_review.ReviewResult(approved=True)]))
    monkeypatch.setattr(
        sl.completion_gate,
        "check",
        _repeating(
            [
                completion_gate.GateResult(passed=False, failed_conditions=["gate issue 1"]),
                completion_gate.GateResult(passed=False, failed_conditions=["gate issue 2"]),
                completion_gate.GateResult(passed=True),
            ]
        ),
    )

    loop = SubtaskLoop(services)
    monkeypatch.setattr(
        loop,
        "_verify",
        _repeating(
            [
                VerifyResult("pytest", True, "", "", 0),
                VerifyResult("pytest", True, "", "", 0),
                VerifyResult("pytest", True, "", "", 0),
            ]
        ),
    )

    store, task, sub = _store_with_one_subtask(workspace)
    outcome = loop._process_subtask(store, task, sub)

    assert outcome == SubtaskOutcome.DONE
    assert len(bus.of_type(events.TASK_ESCALATED)) == 1  # exhausted fix budget -> escalated
    assert len(factory.sessions) == 2  # escalation started a FRESH Worker conversation


# ── findings.md: populated on an ordinary test failure, not on a clean pass ──
def test_findings_recorded_on_ordinary_test_failure_then_fix_succeeds(workspace, bus, monkeypatch):
    services = _services(workspace, bus)
    factory = _FakeWorkerSessionFactory()
    _patch_common(monkeypatch, factory)
    monkeypatch.setattr(sl.code_review, "run", _repeating([code_review.ReviewResult(approved=True)]))
    monkeypatch.setattr(sl.completion_gate, "check", _repeating([completion_gate.GateResult(passed=True)]))

    loop = SubtaskLoop(services)
    monkeypatch.setattr(
        loop,
        "_verify",
        _repeating(
            [
                VerifyResult("pytest", False, "", "AssertionError: boom", 1),
                VerifyResult("pytest", True, "", "", 0),
            ]
        ),
    )

    store, task, sub = _store_with_one_subtask(workspace)
    outcome = loop._process_subtask(store, task, sub)

    assert outcome == SubtaskOutcome.DONE
    findings = workspace.read_agent_doc("findings.md")
    assert findings is not None and f"[{sub['id']}]" in findings
    assert factory.sessions[0].fix_calls == 1
    assert len(bus.of_type(events.FINDINGS_ENTRY_ADDED)) == 1
