"""Feature 3 — per-subtask code review (Manager-as-Reviewer): the APPROVED /
issue-list reply parser, the empty-files short circuit, and failing OPEN on
an unparseable/empty reply so a flaky review call can never block a subtask
forever."""

from __future__ import annotations

import threading

from config import load_config
from context.summaries import SummaryIndex
from llm.client import LLMClient
from services import Services
from stages import code_review


def _services(workspace, bus):
    cfg = load_config()
    services = Services(
        config=cfg, bus=bus, client=LLMClient(cfg, bus, workspace), cancel_event=threading.Event()
    )
    services.workspace = workspace
    services.summaries = SummaryIndex(workspace)
    return services


def test_run_short_circuits_approved_when_no_files_touched(workspace, bus):
    services = _services(workspace, bus)
    result = code_review.run(services, {"id": "T1.1", "type": "implement"}, [])
    assert result.approved and result.issues == []


def test_run_calls_manager_and_parses_approved(workspace, bus, monkeypatch):
    services = _services(workspace, bus)
    workspace.write_file("app.py", "def add(a, b):\n    return a + b\n")
    monkeypatch.setattr(code_review.manager, "call", lambda *a, **k: "APPROVED")
    result = code_review.run(services, {"id": "T1.1", "type": "implement", "role": "backend"}, ["app.py"])
    assert result.approved and result.issues == []


def test_run_parses_issue_list(workspace, bus, monkeypatch):
    services = _services(workspace, bus)
    workspace.write_file("app.py", "def add(a, b):\n    pass\n")
    reply = "1. add() doesn't return the sum\n2. missing error handling for non-numeric input"
    monkeypatch.setattr(code_review.manager, "call", lambda *a, **k: reply)
    result = code_review.run(services, {"id": "T1.1", "type": "implement"}, ["app.py"])
    assert not result.approved
    assert result.issues == [
        "1. add() doesn't return the sum",
        "2. missing error handling for non-numeric input",
    ]


def test_run_fails_open_on_empty_reply(workspace, bus, monkeypatch):
    services = _services(workspace, bus)
    workspace.write_file("app.py", "pass\n")
    monkeypatch.setattr(code_review.manager, "call", lambda *a, **k: "")
    result = code_review.run(services, {"id": "T1.1", "type": "implement"}, ["app.py"])
    assert result.approved  # fail open — never blocks a subtask on a flaky reply


def test_run_reads_role_standards_and_architecture_summary_into_the_prompt(workspace, bus, monkeypatch):
    services = _services(workspace, bus)
    workspace.write_agent_doc("roles/backend.md", "BACKEND STANDARDS TEXT")
    services.summaries.write("architecture.md", "ARCHITECTURE SUMMARY TEXT")
    workspace.write_file("app.py", "x = 1\n")
    captured = {}

    def fake_call(services_, phase, instruction, **kwargs):
        captured["instruction"] = instruction
        return "APPROVED"

    monkeypatch.setattr(code_review.manager, "call", fake_call)
    code_review.run(services, {"id": "T1.1", "type": "implement", "role": "backend"}, ["app.py"])
    assert "BACKEND STANDARDS TEXT" in captured["instruction"]
    assert "ARCHITECTURE SUMMARY TEXT" in captured["instruction"]
    assert "app.py" in captured["instruction"]


def test_parse_treats_approved_anywhere_as_a_safety_net(workspace, bus):
    result = code_review._parse(_services(workspace, bus), "APPROVED")
    assert result.approved
    result2 = code_review._parse(_services(workspace, bus), "some preamble\nAPPROVED")
    assert result2.approved
