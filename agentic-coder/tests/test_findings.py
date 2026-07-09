"""Feature 1 — the shared, append-only error-persistence log (findings.md):
first-sentence truncation, the `[T<id>] <summary>` format, and the
HandoffBuilder's PRIOR FAILURES section (placed after decisions, before file
summaries; capped at the last 10 lines)."""

from __future__ import annotations

import threading

from config import load_config
from context.handoff import HandoffBuilder
from context.manifest import Manifest
from context.summaries import SummaryIndex
from llm.client import LLMClient
from orchestrator.subtask_loop import SubtaskLoop, _first_sentence
from server import events
from services import Services
from taskstore import TaskStore


# ── _first_sentence ──────────────────────────────────────────────────────────
def test_first_sentence_truncates_to_first_sentence():
    text = "pytest-asyncio requires explicit event_loop fixture in v0.23+. Also unrelated context."
    assert _first_sentence(text) == "pytest-asyncio requires explicit event_loop fixture in v0.23+."


def test_first_sentence_collapses_newlines():
    text = "line one\nline two continues.\nmore stuff"
    assert _first_sentence(text) == "line one line two continues."


def test_first_sentence_falls_back_to_whole_text_without_terminal_punctuation():
    assert _first_sentence("no terminal punctuation here") == "no terminal punctuation here"


def test_first_sentence_empty_input():
    assert _first_sentence("") == ""
    assert _first_sentence("   ") == ""


# ── SubtaskLoop._record_finding ──────────────────────────────────────────────
def _loop(workspace, bus):
    cfg = load_config()
    services = Services(config=cfg, bus=bus, client=LLMClient(cfg, bus), cancel_event=threading.Event())
    services.workspace = workspace
    return SubtaskLoop(services)


class _FakeSession:
    def __init__(self, summary):
        self._summary = summary

    def summarize_failure(self):
        return self._summary


def test_record_finding_appends_bracketed_format(workspace, bus):
    loop = _loop(workspace, bus)
    loop._record_finding(
        "T003.2", _FakeSession("pytest-asyncio requires explicit event_loop fixture in v0.23+.")
    )
    text = workspace.read_agent_doc("findings.md")
    assert text == "[T003.2] pytest-asyncio requires explicit event_loop fixture in v0.23+.\n"


def test_record_finding_is_append_only(workspace, bus):
    loop = _loop(workspace, bus)
    loop._record_finding("T003.2", _FakeSession("first failure."))
    loop._record_finding("T005.1", _FakeSession("second failure."))
    lines = workspace.read_agent_doc("findings.md").splitlines()
    assert lines == ["[T003.2] first failure.", "[T005.1] second failure."]


def test_record_finding_emits_event(workspace, bus):
    loop = _loop(workspace, bus)
    loop._record_finding("T003.2", _FakeSession("something broke."))
    added = bus.of_type(events.FINDINGS_ENTRY_ADDED)
    assert len(added) == 1 and added[0].data["id"] == "T003.2"


def test_record_finding_skips_empty_summary(workspace, bus):
    loop = _loop(workspace, bus)
    loop._record_finding("T003.2", _FakeSession(""))
    assert not workspace.agent_doc_exists("findings.md")


def test_record_finding_survives_worker_exception(workspace, bus):
    class _Boom:
        def summarize_failure(self):
            raise RuntimeError("model call failed")

    loop = _loop(workspace, bus)
    loop._record_finding("T003.2", _Boom())  # must not raise
    assert not workspace.agent_doc_exists("findings.md")


# ── HandoffBuilder: PRIOR FAILURES section ───────────────────────────────────
def _builder(workspace, cfg):
    summaries = SummaryIndex(workspace)
    manifest = Manifest(workspace)
    return HandoffBuilder(workspace, summaries, manifest, cfg)


def _subtask_and_store(workspace):
    subtask = {
        "id": "T1.1", "title": "s", "type": "implement", "intent": "do x",
        "files": ["a.py"], "dependencies": [], "test_command": "pytest",
    }
    store = TaskStore.from_data(
        workspace, {"project": "p", "tasks": [{"id": "T1", "title": "t1", "subtasks": [subtask]}]}
    )
    return subtask, store


def test_prior_failures_included_after_decisions_before_summaries(workspace):
    cfg = load_config()
    workspace.write_agent_doc("decisions.md", "# Architectural Decisions (rolling)\n\n- decision one\n")
    workspace.write_agent_doc("findings.md", "[T001.1] first learning.\n[T001.2] second learning.\n")
    builder = _builder(workspace, cfg)
    builder.summaries.write("a.py", "summary of a.py")
    subtask, store = _subtask_and_store(workspace)
    handoff = builder.build(subtask, store)

    ctx = handoff.user_context
    assert "PRIOR FAILURES" in ctx
    decisions_idx = ctx.index("Recent architectural decisions")
    findings_idx = ctx.index("PRIOR FAILURES")
    summary_idx = ctx.index("FILE SUMMARY")
    assert decisions_idx < findings_idx < summary_idx
    assert "first learning" in ctx and "second learning" in ctx


def test_prior_failures_capped_at_last_10_lines(workspace):
    cfg = load_config()
    lines = "\n".join(f"[T{i:03d}] entry-{i:02d} learning." for i in range(1, 16))
    workspace.write_agent_doc("findings.md", lines + "\n")
    builder = _builder(workspace, cfg)
    subtask, store = _subtask_and_store(workspace)
    handoff = builder.build(subtask, store)
    assert "entry-01 " not in handoff.user_context  # oldest of 15, beyond the last 10
    assert "entry-06 " in handoff.user_context  # the oldest of the kept last 10
    assert "entry-15 " in handoff.user_context  # the newest


def test_prior_failures_absent_when_findings_missing(workspace):
    cfg = load_config()
    builder = _builder(workspace, cfg)
    subtask, store = _subtask_and_store(workspace)
    handoff = builder.build(subtask, store)
    assert "PRIOR FAILURES" not in handoff.user_context
