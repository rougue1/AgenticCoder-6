"""Feature 2 — sub-agent roles: role-file generation (mocked Manager calls),
``read_role``'s fallback chain, and the Worker's 3-layer system prompt
assembly (anchor -> role -> per-subtask handoff instructions -> the generic
agent identity/tool protocol, always in that order)."""

from __future__ import annotations

import threading

from config import load_config
from llm.client import LLMClient
from services import Services
from stages import roles
from stages.worker import HARD_RULES, WorkerSession


def _services(workspace, bus):
    cfg = load_config()
    services = Services(
        config=cfg, bus=bus, client=LLMClient(cfg, bus, workspace), cancel_event=threading.Event()
    )
    services.workspace = workspace
    return services


class _FakeStack:
    raw_output = "python-fastapi, python 3.12"


# ── stages.roles.run ──────────────────────────────────────────────────────────
def test_run_generates_all_six_role_files(workspace, bus, monkeypatch):
    services = _services(workspace, bus)
    services.stack = _FakeStack()
    calls: list[tuple[str, str]] = []

    def fake_call(services_, phase, instruction, **kwargs):
        calls.append((phase, instruction))
        return f"ROLE BODY #{len(calls)}"

    monkeypatch.setattr(roles.manager, "call", fake_call)
    roles.run(services)

    assert len(calls) == len(roles.ROLE_DESCRIPTIONS) == 6
    assert all(phase == "roles" for phase, _ in calls)
    for name in roles.ROLE_DESCRIPTIONS:
        assert workspace.agent_doc_exists(f"roles/{name}.md")
        assert workspace.read_agent_doc(f"roles/{name}.md").startswith("ROLE BODY")


def test_run_passes_stack_and_role_description_into_the_prompt(workspace, bus, monkeypatch):
    services = _services(workspace, bus)
    services.stack = _FakeStack()
    seen_instructions: set[str] = set()

    def fake_call(services_, phase, instruction, **kwargs):
        seen_instructions.add(instruction)
        return "body"

    monkeypatch.setattr(roles.manager, "call", fake_call)
    roles.run(services)

    combined = "\n".join(seen_instructions)
    assert "python-fastapi" in combined
    assert roles.ROLE_DESCRIPTIONS["database"] in combined


# ── stages.roles.read_role fallback chain ────────────────────────────────────
def test_read_role_returns_the_requested_role(workspace, bus):
    services = _services(workspace, bus)
    workspace.write_agent_doc("roles/database.md", "DATABASE ROLE TEXT")
    workspace.write_agent_doc("roles/backend.md", "BACKEND ROLE TEXT")
    assert roles.read_role(services, "database") == "DATABASE ROLE TEXT"


def test_read_role_falls_back_to_backend_when_requested_role_missing(workspace, bus):
    services = _services(workspace, bus)
    workspace.write_agent_doc("roles/backend.md", "BACKEND ROLE TEXT")
    assert roles.read_role(services, "frontend") == "BACKEND ROLE TEXT"


def test_read_role_falls_back_to_first_available_role_file(workspace, bus):
    services = _services(workspace, bus)
    workspace.write_agent_doc("roles/review.md", "REVIEW ROLE TEXT")  # backend.md absent
    assert roles.read_role(services, "nonexistent-role") == "REVIEW ROLE TEXT"


def test_read_role_returns_empty_when_no_role_files_exist(workspace, bus):
    services = _services(workspace, bus)
    assert roles.read_role(services, "backend") == ""
    assert roles.read_role(services, "") == ""


def test_read_role_handles_missing_workspace():
    services = Services(config=load_config(), bus=None, client=None, cancel_event=threading.Event())
    assert roles.read_role(services, "backend") == ""


# ── WorkerSession 3-layer system prompt (Feature 2) ──────────────────────────
def test_worker_system_prompt_layers_anchor_role_then_instructions(workspace, bus):
    services = _services(workspace, bus)
    workspace.write_agent_doc("anchor.md", "THE ANCHOR TEXT")
    workspace.write_agent_doc("roles/backend.md", "THE ROLE TEXT")
    subtask = {"id": "T1.1", "title": "s", "type": "implement", "role": "backend"}
    session = WorkerSession(services, subtask, "THE HANDOFF INSTRUCTIONS")
    system = session.messages[0]["content"]

    assert system.startswith("THE ANCHOR TEXT")
    anchor_idx = system.index("THE ANCHOR TEXT")
    role_idx = system.index("THE ROLE TEXT")
    instr_idx = system.index("THE HANDOFF INSTRUCTIONS")
    rules_idx = system.index(HARD_RULES)
    assert anchor_idx < role_idx < instr_idx < rules_idx


def test_worker_system_prompt_omits_role_section_when_no_role_files_exist(workspace, bus):
    services = _services(workspace, bus)
    workspace.write_agent_doc("anchor.md", "THE ANCHOR TEXT")
    subtask = {"id": "T1.1", "title": "s", "type": "scaffold"}  # no role assigned, no role files
    session = WorkerSession(services, subtask, "INSTRUCTIONS HERE")
    system = session.messages[0]["content"]
    assert "# Role:" not in system
    assert "THE ANCHOR TEXT" in system and "INSTRUCTIONS HERE" in system


def test_worker_first_user_turn_no_longer_duplicates_instructions(workspace, bus, monkeypatch):
    """worker.j2 used to repeat the Manager's instructions in the first user
    turn; they now live only in the system prompt (Feature 2), which also
    fixes the latent bug where instructions (the oldest non-system message)
    were the first thing pack_conversation dropped under context pressure."""
    services = _services(workspace, bus)
    workspace.write_agent_doc("anchor.md", "ANCHOR")
    subtask = {"id": "T1.1", "title": "s", "type": "implement", "role": "backend"}
    session = WorkerSession(services, subtask, "UNIQUE-INSTRUCTION-MARKER-XYZ")

    captured: dict = {}

    def fake_drive(instruction):
        captured["instruction"] = instruction
        return True

    monkeypatch.setattr(session, "_drive", fake_drive)
    session.implement()
    assert "UNIQUE-INSTRUCTION-MARKER-XYZ" not in captured["instruction"]
    assert "UNIQUE-INSTRUCTION-MARKER-XYZ" in session.messages[0]["content"]
