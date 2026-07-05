"""Manager-call core + the per-subtask Manager roles (redesign).

Everything the large model does flows through :func:`call`: the anchor text
(``.agent/anchor.md``) is the system prompt for EVERY Manager call after it
exists — without exception — and each call is labeled with a phase for events
and JSONL dumps.

Per-subtask roles:

* :func:`handoff` — Step A. Builds the handoff packet via
  :class:`context.handoff.HandoffBuilder`, asks the Manager for precise Worker
  instructions plus a one-line architectural decision note, appends the note to
  the rolling ``decisions.md`` (trimmed to ``decision_log_max_entries``), and
  emits ``manager.handoff_ready``.
* :func:`escalate` — a completely new implementation plan from the full
  failure history (the Worker conversation is discarded by the caller).
* :func:`decompose` — 2-4 micro-subtasks in the tasks.json schema, validated
  by the caller through :meth:`taskstore.TaskStore.inject_decomposed`.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import promptlib
from config import MANAGER
from llm.tool_parser import extract_json
from server import events
from services import Services, clean_doc

if TYPE_CHECKING:
    from taskstore import TaskStore

# System prompt for Manager calls made BEFORE the anchor exists (Step 1 only).
BOOTSTRAP_SYSTEM = (
    "You are the Manager of an autonomous build pipeline: a pragmatic staff "
    "engineer who makes decisive, concrete choices. Follow the requested output "
    "format exactly. Never use git."
)

_DECISION_RE = re.compile(r"^\s*DECISION\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def call(
    services: Services,
    phase: str,
    instruction: str,
    *,
    system: str | None = None,
    temperature: float | None = None,
) -> str:
    """One Manager completion. The anchor is the system prompt once it exists."""
    services.check_cancel()
    if system is None:
        anchor = services.workspace.read_anchor_text() if services.workspace is not None else ""
        system = anchor or BOOTSTRAP_SYSTEM
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]
    result = services.client.complete(MANAGER, phase, messages, temperature=temperature)
    return clean_doc(result.text) or clean_doc(result.raw)


# ── Step A: the handoff ─────────────────────────────────────────────────────────
def handoff(services: Services, subtask: dict, store: "TaskStore") -> str:
    """Manager handoff for *subtask* -> the Worker's implementation instructions."""
    packet = services.handoff_builder.build(subtask, store)
    if packet.trimmed:
        services.bus.log(
            f"handoff context trimmed to fit {services.config.context.max_handoff_tokens} tokens: "
            + ", ".join(packet.trimmed),
            phase="handoff",
        )
    instruction = promptlib.render(
        "handoff",
        context=packet.user_context,
        subtask_type=str(subtask.get("type", "")),
        test_command=str(subtask.get("test_command") or ""),
    )
    text = call(services, "handoff", instruction, system=packet.system or None)

    instructions, decision = _split_decision(text)
    append_decision(services, str(subtask.get("id", "")), decision)
    services.bus.emit(
        events.MANAGER_HANDOFF_READY,
        "handoff",
        subtask_id=str(subtask.get("id", "")),
        decision=decision,
        instruction_chars=len(instructions),
    )
    return instructions


def _split_decision(text: str) -> tuple[str, str]:
    """Separate the Manager's instructions from its one-line DECISION note."""
    matches = list(_DECISION_RE.finditer(text or ""))
    if not matches:
        return (text or "").strip(), "(no decision note provided)"
    last = matches[-1]
    decision = last.group(1).strip()
    instructions = (text[: last.start()] + text[last.end():]).strip()
    return instructions, decision or "(no decision note provided)"


# ── the rolling decision log ────────────────────────────────────────────────────
_DECISIONS_HEADER = "# Architectural Decisions (rolling)\n"


def append_decision(services: Services, subtask_id: str, note: str) -> None:
    """Append one decision line and trim the log to the configured window."""
    ws = services.workspace
    max_entries = max(1, services.config.context.decision_log_max_entries)
    existing = ws.read_agent_doc("decisions.md", "") or ""
    entries = [ln for ln in existing.splitlines() if ln.startswith("- ")]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    entries.append(f"- [{stamp}] {subtask_id}: {note.strip().replace(chr(10), ' ')}")
    entries = entries[-max_entries:]
    ws.write_agent_doc("decisions.md", _DECISIONS_HEADER + "\n" + "\n".join(entries) + "\n")


# ── escalation ──────────────────────────────────────────────────────────────────
def escalate(services: Services, subtask: dict, failure_history: str, store: "TaskStore") -> str:
    """A fresh implementation plan built from the full failure history."""
    packet = services.handoff_builder.build(subtask, store)
    instruction = promptlib.render(
        "escalation",
        context=packet.user_context,
        failure_history=failure_history,
        test_command=str(subtask.get("test_command") or ""),
    )
    return call(services, "escalation", instruction, system=packet.system or None)


# ── decomposition ───────────────────────────────────────────────────────────────
def decompose(services: Services, subtask: dict, failure_history: str, store: "TaskStore") -> list[dict]:
    """Ask the Manager to split *subtask* into 2-4 micro-subtasks.

    Returns the raw list (possibly empty on a parse failure); the caller
    validates + injects via :meth:`TaskStore.inject_decomposed`.
    """
    packet = services.handoff_builder.build(subtask, store)
    instruction = promptlib.render(
        "decomposition",
        context=packet.user_context,
        failure_history=failure_history,
        subtask_id=str(subtask.get("id", "")),
    )
    text = call(services, "decomposition", instruction, system=packet.system or None)
    data = extract_json(text)
    if isinstance(data, dict):
        # Accept {"subtasks": [...]} as well as a bare array.
        data = data.get("subtasks") or data.get("tasks") or []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []
