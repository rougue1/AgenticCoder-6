"""PER-SUBTASK PLANNER (spec §9 PLAN) — LARGE-model implementation planner.

A single fresh call (no conversation). Assembles, via the ContextBuilder:
steering (verbatim) + design docs + the file manifest + the source files this
subtask touches/depends on, then asks for an exact, unambiguous plan the small
coding model can execute. Used both for the initial plan and for escalation
re-plans (with the full failure history appended).
"""

from __future__ import annotations

import promptlib
from context.compressor import P_DESIGN
from services import Services, clean_doc

_PLANNER_SYSTEM = (
    "You are a meticulous implementation planner in an autonomous build pipeline. "
    "You write plans precise enough that a coding model can execute them with no "
    "further reasoning. You describe WHAT to build — files, signatures, key logic, "
    "tests — but you do NOT write the full implementation code; that is the "
    "implementer's job. You never invent APIs or files; you reference the real, "
    "existing code provided to you."
)


def run(
    services: Services,
    task: dict,
    subtask: dict,
    *,
    failure_history: str = "",
    phase: str = "planner",
) -> str:
    services.check_cancel()
    builder = services.builder

    # Source files relevant to this subtask: the files it touches plus the files
    # produced by the subtasks it depends on (current code it builds on).
    relevant = list(subtask.get("files") or [])
    relevant += _dependency_files(services, subtask.get("depends_on") or [])
    relevant = _dedupe(relevant)
    active = set(subtask.get("files") or [])

    blocks = []
    steering = builder.doc_block("steering.md", required=True)
    if steering:
        blocks.append(steering)
    for name in ("sdd.md", "architecture.md"):
        b = builder.doc_block(name, priority=P_DESIGN)
        if b:
            blocks.append(b)
    blocks.extend(builder.manifest_blocks())
    blocks.extend(builder.source_blocks(relevant, active=active))

    instruction = promptlib.render(
        "planner",
        task_goal=task.get("goal", ""),
        subtask_id=subtask.get("id", ""),
        subtask_title=subtask.get("title", ""),
        subtask_intent=subtask.get("intent", ""),
        files=", ".join(subtask.get("files") or []) or "(none specified)",
        test_strategy=subtask.get("test_strategy", ""),
        failure_history=failure_history,
    )
    messages = builder.assemble(phase, _PLANNER_SYSTEM, blocks, instruction)
    result = services.client.complete(phase, messages)
    return clean_doc(result.text) or clean_doc(result.raw)


def _dependency_files(services: Services, dep_ids: list[str]) -> list[str]:
    from taskstore import TaskStore

    store = TaskStore.load(services.workspace)
    files: list[str] = []
    for dep in dep_ids:
        sub = store.get_subtask(dep)
        if sub:
            files.extend(sub.get("files") or [])
    return files


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
