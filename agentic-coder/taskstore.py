"""tasks.json model + persistence (spec §8).

Wraps the on-disk ``tasks.json`` (the live task/subtask breakdown) with helpers
the orchestrator and subtask loop need: dependency-aware selection of the next
runnable subtask, status updates, resume reset, and a human-readable summary.
Subtasks are plain dicts so (de)serialization stays trivial.
"""

from __future__ import annotations

import json
from typing import Iterator, Optional

from workspace import Workspace

PENDING = "pending"
IN_PROGRESS = "in_progress"
DONE = "done"
BLOCKED = "blocked"
VALID_STATUS = {PENDING, IN_PROGRESS, DONE, BLOCKED}


class TaskStore:
    def __init__(self, workspace: Workspace, data: dict):
        self.workspace = workspace
        self.data = data

    # ── construction ──────────────────────────────────────────────────────────
    @classmethod
    def load(cls, workspace: Workspace) -> "TaskStore":
        raw = workspace.read_agent_doc("tasks.json")
        data = json.loads(raw) if raw else {"project": workspace.root.name, "tasks": []}
        return cls(workspace, normalize(data, workspace.root.name))

    @classmethod
    def from_data(cls, workspace: Workspace, data: dict) -> "TaskStore":
        store = cls(workspace, normalize(data, workspace.root.name))
        store.save()
        return store

    def save(self) -> None:
        self.workspace.write_agent_doc("tasks.json", json.dumps(self.data, indent=2))

    # ── iteration / lookup ────────────────────────────────────────────────────
    @property
    def tasks(self) -> list[dict]:
        return self.data.get("tasks", [])

    def subtasks(self) -> Iterator[tuple[dict, dict]]:
        for task in self.tasks:
            for sub in task.get("subtasks", []):
                yield task, sub

    def get_subtask(self, subtask_id: str) -> Optional[dict]:
        for _, sub in self.subtasks():
            if sub.get("id") == subtask_id:
                return sub
        return None

    def parent_of(self, subtask_id: str) -> Optional[dict]:
        for task, sub in self.subtasks():
            if sub.get("id") == subtask_id:
                return task
        return None

    # ── scheduling ────────────────────────────────────────────────────────────
    def _done_ids(self) -> set[str]:
        return {sub["id"] for _, sub in self.subtasks() if sub.get("status") == DONE}

    def _blocked_ids(self) -> set[str]:
        return {sub["id"] for _, sub in self.subtasks() if sub.get("status") == BLOCKED}

    def next_runnable(self) -> Optional[tuple[dict, dict]]:
        """Next pending subtask whose dependencies are all done.

        Subtasks transitively depending on a blocked subtask are skipped (their
        deps will never be satisfied), so the loop makes progress elsewhere.
        """
        done = self._done_ids()
        unsatisfiable = self._unsatisfiable_ids()
        for task, sub in self.subtasks():
            if sub.get("status") != PENDING:
                continue
            if sub["id"] in unsatisfiable:
                continue
            deps = sub.get("depends_on") or []
            if all(d in done for d in deps):
                return task, sub
        return None

    def _unsatisfiable_ids(self) -> set[str]:
        """Subtask ids that can never run because a dependency is blocked/missing."""
        blocked = self._blocked_ids()
        all_ids = {sub["id"] for _, sub in self.subtasks()}
        unsat = set(blocked)
        changed = True
        while changed:
            changed = False
            for _, sub in self.subtasks():
                if sub["id"] in unsat:
                    continue
                deps = sub.get("depends_on") or []
                if any((d in unsat) or (d not in all_ids) for d in deps):
                    unsat.add(sub["id"])
                    changed = True
        # the blocked ones themselves are "done-ish" for the purpose of skipping,
        # but keep them flagged so we never re-run them
        return unsat

    def has_pending(self) -> bool:
        return any(sub.get("status") == PENDING for _, sub in self.subtasks())

    def all_resolved(self) -> bool:
        """True when nothing is pending/in_progress (everything done or blocked)."""
        return all(sub.get("status") in (DONE, BLOCKED) for _, sub in self.subtasks())

    # ── mutation ──────────────────────────────────────────────────────────────
    def set_status(self, subtask_id: str, status: str) -> None:
        assert status in VALID_STATUS, status
        sub = self.get_subtask(subtask_id)
        if sub is not None:
            sub["status"] = status
            self._roll_up_task_status()
            self.save()

    def reset_in_progress(self) -> int:
        """Reset any in_progress subtask back to pending (for resume). Returns count."""
        n = 0
        for _, sub in self.subtasks():
            if sub.get("status") == IN_PROGRESS:
                sub["status"] = PENDING
                n += 1
        if n:
            self.save()
        return n

    def _roll_up_task_status(self) -> None:
        for task in self.tasks:
            subs = task.get("subtasks", [])
            if not subs:
                continue
            statuses = {s.get("status") for s in subs}
            if statuses <= {DONE}:
                task["status"] = DONE
            elif statuses & {IN_PROGRESS}:
                task["status"] = IN_PROGRESS
            elif statuses <= {BLOCKED, DONE}:
                task["status"] = BLOCKED if BLOCKED in statuses else DONE
            else:
                task["status"] = PENDING

    # ── reporting ─────────────────────────────────────────────────────────────
    def counts(self) -> dict[str, int]:
        c = {PENDING: 0, IN_PROGRESS: 0, DONE: 0, BLOCKED: 0}
        for _, sub in self.subtasks():
            c[sub.get("status", PENDING)] = c.get(sub.get("status", PENDING), 0) + 1
        return c

    def total_subtasks(self) -> int:
        return sum(1 for _ in self.subtasks())

    def summary(self) -> str:
        lines = [f"Project: {self.data.get('project','?')}"]
        for task in self.tasks:
            lines.append(f"- {task.get('id')} [{task.get('status')}] {task.get('title')}")
            for sub in task.get("subtasks", []):
                lines.append(f"    - {sub.get('id')} [{sub.get('status')}] {sub.get('title')}")
        c = self.counts()
        lines.append(
            f"\nTotals: {c[DONE]} done, {c[BLOCKED]} blocked, "
            f"{c[IN_PROGRESS]} in-progress, {c[PENDING]} pending."
        )
        return "\n".join(lines)


def normalize(data: dict, project_default: str) -> dict:
    """Coerce arbitrary planner output into the canonical schema."""
    if not isinstance(data, dict):
        data = {}
    out = {"project": str(data.get("project") or project_default), "tasks": []}
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
    for ti, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            continue
        tid = str(task.get("id") or f"T{ti:03d}")
        norm_task = {
            "id": tid,
            "title": str(task.get("title") or f"Task {ti}"),
            "goal": str(task.get("goal") or task.get("description") or ""),
            "status": _coerce_status(task.get("status")),
            "subtasks": [],
        }
        subs = task.get("subtasks")
        if not isinstance(subs, list):
            subs = []
        for si, sub in enumerate(subs, start=1):
            if not isinstance(sub, dict):
                continue
            sid = str(sub.get("id") or f"{tid}.{si}")
            norm_task["subtasks"].append(
                {
                    "id": sid,
                    "title": str(sub.get("title") or f"Subtask {sid}"),
                    "intent": str(sub.get("intent") or sub.get("description") or ""),
                    "files": _as_str_list(sub.get("files")),
                    "depends_on": _as_str_list(sub.get("depends_on")),
                    "test_strategy": str(sub.get("test_strategy") or sub.get("test") or ""),
                    "status": _coerce_status(sub.get("status")),
                }
            )
        out["tasks"].append(norm_task)
    return out


def _coerce_status(value) -> str:
    s = str(value or PENDING).strip().lower()
    return s if s in VALID_STATUS else PENDING


def _as_str_list(value) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []
