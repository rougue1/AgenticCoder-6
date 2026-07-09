"""tasks.json model + persistence — the single source of truth for task state.

Redesign schema per subtask::

    {id, title, type, intent, role, files[], dependencies[], test_command,
     status, is_decomposed, can_decompose}

``role`` (Feature 2) is one of the built-in role names in
``stages.roles.ROLE_DESCRIPTIONS`` (backend, frontend, database,
infrastructure, testing, review), assigned by the Manager during task
planning. It is a soft field: unknown/missing values just fall back to
``backend.md`` at Worker-prompt-assembly time, never a validation failure.

``type`` ∈ {scaffold, implement, integrate, config, install}. ``status`` ∈
{pending, in_progress, done, blocked, decomposed}. Subtasks born from a
decomposition event carry ``is_decomposed: true`` and ``can_decompose: false``
— decomposed tasks cannot be decomposed again.

:meth:`TaskStore.inject_decomposed` atomically inserts micro-subtasks at the
original's position, re-validating schema + cycles on the full graph and only
persisting when both pass.
"""

from __future__ import annotations

import json
from typing import Iterator, Optional

from validation import VALID_TYPES, validate_all
from workspace import Workspace

PENDING = "pending"
IN_PROGRESS = "in_progress"
DONE = "done"
BLOCKED = "blocked"
DECOMPOSED = "decomposed"
VALID_STATUS = {PENDING, IN_PROGRESS, DONE, BLOCKED, DECOMPOSED}

# Statuses that permanently remove a subtask from scheduling.
_UNRUNNABLE = {BLOCKED, DECOMPOSED}


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

    def all_subtasks(self) -> list[dict]:
        return [sub for _, sub in self.subtasks()]

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

    def _satisfied_ids(self) -> set[str]:
        """Ids that count as satisfied dependencies: done subtasks, plus
        decomposed subtasks whose replacement micro-subtasks are all done
        (the group accomplishes the original's goal)."""
        done = self._done_ids()
        satisfied = set(done)
        for _, sub in self.subtasks():
            if sub.get("status") == DECOMPOSED and self._decomposition_done(sub["id"], done):
                satisfied.add(sub["id"])
        return satisfied

    def next_runnable(self) -> Optional[tuple[dict, dict]]:
        """Next pending subtask whose dependencies are all satisfied.

        Blocked and decomposed subtasks are never picked; subtasks transitively
        depending on an unrunnable/missing dependency are skipped so the loop
        makes progress elsewhere.
        """
        satisfied = self._satisfied_ids()
        unsatisfiable = self._unsatisfiable_ids()
        for task, sub in self.subtasks():
            if sub.get("status") != PENDING:
                continue
            if sub["id"] in unsatisfiable:
                continue
            deps = sub.get("dependencies") or []
            if all(d in satisfied for d in deps):
                return task, sub
        return None

    def _unsatisfiable_ids(self) -> set[str]:
        """Subtask ids that can never run: blocked/decomposed themselves, or
        (transitively) depending on one, or depending on a missing id.

        A dependency on a DECOMPOSED subtask counts as satisfied when all of
        its replacement micro-subtasks are done — the group accomplishes the
        original's goal.
        """
        all_ids = {sub["id"] for _, sub in self.subtasks()}
        done = self._done_ids()
        replaced_ok = {
            sub["id"]
            for _, sub in self.subtasks()
            if sub.get("status") == DECOMPOSED and self._decomposition_done(sub["id"], done)
        }
        unsat = {
            sub["id"] for _, sub in self.subtasks()
            if sub.get("status") in _UNRUNNABLE and sub["id"] not in replaced_ok
        }
        changed = True
        while changed:
            changed = False
            for _, sub in self.subtasks():
                if sub["id"] in unsat:
                    continue
                deps = sub.get("dependencies") or []
                if any((d in unsat) or (d not in all_ids) for d in deps):
                    unsat.add(sub["id"])
                    changed = True
        return unsat

    def _decomposition_done(self, original_id: str, done: set[str]) -> bool:
        children = [
            sub for _, sub in self.subtasks()
            if sub.get("is_decomposed") and sub.get("decomposed_from") == original_id
        ]
        return bool(children) and all(c.get("id") in done for c in children)

    def has_pending(self) -> bool:
        return any(sub.get("status") == PENDING for _, sub in self.subtasks())

    def all_resolved(self) -> bool:
        """True when nothing is pending/in_progress."""
        return all(sub.get("status") in (DONE, BLOCKED, DECOMPOSED) for _, sub in self.subtasks())

    # ── mutation ──────────────────────────────────────────────────────────────
    def set_status(self, subtask_id: str, status: str) -> None:
        assert status in VALID_STATUS, status
        sub = self.get_subtask(subtask_id)
        if sub is not None:
            sub["status"] = status
            self._roll_up_task_status()
            self.save()

    def cascade_block(self, blocked_id: str) -> list[str]:
        """Mark every pending subtask transitively depending on *blocked_id* as
        blocked too. Returns the ids that were newly blocked.

        A decomposed subtask whose replacement group can no longer complete
        (a micro-subtask blocked) propagates the block under its ORIGINAL id —
        dependents declared against the original, not the micro-subtasks.
        """
        newly: list[str] = []
        changed = True
        blocked_set = {blocked_id}
        while changed:
            changed = False
            for _, sub in self.subtasks():
                if sub.get("status") == DECOMPOSED and sub["id"] not in blocked_set:
                    children = [
                        c for _, c in self.subtasks()
                        if c.get("decomposed_from") == sub["id"]
                    ]
                    if any(c.get("status") == BLOCKED or c["id"] in blocked_set for c in children):
                        blocked_set.add(sub["id"])
                        changed = True
            for _, sub in self.subtasks():
                if sub.get("status") != PENDING:
                    continue
                deps = set(sub.get("dependencies") or [])
                if deps & blocked_set:
                    sub["status"] = BLOCKED
                    blocked_set.add(sub["id"])
                    newly.append(sub["id"])
                    changed = True
        if newly:
            self._roll_up_task_status()
            self.save()
        return newly

    def reset_in_progress(self) -> int:
        """Reset any in_progress subtask back to pending (for resume)."""
        n = 0
        for _, sub in self.subtasks():
            if sub.get("status") == IN_PROGRESS:
                sub["status"] = PENDING
                n += 1
        if n:
            self.save()
        return n

    def inject_decomposed(self, original_id: str, new_subtasks: list[dict]) -> list[str]:
        """Atomically replace *original_id* with micro-subtasks at its position.

        The new entries are normalized (``is_decomposed=True``,
        ``can_decompose=False``, ``decomposed_from`` back-reference), schema-
        validated, and the FULL updated graph is cycle-checked. Persists and
        returns ``[]`` only when both validations pass; otherwise the store is
        left untouched and the error list is returned.
        """
        task = self.parent_of(original_id)
        original = self.get_subtask(original_id)
        if task is None or original is None:
            return [f"unknown subtask id {original_id!r}"]

        prepared: list[dict] = []
        for i, raw in enumerate(new_subtasks, start=1):
            sub = _normalize_subtask(raw, fallback_id=f"{original_id}.d{i}")
            sub["is_decomposed"] = True
            sub["can_decompose"] = False
            sub["decomposed_from"] = original_id
            sub["status"] = PENDING
            # Micro-subtasks inherit the original's role when the Manager
            # didn't assign one explicitly.
            if not sub["role"]:
                sub["role"] = str(original.get("role") or "")
            # Micro-subtasks inherit the original's external dependencies unless
            # the Manager supplied an explicit list.
            if not sub["dependencies"]:
                prior = prepared[-1]["id"] if prepared else None
                inherited = list(original.get("dependencies") or [])
                sub["dependencies"] = inherited + ([prior] if prior else [])
            prepared.append(sub)

        # Validate on a deep-copied graph before mutating anything durable.
        trial = json.loads(json.dumps(self.data))
        for t in trial.get("tasks", []):
            subs = t.get("subtasks", [])
            for idx, s in enumerate(subs):
                if s.get("id") == original_id:
                    s["status"] = DECOMPOSED
                    t["subtasks"] = subs[: idx + 1] + prepared + subs[idx + 1:]
                    break
        flat = [s for t in trial.get("tasks", []) for s in t.get("subtasks", [])]
        errors = validate_all(flat)
        if errors:
            return errors

        self.data = trial
        self._roll_up_task_status()
        self.save()
        return []

    def _roll_up_task_status(self) -> None:
        for task in self.tasks:
            subs = task.get("subtasks", [])
            if not subs:
                continue
            statuses = {s.get("status") for s in subs}
            if statuses <= {DONE, DECOMPOSED}:
                task["status"] = DONE
            elif statuses & {IN_PROGRESS}:
                task["status"] = IN_PROGRESS
            elif statuses <= {BLOCKED, DONE, DECOMPOSED}:
                task["status"] = BLOCKED if BLOCKED in statuses else DONE
            else:
                task["status"] = PENDING

    # ── reporting ─────────────────────────────────────────────────────────────
    def counts(self) -> dict[str, int]:
        c = {PENDING: 0, IN_PROGRESS: 0, DONE: 0, BLOCKED: 0, DECOMPOSED: 0}
        for _, sub in self.subtasks():
            status = sub.get("status", PENDING)
            c[status] = c.get(status, 0) + 1
        return c

    def total_subtasks(self) -> int:
        return sum(1 for _ in self.subtasks())

    def summary(self) -> str:
        lines = [f"Project: {self.data.get('project', '?')}"]
        for task in self.tasks:
            lines.append(f"- {task.get('id')} [{task.get('status')}] {task.get('title')}")
            for sub in task.get("subtasks", []):
                lines.append(f"    - {sub.get('id')} [{sub.get('status')}] ({sub.get('type')}) {sub.get('title')}")
        c = self.counts()
        lines.append(
            f"\nTotals: {c[DONE]} done, {c[BLOCKED]} blocked, {c[DECOMPOSED]} decomposed, "
            f"{c[IN_PROGRESS]} in-progress, {c[PENDING]} pending."
        )
        return "\n".join(lines)


def normalize(data: dict, project_default: str) -> dict:
    """Coerce arbitrary planner output into the canonical redesign schema."""
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
            norm_task["subtasks"].append(_normalize_subtask(sub, fallback_id=f"{tid}.{si}"))
        out["tasks"].append(norm_task)
    return out


def _normalize_subtask(sub: dict, *, fallback_id: str) -> dict:
    sid = str(sub.get("id") or fallback_id)
    stype = str(sub.get("type") or "").strip().lower()
    if stype not in VALID_TYPES:
        # Tolerant coercion for planner slips; the validators still flag a
        # missing test_command on implement/integrate.
        stype = {"setup": "scaffold", "configure": "config", "installation": "install"}.get(stype, stype)
    return {
        "id": sid,
        "title": str(sub.get("title") or f"Subtask {sid}"),
        "type": stype if stype in VALID_TYPES else "implement",
        "intent": str(sub.get("intent") or sub.get("description") or ""),
        # Feature 2 (sub-agent roles): soft/optional field — an unknown or
        # missing role just falls back to backend.md at prompt-assembly time
        # (stages/roles.py::read_role), never a schema validation failure.
        "role": str(sub.get("role") or "").strip().lower(),
        "files": _as_str_list(sub.get("files")),
        # accept the legacy key as an alias, canonicalize to `dependencies`
        "dependencies": _as_str_list(sub.get("dependencies") if sub.get("dependencies") is not None else sub.get("depends_on")),
        "test_command": str(sub.get("test_command") or sub.get("test_strategy") or "").strip(),
        "status": _coerce_status(sub.get("status")),
        "is_decomposed": bool(sub.get("is_decomposed", False)),
        "can_decompose": bool(sub.get("can_decompose", not bool(sub.get("is_decomposed", False)))),
        **({"decomposed_from": str(sub["decomposed_from"])} if sub.get("decomposed_from") else {}),
    }


def _coerce_status(value) -> str:
    s = str(value or PENDING).strip().lower()
    return s if s in VALID_STATUS else PENDING


def _as_str_list(value) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return []
