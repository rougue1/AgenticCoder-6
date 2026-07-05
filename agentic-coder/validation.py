"""tasks.json validators — subtask schema + dependency-graph cycle detection.

Used in three places: after Phase 1 task planning (with up to 2 Manager
correction rounds), inside :meth:`taskstore.TaskStore.inject_decomposed`
(decomposition output must pass the same bar), and during resume hardening
(manual edits to tasks.json can introduce cycles).
"""

from __future__ import annotations

VALID_TYPES = {"scaffold", "implement", "integrate", "config", "install"}
TEST_REQUIRED_TYPES = {"implement", "integrate"}


def validate_subtask_schema(subtasks: list[dict]) -> list[str]:
    """Schema errors for *subtasks* (empty list = valid).

    Checks: unique non-empty ids, a valid ``type``, a ``test_command`` on
    implement/integrate subtasks, and a non-empty ``files`` list everywhere.
    """
    errors: list[str] = []
    seen: set[str] = set()
    for i, sub in enumerate(subtasks):
        sid = str(sub.get("id") or "").strip()
        label = sid or f"subtask #{i + 1}"
        if not sid:
            errors.append(f"{label}: missing id")
        elif sid in seen:
            errors.append(f"{label}: duplicate id")
        seen.add(sid)

        stype = str(sub.get("type") or "").strip().lower()
        if stype not in VALID_TYPES:
            errors.append(
                f"{label}: invalid type {stype!r} (must be one of: {', '.join(sorted(VALID_TYPES))})"
            )
        if stype in TEST_REQUIRED_TYPES and not str(sub.get("test_command") or "").strip():
            errors.append(f"{label}: type {stype!r} requires a test_command")

        files = sub.get("files")
        if not isinstance(files, list) or not [f for f in files if str(f).strip()]:
            errors.append(f"{label}: files list must be non-empty")
    return errors


def validate_dependencies(subtasks: list[dict]) -> list[str]:
    """Dependency errors: unknown ids and cycles (via Kahn's topological sort)."""
    errors: list[str] = []
    ids = {str(s.get("id") or "") for s in subtasks} - {""}

    deps: dict[str, set[str]] = {}
    for sub in subtasks:
        sid = str(sub.get("id") or "")
        wanted = {str(d).strip() for d in (sub.get("dependencies") or []) if str(d).strip()}
        unknown = wanted - ids
        for u in sorted(unknown):
            errors.append(f"{sid}: depends on unknown subtask id {u!r}")
        if sid in wanted:
            errors.append(f"{sid}: depends on itself")
        deps[sid] = wanted & ids - {sid}

    # Kahn's algorithm: whatever can't be peeled off is part of a cycle.
    indegree = {sid: len(d) for sid, d in deps.items()}
    dependents: dict[str, set[str]] = {sid: set() for sid in deps}
    for sid, d in deps.items():
        for dep in d:
            dependents.setdefault(dep, set()).add(sid)
    queue = [sid for sid, n in indegree.items() if n == 0]
    resolved = 0
    while queue:
        sid = queue.pop()
        resolved += 1
        for child in dependents.get(sid, ()):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if resolved < len(deps):
        cyclic = sorted(sid for sid, n in indegree.items() if n > 0)
        errors.append(f"dependency cycle detected involving: {', '.join(cyclic)}")
    return errors


def validate_all(subtasks: list[dict]) -> list[str]:
    """Schema + dependency validation in one pass (deduplicated, ordered)."""
    errors = validate_subtask_schema(subtasks) + validate_dependencies(subtasks)
    seen: set[str] = set()
    out: list[str] = []
    for e in errors:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out
