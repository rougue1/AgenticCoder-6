"""Feature 5 — the completion gate.

Extra verification the orchestrator runs before marking a subtask done, on
top of "the test command exited 0" (that condition is already the precondition
for reaching this check at all — see ``orchestrator/subtask_loop.py``):

2. every file declared in the subtask's ``files[]`` exists and is non-empty.
3. none of the files this subtask wrote/patched contain a placeholder marker
   (``NotImplementedError``, ``TODO: implement``, …) — conservative substring
   checks only, so a note like "# TODO: review later" is never flagged.
4. the Worker's last tool dispatch was not an error.
5. no background session this subtask started is still running.

A failure here is injected back into the Worker conversation and counts as a
normal fix attempt against ``pipeline.max_fix_retries`` — see the caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services import Services
    from stages.worker import WorkerSession

# Exact strings (case-insensitive, whitespace-normalized) that indicate
# incomplete implementation logic rather than a genuine note/comment. Kept
# conservative on purpose: no open-ended regex, no flagging generic "TODO"s.
_PLACEHOLDER_NEEDLES = [
    "notimplementederror",
    "raise notimplemented",
    "throw new error('not implemented')",
    'throw new error("not implemented")',
    "pass # placeholder",
    "todo: implement",
    "fixme: implement",
]

_WS_RE = re.compile(r"[ \t]+")


@dataclass
class GateResult:
    passed: bool
    failed_conditions: list[str] = field(default_factory=list)


def check(services: "Services", sub: dict, session: "WorkerSession") -> GateResult:
    failed: list[str] = []

    # 2. declared files exist and are non-empty.
    for rel in sub.get("files") or []:
        rel = str(rel).strip()
        if not rel:
            continue
        try:
            path = services.workspace.resolve_in_root(rel)
        except Exception:
            failed.append(f"declared file missing: {rel}")
            continue
        if not path.is_file():
            failed.append(f"declared file missing: {rel}")
        elif path.stat().st_size == 0:
            failed.append(f"declared file is empty: {rel}")

    # 3. placeholder/incomplete-implementation markers in files touched this subtask.
    for rel in sorted(session.files_touched):
        needle = _find_placeholder(services, rel)
        if needle:
            failed.append(f"placeholder/incomplete implementation marker ({needle!r}) found in {rel}")

    # 4. the Worker's last tool call was not an error.
    if not session.last_call_ok:
        failed.append("the Worker's last tool call did not succeed")

    # 5. no lingering background sessions from this subtask.
    sandbox = services.sandbox
    if sandbox is not None:
        for session_id in sandbox.active_sessions:
            failed.append(
                f"Background session {session_id} is still running. Stop it before "
                "marking this subtask complete."
            )

    return GateResult(passed=not failed, failed_conditions=failed)


def _find_placeholder(services: "Services", rel: str) -> str | None:
    try:
        path = services.workspace.resolve_in_root(rel)
    except Exception:
        return None
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    normalized = _WS_RE.sub(" ", text).lower()
    for needle in _PLACEHOLDER_NEEDLES:
        if needle in normalized:
            return needle
    return None
