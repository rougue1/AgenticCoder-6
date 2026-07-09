"""Feature 3 — Per-subtask code review (Manager-as-Reviewer).

After the Worker's tests pass (and before Step C summarization), the Manager
reviews every file the subtask touched against the architecture summary and
the subtask's role standards. Read-only: no tools are offered.

One review cycle per subtask maximum (enforced by the caller,
``orchestrator/subtask_loop.py``) — this module only answers "is the current
state of these files acceptable," it does not loop itself. Skipped entirely
for scaffold/config/install subtask types (see the caller) since those don't
produce logic code worth reviewing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import promptlib
from services import Services
from stages import manager, roles as roles_stage

_APPROVED = "APPROVED"

# Bound on how much of each touched file is fed to the reviewer, mirroring
# stages/summarizer.py's _cap_content treatment of large generated files.
_FILE_CAP_CHARS = 24_000


@dataclass
class ReviewResult:
    approved: bool
    issues: list[str] = field(default_factory=list)


def run(services: Services, sub: dict, files_touched: list[str]) -> ReviewResult:
    if not files_touched:
        return ReviewResult(approved=True)

    role = str(sub.get("role") or "").strip().lower()
    role_text = roles_stage.read_role(services, role) or "(no role-specific standards recorded)"
    architecture_summary = (
        services.summaries.read("architecture.md") if services.summaries else ""
    ) or "(no architecture summary recorded)"
    files_block = _render_files(services, files_touched)

    instruction = promptlib.render(
        "code_review",
        subtask_id=str(sub.get("id", "")),
        subtask_title=str(sub.get("title", "")),
        intent=str(sub.get("intent", "")),
        role=role or roles_stage.DEFAULT_ROLE,
        role_text=role_text,
        architecture_summary=architecture_summary,
        files_block=files_block,
    )
    text = manager.call(services, "code_review", instruction)
    return _parse(services, text)


def _render_files(services: Services, rel_paths: list[str]) -> str:
    parts: list[str] = []
    for rel in sorted(set(rel_paths)):
        try:
            target = services.workspace.resolve_in_root(rel)
            content = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else None
        except Exception:
            content = None
        if content is None:
            continue
        parts.append(f"=== {rel} ===\n{_cap(content)}")
    return "\n\n".join(parts) or "(no file contents available)"


def _cap(content: str, limit_chars: int = _FILE_CAP_CHARS) -> str:
    if len(content) <= limit_chars:
        return content
    half = limit_chars // 2
    return content[:half] + "\n… (middle omitted for length) …\n" + content[-half:]


def _parse(services: Services, text: str) -> ReviewResult:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        services.bus.log(
            "code review returned an empty/unparseable reply — failing open (approved)",
            phase="code_review",
            level="warn",
        )
        return ReviewResult(approved=True)
    if any(ln.upper() == _APPROVED for ln in lines):
        return ReviewResult(approved=True)
    return ReviewResult(approved=False, issues=lines)
