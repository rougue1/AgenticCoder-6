"""Step C — Post-task summarization (Manager-as-Analyst).

After a subtask completes (passed OR hard-blocked), every unique file it wrote
or patched gets one Analyst call producing a concise summary — primary purpose,
main classes/functions, critical imports/dependencies, behavioral notes —
capped at ``context.max_summary_tokens``. Summaries land in
``.agent/summaries/`` (slugged filenames; latest wins) and feed every future
handoff. The same Analyst also summarizes the Phase 1 documents so handoffs
can always include them without shipping raw docs.

After a successful install-type subtask, the dependency-conflict check
(``pip check`` / ``npm ls --depth=0``) runs silently; conflicts are logged as
warnings, never failures.
"""

from __future__ import annotations

import promptlib
from server import events
from services import Services
from stages import manager
from tokens import estimate_tokens


def summarize_files(services: Services, subtask_id: str, rel_paths: list[str]) -> int:
    """Summarize every file in *rel_paths* (deduplicated). Returns the count."""
    unique = sorted({p for p in rel_paths if p})
    if not unique:
        return 0
    services.bus.emit(events.SUMMARIZER_START, "summarizer", subtask_id=subtask_id, files=len(unique))
    n = 0
    for rel in unique:
        content = _read_source(services, rel)
        if content is None:
            services.bus.log(f"skipping summary for missing file {rel}", phase="summarizer", level="warn")
            continue
        summary = _analyst_call(services, rel, content)
        services.summaries.write(rel, summary)
        services.bus.emit(events.SUMMARIZER_FILE_COMPLETE, "summarizer", path=rel, tokens=estimate_tokens(summary))
        n += 1
    return n


def summarize_doc(services: Services, doc_name: str) -> None:
    """Summarize an ``.agent/`` document (architecture.md / requirements.md)
    under its bare name so the HandoffBuilder's always-include finds it."""
    content = services.loader.doc(doc_name)
    if not content.strip():
        return
    summary = _analyst_call(services, doc_name, content)
    services.summaries.write(doc_name, summary)
    services.bus.emit(events.SUMMARIZER_FILE_COMPLETE, "summarizer", path=doc_name, tokens=estimate_tokens(summary))


def post_install_check(services: Services) -> None:
    """Silent dependency-conflict check after an install-type subtask passes."""
    checks: list[str] = []
    if services.workspace.file_exists("requirements.txt") or (
        services.environment and services.environment.venv_path
    ):
        checks.append("pip check")
    if services.workspace.file_exists("package.json"):
        checks.append("npm ls --depth=0")
    for cmd in checks:
        res = services.sandbox.run(cmd, validate=False)
        if not res.ok:
            services.bus.log(
                f"dependency conflict check `{cmd}` reported issues (exit {res.exit_code}): "
                f"{(res.stdout or res.stderr)[:400]}",
                phase="summarizer",
                level="warn",
            )


# ── internals ─────────────────────────────────────────────────────────────────
def _analyst_call(services: Services, rel: str, content: str) -> str:
    max_tokens = max(50, services.config.context.max_summary_tokens)
    instruction = promptlib.render(
        "analyst",
        path=rel,
        content=_cap_content(content),
        max_tokens=max_tokens,
    )
    summary = manager.call(services, "analyst", instruction)
    return _cap_summary(summary, max_tokens)


def _read_source(services: Services, rel: str) -> str | None:
    try:
        target = services.workspace.resolve_in_root(rel)
    except Exception:
        return None
    if not target.is_file():
        return None
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _cap_content(content: str, limit_chars: int = 48_000) -> str:
    """Bound the raw file fed to the Analyst (huge generated files exist)."""
    if len(content) <= limit_chars:
        return content
    half = limit_chars // 2
    return content[:half] + "\n… (middle omitted for length) …\n" + content[-half:]


def _cap_summary(summary: str, max_tokens: int) -> str:
    """Hard-cap the summary at *max_tokens* (the prompt asks; this enforces)."""
    text = summary.strip()
    if estimate_tokens(text) <= max_tokens:
        return text
    approx_chars = max_tokens * 4
    return text[:approx_chars].rsplit("\n", 1)[0].rstrip() + "\n… (summary truncated to token cap)"
