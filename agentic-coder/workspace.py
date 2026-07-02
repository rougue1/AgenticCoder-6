"""Project-directory resolution and the working-directory jail (spec §5, §12.1).

Every file read/write and every shell command in the generated app must occur
inside the resolved project directory. :class:`Workspace` owns that absolute
path, creates the ``.agent/`` control directory, and validates every path
against the root so nothing can escape via ``..``, ``~`` or an absolute path.
"""

from __future__ import annotations

import re
from pathlib import Path

# Control/spec documents that live under .agent/. Names are referenced by stages.
AGENT_DOCS = (
    "project_brief.md",
    "requirements.md",
    "stack.md",
    "architecture.md",
    "sdd.md",
    "steering.md",
    "tasks.json",
    "file_manifest.md",
    "file-directory.txt",
    "blocked.md",
    "review.md",
    "run.log",
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class PathEscapeError(Exception):
    """Raised when a path or command would escape the project root."""


def slugify(name: str, fallback: str = "project") -> str:
    """Turn an arbitrary project name into a filesystem-safe slug."""
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    slug = slug[:60].strip("-")
    return slug or fallback


class Workspace:
    """The resolved project directory plus helpers for the path jail."""

    def __init__(self, project_root: Path):
        self.root = Path(project_root).resolve()
        self.agent_dir = self.root / ".agent"
        self.llm_calls_dir = self.agent_dir / "llm_calls"

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def resolve(cls, project_dir_cfg: str, slug: str | None, tool_root: Path) -> "Workspace":
        """Resolve the project root per spec §5.

        1. If ``project_dir_cfg`` is set, use it (created if missing).
        2. Otherwise use ``<tool_root>/../sandbox/<slug>/`` — the ``sandbox/``
           directory that sits *next to* the tool source dir, never inside it.
           ``tool_root`` is ``agentic-coder/`` (anchored on ``main.py``'s own
           location, not the CWD), so the default output always lands in
           ``AgenticCoder-6/sandbox/<slug>/`` regardless of where the user runs
           ``python agentic-coder/main.py`` from.
        """
        cfg = (project_dir_cfg or "").strip()
        if cfg:
            root = Path(cfg).expanduser().resolve()
        else:
            if not slug:
                raise ValueError("project_dir is blank and no slug was provided")
            root = (Path(tool_root).resolve().parent / "sandbox" / slug).resolve()
        return cls(root)

    def ensure(self) -> "Workspace":
        """Create the project root, the ``.agent/`` dir and the llm_calls dir."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.llm_calls_dir.mkdir(parents=True, exist_ok=True)
        return self

    # ── path validation ───────────────────────────────────────────────────────
    def resolve_in_root(self, path: str | Path) -> Path:
        """Resolve *path* relative to the root and assert it stays inside it.

        Rejects ``~`` expansion, absolute paths outside the root, and ``..``
        traversal that escapes the root. Returns the absolute, resolved path.
        """
        raw = str(path)
        if "~" in raw:
            raise PathEscapeError(f"Home (~) paths are not allowed: {raw!r}")

        p = Path(raw)
        candidate = p.resolve() if p.is_absolute() else (self.root / p).resolve()

        if candidate != self.root and self.root not in candidate.parents:
            raise PathEscapeError(
                f"Path {raw!r} resolves to {candidate}, outside project root {self.root}"
            )
        return candidate

    def relative(self, path: str | Path) -> str:
        """Return *path* as a POSIX string relative to the root (best effort)."""
        try:
            return self.resolve_in_root(path).relative_to(self.root).as_posix()
        except (PathEscapeError, ValueError):
            return str(path)

    # ── agent-doc IO ──────────────────────────────────────────────────────────
    def agent_path(self, name: str) -> Path:
        return self.agent_dir / name

    def write_agent_doc(self, name: str, content: str) -> Path:
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        path = self.agent_path(name)
        path.write_text(content, encoding="utf-8")
        return path

    def read_agent_doc(self, name: str, default: str | None = None) -> str | None:
        path = self.agent_path(name)
        if not path.exists():
            return default
        return path.read_text(encoding="utf-8")

    def agent_doc_exists(self, name: str) -> bool:
        return self.agent_path(name).exists()

    # ── generic file IO (jailed) ──────────────────────────────────────────────
    def read_file(self, path: str | Path) -> str:
        target = self.resolve_in_root(path)
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str | Path, content: str) -> Path:
        target = self.resolve_in_root(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def file_exists(self, path: str | Path) -> bool:
        try:
            return self.resolve_in_root(path).exists()
        except PathEscapeError:
            return False
