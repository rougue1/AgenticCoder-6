"""Reads SDD/steering docs and source files from disk (spec §10).

Thin, side-effect-free helpers used by the context builder and the stages. All
file access goes through the workspace path-jail.
"""

from __future__ import annotations

from workspace import Workspace


class Loader:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    def doc(self, name: str, default: str = "") -> str:
        """Read an ``.agent/`` document by name (e.g. ``steering.md``)."""
        return self.workspace.read_agent_doc(name, default) or default

    def docs(self, *names: str) -> dict[str, str]:
        return {name: self.doc(name) for name in names}

    def source(self, rel_path: str) -> str | None:
        """Read a project source file; ``None`` if missing/out of bounds."""
        try:
            target = self.workspace.resolve_in_root(rel_path)
        except Exception:
            return None
        if not target.exists() or not target.is_file():
            return None
        try:
            return target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def sources(self, rel_paths) -> dict[str, str]:
        """Read several source files; skips ones that don't exist."""
        out: dict[str, str] = {}
        for rel in rel_paths or []:
            content = self.source(rel)
            if content is not None:
                out[self.workspace.relative(rel)] = content
        return out
