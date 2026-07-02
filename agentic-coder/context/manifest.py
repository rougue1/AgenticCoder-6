"""File manifest + raw directory listing (spec §10).

Maintains two files under ``.agent/`` and keeps them in sync after every write:

* ``file-directory.txt`` — a filtered, recursive ``ls -R``-style listing (the
  ground-truth list of what exists, excluding noise dirs like ``node_modules``).
* ``file_manifest.md`` — an *annotated* tree where each generated file carries
  the one-line ``summary`` recorded at creation time, giving the planner
  semantic awareness without having to ``read_file`` everything.

Summaries are persisted to ``.agent/manifest.json`` so they survive resumes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from workspace import Workspace

if TYPE_CHECKING:
    from server.events import EventBus

# Directories never shown in the listing/manifest (build artefacts & VCS noise).
# Build/VCS/dependency noise. Deliberately conservative: ambiguous names like
# bin/obj/out are NOT excluded, so the listing never hides real user code.
EXCLUDE_DIRS = {
    ".git", ".agent", "node_modules", "dist", "build", "__pycache__", ".venv",
    "venv", ".env", ".pytest_cache", ".mypy_cache", ".ruff_cache", "target",
    ".next", ".nuxt", ".svelte-kit", "coverage", ".gradle", ".idea", ".vscode",
    "vendor", ".cache", ".turbo", ".parcel-cache",
}
# Individual files excluded from the listing (lockfile noise etc.).
EXCLUDE_FILES = {".DS_Store"}
_SIDE_CAR = "manifest.json"


class Manifest:
    def __init__(self, workspace: Workspace, bus: "EventBus | None" = None):
        self.workspace = workspace
        self.bus = bus
        self.summaries: dict[str, str] = {}
        self.load()

    # ── persistence ───────────────────────────────────────────────────────────
    def load(self) -> None:
        path = self.workspace.agent_path(_SIDE_CAR)
        if path.exists():
            try:
                self.summaries = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.summaries = {}

    def _save_sidecar(self) -> None:
        self.workspace.write_agent_doc(_SIDE_CAR, json.dumps(self.summaries, indent=2, sort_keys=True))

    # ── recording ─────────────────────────────────────────────────────────────
    def record(self, rel_path: str, summary: str | None) -> None:
        """Record/refresh a file's one-line summary and regenerate both files."""
        key = self.workspace.relative(rel_path)
        if summary:
            self.summaries[key] = summary.strip().replace("\n", " ")
        elif key not in self.summaries:
            self.summaries[key] = ""
        self._save_sidecar()
        self.regenerate()

    def describe(self, rel_path: str) -> str:
        return self.summaries.get(self.workspace.relative(rel_path), "")

    # ── generation ────────────────────────────────────────────────────────────
    def regenerate(self) -> None:
        files = self._walk()
        self.workspace.write_agent_doc("file-directory.txt", self._render_directory(files))
        self.workspace.write_agent_doc("file_manifest.md", self._render_manifest(files))

    def _walk(self) -> list[str]:
        """Return all project-relative file paths, filtered, sorted."""
        results: list[str] = []
        root = self.workspace.root
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS and not _is_hidden_noise(d))
            rel_dir = Path(dirpath).relative_to(root)
            for fname in sorted(filenames):
                if fname in EXCLUDE_FILES:
                    continue
                rel = (rel_dir / fname).as_posix()
                if rel.startswith(".agent/"):
                    continue
                results.append(rel)
        return sorted(results)

    def _render_directory(self, files: list[str]) -> str:
        header = f"# Project file listing for {self.workspace.root.name}\n# (excludes build/VCS noise; ground truth of what exists)\n\n"
        if not files:
            return header + "(no files yet)\n"
        lines = []
        last_dir = object()
        for rel in files:
            parent = str(Path(rel).parent)
            if parent != last_dir:
                last_dir = parent
                if parent != ".":
                    lines.append(f"{parent}/")
            indent = "  " if parent != "." else ""
            lines.append(f"{indent}{Path(rel).name}")
        return header + "\n".join(lines) + "\n"

    def _render_manifest(self, files: list[str]) -> str:
        header = (
            f"# File Manifest — {self.workspace.root.name}\n\n"
            "Annotated tree. Each line: `path — one-line description`. "
            "Use `read_file` when a one-liner isn't enough.\n\n"
        )
        if not files:
            return header + "_(no files yet)_\n"
        lines = []
        for rel in files:
            desc = self.summaries.get(rel, "")
            lines.append(f"- `{rel}`" + (f" — {desc}" if desc else ""))
        return header + "\n".join(lines) + "\n"


def _is_hidden_noise(name: str) -> bool:
    # Keep meaningful dotfiles/dirs (e.g. .github) but drop obvious caches.
    return name in EXCLUDE_DIRS
