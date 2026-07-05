"""The Worker-written file manifest (redesign).

Records every file the Worker writes/patches, with the one-line summary the
``write_file`` call supplied. This is the source of the **manifest-filtered
file tree** the Manager sees in every handoff: only Worker-written files appear
— never ``.agent/``, never the environment dirs, never ``.agentignore`` matches.

Persisted as ``.agent/manifest.json`` (an internal sidecar; the durable
per-file knowledge base is ``.agent/summaries/``) so the tree survives resume.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from workspace import Workspace

if TYPE_CHECKING:
    from server.events import EventBus

_SIDE_CAR = "manifest.json"


class Manifest:
    def __init__(self, workspace: Workspace, bus: "EventBus | None" = None):
        self.workspace = workspace
        self.bus = bus
        self.entries: dict[str, str] = {}  # rel path -> one-line summary
        self.load()

    # ── persistence ───────────────────────────────────────────────────────────
    def load(self) -> None:
        path = self.workspace.agent_path(_SIDE_CAR)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.entries = {str(k): str(v or "") for k, v in data.items()} if isinstance(data, dict) else {}
            except (json.JSONDecodeError, OSError):
                self.entries = {}

    def _save(self) -> None:
        self.workspace.write_agent_doc(_SIDE_CAR, json.dumps(self.entries, indent=2, sort_keys=True))

    # ── recording ─────────────────────────────────────────────────────────────
    def record(self, rel_path: str, summary: str | None) -> None:
        """Record/refresh a Worker-written file and its one-line summary."""
        key = self.workspace.relative(rel_path)
        if summary:
            self.entries[key] = str(summary).strip().replace("\n", " ")
        elif key not in self.entries:
            self.entries[key] = ""
        self._save()

    def forget_missing(self) -> None:
        """Drop entries whose files no longer exist on disk (resume hygiene)."""
        stale = [rel for rel in self.entries if not self.workspace.file_exists(rel)]
        for rel in stale:
            del self.entries[rel]
        if stale:
            self._save()

    # ── views ─────────────────────────────────────────────────────────────────
    def describe(self, rel_path: str) -> str:
        return self.entries.get(self.workspace.relative(rel_path), "")

    def paths(self) -> list[str]:
        return sorted(self.entries)

    def file_tree(self) -> str:
        """The manifest-filtered tree (Worker-written files only)."""
        return self.workspace.file_tree(self.paths())

    def render_markdown(self) -> str:
        """Annotated manifest for humans / the ``/project/manifest`` endpoint."""
        header = (
            f"# File Manifest — {self.workspace.root.name}\n\n"
            "Files written by the Worker. Each line: `path — one-line description`.\n\n"
        )
        matcher = self.workspace.ignore_matcher()
        lines = [
            f"- `{rel}`" + (f" — {desc}" if desc else "")
            for rel, desc in sorted(self.entries.items())
            if not matcher.is_ignored(rel)
        ]
        return header + ("\n".join(lines) + "\n" if lines else "_(no files yet)_\n")
