"""The ``.agent/summaries/`` index — Manager-as-Analyst file summaries.

One ``.md`` file per unique source file, named ``<sanitized-path>_<sha6>.md``
(:func:`workspace.summary_slug`). Each file carries a one-line ``path:`` header
so the index can be rebuilt from disk on resume (the slug is not reversible).

The in-memory index maps relative file paths -> slug filenames so lookups are
O(1) during handoff assembly. Latest summary always wins (overwrites).
"""

from __future__ import annotations

from workspace import Workspace, summary_slug

_PATH_HEADER = "path:"


class SummaryIndex:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self._index: dict[str, str] = {}  # rel path -> slug filename (with .md)
        self.reload()

    # ── persistence ───────────────────────────────────────────────────────────
    def reload(self) -> None:
        """Rebuild the index by scanning ``.agent/summaries/`` (resume path)."""
        self._index.clear()
        d = self.workspace.summaries_dir
        if not d.is_dir():
            return
        for f in sorted(d.glob("*.md")):
            try:
                first = f.read_text(encoding="utf-8").splitlines()[:1]
            except OSError:
                continue
            if first and first[0].startswith(_PATH_HEADER):
                rel = first[0][len(_PATH_HEADER):].strip()
                if rel:
                    self._index[rel] = f.name

    def write(self, rel_path: str, summary: str) -> str:
        """Write/overwrite the summary for *rel_path*; returns the slug filename."""
        rel = self.workspace.relative(rel_path) if not rel_path.startswith(".agent") else rel_path
        name = summary_slug(rel) + ".md"
        body = f"{_PATH_HEADER} {rel}\n\n{summary.strip()}\n"
        self.workspace.summaries_dir.mkdir(parents=True, exist_ok=True)
        (self.workspace.summaries_dir / name).write_text(body, encoding="utf-8")
        self._index[rel] = name
        return name

    # ── lookups ───────────────────────────────────────────────────────────────
    def exists(self, rel_path: str) -> bool:
        return rel_path in self._index

    def read(self, rel_path: str) -> str:
        """The summary text for *rel_path* ('' when absent)."""
        name = self._index.get(rel_path)
        if not name:
            return ""
        try:
            text = (self.workspace.summaries_dir / name).read_text(encoding="utf-8")
        except OSError:
            return ""
        # Strip the path: header line.
        lines = text.splitlines()
        if lines and lines[0].startswith(_PATH_HEADER):
            lines = lines[1:]
        return "\n".join(lines).strip()

    def paths(self) -> list[str]:
        return sorted(self._index)

    def read_all(self) -> dict[str, str]:
        return {rel: self.read(rel) for rel in self.paths()}
