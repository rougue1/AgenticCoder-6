"""Project-directory resolution, the path jail, and workspace security.

Every file read/write and every shell command in the generated app must occur
inside the resolved project directory. :class:`Workspace` owns that absolute
path, creates the ``.agent/`` control tree, and validates every path against
the root so nothing escapes via ``..``, ``~`` or an absolute path.

Redesign additions:

* **Anchor immutability** — ``.agent/anchor.md`` is written exactly once;
  any later write attempt raises :class:`AnchorImmutableError` (a hard error,
  never a silent skip).
* **.agentignore** — loaded from the workspace root and applied to every
  tool-level ``read_file``/``write_file``/``patch_file`` (the ``run`` tool is
  governed by the command allowlist instead). ``.venv/`` and ``node_modules/``
  are always ignored.
* **Summary slugs** — ``src/api/routes.py`` -> ``src__api__routes_py_<sha6>.md``
  for ``.agent/summaries/`` filenames (deterministic, collision-proof).
* **Manifest-filtered file tree** — the tree shown to the Manager contains only
  Worker-written files, never ``.agent/`` or ignored paths.
"""

from __future__ import annotations

import hashlib
import re
from fnmatch import fnmatch
from pathlib import Path

# The .agent/ control tree (redesign layout). Directories end with "/".
AGENT_DOCS = (
    "anchor.md",
    "tasks.json",
    "run.log",
    "architecture.md",
    "requirements.md",
    "project_brief.md",
    "blocked.md",
    "decisions.md",
    "test_results.jsonl",
    "final_report.md",
)
AGENT_SUBDIRS = ("summaries", "llm_calls/manager", "llm_calls/worker", "decompositions")

# Always ignored for tool access + file trees, regardless of .agentignore.
ALWAYS_IGNORED_DIRS = (".agent", ".venv", "node_modules", "__pycache__", ".git")

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SLUG_STOPWORDS = frozenset(
    "a an the with and or for of to in on that this using please build create make write app application".split()
)


class PathEscapeError(Exception):
    """Raised when a path or command would escape the project root."""


class AnchorImmutableError(Exception):
    """Raised on any attempt to overwrite ``.agent/anchor.md`` after creation."""


class IgnoredPathError(Exception):
    """Raised when a tool touches a path excluded by ``.agentignore``."""


def slugify(name: str, fallback: str = "project") -> str:
    """Turn an arbitrary string into a filesystem-safe slug."""
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    slug = slug[:60].strip("-")
    return slug or fallback


def _strip_dot_prefix(rel: str) -> str:
    """Remove leading ``./`` segments (NOT ``lstrip`` — that eats the dot of
    dotfiles like ``.venv`` and ``.agentignore``)."""
    while rel.startswith("./"):
        rel = rel[2:]
    return rel


def slug_from_prompt(prompt: str, fallback: str = "project") -> str:
    """Derive a short, deterministic project slug from the raw prompt.

    Programmatic (no LLM): the first few meaningful words, kebab-cased, plus a
    short hash of the whole prompt so two similar prompts don't collide."""
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", prompt or "")]
    meaningful = [w for w in words if w not in _SLUG_STOPWORDS][:4] or words[:4]
    base = slugify("-".join(meaningful), fallback)
    digest = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[:6]
    return f"{base}-{digest}"


def summary_slug(rel_path: str) -> str:
    """``src/api/routes.py`` -> ``src__api__routes_py_a3f9c1`` (no extension).

    Path separators become ``__``, dots become ``_``, and the first 6 hex chars
    of the sha256 of the ORIGINAL path guarantee uniqueness after sanitizing."""
    rel = _strip_dot_prefix((rel_path or "").strip().replace("\\", "/"))
    digest = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:6]
    sanitized = rel.replace("/", "__").replace(".", "_")
    sanitized = re.sub(r"[^A-Za-z0-9_\-]", "_", sanitized) or "file"
    return f"{sanitized}_{digest}"


class IgnoreMatcher:
    """Gitignore-lite matcher for ``.agentignore`` patterns.

    Supports blank lines, ``#`` comments, trailing-slash directory patterns,
    ``*`` globs, and bare names (matched at any depth). Always ignores
    :data:`ALWAYS_IGNORED_DIRS`.
    """

    def __init__(self, patterns: list[str] | None = None):
        self.patterns: list[str] = []
        for raw in patterns or []:
            line = str(raw).strip()
            if line and not line.startswith("#"):
                self.patterns.append(line.rstrip("/") if line.endswith("/") else line)

    def is_ignored(self, rel_path: str) -> bool:
        rel = _strip_dot_prefix((rel_path or "").replace("\\", "/"))
        if not rel:
            return False
        parts = rel.split("/")
        if any(p in ALWAYS_IGNORED_DIRS for p in parts):
            return True
        for pattern in self.patterns:
            if "/" in pattern:
                # Anchored path pattern: match the path itself or anything under it.
                if fnmatch(rel, pattern) or fnmatch(rel, pattern + "/*") or rel.startswith(pattern + "/"):
                    return True
            else:
                # Bare name/glob: match any path component or the full path.
                if any(fnmatch(part, pattern) for part in parts) or fnmatch(rel, pattern):
                    return True
        return False


class Workspace:
    """The resolved project directory plus helpers for the path jail."""

    def __init__(self, project_root: Path):
        self.root = Path(project_root).resolve()
        self.agent_dir = self.root / ".agent"
        self.summaries_dir = self.agent_dir / "summaries"
        self.llm_calls_dir = self.agent_dir / "llm_calls"
        self.decompositions_dir = self.agent_dir / "decompositions"
        self._ignore_cache: tuple[float, IgnoreMatcher] | None = None

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def resolve(cls, project_dir_cfg: str, slug: str | None, tool_root: Path) -> "Workspace":
        """Resolve the project root.

        1. If ``project_dir_cfg`` is set, use it (created if missing).
        2. Otherwise use ``<tool_root>/../sandbox/<slug>/`` — the ``sandbox/``
           directory next to the tool source dir, anchored on main.py's own
           location, never the CWD.
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
        """Create the project root and the full ``.agent/`` control tree."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        for sub in AGENT_SUBDIRS:
            (self.agent_dir / sub).mkdir(parents=True, exist_ok=True)
        return self

    # ── path validation ───────────────────────────────────────────────────────
    def resolve_in_root(self, path: str | Path) -> Path:
        """Resolve *path* relative to the root and assert it stays inside it."""
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

    def resolve_tool_path(self, path: str | Path) -> Path:
        """Path validation for the Worker's file tools.

        On top of the jail: rejects anything under ``.agent/`` (pipeline-internal
        state is invisible to the model) and anything matched by ``.agentignore``.
        """
        target = self.resolve_in_root(path)
        rel = target.relative_to(self.root).as_posix() if target != self.root else "."
        if rel == ".agent" or rel.startswith(".agent/"):
            raise IgnoredPathError(f"{rel!r} is pipeline-internal state and not accessible to tools")
        if self.ignore_matcher().is_ignored(rel):
            raise IgnoredPathError(f"{rel!r} is excluded by .agentignore")
        return target

    # ── .agentignore ──────────────────────────────────────────────────────────
    def agentignore_path(self) -> Path:
        return self.root / ".agentignore"

    def write_agentignore(self, patterns: list[str]) -> None:
        """Write ``.agentignore`` to the workspace root AND mirror it to .agent/.

        ``.venv/`` and ``node_modules/`` are always appended — the environment
        directories must never surface in any tree shown to a model."""
        seen: set[str] = set()
        lines: list[str] = []
        for raw in list(patterns) + [".venv/", "node_modules/", ".agent/", "__pycache__/"]:
            line = str(raw).strip()
            if line and line not in seen:
                seen.add(line)
                lines.append(line)
        body = "# Paths the pipeline hides from the model (gitignore-lite syntax).\n" + "\n".join(lines) + "\n"
        self.agentignore_path().write_text(body, encoding="utf-8")
        (self.agent_dir / ".agentignore").write_text(body, encoding="utf-8")
        self._ignore_cache = None

    def ignore_matcher(self) -> IgnoreMatcher:
        """Load (and mtime-cache) the current ``.agentignore`` matcher."""
        path = self.agentignore_path()
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = -1.0
        if self._ignore_cache is not None and self._ignore_cache[0] == mtime:
            return self._ignore_cache[1]
        patterns: list[str] = []
        if mtime >= 0:
            try:
                patterns = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                patterns = []
        matcher = IgnoreMatcher(patterns)
        self._ignore_cache = (mtime, matcher)
        return matcher

    # ── agent-doc IO ──────────────────────────────────────────────────────────
    def agent_path(self, name: str) -> Path:
        return self.agent_dir / name

    def write_agent_doc(self, name: str, content: str) -> Path:
        """Write an ``.agent/`` document. ``anchor.md`` is write-once: a second
        write raises :class:`AnchorImmutableError` — hard, never silent."""
        norm = str(name).strip().lstrip("./")
        path = self.agent_path(norm)
        if norm == "anchor.md" and path.exists():
            raise AnchorImmutableError(
                "anchor.md is immutable after creation — it is the locked system "
                "prompt for every Manager call and must never be rewritten"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def append_agent_doc(self, name: str, content: str) -> Path:
        """Append to an ``.agent/`` document (blocked.md, test_results.jsonl…)."""
        norm = str(name).strip().lstrip("./")
        if norm == "anchor.md":
            raise AnchorImmutableError("anchor.md is immutable after creation")
        path = self.agent_path(norm)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(content)
        return path

    def read_agent_doc(self, name: str, default: str | None = None) -> str | None:
        path = self.agent_path(name)
        if not path.exists():
            return default
        return path.read_text(encoding="utf-8")

    def read_anchor_text(self) -> str:
        """The anchor BODY (frontmatter stripped) — the Manager system prompt.

        The YAML frontmatter duplicates the body in ``combined_anchor`` for
        machine consumers; sending it too would double the anchor's tokens on
        every Manager call."""
        raw = self.read_agent_doc("anchor.md", "") or ""
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return raw.strip()

    def agent_doc_exists(self, name: str) -> bool:
        return self.agent_path(name).exists()

    # ── generic file IO (jailed, pipeline-internal use) ───────────────────────
    def read_file(self, path: str | Path) -> str:
        return self.resolve_in_root(path).read_text(encoding="utf-8")

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

    # ── file tree (Manager-facing) ────────────────────────────────────────────
    def file_tree(self, rel_paths: list[str]) -> str:
        """Render a manifest-filtered tree from *rel_paths* (Worker-written files).

        ``.agent/`` and ``.agentignore`` matches are excluded — the Manager only
        ever sees what the Worker has produced."""
        matcher = self.ignore_matcher()
        visible = sorted(
            {_strip_dot_prefix(p.replace("\\", "/")) for p in rel_paths if p}
            - {""}
        )
        visible = [p for p in visible if not p.startswith(".agent/") and not matcher.is_ignored(p)]
        if not visible:
            return "(no files written yet)"
        lines: list[str] = []
        last_dir: object = object()
        for rel in visible:
            parent = str(Path(rel).parent)
            if parent != last_dir:
                last_dir = parent
                if parent != ".":
                    lines.append(f"{parent}/")
            indent = "  " if parent != "." else ""
            lines.append(f"{indent}{Path(rel).name}")
        return "\n".join(lines)
