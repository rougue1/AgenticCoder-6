"""Feature 4 — post-execution formatting hooks.

After the Worker successfully writes or patches a file (``tools/registry.py``
``_write_file``/``_patch_file``), run the best available formatter for its
extension inside the bwrap sandbox, silently. Best effort only: an
unavailable formatter or a formatting failure never blocks the subtask — it
just logs a warning to run.log and the write/patch still succeeds.

Formatter selection is availability-first: the first candidate in the
extension's list whose binary resolves via ``Sandbox.which`` (the same
login-shell resolution used for nvm/pyenv-managed tools elsewhere in the
pipeline) is the one that runs. If that available formatter's run fails
(not installed after all, syntax error in the file, …), the failure is
logged and no further candidate is tried — the fallback chain is for
AVAILABILITY, not for retrying after a real failure.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.events import EventBus
    from tools.sandbox import Sandbox

# Extension -> ordered (label, shell command template) candidates. "{path}" is
# substituted with the shell-quoted relative path.
_FORMATTERS: dict[str, list[tuple[str, str]]] = {
    ".py": [
        ("ruff format", "ruff format {path}"),
        ("black", "black {path}"),
        ("autopep8", "autopep8 --in-place {path}"),
    ],
    ".ts": [("prettier", "prettier --write {path}"), ("eslint --fix", "eslint --fix {path}")],
    ".tsx": [("prettier", "prettier --write {path}"), ("eslint --fix", "eslint --fix {path}")],
    ".js": [("prettier", "prettier --write {path}"), ("eslint --fix", "eslint --fix {path}")],
    ".jsx": [("prettier", "prettier --write {path}"), ("eslint --fix", "eslint --fix {path}")],
    ".go": [("gofmt", "gofmt -w {path}")],
    ".rs": [("rustfmt", "rustfmt {path}")],
    ".json": [("prettier", "prettier --write {path}")],
    ".md": [("prettier", "prettier --write {path}")],
    ".css": [("prettier", "prettier --write {path}")],
    ".scss": [("prettier", "prettier --write {path}")],
}

# Formatter label -> the binary Sandbox.which should resolve for it.
_BINARY_OF: dict[str, str] = {
    "ruff format": "ruff",
    "black": "black",
    "autopep8": "autopep8",
    "prettier": "prettier",
    "eslint --fix": "eslint",
    "gofmt": "gofmt",
    "rustfmt": "rustfmt",
}


def format_file(sandbox: "Sandbox", rel_path: str, bus: "EventBus | None" = None, phase: str = "") -> None:
    """Best-effort auto-format *rel_path* in place. Never raises."""
    candidates = _FORMATTERS.get(_ext_of(rel_path))
    if not candidates:
        return

    quoted = shlex.quote(rel_path)
    for label, template in candidates:
        binary = _BINARY_OF.get(label, label.split()[0])
        try:
            available = sandbox.which(binary) is not None
        except Exception:
            available = False
        if not available:
            continue  # try the next fallback candidate

        cmd = template.format(path=quoted)
        try:
            result = sandbox.run(cmd, validate=False)
        except Exception as exc:  # pragma: no cover - defensive; sandbox.run doesn't raise
            _warn(bus, phase, f"formatter {label!r} crashed on {rel_path}: {exc}")
            return

        if result.ok:
            if bus is not None:
                from server import events

                bus.emit(events.FORMATTER_RUN, phase or "formatter", path=rel_path, formatter=label)
            return

        _warn(
            bus,
            phase,
            f"formatter {label!r} failed on {rel_path} (exit {result.exit_code}): "
            f"{(result.stderr or result.stdout or '').strip()[:300]}",
        )
        return  # the fallback chain is for availability, not for a real failure

    # No candidate formatter was available on PATH: skip silently (no run.log spam).


def _warn(bus: "EventBus | None", phase: str, message: str) -> None:
    if bus is not None:
        bus.log(message, phase=phase or "formatter", level="warn")


def _ext_of(rel_path: str) -> str:
    name = (rel_path or "").replace("\\", "/").rsplit("/", 1)[-1]
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()
