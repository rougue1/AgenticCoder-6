"""Pre-flight validation — runs after model resolution, before Phase 1.

Checks, in order:

1. the workspace directory exists and is writeable (probe file);
2. the Ollama service is reachable (``/api/tags`` — resolution already talked
   to it, this re-verifies right before the first pipeline call);
3. the default stack profile's required binaries are present (hard fail) and
   its optional binaries are present (warning only).

Every check emits ``preflight.check``; the pass emits ``preflight.passed`` or
``preflight.failed`` + :class:`PreflightError` with a descriptive message.
"""

from __future__ import annotations

import shutil
import uuid
from typing import TYPE_CHECKING

import httpx

from stackprofiles import StackProfile

if TYPE_CHECKING:
    from config import AppConfig
    from server.events import EventBus
    from workspace import Workspace


class PreflightError(RuntimeError):
    """A critical pre-flight dependency is missing/broken."""


def run_preflight(config: "AppConfig", workspace: "Workspace", profile: StackProfile, bus: "EventBus") -> None:
    from server import events

    failures: list[str] = []
    warnings: list[str] = []

    def check(name: str, ok: bool, detail: str, *, hard: bool = True) -> None:
        bus.emit(events.PREFLIGHT_CHECK, "preflight", name=name, ok=ok, detail=detail, hard=hard)
        if not ok:
            (failures if hard else warnings).append(f"{name}: {detail}")

    # 1. Workspace writability.
    ok, detail = _workspace_writeable(workspace)
    check("workspace_writeable", ok, detail)

    # 2. Ollama reachability.
    ok, detail = _ollama_reachable(config.ollama.host)
    check("ollama_reachable", ok, detail)

    # 3. Stack binaries.
    for binary in profile.required_binaries:
        path = shutil.which(binary)
        check(f"binary:{binary}", path is not None, path or f"{binary!r} not found on PATH")
    for binary in profile.optional_binaries:
        path = shutil.which(binary)
        check(f"binary:{binary}", path is not None, path or f"{binary!r} not found on PATH (soft)", hard=False)

    for warning in warnings:
        bus.log(f"pre-flight warning: {warning}", phase="preflight", level="warn")

    if failures:
        bus.emit(events.PREFLIGHT_FAILED, "preflight", failures=failures)
        raise PreflightError(
            "pre-flight validation failed:\n  - " + "\n  - ".join(failures)
        )
    bus.emit(events.PREFLIGHT_PASSED, "preflight", warnings=len(warnings))


def _workspace_writeable(workspace: "Workspace") -> tuple[bool, str]:
    probe = workspace.agent_dir / f".preflight-{uuid.uuid4().hex[:8]}"
    try:
        workspace.ensure()
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, str(workspace.root)
    except OSError as exc:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return False, f"cannot write to {workspace.root}: {exc}"


def _ollama_reachable(base: str) -> tuple[bool, str]:
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=10)
    except httpx.HTTPError as exc:
        return False, f"Ollama unreachable at {base}: {exc}. Is `ollama serve` running?"
    if resp.status_code != 200:
        return False, f"Ollama at {base} answered HTTP {resp.status_code}"
    return True, base
