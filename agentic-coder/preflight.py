"""Pre-flight validation — runs after model resolution, before Phase 1.

Checks, in order:

1. the workspace directory exists and is writeable (probe file);
2. **bubblewrap works**: ``bwrap`` must be on PATH *and* able to actually
   build a sandbox on this machine. Presence alone is not enough — Ubuntu
   23.10+ restricts unprivileged user namespaces via AppArmor, leaving bwrap
   installed but unusable; the functional probe catches that and the failure
   message carries the exact fix. bwrap is a HARD requirement: there is no
   allowlist fallback to degrade to.
3. the Ollama service is reachable (``/api/tags`` — resolution already talked
   to it, this re-verifies right before the first pipeline call);
4. the default stack profile's required binaries are present (hard fail) and
   its optional binaries are present (warning only). When the sandbox probe
   passed, binaries are resolved through the sandbox's login shell — the same
   way Worker commands will see them — so nvm/pyenv-managed tools count.

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
    from tools.sandbox import Sandbox
    from workspace import Workspace


class PreflightError(RuntimeError):
    """A critical pre-flight dependency is missing/broken."""


def run_preflight(
    config: "AppConfig",
    workspace: "Workspace",
    profile: StackProfile,
    bus: "EventBus",
    sandbox: "Sandbox | None" = None,
) -> None:
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

    # 2. bubblewrap: present AND functional (hard requirement, no fallback).
    sandbox_ok = False
    if sandbox is not None:
        sandbox_ok, detail = sandbox.probe()
        check("bwrap_functional", sandbox_ok, detail)
    else:
        path = shutil.which("bwrap")
        check(
            "bwrap_installed",
            path is not None,
            path
            or "bubblewrap (bwrap) is required for the OS-level sandbox but was not found on PATH "
            "— install it (Debian/Ubuntu: `sudo apt install bubblewrap`)",
        )

    # 3. Ollama reachability.
    ok, detail = _ollama_reachable(config.ollama.host)
    check("ollama_reachable", ok, detail)

    # 4. Stack binaries — resolved the way sandboxed commands will resolve them
    #    (login shell) when the sandbox works; host PATH otherwise.
    which = sandbox.which if (sandbox is not None and sandbox_ok) else shutil.which
    for binary in profile.required_binaries:
        path = which(binary)
        check(f"binary:{binary}", path is not None, path or f"{binary!r} not found in a login shell")
    for binary in profile.optional_binaries:
        path = which(binary)
        check(f"binary:{binary}", path is not None, path or f"{binary!r} not found in a login shell (soft)", hard=False)

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
