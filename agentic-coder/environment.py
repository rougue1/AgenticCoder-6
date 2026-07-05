"""Runtime environment setup — done by the ORCHESTRATOR, never by a model.

Phase 1 Step 3: after stack determination, the orchestrator prepares the
runtime environment for the detected stack.

* **Python** — create ``<project>/.venv`` with the best available interpreter
  (``.python-version`` file > Manager-preferred version > ``python3``), then
  validate it loudly: the venv python must exist and ``pip --version`` must
  succeed inside it. The sandbox is then pointed at the venv so every Worker
  command is transparently rewritten/environment-scoped.
* **Node** — verify node/npm exist; the sandbox prepends ``node_modules/.bin``
  to PATH so npx-less tool invocations work and npm stays project-scoped.

``.venv``/``node_modules`` are always excluded from trees via ``.agentignore``
(the workspace appends them unconditionally).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from stackprofiles import StackProfile

if TYPE_CHECKING:
    from server.events import EventBus
    from workspace import Workspace


class EnvironmentError_(RuntimeError):
    """Environment setup failed validation (venv broken, binaries missing…)."""


@dataclass
class EnvInfo:
    stack_profile: str
    venv_path: Path | None = None
    python_bin: Path | None = None
    node_root: Path | None = None
    notes: list[str] = field(default_factory=list)


_VERSION_RE = re.compile(r"^\s*(\d+)\.(\d+)")


def setup_environment(
    workspace: "Workspace",
    profile: StackProfile,
    bus: "EventBus",
    *,
    preferred_python: str = "",
) -> EnvInfo:
    from server import events

    bus.emit(events.ENVIRONMENT_SETUP_START, "environment", profile=profile.name)
    info = EnvInfo(stack_profile=profile.name)

    if profile.uses_venv:
        info.venv_path, info.python_bin = _setup_venv(workspace, bus, preferred_python)
        info.notes.append(f"venv at {info.venv_path} (python: {info.python_bin})")

    if profile.uses_node_modules:
        for binary in ("node", "npm"):
            if shutil.which(binary) is None:
                raise EnvironmentError_(f"node stack requires {binary!r}, which is not on PATH")
        info.node_root = workspace.root
        info.notes.append("node_modules/.bin will be prepended to PATH for project commands")

    bus.emit(events.ENVIRONMENT_SETUP_COMPLETE, "environment", profile=profile.name, notes=info.notes)
    return info


def verify_venv(workspace: "Workspace") -> bool:
    """Resume-hardening check: pyvenv.cfg present and the python binary runs."""
    venv = workspace.root / ".venv"
    py = venv / "bin" / "python"
    if not (venv / "pyvenv.cfg").is_file() or not py.exists():
        return False
    try:
        proc = subprocess.run(
            [str(py), "--version"], capture_output=True, text=True, timeout=20
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


# ── internals ─────────────────────────────────────────────────────────────────
def _setup_venv(workspace: "Workspace", bus: "EventBus", preferred: str) -> tuple[Path, Path]:
    venv = workspace.root / ".venv"
    py_bin = venv / "bin" / "python"

    if verify_venv(workspace):
        bus.log(f"existing venv at {venv} is intact — reusing it", phase="environment")
        return venv, py_bin

    if venv.exists():
        bus.log(f"venv at {venv} is broken — recreating", phase="environment", level="warn")
        shutil.rmtree(venv, ignore_errors=True)

    interpreter = _pick_interpreter(workspace, preferred, bus)
    try:
        proc = subprocess.run(
            [interpreter, "-m", "venv", str(venv)],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(workspace.root),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EnvironmentError_(f"venv creation failed to start ({interpreter} -m venv): {exc}") from exc
    if proc.returncode != 0:
        raise EnvironmentError_(
            f"venv creation failed (exit {proc.returncode}): {(proc.stderr or proc.stdout)[:500]}"
        )

    # Validate explicitly and loudly: binary exists, pip functions inside it.
    if not py_bin.exists():
        raise EnvironmentError_(f"venv python missing after creation: {py_bin}")
    try:
        pip_proc = subprocess.run(
            [str(py_bin), "-m", "pip", "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EnvironmentError_(f"pip validation inside the venv failed to run: {exc}") from exc
    if pip_proc.returncode != 0:
        raise EnvironmentError_(
            f"pip is not functional inside the venv (exit {pip_proc.returncode}): "
            f"{(pip_proc.stderr or pip_proc.stdout)[:500]}"
        )

    bus.log(f"created venv at {venv} ({pip_proc.stdout.strip()})", phase="environment")
    return venv, py_bin


def _pick_interpreter(workspace: "Workspace", preferred: str, bus: "EventBus") -> str:
    """Interpreter precedence: .python-version file > pyproject requires hint >
    the Manager's preferred version > plain python3."""
    candidates: list[str] = []

    pv = workspace.root / ".python-version"
    if pv.is_file():
        try:
            m = _VERSION_RE.match(pv.read_text(encoding="utf-8").strip())
            if m:
                candidates.append(f"python{m.group(1)}.{m.group(2)}")
        except OSError:
            pass

    pyproject = workspace.root / "pyproject.toml"
    if pyproject.is_file():
        try:
            m = re.search(r'requires-python\s*=\s*"[^\d]*(\d+)\.(\d+)', pyproject.read_text(encoding="utf-8"))
            if m:
                candidates.append(f"python{m.group(1)}.{m.group(2)}")
        except OSError:
            pass

    if preferred:
        m = _VERSION_RE.match(preferred.strip())
        if m:
            candidates.append(f"python{m.group(1)}.{m.group(2)}")

    candidates.append("python3")

    for cand in candidates:
        path = shutil.which(cand)
        if path:
            if cand != "python3":
                bus.log(f"using {cand} for the project venv", phase="environment")
            return path
    raise EnvironmentError_(
        f"no usable Python interpreter found (tried: {', '.join(candidates)})"
    )
