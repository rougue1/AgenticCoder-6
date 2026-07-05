"""Long-running process harness (spec §13).

Servers/watchers don't fit a single pass/fail exit code. When ``run`` is invoked
with ``background: true`` this harness:

1. starts the process detached (its own process group), capturing output;
2. polls for health up to ``long_process_timeout`` (a port opening, an expected
   log line, or simply staying alive);
3. runs any smoke commands the plan specified against it;
4. **kills the process group** (guaranteed cleanup, even on error/timeout) and
   reduces everything to a pass/fail :class:`CommandResult`.
"""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tools.sandbox import CommandResult, Sandbox, _truncate
from workspace import Workspace

if TYPE_CHECKING:
    from server.events import EventBus

# Substrings that commonly indicate a dev server became ready.
_READY_HINTS = (
    "listening", "running on", "running at", "started server", "server started",
    "ready in", "ready on", "compiled successfully", "local:", "now listening",
    "serving", "development server", "uvicorn running", "application startup complete",
)
_PORT_RE = re.compile(r"(?:--port[=\s]+|:|\bport\s+|PORT[=\s]+)(\d{2,5})", re.I)


@dataclass
class ProcessHandle:
    proc: subprocess.Popen
    cmd: str
    log_path: str
    port: int | None = None
    started: float = field(default_factory=time.monotonic)

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None


class ProcessManager:
    def __init__(self, workspace: Workspace, sandbox: Sandbox, bus: "EventBus | None" = None):
        self.workspace = workspace
        self.sandbox = sandbox
        self.bus = bus
        self._live: list[ProcessHandle] = []

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self, cmd: str, *, allow_network: bool | None = None) -> ProcessHandle:
        if allow_network is None:
            allow_network = self.sandbox.is_install_command(cmd)
        env = self.sandbox.build_env(allow_network)
        exec_cmd = self.sandbox.rewrite_command(cmd)  # venv transparency, same as foreground
        log_fh = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".log", prefix="aiforge-proc-", delete=False, encoding="utf-8"
        )
        proc = subprocess.Popen(
            exec_cmd,
            shell=True,
            cwd=str(self.workspace.root),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        handle = ProcessHandle(proc=proc, cmd=cmd, log_path=log_fh.name, port=_extract_port(cmd))
        self._live.append(handle)
        return handle

    def stop(self, handle: ProcessHandle) -> None:
        """Kill the whole process group; never raise."""
        proc = handle.proc
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        if handle in self._live:
            self._live.remove(handle)

    def stop_all(self) -> None:
        for handle in list(self._live):
            self.stop(handle)

    # ── health + smoke ────────────────────────────────────────────────────────
    def wait_healthy(self, handle: ProcessHandle, timeout: int) -> tuple[bool, str]:
        """Return (healthy, detail). Healthy if a port opens, a ready line is
        logged, or the process simply survives the grace window."""
        deadline = time.monotonic() + max(1, timeout)
        while time.monotonic() < deadline:
            if not handle.alive:
                rc = handle.proc.returncode
                return (rc == 0, f"process exited early with code {rc}")
            if handle.port and _port_open(handle.port):
                return True, f"port {handle.port} open"
            log = _read_log(handle.log_path).lower()
            if any(hint in log for hint in _READY_HINTS):
                return True, "ready log line detected"
            time.sleep(0.4)
        # Survived the whole window without exiting -> treat as healthy.
        if handle.alive:
            return True, f"alive after {timeout}s grace"
        rc = handle.proc.returncode
        return (rc == 0, f"process exited with code {rc}")

    def run_background_check(
        self,
        cmd: str,
        *,
        health_timeout: int,
        smoke_cmds: list[str] | None = None,
        smoke_timeout: int = 30,
    ) -> CommandResult:
        """Full start -> health -> smoke -> stop cycle, reduced to pass/fail."""
        start = time.monotonic()
        handle = self.start(cmd)
        passed = False
        detail = ""
        smoke_out = ""
        try:
            healthy, detail = self.wait_healthy(handle, health_timeout)
            if healthy and smoke_cmds:
                all_ok = True
                for sc in smoke_cmds:
                    res = self.sandbox.run(sc, timeout=smoke_timeout)
                    smoke_out += f"$ {sc}\n[exit {res.exit_code}] {res.stdout}\n{res.stderr}\n"
                    if not res.ok:
                        all_ok = False
                        break
                passed = all_ok
            else:
                passed = healthy
        finally:
            self.stop(handle)

        log_tail = _read_log(handle.log_path)
        duration = time.monotonic() - start
        return CommandResult(
            exit_code=0 if passed else 1,
            stdout=_truncate(log_tail + ("\n" + smoke_out if smoke_out else "")),
            stderr="" if passed else f"background check failed: {detail}",
            duration=duration,
            reason=detail,
        )


def _extract_port(cmd: str) -> int | None:
    m = _PORT_RE.search(cmd)
    if m:
        try:
            port = int(m.group(1))
            return port if 1 <= port <= 65535 else None
        except ValueError:
            return None
    return None


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        try:
            return sock.connect_ex((host, port)) == 0
        except OSError:
            return False


def _read_log(path: str, limit: int = 8000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()[-limit:]
    except OSError:
        return ""
