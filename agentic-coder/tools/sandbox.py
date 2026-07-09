"""OS-level command sandbox built on bubblewrap (bwrap) — Linux only.

This replaces the old command-allowlist sandbox entirely. Security is enforced
at the filesystem/process level by what a command can SEE and WRITE, not by
which executables it is allowed to type: the old allowlist broke stack
agnosticism (every unknown tool died with exit 126 and the Worker looped), and
the old bare ``subprocess`` never sourced the user's shell environment, so
nvm/fnm/volta/pyenv-managed tools were invisible.

ARCHITECTURAL DECISION — per-command bwrap invocations, NOT a persistent
bwrap session with a long-lived shell inside:

* **Robust failure domains.** Every command gets a fresh sandbox; a command
  that wedges, corrupts its shell state, or dies cannot poison the next one.
  A persistent inner shell needs sentinel-based output/exit-code framing,
  which breaks on binary output, interleaved writes, or a killed shell.
* **Parallelism for free.** N concurrent commands are just N bwrap processes.
  Background sessions are unawaited invocations of the exact same code path —
  no multiplexing protocol, no session daemon to babysit.
* **Precise lifecycle.** Each invocation carries ``--die-with-parent`` (tied
  to the orchestrator's life) and its own host-side process group; killing one
  session cannot disturb another. With ``--unshare-pid`` the command is PID 1
  of its own namespace, so the kernel reaps every descendant when it dies.
* **Negligible cost.** bwrap setup is ~10ms and the login shell ~100ms —
  noise against pip/npm/pytest runtimes, and nothing against the multi-minute
  LLM turns on this box.

The accepted tradeoff: no shell state persists between commands (``cd``,
exports, ``/tmp`` contents) — the workspace filesystem is the only durable
channel, which matches the pipeline's "durable state lives on disk" rule.
Chained commands (``cd x && …``) still work within a single invocation, and
package caches persist via ``XDG_CACHE_HOME`` pointed inside ``.agent/``.

The bwrap profile (see :meth:`Sandbox.build_bwrap_args`):

* read-only binds of the host system (``/usr``, ``/etc``, ``/opt``, ``/snap``,
  the ``/bin``-style merged-usr symlinks) and of ``/home`` (workspace reads);
* the project workspace is the ONLY writable path (rw-bound last, so it wins
  over the read-only ``/home`` even when nested inside it);
* sensitive directories (``~/.ssh``, ``~/.aws``, …) are masked with tmpfs
  (dirs) or a ``/dev/null`` bind (files) so their contents cannot be read;
* fresh ``/proc`` + ``/dev``, private PID namespace, tmpfs ``/tmp`` and
  ``/run`` (with ``/run/systemd/resolve`` re-bound so the systemd-resolved
  ``/etc/resolv.conf`` symlink keeps resolving DNS);
* the host NETWORK namespace is shared (no ``--unshare-net``): package
  installs and testing against background servers need it;
* ``--clearenv`` + explicit ``--setenv`` of a filtered environment (secrets
  never cross the boundary), and commands run under ``bash -l -c`` so the
  user's login environment (version managers included) is sourced;
* bwrap always sets ``no_new_privs``, so sudo/setuid escalation is impossible
  inside regardless of the deny-list.

A small two-category deny-list still runs BEFORE bwrap, purely for fast,
actionable errors: category 1 (always blocked) covers destructive commands and
**all git** (a hard project rule); category 2 (blocked in FOREGROUND only)
covers dev-server commands that would hang a blocking call — the Worker is
told to start them as background sessions instead.

Background sessions: ``run_background`` starts a command inside the same
bwrap profile and returns a UUID session id immediately (after a short grace
poll so instant crashes — port conflicts, bad flags — surface their output
right away). Output streams to ``.agent/sessions/<id>.log``;
``check_session``/``stop_session`` poll and kill; ``terminate_sessions`` is
invoked at every subtask boundary and on orchestrator shutdown, with an
``atexit`` hook as the last resort below ``--die-with-parent``.
"""

from __future__ import annotations

import atexit
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from workspace import Workspace

if TYPE_CHECKING:  # avoid a hard import cycle / optional at runtime
    from server.events import EventBus

# ── category 1: always blocked, in every mode ──────────────────────────────────
# The OS sandbox already makes most of these harmless (no write access outside
# the workspace, no_new_privs kills sudo/setuid); rejecting them up front just
# turns a confusing kernel error into an instant, explicit one.
_DENY_ALWAYS: list[tuple[str, re.Pattern[str]]] = [
    ("git is forbidden", re.compile(r"(?:^|[\s;&|`(])git(?:\s|$)", re.I)),
    ("recursive root delete", re.compile(r"\brm\s+(?:-[a-z]*\s+)*-?[a-z]*r[a-z]*f?[a-z]*\s+(?:/|/\s|~)", re.I)),
    ("rm of / or ~", re.compile(r"\brm\b[^\n]*\s(/|~)(\s|$)", re.I)),
    ("sudo escalation", re.compile(r"(?:^|[\s;&|`(])sudo(?:\s|$)", re.I)),
    ("disk dd", re.compile(r"(?:^|[\s;&|`(])dd(?:\s|$)", re.I)),
    ("mkfs", re.compile(r"\bmkfs\S*\b", re.I)),
    ("fork bomb", re.compile(r":\s*\(\s*\)\s*\{")),
    ("shutdown", re.compile(r"(?:^|[\s;&|`(])shutdown(?:\s|$)", re.I)),
    ("reboot/halt/poweroff", re.compile(r"(?:^|[\s;&|`(])(?:reboot|halt|poweroff)(?:\s|$)", re.I)),
    ("systemctl power control", re.compile(r"\bsystemctl\s+(?:reboot|poweroff|halt|suspend|hibernate)\b", re.I)),
    ("chmod 777 of root", re.compile(r"\bchmod\s+(?:-R\s+)?777\s+/", re.I)),
    ("pipe-to-shell install", re.compile(r"\b(?:curl|wget)\b[^\n]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|python\d?)\b", re.I)),
    ("mv/cp to root", re.compile(r"\b(?:mv|cp)\b[^\n]*\s/(\s|$)", re.I)),
]

# ── category 2: blocked in FOREGROUND only (would hang a blocking call) ────────
# Matching is token-based per command segment (never substring), so
# `echo uvicorn` or `pytest test_vite.py` can never false-positive.
_SERVER_SCRIPT_NAMES = {"start", "dev", "serve", "preview", "watch"}
_PY_SERVER_SCRIPTS = {"app.py", "server.py", "wsgi.py", "asgi.py"}
_ALWAYS_SERVER_EXES = {
    "uvicorn", "gunicorn", "nodemon", "http-server", "live-server", "serve",
    "webpack-dev-server", "flask-dev",
}
_FOREGROUND_BLOCK_MSG = (
    "This command starts a long-running process. Use background mode to start it, "
    'then test against it. (matched: {label} — re-run it with "background": true, '
    "then hit it with foreground commands like curl and inspect it with check_session)"
)

_PYTHON_RE = re.compile(r"^python(?:3(?:\.\d+)?)?$")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_WRAPPER_CMDS = {"env", "nice", "time", "exec", "nohup", "stdbuf"}

# ── environment filtering (what crosses into the sandbox) ──────────────────────
_ENV_ALLOW_EXACT = {"HOME", "USER", "SHELL", "TERM", "LANG", "LC_ALL", "PATH"}
_ENV_ALLOW_PREFIX = (
    "OLLAMA_", "NVM_", "VIRTUAL_ENV", "PYTHONPATH", "NODE_PATH", "npm_config_",
    "CARGO_", "GOPATH", "GOROOT", "RUSTUP_", "LC_",
)
_ENV_DENY_PREFIX = ("AWS_", "GITHUB_", "GITLAB_", "DOCKER_")
_ENV_DENY_EXACT = {"KUBECONFIG"}
_ENV_DENY_SUBSTRING = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "KEY")

# Sensitive paths (relative to $HOME) masked inside the sandbox. Only added
# when they exist on the host — bwrap fails on mount points it cannot create.
_SENSITIVE_HOME_PATHS = (".ssh", ".aws", ".gnupg", ".kube", ".docker", ".config/gh", ".netrc")

# Output captured from a command is truncated to keep events/conversations sane.
_MAX_CAPTURE = 60_000

_BWRAP_MISSING_MSG = (
    "bubblewrap (bwrap) is required for the OS-level sandbox but was not found on "
    "PATH. Install it and re-run (Debian/Ubuntu: `sudo apt install bubblewrap`; "
    "Fedora: `sudo dnf install bubblewrap`; Arch: `sudo pacman -S bubblewrap`)."
)

# Ubuntu 23.10+ ships kernel.apparmor_restrict_unprivileged_userns=1: bwrap can
# create the user namespace but is denied the uid-map write unless an AppArmor
# profile grants it `userns`. The probe error is cryptic, so spell out the fix.
_USERNS_HINT = (
    "bwrap is installed but cannot create its sandbox (unprivileged user "
    "namespaces are restricted — on Ubuntu 23.10+ this is "
    "kernel.apparmor_restrict_unprivileged_userns=1). Fix it once with an "
    "AppArmor profile for bwrap:\n"
    "  sudo tee /etc/apparmor.d/bwrap <<'EOF'\n"
    "  abi <abi/4.0>,\n"
    "  include <tunables/global>\n"
    "  profile bwrap /usr/bin/bwrap flags=(unconfined) {\n"
    "    userns,\n"
    "    include if exists <local/bwrap>\n"
    "  }\n"
    "  EOF\n"
    "  sudo apparmor_parser -r /etc/apparmor.d/bwrap\n"
    "(or, less targeted: `sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0`)"
)


@dataclass
class CommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    timed_out: bool = False
    rejected: bool = False
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.rejected and not self.timed_out

    def to_dict(self) -> dict:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration": round(self.duration, 3),
            "timed_out": self.timed_out,
            "rejected": self.rejected,
            "reason": self.reason,
        }


@dataclass
class SessionStart:
    """Result of ``run_background``: the session id plus its startup fate."""

    session_id: str = ""
    rejected: bool = False
    reason: str = ""
    running: bool = False
    exit_code: int | None = None
    output: str = ""     # log content captured during the startup grace window
    log_path: str = ""   # workspace-relative, for humans reading .agent/

    @property
    def ok(self) -> bool:
        return not self.rejected and bool(self.session_id) and (self.running or self.exit_code == 0)


@dataclass
class SessionStatus:
    """Snapshot of one background session for ``check_session``/``stop_session``."""

    session_id: str
    exists: bool = False
    running: bool = False
    exit_code: int | None = None
    output: str = ""
    cmd: str = ""
    uptime: float = 0.0
    detail: str = ""


@dataclass
class Session:
    id: str
    cmd: str
    proc: subprocess.Popen
    log_path: Path
    started: float = field(default_factory=time.monotonic)

    @property
    def running(self) -> bool:
        return self.proc.poll() is None


class CommandRejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Sandbox:
    def __init__(self, workspace: Workspace, config, bus: "EventBus | None" = None):
        """*config* is the AppConfig ``sandbox`` section (SandboxCfg)."""
        self.workspace = workspace
        self.limits = config  # .timeout / .background_grace / .session_tail_lines
        self.bus = bus
        self.venv_path: Path | None = None
        # bwrap is a HARD requirement (no allowlist fallback exists any more).
        # Missing/broken bwrap is reported by the pre-flight probe with install
        # instructions; any run() before that still fails loudly, never silently.
        self.bwrap_path: str | None = shutil.which("bwrap")
        self.home_dir = Path(os.environ.get("HOME") or Path.home())
        self._sessions: dict[str, Session] = {}
        self._sessions_lock = threading.Lock()
        self._atexit_registered = False

    # ── configuration (orchestrator-side) ─────────────────────────────────────
    def set_venv(self, venv_path: str | Path | None) -> None:
        self.venv_path = Path(venv_path) if venv_path else None

    # ── validation: the two-category deny-list ─────────────────────────────────
    def validate(self, cmd: str, *, background: bool = False) -> None:
        """Raise :class:`CommandRejected` if *cmd* hits the deny-list.

        Category 1 (destructive + git) is blocked in every mode; category 2
        (dev servers/watchers) only when *background* is False — in background
        mode those are exactly the commands this sandbox exists to run.
        """
        if not cmd or not cmd.strip():
            raise CommandRejected("empty command")

        for reason, pattern in _DENY_ALWAYS:
            if pattern.search(cmd):
                self._emit_rejected(cmd, reason)
                raise CommandRejected(reason)

        if not background:
            label = _blocking_server_intent(cmd)
            if label:
                reason = _FOREGROUND_BLOCK_MSG.format(label=label)
                self._emit_rejected(cmd, reason)
                raise CommandRejected(reason)

    def _emit_rejected(self, cmd: str, reason: str) -> None:
        if self.bus is not None:
            from server import events

            self.bus.emit(events.SANDBOX_COMMAND_REJECTED, "sandbox", cmd=cmd[:400], reason=reason)

    # ── environment (filtered host env + sandbox-specific overrides) ───────────
    def sandbox_env(self) -> dict[str, str]:
        """The full environment set inside the sandbox via ``--setenv``."""
        env = {
            # Non-interactive/deterministic package-manager behavior.
            "CI": "1",
            "PYTHONUNBUFFERED": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "npm_config_yes": "true",
            # No .pyc caching: a fix rewriting a module within the same second
            # as the previous test run would otherwise be masked by a stale
            # __pycache__ entry (mtime granularity), re-failing a correct fix.
            "PYTHONDONTWRITEBYTECODE": "1",
            "TERM": "dumb",
        }
        env.update(filter_environment(os.environ))
        env.setdefault("HOME", str(self.home_dir))
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
        # $HOME is read-only inside the sandbox, which would break pip/npm
        # caches (~/.cache, ~/.npm). Point the XDG cache (pip honors it) and
        # the npm cache at a writable dir inside the jail that persists across
        # per-command sandboxes — installs stay fast and nothing escapes.
        cache = self.workspace.agent_dir / "cache"
        try:
            (cache / "npm").mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        env["XDG_CACHE_HOME"] = str(cache)
        env["npm_config_cache"] = str(cache / "npm")
        if self.venv_path is not None:
            env["VIRTUAL_ENV"] = str(self.venv_path)
            env["PATH"] = str(self.venv_path / "bin") + os.pathsep + env["PATH"]
            env.pop("PYTHONHOME", None)
        return env

    # ── bwrap invocation construction ──────────────────────────────────────────
    def build_bwrap_args(self) -> list[str]:
        """The bwrap argv prefix (everything before the command to execute).

        Mount order matters: bwrap applies operations in argv order and later
        mounts override earlier ones, so the read-write workspace bind comes
        LAST — after the read-only ``/home`` and after ``--tmpfs /tmp`` — and
        wins wherever the workspace actually lives.
        """
        if self.bwrap_path is None:
            raise CommandRejected(_BWRAP_MISSING_MSG)
        root = str(self.workspace.root)
        args: list[str] = [
            self.bwrap_path,
            "--die-with-parent",  # sandbox dies with the orchestrator
            "--new-session",      # no controlling-terminal takeover (setsid)
            "--unshare-pid",      # own PID ns: fresh /proc, kernel reaps the tree
            "--proc", "/proc",
            "--dev", "/dev",
        ]
        # Read-only system mounts (existence-checked; merged-usr distros make
        # /bin, /lib, … symlinks into /usr — recreate the symlink, don't bind).
        for p in ("/usr", "/etc", "/opt", "/snap"):
            if os.path.isdir(p) and not os.path.islink(p):
                args += ["--ro-bind", p, p]
        for p in ("/bin", "/sbin", "/lib", "/lib64", "/lib32"):
            if os.path.islink(p):
                args += ["--symlink", os.readlink(p), p]
            elif os.path.isdir(p):
                args += ["--ro-bind", p, p]
        # Writable scratch space.
        args += ["--tmpfs", "/tmp"]
        args += ["--tmpfs", "/run"]
        if os.path.isdir("/run/systemd/resolve"):
            # /etc/resolv.conf is a symlink into /run on systemd-resolved
            # systems; without this re-bind, DNS dies inside the sandbox.
            args += ["--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve"]
        uid = os.getuid()
        args += ["--dir", "/run/user", "--perms", "0700", "--dir", f"/run/user/{uid}"]
        # Home: readable (the agent may read the workspace's surroundings and
        # its own dotfiles for the login shell) but never writable.
        if os.path.isdir("/home"):
            args += ["--ro-bind", "/home", "/home"]
        home = self.home_dir
        if home.is_dir() and not _is_under(home, Path("/home")):
            args += ["--ro-bind", str(home), str(home)]  # e.g. root's /root
        # Mask sensitive paths (tmpfs for dirs, /dev/null for files) — only
        # those that exist, checked on the host.
        for rel in _SENSITIVE_HOME_PATHS:
            p = home / rel
            if p.is_dir():
                args += ["--tmpfs", str(p)]
            elif p.is_file():
                args += ["--ro-bind", "/dev/null", str(p)]
        # The workspace: the ONLY writable path. Bound last so it overrides the
        # read-only /home (or the /tmp tmpfs, when tests place it there).
        args += ["--bind", root, root, "--chdir", root]
        # Environment: start from nothing, then set exactly the filtered set.
        args += ["--clearenv"]
        for key, value in sorted(self.sandbox_env().items()):
            args += ["--setenv", key, value]
        return args

    def build_command_argv(self, cmd: str) -> list[str]:
        """Full argv: bwrap profile + ``bash -l -c`` around *cmd*.

        The login shell sources the user's profile chain, which is what makes
        nvm/pyenv/volta-installed tools resolvable. Two PATH safeguards ride in
        a prelude prepended to the command itself, because Debian-style
        ``/etc/profile`` RESETS ``PATH`` and would clobber anything passed via
        ``--setenv`` alone:

        1. the host PATH (the environment the tool was launched from, which
           already contains the user's version-manager paths) is re-prepended
           after profile sourcing;
        2. when a venv exists, its bin dir is prepended in front of everything
           (it is also in the ``--setenv`` PATH per the venv-transparency
           contract — the prelude just makes it profile-proof).
        """
        env = self.sandbox_env()
        prelude = ["export PATH=" + shlex.quote(env["PATH"]) + '"${PATH:+:$PATH}"']
        if self.venv_path is not None:
            prelude += [
                "export VIRTUAL_ENV=" + shlex.quote(str(self.venv_path)),
                'export PATH="$VIRTUAL_ENV/bin:$PATH"',
                "unset PYTHONHOME",
            ]
        shell_cmd = "\n".join(prelude + [cmd])
        return self.build_bwrap_args() + ["/bin/bash", "-l", "-c", shell_cmd]

    # ── pre-flight probe ───────────────────────────────────────────────────────
    def probe(self) -> tuple[bool, str]:
        """Can bwrap actually build this sandbox on this machine?

        A mere ``which bwrap`` is not enough: Ubuntu 23.10+ restricts
        unprivileged user namespaces via AppArmor, in which case bwrap exists
        but every invocation dies with ``setting up uid map: Permission
        denied``. Returns ``(ok, detail)`` with actionable instructions.
        """
        if self.bwrap_path is None:
            return False, _BWRAP_MISSING_MSG
        res = self.run("true", timeout=30, validate=False)
        if res.ok:
            return True, self.bwrap_path
        detail = (res.stderr or res.stdout or "").strip()
        lowered = detail.lower()
        if "uid map" in lowered or "user namespace" in lowered or "permission denied" in lowered:
            return False, f"{detail}\n{_USERNS_HINT}"
        return False, f"bwrap probe failed (exit {res.exit_code}): {detail or res.reason}"

    def which(self, binary: str) -> str | None:
        """Resolve *binary* the way sandboxed commands will — through the login
        shell — so version-manager-installed tools count. Falls back to the
        host PATH when bwrap is unusable (pre-flight reports that separately)."""
        if self.bwrap_path is None:
            return shutil.which(binary)
        res = self.run(f"command -v -- {shlex.quote(binary)}", timeout=30, validate=False)
        if not res.ok:
            return None
        # Profile scripts may echo noise before the answer; take the last
        # absolute path printed.
        for line in reversed(res.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("/"):
                return line
        return None

    # ── foreground execution ───────────────────────────────────────────────────
    def run(
        self,
        cmd: str,
        timeout: int | None = None,
        *,
        validate: bool = True,
    ) -> CommandResult:
        """Validate and run *cmd* to completion inside the bwrap sandbox."""
        if validate:
            try:
                self.validate(cmd, background=False)
            except CommandRejected as exc:
                return CommandResult(exit_code=126, rejected=True, reason=exc.reason, stderr=exc.reason)

        try:
            argv = self.build_command_argv(cmd)
        except CommandRejected as exc:  # bwrap missing — hard requirement, loud failure
            return CommandResult(exit_code=127, rejected=True, reason=exc.reason, stderr=exc.reason)

        timeout = max(1, int(timeout or self.limits.timeout))
        start = time.monotonic()
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(self.workspace.root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            return CommandResult(exit_code=127, stderr=f"failed to start bwrap: {exc}", reason=str(exc))

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return CommandResult(
                exit_code=proc.returncode,
                stdout=_truncate(stdout),
                stderr=_truncate(stderr),
                duration=time.monotonic() - start,
            )
        except subprocess.TimeoutExpired:
            _kill_proc_tree(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
            if self.bus is not None:
                from server import events

                self.bus.emit(events.SANDBOX_TIMEOUT, "sandbox", cmd=cmd[:400], timeout=timeout)
            return CommandResult(
                exit_code=124,
                stdout=_truncate(stdout or ""),
                stderr=_truncate((stderr or "") + f"\n[timeout after {timeout}s]"),
                duration=time.monotonic() - start,
                timed_out=True,
                reason=f"timeout after {timeout}s",
            )

    # ── background sessions ────────────────────────────────────────────────────
    def run_background(self, cmd: str, *, validate: bool = True) -> SessionStart:
        """Start *cmd* as a background session; return its id immediately.

        A short grace poll (``limits.background_grace`` seconds) catches
        instant deaths — port conflicts, bad flags, missing modules — so their
        output comes back on THIS call and the Worker can self-correct without
        a follow-up check_session.
        """
        if validate:
            try:
                self.validate(cmd, background=True)
            except CommandRejected as exc:
                return SessionStart(rejected=True, reason=exc.reason, output=exc.reason)

        try:
            argv = self.build_command_argv(cmd)
        except CommandRejected as exc:
            return SessionStart(rejected=True, reason=exc.reason, output=exc.reason)

        sessions_dir = self.workspace.agent_dir / "sessions"
        session_id = str(uuid.uuid4())
        log_path = sessions_dir / f"{session_id}.log"
        try:
            sessions_dir.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, "wb")
        except OSError as exc:
            return SessionStart(reason=f"could not open session log: {exc}", output=str(exc))

        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(self.workspace.root),
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            log_fh.close()
            return SessionStart(reason=f"failed to start bwrap: {exc}", output=str(exc))
        finally:
            # The child inherited its own copy of the fd at spawn time; the
            # parent-side handle is not needed for capture.
            if not log_fh.closed:
                log_fh.close()

        session = Session(id=session_id, cmd=cmd, proc=proc, log_path=log_path)
        with self._sessions_lock:
            self._sessions[session_id] = session
        self._register_atexit()
        if self.bus is not None:
            self.bus.log(f"background session {session_id[:8]}… started: {cmd[:160]}", phase="sandbox")

        deadline = time.monotonic() + max(0.0, float(self.limits.background_grace))
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.05)

        rel_log = f".agent/sessions/{session_id}.log"
        rc = proc.poll()
        tail = self._read_log_tail(log_path)
        if rc is not None:
            return SessionStart(
                session_id=session_id, running=False, exit_code=rc, output=tail, log_path=rel_log
            )
        return SessionStart(session_id=session_id, running=True, output=tail, log_path=rel_log)

    def check_session(self, session_id: str) -> SessionStatus:
        """Status + captured output of a background session (running or dead)."""
        session = self._get_session(session_id)
        if session is None:
            return SessionStatus(
                session_id=str(session_id),
                exists=False,
                detail=self._unknown_session_detail(session_id),
            )
        rc = session.proc.poll()
        return SessionStatus(
            session_id=session.id,
            exists=True,
            running=rc is None,
            exit_code=rc,
            output=self._read_log_tail(session.log_path),
            cmd=session.cmd,
            uptime=time.monotonic() - session.started,
        )

    def stop_session(self, session_id: str) -> SessionStatus:
        """Terminate one background session; returns its final status."""
        session = self._get_session(session_id)
        if session is None:
            return SessionStatus(
                session_id=str(session_id),
                exists=False,
                detail=self._unknown_session_detail(session_id),
            )
        self._terminate(session)
        status = self.check_session(session_id)
        status.detail = "terminated"
        if self.bus is not None:
            self.bus.log(f"background session {session.id[:8]}… stopped", phase="sandbox")
        return status

    def terminate_sessions(self, reason: str = "") -> int:
        """Kill every tracked session (subtask boundary / shutdown). Returns
        how many were still running. The dict is cleared: session ids never
        outlive the subtask that created them."""
        with self._sessions_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        killed = 0
        for session in sessions:
            if session.running:
                killed += 1
            self._terminate(session)
        if killed and self.bus is not None:
            suffix = f" ({reason})" if reason else ""
            self.bus.log(f"terminated {killed} background session(s){suffix}", phase="sandbox")
        return killed

    @property
    def active_sessions(self) -> list[str]:
        with self._sessions_lock:
            return [s.id for s in self._sessions.values() if s.running]

    # ── session internals ──────────────────────────────────────────────────────
    def _get_session(self, session_id: str) -> Session | None:
        with self._sessions_lock:
            return self._sessions.get(str(session_id).strip())

    def _unknown_session_detail(self, session_id: str) -> str:
        with self._sessions_lock:
            known = [s.id for s in self._sessions.values()]
        listing = ", ".join(k[:8] + "…" for k in known) if known else "(none)"
        return (
            f"unknown session id {str(session_id)[:64]!r} — sessions do not survive "
            f"subtask or escalation boundaries. Active sessions: {listing}"
        )

    @staticmethod
    def _terminate(session: Session) -> None:
        """SIGTERM bwrap, then SIGKILL. Killing bwrap is enough: with
        ``--die-with-parent`` the sandboxed command holds PDEATHSIG(SIGKILL)
        against bwrap, and as PID 1 of its namespace its death reaps every
        descendant. The killpg sweep is belt-and-suspenders."""
        proc = session.proc
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
        except OSError:
            pass
        try:
            proc.wait(timeout=3)
            return
        except Exception:
            pass
        for kill in (
            lambda: os.killpg(os.getpgid(proc.pid), signal.SIGKILL),
            proc.kill,
        ):
            try:
                kill()
                break
            except (ProcessLookupError, PermissionError, OSError):
                continue
        try:
            proc.wait(timeout=3)
        except Exception:
            pass

    def _read_log_tail(self, log_path: Path) -> str:
        return _tail_file(log_path, max_lines=int(self.limits.session_tail_lines))

    def _register_atexit(self) -> None:
        if not self._atexit_registered:
            self._atexit_registered = True
            atexit.register(self._atexit_cleanup)

    def _atexit_cleanup(self) -> None:
        # Interpreter is shutting down: no events, never raise.
        try:
            with self._sessions_lock:
                sessions = list(self._sessions.values())
                self._sessions.clear()
            for session in sessions:
                self._terminate(session)
        except Exception:
            pass


# ── environment filtering ───────────────────────────────────────────────────────
def filter_environment(environ) -> dict[str, str]:
    """Pick the host env vars allowed to cross into the sandbox.

    Allow: identity/locale basics + tool-ecosystem prefixes. Deny (vetoing any
    allow): cloud/VCS credentials and anything whose NAME smells like a secret
    (…TOKEN…, …KEY…, — so e.g. ``OLLAMA_API_KEY`` stays out even though
    ``OLLAMA_`` is an allowed prefix)."""
    out: dict[str, str] = {}
    for name, value in dict(environ).items():
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        allowed = name in _ENV_ALLOW_EXACT or any(name.startswith(p) for p in _ENV_ALLOW_PREFIX)
        if not allowed:
            continue
        upper = name.upper()
        if name in _ENV_DENY_EXACT or any(name.startswith(p) for p in _ENV_DENY_PREFIX):
            continue
        if any(s in upper for s in _ENV_DENY_SUBSTRING):
            continue
        out[name] = value
    return out


# ── category-2 (foreground-blocking) detection ─────────────────────────────────
def _blocking_server_intent(cmd: str) -> str | None:
    """A human-readable label when any segment of *cmd* starts a long-running
    server/watcher, else None. Token-based per segment; heredoc bodies are
    skipped so script content can never trip it."""
    for tokens in _segment_token_lists(cmd):
        label = _classify_server_tokens(tokens)
        if label:
            return label
    return None


def _classify_server_tokens(tokens: list[str]) -> str | None:
    # Strip env assignments, wrapper commands and their flags to find the
    # real executable (handles `nohup uvicorn …`, `FOO=1 npm start`, …).
    i = 0
    while i < len(tokens) and (
        _ENV_ASSIGN_RE.match(tokens[i]) or tokens[i] in _WRAPPER_CMDS or tokens[i].startswith("-")
    ):
        i += 1
    tokens = tokens[i:]
    if not tokens:
        return None

    exe = os.path.basename(tokens[0]).lower()
    if exe.endswith(".exe"):
        exe = exe[: -len(".exe")]

    if exe in ("npx", "bunx"):  # npx [flags] <real-command …>
        j = 1
        while j < len(tokens) and tokens[j].startswith("-"):
            j += 1
        return _classify_server_tokens(tokens[j:])

    args = tokens[1:]
    first_nonflag = next((a for a in args if not a.startswith("-")), "")

    def arg(n: int) -> str:
        return args[n] if len(args) > n else ""

    if exe in _ALWAYS_SERVER_EXES:
        return exe
    if _PYTHON_RE.match(exe):
        module = ""
        if "-m" in args:
            idx = args.index("-m")
            module = args[idx + 1] if idx + 1 < len(args) else ""
        if module in ("uvicorn", "gunicorn", "http.server"):
            return f"python -m {module}"
        if module == "flask" and "run" in args:
            return "flask run"
        script = os.path.basename(first_nonflag)
        if script in _PY_SERVER_SCRIPTS:
            return f"python {script}"
        if script == "manage.py" and "runserver" in args:
            return "manage.py runserver"
        return None
    if exe == "flask" and "run" in args:
        return "flask run"
    if exe == "vite":
        return None if first_nonflag in ("build", "optimize") else "vite"
    if exe == "webpack" and "serve" in args:
        return "webpack serve"
    if exe == "ng" and arg(0) in ("serve", "s"):
        return "ng serve"
    if exe == "next" and arg(0) in ("dev", "start"):
        return f"next {arg(0)}"
    if exe == "node":
        script = os.path.basename(first_nonflag)
        if "server" in script or script in ("app.js", "app.mjs"):
            return f"node {script}"
        return None
    if exe in ("npm", "yarn", "pnpm", "bun"):
        a0 = arg(0)
        if a0 == "start":
            return f"{exe} start"
        if a0 == "run" and arg(1).split(":")[0] in _SERVER_SCRIPT_NAMES:
            return f"{exe} run {arg(1)}"
        if exe in ("yarn", "pnpm", "bun") and a0.split(":")[0] in _SERVER_SCRIPT_NAMES:
            return f"{exe} {a0}"
        return None
    if exe == "rails" and arg(0) in ("server", "s"):
        return "rails server"
    if exe == "php" and "-S" in args:
        return "php -S"
    if exe == "deno" and arg(0) == "serve":
        return "deno serve"
    return None


def _segment_token_lists(cmd: str) -> list[list[str]]:
    """Token lists for every command segment (split on unquoted ``;``, ``&&``,
    ``||``, ``|``, ``&``), skipping heredoc bodies."""
    out: list[list[str]] = []
    heredoc_end: str | None = None
    for line in (cmd or "").split("\n"):
        if heredoc_end is not None:
            if line.strip() == heredoc_end:
                heredoc_end = None
            continue
        for segment in _split_segments(line):
            try:
                tokens = shlex.split(segment, posix=True)
            except ValueError:
                tokens = segment.split()
            if tokens:
                out.append(tokens)
        m = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
        if m:
            heredoc_end = m.group(1)
    return out


def _split_segments(line: str) -> list[str]:
    """Split one line on unquoted ``;``, ``&&``, ``||``, ``|``, ``&``."""
    segments: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            buf.append(ch)
            if ch == quote and (quote != '"' or line[i - 1] != "\\"):
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        two = line[i : i + 2]
        if two in ("&&", "||"):
            segments.append("".join(buf))
            buf = []
            i += 2
            continue
        if ch in (";", "|", "&"):
            segments.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    segments.append("".join(buf))
    return [s.strip() for s in segments if s.strip()]


# ── small helpers ───────────────────────────────────────────────────────────────
def _is_under(path: Path, ancestor: Path) -> bool:
    try:
        path.relative_to(ancestor)
        return True
    except ValueError:
        return False


def _kill_proc_tree(proc: subprocess.Popen) -> None:
    """Kill a foreground bwrap (the PDEATHSIG/PID-ns chain reaps the inside)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except Exception:
            pass


def _tail_file(path: Path, max_lines: int = 200, max_bytes: int = 32_768) -> str:
    """The last *max_lines* lines (bounded by *max_bytes*) of a session log."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            data = fh.read(max_bytes + 1)
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    clipped = size > max_bytes or len(lines) > max_lines
    lines = lines[-max_lines:]
    body = "\n".join(lines)
    if clipped:
        body = f"[…log clipped to last {len(lines)} line(s)…]\n" + body
    return body


def _truncate(text: str, limit: int = _MAX_CAPTURE) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... <{len(text) - limit} chars truncated> ...\n{tail}"
