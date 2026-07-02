"""Subprocess sandbox with all safety layers (spec §12) — NOT Docker.

Every command requested by a model passes through :meth:`Sandbox.validate`
before execution:

1. Working-directory jail: ``cwd`` is the resolved project root; absolute paths
   outside it, ``..`` escapes and ``~`` are rejected.
2. Command denylist: ``rm -rf /``, ``sudo``, ``dd``, ``mkfs``, fork bombs,
   ``shutdown``/``reboot``, ``chmod -R 777 /``, ``curl|wget | sh`` and **all
   ``git`` commands**.
3. Timeout: every command is killed after ``sandbox_timeout`` seconds (whole
   process group), returning a timeout failure.
4. Network: disabled by default; only commands recognized as dependency-install
   steps are permitted to reach the network. (True socket isolation needs
   namespaces/containers, which the no-Docker constraint forbids; this layer
   classifies + scrubs proxy env and records the decision — see README.)
5. Returns ``{exit_code, stdout, stderr, duration}``; the registry emits a
   ``tool_result`` event for each.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from workspace import PathEscapeError, Workspace

if TYPE_CHECKING:  # avoid a hard import cycle / optional at runtime
    from server.events import EventBus

# Denylisted command patterns — matched case-insensitively against the raw cmd.
_DENY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("git is forbidden", re.compile(r"(?:^|[\s;&|`(])git(?:\s|$)", re.I)),
    ("recursive root delete", re.compile(r"\brm\s+(?:-[a-z]*\s+)*-?[a-z]*r[a-z]*f?[a-z]*\s+(?:/|/\s|~)", re.I)),
    ("rm of / or ~", re.compile(r"\brm\b[^\n]*\s(/|~)(\s|$)", re.I)),
    ("sudo escalation", re.compile(r"(?:^|[\s;&|`(])sudo(?:\s|$)", re.I)),
    ("disk dd", re.compile(r"(?:^|[\s;&|`(])dd(?:\s|$)", re.I)),
    ("mkfs", re.compile(r"\bmkfs\S*\b", re.I)),
    ("fork bomb", re.compile(r":\s*\(\s*\)\s*\{")),
    ("shutdown", re.compile(r"(?:^|[\s;&|`(])shutdown(?:\s|$)", re.I)),
    ("reboot/halt/poweroff", re.compile(r"(?:^|[\s;&|`(])(?:reboot|halt|poweroff)(?:\s|$)", re.I)),
    ("chmod 777 of root", re.compile(r"\bchmod\s+-R\s+777\s+/", re.I)),
    ("pipe-to-shell install", re.compile(r"\b(?:curl|wget)\b[^\n]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|python\d?)\b", re.I)),
    ("mv/cp to root", re.compile(r"\b(?:mv|cp)\b[^\n]*\s/(\s|$)", re.I)),
]

# Absolute path prefixes that are OK to *reference* (system bins/libs/devnull),
# since they're read or executed, never written by the generated app.
_SAFE_ABS_PREFIXES = (
    "/usr/", "/bin/", "/sbin/", "/lib/", "/lib64/", "/opt/", "/etc/",
    "/proc/", "/sys/", "/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr",
    "/System/", "/Library/", "/private/var/",
)

# Shell redirection operators (>, >>, 2>, &>, 1>>, …) — next token is a write target.
_REDIRECT_RE = re.compile(r"^[0-9]*&?>>?$")
# Commands whose (non-flag) arguments are write destinations.
_WRITE_CMDS = {"cp", "mv", "tee", "ln", "rsync", "install"}

# Commands that legitimately need the network (dependency installs).
_INSTALL_RE = re.compile(
    r"\b("
    r"npm\s+(?:i\b|install|ci|add)|yarn\s+(?:add|install)|pnpm\s+(?:add|install|i\b)|"
    r"pip\d?\s+install|pip3?\s+install|python\d?\s+-m\s+pip\s+install|uv\s+(?:add|pip|sync)|"
    r"poetry\s+(?:add|install)|pipenv\s+install|"
    r"cargo\s+(?:add|build|fetch|install)|go\s+(?:get|mod\s+download|install)|"
    r"bundle\s+install|gem\s+install|composer\s+(?:install|require)|"
    r"dotnet\s+(?:restore|add)|mvn\b|gradle\b|"
    r"npx\b|bunx\b|bun\s+(?:add|install|i\b)"
    r")\b",
    re.I,
)


def normalize_pytest_command(cmd: str) -> str:
    """Rewrite bare ``pytest ...`` to ``python -m pytest ...``.

    Bare ``pytest`` does not add the project root to ``sys.path``, so
    ``from mypkg import ...`` raises ModuleNotFoundError (exit 2) even for correct
    code. ``python -m pytest`` adds the CWD to ``sys.path`` and fixes it — the most
    common local-build test failure for Python projects. Applied at the
    orchestration layer (loop + registry) so the sandbox still runs exactly what
    it is handed.
    """
    s = (cmd or "").strip()
    if re.match(r"^pytest(\s|$)", s):
        return "python -m " + s
    return cmd


@dataclass
class CommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    timed_out: bool = False
    rejected: bool = False
    reason: str = ""
    network_allowed: bool = False

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
            "network_allowed": self.network_allowed,
        }


class CommandRejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# Output captured from a command is truncated to keep events/conversations sane.
_MAX_CAPTURE = 60_000


class Sandbox:
    def __init__(self, workspace: Workspace, limits, bus: "EventBus | None" = None):
        self.workspace = workspace
        self.limits = limits
        self.bus = bus

    # ── validation ────────────────────────────────────────────────────────────
    def is_install_command(self, cmd: str) -> bool:
        return bool(_INSTALL_RE.search(cmd))

    def validate(self, cmd: str) -> None:
        """Raise :class:`CommandRejected` if *cmd* violates any safety layer."""
        if not cmd or not cmd.strip():
            raise CommandRejected("empty command")

        for reason, pattern in _DENY_PATTERNS:
            if pattern.search(cmd):
                raise CommandRejected(reason)

        self._check_paths(cmd)

    def _check_paths(self, cmd: str) -> None:
        """Reject path-like tokens that escape the project root.

        Read/exec references to system paths (``/usr/bin/python`` …) are allowed
        via :data:`_SAFE_ABS_PREFIXES`, but **write targets** — redirection
        destinations and the args of ``cp``/``mv``/``tee``/``ln``/``rsync`` —
        must resolve strictly inside the root, with no safe-prefix bypass.
        """
        try:
            tokens = shlex.split(cmd, posix=True)
        except ValueError:
            tokens = cmd.split()

        write_target_next = False  # set after a redirection operator
        write_cmd_active = False   # inside a cp/mv/tee/ln/rsync invocation

        for tok in tokens:
            if tok in (";", "&&", "||", "|", "&"):
                write_cmd_active = False
                continue
            if _REDIRECT_RE.match(tok):
                write_target_next = True
                continue
            if tok in _WRITE_CMDS:
                write_cmd_active = True
                continue

            is_write_target = write_target_next or (write_cmd_active and not tok.startswith("-"))
            write_target_next = False

            # strip a leading flag like --out=/etc/x (still a write if it's an output flag)
            val = tok.split("=", 1)[1] if tok.startswith("-") and "=" in tok else tok
            if not val:
                continue

            if val.startswith("~"):
                raise CommandRejected(f"home (~) path not allowed: {tok!r}")
            if val.startswith("/"):
                if not is_write_target and val.startswith(_SAFE_ABS_PREFIXES):
                    continue
                if not self._within_root(val):
                    what = "write target escapes" if is_write_target else "absolute path escapes"
                    raise CommandRejected(f"{what} project root: {tok!r}")
            elif ".." in val.replace("\\", "/").split("/"):
                if not self._within_root(val):
                    raise CommandRejected(f"'..' path escapes project root: {tok!r}")

    def _within_root(self, path: str) -> bool:
        try:
            self.workspace.resolve_in_root(path)
            return True
        except PathEscapeError:
            return False

    # ── execution ─────────────────────────────────────────────────────────────
    def run(
        self,
        cmd: str,
        timeout: int | None = None,
        *,
        validate: bool = True,
        allow_network: bool | None = None,
    ) -> CommandResult:
        """Validate and run *cmd* (foreground) inside the project root."""
        if validate:
            try:
                self.validate(cmd)
            except CommandRejected as exc:
                return CommandResult(exit_code=126, rejected=True, reason=exc.reason, stderr=exc.reason)

        if allow_network is None:
            allow_network = self.is_install_command(cmd)

        timeout = int(timeout or self.limits.sandbox_timeout)
        env = self._build_env(allow_network)
        start = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=str(self.workspace.root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,  # own process group, so we can kill the whole tree
            )
        except OSError as exc:
            return CommandResult(exit_code=127, stderr=f"failed to start command: {exc}", reason=str(exc))

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            duration = time.monotonic() - start
            return CommandResult(
                exit_code=proc.returncode,
                stdout=_truncate(stdout),
                stderr=_truncate(stderr),
                duration=duration,
                network_allowed=allow_network,
            )
        except subprocess.TimeoutExpired:
            self._kill_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
            duration = time.monotonic() - start
            return CommandResult(
                exit_code=124,
                stdout=_truncate(stdout or ""),
                stderr=_truncate((stderr or "") + f"\n[timeout after {timeout}s]"),
                duration=duration,
                timed_out=True,
                reason=f"timeout after {timeout}s",
                network_allowed=allow_network,
            )

    def _build_env(self, allow_network: bool) -> dict[str, str]:
        env = dict(os.environ)
        # Keep package managers non-interactive and deterministic.
        env.setdefault("CI", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        env.setdefault("npm_config_yes", "true")
        if not allow_network:
            # Scrub proxy hints so non-install steps don't reach out unexpectedly.
            for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
                env.pop(var, None)
            env["NO_NETWORK"] = "1"
        return env

    @staticmethod
    def _kill_group(proc: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass


def _truncate(text: str, limit: int = _MAX_CAPTURE) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... <{len(text) - limit} chars truncated> ...\n{tail}"
