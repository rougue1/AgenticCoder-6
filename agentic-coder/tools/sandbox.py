"""Subprocess sandbox with all safety layers (redesign) — NOT Docker.

Every command requested by a model passes through :meth:`Sandbox.validate`
before execution:

1. **never_allow** — a hard, system-level denylist that no configuration can
   override: ``rm -rf /``, ``sudo``, ``dd``, ``mkfs``, fork bombs,
   ``shutdown``/``reboot``, ``curl|wget`` piped to a shell, and **all git**.
2. **Command allowlist** — each command segment's executable must be in the
   project's ``allowed_commands`` (the Manager's stack-determination output
   extended by the built-in stack profile registry). Rejections emit
   ``sandbox.command_rejected``.
3. **Working-directory jail** — ``cwd`` is the project root; absolute paths
   outside it, ``..`` escapes and ``~`` write targets are rejected.
4. **Venv transparency** — Python commands are rewritten to the project venv
   (``python`` -> ``.venv/bin/python``, bare ``pytest`` ->
   ``.venv/bin/python -m pytest`` …) and the subprocess env gets
   ``VIRTUAL_ENV`` + a venv-first ``PATH`` with ``PYTHONHOME`` removed. The
   Worker never sees any of this.
4b. **Node transparency** — a direct ``node_modules/.bin/<tool>`` invocation
   (at any relative depth) is rewritten to run through ``node`` so a missing
   executable bit (common on npm-installed ``.bin`` entries) can't 126 it, and
   any such scripts found under the project root / its immediate subdirectories
   have their executable bit repaired before every command runs, belt-and-
   suspenders style. The Worker never sees any of this either.
5. **Timeout** — every command is killed after ``sandbox.timeout`` seconds
   (whole process group); timeouts emit ``sandbox.timeout``.
6. **Network classification** — only recognized dependency-install commands
   keep proxy env; everything else gets it scrubbed.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from workspace import PathEscapeError, Workspace

if TYPE_CHECKING:  # avoid a hard import cycle / optional at runtime
    from server.events import EventBus

# ── never_allow: enforced at the system level, cannot be overridden ───────────
_NEVER_ALLOW: list[tuple[str, re.Pattern[str]]] = [
    ("git is forbidden", re.compile(r"(?:^|[\s;&|`(])git(?:\s|$)", re.I)),
    ("recursive root delete", re.compile(r"\brm\s+(?:-[a-z]*\s+)*-?[a-z]*r[a-z]*f?[a-z]*\s+(?:/|/\s|~)", re.I)),
    ("rm of / or ~", re.compile(r"\brm\b[^\n]*\s(/|~)(\s|$)", re.I)),
    ("sudo escalation", re.compile(r"(?:^|[\s;&|`(])sudo(?:\s|$)", re.I)),
    ("disk dd", re.compile(r"(?:^|[\s;&|`(])dd(?:\s|$)", re.I)),
    ("mkfs", re.compile(r"\bmkfs\S*\b", re.I)),
    ("fork bomb", re.compile(r":\s*\(\s*\)\s*\{")),
    ("shutdown", re.compile(r"(?:^|[\s;&|`(])shutdown(?:\s|$)", re.I)),
    ("reboot/halt/poweroff", re.compile(r"(?:^|[\s;&|`(])(?:reboot|halt|poweroff)(?:\s|$)", re.I)),
    ("chmod 777 of root", re.compile(r"\bchmod\s+(?:-R\s+)?777\s+/", re.I)),
    ("pipe-to-shell install", re.compile(r"\b(?:curl|wget)\b[^\n]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|python\d?)\b", re.I)),
    ("mv/cp to root", re.compile(r"\b(?:mv|cp)\b[^\n]*\s/(\s|$)", re.I)),
]

# Absolute path prefixes that are OK to *reference* (system bins/libs/devnull),
# since they're read or executed, never written by the generated app.
_SAFE_ABS_PREFIXES = (
    "/usr/", "/bin/", "/sbin/", "/lib/", "/lib64/", "/opt/", "/etc/",
    "/proc/", "/sys/", "/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr",
    "/System/", "/Library/", "/private/var/", "/tmp/",
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

# Python-ecosystem executables transparently rewritten to their venv equivalents.
_VENV_BIN_TOOLS = {"pip", "pip3", "uvicorn", "gunicorn", "black", "ruff", "mypy", "flake8", "isort", "coverage", "alembic", "flask", "pytest"}
_PY_MODULE_FALLBACK = {  # tool -> `python -m <module>` when the venv has no shim
    "pytest": "pytest", "pip": "pip", "pip3": "pip", "black": "black", "ruff": "ruff",
    "mypy": "mypy", "flake8": "flake8", "isort": "isort", "coverage": "coverage",
    "uvicorn": "uvicorn", "gunicorn": "gunicorn", "flask": "flask", "alembic": "alembic",
}
_PYTHON_RE = re.compile(r"^python(?:3(?:\.\d+)?)?$")

# A direct invocation of an npm-installed node_modules/.bin/<tool> script (any
# relative depth: bare, ``./``, ``../``, or a subproject like ``frontend/…``).
# These are frequently installed without the executable bit set (permission is
# not always preserved by the packing/extraction step), which makes running the
# path directly fail with exit 126 (found, not executable) even though the
# Worker's command is otherwise correct. Routing it through the ``node``
# interpreter needs only read access, so it sidesteps the executable-bit
# requirement entirely — the same "transparent rewrite" treatment bare
# `python`/`pytest` get for the venv.
_NODE_BIN_PATH_RE = re.compile(r"^(?:[\w.\-]+/)*node_modules/\.bin/[\w.\-]+$")

# Segment boundaries at which a new command begins.
_SEGMENT_SPLIT_RE = re.compile(r"(\|\||&&|;|\||&(?!&))")

# Env-assignment prefix tokens (FOO=bar cmd …).
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Output captured from a command is truncated to keep events/conversations sane.
_MAX_CAPTURE = 60_000


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


class Sandbox:
    def __init__(self, workspace: Workspace, config, bus: "EventBus | None" = None):
        """*config* is the AppConfig ``sandbox`` section (SandboxCfg)."""
        self.workspace = workspace
        self.limits = config  # .timeout / .long_process_timeout
        self.bus = bus
        # Project allowlist: stack profile base ∪ Manager's allowed_commands.
        # Empty means "not yet configured" — only never_allow applies then
        # (Phase 1 runs no model commands before the allowlist is set).
        self.allowed_commands: set[str] = set()
        self.venv_path: Path | None = None
        self.node_bin_path: Path | None = None

    # ── configuration (orchestrator-side) ─────────────────────────────────────
    def set_allowed_commands(self, commands: list[str]) -> None:
        self.allowed_commands = {str(c).strip() for c in commands if str(c).strip()}

    def set_venv(self, venv_path: str | Path | None) -> None:
        self.venv_path = Path(venv_path) if venv_path else None

    def set_node_bin(self, project_root: str | Path | None) -> None:
        self.node_bin_path = (Path(project_root) / "node_modules" / ".bin") if project_root else None

    # ── validation ────────────────────────────────────────────────────────────
    def is_install_command(self, cmd: str) -> bool:
        return bool(_INSTALL_RE.search(cmd))

    def validate(self, cmd: str) -> None:
        """Raise :class:`CommandRejected` if *cmd* violates any safety layer."""
        if not cmd or not cmd.strip():
            raise CommandRejected("empty command")

        for reason, pattern in _NEVER_ALLOW:
            if pattern.search(cmd):
                self._emit_rejected(cmd, reason)
                raise CommandRejected(reason)

        self._check_allowlist(cmd)
        self._check_paths(cmd)

    def _check_allowlist(self, cmd: str) -> None:
        """Every command segment's executable must be on the project allowlist."""
        if not self.allowed_commands:
            return  # allowlist not configured yet (pre-stack orchestrator use)
        for exe in _command_words(cmd):
            base = os.path.basename(exe)
            norm = base[:-len(".exe")] if base.endswith(".exe") else base
            if norm in self.allowed_commands:
                continue
            # A venv/als path like .venv/bin/python is fine if its basename is allowed.
            if _PYTHON_RE.match(norm):
                continue
            reason = f"command {norm!r} is not in this project's allowed commands"
            self._emit_rejected(cmd, reason)
            raise CommandRejected(reason)

    def _emit_rejected(self, cmd: str, reason: str) -> None:
        if self.bus is not None:
            from server import events

            self.bus.emit(events.SANDBOX_COMMAND_REJECTED, "sandbox", cmd=cmd[:400], reason=reason)

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
                    hint = "" if is_write_target else " — paths are relative to the project root; drop the leading '/'"
                    raise CommandRejected(f"{what} project root: {tok!r}{hint}")
            elif ".." in val.replace("\\", "/").split("/"):
                if not self._within_root(val):
                    raise CommandRejected(f"'..' path escapes project root: {tok!r}")

    def _within_root(self, path: str) -> bool:
        try:
            self.workspace.resolve_in_root(path)
            return True
        except PathEscapeError:
            return False

    # ── venv / node-bin rewriting (the Worker never sees this) ────────────────
    def rewrite_command(self, cmd: str) -> str:
        """Rewrite Python-ecosystem executables to their venv equivalents, and
        direct ``node_modules/.bin/<tool>`` invocations to run through ``node``.

        Applied at command positions only (start of each segment/line), skipping
        heredoc bodies, so text like ``grep python file`` is never touched.
        """
        if not cmd:
            return cmd
        venv = self.venv_path
        py = str(venv / "bin" / "python") if venv is not None else None

        def _rewrite_word(word: str) -> str:
            if venv is not None:
                if _PYTHON_RE.match(word):
                    return py
                if word in _VENV_BIN_TOOLS:
                    shim = venv / "bin" / word
                    if shim.exists():
                        return str(shim)
                    mod = _PY_MODULE_FALLBACK.get(word)
                    return f"{py} -m {mod}" if mod else word
            if _NODE_BIN_PATH_RE.match(word):
                return f"node {word}"
            return word

        out_lines: list[str] = []
        heredoc_end: str | None = None
        for line in cmd.split("\n"):
            if heredoc_end is not None:
                out_lines.append(line)
                if line.strip() == heredoc_end:
                    heredoc_end = None
                continue
            out_lines.append(_rewrite_line(line, _rewrite_word))
            m = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
            if m:
                heredoc_end = m.group(1)
        return "\n".join(out_lines)

    # ── node_modules/.bin permission repair (the Worker never sees this) ──────
    def _ensure_node_bin_executable(self) -> None:
        """Repair a missing execute bit on installed ``node_modules/.bin/*``
        scripts before they can ever be run.

        npm/yarn/pnpm on this filesystem sometimes extract ``.bin`` entries
        without the executable bit set, which makes a direct
        ``node_modules/.bin/tsc`` invocation fail with exit 126 (found, not
        executable) — a state a Worker command can't tell apart from "wrong
        path" by exit code alone. Fixed proactively here (bounded to the
        project root and its immediate subdirectories, to cover a simple
        ``frontend/``-style layout without walking the whole dependency tree),
        so the Worker never has a reason to hit it.
        """
        root = self.workspace.root
        bin_dirs = [root / "node_modules" / ".bin"]
        try:
            for child in root.iterdir():
                if child.is_dir() and child.name != "node_modules":
                    bin_dirs.append(child / "node_modules" / ".bin")
        except OSError:
            return
        for bin_dir in bin_dirs:
            try:
                entries = list(bin_dir.iterdir())
            except OSError:
                continue
            for entry in entries:
                try:
                    st = entry.stat()
                    if not st.st_mode & 0o111:
                        entry.chmod(st.st_mode | 0o111)
                except OSError:
                    continue

    # ── execution ─────────────────────────────────────────────────────────────
    def run(
        self,
        cmd: str,
        timeout: int | None = None,
        *,
        validate: bool = True,
        allow_network: bool | None = None,
    ) -> CommandResult:
        """Validate, venv-rewrite, and run *cmd* (foreground) in the project root."""
        if validate:
            try:
                self.validate(cmd)
            except CommandRejected as exc:
                return CommandResult(exit_code=126, rejected=True, reason=exc.reason, stderr=exc.reason)

        if self.node_bin_path is not None:
            self._ensure_node_bin_executable()

        if allow_network is None:
            allow_network = self.is_install_command(cmd)

        exec_cmd = self.rewrite_command(cmd)
        timeout = int(timeout or self.limits.timeout)
        env = self.build_env(allow_network)
        start = time.monotonic()
        try:
            proc = subprocess.Popen(
                exec_cmd,
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
            if self.bus is not None:
                from server import events

                self.bus.emit(events.SANDBOX_TIMEOUT, "sandbox", cmd=cmd[:400], timeout=timeout)
            return CommandResult(
                exit_code=124,
                stdout=_truncate(stdout or ""),
                stderr=_truncate((stderr or "") + f"\n[timeout after {timeout}s]"),
                duration=duration,
                timed_out=True,
                reason=f"timeout after {timeout}s",
                network_allowed=allow_network,
            )

    def build_env(self, allow_network: bool) -> dict[str, str]:
        env = dict(os.environ)
        # Keep package managers non-interactive and deterministic.
        env.setdefault("CI", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        env.setdefault("npm_config_yes", "true")
        # No .pyc caching: a fix that rewrites a module within the same second
        # as the previous test run would otherwise be masked by a stale
        # __pycache__ entry (mtime granularity), re-failing a correct fix.
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        if self.venv_path is not None:
            env["VIRTUAL_ENV"] = str(self.venv_path)
            env["PATH"] = str(self.venv_path / "bin") + os.pathsep + env.get("PATH", "")
            env.pop("PYTHONHOME", None)
        if self.node_bin_path is not None:
            env["PATH"] = str(self.node_bin_path) + os.pathsep + env.get("PATH", "")
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


# ── command scanning helpers ───────────────────────────────────────────────────
def _command_words(cmd: str) -> list[str]:
    """The executable word of every command segment (quote-aware, heredoc-safe)."""
    words: list[str] = []
    heredoc_end: str | None = None
    for line in (cmd or "").split("\n"):
        if heredoc_end is not None:
            if line.strip() == heredoc_end:
                heredoc_end = None
            continue
        for segment in _split_segments(line):
            word = _first_word(segment)
            if word:
                words.append(word)
        m = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
        if m:
            heredoc_end = m.group(1)
    return words


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


def _first_word(segment: str) -> str | None:
    """The executable of a segment, skipping env assignments and wrappers."""
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        tokens = segment.split()
    skip_next_value = False
    for tok in tokens:
        if skip_next_value:
            skip_next_value = False
            continue
        if _ENV_ASSIGN_RE.match(tok):
            continue
        if tok in ("env", "nice", "time", "exec", "nohup"):
            continue
        if tok.startswith("-"):
            continue  # flags of a wrapper like env/nice
        if _REDIRECT_RE.match(tok):
            skip_next_value = True
            continue
        return tok
    return None


def _rewrite_line(line: str, rewrite_word) -> str:
    """Rewrite the command word of each segment in *line* (quote-aware)."""
    out: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    at_cmd_pos = True
    token: list[str] = []

    def _flush_token() -> None:
        nonlocal at_cmd_pos
        if not token:
            return
        word = "".join(token)
        token.clear()
        if at_cmd_pos and not _ENV_ASSIGN_RE.match(word) and word not in ("env", "nice", "time", "exec", "nohup"):
            buf.append(rewrite_word(word))
            at_cmd_pos = False
        else:
            buf.append(word)

    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            token.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            token.append(ch)
            i += 1
            continue
        two = line[i : i + 2]
        if two in ("&&", "||"):
            _flush_token()
            buf.append(two)
            at_cmd_pos = True
            i += 2
            continue
        if ch in (";", "|", "&"):
            _flush_token()
            buf.append(ch)
            at_cmd_pos = True
            i += 1
            continue
        if ch in (" ", "\t"):
            _flush_token()
            buf.append(ch)
            i += 1
            continue
        token.append(ch)
        i += 1
    _flush_token()
    out.append("".join(buf))
    return "".join(out)


def _truncate(text: str, limit: int = _MAX_CAPTURE) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... <{len(text) - limit} chars truncated> ...\n{tail}"
