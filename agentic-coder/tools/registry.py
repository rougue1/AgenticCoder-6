"""The Worker tools + dispatch (bwrap redesign).

* ``read_file``      ``{path}`` — jailed + .agentignore-checked read.
* ``write_file``     ``{path, content, summary}`` — create/overwrite; ``summary``
  is REQUIRED and recorded in the manifest; emits ``file_written``.
* ``patch_file``     ``{path, old_string, new_string}`` — exact-match single
  replacement with precise diagnostics (zero matches / ambiguous / whitespace).
* ``run``            ``{cmd, background?, timeout?}`` — deny-list-checked
  execution inside the bwrap OS sandbox (login shell, workspace-only writes).
  Foreground blocks to completion; ``background: true`` starts a session and
  returns its ``session_id`` immediately.
* ``check_session``  ``{session_id}`` — status + captured output of a
  background session (running or crashed).
* ``stop_session``   ``{session_id}`` — terminate a background session.

Every dispatch emits ``worker.tool_call`` then ``worker.tool_result``. The
model never runs anything itself — it only requests; this layer validates and
executes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from context.conversation import cap_tool_output
from llm.tool_parser import ToolCall
from tools import formatters
from tools.sandbox import Sandbox, SessionStatus
from workspace import IgnoredPathError, PathEscapeError, Workspace

if TYPE_CHECKING:
    from context.manifest import Manifest
    from server.events import EventBus

# Description injected into Worker prompts so the model knows the protocol.
# NOTE: the wrapper tag is <agentic_call>, NOT the more obvious tool_call tag.
# ornith (qwen3.5-family) is recognized by Ollama's own renderer/parser, which
# natively scans raw output for the literal qwen tool-call markers and tries to
# lift them into message.tool_calls using ITS OWN schema (name/arguments)
# before we ever see the text. Our schema uses tool/args, so Ollama's parser
# can't find what it expects, hits EOF mid-parse, and the whole /api/chat
# response comes back as {"error": "EOF"} instead of real content (litellm
# then dies with a raw KeyError on the missing "message" key). Using a
# non-native tag keeps this entirely our own protocol, parsed only by
# llm/tool_parser.py.
TOOL_INSTRUCTIONS = """\
You act ONLY by emitting tool calls. You never run commands yourself.
Emit EXACTLY ONE tool call per message, wrapped in <agentic_call>...</agentic_call>
tags with a JSON body. After each call you receive the result, then continue.

Available tools:

1. read_file — read an existing file.
   <agentic_call>{"tool": "read_file", "args": {"path": "src/app.py"}}</agentic_call>

2. write_file — create or overwrite a WHOLE file. The one-line `summary`
   describing what the file does is REQUIRED (it feeds the project manifest).
   <agentic_call>{"tool": "write_file", "args": {"path": "src/app.py", "content": "<full file contents>", "summary": "FastAPI entrypoint"}}</agentic_call>

3. patch_file — change an EXISTING file surgically: replace `old_string` (which
   must appear EXACTLY ONCE — copy the exact bytes including indentation, and
   include enough surrounding context to make it unique) with `new_string`.
   Prefer this over write_file for any change short of a full rewrite.
   <agentic_call>{"tool": "patch_file", "args": {"path": "src/app.py", "old_string": "def add(a, b):\\n    return a - b", "new_string": "def add(a, b):\\n    return a + b"}}</agentic_call>

4. run — run a shell command in the project root. Full shell syntax works and
   your user-level tools (node via nvm, pyenv pythons, …) are on PATH. Only
   the project directory is writable. Two modes:
   - FOREGROUND (default): blocks until the command finishes, returns
     stdout/stderr/exit code. For tests, builds, installs, one-off commands.
     Long-running server commands are REJECTED in foreground — see background.
     <agentic_call>{"tool": "run", "args": {"cmd": "python -m pytest -q", "timeout": 120}}</agentic_call>
   - BACKGROUND: set "background": true for anything that keeps running (dev
     servers, watchers, databases). Returns a session_id immediately while the
     process keeps running; then verify it with normal foreground commands
     (curl, a test suite, …) and inspect it with check_session.
     <agentic_call>{"tool": "run", "args": {"cmd": "uvicorn app.main:app --port 8100", "background": true}}</agentic_call>

5. check_session — status of a background session: running or exited, its exit
   code, and its captured output (server logs, crash traces).
   <agentic_call>{"tool": "check_session", "args": {"session_id": "<id returned by run>"}}</agentic_call>

6. stop_session — terminate a background session (e.g. to free its port before
   restarting a reconfigured server).
   <agentic_call>{"tool": "stop_session", "args": {"session_id": "<id>"}}</agentic_call>

Rules:
- All paths are relative to the project root. Never use absolute paths, "..",
  "~", or git in any form. The .agent/ directory does not exist for you.
- EXPLORE before writing when unsure: run `ls -R` (or `find . -type f`) and
  read_file the files you are about to depend on, instead of guessing.
- Installing packages requires actually RUNNING the installer: after you write
  requirements.txt (or package.json) run it — e.g. run `pip install -r
  requirements.txt`. Writing the manifest file does not install anything.
- write_file writes COMPLETE file contents, never diffs or "... unchanged ...".
- To change a file that already exists, prefer patch_file over rewriting it.
- Background processes run in their own PID namespace: `kill <pid>` cannot
  reach them — use stop_session. All sessions are terminated automatically
  when this subtask ends; stop_session servers you started once you are done
  verifying against them (a leftover server can hold the port the final test
  run needs).
- For a LARGE new file (more than a few hundred lines) build it with run + bash
  heredoc appends from the FIRST call (`cat > path <<'EOF' ... EOF`, then a
  `cat >> path <<'EOF' ... EOF` per part) — a single huge write_file body can be
  truncated mid-stream. Verify with `wc -c path`.
- When you are completely done with this step, reply with the single line: DONE
"""

# Appended to a write_file/patch_file result when the call had to be salvaged
# from malformed JSON — almost always because the server truncated the streamed
# arguments at its output-token limit, leaving a possibly-incomplete file.
_TRUNCATED_ARGS_NOTE = (
    "\n\nNOTE: this call's arguments looked truncated (the server likely cut them at "
    "its output-token limit), so the file may be INCOMPLETE. Do NOT retry the same "
    "whole-file write. Verify with run `wc -c <path>` (and read the tail); if it is "
    "short, rebuild it in parts with bash heredoc appends: run `cat > <path> <<'EOF' "
    "… EOF` for the first part, then a `cat >> <path> <<'EOF' … EOF` per following part."
)

_VALID_TOOLS_LINE = "Valid tools: read_file, write_file, patch_file, run, check_session, stop_session."


@dataclass
class ToolResult:
    """Normalized result of executing a tool call."""

    tool: str
    ok: bool
    payload: dict
    display: str  # text fed back into the model conversation

    def to_dict(self) -> dict:
        return {"tool": self.tool, "ok": self.ok, **self.payload}


class ToolRegistry:
    def __init__(
        self,
        workspace: Workspace,
        sandbox: Sandbox,
        manifest: "Manifest",
        bus: "EventBus",
    ):
        self.workspace = workspace
        self.sandbox = sandbox
        self.manifest = manifest
        self.bus = bus

    # ── dispatch ──────────────────────────────────────────────────────────────
    def dispatch(self, call: ToolCall, phase: str) -> ToolResult:
        self.bus.tool_call(phase, call.name, call.args)
        handler = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "patch_file": self._patch_file,
            "run": self._run,
            "check_session": self._check_session,
            "stop_session": self._stop_session,
        }.get(call.name)
        if handler is None:
            result = ToolResult(
                tool=call.name,
                ok=False,
                payload={"error": f"unknown tool {call.name!r}"},
                display=f"ERROR: unknown tool {call.name!r}. {_VALID_TOOLS_LINE}",
            )
        else:
            result = handler(call.args, phase)
        # A salvaged call means the parser had to reconstruct the arguments — the
        # strong signal the server truncated them mid-stream, so a write may be
        # incomplete. Tell the model how to recover instead of silently accepting
        # a half-written file and letting it re-emit the same too-large write.
        if getattr(call, "salvaged", False) and result.tool in ("write_file", "patch_file") and result.ok:
            result.display = result.display.rstrip() + _TRUNCATED_ARGS_NOTE
        # Cap the text fed back into the conversation so one noisy read or test
        # log can't dominate the worker's window (the full payload still reaches
        # the event stream via _event_payload below).
        result.display = cap_tool_output(result.display)
        self.bus.tool_result(phase, result.tool, ok=result.ok, **_event_payload(result))
        return result

    # ── path gate shared by the file tools ────────────────────────────────────
    def _tool_target(self, tool: str, path: str):
        try:
            return self.workspace.resolve_tool_path(path), None
        except PathEscapeError as exc:
            return None, _err(tool, str(exc))
        except IgnoredPathError as exc:
            return None, _err(tool, str(exc))

    # ── read_file ─────────────────────────────────────────────────────────────
    def _read_file(self, args: dict, phase: str) -> ToolResult:
        path = (args or {}).get("path")
        if not path:
            return _err("read_file", "missing required arg 'path'")
        target, err = self._tool_target("read_file", path)
        if err:
            return err
        if not target.exists():
            return _err("read_file", f"file not found: {path}")
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return _err("read_file", f"could not read {path}: {exc}")
        rel = self.workspace.relative(path)
        return ToolResult(
            tool="read_file",
            ok=True,
            payload={"path": rel, "bytes": len(content)},
            display=f"Contents of {rel}:\n```\n{content}\n```",
        )

    # ── write_file ────────────────────────────────────────────────────────────
    def _write_file(self, args: dict, phase: str) -> ToolResult:
        args = args or {}
        path = args.get("path")
        content = args.get("content")
        summary = (args.get("summary") or args.get("description") or "").strip()
        if not path:
            return _err("write_file", "missing required arg 'path'")
        if content is None:
            return _err("write_file", "missing required arg 'content'")
        if not summary:
            return _err("write_file", "missing required arg 'summary' — one line describing what this file does")
        if not isinstance(content, str):
            content = str(content)
        target, err = self._tool_target("write_file", path)
        if err:
            return err

        action = "edit" if target.exists() else "create"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return _err("write_file", f"could not write {path}: {exc}")

        rel = self.workspace.relative(path)
        self.manifest.record(rel, summary)
        # Feature 4 — post-execution formatting hook: silent, best-effort, never
        # blocks. The Worker never sees that formatting happened; we re-read the
        # file so the event/manifest/display bytes reflect the formatted content.
        formatters.format_file(self.sandbox, rel, self.bus, phase)
        final_content = _read_after_format(target, content)
        self.bus.file_written(phase, rel, action, content=final_content)
        return ToolResult(
            tool="write_file",
            ok=True,
            payload={"path": rel, "action": action, "bytes": len(final_content)},
            display=f"Wrote {rel} ({action}, {len(final_content)} bytes).",
        )

    # ── patch_file ────────────────────────────────────────────────────────────
    def _patch_file(self, args: dict, phase: str) -> ToolResult:
        """Exact-string single replacement on an existing file.

        ``old_string`` must appear EXACTLY ONCE. Zero matches, ambiguous
        matches, and whitespace-only mismatches each get their own precise
        error so the model can correct itself instead of burning retries.
        """
        args = args or {}
        path = args.get("path")
        old = args.get("old_string")
        new = args.get("new_string")
        if not path:
            return _err("patch_file", "missing required arg 'path'")
        if old is None or old == "":
            return _err("patch_file", "missing/empty 'old_string' (need a unique anchor to replace)")
        if new is None:
            return _err("patch_file", "missing required arg 'new_string' (use \"\" to delete the anchor)")
        if not isinstance(old, str):
            old = str(old)
        if not isinstance(new, str):
            new = str(new)
        if old == new:
            return _err("patch_file", "no change: old_string equals new_string")
        target, err = self._tool_target("patch_file", path)
        if err:
            return err
        if not target.exists():
            return _err("patch_file", f"file not found: {path} (use write_file to create it)")
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return _err("patch_file", f"could not read {path}: {exc}")

        n = content.count(old)
        if n == 0:
            # A near-miss differing only in whitespace (indentation/tabs/newlines)
            # is the most common patch failure for a local model — name it so it
            # fixes the bytes instead of guessing.
            if _differs_only_in_whitespace(content, old):
                return _err(
                    "patch_file",
                    f"no exact match in {path} — a block there differs from old_string ONLY in "
                    f"whitespace (indentation/tabs/newlines); copy the exact bytes, including indentation",
                )
            return _err("patch_file", f"old_string not found in {path} — copy the exact bytes, including indentation")
        if n > 1:
            return _err("patch_file", f"old_string appears {n}× in {path} — add surrounding context to make it unique")
        # str.count is non-overlapping, so a self-overlapping anchor ("==" in
        # "a === b") can pass n==1 yet match two ways. Reject the overlap.
        idx = content.find(old)
        if content.find(old, idx + 1) != -1:
            return _err("patch_file", "old_string overlaps itself — add surrounding context to make it unique")

        updated = content[:idx] + new + content[idx + len(old):]
        try:
            target.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return _err("patch_file", f"could not write {path}: {exc}")

        rel = self.workspace.relative(path)
        self.manifest.record(rel, args.get("summary") or args.get("description"))
        # Feature 4 — post-execution formatting hook (see _write_file for the
        # full rationale); re-read so the reported bytes reflect the formatted file.
        formatters.format_file(self.sandbox, rel, self.bus, phase)
        final_content = _read_after_format(target, updated)
        self.bus.file_written(phase, rel, "edit", content=final_content)
        return ToolResult(
            tool="patch_file",
            ok=True,
            payload={"path": rel, "action": "edit", "bytes": len(final_content)},
            display=f"Patched {rel} (-{len(old)} +{len(new)} bytes).",
        )

    # ── run ───────────────────────────────────────────────────────────────────
    def _run(self, args: dict, phase: str) -> ToolResult:
        args = args or {}
        cmd = args.get("cmd") or args.get("command")
        if not cmd:
            return _err("run", "missing required arg 'cmd'")
        background = bool(args.get("background", False))
        timeout = args.get("timeout")

        note = ""
        if args.get("smoke") or args.get("smoke_cmds"):
            # The pre-bwrap harness took "smoke" commands; sessions replaced it.
            note = (
                '\n[note] "smoke" is no longer a run argument. The background session keeps '
                "running: verify it yourself with foreground run calls (curl …) and "
                "check_session, then stop_session when done."
            )

        if background:
            return self._run_background(cmd, note)

        result = self.sandbox.run(cmd, timeout=int(timeout) if timeout else None)
        if result.rejected:
            return ToolResult(
                tool="run",
                ok=False,
                payload={"cmd": cmd, "exit_code": result.exit_code, "rejected": True, "reason": result.reason},
                display=f"COMMAND REJECTED ({result.reason}): {cmd}",
            )
        return ToolResult(
            tool="run",
            ok=result.ok,
            payload={
                "cmd": cmd,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": round(result.duration, 3),
                "timed_out": result.timed_out,
                "background": False,
            },
            display=_render_run(cmd, result, background=False) + note,
        )

    def _run_background(self, cmd: str, note: str = "") -> ToolResult:
        start = self.sandbox.run_background(cmd)
        if start.rejected:
            return ToolResult(
                tool="run",
                ok=False,
                payload={"cmd": cmd, "exit_code": 126, "rejected": True, "reason": start.reason, "background": True},
                display=f"COMMAND REJECTED ({start.reason}): {cmd}",
            )
        if not start.session_id:
            return _err("run", f"could not start background session: {start.reason}")

        payload = {
            "cmd": cmd,
            "background": True,
            "session_id": start.session_id,
            "running": start.running,
            "exit_code": start.exit_code,
            "log_path": start.log_path,
            "output": start.output,
        }
        lines = [f"$ {cmd}"]
        if start.running:
            lines.append(f"[background] started, session_id={start.session_id} (still running)")
            if start.output.strip():
                lines.append(f"--- early output ---\n{start.output}")
            lines.append(
                "Verify it with foreground run calls (e.g. curl), inspect with check_session "
                f'{{"session_id": "{start.session_id}"}}, terminate with stop_session.'
            )
        elif start.exit_code == 0:
            lines.append(
                f"[background] session_id={start.session_id} exited immediately with code 0 — "
                "it finished (or daemonized) instead of staying in the foreground of its session."
            )
            lines.append(f"--- output ---\n{start.output or '(no output captured)'}")
        else:
            lines.append(
                f"[background] session_id={start.session_id} EXITED IMMEDIATELY with code {start.exit_code} "
                "— it did not stay running. Diagnose from its output below (port already in use? "
                "bad flag? missing dependency?) and fix before retrying."
            )
            lines.append(f"--- output ---\n{start.output or '(no output captured)'}")
        return ToolResult(tool="run", ok=start.ok, payload=payload, display="\n".join(lines) + note)

    # ── check_session / stop_session ──────────────────────────────────────────
    def _check_session(self, args: dict, phase: str) -> ToolResult:
        session_id = str((args or {}).get("session_id") or "").strip()
        if not session_id:
            return _err("check_session", "missing required arg 'session_id'")
        status = self.sandbox.check_session(session_id)
        if not status.exists:
            return _err("check_session", status.detail)
        ok = status.running or status.exit_code == 0
        return ToolResult(
            tool="check_session",
            ok=ok,
            payload=_session_payload(status),
            display=_render_session(status),
        )

    def _stop_session(self, args: dict, phase: str) -> ToolResult:
        session_id = str((args or {}).get("session_id") or "").strip()
        if not session_id:
            return _err("stop_session", "missing required arg 'session_id'")
        status = self.sandbox.stop_session(session_id)
        if not status.exists:
            return _err("stop_session", status.detail)
        return ToolResult(
            tool="stop_session",
            ok=True,
            payload=_session_payload(status),
            display=f"Session {status.session_id} terminated.\n" + _render_session(status),
        )


# ── helpers ─────────────────────────────────────────────────────────────────────
def _err(tool: str, message: str) -> ToolResult:
    return ToolResult(tool=tool, ok=False, payload={"error": message}, display=f"ERROR: {message}")


def _read_after_format(target: Path, fallback: str) -> str:
    """Re-read *target* after the formatting hook (Feature 4); *fallback* (the
    content this call itself just wrote) covers a formatter that deleted the
    file or a transient read error — writing/patching must never fail here."""
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return fallback


def _differs_only_in_whitespace(content: str, old: str) -> bool:
    """True if *old* matches *content* at exactly one spot once every whitespace
    run is collapsed — i.e. the only mismatch is indentation/tabs/newlines."""
    def norm(s: str) -> str:
        return " " + " ".join(s.split()) + " "
    return norm(content).count(norm(old)) == 1


def _event_payload(result: ToolResult) -> dict:
    """Pick the fields worth putting on the tool_result event (trim big stdout)."""
    p = dict(result.payload)
    for key in ("stdout", "stderr", "output"):
        if isinstance(p.get(key), str) and len(p[key]) > 2000:
            p[key] = p[key][:2000] + " …<truncated>"
    return p


def _session_payload(status: SessionStatus) -> dict:
    return {
        "session_id": status.session_id,
        "running": status.running,
        "exit_code": status.exit_code,
        "uptime": round(status.uptime, 1),
        "cmd": status.cmd,
        "output": status.output,
    }


def _render_session(status: SessionStatus) -> str:
    if status.running:
        head = f"[session {status.session_id}] RUNNING for {status.uptime:.1f}s: {status.cmd}"
    else:
        head = f"[session {status.session_id}] EXITED with code {status.exit_code}: {status.cmd}"
    body = status.output.strip() or "(no output captured yet)"
    return f"{head}\n--- captured output (tail) ---\n{body}"


def _render_run(cmd: str, result, background: bool) -> str:
    tag = "background check" if background else "command"
    out = [f"$ {cmd}", f"[{tag}] exit_code={result.exit_code} duration={result.duration:.2f}s"]
    if result.stdout:
        out.append(f"--- stdout ---\n{result.stdout}")
    if result.stderr:
        out.append(f"--- stderr ---\n{result.stderr}")
    if not result.stdout and not result.stderr:
        out.append("(no output)")
    if result.exit_code == 126 and not getattr(result, "rejected", False):
        out.append(
            "[hint] exit 126 means the file was FOUND but is NOT EXECUTABLE — retrying the same "
            "path will not help. This is usually a missing execute-permission bit (common for "
            "npm-installed node_modules/.bin/* scripts, or a script you wrote yourself without "
            "chmod). Instead, run it through its interpreter directly, e.g. `node "
            "node_modules/.bin/<tool>` or `npx <tool>` for a Node script, `sh <script>` for a "
            "shell script, or `chmod +x <path>` first."
        )
    return "\n".join(out)
