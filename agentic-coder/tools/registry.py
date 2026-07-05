"""The four Worker tools + dispatch (redesign).

* ``read_file``  ``{path}`` — jailed + .agentignore-checked read.
* ``write_file`` ``{path, content, summary}`` — create/overwrite; ``summary``
  is REQUIRED and recorded in the manifest; emits ``file_written``.
* ``patch_file`` ``{path, old_string, new_string}`` — exact-match single
  replacement with precise diagnostics (zero matches / ambiguous / whitespace).
* ``run``        ``{cmd, background?, timeout?, smoke?}`` — allowlist-validated,
  venv-rewritten execution; background mode for dev servers.

Every dispatch emits ``worker.tool_call`` then ``worker.tool_result``. The
model never runs anything itself — it only requests; this layer validates and
executes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from context.conversation import cap_tool_output
from llm.tool_parser import ToolCall
from tools.process_manager import ProcessManager
from tools.sandbox import Sandbox
from workspace import IgnoredPathError, PathEscapeError, Workspace

if TYPE_CHECKING:
    from context.manifest import Manifest
    from server.events import EventBus

# Description injected into Worker prompts so the model knows the protocol.
# NOTE: the wrapper tag is <agentic_call>, NOT the more obvious <tool_call>.
# ornith (qwen3.5-family) is recognized by Ollama's own renderer/parser, which
# natively scans raw output for the literal <tool_call>...</tool_call> markers
# and tries to lift them into message.tool_calls using ITS OWN schema
# (name/arguments) before we ever see the text. Our schema uses tool/args, so
# Ollama's parser can't find what it expects, hits EOF mid-parse, and the whole
# /api/chat response comes back as {"error": "EOF"} instead of real content
# (litellm then dies with a raw KeyError on the missing "message" key). Using a
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

4. run — run a shell command in the project root. NEVER start a server in the
   foreground (it would hang until killed): for dev servers/watchers set
   "background": true and pass the verification calls as "smoke" commands — the
   server is started in its own process, health-checked, the smoke commands run
   against it concurrently, and it is stopped afterwards.
   <agentic_call>{"tool": "run", "args": {"cmd": "pytest -q", "timeout": 120}}</agentic_call>
   <agentic_call>{"tool": "run", "args": {"cmd": "uvicorn app.main:app --port 8100", "background": true, "smoke": ["curl -s http://127.0.0.1:8100/health"]}}</agentic_call>

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

# Foreground commands that would block forever (dev servers). Auto-routed to the
# background harness so the pipeline never hangs on `uvicorn app:app`.
_SERVER_CMD_RE = re.compile(
    r"^\s*(?:\S*/)?(?:uvicorn|gunicorn|flask\s+run|python\d?\s+-m\s+(?:uvicorn|flask|http\.server)|"
    r"npm\s+(?:run\s+)?(?:dev|start|serve)|npx\s+(?:vite|next|serve)|node\s+\S*server\S*|"
    r"yarn\s+(?:dev|start)|pnpm\s+(?:dev|start)|vite\b|next\s+dev)",
    re.I,
)


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
        process_manager: ProcessManager,
    ):
        self.workspace = workspace
        self.sandbox = sandbox
        self.manifest = manifest
        self.bus = bus
        self.process_manager = process_manager

    # ── dispatch ──────────────────────────────────────────────────────────────
    def dispatch(self, call: ToolCall, phase: str) -> ToolResult:
        self.bus.tool_call(phase, call.name, call.args)
        handler = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "patch_file": self._patch_file,
            "run": self._run,
        }.get(call.name)
        if handler is None:
            result = ToolResult(
                tool=call.name,
                ok=False,
                payload={"error": f"unknown tool {call.name!r}"},
                display=f"ERROR: unknown tool {call.name!r}. Valid tools: read_file, write_file, patch_file, run.",
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
        self.bus.file_written(phase, rel, action, content=content)
        return ToolResult(
            tool="write_file",
            ok=True,
            payload={"path": rel, "action": action, "bytes": len(content)},
            display=f"Wrote {rel} ({action}, {len(content)} bytes).",
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
        self.bus.file_written(phase, rel, "edit", content=updated)
        return ToolResult(
            tool="patch_file",
            ok=True,
            payload={"path": rel, "action": "edit", "bytes": len(updated)},
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
        smoke = args.get("smoke") or args.get("smoke_cmds") or []
        if isinstance(smoke, str):
            smoke = [smoke]

        note = ""
        if not background and _SERVER_CMD_RE.match(cmd.strip()):
            # A foreground dev server would block until the timeout kills it.
            # Route it through the background harness instead so the pipeline
            # never hangs on a long-running process.
            background = True
            note = (
                "\n[note] this looks like a long-running server, so it was started in the "
                "background, health-checked, and stopped. Pass \"smoke\" commands to test it."
            )

        try:
            self.sandbox.validate(cmd)
        except Exception as exc:  # CommandRejected
            reason = getattr(exc, "reason", str(exc))
            return ToolResult(
                tool="run",
                ok=False,
                payload={"cmd": cmd, "exit_code": 126, "rejected": True, "reason": reason},
                display=f"COMMAND REJECTED ({reason}): {cmd}",
            )

        if background:
            result = self.process_manager.run_background_check(
                cmd,
                health_timeout=self.sandbox.limits.long_process_timeout,
                smoke_cmds=list(smoke),
                smoke_timeout=int(timeout) if timeout else self.sandbox.limits.timeout,
            )
        else:
            result = self.sandbox.run(cmd, timeout=int(timeout) if timeout else None)

        display = _render_run(cmd, result, background) + note
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
                "background": background,
            },
            display=display,
        )


# ── helpers ─────────────────────────────────────────────────────────────────────
def _err(tool: str, message: str) -> ToolResult:
    return ToolResult(tool=tool, ok=False, payload={"error": message}, display=f"ERROR: {message}")


def _differs_only_in_whitespace(content: str, old: str) -> bool:
    """True if *old* matches *content* at exactly one spot once every whitespace
    run is collapsed — i.e. the only mismatch is indentation/tabs/newlines."""
    def norm(s: str) -> str:
        return " " + " ".join(s.split()) + " "
    return norm(content).count(norm(old)) == 1


def _event_payload(result: ToolResult) -> dict:
    """Pick the fields worth putting on the tool_result event (trim big stdout)."""
    p = dict(result.payload)
    for key in ("stdout", "stderr"):
        if isinstance(p.get(key), str) and len(p[key]) > 2000:
            p[key] = p[key][:2000] + " …<truncated>"
    return p


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
