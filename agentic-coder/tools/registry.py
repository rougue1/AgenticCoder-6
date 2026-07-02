"""Tool definitions + dispatch (spec §11).

Three tools are exposed to the model and executed by the orchestrator:

* ``read_file`` ``{path}`` -> file contents.
* ``write_file`` ``{path, content, summary}`` -> writes (path-validated), records
  the summary in the manifest, emits ``file_written``.
* ``run`` ``{cmd, background?, timeout?}`` -> executed by the sandbox; returns
  ``{exit_code, stdout, stderr}`` (or a background pass/fail via the harness).

Every dispatch emits ``tool_call`` then ``tool_result``. The model never runs
anything itself — it only requests; this layer validates and executes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from context.conversation import cap_tool_output
from llm.tool_parser import ToolCall
from tools.process_manager import ProcessManager
from tools.sandbox import Sandbox, normalize_pytest_command
from workspace import PathEscapeError, Workspace

if TYPE_CHECKING:
    from context.manifest import Manifest
    from server.events import EventBus

# Description injected into code-gen prompts so the model knows the protocol.
TOOL_INSTRUCTIONS = """\
You act ONLY by emitting tool calls. You never run commands yourself.
Emit ONE tool call at a time, wrapped in <tool_call>...</tool_call> tags with a
JSON body. After each call you will receive the result, then continue.

Available tools:

1. read_file — read an existing file.
   <tool_call>{"tool": "read_file", "args": {"path": "src/app.py"}}</tool_call>

2. write_file — create or overwrite a WHOLE file. ALWAYS include a one-line
   `summary` describing what the file does (used in the project manifest).
   <tool_call>{"tool": "write_file", "args": {"path": "src/app.py", "content": "<full file contents>", "summary": "FastAPI entrypoint"}}</tool_call>

3. edit_file — change an EXISTING file surgically: replace `old_string` (which must
   appear EXACTLY ONCE — include enough surrounding context to make it unique) with
   `new_string`. Prefer this over write_file for any change short of a full rewrite
   (a typo fix, a single line, swapping a function body); a whole-file rewrite is a
   fresh chance to inject a typo that breaks the file.
   <tool_call>{"tool": "edit_file", "args": {"path": "src/app.py", "old_string": "def add(a, b):\n    return a - b", "new_string": "def add(a, b):\n    return a + b"}}</tool_call>

4. run — run a shell command in the project directory. Set "background": true for
   dev servers/watchers (it will be started, health-checked, then stopped).
   <tool_call>{"tool": "run", "args": {"cmd": "npm test", "timeout": 120}}</tool_call>

Rules:
- All paths are relative to the project root. Never use absolute paths, "..", "~",
  or git in any form.
- EXPLORE before writing when unsure: run `ls -R` (or `find . -type f`) and
  read_file the files you are about to depend on, instead of guessing their
  contents.
- Installing packages requires actually RUNNING the installer: after you write
  requirements.txt (or package.json) run it — e.g. run `pip install -r
  requirements.txt`. Writing the manifest file does not install anything.
- write_file writes COMPLETE file contents, never diffs or "... unchanged ...".
- To change a file that already exists, prefer edit_file over rewriting it whole.
- For a LARGE new file (more than a few hundred lines) build it with run + bash
  heredoc appends from the FIRST call — `run` with `cat > path <<'EOF' ... EOF`,
  then a `cat >> path <<'EOF' ... EOF` per part — because a single huge write_file
  body can be truncated mid-stream by the server. Verify with `wc -c path`.
- When you are completely done with this step, reply with the single line: DONE
"""

# Appended to a write_file/edit_file result when the call had to be salvaged from
# malformed JSON — almost always because the server truncated the streamed
# arguments at its output-token limit, leaving a possibly-incomplete file. Mirrors
# codehamr's truncated-args recovery message (spec adoption #4).
_TRUNCATED_ARGS_NOTE = (
    "\n\nNOTE: this call's arguments looked truncated (the server likely cut them at "
    "its output-token limit), so the file may be INCOMPLETE. Do NOT retry the same "
    "whole-file write. Verify with run `wc -c <path>` (and read the tail); if it is "
    "short, rebuild it in parts with bash heredoc appends: run `cat > <path> <<'EOF' "
    "… EOF` for the first part, then a `cat >> <path> <<'EOF' … EOF` per following part."
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
            "edit_file": self._edit_file,
            "run": self._run,
        }.get(call.name)
        if handler is None:
            result = ToolResult(
                tool=call.name,
                ok=False,
                payload={"error": f"unknown tool {call.name!r}"},
                display=f"ERROR: unknown tool {call.name!r}. Valid tools: read_file, write_file, run.",
            )
        else:
            result = handler(call.args, phase)
        # A salvaged call means the parser had to reconstruct the arguments — the
        # strong signal the server truncated them mid-stream, so a write may be
        # incomplete. Tell the model how to recover instead of silently accepting a
        # half-written file and letting it re-emit the same too-large write (#4).
        if getattr(call, "salvaged", False) and result.tool in ("write_file", "edit_file") and result.ok:
            result.display = result.display.rstrip() + _TRUNCATED_ARGS_NOTE
        # Cap the text fed back into the ephemeral conversation so one noisy read
        # or test log can't dominate a local model's window (the full payload
        # still reaches the event stream / UI via _event_payload below).
        result.display = cap_tool_output(result.display)
        self.bus.tool_result(phase, result.tool, ok=result.ok, **_event_payload(result))
        return result

    # ── read_file ─────────────────────────────────────────────────────────────
    def _read_file(self, args: dict, phase: str) -> ToolResult:
        path = (args or {}).get("path")
        if not path:
            return _err("read_file", "missing required arg 'path'")
        try:
            target = self.workspace.resolve_in_root(path)
        except PathEscapeError as exc:
            return _err("read_file", str(exc))
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
        summary = args.get("summary") or args.get("description") or ""
        if not path:
            return _err("write_file", "missing required arg 'path'")
        if content is None:
            return _err("write_file", "missing required arg 'content'")
        if not isinstance(content, str):
            content = str(content)
        try:
            target = self.workspace.resolve_in_root(path)
        except PathEscapeError as exc:
            return _err("write_file", str(exc))

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

    # ── edit_file ─────────────────────────────────────────────────────────────
    def _edit_file(self, args: dict, phase: str) -> ToolResult:
        """Surgical single-anchor replace on an existing file (spec adoption #3).

        ``old_string`` must appear EXACTLY ONCE. Errors (not found, ambiguous,
        missing file) come back in ``display`` like every other tool, so the model
        reacts the way it does to a non-zero exit. Preferred over ``write_file``
        for any change short of a full rewrite: a whole-file rewrite is a fresh
        chance to inject a typo that dead-stops the file.
        """
        args = args or {}
        path = args.get("path")
        old = args.get("old_string")
        new = args.get("new_string")
        if not path:
            return _err("edit_file", "missing required arg 'path'")
        if old is None or old == "":
            return _err("edit_file", "missing/empty 'old_string' (need a unique anchor to replace)")
        if new is None:
            return _err("edit_file", "missing required arg 'new_string' (use \"\" to delete the anchor)")
        if not isinstance(old, str):
            old = str(old)
        if not isinstance(new, str):
            new = str(new)
        if old == new:
            return _err("edit_file", "no change: old_string equals new_string")
        try:
            target = self.workspace.resolve_in_root(path)
        except PathEscapeError as exc:
            return _err("edit_file", str(exc))
        if not target.exists():
            return _err("edit_file", f"file not found: {path} (use write_file to create it)")
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return _err("edit_file", f"could not read {path}: {exc}")

        n = content.count(old)
        if n == 0:
            # A near-miss differing only in whitespace (indentation/tabs/newlines)
            # is the most common edit failure for a local model — name it so it
            # fixes the bytes instead of burning retries.
            if _differs_only_in_whitespace(content, old):
                return _err(
                    "edit_file",
                    f"no exact match in {path} — a block there differs only in whitespace "
                    f"(indentation/tabs/newlines); copy the exact bytes, including indentation",
                )
            return _err("edit_file", f"old_string not found in {path} (copy the exact bytes to anchor on)")
        if n > 1:
            return _err("edit_file", f"old_string appears {n}× in {path} — add surrounding context to make it unique")
        # str.count is non-overlapping, so a self-overlapping anchor ("==" in
        # "a === b") can pass n==1 yet match two ways. Reject the overlap.
        idx = content.find(old)
        if content.find(old, idx + 1) != -1:
            return _err("edit_file", "old_string overlaps itself — add surrounding context to make it unique")

        updated = content[:idx] + new + content[idx + len(old):]
        try:
            target.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return _err("edit_file", f"could not write {path}: {exc}")

        rel = self.workspace.relative(path)
        self.manifest.record(rel, args.get("summary") or args.get("description"))
        self.bus.file_written(phase, rel, "edit", content=updated)
        return ToolResult(
            tool="edit_file",
            ok=True,
            payload={"path": rel, "action": "edit", "bytes": len(updated)},
            display=f"Edited {rel} (-{len(old)} +{len(new)} bytes).",
        )

    # ── run ───────────────────────────────────────────────────────────────────
    def _run(self, args: dict, phase: str) -> ToolResult:
        args = args or {}
        cmd = args.get("cmd") or args.get("command")
        if not cmd:
            return _err("run", "missing required arg 'cmd'")
        cmd = normalize_pytest_command(cmd)  # bare pytest -> python -m pytest (importable root)
        background = bool(args.get("background", False))
        timeout = args.get("timeout")
        smoke = args.get("smoke") or args.get("smoke_cmds") or []
        if isinstance(smoke, str):
            smoke = [smoke]

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
                smoke_timeout=int(timeout) if timeout else self.sandbox.limits.sandbox_timeout,
            )
        else:
            result = self.sandbox.run(cmd, timeout=int(timeout) if timeout else None)

        display = _render_run(cmd, result, background)
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
    run is collapsed — i.e. the only mismatch is indentation/tabs/newlines. Padded
    with spaces so a match can't straddle a token boundary and mislabel an
    unrelated near-miss. Mirrors codehamr's differsOnlyInWhitespace."""
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
    return "\n".join(out)
