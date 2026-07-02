"""Live terminal renderer (spec §15).

Subscribes to the server's ``/events`` SSE stream and renders the run live with
``rich``: current stage, streaming LLM tokens (thinking vs output shown
differently), tool calls + results, test outcomes, subtask progress, escalations,
and a final summary. It holds NO orchestration logic — it is purely a consumer of
the event stream, proving the event layer is sufficient for a future frontend.
"""

from __future__ import annotations

import json
import time
from collections import deque
from typing import Callable, Optional

import httpx
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Event types (kept in sync with server.events).
from server import events as ev

_OUTPUT_KEEP = 2400  # chars of streamed output kept on screen
_THINK_KEEP = 1200
_LOG_KEEP = 14


class Renderer:
    def __init__(self, base_url: str, console: Optional[Console] = None):
        self.base_url = base_url.rstrip("/")
        self.console = console or Console()
        self.phase = "starting"
        self.title = ""
        self.output = ""
        self.thinking = ""
        self.log: deque[Text] = deque(maxlen=_LOG_KEEP)
        self.counts = {"files": 0, "tests": 0, "subtasks_done": 0, "blocked": 0, "escalations": 0}
        self.model = ""
        self.finished = False
        self.final: dict = {}
        # Live generation throughput (tok/s). gen_chars/4 ≈ tokens; the authoritative
        # value lands on llm_complete (tokens_per_second) and replaces the estimate.
        self.tps = 0.0
        self._gen_start: Optional[float] = None
        self._gen_chars = 0

    # ── connection ────────────────────────────────────────────────────────────
    def run(self, on_connected: Optional[Callable[[], None]] = None) -> dict:
        """Open the SSE stream and render until pipeline_complete. Returns summary."""
        url = f"{self.base_url}/events"
        with Live(self._render(), console=self.console, refresh_per_second=12, screen=False) as live:
            with httpx.Client(timeout=None) as client:
                with client.stream("GET", url) as resp:
                    if on_connected:
                        on_connected()
                    for event in _iter_sse(resp):
                        self._handle(event)
                        live.update(self._render())
                        if event.get("type") in ev.TERMINAL_TYPES:
                            self.finished = True
                            break
            live.update(self._render())
        self._print_final()
        return self.final

    # ── event handling ────────────────────────────────────────────────────────
    def _handle(self, event: dict) -> None:
        etype = event.get("type", "")
        data = event.get("data", {}) or {}
        phase = event.get("phase", "") or self.phase

        if etype == ev.STAGE_START:
            self.phase = phase
            self.title = data.get("title", "")
            self.output = self.thinking = ""
            self._add(f"▶ stage: {phase} — {self.title}", "bold cyan")
        elif etype == ev.STAGE_END:
            self._add(f"■ stage complete: {phase}", "cyan")
        elif etype == ev.LLM_REQUEST:
            self.phase = phase
            self.model = data.get("model", "")
            self.output = self.thinking = ""
            self.tps = 0.0
            self._gen_start = None  # generation clock starts at the first token (set in _tick_tps)
            self._gen_chars = 0
            self._add(f"… {phase}: calling {self.model} (~{data.get('prompt_token_estimate','?')} tok)", "dim")
        elif etype == ev.LLM_TOKEN:
            tok = data.get("token", "")
            self.output = _tail(self.output + tok, _OUTPUT_KEEP)
            self._tick_tps(tok)
        elif etype == ev.LLM_THINKING_TOKEN:
            tok = data.get("token", "")
            self.thinking = _tail(self.thinking + tok, _THINK_KEEP)
            self._tick_tps(tok)
        elif etype == ev.LLM_COMPLETE:
            self.tps = float(data.get("tokens_per_second") or self.tps or 0.0)
            self._gen_start = None
            self._add(
                f"✓ {phase}: model done ({data.get('total_tokens','?')} tok, "
                f"{data.get('duration','?')}s, {self.tps:.1f} tok/s)",
                "dim",
            )
        elif etype == ev.TOOL_CALL:
            self._add(f"🔧 {data.get('tool')} {_brief_args(data.get('args', {}))}", "yellow")
        elif etype == ev.TOOL_RESULT:
            ok = data.get("ok", True)
            extra = f"exit={data['exit_code']}" if "exit_code" in data else ("ok" if ok else "error")
            self._add(f"   ↳ {data.get('tool')} {extra}", "green" if ok else "red")
        elif etype == ev.FILE_WRITTEN:
            self.counts["files"] += 1
            self._add(f"📝 {data.get('action','write')} {data.get('path')}", "green")
        elif etype == ev.TEST_RUN:
            self.counts["tests"] += 1
            passed = data.get("passed")
            self._add(f"🧪 tests {'PASS' if passed else 'FAIL'} — {data.get('cmd','')}", "green" if passed else "red")
        elif etype == ev.SUBTASK_START:
            self._add(f"➤ subtask {data.get('id')} — {data.get('title','')}", "bold magenta")
        elif etype == ev.SUBTASK_DONE:
            self.counts["subtasks_done"] += 1
            self._add(f"✅ subtask {data.get('id')} done", "bold green")
        elif etype == ev.SUBTASK_FAILED:
            self._add(f"✗ subtask {data.get('id')} attempt {data.get('attempt')} failed (exit {data.get('exit_code')})", "red")
        elif etype == ev.ESCALATION:
            self.counts["escalations"] += 1
            self._add(f"⏫ escalating {data.get('id')} ({data.get('escalations_left')} left)", "bold yellow")
        elif etype == ev.BLOCKED:
            self.counts["blocked"] += 1
            self._add(f"⛔ blocked {data.get('id')} after {data.get('attempts')} attempts", "bold red")
        elif etype == ev.COMPRESSION:
            n = len(data.get("summarized_files", []))
            self._add(f"🗜 compressed {n} block(s) to fit budget", "blue")
        elif etype == ev.COMPRESSION_FAILURE:
            self._add(f"🗜! compression failure: {data.get('message','')}", "bold red")
        elif etype == ev.ERROR:
            self._add(f"❌ error: {data.get('message','')}", "bold red")
        elif etype == ev.LOG:
            self._add(f"· {data.get('message','')}", "dim")
        elif etype == ev.PIPELINE_COMPLETE:
            self.final = {**data, "result": data.get("result", "done")}
            self._add(f"🏁 pipeline {self.final['result']}", "bold green")

    # ── rendering ─────────────────────────────────────────────────────────────
    def _render(self) -> Panel:
        header = Table.grid(expand=True)
        header.add_column(justify="left")
        header.add_column(justify="right")
        c = self.counts
        rate = f"  ·  {self.tps:.1f} tok/s" if self.tps > 0 else ""
        header.add_row(
            Text(f"stage: {self.phase}  {self.title}", style="bold cyan"),
            Text(
                f"files {c['files']} · tests {c['tests']} · subtasks ✓{c['subtasks_done']} "
                f"⛔{c['blocked']} ⏫{c['escalations']}{rate}",
                style="dim",
            ),
        )

        stream = Table.grid(expand=True)
        stream.add_column()
        if self.thinking:
            stream.add_row(Text("thinking", style="bold dim"))
            stream.add_row(Text(_last_lines(self.thinking, 6), style="italic grey58"))
        stream.add_row(Text("output", style="bold dim"))
        stream.add_row(Text(_last_lines(self.output, 10) or "…", style="white"))

        activity = Group(*self.log) if self.log else Text("(waiting for events)", style="dim")

        body = Group(
            header,
            Panel(stream, title="model stream", border_style="grey37", padding=(0, 1)),
            Panel(activity, title="activity", border_style="grey37", padding=(0, 1)),
        )
        return Panel(body, title="AIForge — autonomous build", border_style="cyan")

    def _print_final(self) -> None:
        result = self.final.get("result", "unknown")
        style = {"done": "bold green", "error": "bold red", "cancelled": "bold yellow"}.get(result, "bold")
        table = Table(title="Run Summary", show_header=False, border_style=style)
        table.add_row("result", Text(result, style=style))
        if "project_dir" in self.final:
            table.add_row("project_dir", str(self.final["project_dir"]))
        if "elapsed" in self.final:
            table.add_row("elapsed", f"{self.final['elapsed']}s")
        tasks = self.final.get("tasks")
        if isinstance(tasks, dict):
            table.add_row("subtasks", ", ".join(f"{k}={v}" for k, v in tasks.items()))
        if self.final.get("message"):
            table.add_row("message", Text(str(self.final["message"]), style="red"))
        table.add_row("files written", str(self.counts["files"]))
        self.console.print(table)

    def _add(self, text: str, style: str = "") -> None:
        self.log.append(Text(text, style=style))

    def _tick_tps(self, token: str) -> None:
        """Update the live tok/s estimate from streamed chars (~4 chars/token)."""
        if self._gen_start is None:
            self._gen_start = time.monotonic()
            self._gen_chars = 0
        self._gen_chars += len(token)
        elapsed = time.monotonic() - self._gen_start
        if elapsed > 0.4:  # ignore the first burst so the rate isn't wildly spiky
            self.tps = (self._gen_chars / 4) / elapsed


# ── SSE parsing ────────────────────────────────────────────────────────────────
def _iter_sse(resp: httpx.Response):
    """Yield parsed event dicts from an SSE response."""
    etype = None
    data_lines: list[str] = []
    for line in resp.iter_lines():
        if line is None:
            continue
        if line == "":  # dispatch on blank line
            if data_lines:
                raw = "\n".join(data_lines)
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {"type": etype or "log", "data": {"message": raw}}
                if "type" not in payload and etype:
                    payload["type"] = etype
                yield payload
            etype, data_lines = None, []
            continue
        if line.startswith(":"):  # comment / keepalive
            continue
        if line.startswith("event:"):
            etype = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())


def _tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[-limit:]


def _last_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _brief_args(args: dict) -> str:
    if not args:
        return ""
    if "cmd" in args:
        return f"`{args['cmd']}`"
    if "path" in args:
        return str(args["path"])
    return json.dumps(args, default=str)[:80]
