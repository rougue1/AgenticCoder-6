"""Event taxonomy + async pub/sub EventBus (redesign).

The orchestrator runs in a worker thread and publishes events synchronously via
:meth:`EventBus.emit`. The SSE endpoint runs on the server's asyncio loop and
consumes them through per-subscriber ``asyncio.Queue`` objects. ``emit`` bridges
the two by scheduling delivery onto the bound loop with ``call_soon_threadsafe``.

Event types use the dotted redesign taxonomy (``pipeline.start``,
``manager.call_start``, ``task.blocked``, …). The CLI renderer imports the
*constant names* below (``ev.STAGE_START`` etc.) rather than string literals, so
the legacy constant names are kept as aliases bound to the new taxonomy values —
the renderer keeps working untouched while every wire event carries the new name.

Two event families sit outside the lifecycle taxonomy by design:

* ``llm_request`` / ``llm_token`` / ``llm_thinking_token`` / ``llm_complete`` —
  the raw model-stream layer (fires for BOTH tiers; powers the live renderer).
  Token events are stream-only and never persisted.
* ``file_written`` — emitted by ``write_file``/``patch_file`` per the tool spec.

``run.log`` is a HUMAN-READABLE append-only log: one formatted line per event
(timestamp, type, phase, compact data), not raw JSON. Token events are excluded
so the log stays bounded; full prompts/responses live in ``.agent/llm_calls/``.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Redesign taxonomy ──────────────────────────────────────────────────────────
PIPELINE_START = "pipeline.start"
PIPELINE_COMPLETE = "pipeline.complete"
PIPELINE_CANCELLED = "pipeline.cancelled"
PIPELINE_PAUSED = "pipeline.paused"
PIPELINE_RESUMED = "pipeline.resumed"

STAGE_START = "stage.start"
STAGE_END = "stage.end"

MANAGER_CALL_START = "manager.call_start"
MANAGER_CALL_END = "manager.call_end"
MANAGER_HANDOFF_READY = "manager.handoff_ready"

WORKER_CALL_START = "worker.call_start"
WORKER_TOOL_CALL = "worker.tool_call"
WORKER_TOOL_RESULT = "worker.tool_result"
WORKER_FIX_ATTEMPT = "worker.fix_attempt"

SUMMARIZER_START = "summarizer.start"
SUMMARIZER_FILE_COMPLETE = "summarizer.file_complete"

TASK_START = "task.start"
TASK_DONE = "task.done"
TASK_BLOCKED = "task.blocked"
TASK_DECOMPOSED = "task.decomposed"
# Taxonomy extension (same dotted style): escalation is a first-class ladder
# rung and needs its own lifecycle event.
TASK_ESCALATED = "task.escalated"

TEST_RUN = "test.run"
TEST_PASSED = "test.passed"
TEST_FAILED = "test.failed"

SANDBOX_COMMAND_REJECTED = "sandbox.command_rejected"
SANDBOX_TIMEOUT = "sandbox.timeout"

PREFLIGHT_CHECK = "preflight.check"
PREFLIGHT_PASSED = "preflight.passed"
PREFLIGHT_FAILED = "preflight.failed"

ENVIRONMENT_SETUP_START = "environment.setup_start"
ENVIRONMENT_SETUP_COMPLETE = "environment.setup_complete"

# ── Model-stream layer (both tiers; not lifecycle events) ─────────────────────
LLM_REQUEST = "llm_request"
LLM_TOKEN = "llm_token"
LLM_THINKING_TOKEN = "llm_thinking_token"
LLM_COMPLETE = "llm_complete"

# ── Tool-spec events ───────────────────────────────────────────────────────────
FILE_WRITTEN = "file_written"

# ── Generic ───────────────────────────────────────────────────────────────────
ERROR = "error"
LOG = "log"  # human-readable status line (renderer convenience)

# ── Legacy constant-name aliases (the CLI renderer imports these NAMES; their
#    VALUES are the new taxonomy, so the untouched renderer stays live) ─────────
TOOL_CALL = WORKER_TOOL_CALL
TOOL_RESULT = WORKER_TOOL_RESULT
SUBTASK_START = TASK_START
SUBTASK_DONE = TASK_DONE
SUBTASK_FAILED = WORKER_FIX_ATTEMPT
ESCALATION = TASK_ESCALATED
BLOCKED = TASK_BLOCKED
# No compression events exist in the redesign (the HandoffBuilder trims by
# priority instead); the names stay defined so the renderer's elif chain,
# which references them, keeps importing. Nothing emits them.
COMPRESSION = "compression"
COMPRESSION_FAILURE = "compression_failure"

# Terminal events after which an SSE consumer may safely stop.
TERMINAL_TYPES = {PIPELINE_COMPLETE, PIPELINE_CANCELLED}

# High-frequency token events are streamed live to SSE subscribers but NOT
# persisted to run.log — otherwise the log balloons to megabytes per run.
STREAM_ONLY_TYPES = {LLM_TOKEN, LLM_THINKING_TOKEN}

# Largest string value shown per key in a run.log line.
_LOG_VALUE_LIMIT = 400


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Event:
    """One pipeline event. Serialized to the SSE stream and run.log."""

    type: str
    phase: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class EventBus:
    """Thread-safe async pub/sub with a durable human-readable log sink."""

    def __init__(self, run_log_path: str | Path | None = None):
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._log_lock = threading.Lock()
        self._run_log_path: Path | None = Path(run_log_path) if run_log_path else None
        self._buffer: list[str] = []  # lines emitted before the log path was set

    # ── loop binding (called from the server's startup, on its loop) ──────────
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def set_log_path(self, path: str | Path) -> None:
        """Point run.log at *path* and flush any buffered lines."""
        with self._log_lock:
            self._run_log_path = Path(path)
            self._run_log_path.parent.mkdir(parents=True, exist_ok=True)
            if self._buffer:
                with self._run_log_path.open("a", encoding="utf-8") as fh:
                    fh.write("\n".join(self._buffer) + "\n")
                self._buffer.clear()

    # ── subscription (called on the server loop) ──────────────────────────────
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    # ── publishing (called from any thread) ───────────────────────────────────
    def emit(self, type_: str, phase: str = "", **data: Any) -> Event:
        event = Event(type=type_, phase=phase, data=data)
        self._append_log(event)
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._dispatch, event)
            except RuntimeError:
                pass  # loop shutting down; durable log already has the event
        return event

    def _dispatch(self, event: Event) -> None:
        # Runs on the server loop thread; put_nowait is safe here.
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - unbounded queues
                pass

    def _append_log(self, event: Event) -> None:
        if event.type in STREAM_ONLY_TYPES:
            return  # streamed live, never persisted (keeps run.log bounded)
        line = _format_human(event)
        with self._log_lock:
            if self._run_log_path is None:
                self._buffer.append(line)
                return
            with self._run_log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    # ── typed emit helpers (thin, for readability at call sites) ──────────────
    def stage_start(self, phase: str, title: str = "", **extra: Any) -> Event:
        return self.emit(STAGE_START, phase, title=title, **extra)

    def stage_end(self, phase: str, **extra: Any) -> Event:
        return self.emit(STAGE_END, phase, **extra)

    def llm_request(self, phase: str, model: str, prompt_token_estimate: int, **extra: Any) -> Event:
        return self.emit(LLM_REQUEST, phase, model=model, prompt_token_estimate=prompt_token_estimate, **extra)

    def llm_token(self, phase: str, token: str) -> Event:
        return self.emit(LLM_TOKEN, phase, token=token)

    def llm_thinking_token(self, phase: str, token: str) -> Event:
        return self.emit(LLM_THINKING_TOKEN, phase, token=token)

    def llm_complete(self, phase: str, total_tokens: int, **extra: Any) -> Event:
        return self.emit(LLM_COMPLETE, phase, total_tokens=total_tokens, **extra)

    def tool_call(self, phase: str, tool: str, args: dict[str, Any]) -> Event:
        return self.emit(WORKER_TOOL_CALL, phase, tool=tool, args=_safe_args(args))

    def tool_result(self, phase: str, tool: str, **result: Any) -> Event:
        return self.emit(WORKER_TOOL_RESULT, phase, tool=tool, **result)

    def file_written(self, phase: str, path: str, action: str, content: str = "") -> Event:
        # `content` is the complete file body so a UI can stream/highlight it
        # live without a follow-up fetch. The run.log copy is trimmed.
        return self.emit(FILE_WRITTEN, phase, path=path, action=action, content=content)

    def test_run(self, phase: str, cmd: str, exit_code: int, passed: bool, output: str = "") -> Event:
        event = self.emit(TEST_RUN, phase, cmd=cmd, exit_code=exit_code, passed=passed, output=output)
        self.emit(TEST_PASSED if passed else TEST_FAILED, phase, cmd=cmd, exit_code=exit_code)
        return event

    def error(self, message: str, context: str = "", phase: str = "") -> Event:
        return self.emit(ERROR, phase, message=message, context=context)

    def log(self, message: str, phase: str = "", level: str = "info") -> Event:
        return self.emit(LOG, phase, message=message, level=level)


def _safe_args(args: dict[str, Any], limit: int = 600) -> dict[str, Any]:
    """Trim very large string values (e.g. write_file content) for event payloads."""
    out: dict[str, Any] = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > limit:
            out[k] = v[:limit] + f"... <+{len(v) - limit} chars>"
        else:
            out[k] = v
    return out


def _format_human(event: Event) -> str:
    """One readable run.log line: ``<ts> [<type>] (<phase>) key=value …``.

    ``log``/``error`` events print their message as plain prose (multi-line
    messages are indented so blocks like the model-resolution summary stay
    readable); other events render compact key=value pairs with long strings
    elided.
    """
    ts = event.timestamp
    head = f"{ts} [{event.type}]" + (f" ({event.phase})" if event.phase else "")
    data = event.data or {}

    if event.type in (LOG, ERROR):
        msg = str(data.get("message", ""))
        if "\n" in msg:
            msg = msg.replace("\n", "\n    ")
        extra = ""
        if event.type == ERROR and data.get("context"):
            ctx = str(data["context"])[-_LOG_VALUE_LIMIT:]
            extra = f"\n    context: {ctx}".replace("\n", "\n    ")
        level = data.get("level")
        tag = f" {level.upper()}:" if isinstance(level, str) and level not in ("", "info") else ""
        return f"{head}{tag} {msg}{extra}"

    parts: list[str] = []
    for k, v in data.items():
        if isinstance(v, str):
            v = v.replace("\n", "\\n")
            if len(v) > _LOG_VALUE_LIMIT:
                v = v[:_LOG_VALUE_LIMIT] + f"... <+{len(v) - _LOG_VALUE_LIMIT} chars>"
            parts.append(f'{k}="{v}"')
        else:
            try:
                parts.append(f"{k}={json.dumps(v, default=str)}")
            except (TypeError, ValueError):
                parts.append(f"{k}={v!r}")
    return f"{head} {' '.join(parts)}".rstrip()
