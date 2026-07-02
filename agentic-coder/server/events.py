"""Event schema + async pub/sub EventBus (spec §14).

The orchestrator runs in a worker thread and publishes events synchronously via
:meth:`EventBus.emit`. The SSE endpoint runs on the server's asyncio loop and
consumes them through per-subscriber ``asyncio.Queue`` objects. ``emit`` bridges
the two by scheduling delivery onto the bound loop with ``call_soon_threadsafe``.

Every event is also appended to ``.agent/run.log`` as one JSON line. The log
path is unknown until the project dir is resolved, so early events are buffered
in memory and flushed once :meth:`EventBus.set_log_path` is called.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Event type constants (the full schema from spec §14) ──────────────────────
STAGE_START = "stage_start"
STAGE_END = "stage_end"
LLM_REQUEST = "llm_request"
LLM_TOKEN = "llm_token"
LLM_THINKING_TOKEN = "llm_thinking_token"
LLM_COMPLETE = "llm_complete"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
FILE_WRITTEN = "file_written"
TEST_RUN = "test_run"
SUBTASK_START = "subtask_start"
SUBTASK_DONE = "subtask_done"
SUBTASK_FAILED = "subtask_failed"
ESCALATION = "escalation"
BLOCKED = "blocked"
COMPRESSION = "compression"
COMPRESSION_FAILURE = "compression_failure"
PIPELINE_COMPLETE = "pipeline_complete"
PIPELINE_PAUSED = "pipeline_paused"
PIPELINE_RESUMED = "pipeline_resumed"
ERROR = "error"
LOG = "log"  # generic human-readable status line (renderer convenience)

# Terminal events after which an SSE consumer may safely stop.
TERMINAL_TYPES = {PIPELINE_COMPLETE}

# High-frequency token events are streamed live to SSE subscribers but NOT
# persisted to run.log — otherwise the log balloons to megabytes per run. The
# full text is recoverable from llm_complete + the .agent/llm_calls/ dumps.
STREAM_ONLY_TYPES = {LLM_TOKEN, LLM_THINKING_TOKEN}

# Largest string value persisted to run.log. The full payload still reaches live
# SSE subscribers (e.g. file_written carries complete file content for the UI to
# stream); only the durable JSONL copy is trimmed so the log stays bounded.
_LOG_VALUE_LIMIT = 4000


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
    """Thread-safe async pub/sub with a durable JSONL sink."""

    def __init__(self, run_log_path: str | Path | None = None):
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._log_lock = threading.Lock()
        self._run_log_path: Path | None = Path(run_log_path) if run_log_path else None
        self._buffer: list[str] = []  # events emitted before the log path was set

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
        line = json.dumps(_log_safe_dict(event), default=str)
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

    def llm_request(self, phase: str, model: str, prompt_token_estimate: int) -> Event:
        return self.emit(LLM_REQUEST, phase, model=model, prompt_token_estimate=prompt_token_estimate)

    def llm_token(self, phase: str, token: str) -> Event:
        return self.emit(LLM_TOKEN, phase, token=token)

    def llm_thinking_token(self, phase: str, token: str) -> Event:
        return self.emit(LLM_THINKING_TOKEN, phase, token=token)

    def llm_complete(self, phase: str, total_tokens: int, **extra: Any) -> Event:
        return self.emit(LLM_COMPLETE, phase, total_tokens=total_tokens, **extra)

    def tool_call(self, phase: str, tool: str, args: dict[str, Any]) -> Event:
        return self.emit(TOOL_CALL, phase, tool=tool, args=_safe_args(args))

    def tool_result(self, phase: str, tool: str, **result: Any) -> Event:
        return self.emit(TOOL_RESULT, phase, tool=tool, **result)

    def file_written(self, phase: str, path: str, action: str, content: str = "") -> Event:
        # `content` is the complete file body so the UI can stream/syntax-highlight
        # it live without a follow-up fetch. The persisted log copy is trimmed.
        return self.emit(FILE_WRITTEN, phase, path=path, action=action, content=content)

    def test_run(self, phase: str, cmd: str, exit_code: int, passed: bool, output: str = "") -> Event:
        return self.emit(TEST_RUN, phase, cmd=cmd, exit_code=exit_code, passed=passed, output=output)

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


def _log_safe_dict(event: Event) -> dict[str, Any]:
    """Event as a dict with oversized string values trimmed for the durable log.

    Keeps run.log bounded even when an event (e.g. ``file_written``) carries a
    large payload for live SSE consumers; the on-disk copy notes the elision.
    """
    d = event.to_dict()
    data = d.get("data")
    if isinstance(data, dict):
        d["data"] = {
            k: (v[:_LOG_VALUE_LIMIT] + f"... <+{len(v) - _LOG_VALUE_LIMIT} chars>")
            if isinstance(v, str) and len(v) > _LOG_VALUE_LIMIT
            else v
            for k, v in data.items()
        }
    return d
