"""Tier-based LiteLLM streaming client for Ollama (redesign).

Two named tiers — ``manager`` and ``worker`` — each backed by a
:class:`llm.resolution.RuntimeModelConfig` resolved once at startup. Every call
site passes the tier plus a human ``phase`` label; the client:

* picks the tier's model, temperature, and effective ``num_ctx``;
* drives thinking via Ollama's native ``think`` parameter (only when the model
  advertises the capability and the tier's resolved ``thinking_enabled`` is on);
* streams tokens to the EventBus, splitting reasoning (``reasoning_content``
  deltas AND inline ``<think>`` tags) from output;
* enforces the **one-model-resident** memory policy: keep_alive keeps the same
  model warm across consecutive calls (the worker's tool loop), while a switch
  to a different model unloads the old one first so two models never stack in
  16GB VRAM (``evict_on_model_switch``);
* emits ``manager.call_start``/``manager.call_end`` lifecycle events for
  manager-tier calls on top of the raw ``llm_request``/``llm_complete`` stream
  events both tiers get;
* dumps each call as one JSONL record under ``.agent/llm_calls/<tier>/``
  (manager: one file per call; worker: appended to a per-subtask session file).
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from config import MANAGER, AppConfig
from llm.resolution import RuntimeModelConfig
from tokens import estimate_messages_tokens, estimate_tokens

if TYPE_CHECKING:
    from server.events import EventBus
    from workspace import Workspace


class LLMError(RuntimeError):
    """Raised when an LLM call fails (network, model missing, stalled stream…)."""


@dataclass
class CompletionResult:
    text: str = ""        # the model's final answer (thinking stripped out)
    thinking: str = ""    # accumulated reasoning content
    raw: str = ""         # everything, in order, as produced
    model: str = ""
    tier: str = ""
    phase: str = ""
    total_tokens: int = 0
    duration: float = 0.0      # whole call (model load + prefill + generation)
    gen_duration: float = 0.0  # first token -> last token (for tok/s)

    def __bool__(self) -> bool:
        return bool(self.text.strip() or self.raw.strip())


class ThinkSplitter:
    """Incrementally split a token stream into (output, thinking) segments.

    Handles ``<think>``/``</think>`` tags that arrive split across chunks by
    holding back a small tail that could be the start of a tag. Path B for
    models whose template inlines thinking instead of using the native field.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self.in_think = False
        self.buf = ""

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        """Return a list of ``(kind, text)`` where kind is 'output' or 'think'."""
        self.buf += chunk
        out: list[tuple[str, str]] = []
        while True:
            if not self.in_think:
                idx = self.buf.find(self.OPEN)
                if idx == -1:
                    emit, self.buf = _split_keep_tail(self.buf, self.OPEN)
                    if emit:
                        out.append(("output", emit))
                    break
                if idx > 0:
                    out.append(("output", self.buf[:idx]))
                self.buf = self.buf[idx + len(self.OPEN):]
                self.in_think = True
            else:
                idx = self.buf.find(self.CLOSE)
                if idx == -1:
                    emit, self.buf = _split_keep_tail(self.buf, self.CLOSE)
                    if emit:
                        out.append(("think", emit))
                    break
                if idx > 0:
                    out.append(("think", self.buf[:idx]))
                self.buf = self.buf[idx + len(self.CLOSE):]
                self.in_think = False
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Emit any held-back tail at end of stream."""
        if not self.buf:
            return []
        kind = "think" if self.in_think else "output"
        out = [(kind, self.buf)]
        self.buf = ""
        return out


def _split_keep_tail(buf: str, tag: str) -> tuple[str, str]:
    """Return (emittable, held_tail) keeping back a possible partial *tag*."""
    keep = len(tag) - 1
    for k in range(keep, 0, -1):
        if buf.endswith(tag[:k]):
            return buf[:-k], buf[-k:]
    return buf, ""


class LLMClient:
    def __init__(self, config: AppConfig, bus: "EventBus", workspace: "Workspace | None" = None):
        self.config = config
        self.bus = bus
        self.workspace = workspace
        # Set once by the orchestrator after llm.resolution.resolve_all(). All
        # model parameters are read from here — never from raw config fields.
        self.runtime: dict[str, RuntimeModelConfig] = {}
        # The model Ollama currently holds warm (the last one we called). Used to
        # enforce "at most one model resident at a time": a DIFFERENT model is
        # unloaded before the next loads, a re-called model stays warm.
        self._resident_model: str | None = None
        self._mem_lock = threading.Lock()

    def set_workspace(self, workspace: "Workspace") -> None:
        self.workspace = workspace

    def set_runtime(self, runtime: dict[str, RuntimeModelConfig]) -> None:
        self.runtime = dict(runtime)

    def runtime_for(self, tier: str) -> RuntimeModelConfig:
        rmc = self.runtime.get(tier)
        if rmc is None:
            raise LLMError(
                f"no resolved runtime config for tier {tier!r} — model resolution must run "
                "before any LLM call"
            )
        return rmc

    # ── model memory management (one model resident at a time) ────────────────
    def _ensure_only(self, model: str) -> None:
        """Make *model* the only resident model before a call (when eviction is
        on). A model re-called consecutively (the worker's implement/fix loop)
        stays warm; a switch unloads the previous model first so the 21GB
        manager and the worker can never stack in VRAM/RAM."""
        if not self.config.ollama.evict_on_model_switch:
            return
        with self._mem_lock:
            prev = self._resident_model
            if prev and prev != model:
                self._ollama_unload(prev)
            self._resident_model = model

    def unload_all(self) -> None:
        """Unload the resident model (called at pipeline end so nothing lingers)."""
        if not self.config.ollama.evict_on_model_switch:
            return
        with self._mem_lock:
            if self._resident_model:
                self._ollama_unload(self._resident_model)
                self._resident_model = None

    def _ollama_unload(self, model: str) -> None:
        """Ask Ollama to evict *model* immediately (keep_alive=0). Best-effort.

        Skips the call if the model isn't actually loaded (per ``/api/ps``) so
        we never reload a model just to unload it."""
        name = (model or "").split("/", 1)[-1]  # tolerate a litellm-prefixed name
        if not name:
            return
        base = self.config.ollama.host
        try:
            ps = httpx.get(f"{base}/api/ps", timeout=10)
            if ps.status_code == 200:
                loaded = {str(m.get("name") or m.get("model") or "") for m in (ps.json().get("models") or [])}
                if name not in loaded and f"{name}:latest" not in loaded:
                    return  # already evicted by Ollama's own keep_alive
        except (httpx.HTTPError, ValueError, KeyError):
            pass  # can't check — fall through and try the unload anyway
        try:
            httpx.post(f"{base}/api/generate", json={"model": name, "keep_alive": 0}, timeout=60)
            self.bus.log(f"evicted {name} to free memory before loading a different model", phase="setup")
        except httpx.HTTPError:
            pass  # best-effort; a failed unload shouldn't break the run

    # ── main entry ────────────────────────────────────────────────────────────
    def complete(
        self,
        tier: str,
        phase: str,
        messages: list[dict],
        *,
        stream: bool = True,
        temperature: float | None = None,
        dump: bool = True,
        dump_path: Path | None = None,
    ) -> CompletionResult:
        """One completion for *tier*, labeled *phase* on every event.

        ``dump_path`` (worker sessions) appends the call record to that file;
        otherwise a fresh ``<timestamp>_<phase>.jsonl`` is created per call.
        """
        rmc = self.runtime_for(tier)
        model = f"ollama_chat/{rmc.model}"
        temp = rmc.temperature if temperature is None else temperature
        prepared = [_clean_message(m) for m in messages]

        prompt_tokens = estimate_messages_tokens(prepared)
        self.bus.llm_request(phase, rmc.model, prompt_tokens, tier=tier)
        if tier == MANAGER:
            from server import events

            self.bus.emit(events.MANAGER_CALL_START, phase, model=rmc.model, prompt_token_estimate=prompt_tokens)

        # Evict a different warm model first so two models never stack (OOM/EOF).
        self._ensure_only(rmc.model)

        start = time.monotonic()
        try:
            result = (
                self._stream(phase, model, rmc, prepared, temp)
                if stream
                else self._oneshot(phase, model, rmc, prepared, temp)
            )
        except LLMError:
            raise
        except Exception as exc:  # normalize any provider error
            self.bus.error(f"LLM call failed: {exc}", context=f"tier={tier} phase={phase} model={rmc.model}", phase=phase)
            raise LLMError(f"{phase}: {exc}") from exc

        result.duration = time.monotonic() - start
        result.model = rmc.model
        result.tier = tier
        result.phase = phase
        result.total_tokens = estimate_tokens(result.raw)
        # Raw generation throughput: tokens / generation time (first->last token)
        # so decode speed isn't diluted by load/prefill (models are evicted +
        # reloaded on switch). Falls back to wall time for the one-shot path.
        gen_dur = result.gen_duration or result.duration
        tps = round(result.total_tokens / gen_dur, 1) if gen_dur > 0 else 0.0
        self.bus.llm_complete(
            phase, result.total_tokens, duration=round(result.duration, 2), tokens_per_second=tps, tier=tier
        )
        if tier == MANAGER:
            from server import events

            self.bus.emit(
                events.MANAGER_CALL_END,
                phase,
                model=rmc.model,
                total_tokens=result.total_tokens,
                duration=round(result.duration, 2),
                tokens_per_second=tps,
            )

        if dump and self.config.dump_llm_calls:
            self._dump(tier, phase, rmc.model, prepared, result, dump_path)
        return result

    # ── request options ───────────────────────────────────────────────────────
    def _request_kwargs(self, rmc: RuntimeModelConfig, temperature: float) -> dict:
        kwargs: dict = {
            "api_base": self.config.ollama.host,
            "api_key": "ollama",
            "temperature": temperature,
            "num_ctx": rmc.num_ctx,
            # keep_alive keeps THIS model warm for the next same-model call; a
            # different model evicts it explicitly first (_ensure_only), so the
            # effective policy is dynamic rather than a fixed TTL.
            "keep_alive": self.config.ollama.keep_alive,
            "timeout": self.config.ollama.request_timeout,
        }
        # Only models that advertise the capability get the native think flag —
        # sending it to a non-thinking model is an Ollama 400.
        if rmc.supports_think_param:
            kwargs["think"] = rmc.thinking_enabled
        return kwargs

    # ── streaming ─────────────────────────────────────────────────────────────
    def _stream(
        self, phase: str, model: str, rmc: RuntimeModelConfig, messages: list[dict], temperature: float
    ) -> CompletionResult:
        import litellm

        splitter = ThinkSplitter()
        text_parts: list[str] = []
        think_parts: list[str] = []
        raw_parts: list[str] = []

        response = litellm.completion(
            model=model,
            messages=messages,
            stream=True,
            **self._request_kwargs(rmc, temperature),
        )
        idle = max(30, int(self.config.ollama.stream_idle_timeout))
        first_tok: float | None = None
        for chunk in _iter_with_idle_timeout(response, idle):
            content, reasoning = _chunk_parts(chunk)
            if first_tok is None and (content or reasoning):
                first_tok = time.monotonic()  # generation clock starts at the 1st token
            # Path A: LiteLLM/Ollama surface reasoning in a separate field.
            if reasoning:
                raw_parts.append(reasoning)
                think_parts.append(reasoning)
                self.bus.llm_thinking_token(phase, reasoning)
            # Path B: model inlines <think>...</think> in the content stream.
            if content:
                raw_parts.append(content)
                for kind, seg in splitter.feed(content):
                    if not seg:
                        continue
                    if kind == "think":
                        think_parts.append(seg)
                        self.bus.llm_thinking_token(phase, seg)
                    else:
                        text_parts.append(seg)
                        self.bus.llm_token(phase, seg)
        for kind, seg in splitter.flush():
            if not seg:
                continue
            (think_parts if kind == "think" else text_parts).append(seg)
            (self.bus.llm_thinking_token if kind == "think" else self.bus.llm_token)(phase, seg)

        return CompletionResult(
            text="".join(text_parts).strip(),
            thinking="".join(think_parts).strip(),
            raw="".join(raw_parts),
            gen_duration=(time.monotonic() - first_tok) if first_tok is not None else 0.0,
        )

    def _oneshot(
        self, phase: str, model: str, rmc: RuntimeModelConfig, messages: list[dict], temperature: float
    ) -> CompletionResult:
        import litellm

        response = litellm.completion(
            model=model,
            messages=messages,
            stream=False,
            **self._request_kwargs(rmc, temperature),
        )
        message = response.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
        text, inline_think = _strip_think(content)
        thinking = reasoning + ("\n" if reasoning and inline_think else "") + inline_think
        if thinking:
            self.bus.llm_thinking_token(phase, thinking)
        self.bus.llm_token(phase, text)
        return CompletionResult(text=text.strip(), thinking=thinking.strip(), raw=reasoning + content)

    # ── debugging dump (JSONL) ────────────────────────────────────────────────
    def _dump(
        self,
        tier: str,
        phase: str,
        model: str,
        messages: list[dict],
        result: CompletionResult,
        dump_path: Path | None,
    ) -> None:
        if self.workspace is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tier": tier,
            "phase": phase,
            "model": model,
            "duration_s": round(result.duration, 2),
            "total_tokens": result.total_tokens,
            "messages": messages,
            "thinking": result.thinking,
            "response": result.text,
        }
        try:
            if dump_path is None:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
                dump_path = self.workspace.llm_calls_dir / tier / f"{ts}_{phase}.jsonl"
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            with dump_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass

    def session_dump_path(self, tier: str, label: str) -> Path | None:
        """Path for a per-session JSONL dump (worker: one file per subtask)."""
        if self.workspace is None:
            return None
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in label) or "session"
        return self.workspace.llm_calls_dir / tier / f"{ts}_{safe}.jsonl"


def _clean_message(msg: dict) -> dict:
    """Strip pipeline-internal keys (conversation bookkeeping) before sending."""
    return {"role": msg.get("role", "user"), "content": str(msg.get("content", ""))}


def _chunk_parts(chunk) -> tuple[str, str]:
    """Extract (content, reasoning_content) from a LiteLLM streaming chunk.

    LiteLLM exposes Ollama's thinking via ``reasoning_content`` (or
    ``reasoning``) on the delta, separate from the answer in ``content``."""
    try:
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None) or getattr(choice, "message", None)
        if delta is None:
            return "", ""
        content = getattr(delta, "content", None) or ""
        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) or ""
        return content, reasoning
    except (AttributeError, IndexError, TypeError):
        return "", ""


def _strip_think(raw: str) -> tuple[str, str]:
    """Split a full (non-streamed) response into (output, thinking)."""
    splitter = ThinkSplitter()
    text_parts, think_parts = [], []
    for kind, seg in splitter.feed(raw) + splitter.flush():
        (think_parts if kind == "think" else text_parts).append(seg)
    return "".join(text_parts), "".join(think_parts)


def _iter_with_idle_timeout(response, idle: float):
    """Yield streamed chunks, raising :class:`LLMError` if no chunk arrives for
    *idle* seconds.

    This is an INTER-FRAME idle timeout — reset on every chunk — not an
    end-to-end cap. A slow-but-alive local model doing long prefill on a big
    context keeps the stream arriving; only a stream gone genuinely silent
    trips it. The blocking provider generator is drained on a daemon thread
    feeding a bounded queue, and the consumer waits at most *idle* per item.
    """
    q: queue.Queue = queue.Queue(maxsize=64)
    sentinel = object()
    box: dict = {}

    def _produce() -> None:
        try:
            for chunk in response:
                q.put(chunk)
        except Exception as exc:  # surface provider errors to the consumer
            box["err"] = exc
        finally:
            q.put(sentinel)

    threading.Thread(target=_produce, daemon=True, name="aiforge-llm-stream").start()
    while True:
        try:
            item = q.get(timeout=idle)
        except queue.Empty:
            raise LLMError(
                f"stream stalled: no tokens for {idle:.0f}s — the server may be hung "
                f"(raise ollama.stream_idle_timeout if the model legitimately needs longer to prefill)"
            )
        if item is sentinel:
            if "err" in box:
                raise LLMError(str(box["err"]))
            return
        yield item
