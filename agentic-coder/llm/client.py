"""LiteLLM streaming wrapper with per-phase model selection (spec §14).

Responsibilities:

* pick the model + thinking mode for a phase (from :class:`config.AppConfig`);
* apply the qwen3 ``/think`` / ``/no_think`` suffix where required;
* stream tokens to the EventBus, splitting ``<think>...</think>`` reasoning into
  ``llm_thinking_token`` events and everything else into ``llm_token`` events
  (tags may be split across stream chunks — handled by :class:`ThinkSplitter`);
* emit ``llm_request`` / ``llm_complete`` and optionally dump prompt+response to
  ``.agent/llm_calls/`` for debugging.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx

from config import (
    THINK_DEEPSEEK,
    THINK_NONE,
    THINK_QWEN3_OFF,
    THINK_QWEN3_ON,
    AppConfig,
)
from tokens import estimate_messages_tokens, estimate_tokens

if TYPE_CHECKING:
    from server.events import EventBus
    from workspace import Workspace


class LLMError(RuntimeError):
    """Raised when an LLM call fails (network, model missing, etc.)."""


@dataclass
class CompletionResult:
    text: str = ""        # the model's final answer (thinking stripped out)
    thinking: str = ""    # accumulated <think> content
    raw: str = ""         # everything, in order, as produced
    model: str = ""
    phase: str = ""
    total_tokens: int = 0
    duration: float = 0.0      # whole call (model load + prefill + generation)
    gen_duration: float = 0.0  # first token -> last token (generation only; for tok/s)

    def __bool__(self) -> bool:
        return bool(self.text.strip() or self.raw.strip())


class ThinkSplitter:
    """Incrementally split a token stream into (output, thinking) segments.

    Handles ``<think>``/``</think>`` tags that arrive split across chunks by
    holding back a small tail that could be the start of a tag.
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
                self.buf = self.buf[idx + len(self.OPEN) :]
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
                self.buf = self.buf[idx + len(self.CLOSE) :]
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
        # The model Ollama is currently holding warm (the last one we issued a call
        # to). Used to enforce "at most one model resident at a time" by unloading it
        # before a DIFFERENT model loads — the fix for the 27B->30B swap OOM/EOF.
        self._resident_model: str | None = None
        self._mem_lock = threading.Lock()

    def set_workspace(self, workspace: "Workspace") -> None:
        self.workspace = workspace

    # ── model memory management (spec adoption: no model stacking) ──────────────
    def _ensure_only(self, model: str) -> None:
        """Make *model* the only resident model before a call (when eviction is on).

        If a DIFFERENT model is currently warm, unload it first so the two never
        stack in VRAM/RAM (which OOM-kills the Ollama runner on a 16GB-VRAM box and
        surfaces as ``OllamaException {"error":"EOF"}``). A model re-called
        consecutively (e.g. the implement/fix loop) is left warm. No-op when
        ``evict_on_model_switch`` is false (big-memory box that can stack)."""
        if not self.config.evict_on_model_switch:
            return
        with self._mem_lock:
            prev = self._resident_model
            if prev and prev != model:
                self._ollama_unload(prev)
            self._resident_model = model

    def unload_all(self) -> None:
        """Unload the resident model (called at pipeline end so nothing lingers)."""
        if not self.config.evict_on_model_switch:
            return
        with self._mem_lock:
            if self._resident_model:
                self._ollama_unload(self._resident_model)
                self._resident_model = None

    def _ollama_unload(self, model: str) -> None:
        """Ask Ollama to evict *model* immediately (keep_alive=0). Best-effort.

        Skips the call if the model isn't actually loaded (per ``/api/ps``) so we
        never reload a model just to unload it."""
        name = (model or "").split("/", 1)[-1]  # strip the LiteLLM "ollama/" prefix
        if not name:
            return
        base = (self.config.ollama_base_url or "http://localhost:11434").rstrip("/")
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
        phase: str,
        messages: list[dict],
        *,
        stream: bool = True,
        temperature: float = 0.2,
        dump: bool = True,
    ) -> CompletionResult:
        model = self.config.model_for(phase)
        mode = self.config.thinking_mode_for(phase)
        prepared = self._apply_thinking(messages, mode)

        prompt_tokens = estimate_messages_tokens(prepared)
        self.bus.llm_request(phase, model, prompt_tokens)

        # Evict a different warm model first so two big models never stack (OOM/EOF).
        self._ensure_only(model)

        start = time.monotonic()
        try:
            result = self._stream(phase, model, prepared, temperature) if stream else self._oneshot(
                phase, model, prepared, temperature
            )
        except LLMError:
            raise
        except Exception as exc:  # normalize any provider error
            self.bus.error(f"LLM call failed: {exc}", context=f"phase={phase} model={model}", phase=phase)
            raise LLMError(f"{phase}: {exc}") from exc

        result.duration = time.monotonic() - start
        result.model = model
        result.phase = phase
        result.total_tokens = estimate_tokens(result.raw)
        # Raw generation throughput: tokens / generation time (from first token to
        # last), so a model's decode speed isn't diluted by load/prefill (important
        # now that models are evicted+reloaded on switch). Falls back to total wall
        # time for the non-streaming path. Surfaced on the event for the CLI + web UI.
        gen_dur = result.gen_duration or result.duration
        tps = round(result.total_tokens / gen_dur, 1) if gen_dur > 0 else 0.0
        self.bus.llm_complete(phase, result.total_tokens, duration=round(result.duration, 2), tokens_per_second=tps)

        if dump and self.config.dump_llm_calls:
            self._dump(phase, model, prepared, result)
        return result

    # ── thinking-mode prompt shaping ──────────────────────────────────────────
    def _apply_thinking(self, messages: list[dict], mode: str) -> list[dict]:
        if mode in (THINK_NONE, THINK_DEEPSEEK):
            return list(messages)  # deepseek thinks automatically; nothing to add
        suffix = {THINK_QWEN3_ON: " /think", THINK_QWEN3_OFF: " /no_think"}.get(mode)
        if not suffix:
            return list(messages)
        out = [dict(m) for m in messages]
        for msg in reversed(out):
            if msg.get("role") == "user":
                msg["content"] = f"{msg.get('content', '')}{suffix}"
                break
        return out

    # ── streaming ─────────────────────────────────────────────────────────────
    def _stream(self, phase: str, model: str, messages: list[dict], temperature: float) -> CompletionResult:
        import litellm

        splitter = ThinkSplitter()
        text_parts: list[str] = []
        think_parts: list[str] = []
        raw_parts: list[str] = []

        response = litellm.completion(
            model=model,
            messages=messages,
            stream=True,
            api_base=self.config.ollama_base_url,
            api_key="ollama",
            temperature=temperature,
            num_ctx=self.config.num_ctx,
            keep_alive=self.config.keep_alive,  # 0 = unload after this call (no model stacking in RAM)
            timeout=self.config.limits.sandbox_timeout * 30,
        )
        idle = max(30, int(self.config.limits.stream_idle_timeout))
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

    def _oneshot(self, phase: str, model: str, messages: list[dict], temperature: float) -> CompletionResult:
        import litellm

        response = litellm.completion(
            model=model,
            messages=messages,
            stream=False,
            api_base=self.config.ollama_base_url,
            api_key="ollama",
            temperature=temperature,
            num_ctx=self.config.num_ctx,
            keep_alive=self.config.keep_alive,  # 0 = unload after this call (no model stacking in RAM)
        )
        message = response.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
        text, inline_think = _strip_think(content)
        thinking = (reasoning + ("\n" if reasoning and inline_think else "") + inline_think)
        if thinking:
            self.bus.llm_thinking_token(phase, thinking)
        self.bus.llm_token(phase, text)
        return CompletionResult(text=text.strip(), thinking=thinking.strip(), raw=reasoning + content)

    # ── debugging dump ────────────────────────────────────────────────────────
    def _dump(self, phase: str, model: str, messages: list[dict], result: CompletionResult) -> None:
        if self.workspace is None:
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        body = [f"# LLM call — {phase}", f"model: `{model}`", f"duration: {result.duration:.2f}s", ""]
        body.append("## Prompt\n")
        for msg in messages:
            body.append(f"### {msg.get('role')}\n\n{msg.get('content','')}\n")
        if result.thinking:
            body.append("## Thinking\n\n```\n" + result.thinking + "\n```\n")
        body.append("## Output\n\n" + result.text + "\n")
        try:
            (self.workspace.llm_calls_dir / f"{phase}_{ts}.md").write_text("\n".join(body), encoding="utf-8")
        except OSError:
            pass


def _chunk_parts(chunk) -> tuple[str, str]:
    """Extract (content, reasoning_content) from a LiteLLM streaming chunk.

    LiteLLM exposes a model's thinking via ``reasoning_content`` (or ``reasoning``)
    on the delta, separate from the answer in ``content``. We read both so thinking
    is captured whether the provider separates it or inlines ``<think>`` tags.
    """
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
    *idle* seconds (spec adoption #7).

    This is an INTER-FRAME idle timeout — reset on every chunk — not an end-to-end
    cap. A slow-but-alive local model doing long prefill on a big context keeps the
    stream arriving, so only a stream gone genuinely silent trips it; a fixed total
    timeout, by contrast, can't tell "slow but working" from "hung". The blocking
    provider generator is drained on a daemon thread feeding a bounded queue, and the
    consumer waits at most *idle* per item. Mirrors codehamr's streamIdleTimeout.
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
                f"(raise limits.stream_idle_timeout if the model legitimately needs longer to prefill)"
            )
        if item is sentinel:
            if "err" in box:
                raise LLMError(str(box["err"]))
            return
        yield item
