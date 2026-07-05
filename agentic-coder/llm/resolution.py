"""Model capability resolution — runs once at startup, before pre-flight.

For each tier (manager, worker) this module turns the user's raw config into a
:class:`RuntimeModelConfig` that ALL downstream code reads. Nothing else in the
codebase reads raw model config fields after this step.

Per tier, in order:

1. **/api/show** — fetch the model's metadata from Ollama. A missing model or a
   failed call is a hard startup failure with a clear message. The usable
   context window is the ``num_ctx`` the modelfile pins if any, else the
   architecture's ``context_length``; it is then clamped to the hardware
   ceiling ``ollama.max_num_ctx`` (ornith models report 262144 — allocating
   that KV cache would OOM a 16GB-VRAM box instantly).
2. **max_tokens** — ``max_tokens_override`` verbatim when set, else
   ``int(num_ctx * context_window_pct)``.
3. **Thinking, two gates** — gate one is config intent (``use_thinking: false``
   disables immediately, no probe). Gate two is a live probe: one minimal
   ``/api/chat`` call with ``think: true`` and a 120s timeout (long enough to
   cover a cold model load into VRAM). An explicit "does not support thinking"
   error records ``thinking_enabled=False`` with a single warning; ANY other
   failure is a hard startup failure, never a silent fallback.
4. **Worker override** — the worker NEVER thinks, regardless of config or
   probe (a tight tool loop gains nothing from deliberation). Asking for it
   only earns a warning.

Resolution is re-run from scratch on ``--resume`` — model availability and
configuration can change between sessions, so nothing here is persisted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from config import MANAGER, WORKER, AppConfig, ModelTierCfg

if TYPE_CHECKING:
    from server.events import EventBus

_NUM_CTX_RE = re.compile(r"^\s*num_ctx\s+(\d+)\s*$", re.MULTILINE)
_UNSUPPORTED_THINK_RE = re.compile(r"does not support thinking|thinking is not supported|unknown.*think", re.I)

# Conservative floor used only if Ollama reports no context metadata at all.
_FALLBACK_CTX = 8192


class ModelResolutionError(RuntimeError):
    """Hard startup failure while resolving a model's runtime configuration."""


@dataclass(frozen=True)
class RuntimeModelConfig:
    """The resolved, immutable runtime view of one model tier."""

    tier: str
    model: str
    reported_num_ctx: int      # what Ollama says the model supports/pins
    num_ctx: int               # effective per-request window (clamped to hardware cap)
    max_tokens: int            # input-token budget for this tier
    context_window_pct: float
    temperature: float
    use_thinking: bool         # the raw config intent
    thinking_enabled: bool     # the final resolved decision
    supports_think_param: bool  # /api/show capabilities include "thinking"

    def describe(self) -> str:
        return (
            f"{self.tier}: model={self.model} | reported num_ctx={self.reported_num_ctx} | "
            f"effective num_ctx={self.num_ctx} | max_tokens={self.max_tokens} "
            f"(pct={self.context_window_pct}) | use_thinking={self.use_thinking} -> "
            f"thinking_enabled={self.thinking_enabled}"
        )


def resolve_all(config: AppConfig, bus: "EventBus") -> dict[str, RuntimeModelConfig]:
    """Resolve both tiers (manager first — it is also the first model Phase 1
    uses, so its probe leaves it warm) and log the summary block to run.log."""
    resolved = {
        MANAGER: resolve_tier(config, config.tier(MANAGER), bus),
        WORKER: resolve_tier(config, config.tier(WORKER), bus, force_no_thinking=True),
    }
    summary = "Model resolution complete:\n" + "\n".join(
        "  " + rmc.describe() for rmc in resolved.values()
    )
    bus.log(summary, phase="resolution")
    return resolved


def resolve_tier(
    config: AppConfig,
    tier_cfg: ModelTierCfg,
    bus: "EventBus",
    *,
    force_no_thinking: bool = False,
) -> RuntimeModelConfig:
    base = config.ollama.host
    model = tier_cfg.model

    # 1. /api/show — hard fail if the model isn't known to this Ollama instance.
    show = _api_show(base, model)
    reported = _extract_num_ctx(show)
    if reported is None:
        bus.log(
            f"{model}: Ollama reported no context-window metadata; assuming {_FALLBACK_CTX}",
            phase="resolution",
            level="warn",
        )
        reported = _FALLBACK_CTX
    num_ctx = max(1024, min(reported, config.ollama.max_num_ctx))
    supports_think = "thinking" in {str(c).lower() for c in (show.get("capabilities") or [])}

    # 2. Effective max_tokens for the tier.
    if tier_cfg.max_tokens_override is not None:
        max_tokens = int(tier_cfg.max_tokens_override)
    else:
        max_tokens = int(num_ctx * tier_cfg.context_window_pct)

    # 3+4. Thinking resolution (two gates + the worker architectural override).
    thinking_enabled = False
    if force_no_thinking:
        if tier_cfg.use_thinking:
            bus.log(
                f"{tier_cfg.name}: use_thinking=true is IGNORED for the worker role — the worker "
                "runs a tight tool-calling loop where deliberation burns tokens with no quality "
                "benefit; proceeding with thinking disabled",
                phase="resolution",
                level="warn",
            )
    elif tier_cfg.use_thinking:
        thinking_enabled = _probe_thinking(config, model, bus)

    return RuntimeModelConfig(
        tier=tier_cfg.name,
        model=model,
        reported_num_ctx=reported,
        num_ctx=num_ctx,
        max_tokens=max_tokens,
        context_window_pct=tier_cfg.context_window_pct,
        temperature=tier_cfg.temperature,
        use_thinking=tier_cfg.use_thinking,
        thinking_enabled=thinking_enabled,
        supports_think_param=supports_think,
    )


# ── Ollama plumbing ────────────────────────────────────────────────────────────
def _api_show(base: str, model: str) -> dict:
    try:
        resp = httpx.post(f"{base}/api/show", json={"model": model}, timeout=30)
    except httpx.HTTPError as exc:
        raise ModelResolutionError(
            f"could not reach Ollama at {base} to resolve model {model!r}: {exc}. "
            "Is `ollama serve` running?"
        ) from exc
    if resp.status_code == 404:
        raise ModelResolutionError(
            f"model {model!r} was not found in Ollama. Pull it (`ollama pull {model}`) "
            "or fix the model name in config.yaml."
        )
    if resp.status_code >= 400:
        raise ModelResolutionError(
            f"Ollama /api/show failed for {model!r}: HTTP {resp.status_code} {resp.text[:300]}"
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise ModelResolutionError(f"Ollama /api/show returned non-JSON for {model!r}") from exc


def _extract_num_ctx(show: dict) -> int | None:
    """The usable context window as configured in this Ollama instance.

    A ``num_ctx`` pinned in the modelfile parameters wins (it IS the instance
    configuration); otherwise the architecture's ``<family>.context_length``
    from model_info.
    """
    params = show.get("parameters")
    if isinstance(params, str):
        m = _NUM_CTX_RE.search(params)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    info = show.get("model_info")
    if isinstance(info, dict):
        best: int | None = None
        for key, value in info.items():
            if str(key).endswith(".context_length"):
                try:
                    v = int(value)
                except (TypeError, ValueError):
                    continue
                best = v if best is None else max(best, v)
        if best:
            return best
    return None


def _probe_thinking(config: AppConfig, model: str, bus: "EventBus") -> bool:
    """Gate two: a live capability probe with ``think: true``.

    Success -> True. An explicit unsupported-think error -> False plus one
    warning line. Anything else (network failure, timeout, malformed response)
    -> :class:`ModelResolutionError`, never a silent fallback. The generous
    timeout exists because the first call loads the model into VRAM (30-60s+
    for a large model) — do not shorten it.
    """
    base = config.ollama.host
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say the word hello."}],
        "think": True,
        "stream": False,
        "options": {"num_predict": 32},
        # Keep the probed model warm — the manager is the next model Phase 1 uses.
        "keep_alive": config.ollama.keep_alive,
    }
    try:
        resp = httpx.post(f"{base}/api/chat", json=body, timeout=config.ollama.probe_timeout)
    except httpx.TimeoutException as exc:
        raise ModelResolutionError(
            f"thinking probe for {model!r} timed out after {config.ollama.probe_timeout}s — "
            "the model may be too large to load on this machine, or Ollama is stuck"
        ) from exc
    except httpx.HTTPError as exc:
        raise ModelResolutionError(f"thinking probe for {model!r} failed to reach Ollama: {exc}") from exc

    if resp.status_code >= 400:
        detail = _error_text(resp)
        if _UNSUPPORTED_THINK_RE.search(detail):
            bus.log(
                f"{model}: thinking was requested but is unavailable for this model — "
                "falling back to standard completion",
                phase="resolution",
                level="warn",
            )
            return False
        raise ModelResolutionError(
            f"thinking probe for {model!r} failed: HTTP {resp.status_code} {detail[:300]}"
        )

    try:
        payload = resp.json()
    except ValueError as exc:
        raise ModelResolutionError(f"thinking probe for {model!r} returned non-JSON") from exc
    if not isinstance(payload, dict) or "message" not in payload:
        raise ModelResolutionError(
            f"thinking probe for {model!r} returned an unexpected shape: {str(payload)[:200]}"
        )
    return True


def _error_text(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
    except ValueError:
        pass
    return resp.text or ""
