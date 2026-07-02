"""Configuration loading and per-phase resolution for AIForge.

Loads ``config.yaml`` (models, limits, server) and ``context_budget.yaml``
(token budgets + compression policy), and exposes a single :class:`AppConfig`
object with convenience lookups used everywhere in the pipeline:

* :meth:`AppConfig.model_for` -> the LiteLLM model string for a phase.
* :meth:`AppConfig.thinking_mode_for` -> how the phase drives a reasoning model.
* :meth:`AppConfig.budget_for` -> the input-token budget for a phase.

Any phase omitted from ``config.yaml`` falls back to a documented default
(see ``DEFAULT_MODELS``). Model strings are LiteLLM-format (e.g.
``ollama/qwen2.5-coder:14b``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Thinking-mode tokens. A phase resolves to exactly one of these.
THINK_DEEPSEEK = "deepseek"        # model emits <think>...</think> automatically
THINK_QWEN3_ON = "qwen3_think"     # append "/think" to the user message
THINK_QWEN3_OFF = "qwen3_no_think"  # append "/no_think" to the user message
THINK_NONE = "none"                # no thinking mode available


def detect_thinking_mode(model: str) -> str:
    """Infer a phase's thinking mode from its model name — no manual config.

    The rule the user asked for: *default to thinking wherever the model family
    supports it, and fall back to no-thinking where it doesn't.* So swapping a
    model in ``config.yaml`` automatically carries the right thinking behavior and
    you never hand-maintain a per-phase list. An explicit ``model_options`` entry
    still overrides this (see :meth:`AppConfig.thinking_mode_for`).

    * DeepSeek-R1 distills think on their own → :data:`THINK_DEEPSEEK` (no suffix).
    * Qwen3 family (qwen3, qwen3.5, qwen3.6, qwen3-coder, qwq…) uses the ``/think``
      soft switch → :data:`THINK_QWEN3_ON`.
    * Everything else (qwen2.5, llama3.x, mistral…) has no thinking toggle →
      :data:`THINK_NONE`.
    """
    name = (model or "").split("/", 1)[-1].strip().lower()  # drop any "ollama/" prefix
    if not name:
        return THINK_NONE
    if "deepseek-r1" in name or ("deepseek" in name and "r1" in name):
        return THINK_DEEPSEEK
    if name.startswith("qwen3") or name.startswith("qwq"):
        return THINK_QWEN3_ON
    return THINK_NONE


# Phase groups for the coarse `thinking:` config toggle (coding vs planning). An
# explicit ``model_options`` entry still wins over the toggle, and the toggle wins
# over auto-detection. ``tester`` is kept here for robustness even though the
# implementer now owns test-writing (no separate tester phase runs).
CODING_PHASES = frozenset({"implementer", "tester", "reviewer"})
PLANNING_PHASES = frozenset(
    {"requirements", "stack_decider", "architect", "sdd_generator", "task_planner", "planner", "escalation"}
)


def _toggle_thinking(model: str, want: bool) -> str:
    """Resolve a thinking mode for *model* under an explicit on/off *want* toggle.

    Only flips what the model family actually supports: for the qwen3 family the
    soft switch becomes ``/think`` or ``/no_think``; DeepSeek-R1 can be left on but
    not suffix-disabled; a no-thinking model ignores the toggle entirely.
    """
    auto = detect_thinking_mode(model)
    if auto == THINK_QWEN3_ON:
        return THINK_QWEN3_ON if want else THINK_QWEN3_OFF
    if auto == THINK_DEEPSEEK:
        return THINK_DEEPSEEK if want else THINK_NONE
    return THINK_NONE

# Documented fallbacks, used ONLY when a phase key is missing from config.yaml so
# the pipeline never crashes on a partial config. ⚠️ config.yaml is the single
# source of truth for models and OVERRIDES everything here — these are a safety net,
# not the live selection. Tuned for a 16GB-VRAM / 32GB-RAM machine: the strongest
# models that fit (no 70b). There is no separate ``tester`` phase — the implementer
# writes tests inside its own ephemeral conversation (it has the implementation
# context). ``reviewer`` reuses the coder so it stays warm.
DEFAULT_MODELS: dict[str, str] = {
    "architect": "ollama/qwen3.6:27b",
    "sdd_generator": "ollama/qwen3.6:27b",
    "requirements": "ollama/qwen3.6:27b",
    "stack_decider": "ollama/qwen3.6:27b",
    "task_planner": "ollama/qwen3.6:27b",
    "planner": "ollama/qwen3.6:27b",
    "escalation": "ollama/qwen3.6:27b",
    "implementer": "ollama/qwen3-coder:30b",
    "tool_caller": "ollama/qwen2.5-coder:14b",
    "intake": "ollama/llama3.2:3b",
    "reviewer": "ollama/qwen3-coder:30b",
}

DEFAULT_BUDGET = 32768
DEFAULT_RESERVE = 8192

# Phases not listed in any model_options group default to THINK_NONE.
DEFAULT_LIMITS = {
    "sandbox_timeout": 120,
    "max_fix_retries": 3,
    "max_escalations": 2,
    "long_process_timeout": 30,
    "stream_idle_timeout": 600,
}


@dataclass
class Limits:
    sandbox_timeout: int = 120
    max_fix_retries: int = 3
    max_escalations: int = 2
    long_process_timeout: int = 30
    # Inter-frame idle timeout for an LLM stream (spec adoption #7): the wait for the
    # NEXT streamed chunk, reset on every chunk — not an end-to-end cap. A live-but-slow
    # local model doing long prefill on big context keeps the stream alive; only a
    # stream gone silent after the request was accepted trips this. Generous because a
    # 32B on modest hardware can stay silent for minutes during prefill.
    stream_idle_timeout: int = 600


@dataclass
class ServerCfg:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class AppConfig:
    """Resolved, validated configuration for one pipeline run."""

    raw: dict[str, Any]
    budgets_raw: dict[str, Any]
    tool_root: Path

    models: dict[str, str] = field(default_factory=dict)
    thinking: dict[str, str] = field(default_factory=dict)
    budgets: dict[str, int] = field(default_factory=dict)
    reserve_for_output: int = DEFAULT_RESERVE
    limits: Limits = field(default_factory=Limits)
    server: ServerCfg = field(default_factory=ServerCfg)
    ollama_base_url: str = "http://localhost:11434"
    project_dir: str = ""
    # num_ctx sizes the KV cache Ollama allocates at load time — the dominant memory
    # cost. 32768 suits a 16GB-VRAM / 32GB-RAM box; lower it (and the
    # context_budget.yaml budgets) on a smaller machine, raise it with RAM to spare.
    num_ctx: int = 32768
    # keep_alive is how long Ollama keeps a model resident after a call (passed per
    # request). A duration like "5m" keeps the active model warm so a phase's tool
    # loop doesn't reload it every call; 0 unloads immediately (safest on a tight
    # machine). int seconds or an Ollama duration string.
    keep_alive: int | str = "5m"
    dump_llm_calls: bool = True
    # When true (right for <=16GB VRAM) the LLM client never keeps two DIFFERENT
    # models resident: the prior phase's model is unloaded before a different one
    # loads, so big model swaps (e.g. 27B planner -> 30B coder) can't stack and OOM.
    # A model re-called consecutively (the implement/fix loop) stays warm. Set false
    # on a big box to allow stacking. See llm/client.py.
    evict_on_model_switch: bool = True
    # Coarse thinking control (config.yaml `thinking:` block). None = auto-detect from
    # the model name; True/False force thinking on/off for that phase group. A
    # per-phase ``model_options`` entry still wins over these.
    think_coding: bool | None = None     # implementer / reviewer (the coder phases)
    think_planning: bool | None = None   # requirements..task_planner, planner, escalation

    # ── lookups ────────────────────────────────────────────────────────────
    def model_for(self, phase: str) -> str:
        """Return the LiteLLM model string for *phase* (with fallback)."""
        return self.models.get(phase) or DEFAULT_MODELS.get(phase) or DEFAULT_MODELS["implementer"]

    def thinking_mode_for(self, phase: str) -> str:
        """Return the thinking-mode token for *phase* (see THINK_* constants).

        Precedence: an explicit ``model_options`` entry (``self.thinking``) wins;
        then the coarse coding/planning ``thinking:`` toggle; otherwise the mode is
        auto-detected from the phase's model (default-on where supported, off where
        not), so thinking follows the model automatically."""
        explicit = self.thinking.get(phase)
        if explicit is not None:
            return explicit
        model = self.model_for(phase)
        if phase in CODING_PHASES and self.think_coding is not None:
            return _toggle_thinking(model, self.think_coding)
        if phase in PLANNING_PHASES and self.think_planning is not None:
            return _toggle_thinking(model, self.think_planning)
        return detect_thinking_mode(model)

    def budget_for(self, phase: str) -> int:
        """Return the input-token budget for *phase*."""
        return int(self.budgets.get(phase, DEFAULT_BUDGET))

    def usable_budget_for(self, phase: str) -> int:
        """Budget minus the output reserve — the real ceiling for input tokens."""
        return max(1024, self.budget_for(phase) - self.reserve_for_output)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _opt_bool(value: Any) -> bool | None:
    """Parse an optional yaml bool: absent/None -> None (auto), else a real bool."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _build_thinking_map(model_options: dict[str, Any]) -> dict[str, str]:
    """Invert the model_options group-lists into a {phase: mode} map."""
    mapping: dict[str, str] = {}
    groups = (
        ("deepseek_thinking_always_on", THINK_DEEPSEEK),
        ("qwen3_thinking", THINK_QWEN3_ON),
        ("qwen3_no_thinking", THINK_QWEN3_OFF),
        ("no_thinking", THINK_NONE),
    )
    for key, mode in groups:
        for phase in model_options.get(key) or []:
            mapping[str(phase).strip()] = mode
    return mapping


# Recognized top-level keys in config.yaml. A key outside this set is almost always
# a typo (e.g. `modeltls:` or `limts:`) that would otherwise be silently ignored and
# leave the documented defaults quietly in force — a confusing debugging session.
_KNOWN_CONFIG_KEYS = {
    "models",
    "model_options",
    "thinking",
    "limits",
    "server",
    "project_dir",
    "ollama_base_url",
    "num_ctx",
    "keep_alive",
    "evict_on_model_switch",
    "dump_llm_calls",
}


def _warn_unknown_config_keys(raw: dict[str, Any], path: Path) -> None:
    """Print a stderr warning for unrecognized top-level config.yaml keys.

    A warning (not a hard error): a local single-user tool should surface a likely
    typo without refusing to start over a harmless extra key. Stderr because config
    loads before the event bus exists (mirrors main.py's startup warnings)."""
    unknown = sorted(str(k) for k in raw if k not in _KNOWN_CONFIG_KEYS)
    if unknown:
        print(
            f"warning: {path.name}: ignoring unrecognized config key(s): {', '.join(unknown)} "
            f"(known keys: {', '.join(sorted(_KNOWN_CONFIG_KEYS))})",
            file=sys.stderr,
        )


def load_config(
    config_path: str | Path | None = None,
    budget_path: str | Path | None = None,
    *,
    project_dir_override: str | None = None,
    tool_root: str | Path | None = None,
    dump_llm_calls: bool | None = None,
) -> AppConfig:
    """Load ``config.yaml`` + ``context_budget.yaml`` into an :class:`AppConfig`.

    ``tool_root`` is the directory the tool lives in (where ``sandbox/`` is
    created). Defaults to the directory containing this file.
    """
    root = Path(tool_root).resolve() if tool_root else Path(__file__).resolve().parent
    cfg_file = Path(config_path) if config_path else root / "config.yaml"
    bud_file = Path(budget_path) if budget_path else root / "context_budget.yaml"

    raw = _read_yaml(cfg_file)
    budgets_raw = _read_yaml(bud_file)
    _warn_unknown_config_keys(raw, cfg_file)

    models = dict(raw.get("models") or {})
    thinking = _build_thinking_map(raw.get("model_options") or {})
    think_cfg = raw.get("thinking") or {}
    think_coding = _opt_bool(think_cfg.get("coding"))
    think_planning = _opt_bool(think_cfg.get("planning"))

    budgets = {k: _coerce_int(v, DEFAULT_BUDGET) for k, v in (budgets_raw.get("budgets") or {}).items()}
    reserve = _coerce_int(budgets_raw.get("reserve_for_output"), DEFAULT_RESERVE)

    limits_raw = {**DEFAULT_LIMITS, **(raw.get("limits") or {})}
    limits = Limits(
        sandbox_timeout=_coerce_int(limits_raw.get("sandbox_timeout"), 120),
        max_fix_retries=_coerce_int(limits_raw.get("max_fix_retries"), 3),
        max_escalations=_coerce_int(limits_raw.get("max_escalations"), 2),
        long_process_timeout=_coerce_int(limits_raw.get("long_process_timeout"), 30),
        stream_idle_timeout=_coerce_int(limits_raw.get("stream_idle_timeout"), 600),
    )

    server_raw = raw.get("server") or {}
    server = ServerCfg(
        host=str(server_raw.get("host", "127.0.0.1")),
        port=_coerce_int(server_raw.get("port"), 8765),
    )

    project_dir = project_dir_override if project_dir_override is not None else str(raw.get("project_dir", "") or "")

    return AppConfig(
        raw=raw,
        budgets_raw=budgets_raw,
        tool_root=root,
        models=models,
        thinking=thinking,
        budgets=budgets,
        reserve_for_output=reserve,
        limits=limits,
        server=server,
        ollama_base_url=str(raw.get("ollama_base_url", "http://localhost:11434")),
        project_dir=project_dir,
        num_ctx=_coerce_int(raw.get("num_ctx"), 32768),
        keep_alive=raw.get("keep_alive", "5m") if raw.get("keep_alive", "5m") not in (None, "") else "5m",
        evict_on_model_switch=_opt_bool(raw.get("evict_on_model_switch")) if raw.get("evict_on_model_switch") is not None else True,
        think_coding=think_coding,
        think_planning=think_planning,
        dump_llm_calls=bool(raw.get("dump_llm_calls", True)) if dump_llm_calls is None else dump_llm_calls,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}
