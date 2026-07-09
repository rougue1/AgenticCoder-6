"""Configuration loading for AIForge (two-tier redesign).

Loads ``config.yaml`` into a single :class:`AppConfig`. The schema is a clean
break from the old per-phase model map: exactly two model tiers (``manager`` and
``worker``), a ``pipeline`` ladder-limits block, a ``sandbox`` block, a
``context`` block, a ``server`` block, and an ``ollama`` hardware/runtime block.

Raw model fields (``model``, ``temperature``, ``context_window_pct``,
``max_tokens_override``, ``use_thinking``) are the only things a user edits.
They are NOT read directly by pipeline code: at startup
:func:`llm.resolution.resolve_all` turns each tier into a
:class:`llm.resolution.RuntimeModelConfig` (true num_ctx from Ollama, computed
max_tokens, probed thinking capability) and everything downstream reads that.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MANAGER = "manager"
WORKER = "worker"
TIERS = (MANAGER, WORKER)

# Documented fallbacks used ONLY when a tier is missing from config.yaml, so a
# partial config never crashes the pipeline. config.yaml overrides these.
# Discovered on this machine via `ollama list` (16GB VRAM + 32GB RAM):
#   ornith:35b    — qwen3.5-MoE 34.7B Q4_K_M (~21GB) — Manager/Architect
#   ornith:latest — qwen3.5 9B Q4_K_M (~5.6GB)       — Worker/Coder
DEFAULT_TIERS: dict[str, dict[str, Any]] = {
    MANAGER: {
        "model": "ornith:35b",
        "temperature": 0.3,
        "context_window_pct": 0.75,
        "max_tokens_override": None,
        "use_thinking": True,
    },
    WORKER: {
        "model": "ornith:latest",
        "temperature": 0.1,
        "context_window_pct": 0.55,
        "max_tokens_override": None,
        "use_thinking": False,
    },
}


@dataclass
class ModelTierCfg:
    """User-editable knobs for one tier. Never read directly after startup —
    see :class:`llm.resolution.RuntimeModelConfig` for the resolved values."""

    name: str
    model: str
    temperature: float
    context_window_pct: float
    max_tokens_override: int | None
    use_thinking: bool


@dataclass
class PipelineCfg:
    max_fix_retries: int = 3
    max_escalations: int = 2
    max_decompositions: int = 1
    tdd_hard_fail: bool = True


@dataclass
class SandboxCfg:
    timeout: int = 60             # seconds per foreground command
    background_grace: float = 2.0  # seconds a background start is watched for instant crashes
    session_tail_lines: int = 200  # log lines check_session returns (tail)
    stack_profile: str = "auto"    # auto | python | node — env strategy + preflight binaries


@dataclass
class ContextCfg:
    max_summary_tokens: int = 300
    max_handoff_tokens: int = 6000
    always_include: list[str] = field(default_factory=lambda: ["architecture.md", "requirements.md"])
    decision_log_max_entries: int = 20


@dataclass
class ServerCfg:
    host: str = "0.0.0.0"
    port: int = 8765


@dataclass
class OllamaCfg:
    """Hardware/runtime tuning. Defaults suit a 16GB-VRAM / 32GB-RAM box."""

    host: str = "http://localhost:11434"
    # Ceiling on the per-request num_ctx (the KV cache Ollama allocates at load
    # time — the dominant memory cost). The resolver clamps a model's reported
    # context to this.
    max_num_ctx: int = 32768
    # Warm window between consecutive SAME-model calls. Switching to a different
    # model evicts explicitly (see evict_on_model_switch), so the effective
    # keep-alive behavior is dynamic: warm inside a loop, freed on swap.
    keep_alive: int | str = "10m"
    evict_on_model_switch: bool = True
    stream_idle_timeout: int = 600
    request_timeout: int = 3600
    probe_timeout: int = 120


@dataclass
class AppConfig:
    """Resolved, validated configuration for one pipeline run."""

    raw: dict[str, Any]
    tool_root: Path

    tiers: dict[str, ModelTierCfg] = field(default_factory=dict)
    pipeline: PipelineCfg = field(default_factory=PipelineCfg)
    sandbox: SandboxCfg = field(default_factory=SandboxCfg)
    context: ContextCfg = field(default_factory=ContextCfg)
    server: ServerCfg = field(default_factory=ServerCfg)
    ollama: OllamaCfg = field(default_factory=OllamaCfg)

    project_dir: str = ""
    dump_llm_calls: bool = True

    def tier(self, name: str) -> ModelTierCfg:
        return self.tiers[name]

    @property
    def manager(self) -> ModelTierCfg:
        return self.tiers[MANAGER]

    @property
    def worker(self) -> ModelTierCfg:
        return self.tiers[WORKER]


# Recognized top-level keys. Anything else is almost always a typo that would
# otherwise silently leave a default in force.
_KNOWN_CONFIG_KEYS = {
    "models",
    "pipeline",
    "sandbox",
    "context",
    "server",
    "ollama",
    "project_dir",
    "dump_llm_calls",
}


def load_config(
    config_path: str | Path | None = None,
    *,
    project_dir_override: str | None = None,
    tool_root: str | Path | None = None,
    dump_llm_calls: bool | None = None,
) -> AppConfig:
    """Load ``config.yaml`` into an :class:`AppConfig`.

    ``tool_root`` is the directory the tool lives in (where ``../sandbox/`` is
    created). Defaults to the directory containing this file.
    """
    root = Path(tool_root).resolve() if tool_root else Path(__file__).resolve().parent
    cfg_file = Path(config_path) if config_path else root / "config.yaml"

    raw = _read_yaml(cfg_file)
    _warn_unknown_config_keys(raw, cfg_file)

    tiers = _load_tiers(raw.get("models") or {})

    pl = raw.get("pipeline") or {}
    pipeline = PipelineCfg(
        max_fix_retries=_int(pl.get("max_fix_retries"), 3),
        max_escalations=_int(pl.get("max_escalations"), 2),
        max_decompositions=_int(pl.get("max_decompositions"), 1),
        tdd_hard_fail=_bool(pl.get("tdd_hard_fail"), True),
    )

    sb = raw.get("sandbox") or {}
    sandbox = SandboxCfg(
        timeout=_int(sb.get("timeout"), 60),
        background_grace=_float(sb.get("background_grace"), 2.0),
        session_tail_lines=_int(sb.get("session_tail_lines"), 200),
        stack_profile=str(sb.get("stack_profile") or "auto").strip().lower(),
    )

    cx = raw.get("context") or {}
    always = cx.get("always_include")
    context = ContextCfg(
        max_summary_tokens=_int(cx.get("max_summary_tokens"), 300),
        max_handoff_tokens=_int(cx.get("max_handoff_tokens"), 6000),
        always_include=[str(x) for x in always] if isinstance(always, list) else ["architecture.md", "requirements.md"],
        decision_log_max_entries=_int(cx.get("decision_log_max_entries"), 20),
    )

    sv = raw.get("server") or {}
    server = ServerCfg(host=str(sv.get("host", "0.0.0.0")), port=_int(sv.get("port"), 8765))

    ol = raw.get("ollama") or {}
    keep_alive = ol.get("keep_alive", "10m")
    ollama = OllamaCfg(
        host=str(ol.get("host") or "http://localhost:11434").rstrip("/"),
        max_num_ctx=_int(ol.get("max_num_ctx"), 32768),
        keep_alive=keep_alive if keep_alive not in (None, "") else "10m",
        evict_on_model_switch=_bool(ol.get("evict_on_model_switch"), True),
        stream_idle_timeout=_int(ol.get("stream_idle_timeout"), 600),
        request_timeout=_int(ol.get("request_timeout"), 3600),
        probe_timeout=_int(ol.get("probe_timeout"), 120),
    )

    project_dir = project_dir_override if project_dir_override is not None else str(raw.get("project_dir", "") or "")

    return AppConfig(
        raw=raw,
        tool_root=root,
        tiers=tiers,
        pipeline=pipeline,
        sandbox=sandbox,
        context=context,
        server=server,
        ollama=ollama,
        project_dir=project_dir,
        dump_llm_calls=bool(raw.get("dump_llm_calls", True)) if dump_llm_calls is None else dump_llm_calls,
    )


def _load_tiers(models_raw: dict[str, Any]) -> dict[str, ModelTierCfg]:
    tiers: dict[str, ModelTierCfg] = {}
    for name in TIERS:
        merged = {**DEFAULT_TIERS[name], **(models_raw.get(name) or {})}
        override = merged.get("max_tokens_override")
        tiers[name] = ModelTierCfg(
            name=name,
            model=_normalize_model_name(str(merged.get("model") or DEFAULT_TIERS[name]["model"])),
            temperature=_float(merged.get("temperature"), float(DEFAULT_TIERS[name]["temperature"])),
            context_window_pct=_clamp_pct(
                _float(merged.get("context_window_pct"), float(DEFAULT_TIERS[name]["context_window_pct"]))
            ),
            max_tokens_override=None if override in (None, "", "null") else _int(override, 0) or None,
            use_thinking=_bool(merged.get("use_thinking"), bool(DEFAULT_TIERS[name]["use_thinking"])),
        )
    return tiers


def _normalize_model_name(name: str) -> str:
    """Model names are bare Ollama names (``ornith:35b``). Tolerate a pasted
    LiteLLM-style prefix by stripping it — the client adds its own prefix."""
    name = name.strip()
    for prefix in ("ollama_chat/", "ollama/"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _warn_unknown_config_keys(raw: dict[str, Any], path: Path) -> None:
    """Stderr warning for unrecognized top-level keys (config loads before the
    event bus exists). A warning, not a hard error — a local single-user tool
    should surface a likely typo without refusing to start."""
    unknown = sorted(str(k) for k in raw if k not in _KNOWN_CONFIG_KEYS)
    if unknown:
        print(
            f"warning: {path.name}: ignoring unrecognized config key(s): {', '.join(unknown)} "
            f"(known keys: {', '.join(sorted(_KNOWN_CONFIG_KEYS))})",
            file=sys.stderr,
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _clamp_pct(value: float) -> float:
    return min(0.95, max(0.05, value))
