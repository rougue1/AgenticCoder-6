"""Built-in stack profile registry.

A :class:`StackProfile` supplies the per-stack baseline the pipeline needs in
three places:

* the **required/optional binaries** pre-flight checks for (resolved through
  the sandbox's login shell, so version-manager installs count);
* the **environment strategy** (``uses_venv`` / ``uses_node_modules``) the
  orchestrator's environment setup follows;
* the default **.agentignore** patterns for the stack.

There is deliberately NO per-stack command list here any more: command
execution is jailed by the bwrap OS sandbox (``tools/sandbox.py``), which
enforces what a process can see and write instead of which executables it may
name — the tool stays stack-agnostic by construction.

``sandbox.stack_profile: auto`` in config resolves against the Manager's chosen
stack name after Step 1; ``python``/``node`` force a profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StackProfile:
    name: str
    required_binaries: list[str] = field(default_factory=list)  # pre-flight hard fail
    optional_binaries: list[str] = field(default_factory=list)  # pre-flight warning
    uses_venv: bool = False
    uses_node_modules: bool = False
    default_agentignore: list[str] = field(default_factory=list)
    install_check_cmd: str = ""  # post-install dependency-conflict check


PROFILES: dict[str, StackProfile] = {
    "python": StackProfile(
        name="python",
        required_binaries=["python3"],
        optional_binaries=["node", "npm"],
        uses_venv=True,
        default_agentignore=[".venv/", "__pycache__/", "*.pyc", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/", "*.egg-info/"],
        install_check_cmd="pip check",
    ),
    "node": StackProfile(
        name="node",
        required_binaries=["node", "npm"],
        optional_binaries=["python3"],
        uses_node_modules=True,
        default_agentignore=["node_modules/", "dist/", "build/", ".next/", "coverage/", "*.tsbuildinfo"],
        install_check_cmd="npm ls --depth=0",
    ),
}

_NODE_HINTS = ("node", "react", "next", "vue", "svelte", "express", "typescript", "javascript", "vite", "nest")
_PYTHON_HINTS = ("python", "fastapi", "flask", "django", "pytest", "uvicorn")


def resolve_profile(configured: str, stack_name: str = "") -> StackProfile:
    """Resolve the effective profile.

    ``configured`` is ``sandbox.stack_profile`` from config (``auto`` defers to
    the Manager's stack determination); *stack_name* is the Manager's chosen
    stack. Unknown/ambiguous names default to python — the redesign's default
    stack is modern Python FastAPI.
    """
    key = (configured or "auto").strip().lower()
    if key in PROFILES:
        return PROFILES[key]
    name = (stack_name or "").lower()
    if any(h in name for h in _NODE_HINTS) and not any(h in name for h in _PYTHON_HINTS):
        return PROFILES["node"]
    return PROFILES["python"]


def default_profile(configured: str) -> StackProfile:
    """Profile used for pre-flight, before the stack is determined."""
    return resolve_profile(configured, "")
