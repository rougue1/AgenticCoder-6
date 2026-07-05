"""Built-in stack profile registry.

A :class:`StackProfile` supplies the per-stack baseline the redesign needs in
three places:

* the sandbox **command allowlist** — the Manager's stack-determination output
  extends this base set (never replaces it; the ``never_allow`` list in
  ``tools/sandbox.py`` is enforced above both);
* the **required/optional binaries** pre-flight checks for;
* the default **.agentignore** patterns for the stack.

``sandbox.stack_profile: auto`` in config resolves against the Manager's chosen
stack name after Step 1; ``python``/``node`` force a profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Shell builtins + coreutils any stack may use. Deliberately conservative:
# read/navigate/create, no package publishing, no VCS (git is denylisted anyway).
CORE_UTILS = [
    "cd", "ls", "cat", "echo", "printf", "pwd", "mkdir", "touch", "rm", "mv", "cp",
    "find", "grep", "egrep", "fgrep", "sed", "awk", "head", "tail", "wc", "sort",
    "uniq", "cut", "tr", "diff", "which", "env", "export", "test", "true", "false",
    "sleep", "date", "basename", "dirname", "xargs", "tee", "chmod", "sh", "bash",
    "curl",  # local API smoke tests against a background server (pipe-to-shell is denylisted)
]

PYTHON_COMMANDS = [
    "python", "python3", "pip", "pip3", "pytest", "uvicorn", "gunicorn",
    "black", "ruff", "mypy", "flake8", "isort", "coverage", "alembic", "flask",
]

NODE_COMMANDS = [
    "node", "npm", "npx", "yarn", "pnpm", "tsc", "vite", "jest", "vitest",
    "eslint", "prettier", "next",
]


@dataclass
class StackProfile:
    name: str
    base_allowed_commands: list[str] = field(default_factory=list)
    required_binaries: list[str] = field(default_factory=list)  # pre-flight hard fail
    optional_binaries: list[str] = field(default_factory=list)  # pre-flight warning
    uses_venv: bool = False
    uses_node_modules: bool = False
    default_agentignore: list[str] = field(default_factory=list)
    install_check_cmd: str = ""  # post-install dependency-conflict check


PROFILES: dict[str, StackProfile] = {
    "python": StackProfile(
        name="python",
        base_allowed_commands=CORE_UTILS + PYTHON_COMMANDS,
        required_binaries=["python3"],
        optional_binaries=["node", "npm"],
        uses_venv=True,
        default_agentignore=[".venv/", "__pycache__/", "*.pyc", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/", "*.egg-info/"],
        install_check_cmd="pip check",
    ),
    "node": StackProfile(
        name="node",
        base_allowed_commands=CORE_UTILS + NODE_COMMANDS + PYTHON_COMMANDS,
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
