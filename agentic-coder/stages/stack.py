"""Phase 1, Step 1 — Stack Determination (Manager).

The Manager evaluates the user prompt and determines: the technology stack
name, a preferred Python version (when Python-based), the explicit shell
command allowlist the Worker may run (extending the built-in stack profile),
the ``.agentignore`` content for this stack, and a short project brief. When
the prompt implies no stack, the default is modern Python + FastAPI.

The orchestrator then (Step 2) concatenates the original prompt with this
output into the immutable Stack-Specific Anchor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import promptlib
from llm.tool_parser import extract_json
from services import Services
from stages import manager


@dataclass
class StackInfo:
    stack_name: str = "python-fastapi"
    python_version: str = ""
    allowed_commands: list[str] = field(default_factory=list)
    agentignore: list[str] = field(default_factory=list)
    brief: str = ""
    raw_output: str = ""  # the Manager's full determination (anchor material)


def run(services: Services, prompt: str) -> StackInfo:
    instruction = promptlib.render("stack", prompt=prompt)
    text = manager.call(services, "stack", instruction, system=manager.BOOTSTRAP_SYSTEM)

    data = extract_json(text)
    info = StackInfo(raw_output=text)
    if isinstance(data, dict):
        info.stack_name = str(data.get("stack") or data.get("stack_name") or info.stack_name).strip()
        info.python_version = str(data.get("python_version") or "").strip()
        info.allowed_commands = [str(c).strip() for c in (data.get("allowed_commands") or []) if str(c).strip()]
        ignore = data.get("agentignore")
        if isinstance(ignore, str):
            info.agentignore = [ln for ln in ignore.splitlines() if ln.strip()]
        elif isinstance(ignore, list):
            info.agentignore = [str(p).strip() for p in ignore if str(p).strip()]
        info.brief = str(data.get("brief") or "").strip()
    else:
        services.bus.log(
            "stack determination returned no parseable JSON — defaulting to python-fastapi",
            phase="stack",
            level="warn",
        )

    if not info.brief:
        info.brief = f"## Interpreted Request\n\n{prompt.strip()}"
    return info


def write_project_brief(services: Services, prompt: str, info: StackInfo) -> None:
    body = (
        "# Project Brief\n\n"
        f"{info.brief.strip()}\n\n"
        "## Original Prompt\n\n"
        f"{prompt.strip()}\n\n"
        f"## Stack\n\n{info.stack_name}\n"
    )
    services.workspace.write_agent_doc("project_brief.md", body)
