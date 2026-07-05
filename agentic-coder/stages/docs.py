"""Phase 1, Steps 5-6 — Requirements + Architecture documents (Manager).

Each is one anchored Manager call writing one ``.agent/`` document. After
writing, each document is summarized by the Analyst (``stages/summarizer``) so
every future handoff can always include a compact version without ever
shipping the raw document.
"""

from __future__ import annotations

import promptlib
from services import Services
from stages import manager


def run_requirements(services: Services) -> str:
    brief = services.loader.doc("project_brief.md")
    instruction = promptlib.render("requirements", project_brief=brief)
    content = manager.call(services, "requirements", instruction)
    services.workspace.write_agent_doc("requirements.md", content)
    return content


def run_architecture(services: Services) -> str:
    instruction = promptlib.render(
        "architecture",
        project_brief=services.loader.doc("project_brief.md"),
        requirements=services.loader.doc("requirements.md"),
        stack=services.stack.raw_output if services.stack else "",
    )
    content = manager.call(services, "architecture", instruction)
    services.workspace.write_agent_doc("architecture.md", content)
    return content
