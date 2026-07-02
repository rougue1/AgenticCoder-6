"""STACK DECISION stage (spec §7.3) — choose + LOCK a concrete stack -> stack.md."""

from __future__ import annotations

from services import Services
from stages.common import generate_doc


def run(services: Services) -> str:
    return generate_doc(
        services,
        phase="stack_decider",
        template="stack_decider",
        out_name="stack.md",
        project_brief=services.loader.doc("project_brief.md"),
        requirements=services.loader.doc("requirements.md"),
    )
