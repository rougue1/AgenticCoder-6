"""REQUIREMENTS stage (spec §7.2) — brief -> requirements.md. Stack-agnostic."""

from __future__ import annotations

from services import Services
from stages.common import generate_doc


def run(services: Services) -> str:
    brief = services.loader.doc("project_brief.md")
    return generate_doc(
        services,
        phase="requirements",
        template="requirements",
        out_name="requirements.md",
        project_brief=brief,
    )
