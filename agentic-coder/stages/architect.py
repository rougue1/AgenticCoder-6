"""ARCHITECT stage (spec §7.4) — stack + requirements -> architecture.md.

Fully stack-specific: references the locked technologies by name.
"""

from __future__ import annotations

from services import Services
from stages.common import generate_doc


def run(services: Services) -> str:
    return generate_doc(
        services,
        phase="architect",
        template="architect",
        out_name="architecture.md",
        project_brief=services.loader.doc("project_brief.md"),
        requirements=services.loader.doc("requirements.md"),
        stack=services.loader.doc("stack.md"),
    )
