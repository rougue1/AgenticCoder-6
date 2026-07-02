"""SDD GENERATION stage (spec §7.5) — produces sdd.md AND steering.md.

``sdd.md`` is the detailed, stack-specific design. ``steering.md`` is the
conventions/rules file injected into every implement/fix call to keep dozens of
stateless code-gen calls coherent.
"""

from __future__ import annotations

from services import Services
from stages.common import generate_doc


def run(services: Services) -> tuple[str, str]:
    stack = services.loader.doc("stack.md")
    architecture = services.loader.doc("architecture.md")
    requirements = services.loader.doc("requirements.md")

    sdd = generate_doc(
        services,
        phase="sdd_generator",
        template="sdd",
        out_name="sdd.md",
        stack=stack,
        architecture=architecture,
        requirements=requirements,
    )
    steering = generate_doc(
        services,
        phase="sdd_generator",
        template="steering",
        out_name="steering.md",
        stack=stack,
        architecture=architecture,
    )
    return sdd, steering
