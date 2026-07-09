"""Feature 2 — Sub-agent roles (Manager).

Runs between stack determination (Step 1) and requirements generation (Step
5): one Manager call per fixed built-in role writes ``.agent/roles/<role>.md``,
tailored to the locked stack. Subtasks are later assigned one of these roles
during task planning (Step 7 — see ``taskstore._normalize_subtask``'s ``role``
field), and the Worker's system prompt is assembled from anchor + role file +
handoff instructions (``stages/worker.py::WorkerSession._system_prompt``).
"""

from __future__ import annotations

import promptlib
from services import Services
from stages import manager

# Fixed set of built-in roles + the domain each covers (used verbatim in the
# generation prompt). Order is deterministic (dict insertion order == the
# order role files are generated in).
ROLE_DESCRIPTIONS: dict[str, str] = {
    "backend": "backend service development, API design, database models, server logic, business rules",
    "frontend": "UI components, styling, client-side state management, user interaction handling",
    "database": "schema design, migrations, queries, ORM models, indexing",
    "infrastructure": "config files, Dockerfiles, CI/CD pipelines, deployment scripts, environment config",
    "testing": "test setup, fixtures, mocks, integration test infrastructure, test patterns",
    "review": "code review checklist, quality standards, common anti-patterns to flag",
}

DEFAULT_ROLE = "backend"


def run(services: Services) -> None:
    """Generate every role file under ``.agent/roles/`` (one Manager call each)."""
    stack = services.stack.raw_output if services.stack else ""
    brief = services.loader.doc("project_brief.md") if services.loader else ""
    for role, description in ROLE_DESCRIPTIONS.items():
        instruction = promptlib.render(
            "roles",
            role=role,
            role_description=description,
            stack=stack,
            project_brief=brief,
        )
        content = manager.call(services, "roles", instruction)
        services.workspace.write_agent_doc(f"roles/{role}.md", content)
    services.bus.log(
        f"generated {len(ROLE_DESCRIPTIONS)} role definition(s) in .agent/roles/", phase="roles"
    )


def read_role(services: Services, role: str) -> str:
    """Role instructions for *role* (case-insensitive).

    Falls back to ``backend.md``, then to the first role file found on disk,
    then to ``""`` when ``.agent/roles/`` doesn't exist at all (e.g. a project
    resumed from before this feature existed) — the Worker's system prompt
    just omits the role section in that case.
    """
    ws = services.workspace
    if ws is None:
        return ""
    role = (role or "").strip().lower()
    if role:
        text = ws.read_agent_doc(f"roles/{role}.md")
        if text and text.strip():
            return text
    fallback = ws.read_agent_doc(f"roles/{DEFAULT_ROLE}.md")
    if fallback and fallback.strip():
        return fallback
    for name in ROLE_DESCRIPTIONS:
        text = ws.read_agent_doc(f"roles/{name}.md")
        if text and text.strip():
            return text
    return ""
