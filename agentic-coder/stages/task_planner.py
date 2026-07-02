"""TASK PLANNING stage (spec §7.6) — produce tasks.json (tasks + subtasks).

Calls the planner model for a strict JSON document, extracts it tolerantly,
normalizes it to the canonical schema, and persists it as ``tasks.json``.
"""

from __future__ import annotations

import promptlib
from llm.tool_parser import extract_json
from services import Services
from stages.common import SYSTEM_PROMPT
from taskstore import TaskStore


def run(services: Services) -> TaskStore:
    services.check_cancel()
    instruction = promptlib.render(
        "task_plan",
        stack=services.loader.doc("stack.md"),
        project_brief=services.loader.doc("project_brief.md"),
        requirements=services.loader.doc("requirements.md"),
        architecture=services.loader.doc("architecture.md"),
        sdd=services.loader.doc("sdd.md"),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + " Output ONLY valid JSON."},
        {"role": "user", "content": instruction},
    ]
    result = services.client.complete("task_planner", messages)

    data = extract_json(result.text) or extract_json(result.raw)
    if not isinstance(data, dict) or not data.get("tasks"):
        services.bus.error(
            "task planner did not return a usable tasks.json",
            context=(result.text or result.raw)[:1000],
            phase="task_planner",
        )
        raise ValueError("task planner produced no parseable tasks")

    store = TaskStore.from_data(services.workspace, data)
    services.manifest.regenerate()  # ensure manifest/dir files exist from the start
    return store
