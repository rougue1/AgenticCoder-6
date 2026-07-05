"""Phase 1, Steps 7-8 — Task Planning + Validation (Manager).

One anchored Manager call produces the full task/subtask breakdown in the
redesign schema. Immediately after, two validators run: per-subtask schema
(valid ``type``, ``test_command`` on implement/integrate, non-empty ``files``)
and a topological cycle check on the dependency graph. On failure the specific
error list is fed back to the Manager IN THE SAME SESSION for a corrected
tasks.json — up to 2 correction attempts before a fatal abort.
"""

from __future__ import annotations

import promptlib
from config import MANAGER
from llm.tool_parser import extract_json
from services import Services
from taskstore import TaskStore, normalize
from validation import validate_all

_MAX_CORRECTIONS = 2


def run(services: Services) -> TaskStore:
    services.check_cancel()
    ws = services.workspace
    anchor = ws.read_anchor_text()
    instruction = promptlib.render(
        "task_plan",
        project_brief=services.loader.doc("project_brief.md"),
        requirements=services.loader.doc("requirements.md"),
        architecture=services.loader.doc("architecture.md"),
    )
    # A growing session so validation feedback lands in context (Step 8).
    messages: list[dict] = [
        {"role": "system", "content": anchor or "You are the task-planning Manager."},
        {"role": "user", "content": instruction},
    ]

    attempts = 0
    while True:
        services.check_cancel()
        result = services.client.complete(MANAGER, "task_planner", messages)
        reply = result.text or result.raw
        messages.append({"role": "assistant", "content": reply})

        data = extract_json(reply)
        errors: list[str]
        normalized: dict | None = None
        if not isinstance(data, dict) or not data.get("tasks"):
            errors = ["output did not contain a parseable JSON object with a non-empty 'tasks' array"]
        else:
            normalized = normalize(data, ws.root.name)
            flat = [s for t in normalized.get("tasks", []) for s in t.get("subtasks", [])]
            if not flat:
                errors = ["the plan contains zero subtasks"]
            else:
                errors = validate_all(flat)

        if not errors and normalized is not None:
            store = TaskStore.from_data(ws, normalized)
            services.bus.log(
                f"task plan validated: {store.total_subtasks()} subtasks across {len(store.tasks)} tasks",
                phase="task_planner",
            )
            return store

        attempts += 1
        services.bus.log(
            f"task plan validation failed (attempt {attempts}/{_MAX_CORRECTIONS + 1}): "
            + "; ".join(errors[:8]),
            phase="task_planner",
            level="warn",
        )
        if attempts > _MAX_CORRECTIONS:
            raise ValueError(
                "task planning failed validation after "
                f"{_MAX_CORRECTIONS} correction attempts:\n  - " + "\n  - ".join(errors)
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your tasks.json failed validation with these specific errors:\n  - "
                    + "\n  - ".join(errors)
                    + "\n\nOutput the CORRECTED, complete tasks.json now. Output ONLY the "
                    "JSON object — no prose, no code fences, no commentary."
                ),
            }
        )
