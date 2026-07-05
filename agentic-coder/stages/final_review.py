"""Phase 3 — Final Review (Manager, read-only).

After the subtask loop, one Manager call receives every summary from
``.agent/summaries/``, aggregated statistics from ``test_results.jsonl``
(total subtasks, first-try passes, fixed, escalated, decomposed, blocked),
``blocked.md``, and ``decisions.md``, and writes ``final_report.md``: project
summary, test/health statistics, blocked-subtask diagnosis, the major
architectural decisions, and recommended next steps / known limitations.

Read-only by construction: no tools are offered and nothing but the report
file is written.
"""

from __future__ import annotations

import json

import promptlib
from services import Services
from stages import manager
from taskstore import BLOCKED, DECOMPOSED, DONE, TaskStore


def run(services: Services) -> str:
    services.check_cancel()
    ws = services.workspace
    store = TaskStore.load(ws)

    stats = aggregate_stats(services, store)
    summaries = services.summaries.read_all()
    summaries_block = "\n\n".join(
        f"### {rel}\n{text}" for rel, text in summaries.items() if text
    ) or "(no file summaries recorded)"

    instruction = promptlib.render(
        "final_review",
        stats=json.dumps(stats, indent=2),
        task_summary=store.summary(),
        summaries=summaries_block,
        blocked=ws.read_agent_doc("blocked.md", "") or "(no blocked subtasks)",
        decisions=ws.read_agent_doc("decisions.md", "") or "(no decisions recorded)",
    )
    report = manager.call(services, "final_review", instruction)
    if not report.strip():
        report = "# Final Report\n\n_(the Manager produced no report content)_"
    ws.write_agent_doc("final_report.md", report)
    services.bus.log("final_report.md written", phase="final_review")
    return report


def aggregate_stats(services: Services, store: TaskStore) -> dict:
    """Roll test_results.jsonl + tasks.json into the report's health numbers."""
    counts = store.counts()
    attempts_by_subtask: dict[str, list[dict]] = {}
    raw = services.workspace.read_agent_doc("test_results.jsonl", "") or ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        attempts_by_subtask.setdefault(str(rec.get("subtask_id", "?")), []).append(rec)

    first_try = 0
    needed_fixes = 0
    for _, sub in store.subtasks():
        if sub.get("status") != DONE:
            continue
        runs = attempts_by_subtask.get(str(sub.get("id")), [])
        if len(runs) <= 1:
            first_try += 1
        else:
            needed_fixes += 1

    escalated = sum(
        1 for runs in attempts_by_subtask.values()
        if any(rec.get("escalation") for rec in runs)
    )

    return {
        "total_subtasks": store.total_subtasks(),
        "done": counts.get(DONE, 0),
        "passed_first_try": first_try,
        "needed_fixes": needed_fixes,
        "escalated": escalated,
        "decomposed": counts.get(DECOMPOSED, 0),
        "blocked": counts.get(BLOCKED, 0),
        "test_runs_logged": sum(len(v) for v in attempts_by_subtask.values()),
    }
