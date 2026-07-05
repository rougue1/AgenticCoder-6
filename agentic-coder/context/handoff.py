"""HandoffBuilder — all context assembly for per-subtask Manager calls.

The Manager's handoff input consists of exactly and only:

1. the full anchor text (never truncated — it is the system prompt),
2. the current subtask description (never truncated),
3. the architecture.md / requirements.md doc summaries (always included when
   they exist; the ``context.always_include`` list),
4. the recent entries of decisions.md (always included, already capped to
   ``decision_log_max_entries`` on write),
5. file summaries for the files this subtask touches + files its dependency
   subtasks touched (added in priority order up to the token budget),
6. the manifest-filtered file tree (added last, trimmed first).

Raw file contents NEVER appear here. ``.agent/`` never appears in the tree.
The assembled user-content is capped at ``context.max_handoff_tokens``,
trimming lowest-priority items first (tree, then distant summaries) — the
anchor rides separately as the system message and is never counted against
the trimmable budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tokens import estimate_tokens

if TYPE_CHECKING:
    from config import AppConfig
    from context.manifest import Manifest
    from context.summaries import SummaryIndex
    from taskstore import TaskStore
    from workspace import Workspace


@dataclass
class Handoff:
    """The assembled Manager input for one subtask."""

    system: str        # the anchor text (verbatim)
    user_context: str  # everything else, budget-capped
    trimmed: list[str]  # titles of sections dropped/cut to fit (for logging)


class HandoffBuilder:
    def __init__(self, workspace: "Workspace", summaries: "SummaryIndex", manifest: "Manifest", config: "AppConfig"):
        self.workspace = workspace
        self.summaries = summaries
        self.manifest = manifest
        self.config = config

    # ── public API ────────────────────────────────────────────────────────────
    def build(self, subtask: dict, store: "TaskStore") -> Handoff:
        budget = max(512, int(self.config.context.max_handoff_tokens))
        anchor = self.workspace.read_anchor_text()
        trimmed: list[str] = []

        # Priority 0-1: never truncated.
        fixed_parts: list[str] = [_subtask_block(subtask)]

        # Priority 2: always-include doc summaries.
        for doc in self.config.context.always_include:
            text = self.summaries.read(doc)
            if text:
                fixed_parts.append(f"=== {doc} (summary) ===\n{text}")

        # Priority 3: recent architectural decisions (kept trimmed on write).
        decisions = (self.workspace.read_agent_doc("decisions.md", "") or "").strip()
        if decisions:
            fixed_parts.append(f"=== Recent architectural decisions ===\n{decisions}")

        used = sum(estimate_tokens(p) for p in fixed_parts)

        # Priority 4: dependency-relevant file summaries, up to the budget.
        optional_parts: list[str] = []
        for rel in self._relevant_files(subtask, store):
            text = self.summaries.read(rel)
            if not text:
                continue
            part = f"=== FILE SUMMARY {rel} ===\n{text}"
            cost = estimate_tokens(part)
            if used + cost > budget:
                trimmed.append(f"summary:{rel}")
                continue
            optional_parts.append(part)
            used += cost

        # Priority 5 (lowest): the manifest-filtered file tree, trimmed to fit.
        tree = self.manifest.file_tree()
        tree_part = f"=== Workspace file tree (files written so far) ===\n{tree}"
        cost = estimate_tokens(tree_part)
        if used + cost <= budget:
            optional_parts.append(tree_part)
        else:
            room = budget - used
            if room > 64:  # include a truncated tree only if a useful amount fits
                keep_chars = max(0, room * 4 - 80)
                optional_parts.append(tree_part[:keep_chars] + "\n… (tree trimmed to fit context budget)")
                trimmed.append("file_tree:partial")
            else:
                trimmed.append("file_tree:dropped")

        return Handoff(
            system=anchor,
            user_context="\n\n".join(fixed_parts + optional_parts).strip(),
            trimmed=trimmed,
        )

    # ── internals ─────────────────────────────────────────────────────────────
    def _relevant_files(self, subtask: dict, store: "TaskStore") -> list[str]:
        """Files this subtask touches first, then files its dependencies touched."""
        ordered: list[str] = list(subtask.get("files") or [])
        for dep_id in subtask.get("dependencies") or []:
            dep = store.get_subtask(dep_id)
            if dep:
                ordered.extend(dep.get("files") or [])
        seen: set[str] = set()
        out: list[str] = []
        for f in ordered:
            rel = str(f).strip().replace("\\", "/")
            if rel and rel not in seen:
                seen.add(rel)
                out.append(rel)
        return out


def _subtask_block(subtask: dict) -> str:
    deps = ", ".join(subtask.get("dependencies") or []) or "(none)"
    files = "\n".join(f"  - {f}" for f in (subtask.get("files") or [])) or "  (none declared)"
    return (
        "=== CURRENT SUBTASK ===\n"
        f"id: {subtask.get('id', '')}\n"
        f"title: {subtask.get('title', '')}\n"
        f"type: {subtask.get('type', '')}\n"
        f"intent: {subtask.get('intent', '')}\n"
        f"dependencies: {deps}\n"
        f"test_command: {subtask.get('test_command') or '(none)'}\n"
        f"files:\n{files}"
    )
