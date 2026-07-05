"""Pipeline + subtask state enums and legal transitions (redesign)."""

from __future__ import annotations

from enum import Enum


class PipelineState(str, Enum):
    IDLE = "idle"
    RESOLUTION = "resolution"      # model capability resolution
    PREFLIGHT = "preflight"
    STACK = "stack"                # Step 1-2: determination + anchor
    ENVIRONMENT = "environment"    # Step 3-4: venv/node + .agentignore
    REQUIREMENTS = "requirements"  # Step 5
    ARCHITECTURE = "architecture"  # Step 6
    TASK_PLANNING = "task_planning"  # Step 7-8
    SUBTASK_LOOP = "subtask_loop"  # Phase 2
    FINAL_REVIEW = "final_review"  # Phase 3
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


# Linear pipeline order (resume jumps from PREFLIGHT to SUBTASK_LOOP).
PIPELINE_ORDER = [
    PipelineState.RESOLUTION,
    PipelineState.PREFLIGHT,
    PipelineState.STACK,
    PipelineState.ENVIRONMENT,
    PipelineState.REQUIREMENTS,
    PipelineState.ARCHITECTURE,
    PipelineState.TASK_PLANNING,
    PipelineState.SUBTASK_LOOP,
    PipelineState.FINAL_REVIEW,
]

_TERMINAL = {PipelineState.DONE, PipelineState.ERROR, PipelineState.CANCELLED}

LEGAL_TRANSITIONS: dict[PipelineState, set[PipelineState]] = {
    PipelineState.IDLE: {PipelineState.RESOLUTION},
    PipelineState.RESOLUTION: {PipelineState.PREFLIGHT},
    PipelineState.PREFLIGHT: {PipelineState.STACK, PipelineState.SUBTASK_LOOP},  # loop = resume
    PipelineState.STACK: {PipelineState.ENVIRONMENT},
    PipelineState.ENVIRONMENT: {PipelineState.REQUIREMENTS},
    PipelineState.REQUIREMENTS: {PipelineState.ARCHITECTURE},
    PipelineState.ARCHITECTURE: {PipelineState.TASK_PLANNING},
    PipelineState.TASK_PLANNING: {PipelineState.SUBTASK_LOOP},
    PipelineState.SUBTASK_LOOP: {PipelineState.FINAL_REVIEW},
    PipelineState.FINAL_REVIEW: {PipelineState.DONE},
}
# Any non-terminal state may transition to a terminal failure/cancel state.
for _s in list(PipelineState):
    LEGAL_TRANSITIONS.setdefault(_s, set())
    if _s not in _TERMINAL:
        LEGAL_TRANSITIONS[_s] |= {PipelineState.ERROR, PipelineState.CANCELLED}


def can_transition(src: PipelineState, dst: PipelineState) -> bool:
    return dst in LEGAL_TRANSITIONS.get(src, set())


class SubtaskOutcome(str, Enum):
    """Terminal outcome of one pass through the subtask ladder."""

    DONE = "done"
    BLOCKED = "blocked"
    DECOMPOSED = "decomposed"
