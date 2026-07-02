"""Pipeline + subtask state enums and legal transitions (spec §7, §9)."""

from __future__ import annotations

from enum import Enum


class PipelineState(str, Enum):
    IDLE = "idle"
    INTAKE = "intake"
    REQUIREMENTS = "requirements"
    STACK = "stack"
    ARCHITECT = "architect"
    SDD = "sdd"
    TASK_PLANNING = "task_planning"
    SUBTASK_LOOP = "subtask_loop"
    REVIEW = "review"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


# Linear pipeline order (resumability jumps straight to SUBTASK_LOOP).
PIPELINE_ORDER = [
    PipelineState.INTAKE,
    PipelineState.REQUIREMENTS,
    PipelineState.STACK,
    PipelineState.ARCHITECT,
    PipelineState.SDD,
    PipelineState.TASK_PLANNING,
    PipelineState.SUBTASK_LOOP,
    PipelineState.REVIEW,
]

_TERMINAL = {PipelineState.DONE, PipelineState.ERROR, PipelineState.CANCELLED}

LEGAL_TRANSITIONS: dict[PipelineState, set[PipelineState]] = {
    PipelineState.IDLE: {PipelineState.INTAKE, PipelineState.SUBTASK_LOOP},  # SUBTASK_LOOP for resume
    PipelineState.INTAKE: {PipelineState.REQUIREMENTS},
    PipelineState.REQUIREMENTS: {PipelineState.STACK},
    PipelineState.STACK: {PipelineState.ARCHITECT},
    PipelineState.ARCHITECT: {PipelineState.SDD},
    PipelineState.SDD: {PipelineState.TASK_PLANNING},
    PipelineState.TASK_PLANNING: {PipelineState.SUBTASK_LOOP},
    PipelineState.SUBTASK_LOOP: {PipelineState.REVIEW},
    PipelineState.REVIEW: {PipelineState.DONE},
}
# Any state may transition to a terminal failure/cancel state.
for _s in list(PipelineState):
    LEGAL_TRANSITIONS.setdefault(_s, set())
    if _s not in _TERMINAL:
        LEGAL_TRANSITIONS[_s] |= {PipelineState.ERROR, PipelineState.CANCELLED}


def can_transition(src: PipelineState, dst: PipelineState) -> bool:
    return dst in LEGAL_TRANSITIONS.get(src, set())


class SubtaskState(str, Enum):
    SELECT = "select_subtask"
    PLAN = "plan"
    IMPLEMENT = "implement"
    WRITE_TESTS = "write_tests"
    RUN = "run"
    FIX = "fix"
    ESCALATE = "escalate"
    BLOCK = "block"
    DONE = "done"
