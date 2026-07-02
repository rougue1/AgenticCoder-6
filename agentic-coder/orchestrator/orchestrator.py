"""Top-level pipeline state-machine driver (spec §7, §16, §18.7).

Wires the stages together: intake -> requirements -> stack -> architect -> sdd
-> task planning -> subtask loop -> review. Resolves the project directory
(deriving the slug from intake when ``project_dir`` is blank), builds the service
container once the workspace exists, drives each stage with start/end events,
supports cooperative cancellation, and resumes an existing run from disk.

Designed to run in a worker thread (started by ``POST /start``); it publishes to
the EventBus, which bridges back to the server's asyncio loop.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Optional

from config import AppConfig
from llm.client import LLMClient
from orchestrator.states import PipelineState, can_transition
from orchestrator.subtask_loop import SubtaskLoop
from server import events
from server.events import EventBus
from services import PipelineCancelled, Services
from stages import (
    architect,
    intake,
    requirements,
    reviewer,
    sdd_generator,
    stack_decider,
    task_planner,
)
from taskstore import TaskStore
from workspace import Workspace


class Orchestrator:
    def __init__(self, config: AppConfig, bus: EventBus):
        self.config = config
        self.bus = bus
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.client = LLMClient(config, bus)
        self.services = Services(
            config=config,
            bus=bus,
            client=self.client,
            cancel_event=self.cancel_event,
            pause_event=self.pause_event,
        )

        self.state: PipelineState = PipelineState.IDLE
        self.phase: str = ""
        self.prompt: str = ""
        self.workspace: Optional[Workspace] = None
        self.started_at: float = 0.0
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start_async(self, prompt: str, *, resume: bool = False) -> None:
        """Kick off the pipeline in a daemon worker thread."""
        if self._thread and self._thread.is_alive():
            raise RuntimeError("a pipeline run is already in progress")
        self.cancel_event.clear()
        self.pause_event.clear()
        self._thread = threading.Thread(
            target=self.run, args=(prompt,), kwargs={"resume": resume}, daemon=True, name="aiforge-pipeline"
        )
        self._thread.start()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.pause_event.clear()  # don't let a paused run get stuck ignoring cancel

    def pause(self) -> bool:
        """Cooperatively pause a running pipeline. Returns True if it took effect.

        Sets a flag the worker honors at its next atomic boundary (after the
        current tool call / LLM stream completes). No-op if nothing is running or
        it is already paused.
        """
        if not self.is_running() or self.pause_event.is_set():
            return False
        self.pause_event.set()
        self.bus.emit(events.PIPELINE_PAUSED, self.phase or "pipeline", **self._summary())
        return True

    def resume_pause(self) -> bool:
        """Unpause a paused, running pipeline. Returns True if it took effect."""
        if not self.pause_event.is_set():
            return False
        self.pause_event.clear()
        self.bus.emit(events.PIPELINE_RESUMED, self.phase or "pipeline", **self._summary())
        return True

    def is_paused(self) -> bool:
        return self.pause_event.is_set()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict:
        snap = {
            "state": self.state.value,
            "phase": self.phase,
            "running": self.is_running(),
            "project_dir": str(self.workspace.root) if self.workspace else self.config.project_dir,
            "elapsed": round(time.monotonic() - self.started_at, 1) if self.started_at else 0,
        }
        if self.workspace and self.workspace.agent_doc_exists("tasks.json"):
            try:
                snap["tasks"] = TaskStore.load(self.workspace).counts()
            except Exception:
                pass
        return snap

    # ── frontend-facing state (spec §Phase2 GET /project/state) ────────────────
    def active_workspace(self) -> Optional[Workspace]:
        """The workspace the UI should read files/manifest from.

        The live one if a run is (or was) in progress this session, otherwise one
        built from a configured ``project_dir`` so an existing on-disk project can
        be inspected/resumed before any run starts. ``None`` when neither exists.
        """
        if self.workspace is not None:
            return self.workspace
        pd = (self.config.project_dir or "").strip()
        if not pd:
            return None
        try:
            return Workspace.resolve(pd, None, self.config.tool_root)
        except Exception:
            return None

    def _disk_counts(self) -> Optional[dict]:
        ws = self.active_workspace()
        if ws is not None and ws.agent_doc_exists("tasks.json"):
            try:
                return TaskStore.load(ws).counts()
            except Exception:
                return None
        return None

    def derive_status(self) -> str:
        """Coarse status the frontend switches views on.

        idle | running | paused | done | blocked | error | cancelled.
        ``blocked`` means "not running, but an on-disk run has unfinished work
        that could be resumed" — what the launch screen offers a resume for.
        """
        if self.is_running():
            return "paused" if self.is_paused() else "running"
        if self.state == PipelineState.DONE:
            return "done"
        counts = self._disk_counts() or {}
        resumable = bool(counts.get("pending", 0) or counts.get("in_progress", 0) or counts.get("blocked", 0))
        if self.state == PipelineState.ERROR:
            return "blocked" if resumable else "error"
        if self.state == PipelineState.CANCELLED:
            return "blocked" if resumable else "cancelled"
        return "blocked" if resumable else "idle"

    def project_state(self) -> dict:
        """Full snapshot for ``GET /project/state`` (disk-derived where possible)."""
        ws = self.active_workspace()
        prog = self.services.progress
        out: dict = {
            "status": self.derive_status(),
            "phase": self.phase or "",
            "activity": prog.activity if self.state == PipelineState.SUBTASK_LOOP else "",
            "running": self.is_running(),
            "paused": self.is_paused(),
            "elapsed_seconds": round(time.monotonic() - self.started_at, 1) if self.started_at else 0.0,
            "subtask_elapsed_seconds": prog.subtask_elapsed(),
            "project_dir": str(ws.root) if ws else (self.config.project_dir or ""),
            "project_name": ws.root.name if ws else "",
            "current_task": "",
            "current_subtask": "",
            "current_subtask_intent": "",
            "subtask_index": 0,
            "subtask_total": 0,
            "task_index": 0,
            "task_total": 0,
            "subtask_local_index": 0,
            "subtask_local_total": 0,
            "done_count": 0,
            "blocked_count": 0,
            "pending_count": 0,
            "in_progress_count": 0,
        }
        if ws is None or not ws.agent_doc_exists("tasks.json"):
            if ws is not None:
                out["project_name"] = _project_name_from_brief(ws) or out["project_name"]
            return out
        try:
            store = TaskStore.load(ws)
        except Exception:
            return out

        counts = store.counts()
        out["done_count"] = counts.get("done", 0)
        out["blocked_count"] = counts.get("blocked", 0)
        out["pending_count"] = counts.get("pending", 0)
        out["in_progress_count"] = counts.get("in_progress", 0)
        out["subtask_total"] = store.total_subtasks()
        out["task_total"] = len(store.tasks)
        out["project_name"] = str(store.data.get("project") or out["project_name"])

        flat = list(store.subtasks())
        idx = _locate_current(flat, prog.subtask_id, store)
        if idx is not None:
            task, sub = flat[idx]
            out["current_task"] = str(task.get("title", ""))
            out["current_subtask"] = str(sub.get("title", ""))
            out["current_subtask_intent"] = str(sub.get("intent", "") or prog.subtask_intent)
            out["subtask_index"] = idx + 1
            subs = task.get("subtasks", [])
            out["subtask_local_total"] = len(subs)
            out["subtask_local_index"] = next(
                (i + 1 for i, s in enumerate(subs) if s.get("id") == sub.get("id")), 0
            )
            out["task_index"] = next(
                (i + 1 for i, t in enumerate(store.tasks) if t.get("id") == task.get("id")), 0
            )
        return out

    # ── main run ──────────────────────────────────────────────────────────────
    def run(self, prompt: str, *, resume: bool = False) -> None:
        self.prompt = prompt
        self.started_at = time.monotonic()
        try:
            if resume:
                self._resume()
            else:
                self._fresh_run(prompt)
            self._finish()
        except PipelineCancelled:
            self._set_state(PipelineState.CANCELLED)
            self.bus.emit(events.PIPELINE_COMPLETE, self.phase or "pipeline", result="cancelled", **self._summary())
        except Exception as exc:  # noqa: BLE001 - surface everything as an error event
            self._set_state(PipelineState.ERROR)
            self.bus.error(str(exc), context=traceback.format_exc()[-2000:], phase=self.phase or "pipeline")
            self.bus.emit(events.PIPELINE_COMPLETE, self.phase or "pipeline", result="error", message=str(exc), **self._summary())
        finally:
            # Free the last warm model so nothing lingers in VRAM/RAM after a
            # walk-away run (honors evict_on_model_switch; no-op when it's off).
            self.client.unload_all()

    def _fresh_run(self, prompt: str) -> None:
        tool_root = self.config.tool_root

        # INTAKE — may need to run before the workspace exists (slug source).
        if self.config.project_dir.strip():
            self._attach(Workspace.resolve(self.config.project_dir, None, tool_root))
            with self._stage(PipelineState.INTAKE, "intake", "Interpreting the request"):
                res = intake.run(self.services, prompt)
                self.workspace.write_agent_doc("project_brief.md", res.brief)
        else:
            with self._stage(PipelineState.INTAKE, "intake", "Interpreting the request"):
                res = intake.run(self.services, prompt)
                self._attach(Workspace.resolve("", res.slug, tool_root))
                self.workspace.write_agent_doc("project_brief.md", res.brief)
        self.bus.log(f"Project name: {res.project_name} (slug: {res.slug})", phase="intake")

        with self._stage(PipelineState.REQUIREMENTS, "requirements", "Deriving requirements"):
            requirements.run(self.services)
        with self._stage(PipelineState.STACK, "stack_decider", "Choosing the stack"):
            stack_decider.run(self.services)
        with self._stage(PipelineState.ARCHITECT, "architect", "Designing the architecture"):
            architect.run(self.services)
        with self._stage(PipelineState.SDD, "sdd_generator", "Writing the SDD + steering"):
            sdd_generator.run(self.services)
        with self._stage(PipelineState.TASK_PLANNING, "task_planner", "Planning tasks & subtasks"):
            store = task_planner.run(self.services)
            self.bus.log(f"Planned {store.total_subtasks()} subtasks across {len(store.tasks)} tasks", phase="task_planner")

        self._run_loop_and_review()

    def _resume(self) -> None:
        tool_root = self.config.tool_root
        if not self.config.project_dir.strip():
            raise ValueError("--resume requires a known project_dir (set it in config.yaml or pass --project-dir)")
        ws = Workspace.resolve(self.config.project_dir, None, tool_root)
        if not ws.agent_doc_exists("tasks.json"):
            raise ValueError(f"cannot resume: no tasks.json found in {ws.agent_dir}")
        self._attach(ws)
        store = TaskStore.load(ws)
        reset = store.reset_in_progress()
        self.bus.log(f"Resuming run in {ws.root} ({reset} in-progress subtask(s) reset to pending)", phase="resume")
        self._run_loop_and_review()

    def _run_loop_and_review(self) -> None:
        with self._stage(PipelineState.SUBTASK_LOOP, "subtask_loop", "Implementing subtasks"):
            loop_result = SubtaskLoop(self.services).run()
            self.bus.log(
                f"Subtask loop finished: {loop_result.done} done, {loop_result.blocked} blocked",
                phase="subtask_loop",
            )
        with self._stage(PipelineState.REVIEW, "reviewer", "Final review pass"):
            reviewer.run(self.services)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _attach(self, ws: Workspace) -> None:
        ws.ensure()
        self.workspace = ws
        self.services.attach_workspace(ws)
        self.bus.set_log_path(ws.agent_path("run.log"))
        self.bus.log(f"Project directory: {ws.root}", phase="setup")

    def _finish(self) -> None:
        self._set_state(PipelineState.DONE)
        self.bus.emit(events.PIPELINE_COMPLETE, "pipeline", result="done", **self._summary())

    def _summary(self) -> dict:
        out: dict = {"elapsed": round(time.monotonic() - self.started_at, 1) if self.started_at else 0}
        if self.workspace:
            out["project_dir"] = str(self.workspace.root)
            if self.workspace.agent_doc_exists("tasks.json"):
                try:
                    out["tasks"] = TaskStore.load(self.workspace).counts()
                except Exception:
                    pass
        return out

    def _set_state(self, state: PipelineState) -> None:
        if not can_transition(self.state, state):
            self.bus.log(f"unexpected state transition {self.state.value} -> {state.value}", level="warn")
        self.state = state

    def _stage(self, state: PipelineState, phase: str, title: str) -> "_StageCtx":
        return _StageCtx(self, state, phase, title)


class _StageCtx:
    """Context manager: set state, emit stage_start/stage_end, honor cancel."""

    def __init__(self, orch: Orchestrator, state: PipelineState, phase: str, title: str):
        self.orch = orch
        self.state = state
        self.phase = phase
        self.title = title

    def __enter__(self):
        self.orch.services.check_cancel()
        self.orch._set_state(self.state)
        self.orch.phase = self.phase
        self.orch.bus.stage_start(self.phase, self.title)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Always emit stage_end (note failure if one occurred); don't suppress.
        self.orch.bus.stage_end(self.phase, ok=exc_type is None)
        return False


def _locate_current(flat: list, tracked_id: str, store: "TaskStore") -> Optional[int]:
    """Index (into the flat (task, sub) list) of the subtask to report as current.

    Preference order: the one the loop says it's on -> any in_progress on disk ->
    the next runnable -> the last one (so a finished run reads as 100%).
    """
    if not flat:
        return None
    if tracked_id:
        for i, (_, sub) in enumerate(flat):
            if sub.get("id") == tracked_id:
                return i
    for i, (_, sub) in enumerate(flat):
        if sub.get("status") == "in_progress":
            return i
    nxt = store.next_runnable()
    if nxt is not None:
        for i, (_, sub) in enumerate(flat):
            if sub.get("id") == nxt[1].get("id"):
                return i
    return len(flat) - 1


def _project_name_from_brief(ws: "Workspace") -> str:
    """Best-effort project name from the first ``# Heading`` of project_brief.md."""
    brief = ws.read_agent_doc("project_brief.md") or ""
    for line in brief.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""
