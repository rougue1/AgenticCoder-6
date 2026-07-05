"""Top-level pipeline driver (redesign).

Fresh run:  model resolution -> pre-flight -> Phase 1 (stack determination ->
anchor -> environment -> .agentignore -> requirements -> architecture -> task
planning + validation) -> Phase 2 (subtask loop) -> Phase 3 (final review).

Resume: model resolution re-runs from scratch (never persisted), then the
hardening checks (anchor integrity, dependency re-validation, venv integrity,
missing summaries re-generated, in_progress reset) before jumping straight to
the subtask loop.

Designed to run in a worker thread (started by ``POST /start``); it publishes
to the EventBus, which bridges back to the server's asyncio loop.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Optional

import yaml

from config import AppConfig
from environment import setup_environment, verify_venv
from llm.client import LLMClient
from llm.resolution import resolve_all
from orchestrator.states import PipelineState, can_transition
from orchestrator.subtask_loop import SubtaskLoop
from preflight import run_preflight
from server import events
from server.events import EventBus
from services import PipelineCancelled, Services
from stackprofiles import StackProfile, default_profile, resolve_profile
from stages import docs, final_review, manager, summarizer, task_planner
from stages import stack as stack_stage
from stages.stack import StackInfo
from taskstore import DONE, PENDING, TaskStore
from validation import validate_dependencies
from workspace import Workspace, slug_from_prompt


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
        """Cooperatively pause a running pipeline. Returns True if it took effect."""
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

    # ── frontend-facing state (GET /project/state) ─────────────────────────────
    def active_workspace(self) -> Optional[Workspace]:
        """The workspace the UI should read files/manifest from."""
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
        """Coarse status a frontend switches views on:
        idle | running | paused | done | blocked | error | cancelled."""
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
            "decomposed_count": 0,
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
        out["decomposed_count"] = counts.get("decomposed", 0)
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
        self.bus.emit(events.PIPELINE_START, "pipeline", mode="resume" if resume else "fresh")
        try:
            # MODEL CAPABILITY RESOLUTION — before pre-flight, once per run,
            # never persisted (models can change between sessions).
            with self._stage(PipelineState.RESOLUTION, "resolution", "Resolving model capabilities"):
                self.services.set_runtime(resolve_all(self.config, self.bus))

            if resume:
                self._resume()
            else:
                self._fresh_run(prompt)
            self._finish()
        except PipelineCancelled:
            self._set_state(PipelineState.CANCELLED)
            self.bus.emit(events.PIPELINE_CANCELLED, self.phase or "pipeline", result="cancelled", **self._summary())
        except Exception as exc:  # noqa: BLE001 - surface everything as an error event
            self._set_state(PipelineState.ERROR)
            self.bus.error(str(exc), context=traceback.format_exc()[-2000:], phase=self.phase or "pipeline")
            self.bus.emit(events.PIPELINE_COMPLETE, self.phase or "pipeline", result="error", message=str(exc), **self._summary())
        finally:
            # Free the last warm model so nothing lingers in VRAM/RAM after a
            # walk-away run (honors evict_on_model_switch; no-op when off).
            self.client.unload_all()

    # ── fresh run: Phase 1 ─────────────────────────────────────────────────────
    def _fresh_run(self, prompt: str) -> None:
        tool_root = self.config.tool_root

        # The workspace must exist BEFORE pre-flight (writability check), so the
        # slug is derived programmatically from the prompt — no LLM involved.
        slug = slug_from_prompt(prompt)
        self._attach(Workspace.resolve(self.config.project_dir, slug, tool_root))

        with self._stage(PipelineState.PREFLIGHT, "preflight", "Pre-flight validation"):
            run_preflight(self.config, self.workspace, default_profile(self.config.sandbox.stack_profile), self.bus)

        # Step 1+2 — stack determination, project brief, the immutable anchor.
        with self._stage(PipelineState.STACK, "stack", "Determining the stack + anchor"):
            info = stack_stage.run(self.services, prompt)
            self.services.stack = info
            stack_stage.write_project_brief(self.services, prompt, info)
            self._write_anchor(prompt, info)
            profile = resolve_profile(self.config.sandbox.stack_profile, info.stack_name)
            self._configure_sandbox(profile, info.allowed_commands)
            self.bus.log(f"stack locked: {info.stack_name} (profile: {profile.name})", phase="stack")

        # Step 3+4 — environment setup (orchestrator-side) + .agentignore.
        with self._stage(PipelineState.ENVIRONMENT, "environment", "Setting up the environment"):
            env = setup_environment(self.workspace, profile, self.bus, preferred_python=info.python_version)
            self.services.environment = env
            self._apply_env_to_sandbox(env)
            self.workspace.write_agentignore(profile.default_agentignore + info.agentignore)

        # Step 5 — requirements (+ Analyst summary for future handoffs).
        with self._stage(PipelineState.REQUIREMENTS, "requirements", "Deriving requirements"):
            docs.run_requirements(self.services)
            summarizer.summarize_doc(self.services, "requirements.md")

        # Step 6 — architecture (+ Analyst summary).
        with self._stage(PipelineState.ARCHITECTURE, "architecture", "Designing the architecture"):
            docs.run_architecture(self.services)
            summarizer.summarize_doc(self.services, "architecture.md")

        # Step 7+8 — task planning with schema + cycle validation (2 corrections).
        with self._stage(PipelineState.TASK_PLANNING, "task_planner", "Planning tasks & subtasks"):
            store = task_planner.run(self.services)
            self.bus.log(
                f"planned {store.total_subtasks()} subtasks across {len(store.tasks)} tasks",
                phase="task_planner",
            )

        self._run_loop_and_review()

    # ── the anchor (Step 2) ─────────────────────────────────────────────────────
    def _write_anchor(self, prompt: str, info: StackInfo) -> None:
        """Programmatic concatenation: prompt + stack determination -> anchor.

        Written exactly once; the workspace layer hard-rejects any rewrite."""
        combined = (
            "You are the Manager of an autonomous build pipeline: a pragmatic staff "
            "engineer. Everything you plan, hand off, or review MUST stay inside the "
            "locked stack below. Never use git. Be concrete and decisive.\n\n"
            f"## Original Request\n\n{prompt.strip()}\n\n"
            f"## Stack Determination (locked)\n\n{info.raw_output.strip()}"
        )
        frontmatter = yaml.safe_dump(
            {
                "original_prompt": prompt,
                "stack": info.stack_name,
                "combined_anchor": combined,
            },
            sort_keys=False,
            allow_unicode=True,
            width=100000,
        )
        self.workspace.write_agent_doc("anchor.md", f"---\n{frontmatter}---\n\n{combined}\n")
        self.bus.log("anchor.md written (immutable from now on)", phase="stack")

    def _configure_sandbox(self, profile: StackProfile, manager_commands: list[str]) -> None:
        allowed = list(dict.fromkeys(profile.base_allowed_commands + manager_commands))
        self.services.sandbox.set_allowed_commands(allowed)
        self.bus.log(f"sandbox allowlist: {len(allowed)} commands", phase="stack")

    def _apply_env_to_sandbox(self, env) -> None:
        if env.venv_path is not None:
            self.services.sandbox.set_venv(env.venv_path)
        if env.node_root is not None:
            self.services.sandbox.set_node_bin(env.node_root)

    # ── resume (hardened) ───────────────────────────────────────────────────────
    def _resume(self) -> None:
        tool_root = self.config.tool_root
        if not self.config.project_dir.strip():
            raise ValueError("--resume requires a known project_dir (set it in config.yaml or pass --project-dir)")
        ws = Workspace.resolve(self.config.project_dir, None, tool_root)
        if not ws.agent_doc_exists("tasks.json"):
            raise ValueError(f"cannot resume: no tasks.json found in {ws.agent_dir}")
        self._attach(ws)

        # 1. Anchor integrity: must exist and parse with the required fields.
        anchor_meta, anchor_body = _parse_anchor(ws)
        info = _stack_info_from_anchor(anchor_meta, anchor_body)
        self.services.stack = info

        profile = resolve_profile(self.config.sandbox.stack_profile, info.stack_name)
        with self._stage(PipelineState.PREFLIGHT, "preflight", "Pre-flight validation (resume)"):
            run_preflight(self.config, ws, profile, self.bus)
        self._configure_sandbox(profile, info.allowed_commands)

        store = TaskStore.load(ws)

        # 2. Re-validate the dependency graph (manual edits can introduce cycles).
        dep_errors = validate_dependencies(store.all_subtasks())
        if dep_errors:
            raise ValueError(
                "cannot resume: tasks.json failed dependency validation:\n  - " + "\n  - ".join(dep_errors)
            )

        # 3. Venv integrity: recreate if broken and re-run done install subtasks.
        if profile.uses_venv:
            venv_ok = verify_venv(ws)
            env = setup_environment(self.workspace, profile, self.bus, preferred_python=info.python_version)
            self.services.environment = env
            self._apply_env_to_sandbox(env)
            if not venv_ok:
                rerun = [
                    sub["id"]
                    for _, sub in store.subtasks()
                    if sub.get("status") == DONE and str(sub.get("type") or "").lower() == "install"
                ]
                for sid in rerun:
                    store.set_status(sid, PENDING)
                if rerun:
                    self.bus.log(
                        f"venv was missing/broken — recreated it and re-queued install subtask(s): "
                        + ", ".join(rerun),
                        phase="resume",
                        level="warn",
                    )
        self.services.manifest.forget_missing()

        # 4. Summaries must exist for every done subtask; re-run Step C if not.
        self._backfill_summaries(store)

        # 5. Reset any in_progress subtasks back to pending.
        reset = store.reset_in_progress()
        self.bus.log(f"resuming run in {ws.root} ({reset} in-progress subtask(s) reset to pending)", phase="resume")

        self._run_loop_and_review()

    def _backfill_summaries(self, store: TaskStore) -> None:
        for _, sub in store.subtasks():
            if sub.get("status") != DONE:
                continue
            existing_files = [
                f for f in (sub.get("files") or []) if self.workspace.file_exists(str(f))
            ]
            missing = [f for f in existing_files if not self.services.summaries.exists(str(f))]
            if missing:
                self.bus.log(
                    f"re-running Step C for {sub.get('id')}: {len(missing)} missing summar(ies)",
                    phase="resume",
                )
                summarizer.summarize_files(self.services, str(sub.get("id")), [str(f) for f in missing])
        for doc in self.config.context.always_include:
            if self.workspace.agent_doc_exists(doc) and not self.services.summaries.exists(doc):
                summarizer.summarize_doc(self.services, doc)

    # ── Phase 2 + 3 ─────────────────────────────────────────────────────────────
    def _run_loop_and_review(self) -> None:
        with self._stage(PipelineState.SUBTASK_LOOP, "subtask_loop", "Implementing subtasks"):
            loop_result = SubtaskLoop(self.services).run()
            self.bus.log(
                f"subtask loop finished: {loop_result.done} done, {loop_result.blocked} blocked, "
                f"{loop_result.decomposed} decomposed",
                phase="subtask_loop",
            )
        with self._stage(PipelineState.FINAL_REVIEW, "final_review", "Final review (read-only)"):
            final_review.run(self.services)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _attach(self, ws: Workspace) -> None:
        ws.ensure()
        self.workspace = ws
        self.services.attach_workspace(ws)
        self.bus.set_log_path(ws.agent_path("run.log"))
        self.bus.log(f"project directory: {ws.root}", phase="setup")

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
    """Context manager: set state, emit stage.start/stage.end, honor cancel."""

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
        # Always emit stage.end (noting failure if one occurred); don't suppress.
        self.orch.bus.stage_end(self.phase, ok=exc_type is None)
        return False


# ── anchor parsing (resume hardening) ───────────────────────────────────────────
def _parse_anchor(ws: Workspace) -> tuple[dict, str]:
    raw = ws.read_agent_doc("anchor.md")
    if not raw:
        raise ValueError("cannot resume: .agent/anchor.md is missing — the anchor is required")
    if not raw.startswith("---"):
        raise ValueError("cannot resume: anchor.md has no YAML frontmatter")
    parts = raw.split("---", 2)
    if len(parts) < 3:
        raise ValueError("cannot resume: anchor.md frontmatter is malformed (unterminated ---)")
    try:
        meta = yaml.safe_load(parts[1])
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot resume: anchor.md frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError("cannot resume: anchor.md frontmatter is not a YAML mapping")
    missing = [k for k in ("original_prompt", "stack", "combined_anchor") if not meta.get(k)]
    if missing:
        raise ValueError(f"cannot resume: anchor.md frontmatter is missing required field(s): {', '.join(missing)}")
    return meta, parts[2].strip()


def _stack_info_from_anchor(meta: dict, body: str) -> StackInfo:
    """Reconstruct the stack determination (incl. allowed_commands) from the
    anchor body — the determination output is embedded there verbatim."""
    from llm.tool_parser import extract_json

    info = StackInfo(stack_name=str(meta.get("stack") or "python-fastapi"), raw_output=body)
    data = extract_json(body)
    if isinstance(data, dict):
        info.python_version = str(data.get("python_version") or "").strip()
        info.allowed_commands = [str(c).strip() for c in (data.get("allowed_commands") or []) if str(c).strip()]
    return info


def _locate_current(flat: list, tracked_id: str, store: "TaskStore") -> Optional[int]:
    """Index (into the flat (task, sub) list) of the subtask to report as current."""
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
