"""Shared service container + small text helpers used across stages.

:class:`Services` bundles every long-lived dependency (config, event bus, LLM
client, resolved model runtimes, context tooling, sandbox, registry) so stages
take a single argument. It is populated in phases by the orchestrator: the LLM
client and bus exist immediately; the model runtimes land after resolution;
the workspace-dependent services are built once the project directory is
resolved; the stack/environment info lands after Phase 1 Steps 1+3.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from config import AppConfig
from context.handoff import HandoffBuilder
from context.loader import Loader
from context.manifest import Manifest
from context.summaries import SummaryIndex
from llm.client import LLMClient
from server.events import EventBus
from tools.process_manager import ProcessManager
from tools.registry import ToolRegistry
from tools.sandbox import Sandbox
from workspace import Workspace

if TYPE_CHECKING:
    from environment import EnvInfo
    from llm.resolution import RuntimeModelConfig
    from stages.stack import StackInfo


@dataclass
class Progress:
    """In-memory, fine-grained run state surfaced by ``GET /project/state``.

    The durable task breakdown lives on disk in ``tasks.json``; this captures
    the sub-stage detail that only exists while the subtask loop is running so
    a UI can show an accurate phase badge and a per-subtask timer.
    """

    activity: str = ""  # handoff | implement | test | fix | escalate | decompose | summarize | ""
    subtask_id: str = ""
    subtask_title: str = ""
    subtask_intent: str = ""
    subtask_started_at: float = 0.0  # time.monotonic() at subtask start

    def begin_subtask(self, sub: dict) -> None:
        self.subtask_id = str(sub.get("id", ""))
        self.subtask_title = str(sub.get("title", ""))
        self.subtask_intent = str(sub.get("intent", ""))
        self.subtask_started_at = time.monotonic()
        self.activity = "handoff"

    def subtask_elapsed(self) -> float:
        return round(time.monotonic() - self.subtask_started_at, 1) if self.subtask_started_at else 0.0

    def clear(self) -> None:
        self.activity = ""
        self.subtask_id = self.subtask_title = self.subtask_intent = ""
        self.subtask_started_at = 0.0


@dataclass
class Services:
    config: AppConfig
    bus: EventBus
    client: LLMClient
    cancel_event: threading.Event
    pause_event: threading.Event = field(default_factory=threading.Event)
    progress: Progress = field(default_factory=Progress)

    # After model resolution (read via client.runtime_for as well).
    runtime: dict[str, "RuntimeModelConfig"] = field(default_factory=dict)

    workspace: Optional[Workspace] = None
    loader: Optional[Loader] = None
    manifest: Optional[Manifest] = None
    summaries: Optional[SummaryIndex] = None
    handoff_builder: Optional[HandoffBuilder] = None
    sandbox: Optional[Sandbox] = None
    process_manager: Optional[ProcessManager] = None
    registry: Optional[ToolRegistry] = None

    # After Phase 1 Steps 1+3.
    stack: Optional["StackInfo"] = None
    environment: Optional["EnvInfo"] = None

    def set_runtime(self, runtime: dict[str, "RuntimeModelConfig"]) -> None:
        self.runtime = dict(runtime)
        self.client.set_runtime(runtime)

    def attach_workspace(self, workspace: Workspace) -> None:
        """Build all workspace-dependent services once the project dir is known."""
        self.workspace = workspace
        self.client.set_workspace(workspace)
        self.loader = Loader(workspace)
        self.manifest = Manifest(workspace, self.bus)
        self.summaries = SummaryIndex(workspace)
        self.handoff_builder = HandoffBuilder(workspace, self.summaries, self.manifest, self.config)
        self.sandbox = Sandbox(workspace, self.config.sandbox, self.bus)
        self.process_manager = ProcessManager(workspace, self.sandbox, self.bus)
        self.registry = ToolRegistry(workspace, self.sandbox, self.manifest, self.bus, self.process_manager)

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def paused(self) -> bool:
        return self.pause_event.is_set()

    def check_cancel(self) -> None:
        """Cooperative checkpoint: hold here while paused, raise if cancelled.

        Called at every atomic boundary in the pipeline (between stages,
        subtasks, and worker tool calls). Because it is *not* called
        mid-LLM-stream, a pause requested during a stream takes effect once
        that stream completes.
        """
        self.wait_if_paused()
        if self.cancel_event.is_set():
            raise PipelineCancelled()

    def wait_if_paused(self) -> None:
        while self.pause_event.is_set():
            if self.cancel_event.is_set():
                raise PipelineCancelled()
            time.sleep(0.15)


class PipelineCancelled(Exception):
    """Raised cooperatively when a run is cancelled."""


# ── text helpers ──────────────────────────────────────────────────────────────
_FENCE_FULL_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*)\n```\s*$", re.DOTALL)


def clean_doc(text: str) -> str:
    """Unwrap a single outer code fence if the whole response is fenced.

    Local models sometimes wrap an entire markdown document in ``` fences. We
    strip exactly one such outer fence; inner fenced code stays intact.
    """
    if not text:
        return ""
    m = _FENCE_FULL_RE.match(text.strip())
    if m:
        return m.group(1).strip()
    return text.strip()
