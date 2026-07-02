"""Shared pytest fixtures for the AIForge tool test-suite.

These are FUNCTIONAL tests of the tool's own logic (parsers, sandbox, scheduling,
context budgeting, model-memory policy, throughput). They need no LLM. The
opt-in end-to-end test (``test_e2e_small.py``) drives the real pipeline with
SMALL Ollama models only — never the large 27B/30B models — so it can't OOM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make agentic-coder/ importable regardless of how pytest is invoked.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.events import EventBus  # noqa: E402
from workspace import Workspace  # noqa: E402
from config import load_config  # noqa: E402


class CaptureBus(EventBus):
    """A real EventBus (no asyncio loop bound) that also records every Event.

    ``emit`` with no bound loop just buffers the durable log line and skips SSE
    dispatch, so this is safe to use synchronously in tests while still letting
    them assert on what was emitted."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list = []

    def emit(self, type_: str, phase: str = "", **data):
        e = super().emit(type_, phase, **data)
        self.events.append(e)
        return e

    def of_type(self, t: str) -> list:
        return [e for e in self.events if e.type == t]


@pytest.fixture
def bus() -> CaptureBus:
    return CaptureBus()


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    ws = Workspace(tmp_path / "proj")
    ws.ensure()
    return ws


@pytest.fixture
def config():
    """The real shipped config (models/budgets/limits); override fields per test."""
    return load_config()
