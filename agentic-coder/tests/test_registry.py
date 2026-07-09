"""Worker-facing text for tool results.

The exit-code framing has to give the Worker enough context to self-correct
instead of blindly retrying the same failing command — this covers the
exit-126 ("found, not executable") hint and the tool-surface messaging around
the new session tools.
"""

from __future__ import annotations

import pytest

from config import SandboxCfg
from context.manifest import Manifest
from llm.tool_parser import ToolCall
from tools.registry import TOOL_INSTRUCTIONS, ToolRegistry, _render_run
from tools.sandbox import CommandResult, Sandbox


@pytest.fixture
def registry(workspace, bus):
    sandbox = Sandbox(workspace, SandboxCfg(), bus)
    return ToolRegistry(workspace, sandbox, Manifest(workspace, bus), bus)


def test_render_run_hints_at_permission_bit_on_exit_126():
    result = CommandResult(
        exit_code=126,
        stderr="sh: 1: node_modules/.bin/tsc: Permission denied",
        duration=0.01,
    )
    text = _render_run("node_modules/.bin/tsc --noEmit", result, background=False)
    assert "[hint]" in text
    assert "NOT EXECUTABLE" in text
    assert "npx" in text and "node node_modules/.bin/<tool>" in text


def test_render_run_omits_the_126_hint_for_a_rejected_command():
    """A validate()-rejected command also carries exit_code=126 by convention,
    but it was never even executed — the permission-bit hint would be
    misleading there, so it must not appear."""
    result = CommandResult(exit_code=126, rejected=True, reason="git is forbidden", duration=0.0)
    text = _render_run("git status", result, background=False)
    assert "[hint]" not in text


def test_render_run_omits_the_126_hint_when_exit_code_is_not_126():
    result = CommandResult(exit_code=1, stderr="AssertionError", duration=0.01)
    text = _render_run("pytest -q", result, background=False)
    assert "[hint]" not in text


def test_unknown_tool_error_lists_the_full_tool_surface(registry):
    result = registry.dispatch(ToolCall(name="bogus_tool", args={}), "test")
    assert not result.ok
    for tool in ("read_file", "write_file", "patch_file", "run", "check_session", "stop_session"):
        assert tool in result.display


def test_run_tool_reports_deny_list_rejection(registry):
    result = registry.dispatch(ToolCall(name="run", args={"cmd": "git status"}), "test")
    assert not result.ok
    assert result.payload.get("rejected") is True
    assert "COMMAND REJECTED" in result.display and "git" in result.display


def test_session_tools_require_a_session_id(registry):
    for tool in ("check_session", "stop_session"):
        result = registry.dispatch(ToolCall(name=tool, args={}), "test")
        assert not result.ok and "session_id" in result.display


def test_tool_instructions_document_the_session_workflow():
    """The prompt must teach the background/check/stop loop (and never the
    Ollama-native tool tag — that regression is covered in
    test_worker_tool_protocol, but the session surface is asserted here)."""
    for needle in ("background", "check_session", "stop_session", "session_id"):
        assert needle in TOOL_INSTRUCTIONS
    assert "smoke" not in TOOL_INSTRUCTIONS  # the old harness's arg is gone
