"""Worker-facing text for `run` tool results.

The exit-code framing has to give the Worker enough context to self-correct
instead of blindly retrying the same failing command — this is a regression
guard for the exit-126 ("found, not executable") hint added alongside the
node_modules/.bin transparent rewriting (see tools/sandbox.py).
"""

from __future__ import annotations

from tools.registry import _render_run
from tools.sandbox import CommandResult


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
