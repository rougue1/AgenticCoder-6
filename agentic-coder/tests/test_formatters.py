"""Feature 4 — post-execution formatting hooks: extension -> formatter
selection (availability-first, not cascading after a real failure), "skip
silently when nothing is available," "never blocks on a formatting failure,"
and the ``formatter.run`` event on success."""

from __future__ import annotations

from server import events
from tools import formatters
from tools.sandbox import CommandResult


class _FakeSandbox:
    def __init__(self, available: set[str], results: dict[str, CommandResult] | None = None):
        self._available = available
        self._results = results or {}
        self.run_calls: list[str] = []

    def which(self, binary: str):
        return f"/usr/bin/{binary}" if binary in self._available else None

    def run(self, cmd: str, validate: bool = True):
        self.run_calls.append(cmd)
        for binary, result in self._results.items():
            if binary in cmd:
                return result
        return CommandResult(exit_code=0, stdout="", stderr="")


def test_unknown_extension_is_skipped_silently(bus):
    sandbox = _FakeSandbox(available={"ruff", "black", "prettier"})
    formatters.format_file(sandbox, "README.txt", bus, "test")
    assert sandbox.run_calls == []
    assert not bus.of_type(events.FORMATTER_RUN)


def test_no_available_formatter_is_skipped_silently(bus):
    sandbox = _FakeSandbox(available=set())
    formatters.format_file(sandbox, "app.py", bus, "test")
    assert sandbox.run_calls == []
    assert not bus.of_type(events.FORMATTER_RUN)
    assert not bus.of_type("log")


def test_picks_first_available_candidate_python(bus):
    sandbox = _FakeSandbox(available={"black"})  # ruff unavailable, black is
    formatters.format_file(sandbox, "app.py", bus, "test")
    assert len(sandbox.run_calls) == 1
    assert "black" in sandbox.run_calls[0] and "app.py" in sandbox.run_calls[0]


def test_prefers_ruff_over_black_when_both_available(bus):
    sandbox = _FakeSandbox(available={"ruff", "black"})
    formatters.format_file(sandbox, "app.py", bus, "test")
    assert "ruff format" in sandbox.run_calls[0]


def test_success_emits_formatter_run_event_with_label(bus):
    sandbox = _FakeSandbox(available={"prettier"})
    formatters.format_file(sandbox, "src/app.ts", bus, "worker")
    ran = bus.of_type(events.FORMATTER_RUN)
    assert len(ran) == 1
    assert ran[0].data["path"] == "src/app.ts"
    assert ran[0].data["formatter"] == "prettier"


def test_failure_logs_a_warning_and_never_raises(bus):
    sandbox = _FakeSandbox(
        available={"ruff"},
        results={"ruff": CommandResult(exit_code=1, stdout="", stderr="syntax error")},
    )
    formatters.format_file(sandbox, "app.py", bus, "test")  # must not raise
    warnings = [e for e in bus.of_type("log") if e.data.get("level") == "warn"]
    assert warnings and "app.py" in warnings[0].data["message"]
    assert not bus.of_type(events.FORMATTER_RUN)


def test_failure_does_not_cascade_to_the_next_fallback_candidate(bus):
    """A real formatting failure stops there — the fallback chain is for
    AVAILABILITY, not for retrying after black failed."""
    sandbox = _FakeSandbox(
        available={"ruff", "black"},
        results={"ruff": CommandResult(exit_code=1, stderr="boom")},
    )
    formatters.format_file(sandbox, "app.py", bus, "test")
    assert len(sandbox.run_calls) == 1  # only ruff was tried, never black


def test_gofmt_selected_for_go_files(bus):
    sandbox = _FakeSandbox(available={"gofmt"})
    formatters.format_file(sandbox, "main.go", bus, "test")
    assert "gofmt -w" in sandbox.run_calls[0]


def test_json_and_md_use_prettier_when_available(bus):
    sandbox = _FakeSandbox(available={"prettier"})
    formatters.format_file(sandbox, "data.json", bus, "test")
    formatters.format_file(sandbox, "README.md", bus, "test")
    assert len(sandbox.run_calls) == 2
    assert all("prettier --write" in c for c in sandbox.run_calls)
