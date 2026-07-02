"""Test-command extraction from plan -> test_strategy -> stack.md."""

from orchestrator.subtask_loop import extract_test_command


def test_from_plan_fenced_block():
    plan = "## Exact Command to Run Tests\n```bash\npython -m pytest tests/test_x.py -v\n```\n"
    assert extract_test_command(plan, {}, "") == "python -m pytest tests/test_x.py -v"


def test_from_test_strategy_inline_backtick():
    sub = {"test_strategy": "Run `pytest tests/` and expect a pass"}
    assert "pytest tests/" in extract_test_command("no command here", sub, "")


def test_from_stack_doc_test_line():
    assert extract_test_command("", {}, "- test: `npm test`\n") == "npm test"


def test_none_when_no_command_anywhere():
    assert extract_test_command("prose only", {}, "") is None
