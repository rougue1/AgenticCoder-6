"""Subtask verification: the redesign has no test-command *extraction* step —
``test_command`` is a plain declarative field on each subtask — but the
orchestrator's fallback verification logic (pytest exit 5 handling,
declared-file-presence acceptance, install-manifest detection) is still a
fragile heuristic worth covering directly on ``SubtaskLoop``."""

import threading

from config import load_config
from llm.client import LLMClient
from orchestrator.subtask_loop import SubtaskLoop
from services import Services
from tools.sandbox import CommandResult, Sandbox

from test_sandbox import require_bwrap


def _loop(workspace, bus):
    cfg = load_config()
    services = Services(config=cfg, bus=bus, client=LLMClient(cfg, bus), cancel_event=threading.Event())
    services.workspace = workspace
    services.sandbox = Sandbox(workspace, cfg.sandbox, bus)
    return SubtaskLoop(services)


def test_verify_runs_declared_test_command(workspace, bus):
    require_bwrap()  # the one test here that really executes a command
    loop = _loop(workspace, bus)
    sub = {"test_command": "echo ok", "type": "implement"}
    result = loop._verify(sub)
    assert result.passed is True
    assert result.cmd == "echo ok"


def test_pytest_exit5_treated_as_pass_for_non_tdd_type(workspace, bus):
    loop = _loop(workspace, bus)
    loop.services.sandbox.run = lambda cmd, **k: CommandResult(exit_code=5, stdout="no tests ran", stderr="")
    sub = {"test_command": "pytest", "type": "scaffold"}
    assert loop._verify(sub).passed is True


def test_pytest_exit5_fails_for_tdd_enforced_type(workspace, bus):
    loop = _loop(workspace, bus)
    loop.services.sandbox.run = lambda cmd, **k: CommandResult(exit_code=5, stdout="no tests ran", stderr="")
    sub = {"test_command": "pytest", "type": "implement"}
    assert loop._verify(sub).passed is False


def test_no_command_falls_back_to_declared_file_presence(workspace, bus):
    loop = _loop(workspace, bus)
    workspace.write_file("a.py", "print(1)\n")
    sub = {"test_command": "", "type": "config", "files": ["a.py"]}
    result = loop._verify(sub)
    assert result.passed is True
    assert result.cmd == "(declared files present)"


def test_no_command_and_missing_file_fails(workspace, bus):
    loop = _loop(workspace, bus)
    sub = {"test_command": "", "type": "config", "files": ["missing.py"]}
    assert loop._verify(sub).passed is False


def test_install_type_verified_via_manifest_install_command(workspace, bus):
    loop = _loop(workspace, bus)
    workspace.write_file("requirements.txt", "requests\n")
    seen: list[str] = []

    def _fake_run(cmd, **k):
        seen.append(cmd)
        return CommandResult(exit_code=0)

    loop.services.sandbox.run = _fake_run
    sub = {"test_command": "", "type": "install", "files": ["requirements.txt"]}
    result = loop._verify(sub)
    assert result.passed is True
    assert seen == ["pip install -r requirements.txt"]
