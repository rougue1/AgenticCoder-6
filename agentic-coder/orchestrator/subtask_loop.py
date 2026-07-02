"""The subtask loop (spec §9) — implement -> test -> fix -> escalate -> block.

This is the heart of the pipeline. For each pending subtask whose dependencies
are done it: plans (large model), opens an ephemeral conversation to implement +
write tests (small model), runs the tests (orchestrator executes — the model only
*requests* the command), and on failure walks the bounded ladder:

    fix (same convo, small model)  x max_fix_retries
      -> escalate (fresh plan from large model + full failure history)  x max_escalations
        -> block (record to blocked.md, skip dependents)

Caps come from ``config.limits``; the loop can never run unbounded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from orchestrator.states import SubtaskState
from server import events
from services import Services
from stages import planner
from stages.implementer import Implementer
from taskstore import BLOCKED, DONE, IN_PROGRESS, TaskStore
from tools.sandbox import normalize_pytest_command


def _npm_install(path: str) -> str:
    """npm install for a package.json; --prefix targets a manifest in a subdir."""
    d = path.rsplit("/", 1)[0] if "/" in path else ""
    return f"npm --prefix {d} install" if d else "npm install"


# Dependency manifests we know how to install from when a subtask ships no test
# command. The sandbox classifies these as install steps and lets them reach the
# network. Order matters only for which is tried first.
_MANIFEST_INSTALL = [
    ("requirements.txt", lambda p: f"pip install -r {p}"),
    ("requirements-dev.txt", lambda p: f"pip install -r {p}"),
    ("package.json", _npm_install),
]


@dataclass
class FailureRecord:
    attempt: int
    cmd: str
    exit_code: int
    stdout: str
    stderr: str
    note: str = ""

    def render(self) -> str:
        return (
            f"### Attempt {self.attempt} — `{self.cmd}` (exit {self.exit_code})\n"
            f"{self.note}\n"
            f"STDOUT:\n{_tail(self.stdout)}\n\nSTDERR:\n{_tail(self.stderr)}"
        )


@dataclass
class LoopResult:
    done: int = 0
    blocked: int = 0
    blocked_ids: list[str] = field(default_factory=list)


class SubtaskLoop:
    def __init__(self, services: Services):
        self.services = services
        self.cfg = services.config.limits
        self.bus = services.bus
        self.stack_doc = ""

    def run(self) -> LoopResult:
        """Process every runnable subtask until none remain."""
        self.stack_doc = self.services.loader.doc("stack.md")
        result = LoopResult()
        while True:
            self.services.check_cancel()
            store = TaskStore.load(self.services.workspace)
            picked = store.next_runnable()
            if picked is None:
                break
            task, sub = picked
            outcome = self._process_subtask(store, task, sub)
            if outcome == SubtaskState.DONE:
                result.done += 1
            elif outcome == SubtaskState.BLOCK:
                result.blocked += 1
                result.blocked_ids.append(sub["id"])
        self.services.progress.clear()
        return result

    # ── one subtask through the ladder ────────────────────────────────────────
    def _process_subtask(self, store: TaskStore, task: dict, sub: dict) -> SubtaskState:
        sid = sub["id"]
        store.set_status(sid, IN_PROGRESS)
        self.services.progress.begin_subtask(sub)
        self.bus.emit(events.SUBTASK_START, "subtask_loop", id=sid, title=sub.get("title", ""))

        failures: list[FailureRecord] = []
        escalations_left = self.cfg.max_escalations

        # initial plan (PLAN state)
        self.services.progress.activity = "plan"
        plan = planner.run(self.services, task, sub)

        while True:  # escalation loop
            self.services.check_cancel()
            self.services.progress.activity = "implement"
            impl = Implementer(self.services, task, sub, plan)  # fresh ephemeral conversation
            impl.implement_and_write_tests()  # IMPLEMENT + WRITE_TESTS

            fix_left = self.cfg.max_fix_retries
            attempt = 0
            while True:  # implement/fix -> run loop
                self.services.check_cancel()
                attempt += 1
                self.services.progress.activity = "test"
                cmd, passed, out, err, code = self._run_tests(plan, sub)
                self.bus.test_run(
                    "subtask_loop",
                    cmd or "(no test command)",
                    code,
                    passed,
                    output=_tail(((err or "") + ("\n" + out if out else "")).strip(), 1500),
                )

                if passed:
                    self._mark_done(store, sid)
                    return SubtaskState.DONE

                failures.append(
                    FailureRecord(
                        attempt=len(failures) + 1,
                        cmd=cmd or "(no test command found)",
                        exit_code=code,
                        stdout=out,
                        stderr=err,
                        note="" if cmd else "Plan did not specify a runnable test command.",
                    )
                )
                self.bus.emit(events.SUBTASK_FAILED, "subtask_loop", id=sid, attempt=attempt, exit_code=code)

                if fix_left > 0:
                    fix_left -= 1
                    self.services.progress.activity = "fix"
                    impl.fix(  # FIX (same conversation)
                        cmd=cmd or "(no test command)",
                        exit_code=code,
                        stdout=out,
                        stderr=err,
                        attempt=self.cfg.max_fix_retries - fix_left,
                        max_attempts=self.cfg.max_fix_retries,
                    )
                    continue
                break  # fixes exhausted -> escalate or block

            if escalations_left > 0:  # ESCALATE
                escalations_left -= 1
                self.services.progress.activity = "escalate"
                self.bus.emit(
                    events.ESCALATION,
                    "subtask_loop",
                    id=sid,
                    escalations_left=escalations_left,
                    failures=len(failures),
                )
                plan = planner.run(
                    self.services,
                    task,
                    sub,
                    failure_history="\n\n".join(f.render() for f in failures),
                    phase="escalation",
                )
                continue  # discard old convo, re-enter IMPLEMENT with the new plan

            self._mark_blocked(store, task, sub, failures)  # BLOCK
            return SubtaskState.BLOCK

    # ── RUN ───────────────────────────────────────────────────────────────────
    def _run_tests(self, plan: str, sub: dict):
        cmd = extract_test_command(plan, sub, self.stack_doc)
        if not cmd:
            # The planner gave no runnable test command (common for scaffold/manifest
            # subtasks like "create requirements.txt"). Don't report a guaranteed
            # failure every attempt and burn the whole fix/escalate ladder for
            # nothing — verify what is actually checkable instead.
            return self._fallback_verification(sub)
        cmd = normalize_pytest_command(cmd)
        result = self.services.sandbox.run(cmd)
        passed = result.ok
        # pytest exit 5 = "no tests collected". For scaffold/setup subtasks the
        # spec's own criterion is "the test runner executes an empty suite
        # successfully", so a working runner that finds nothing is a pass.
        if not passed and result.exit_code == 5 and "pytest" in cmd:
            passed = True
            self.bus.log("pytest collected no tests (exit 5) — empty suite treated as pass", phase="subtask_loop")
        return cmd, passed, result.stdout, result.stderr, result.exit_code

    # ── fallback verification (no planner-provided test command) ──────────────
    def _fallback_verification(self, sub: dict):
        """Verify a subtask that declared no runnable test command.

        Every subtask is supposed to ship a concrete test command, but local models
        sometimes omit one (typically for scaffold/manifest subtasks). Rather than
        report a guaranteed failure each attempt — which pointlessly walks the whole
        fix -> escalate -> block ladder — verify what is actually checkable:

        1. If the subtask produced a dependency manifest, prove the dependencies
           INSTALL. This also performs the install the scaffold needs so later
           subtasks' imports resolve, and surfaces a broken/hallucinated manifest as
           a real, fixable error instead of a silent skip.
        2. Otherwise accept iff every file the subtask declared now exists and is
           non-empty; if not, report a clear failure.
        """
        install = self._manifest_install_command(sub)
        if install:
            res = self.services.sandbox.run(install)
            if res.ok:
                self.bus.log(f"no test command; verified dependencies install via `{install}`", phase="subtask_loop")
            return install, res.ok, res.stdout, res.stderr, res.exit_code

        files = [str(f).strip() for f in (sub.get("files") or []) if str(f).strip()]
        missing = [f for f in files if not self._file_present(f)]
        if files and not missing:
            self.bus.log(f"no test command; accepted on file presence ({len(files)} file(s))", phase="subtask_loop")
            return "(no test command; declared files present)", True, "", "", 0
        msg = (
            "No runnable test command for this subtask and it produced no verifiable "
            f"output. Missing or empty files: {', '.join(missing) or '(none declared)'}."
        )
        return None, False, "", msg, 1

    def _manifest_install_command(self, sub: dict) -> str | None:
        """Install command for a dependency manifest this subtask created, if any."""
        declared = [str(f).strip().replace("\\", "/") for f in (sub.get("files") or [])]
        for name, make_cmd in _MANIFEST_INSTALL:
            for d in declared:
                if (d == name or d.endswith("/" + name)) and self._file_present(d):
                    return make_cmd(d)
        # Catch a root requirements.txt even if this subtask didn't declare it.
        if self._file_present("requirements.txt"):
            return "pip install -r requirements.txt"
        return None

    def _file_present(self, rel: str) -> bool:
        """True if *rel* resolves inside the project root and is a non-empty file."""
        try:
            p = self.services.workspace.resolve_in_root(rel)
        except Exception:
            return False
        try:
            return p.is_file() and p.stat().st_size > 0
        except OSError:
            return False

    # ── state updates ─────────────────────────────────────────────────────────
    def _mark_done(self, store: TaskStore, sid: str) -> None:
        store.set_status(sid, DONE)
        self.services.manifest.regenerate()
        self.bus.emit(events.SUBTASK_DONE, "subtask_loop", id=sid)

    def _mark_blocked(self, store: TaskStore, task: dict, sub: dict, failures: list[FailureRecord]) -> None:
        sid = sub["id"]
        store.set_status(sid, BLOCKED)
        last = failures[-1] if failures else None
        summary = (
            f"## {sid} — {sub.get('title','')}\n"
            f"- Parent task: {task.get('id')} {task.get('title','')}\n"
            f"- Intent: {sub.get('intent','')}\n"
            f"- Attempts: {len(failures)} (fixes + escalations exhausted)\n"
            f"- Last command: `{last.cmd if last else 'n/a'}` (exit {last.exit_code if last else 'n/a'})\n"
            f"- Last error:\n```\n{_tail(last.stderr if last else '', 1500)}\n```\n"
            f"- Recorded: {datetime.now(timezone.utc).isoformat()}\n"
        )
        existing = self.services.workspace.read_agent_doc("blocked.md") or "# Blocked Subtasks\n\n"
        self.services.workspace.write_agent_doc("blocked.md", existing.rstrip() + "\n\n" + summary)
        self.bus.emit(events.BLOCKED, "subtask_loop", id=sid, attempts=len(failures))


# ── test-command extraction ────────────────────────────────────────────────────
_RUNNER_RE = re.compile(
    r"^\s*(?:\$\s*)?((?:npm|npx|pnpm|yarn|bun|bunx|pytest|python\d?|py|node|deno|go|cargo|"
    r"mvn|gradle|dotnet|ruby|rspec|bundle|make|jest|vitest|mocha|phpunit|composer|"
    r"./gradlew|sh|bash)\b.*)$",
    re.I | re.M,
)
_HEADING_RE = re.compile(
    r"(?:Exact Command to Run Tests|Command to Run Tests|Run Tests|Test Command)\s*:?",
    re.I,
)


def extract_test_command(plan: str, sub: dict, stack_doc: str) -> str | None:
    """Find the command to verify the subtask (plan -> test_strategy -> stack)."""
    for source in (_plan_command_region(plan), plan, sub.get("test_strategy", "")):
        cmd = _first_command(source)
        if cmd:
            return cmd.strip()
    return _stack_test_command(stack_doc) or _first_command(stack_doc)


def _stack_test_command(stack_doc: str) -> str | None:
    """Prefer the explicitly-labeled ``test:`` command from stack.md."""
    m = re.search(r"^\s*[-*]?\s*test\s*:\s*`([^`]+)`", stack_doc or "", re.I | re.M)
    return m.group(1).strip() if m else None


def _plan_command_region(plan: str) -> str:
    m = _HEADING_RE.search(plan or "")
    if not m:
        return ""
    return plan[m.end() : m.end() + 600]


def _first_command(text: str) -> str | None:
    if not text:
        return None
    # 1) fenced code block
    fence = re.search(r"```(?:bash|sh|shell|console)?\s*\n(.+?)```", text, re.S)
    if fence:
        for line in fence.group(1).splitlines():
            line = line.strip().lstrip("$ ").strip()
            if line and not line.startswith("#"):
                return line
    # 2) inline backticked command
    for inline in re.findall(r"`([^`]+)`", text):
        if _RUNNER_RE.match(inline.strip()):
            return inline.strip()
    # 3) a bare line that begins with a known runner
    m = _RUNNER_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def _tail(text: str, limit: int = 4000) -> str:
    if not text:
        return "(empty)"
    return text if len(text) <= limit else "...\n" + text[-limit:]
