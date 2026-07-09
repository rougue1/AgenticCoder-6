"""Phase 2 — the subtask execution loop (redesign).

For each runnable subtask (all dependencies done):

STEP A  Manager handoff — anchor + subtask + doc summaries + decisions +
        dependency file summaries + manifest tree -> precise Worker
        instructions + a one-line decision note (``stages/manager.handoff``).
STEP B  Worker TDD loop — a stateful conversation (``stages/worker``) writes
        tests-first, then implementation; the ORCHESTRATOR runs the subtask's
        test command after each turn and walks the bounded ladder on failure:

            fix (same conversation)            x max_fix_retries
              -> escalate (new Manager plan,
                 fresh conversation)           x max_escalations
                -> decompose (2-4 micro-
                   subtasks, injected once)    x max_decompositions
                  -> hard block (+ cascade to dependents)

        Every failed verification also gets a one-line Worker-written entry in
        the shared ``.agent/findings.md`` log (Feature 1). Once tests pass,
        implement/integrate subtasks get one Manager-as-Reviewer code-review
        cycle (Feature 3, at most once per subtask) and then the completion
        gate (Feature 5) before the subtask is marked done; either failing
        feeds back into the SAME fix-retry budget as a test failure, and
        exhausting it escalates exactly like a test failure would.

STEP C  Post-task summarization — every file the subtask touched gets a
        Manager-as-Analyst summary; install subtasks get a silent dependency
        conflict check; every test run lands in ``test_results.jsonl``
        (retries ``is_final=false``, the terminal run ``is_final=true``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from orchestrator import completion_gate
from orchestrator.states import SubtaskOutcome
from server import events
from services import Services
from stages import code_review, manager, summarizer
from stages.worker import WorkerSession
from taskstore import BLOCKED, DECOMPOSED, DONE, IN_PROGRESS, TaskStore

# Dependency manifests we know how to install from when an install-type subtask
# ships no test command. Doubles as the environment-population step.
def _npm_install(path: str) -> str:
    d = path.rsplit("/", 1)[0] if "/" in path else ""
    return f"npm --prefix {d} install" if d else "npm install"


_MANIFEST_INSTALL = [
    ("requirements.txt", lambda p: f"pip install -r {p}"),
    ("requirements-dev.txt", lambda p: f"pip install -r {p}"),
    ("package.json", _npm_install),
]

# Subtask types under TDD enforcement (pytest exit 5 = "no tests collected" is a
# FAILURE for these — a green empty suite would silently defeat tests-first).
_TDD_TYPES = ("implement", "integrate")

# Feature 3: subtask types worth a code-review cycle — scaffold/config/install
# don't produce logic code worth reviewing (same set as _TDD_TYPES today, kept
# as its own name since the two concerns are independent).
_REVIEW_TYPES = ("implement", "integrate")

_SENTENCE_RE = re.compile(r"^(.*?[.!?])(?:\s|$)")


def _first_sentence(text: str) -> str:
    """Feature 1 — collapse to one line and keep only the first sentence."""
    flat = " ".join((text or "").split())
    if not flat:
        return ""
    m = _SENTENCE_RE.match(flat)
    return (m.group(1) if m else flat).strip()


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
class VerifyResult:
    cmd: str
    passed: bool
    stdout: str
    stderr: str
    exit_code: int
    duration: float = 0.0


@dataclass
class LoopResult:
    done: int = 0
    blocked: int = 0
    decomposed: int = 0
    blocked_ids: list[str] = field(default_factory=list)


class SubtaskLoop:
    def __init__(self, services: Services):
        self.services = services
        self.cfg = services.config.pipeline
        self.bus = services.bus

    def run(self) -> LoopResult:
        """Process every runnable subtask until none remain."""
        result = LoopResult()
        while True:
            self.services.check_cancel()
            store = TaskStore.load(self.services.workspace)
            picked = store.next_runnable()
            if picked is None:
                break
            task, sub = picked
            outcome = self._process_subtask(store, task, sub)
            if outcome == SubtaskOutcome.DONE:
                result.done += 1
            elif outcome == SubtaskOutcome.DECOMPOSED:
                result.decomposed += 1
            elif outcome == SubtaskOutcome.BLOCKED:
                result.blocked += 1
                result.blocked_ids.append(sub["id"])
        self.services.progress.clear()
        return result

    # ── one subtask through the ladder ────────────────────────────────────────
    def _process_subtask(self, store: TaskStore, task: dict, sub: dict) -> SubtaskOutcome:
        """Run one subtask through the ladder. Whatever the outcome (done,
        blocked, decomposed, cancel/crash), every background session the
        Worker started is terminated on the way out — no server survives the
        subtask that started it."""
        try:
            return self._process_subtask_inner(store, task, sub)
        finally:
            self._terminate_sessions("subtask boundary")

    def _terminate_sessions(self, reason: str) -> None:
        sandbox = self.services.sandbox
        if sandbox is None:
            return
        try:
            sandbox.terminate_sessions(reason)
        except Exception as exc:  # cleanup must never mask the subtask outcome
            self.bus.log(f"session cleanup failed: {exc}", phase="subtask_loop", level="warn")

    def _process_subtask_inner(self, store: TaskStore, task: dict, sub: dict) -> SubtaskOutcome:
        sid = sub["id"]
        store.set_status(sid, IN_PROGRESS)
        self.services.progress.begin_subtask(sub)
        self.bus.emit(events.TASK_START, "subtask_loop", id=sid, title=sub.get("title", ""), type=sub.get("type", ""))

        failures: list[FailureRecord] = []
        escalations_left = self.cfg.max_escalations
        attempt = 0
        # Feature 3: one review cycle for the WHOLE subtask (never reset by
        # escalation) — the explicit anti-infinite-review-loop guarantee.
        review_done = False

        # STEP A — Manager handoff.
        self.services.progress.activity = "handoff"
        instructions = manager.handoff(self.services, sub, store)

        while True:  # escalation loop (each iteration = a fresh Worker conversation)
            self.services.check_cancel()
            self.services.progress.activity = "implement"
            session = WorkerSession(self.services, sub, instructions)
            session.attempt = attempt + 1
            session.implement()

            fix_left = self.cfg.max_fix_retries
            while True:  # test / fix / review / gate loop (one Worker conversation)
                self.services.check_cancel()
                attempt += 1
                session.attempt = attempt
                self.services.progress.activity = "test"
                verify = self._verify(sub)

                # Once tests pass: one code-review cycle (Feature 3, at most
                # once per subtask), then the completion gate (Feature 5).
                # Both are skipped (post_pass_ok stays True) once review is
                # already done / not applicable, or the gate already ran this
                # iteration and passed.
                review_result = None
                gate_result = None
                post_pass_ok = True
                if verify.passed:
                    if not review_done and str(sub.get("type") or "").lower() in _REVIEW_TYPES:
                        review_done = True
                        review_result = self._run_review(sub, session)
                        post_pass_ok = review_result.approved
                    if post_pass_ok:
                        gate_result = completion_gate.check(self.services, sub, session)
                        self._emit_completion_check(sid, gate_result)
                        post_pass_ok = gate_result.passed

                escalation_round = self.cfg.max_escalations - escalations_left
                # Final = passed with review+gate both clear, or the last rung
                # of the ladder (decomposition/block both end THIS subtask's
                # test history).
                is_final = (verify.passed and post_pass_ok) or (fix_left <= 0 and escalations_left <= 0)
                self._log_test_result(sid, verify, attempt, is_final=is_final, escalation=escalation_round)
                self.bus.test_run(
                    "subtask_loop",
                    verify.cmd or "(no test command)",
                    verify.exit_code,
                    verify.passed,
                    output=_tail(((verify.stderr or "") + ("\n" + verify.stdout if verify.stdout else "")).strip(), 1500),
                )

                if verify.passed and post_pass_ok:
                    self._finish_done(store, sub, session)
                    return SubtaskOutcome.DONE

                if verify.passed and not post_pass_ok:
                    # Tests passed but the review and/or completion gate
                    # didn't: address it and re-verify, spending the SAME
                    # fix-retry budget a test failure would.
                    if fix_left > 0:
                        fix_left -= 1
                        fix_no = self.cfg.max_fix_retries - fix_left
                        self.bus.emit(
                            events.WORKER_FIX_ATTEMPT,
                            "subtask_loop",
                            id=sid,
                            attempt=fix_no,
                            exit_code=verify.exit_code,
                        )
                        self.services.progress.activity = "fix"
                        if review_result is not None and not review_result.approved:
                            session.address_review(review_result.issues)
                        elif gate_result is not None and not gate_result.passed:
                            session.address_completion_gate(gate_result.failed_conditions)
                        continue
                    # Fix budget exhausted: same fate as an exhausted test
                    # failure — fall through to escalate/decompose/block,
                    # with a synthetic failure record so escalation has
                    # something concrete to diagnose.
                    failed_list = (
                        (review_result.issues if review_result else [])
                        + (gate_result.failed_conditions if gate_result else [])
                    )
                    failures.append(
                        FailureRecord(
                            attempt=len(failures) + 1,
                            cmd=verify.cmd or "(post-pass checks)",
                            exit_code=1,
                            stdout="",
                            stderr="Post-pass checks failed:\n" + "\n".join(failed_list),
                            note="Tests passed but code review/completion gate failed and fix retries are exhausted.",
                        )
                    )
                    break  # -> escalate / decompose / block

                # verify.passed is False: the ordinary test-failure path.
                failures.append(
                    FailureRecord(
                        attempt=len(failures) + 1,
                        cmd=verify.cmd or "(no test command found)",
                        exit_code=verify.exit_code,
                        stdout=verify.stdout,
                        stderr=verify.stderr,
                        note="" if verify.cmd else "Subtask declared no runnable test command.",
                    )
                )
                self._record_finding(sid, session)

                if fix_left > 0:
                    fix_left -= 1
                    fix_no = self.cfg.max_fix_retries - fix_left
                    self.bus.emit(
                        events.WORKER_FIX_ATTEMPT,
                        "subtask_loop",
                        id=sid,
                        attempt=fix_no,
                        exit_code=verify.exit_code,
                    )
                    self.services.progress.activity = "fix"
                    session.fix(
                        cmd=verify.cmd or "(no test command)",
                        exit_code=verify.exit_code,
                        stdout=verify.stdout,
                        stderr=verify.stderr,
                        attempt=fix_no,
                        max_attempts=self.cfg.max_fix_retries,
                    )
                    continue
                break  # fixes exhausted -> escalate / decompose / block

            history = "\n\n".join(f.render() for f in failures)

            if escalations_left > 0:  # ESCALATE — discard the conversation, re-plan
                escalations_left -= 1
                self.services.progress.activity = "escalate"
                self.bus.emit(
                    events.TASK_ESCALATED,
                    "subtask_loop",
                    id=sid,
                    escalations_left=escalations_left,
                    failures=len(failures),
                )
                # The fresh conversation knows nothing about the old attempt's
                # sessions; kill them so stale servers can't hold ports or serve
                # stale code against the new plan.
                self._terminate_sessions("escalation — fresh plan")
                instructions = manager.escalate(self.services, sub, history, store)
                continue

            # DECOMPOSE — once, if this subtask still may.
            decomposition_errors: list[str] | None = None
            if sub.get("can_decompose", True) and not sub.get("is_decomposed", False) and self.cfg.max_decompositions > 0:
                self.services.progress.activity = "decompose"
                micro = manager.decompose(self.services, sub, history, store)
                if micro:
                    decomposition_errors = store.inject_decomposed(sid, micro)
                else:
                    decomposition_errors = ["the Manager produced no parseable micro-subtasks"]
                self._record_decomposition(sub, history, micro, decomposition_errors)
                if not decomposition_errors:
                    self.bus.emit(events.TASK_DECOMPOSED, "subtask_loop", id=sid, micro_count=len(micro))
                    summarizer_count = self._summarize_session(sub, session)
                    self.bus.log(
                        f"{sid} decomposed into {len(micro)} micro-subtask(s); "
                        f"{summarizer_count} file summar(ies) refreshed",
                        phase="subtask_loop",
                    )
                    return SubtaskOutcome.DECOMPOSED
                self.bus.log(
                    f"decomposition of {sid} failed validation: {'; '.join(decomposition_errors[:5])}",
                    phase="subtask_loop",
                    level="warn",
                )

            self._finish_blocked(store, task, sub, failures, session)  # HARD BLOCK
            return SubtaskOutcome.BLOCKED

    # ── verification (the orchestrator runs tests — never the model) ──────────
    def _verify(self, sub: dict) -> VerifyResult:
        cmd = str(sub.get("test_command") or "").strip()
        stype = str(sub.get("type") or "").lower()
        if cmd:
            res = self.services.sandbox.run(cmd)
            passed = res.ok
            # pytest exit 5 = "no tests collected": acceptable for scaffold-ish
            # types, a FAILURE for TDD-enforced types (an empty suite must not
            # green an implementation subtask).
            if not passed and res.exit_code == 5 and "pytest" in cmd and stype not in _TDD_TYPES:
                passed = True
                self.bus.log("pytest collected no tests (exit 5) — empty suite passes for this subtask type", phase="subtask_loop")
            return VerifyResult(cmd, passed, res.stdout, res.stderr, res.exit_code, res.duration)
        return self._fallback_verification(sub)

    def _fallback_verification(self, sub: dict) -> VerifyResult:
        """Verify a subtask that declared no test command (scaffold/config/install).

        1. install-type with a dependency manifest -> prove the dependencies
           INSTALL (this also populates the environment for later subtasks).
        2. otherwise accept iff every declared file exists non-empty.
        """
        install = self._manifest_install_command(sub)
        if install and str(sub.get("type") or "").lower() == "install":
            res = self.services.sandbox.run(install)
            if res.ok:
                self.bus.log(f"no test command; verified dependencies install via `{install}`", phase="subtask_loop")
            return VerifyResult(install, res.ok, res.stdout, res.stderr, res.exit_code, res.duration)

        files = [str(f).strip() for f in (sub.get("files") or []) if str(f).strip()]
        missing = [f for f in files if not self._file_present(f)]
        if files and not missing:
            self.bus.log(f"no test command; accepted on file presence ({len(files)} file(s))", phase="subtask_loop")
            return VerifyResult("(declared files present)", True, "", "", 0)
        msg = (
            "No runnable test command for this subtask and it produced no verifiable "
            f"output. Missing or empty files: {', '.join(missing) or '(none declared)'}."
        )
        return VerifyResult("", False, "", msg, 1)

    def _manifest_install_command(self, sub: dict) -> str | None:
        declared = [str(f).strip().replace("\\", "/") for f in (sub.get("files") or [])]
        for name, make_cmd in _MANIFEST_INSTALL:
            for d in declared:
                if (d == name or d.endswith("/" + name)) and self._file_present(d):
                    return make_cmd(d)
        if self._file_present("requirements.txt"):
            return "pip install -r requirements.txt"
        return None

    def _file_present(self, rel: str) -> bool:
        try:
            p = self.services.workspace.resolve_in_root(rel)
        except Exception:
            return False
        try:
            return p.is_file() and p.stat().st_size > 0
        except OSError:
            return False

    # ── test_results.jsonl ─────────────────────────────────────────────────────
    def _log_test_result(self, sid: str, verify: VerifyResult, attempt: int, *, is_final: bool, escalation: int) -> None:
        record = {
            "subtask_id": sid,
            "cmd": verify.cmd,
            "exit_code": verify.exit_code,
            "duration": round(verify.duration, 3),
            "attempt": attempt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_final": is_final,
            "passed": verify.passed,
            "escalation": escalation,
        }
        self.services.workspace.append_agent_doc("test_results.jsonl", json.dumps(record) + "\n")

    # ── outcomes ──────────────────────────────────────────────────────────────
    def _finish_done(self, store: TaskStore, sub: dict, session: WorkerSession) -> None:
        sid = sub["id"]
        store.set_status(sid, DONE)
        n = self._summarize_session(sub, session)
        if str(sub.get("type") or "").lower() == "install":
            summarizer.post_install_check(self.services)
        self.bus.emit(events.TASK_DONE, "subtask_loop", id=sid, files_summarized=n)

    def _finish_blocked(
        self,
        store: TaskStore,
        task: dict,
        sub: dict,
        failures: list[FailureRecord],
        session: WorkerSession,
    ) -> None:
        sid = sub["id"]
        store.set_status(sid, BLOCKED)
        cascaded = store.cascade_block(sid)
        last = failures[-1] if failures else None
        record = (
            f"## {sid} — {sub.get('title', '')}\n"
            f"- Parent task: {task.get('id')} {task.get('title', '')}\n"
            f"- Type: {sub.get('type', '')}\n"
            f"- Intent: {sub.get('intent', '')}\n"
            f"- Attempts: {len(failures)} (fixes + escalations exhausted)\n"
            f"- Last command: `{last.cmd if last else 'n/a'}` (exit {last.exit_code if last else 'n/a'})\n"
            f"- Last error:\n```\n{_tail(last.stderr if last else '', 1500)}\n```\n"
            f"- Dependents blocked with it: {', '.join(cascaded) or '(none)'}\n"
            f"- Recorded: {datetime.now(timezone.utc).isoformat()}\n"
        )
        existing = self.services.workspace.read_agent_doc("blocked.md") or "# Blocked Subtasks\n\n"
        self.services.workspace.write_agent_doc("blocked.md", existing.rstrip() + "\n\n" + record)
        self._summarize_session(sub, session)
        self.bus.emit(events.TASK_BLOCKED, "subtask_loop", id=sid, attempts=len(failures), cascaded=cascaded)

    # ── Feature 3: per-subtask code review ────────────────────────────────────
    def _run_review(self, sub: dict, session: WorkerSession) -> code_review.ReviewResult:
        sid = sub["id"]
        self.services.progress.activity = "review"
        self.bus.emit(events.TASK_REVIEW_START, "subtask_loop", id=sid)
        result = code_review.run(self.services, sub, sorted(session.files_touched))
        self.bus.emit(
            events.TASK_REVIEW_COMPLETE,
            "subtask_loop",
            id=sid,
            status="approved" if result.approved else "issues_found",
            count=len(result.issues),
        )
        self.bus.log(
            f"review approved for {sid}"
            if result.approved
            else f"review: {len(result.issues)} issues found for {sid}, worker addressing.",
            phase="subtask_loop",
        )
        return result

    # ── Feature 5: completion gate ────────────────────────────────────────────
    def _emit_completion_check(self, sid: str, gate_result: completion_gate.GateResult) -> None:
        self.bus.emit(
            events.TASK_COMPLETION_CHECK,
            "subtask_loop",
            id=sid,
            status="passed" if gate_result.passed else "failed",
            failed_conditions=gate_result.failed_conditions,
        )
        if not gate_result.passed:
            self.bus.log(
                f"completion gate: {len(gate_result.failed_conditions)} issue(s) for {sid}, worker addressing.",
                phase="subtask_loop",
            )

    # ── Feature 1: findings.md ────────────────────────────────────────────────
    def _record_finding(self, sid: str, session: WorkerSession) -> None:
        """One Worker turn summarizing the failure, appended to the shared
        error-persistence log. Never allowed to fail the subtask."""
        try:
            summary = session.summarize_failure()
        except Exception as exc:
            self.bus.log(f"findings summary failed: {exc}", phase="subtask_loop", level="warn")
            return
        sentence = _first_sentence(summary)
        if not sentence:
            return
        self.services.workspace.append_agent_doc("findings.md", f"[{sid}] {sentence}\n")
        self.bus.emit(events.FINDINGS_ENTRY_ADDED, "subtask_loop", id=sid, summary=sentence)

    def _summarize_session(self, sub: dict, session: WorkerSession) -> int:
        """STEP C — Analyst summaries for every file this subtask touched."""
        self.services.progress.activity = "summarize"
        try:
            return summarizer.summarize_files(self.services, sub["id"], sorted(session.files_touched))
        except Exception as exc:  # a summary failure must not fail the subtask
            self.bus.log(f"post-task summarization failed: {exc}", phase="summarizer", level="warn")
            return 0

    # ── decomposition record ──────────────────────────────────────────────────
    def _record_decomposition(
        self, sub: dict, history: str, micro: list[dict], errors: list[str] | None
    ) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        sid = str(sub.get("id", "unknown"))
        body = (
            f"# Decomposition — {sid}\n\n"
            f"- When: {datetime.now(timezone.utc).isoformat()}\n"
            f"- Outcome: {'INJECTED' if not errors else 'REJECTED: ' + '; '.join(errors)}\n\n"
            f"## Original subtask\n\n```json\n{json.dumps(sub, indent=2)}\n```\n\n"
            f"## Proposed micro-subtasks\n\n```json\n{json.dumps(micro, indent=2)}\n```\n\n"
            f"## Failure history that triggered it\n\n{history}\n"
        )
        path = self.services.workspace.decompositions_dir / f"{ts}_{sid}.md"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        except OSError as exc:
            self.bus.log(f"could not write decomposition record: {exc}", phase="subtask_loop", level="warn")


def _tail(text: str, limit: int = 4000) -> str:
    if not text:
        return "(empty)"
    return text if len(text) <= limit else "...\n" + text[-limit:]
