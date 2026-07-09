"""WORKER — the stateful TDD tool loop for one subtask (Step B).

The Worker receives the Manager's instructions plus TOOL_INSTRUCTIONS and
drives a conversation that persists across the implement/test/fix cycles of
the CURRENT subtask only — it is discarded the moment the subtask ends.

Protocol (enforced here, not trusted to the model):

* **One tool call per reply.** Extra calls in one reply are discarded with a
  warning; only the first executes.
* **Protocol corrections** — a reply with no parseable call and no ``DONE``
  gets a correction message; 3 strikes per drive turn = a failed attempt.
* **TDD enforcement** — for ``implement``/``integrate`` subtasks the first
  written file must be a test file (``test_*.py``, ``*_test.py``,
  ``*.test.ts``/``.js``, ``*.spec.ts``/``.js``, or anything under ``tests/`` /
  ``__tests__/``). An implementation write before that is REJECTED (not
  executed) with a hard correction; 3 strikes = a failed attempt when
  ``pipeline.tdd_hard_fail`` is true, otherwise a warning and the gate opens.
  ``scaffold``/``config``/``install`` subtasks skip the gate entirely.
* **Context safety** — every send goes through ``pack_conversation`` (pinned
  system prompt, newest-first); the fix loop additionally runs the destructive
  70%-of-max_tokens compression from ``context.conversation``.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

import promptlib
from config import WORKER
from context.conversation import compress_conversation, pack_conversation
from llm.tool_parser import ToolCall, extract_all_tool_calls
from server import events
from services import Services
from stages import roles as roles_stage
from tools.registry import TOOL_INSTRUCTIONS

_MAX_STEPS = 48          # hard cap on model turns per drive() call
_MAX_PROTOCOL_CORRECTIONS = 3
_MAX_TDD_CORRECTIONS = 3

_PROTOCOL_CORRECTION = (
    "Your last message did not contain a valid tool call. Respond with EXACTLY one "
    "agentic_call block in this format and nothing else:\n"
    '<agentic_call>{"tool": "write_file", "args": {"path": "...", "content": "...", '
    '"summary": "..."}}</agentic_call>\n'
    "Or, if you are finished with this step, reply with the single line: DONE"
)

_TDD_CORRECTION = (
    "STOP: you attempted to write an implementation file before any test file "
    "exists. This project is built test-first. Write the FAILING tests for this "
    "subtask now — a file matching test_*.py / *_test.py / *.test.ts / *.spec.ts "
    "or under tests/ — before any other file. The write you just attempted was "
    "NOT executed."
)

# Feature 1 (findings.md): asked once per failed verification, in the same
# conversation, so the Worker has full context of what it just attempted.
FINDINGS_PROMPT = (
    "In one sentence, summarize what went wrong and the key learning from this "
    "failure. Reply with PLAIN TEXT ONLY — do not emit a tool call for this."
)

# Non-negotiable rules injected into every Worker call. These encode failure
# modes seen with local models — chiefly imports that don't match the layout.
HARD_RULES = """\
# Non-negotiable rules
- TESTS FIRST: for implementation work, write the failing tests before the code
  under test. The orchestrator enforces this.
- EXPLORE before you write when anything is uncertain: run `ls -R` (or
  `find . -type f`) and read_file the files the instructions reference. Never
  invent the contents of a file you could have read.
- ACTUALLY INSTALL dependencies you rely on — writing requirements.txt (or
  package.json) installs nothing. Run the installer (e.g. run `pip install -r
  requirements.txt`) so imports resolve.
- Use valid, importable identifiers for package/module/directory names (Python
  packages use underscores, never hyphens). Every import MUST match the actual
  file layout you create; add __init__.py where the test runner needs it.
- Write COMPLETE file contents — never diffs, ellipses, or "... unchanged ..."
  placeholders. Tests must import the real modules and assert real behavior.
- To change an EXISTING file use patch_file with a unique anchor; do not
  rewrite the whole file to change a few lines.
- NEVER run a server/watcher in the foreground (uvicorn, npm run dev, …) — the
  sandbox rejects it because it would block forever. Start it with run
  {"background": true} (you get a session_id back while it keeps running),
  verify it with normal foreground commands (curl …), inspect its logs with
  check_session, and stop_session it when you are done — a leftover server can
  hold the port the final verification needs.
- Never make a check pass dishonestly: no `|| true`, no `2>/dev/null` to hide a
  real error, no deleting or weakening an assertion to go green.
- Never use git, absolute paths, `..`, or `~`."""

_TEST_FILE_RE = re.compile(
    r"(?:^|/)(?:test_[^/]+\.py|[^/]+_test\.py|[^/]+\.test\.(?:ts|js|tsx|jsx)|[^/]+\.spec\.(?:ts|js|tsx|jsx))$",
    re.IGNORECASE,
)


def is_test_path(rel_path: str) -> bool:
    """True when *rel_path* is a test file per the redesign's path patterns."""
    rel = (rel_path or "").replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    if not rel:
        return False
    parts = PurePosixPath(rel).parts
    if any(p in ("tests", "__tests__") for p in parts[:-1]):
        return True
    return bool(_TEST_FILE_RE.search(rel))


class WorkerSession:
    """One subtask's Worker conversation (implement -> fix -> fix …)."""

    def __init__(self, services: Services, subtask: dict, instructions: str):
        self.services = services
        self.subtask = subtask
        self.phase = "worker"
        self.subtask_id = str(subtask.get("id", ""))
        self.role = str(subtask.get("role") or "").strip().lower()
        self.tdd_enforced = str(subtask.get("type", "")).lower() in ("implement", "integrate")
        self.test_file_written = False
        self.tdd_corrections = 0
        self.files_touched: set[str] = set()
        self.last_call_ok = True  # Feature 5 (completion gate): last tool dispatch's ok flag
        self.attempt = 0  # test attempts, for tool-result bookkeeping
        self.dump_path = services.client.session_dump_path(WORKER, self.subtask_id)
        self.messages: list[dict] = [
            {"role": "system", "content": self._system_prompt(instructions)},
        ]
        services.bus.emit(
            events.WORKER_CALL_START,
            self.phase,
            subtask_id=self.subtask_id,
            tdd_enforced=self.tdd_enforced,
        )

    # ── public steps ──────────────────────────────────────────────────────────
    def implement(self) -> bool:
        """Drive the initial implement+tests turn. True if the turn completed."""
        instruction = promptlib.render(
            "worker",
            subtask_id=self.subtask_id,
            subtask_title=str(self.subtask.get("title", "")),
            subtask_type=str(self.subtask.get("type", "")),
            tdd=self.tdd_enforced,
        )
        return self._drive(instruction)

    def fix(self, *, cmd: str, exit_code: int, stdout: str, stderr: str, attempt: int, max_attempts: int) -> bool:
        """One fix turn against the latest failure (same conversation).

        Before every fix after the first, the conversation is compressed when
        it exceeds 70% of the Worker's resolved max_tokens (old tool results
        collapse to one-liners; the latest failure and system stay verbatim).
        """
        if attempt > 1:
            rmc = self.services.client.runtime_for(WORKER)
            collapsed = compress_conversation(self.messages, rmc.max_tokens)
            if collapsed:
                self.services.bus.log(
                    f"compressed worker conversation: {collapsed} old tool result(s) collapsed",
                    phase=self.phase,
                )
        instruction = promptlib.render(
            "fix",
            cmd=cmd,
            exit_code=exit_code,
            stdout=_tail(stdout),
            stderr=_tail(stderr),
            attempt=attempt,
            max_attempts=max_attempts,
        )
        return self._drive(instruction)

    def address_review(self, issues: list[str]) -> bool:
        """Feature 3 — inject the Manager-as-Reviewer's issues and drive one
        more turn (the single review-fix cycle; reuses the normal tool-call
        loop machinery)."""
        numbered = "\n".join(f"{i}. {issue}" for i, issue in enumerate(issues, start=1))
        instruction = (
            "Code review identified the following issues that must be addressed before "
            f"this subtask can be marked complete:\n{numbered}\n\nUse tool calls — one per "
            "message. When done, reply with the single line: DONE."
        )
        return self._drive(instruction)

    def address_completion_gate(self, failed_conditions: list[str]) -> bool:
        """Feature 5 — inject the gate's specific failures and drive one more
        turn (reuses the normal tool-call loop)."""
        desc = "; ".join(failed_conditions)
        instruction = (
            f"COMPLETION GATE FAILED: {desc}. Address this before this subtask can be "
            "marked complete.\n\nUse tool calls — one per message. When done, reply with "
            "the single line: DONE."
        )
        return self._drive(instruction)

    def summarize_failure(self) -> str:
        """Feature 1 — one plain-text turn (no tool call): a one-line failure
        summary for the shared ``.agent/findings.md`` log. Appended to the
        running conversation so the Worker has full context of the failure it
        just hit; does not consume the ``_drive()`` step/correction budgets."""
        svc = self.services
        self.messages.append({"role": "user", "content": FINDINGS_PROMPT})
        rmc = svc.client.runtime_for(WORKER)
        packed = pack_conversation(self.messages, rmc.max_tokens)
        result = svc.client.complete(WORKER, self.phase, packed, dump_path=self.dump_path)
        reply = (result.text or result.raw).strip()
        self.messages.append({"role": "assistant", "content": reply})
        if not reply or "<agentic_call" in reply.lower():
            return ""
        return reply

    # ── conversation driver ───────────────────────────────────────────────────
    def _drive(self, instruction: str) -> bool:
        svc = self.services
        self.messages.append({"role": "user", "content": instruction})
        corrections = 0

        for _ in range(_MAX_STEPS):
            svc.check_cancel()
            # Send-time packing: pinned system prompt, newest-first budget view.
            rmc = svc.client.runtime_for(WORKER)
            packed = pack_conversation(self.messages, rmc.max_tokens)
            result = svc.client.complete(WORKER, self.phase, packed, dump_path=self.dump_path)
            reply = result.text or result.raw
            self.messages.append({"role": "assistant", "content": reply})

            calls = extract_all_tool_calls(reply)
            done = says_done(reply)

            if not calls:
                if done:
                    return True
                corrections += 1
                if corrections > _MAX_PROTOCOL_CORRECTIONS:
                    svc.bus.error(
                        "worker: no parseable tool call after "
                        f"{_MAX_PROTOCOL_CORRECTIONS} protocol corrections",
                        context=reply[:600],
                        phase=self.phase,
                    )
                    return False
                self.messages.append({"role": "user", "content": _PROTOCOL_CORRECTION})
                continue

            # One tool call at a time: execute the first, discard the rest.
            if len(calls) > 1:
                svc.bus.log(
                    f"worker emitted {len(calls)} tool calls in one reply — executing the first, "
                    "discarding the rest",
                    phase=self.phase,
                    level="warn",
                )
            call = calls[0]

            if not call.is_known:
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Unknown tool {call.name!r}. Use only read_file, write_file, "
                            "patch_file, run, check_session, or stop_session."
                        ),
                    }
                )
                continue

            # TDD gate: reject implementation writes until a test file exists.
            if self._tdd_violation(call):
                self.tdd_corrections += 1
                if self.tdd_corrections > _MAX_TDD_CORRECTIONS:
                    if svc.config.pipeline.tdd_hard_fail:
                        svc.bus.error(
                            "worker: still writing implementation before tests after "
                            f"{_MAX_TDD_CORRECTIONS} TDD corrections — treating as a failed attempt",
                            phase=self.phase,
                        )
                        return False
                    svc.bus.log(
                        "TDD gate released after repeated corrections (pipeline.tdd_hard_fail=false)",
                        phase=self.phase,
                        level="warn",
                    )
                    self.test_file_written = True  # open the gate; proceed untested
                else:
                    self.messages.append({"role": "user", "content": _TDD_CORRECTION})
                    continue

            tr = svc.registry.dispatch(call, self.phase)
            self._track(call, tr)
            self.messages.append(
                {
                    "role": "user",
                    "content": tr.display,
                    "_kind": "tool_result",
                    "_attempt": self.attempt or 1,
                    "_exit": tr.payload.get("exit_code", 0 if tr.ok else 1),
                }
            )

        svc.bus.log("worker hit the max-steps cap for this turn", phase=self.phase, level="warn")
        return bool(self.files_touched)

    # ── bookkeeping ───────────────────────────────────────────────────────────
    def _tdd_violation(self, call: ToolCall) -> bool:
        if not self.tdd_enforced or self.test_file_written:
            return False
        if call.name not in ("write_file", "patch_file"):
            return False
        path = str((call.args or {}).get("path") or "")
        if is_test_path(path):
            return False
        # Config-ish artifacts (manifests, configs) may precede the first test.
        name = PurePosixPath(path.replace("\\", "/")).name.lower()
        if name in ("requirements.txt", "requirements-dev.txt", "package.json", "pyproject.toml",
                    "setup.py", "setup.cfg", "pytest.ini", "conftest.py", ".python-version",
                    "tsconfig.json", "vitest.config.ts", "jest.config.js", "__init__.py"):
            return False
        return True

    def _track(self, call: ToolCall, tr) -> None:
        self.last_call_ok = bool(tr.ok)  # Feature 5 (completion gate)
        if call.name in ("write_file", "patch_file") and tr.ok:
            rel = str(tr.payload.get("path") or (call.args or {}).get("path") or "")
            if rel:
                self.files_touched.add(rel)
                if is_test_path(rel):
                    self.test_file_written = True

    def _system_prompt(self, instructions: str) -> str:
        """Feature 2 — three layers, always in this order: the anchor (immutable,
        pinned by pack_conversation as the system message), the role-specific
        instructions for this subtask's role (falls back to backend.md, or is
        omitted entirely if no role files exist yet), then the Manager's
        handoff instructions for THIS subtask. The generic agent identity +
        tool protocol always comes last."""
        anchor = self.services.workspace.read_anchor_text() if self.services.workspace else ""
        role_text = roles_stage.read_role(self.services, self.role)
        parts: list[str] = []
        if anchor:
            parts.append(anchor)
        if role_text:
            parts.append(f"# Role: {self.role or roles_stage.DEFAULT_ROLE}\n\n{role_text}")
        parts.append(f"# Instructions for this subtask\n\n{instructions}")
        parts.append(
            "You are an expert coding agent in an autonomous build pipeline. You "
            "implement exactly one subtask by emitting tool calls, one per message. "
            "You write complete, working files that obey the rules below.\n\n"
            + HARD_RULES
            + "\n\n# Tool Protocol\n"
            + TOOL_INSTRUCTIONS
        )
        return "\n\n".join(parts)


def says_done(text: str) -> bool:
    """True if the model signalled completion with a standalone ``DONE`` line.

    A standalone line keeps the word "done" inside prose (or inside a written
    file, which never reaches this check anyway) from triggering early exit.
    """
    if not text:
        return False
    for ln in text.splitlines():
        if ln.strip().upper() == "DONE":
            return True
    return text.strip().upper() == "DONE"


def _tail(text: str, limit: int = 6000) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else "...\n" + text[-limit:]
