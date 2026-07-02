"""IMPLEMENTER (spec §9 IMPLEMENT / WRITE_TESTS / FIX).

Owns the *ephemeral conversation* for one subtask. The small coding model emits
tool calls (write_file / read_file / run); this class executes them via the
registry, appends results, and keeps going until the model signals ``DONE``. The
conversation accumulates so the model never loses what it just wrote — and it is
discarded the moment the subtask passes (the durable outputs are the files on
disk + the manifest/task status).

When the model emits an unparseable tool call, the orchestrator re-asks using the
fast ``tool_caller`` model to salvage a well-formed call (spec §11).
"""

from __future__ import annotations

import promptlib
from context.conversation import pack_conversation
from llm.tool_parser import ToolCall, extract_all_tool_calls
from llm.tool_router import looks_like_tool_content, salvage_calls
from services import Services
from tools.registry import TOOL_INSTRUCTIONS

# Caps that prevent an unbounded ephemeral loop within a single turn.
_MAX_STEPS = 48        # max model turns per drive() turn
_MAX_REASKS = 3        # malformed-call re-asks before the turn fails (spec §11)
_MAX_CALLS_PER_REPLY = 16  # execute at most this many tool calls from one reply

_REASK_HINT = (
    "Your last message did not contain a valid tool call. Respond with EXACTLY one "
    "tool call in this format and nothing else:\n"
    '<tool_call>{"tool": "write_file", "args": {"path": "...", "content": "...", '
    '"summary": "..."}}</tool_call>\n'
    "Or, if you are finished, reply with the single line: DONE"
)

# Non-negotiable rules injected into every implement/fix call, independent of the
# generated steering.md. These encode failure modes seen with local models —
# chiefly hyphenated package dirs and imports that don't match the file layout.
HARD_RULES = """\
# Non-negotiable rules
- EXPLORE before you write when anything is uncertain: if you are not sure of the
  current project layout or what an existing file already contains, first run
  `ls -R` (or `find . -type f`) and read_file the files the plan references. Never
  invent the contents of a file you could have read.
- ACTUALLY INSTALL dependencies you rely on — writing a manifest does not install
  anything. After creating/updating requirements.txt (or package.json), run the
  install (e.g. run `pip install -r requirements.txt`) so imports resolve and tests
  can import the real packages.
- Use valid, importable identifiers for package/module/directory names: no hyphens
  or spaces where the language forbids them (e.g. Python packages use underscores,
  never hyphens). The project FOLDER may contain hyphens, but import paths must not.
- Every import/reference MUST match the actual file layout you create. If a test
  imports `pkg.mod`, put the file at `pkg/mod.py` and make `pkg` importable
  (add `__init__.py` where the language/test-runner needs it).
- Keep file paths consistent across the whole subtask: source and its tests must
  agree on the same module path.
- Write COMPLETE file contents — never diffs, ellipses, or "... unchanged ..."
  placeholders. Tests must import the real modules and assert real behavior.
- To change a file that ALREADY exists, use edit_file with a unique anchor; do not
  rewrite the whole file to change a few lines (a full rewrite is a fresh chance to
  inject a typo that dead-stops the file).
- For a LARGE new file (more than a few hundred lines), build it with `run` + bash
  heredoc appends from the FIRST call (`cat > path <<'EOF' … EOF`, then `cat >>` per
  part) — a single huge write_file body can be truncated mid-stream. Verify with
  `wc -c path`. If a write comes back flagged as possibly truncated, do NOT retry it
  whole; rebuild with heredoc appends.
- Never make a check pass dishonestly: no `|| true`, no `2>/dev/null` to hide a real
  error, no deleting or weakening an assertion to get green. Fix the real cause — a
  test that cannot fail proves nothing.
- Never use git, absolute paths, `..`, or `~`."""

# ── soft-nudge backstops (spec adoption #6) ───────────────────────────────────────
# Deterministic, in-loop safety nets for failure modes the hard fix/escalate ladder
# (which governs the OUTER loop) can't see WITHIN a single drive turn. Each is a
# nudge — an injected note that changes the model's approach — never a hard yield;
# the loop stays bounded by _MAX_STEPS regardless. Ported from codehamr's model.go.

# Prefix on every injected note. A weak (small) model reads a bare mid-conversation
# instruction as a fresh human turn ("the user gave me nothing new, I'll stop"); naming
# it as the orchestrator's own automated check keeps it oriented.
NUDGE_ORIGIN = "[Automated check — not a message from your user.] "

# Same tool + same target failing the same way this many times running -> nudge to
# change approach. Generous so honest trial-and-error on a hard edit isn't interrupted.
_MAX_TOOL_FAIL_STREAK = 4
# Tool calls in one drive turn before a one-time runaway self-check (below _MAX_STEPS
# so the model gets a chance to self-correct before hitting the hard cap).
_MAX_TOOL_ROUNDS = 36

_RUNAWAY_NUDGE = (
    "You've made many tool calls this step without finishing. If you're still making "
    "real progress, continue. If you're repeating something that can't work here — the "
    "same command or edit failing the same way — stop chasing it (that loop burns the "
    "turn) and either fix the root cause differently or say what's blocking you, then "
    "finish by writing the plan's files and tests."
)


def _failure_nudge(n: int) -> str:
    return (
        f"The last {n} tool calls to the same target failed the same way. Stop repeating "
        "it — read the error and change your approach (a different anchor, a different "
        "command, or read_file the current contents first), or say what's blocking you."
    )


def _tool_target_key(call: ToolCall) -> str:
    """Stable identity for repeated-failure detection: tool + its target (path for
    file tools, the command's first line for run). NOT the full args, so a cosmetic
    change between retries (a reworded command, a regenerated body) can't defeat it."""
    if call.name in ("write_file", "edit_file", "read_file"):
        return f"{call.name}|{(call.args or {}).get('path', '')}"
    if call.name == "run":
        cmd = str((call.args or {}).get("cmd") or (call.args or {}).get("command") or "")
        first = cmd.splitlines()[0].strip() if cmd else ""
        return f"run|{first}"
    return call.name


def _looks_like_stranded_tool_call(text: str) -> bool:
    """True when a reply LOOKS like it meant to call a tool but wouldn't parse — a
    ``<tool_call>`` opener or bare tool JSON the parser couldn't extract. Lets the
    re-ask-exhausted diagnostic distinguish 'the model's tool formatting is broken'
    from 'the model just replied in prose' (spec adoption #8)."""
    if not text:
        return False
    low = text.lower()
    return "<tool_call>" in low or '"tool"' in low or '"tool_name"' in low


class Implementer:
    """Drives the implement/test/fix conversation for one subtask."""

    def __init__(self, services: Services, task: dict, subtask: dict, plan: str):
        self.services = services
        self.task = task
        self.subtask = subtask
        self.plan = plan
        self.phase = "implementer"
        self.files_written = 0
        # Repeated-failure tracking (spec adoption #6). Instance state, so a model
        # hammering the same broken target is caught even across implement -> fix calls
        # within this subtask's ephemeral conversation.
        self._fail_key = ""
        self._fail_streak = 0
        self.messages: list[dict] = [
            {"role": "system", "content": self._system_prompt()},
        ]

    # ── public steps ──────────────────────────────────────────────────────────
    def implement_and_write_tests(self) -> bool:
        """Run the IMPLEMENT + WRITE_TESTS step. Returns True if it produced work."""
        instruction = promptlib.render(
            "implementer",
            subtask_id=self.subtask.get("id", ""),
            subtask_title=self.subtask.get("title", ""),
            plan=self.plan,
        )
        return self._drive(instruction)

    def fix(self, *, cmd: str, exit_code: int, stdout: str, stderr: str, attempt: int, max_attempts: int) -> bool:
        """Run a FIX step against the latest failure (same conversation)."""
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

    # ── conversation driver ───────────────────────────────────────────────────
    def _drive(self, instruction: str) -> bool:
        svc = self.services
        self.messages.append({"role": "user", "content": instruction})
        reasks = 0
        nudged = False
        tool_rounds = 0
        runaway_nudged = False

        for _ in range(_MAX_STEPS):
            svc.check_cancel()
            # Pack newest-first to the phase budget before sending. self.messages
            # is the durable record and only grows; sending the full list would
            # eventually overrun num_ctx and make Ollama silently drop the system
            # prompt (steering + tool protocol + plan). The packed copy keeps the
            # system message pinned and the most recent rounds verbatim.
            packed = pack_conversation(self.messages, svc.config.usable_budget_for(self.phase))
            result = svc.client.complete(self.phase, packed)
            reply = result.text or result.raw
            self.messages.append({"role": "assistant", "content": reply})

            # Parse EVERY well-formed tool call in the reply — a strong model often
            # emits several files in one message; run them all rather than forcing a
            # round-trip per file (this also helps the turn converge instead of
            # churning). If nothing parsed but the reply still LOOKS like it tried to
            # act (narrated edits, fenced file dumps, malformed tool JSON), route the
            # whole thing to the tool_caller model to recover real calls (#8) so the
            # work lands on disk instead of being lost.
            calls = extract_all_tool_calls(reply, limit=_MAX_CALLS_PER_REPLY)
            done = says_done(reply)
            if not calls and not done and looks_like_tool_content(reply):
                calls = salvage_calls(svc.client, reply, max_calls=_MAX_CALLS_PER_REPLY)
                if calls:
                    svc.bus.log(
                        f"recovered {len(calls)} tool call(s) from mis-formatted output via tool_caller",
                        phase=self.phase,
                    )

            if not calls:
                if done:
                    if self.files_written == 0 and not nudged:
                        nudged = True
                        self.messages.append(
                            {"role": "user", "content": NUDGE_ORIGIN + "You said DONE but wrote no files. Implement the plan now using write_file or edit_file (or run for installs / heredoc builds)."}
                        )
                        continue
                    return True
                reasks += 1
                if reasks > _MAX_REASKS:
                    if _looks_like_stranded_tool_call(reply):
                        svc.bus.error(
                            "implementer: the model kept emitting tool-call-like text that would not "
                            "parse — its tool-call formatting is malformed (a stranded <tool_call>/JSON). "
                            "Check the model's tool-call output format.",
                            context=reply[:600],
                            phase=self.phase,
                        )
                    else:
                        svc.bus.error(
                            "implementer: no parseable tool call after re-asks (the model replied in "
                            "prose instead of emitting a tool call).",
                            context=reply[:600],
                            phase=self.phase,
                        )
                    return False
                self.messages.append({"role": "user", "content": _REASK_HINT})
                continue

            # Execute each recovered/parsed call in document order. Completion is only
            # recognized on a separate no-call message, so a file whose content
            # contains a "DONE" line can't end the turn early.
            for call in calls:
                if not call.is_known:
                    self.messages.append(
                        {"role": "user", "content": f"Unknown tool {call.name!r}. Use only read_file, write_file, edit_file, or run."}
                    )
                    continue
                tr = svc.registry.dispatch(call, self.phase)
                if call.name in ("write_file", "edit_file") and tr.ok:
                    self.files_written += 1
                self._record_tool_outcome(call, tr)
                tool_rounds += 1
                self.messages.append({"role": "user", "content": tr.display})

                # Soft backstops (spec adoption #6): catch a runaway loop or a target
                # that keeps failing the same way. Injected as user notes (the
                # conversation's nudge channel); the loop stays capped by _MAX_STEPS.
                if not runaway_nudged and tool_rounds >= _MAX_TOOL_ROUNDS:
                    runaway_nudged = True
                    svc.bus.log(f"runaway-iteration nudge at {tool_rounds} tool calls", phase=self.phase, level="warn")
                    self.messages.append({"role": "user", "content": NUDGE_ORIGIN + _RUNAWAY_NUDGE})
                elif self._fail_streak >= _MAX_TOOL_FAIL_STREAK:
                    svc.bus.log(f"repeated-failure nudge after {self._fail_streak} same-target failures", phase=self.phase, level="warn")
                    self.messages.append({"role": "user", "content": NUDGE_ORIGIN + _failure_nudge(self._fail_streak)})
                    self._fail_key, self._fail_streak = "", 0

        svc.bus.log("implementer hit max steps for this turn", phase=self.phase, level="warn")
        return self.files_written > 0

    def _record_tool_outcome(self, call: ToolCall, tr) -> None:
        """Update the same-target failure streak from one tool result. ToolResult.ok
        is authoritative (no display-string parsing needed): a non-zero `run`, a failed
        edit/write, or a missing read all count as failures; any success resets."""
        key = _tool_target_key(call)
        if tr.ok:
            self._fail_key, self._fail_streak = "", 0
            return
        if key == self._fail_key and self._fail_key:
            self._fail_streak += 1
        else:
            self._fail_key, self._fail_streak = key, 1

    # ── helpers ───────────────────────────────────────────────────────────────
    def _system_prompt(self) -> str:
        steering = self.services.loader.doc("steering.md")
        parts = [
            "You are an expert coding agent in an autonomous build pipeline. You "
            "implement exactly one subtask by emitting tool calls. You write "
            "complete, working files that obey the rules below.",
            "",
            HARD_RULES,
            "",
            "# Steering (project conventions)",
            steering or "(no steering doc available)",
            "",
            "# Tool Protocol",
            TOOL_INSTRUCTIONS,
        ]
        return "\n".join(parts)


def says_done(text: str) -> bool:
    """True if the model signalled completion with a standalone ``DONE`` line.

    Matches DONE whether it is the sole content, the last line (implementer), or
    the first line followed by a summary (reviewer). A standalone line keeps the
    word "done" appearing inside prose from triggering a false positive.
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
