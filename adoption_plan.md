# Adoption Plan — porting codehamr's hard-won local-LLM lessons into agentic-coder

> Status: **planning only**. Nothing in here is implemented. Awaiting approval before any code changes.

This document is the output of three passes:

1. A full read of `agentic-coder/` (Python pipeline + React IDE + prompts + config).
2. A full read of `codehamr/` (Go terminal coding agent + embedded system prompt).
3. A cross-reference identifying what is genuinely worth adopting, given that the
   two tools have **different shapes** and not everything transfers.

---

## 1. The two codebases at a glance

### agentic-coder — "prompt → working app", autonomous, phased

A Python state machine that turns one natural-language prompt into a built, tested
app on disk using **local models via Ollama/LiteLLM**.

- **Pipeline (stateless across steps):** intake → requirements → stack → architect →
  SDD+steering → task planning → **subtask loop** → review. Each phase has its own
  configurable model and token budget (`config.yaml`, `context_budget.yaml`).
- **Two-level context model:** the pipeline rebuilds context from disk for every new
  subtask; a *single* subtask runs as one accumulating ephemeral conversation
  (`stages/implementer.py`) that is discarded the moment its tests pass.
- **Subtask loop (`orchestrator/subtask_loop.py`) — the heart:** PLAN (large model) →
  IMPLEMENT + WRITE_TESTS (small model, ephemeral convo) → RUN (orchestrator executes) →
  **hard bounded ladder**: FIX ×`max_fix_retries` → ESCALATE ×`max_escalations`
  (large model re-plans with full failure history) → BLOCK.
- **Tools (`tools/registry.py`):** `read_file`, `write_file` (whole-file only), `run`.
  The model emits **text-based `<tool_call>` JSON**, parsed tolerantly by
  `llm/tool_parser.py` (handles fences, trailing commas, unescaped newlines, and even
  *salvages* truncated `write_file` bodies). The orchestrator validates + executes.
- **Sandbox (`tools/sandbox.py`):** path jail + command denylist (incl. all git) +
  per-command timeout + network classification. No Docker.
- **Context (`context/`):** `builder` assembles priority-ordered `Block`s
  (steering/design/manifest/source); `compressor` swaps the most distant blocks for
  their one-line manifest summaries to fit the per-phase budget.
- **Event layer (`server/`):** `EventBus` (worker-thread → asyncio bridge) → SSE →
  terminal `cli/renderer.py` **and** a React/Zustand web IDE (`ui/`). `run.log` JSONL.
- **LLM client (`llm/client.py`):** LiteLLM streaming; `ThinkSplitter` separates
  `<think>…</think>` and `reasoning_content` into thinking vs output events.

### codehamr — minimal interactive terminal agent, single loop

A Go TUI agent for local LLMs over **any OpenAI-compatible endpoint**.

- **One plain agent loop** (`internal/tui/model.go`): the model calls tools until the
  work is done, then replies. A turn ends precisely when the model emits no tool call.
  No phases, no router, no sub-agents, no skill system.
- **Four tools** (`internal/tools/`): `bash`, `read_file`, `write_file`, **`edit_file`**
  (surgical single-anchor replace). **Native OpenAI tool-calling**, not text parsing.
- **One embedded system prompt** (`internal/config/PROMPT_SYS.md`) that instils:
  execution-before-explanation, verify-as-a-habit, anti-truncation file discipline,
  surgical edits, and `unverified:` honesty.
- **Conversation packing (`internal/ctx/ctx.go`):** newest-first whole-message packing
  to a token budget; tool-output truncation (head+tail+marker); wire-shape repair so
  local backends don't 400; headroom margin against char/4 undercount.
- **LLM client (`internal/llm/llm.go`):** single chat-completions path; inter-frame
  **idle watchdog**; `reasoning_effort` auto-fallback; `X-Context-Window` harvest;
  streamed tool-call accumulation; `_parse_error` sentinel for server-truncated args.
- **Soft-nudge backstops (`model.go`):** repeated-failure, runaway-iteration,
  empty-reply, and finish-verify nudges — injected system notes, never hard yields.

### The key architectural difference (drives every recommendation below)

agentic-coder is a **phased autonomous pipeline with hard bounded retries**; codehamr is
a **single continuous loop with soft nudges**. We are **not** proposing to collapse
agentic-coder into a single loop — its SDD pipeline is its identity. We are proposing to
adopt the **tool-level and conversation-level robustness** codehamr earned the hard way,
which slots cleanly into agentic-coder's existing ephemeral-conversation loops without
touching the pipeline shape.

One structural gap surfaced during the read that reframes several adoptions:
**agentic-coder's ephemeral implementer/reviewer conversations have no token
management at all.** `Implementer._drive` (and `reviewer._drive`) append assistant
replies and full tool outputs to `self.messages` and never compress or truncate them —
`client.complete` hands the whole list to LiteLLM with `num_ctx=131072`. A long subtask
(up to `_MAX_STEPS = 48` rounds) with a few big file reads or verbose test logs can
exceed the window, at which point **Ollama silently front-truncates and drops the system
prompt** (steering + tool protocol + the plan). This is the exact failure
`internal/ctx/ctx.go` was written to prevent. Adoptions #1 and #2 below close it.

---

## 2. Adoption catalog

Each item: **what & where**, **problem it solves**, **how to implement in agentic-coder**,
**risks/tradeoffs**. Ordered by value.

---

### TIER 1 — High value, closes a real silent-failure hole or a clear capability gap

---

#### #1 — Budget-aware packing for the ephemeral implementer/reviewer conversations

**What & where (reference):** `internal/ctx/ctx.go` — `Pack(history, budget)` keeps whole
messages newest-first until the budget fills (always keeping the newest), plus
`Budget(ctxSize)` (subtract fixed reservations, leave a headroom margin) and
`ResponseReserve`. codehamr never lets the conversation silently overflow the server's
window.

**Problem it solves:** agentic-coder's `stages/implementer.py::_drive` and
`stages/reviewer.py::_drive` grow `self.messages` unboundedly and pass it straight to the
model. There is **no packing, no compression, no truncation** on these loops (the
`context/compressor.py` path is only used by the *planner*, via `builder.assemble`). On a
long subtask this overruns `num_ctx`, Ollama front-truncates, and the model loses its
steering + tool protocol + plan mid-task — a silent, hard-to-debug failure.

**How to implement:**

- New module `agentic-coder/context/conversation.py`:
  ```python
  from tokens import estimate_messages_tokens, estimate_tokens

  def pack_conversation(messages: list[dict], budget: int) -> list[dict]:
      """Keep messages[0] (system) pinned, then keep newest-first until budget.
      The newest non-system message is always kept even if it alone exceeds budget.
      agentic-coder uses plain user/assistant roles (no tool_call_id pairing), so the
      dangling/orphan/anchor repair passes codehamr needs are NOT required here."""
      if not messages:
          return messages
      system = messages[0:1] if messages[0].get("role") == "system" else []
      body = messages[len(system):]
      used = estimate_messages_tokens(system)
      kept: list[dict] = []
      for msg in reversed(body):
          cost = estimate_tokens(str(msg.get("content", ""))) + 4
          if kept and used + cost > budget:
              break
          kept.append(msg); used += cost
      kept.reverse()
      return system + kept
  ```
- In `Implementer._drive` and `reviewer._drive`, before each `services.client.complete(...)`:
  ```python
  budget = self.services.config.usable_budget_for(self.phase)
  msgs = pack_conversation(self.messages, budget)
  result = svc.client.complete(self.phase, msgs)
  ```
  Note: pack the **copy sent to the model**; keep `self.messages` intact as the durable
  record (or trim it if memory matters — sending a packed view is enough to fix the bug).
- Reuse the existing `usable_budget_for(phase)` (budget − `reserve_for_output`) and
  `tokens.py` estimators — no new token machinery needed.

**Risks/tradeoffs:** Dropping the oldest turns can lose early context within a subtask
(e.g. the first file the model wrote). Mitigations: the plan + steering live in the pinned
system message and survive; the durable file state is on disk and re-readable via
`read_file`; budgets are generous (114k usable). Low risk, high upside. Keep it simple —
do **not** port codehamr's tool-pairing repair passes; they are specific to native
tool-calling and don't apply to agentic-coder's text protocol.

---

#### #2 — Tool-output truncation with an actionable marker

**What & where (reference):** `internal/ctx/ctx.go::Truncate` collapses any tool output
over `ToolOutputCap` (6k tokens) to `first 2k + last 2k` with a marker that *tells the
model what to do*: "the middle is OMITTED … re-run narrower (grep/sed/head/tail) to read
the omitted span." Applied to **every** tool result before it enters history
(`tools/bash.go::Execute`).

**Problem it solves:** agentic-coder truncates command output to **60,000 chars** in the
sandbox (`tools/sandbox.py::_truncate`) and feeds the *whole* thing into the ephemeral
conversation via `ToolResult.display` (`_render_run`, `_read_file`). 60k chars ≈ 15k
tokens **per tool result**; a handful of big reads or noisy test logs dominate a local
model's window and accelerate the overflow in #1. The model also isn't told the output was
cut or how to recover.

**How to implement:**

- Add to `agentic-coder/context/conversation.py` (or `tokens.py`):
  ```python
  def cap_tool_output(text: str, max_tokens: int = 6000, head_tail: int = 2000) -> str:
      if estimate_tokens(text) <= max_tokens:
          return text
      limit = head_tail * 4  # chars
      head, tail = text[:limit], text[-limit:]
      marker = (f"\n───── output truncated: ~{estimate_tokens(text)} tokens, "
                f"first ~{head_tail} + last ~{head_tail} shown, middle OMITTED. "
                f"This is a PARTIAL view — re-run narrower (grep/sed/head/tail) "
                f"to read the omitted span. ─────\n")
      return head + marker + tail
  ```
- Apply it in `tools/registry.py` where `display` is built (`_render_run`, `_read_file`),
  **or** at the append site in `implementer._drive` / `reviewer._drive`. Applying inside
  the registry keeps both consumers (and any future ones) consistent.
- Keep the sandbox's 60k cap as the outer safety net; this adds the conversation-facing,
  model-actionable cap on top.

**Risks/tradeoffs:** A test failure whose key assertion sits in the omitted middle could
be hidden — but 4k tokens of head+tail almost always contains the failing assertion and
the traceback, and the marker explicitly steers the model to `grep` for the rest. codehamr
ships exactly this tradeoff. The 6k cap should be a named constant so it's tunable.

---

#### #3 — Add an `edit_file` tool (surgical single-anchor replace)

**What & where (reference):** `internal/tools/edit.go::EditFile(path, old_string, new_string)`
— `old_string` must appear **exactly once**; rejects empty anchor, no-op edits,
ambiguous/overlapping matches; and detects the #1 LLM edit failure (a near-miss that
"differs only in whitespace") with a targeted error. The schema description explicitly
steers the model to prefer it over `write_file` for anything short of a full rewrite.

**Problem it solves:** agentic-coder has **only whole-file `write_file`**. The system has
no way to make a one-line change without re-emitting the entire file. As codehamr's prompt
puts it, "every rewrite is a fresh chance to inject the one-character typo that … dead-stops
the whole file." This is most acute in the **FIX loop** (`prompts/fix.j2` literally tells
the model to "Use write_file with the corrected COMPLETE file contents") — exactly where a
surgical edit is safest and a full rewrite is riskiest.

**How to implement:**

- `tools/registry.py`: add `_edit_file(self, args, phase)`:
  ```python
  def _edit_file(self, args, phase):
      path = (args or {}).get("path")
      old = (args or {}).get("old_string"); new = (args or {}).get("new_string")
      if not path: return _err("edit_file", "missing required arg 'path'")
      if not old: return _err("edit_file", "missing required arg 'old_string'")
      if old == new: return _err("edit_file", "no change: old_string equals new_string")
      try: target = self.workspace.resolve_in_root(path)
      except PathEscapeError as exc: return _err("edit_file", str(exc))
      if not target.exists(): return _err("edit_file", f"file not found: {path}")
      content = target.read_text(encoding="utf-8", errors="replace")
      n = content.count(old)
      if n == 0:
          # whitespace-diff hint, like codehamr, helps the model fix the bytes
          ...
          return _err("edit_file", f"old_string not found in {path}")
      if n > 1:
          return _err("edit_file", f"old_string appears {n}× — add context to make it unique")
      target.write_text(content.replace(old, new, 1), encoding="utf-8")
      rel = self.workspace.relative(path)
      self.manifest.record(rel, args.get("summary"))  # keep manifest current
      self.bus.file_written(phase, rel, "edit", content=target.read_text(encoding="utf-8"))
      return ToolResult("edit_file", True, {"path": rel}, display=f"edited {rel}")
  ```
  and register `"edit_file": self._edit_file` in `dispatch`'s handler map.
- `llm/tool_parser.py`: add `"edit_file"` to `_KNOWN_TOOLS`, and an `edit_file` branch in
  `_salvage_tool_call` (grab `path`, `old_string`, `new_string`) so a truncated edit still
  recovers.
- `tools/registry.py::TOOL_INSTRUCTIONS`: document `edit_file` and steer "prefer edit_file
  over write_file for any change short of a full rewrite."
- `prompts/fix.j2`: change step 2 from "Use write_file with the corrected COMPLETE file
  contents" to "Use **edit_file** for a targeted fix (preferred); only rewrite the whole
  file with write_file if the change is structural."
- UI: `ui/src/.../ToolCallsTab.tsx` and `lib/phase.ts`/`format.ts` already key on tool
  name strings; add `edit_file` to `TOOL_COLOR` and any tool-name switches.

**Risks/tradeoffs:** A second write path for the model to choose wrongly. Mitigated by the
exact-match-once guarantee (a bad anchor fails loudly and the model reacts, same convention
as a non-zero exit) and the whitespace-diff hint that turns the most common failure into a
one-step correction. This is the single biggest capability gain and is well-isolated.

---

#### #4 — Anti-truncation file-writing discipline + a recovery message on truncated calls

**What & where (reference):** `internal/config/PROMPT_SYS.md` ("Writing files — the rule
that decides whether your artifact ships working") + `internal/tools/bash.go::runRaw`,
which, when a tool call's args came back as `_parse_error` (server cut the stream at its
output-token limit), returns a precise recovery instruction: *don't retry the whole-file
write; build it in chunks with `cat > path <<'EOF'` … then `cat >> path <<'EOF'` appends;
verify with `wc -c`.* The `write_file`/`edit_file` schema descriptions mirror this so both
instruction channels agree.

**Problem it solves:** Local models + a streaming server truncate large `write_file` bodies
mid-stream. agentic-coder already *detects and salvages* this in `tool_parser._salvage_tool_call`
— impressive — but it **silently writes the imperfect file and says nothing**, so the model
has no idea it was truncated and often re-emits the same too-large write, burning the turn.
It also never teaches the heredoc-append discipline that avoids the wall in the first place.

**How to implement:**

- In `tool_parser.py`, make salvage observable: when `_salvage_tool_call` fired (vs. a clean
  parse), tag the `ToolCall` (e.g. `salvaged=True`). In `registry._write_file`, if the call
  was salvaged or the content looks truncated (ends mid-token / unbalanced braces), append a
  codehamr-style note to `display`:
  > "NOTE: this write may have been truncated server-side. Do NOT retry the same whole-file
  > write. For large files use `run` with heredoc appends: `cat > path <<'EOF'` … `EOF`
  > for the first part, then `cat >> path <<'EOF'` … `EOF` per part; verify with `wc -c path`."
- Add the same discipline to the always-injected rules in
  `stages/implementer.py::HARD_RULES` (a few lines) and to `TOOL_INSTRUCTIONS`, so the
  guidance is present *before* the model hits the wall, not only after.
- Optional: extend the SDD `steering.j2` template so generated `steering.md` includes a
  "large files → heredoc appends; small edits → edit_file" rule.

**Risks/tradeoffs:** Heredoc appends run through the sandbox, which must already permit them
(it does — `cat`/redirection inside the root pass the path jail). Slightly more prose in the
hard rules (cheap; these are small models, but this rule directly prevents a common
budget-burning loop). Coupled to #3 (edit_file) for the "small change" half of the advice.

---

#### #5 — Verification-as-a-habit and `unverified:` honesty in the prompts

**What & where (reference):** `PROMPT_SYS.md` §"Verify your work" — tie a check's exit code
to its assertion; **no false greens** (`|| true`, `2>/dev/null`, deleting the failing
assertion); **don't manufacture proof** (counting braces / grepping a name proves nothing);
when a check genuinely can't run, say `unverified: <what> — <why>` and lead with it, never
bury it under a confident "works." Reinforced at runtime by the finish-verify nudge
(`model.go::maybeVerifyNudge`).

**Problem it solves:** agentic-coder's loop runs tests, but nothing stops the small model
from writing **trivially-passing tests** (the task-planner prompt even warns about this) or
the reviewer from declaring success it didn't verify. agentic-coder's prompts
(`reviewer.j2`, `steering.j2`, `planner.j2`'s "Tests to Write") are thin on test-honesty
compared to codehamr's battle-tested wording.

**How to implement:**

- `prompts/planner.j2` → expand "## Tests to Write": "Each test must FAIL if the behavior is
  broken — assert real outputs, not that a function exists or a file is non-empty. No
  `assert True`, no tests that pass by default."
- `prompts/reviewer.j2` → add: "Never report a check you didn't run. Don't silence a failing
  check to make it pass (`|| true`, `2>/dev/null`, deleting an assertion). If something
  can't be verified here, write `unverified: <what> — <why>` and lead the summary with it."
- `stages/implementer.py::HARD_RULES` → add the false-green sentence (it's the always-on
  channel that survives every compression).
- `stages/sdd_generator.py` steering template → instruct the generated `steering.md` to
  carry a "Definition of Done = the check that proves it, actually run" rule.

**Risks/tradeoffs:** Pure prompt text — zero architectural risk, immediately reversible.
Local models follow instructions imperfectly, so treat this as a quality nudge, not a
guarantee; pair with #6's runtime backstops for teeth. Highest value-to-effort ratio in the
whole plan.

---

### TIER 2 — Strong robustness gains, architecturally compatible

---

#### #6 — Soft-nudge backstops inside the ephemeral drive loops

**What & where (reference):** `internal/tui/model.go` — four deterministic backstops, each
an **injected system note, never a hard yield**, all prefixed by `nudgeOrigin`
(`"[Automated codehamr check — not a message from your user.] "`):
- `maybeFailureNudge` — same tool+target failing the same way `maxToolFailStreak` (5×) →
  "stop repeating it; change approach." Keyed on tool+target (`toolTargetKey`), not full
  args, so cosmetic retry differences can't defeat it.
- `maybeRunawayNudge` — `maxToolRounds` (75) tool calls without finishing → self-assess.
- empty-reply handling (`newestAssistantEmpty`) — turn ended with no content and no tool
  call → re-prompt once.
- `maybeVerifyNudge` — a substantial turn finishing with a confident summary → re-ground to
  the request and actually verify (skipped if it already said `unverified`).

**Problem it solves:** agentic-coder's hard ladder (fix×3 → escalate×2 → block) governs the
**outer** loop, but the **inner** `Implementer._drive` (up to 48 steps) has weak backstops:
it only catches "said DONE but wrote no files" (one nudge) and malformed calls (re-ask).
It has **no** detection for a model that (a) re-runs the same failing command every step,
(b) re-reads the same files in a loop without progress, or (c) returns `files_written > 0`
as a false success after spinning. These map directly onto codehamr's failure/runaway
nudges.

**How to implement (in `stages/implementer.py::_drive`):**

- Track `last_tool_key` (tool name + path, or bash first line — port `toolTargetKey`) and a
  `fail_streak`. After each `registry.dispatch`, classify the result as failed (port
  `toolResultFailed`: file tools fail with a leading `(`/`ERROR:`; `run` fails on non-zero
  `exit_code`). On the same target failing `MAX_TOOL_FAIL_STREAK` (≈4) times, append:
  ```python
  self.messages.append({"role": "user", "content": NUDGE_ORIGIN +
      "The last N calls to the same target failed the same way. Stop repeating it — "
      "read the error and change approach, or report what's blocking you."})
  ```
- Track `tool_rounds`; at a `MAX_TOOL_ROUNDS` threshold inject a one-time runaway self-check.
- Add `NUDGE_ORIGIN = "[automated check — not from your user] "` and prefix the existing
  "You said DONE but wrote no files…" nudge with it (codehamr's hard-won insight: a weak
  model reads a bare mid-turn note as the user's turn and stops).

**Risks/tradeoffs:** Nudges are heuristic; the thresholds need the same care codehamr took
(generous enough not to interrupt honest trial-and-error). They complement, not replace, the
hard ladder — the loop stays bounded by `_MAX_STEPS`. Keep the classifier conservative
(`read_file` legitimately returns content starting with `(`, e.g. Lisp — codehamr matches
only its real failure strings; mirror that).

---

#### #7 — Inter-frame idle watchdog instead of a flat end-to-end stream timeout

**What & where (reference):** `internal/llm/llm.go` — `streamIdleTimeout` resets on **every**
SSE frame; only a stream gone silent after `200 OK` trips it, so a slow-but-alive local
model doing long prefill on big context is never killed mid-stream, while a genuinely dead
socket is still caught (by OS TCP keepalive + the idle timer). Overridable via
`CODEHAMR_IDLE_TIMEOUT`.

**Problem it solves:** `llm/client.py::_stream` passes a flat
`timeout=self.config.limits.sandbox_timeout * 30` (≈3600s) to LiteLLM. A flat timeout is the
wrong tool: a big-context call where the model legitimately prefills for >2 min looks
identical to a hang. The right signal is *inter-frame silence*, not total elapsed.

**How to implement:** LiteLLM streaming yields chunks from a generator; wrap the iteration in
a watchdog that records the time of the last chunk and aborts only after `idle_timeout`
seconds with no new chunk (e.g. iterate in a thread feeding a `queue.Queue` and `get(timeout=idle)`,
or use httpx-level read timeouts under the hood). Surface a clear "server stopped sending data
for Ns" error like codehamr does. Add a config/env knob.

**Risks/tradeoffs:** More plumbing than a flat timeout; LiteLLM may not expose per-frame
timeouts cleanly, so this might need a thread+queue wrapper around the stream. Medium effort,
medium payoff — worth doing once large-context subtasks start timing out spuriously. Lower
priority than #1–#6.

---

#### #8 — Tool-call-leak / stranded-intent diagnostic

**What & where (reference):** `internal/tui/model.go::toolCallLeakWarning` — when the newest
assistant message carries a `<tool_call>` opener as **text** but no structured call, codehamr
tells the user the server's tool-call parser is misconfigured and points at the fix.

**Problem it solves:** agentic-coder uses text parsing, so it's more tolerant — but the
parser still fails sometimes (the `_drive` re-ask path). When the implementer exhausts
re-asks and returns failure, the event stream just shows a generic error. A targeted
diagnostic ("the model emitted prose instead of a parseable tool call N times") is more
actionable in the renderer/UI.

**How to implement:** In `_drive`, when re-asks are exhausted, emit a specific `error`/`log`
event ("model never produced a parseable tool call — last reply was prose; check the model's
tool-call formatting"). Surface it in `cli/renderer.py` and the web `EventLogTab`. Small,
self-contained.

**Risks/tradeoffs:** Cosmetic/diagnostic only; no behavior change. Low effort, low-but-real
debugging value.

---

### TIER 3 — Config & robustness niceties (marginal, do if touching that code)

---

#### #9 — Strict config decode + clearer config errors

**What & where (reference):** `internal/config/config.go::Bootstrap` — strict YAML decode
(`KnownFields(true)`): an unknown/typo'd top-level key fails loudly instead of being silently
ignored; nil-profile and dangling-`active` get readable errors; symlink defences on the
config dir/file.

**Problem it solves:** agentic-coder's `config.py::_read_yaml` + `load_config` silently ignore
unknown keys, so a typo like `modeltls:` or `limts:` is accepted and the defaults quietly
apply — a confusing "why is it using deepseek for everything" debugging session.

**How to implement:** In `load_config`, after parsing, warn (or error) on unrecognized
top-level keys against a known set (`models`, `model_options`, `limits`, `server`,
`project_dir`, `ollama_base_url`, `num_ctx`, `dump_llm_calls`). A warning to stderr is enough
given the local, single-user context; don't over-engineer to codehamr's symlink-hardening
(agentic-coder's config is local and read-only, no bearer tokens at stake).

**Risks/tradeoffs:** Tiny. Only risk is being *too* strict and rejecting a harmless extra key;
a warning (not a hard fail) is the safe default for a local tool.

---

#### #10 — Context-budget headroom against tokenizer undercount

**What & where (reference):** `internal/ctx/ctx.go` — `budgetHeadroomDivisor` packs to ~90%
of the declared window because char/4 *undercounts* code/JSON-heavy histories, so packing to
the literal ceiling risks the real prompt spilling past the window (silent Ollama
front-truncation).

**Problem it solves:** agentic-coder already mitigates this better than codehamr (it uses
**tiktoken**, not char/4, and reserves 8192 for output). But tiktoken's `cl100k_base` still
isn't the local model's exact tokenizer, and the compressor packs right up to
`usable_budget_for`. A small headroom margin would harden the same silent-truncation edge.

**How to implement:** In `context/compressor.py::fit` (and the #1 `pack_conversation`), shave
a few percent off the budget: `budget = int(self.config.usable_budget_for(phase) * 0.95)`.
One line, plus a comment explaining why.

**Risks/tradeoffs:** Negligible. Slightly more aggressive compression in rare near-budget
cases, which is exactly the safe direction.

---

## 3. Considered and **not** recommended (with rationale)

Being explicit about what *not* to take is as important as the adoptions.

- **Native OpenAI tool-calling instead of text `<tool_call>` parsing.** codehamr relies on
  the server's tool-call parser and structured `tool_calls` deltas. agentic-coder
  *deliberately* uses text parsing because "local models have unreliable native
  function-calling" — and its tolerant parser (with salvage) is a genuine asset. Switching is
  a large rewrite (client, parser, registry, conversation shape, and the `ThinkSplitter` path)
  with real regression risk across the model zoo in `config.yaml`. **Recommendation:** keep
  text parsing. If ever revisited, do it as a *hybrid* (try native `tool_calls`, fall back to
  text) behind a per-model flag — but that's a project, not an adoption, and out of scope here.
  (Note: if it *were* adopted, codehamr's wire-shape repair passes in `ctx.go` —
  dropDanglingToolCalls/dropOrphanTools/anchorUserMessage/demoteSystemMessages — would become
  necessary; they are pointless under the current text protocol.)

- **Collapsing the phased pipeline into a single loop.** This is the core identity difference.
  agentic-coder's intake→…→review SDD pipeline with per-phase models and budgets is the
  product. codehamr's single loop is a *different product*. Don't merge them.

- **Self-update (`internal/update/`).** agentic-coder is a local, git-free dev tool with no
  release/binary distribution channel. Irrelevant.

- **The TUI itself (`internal/tui/` chips, popover, keys, prompt history, status bar render).**
  agentic-coder's UX is the React/SSE web IDE, not a terminal app. These don't port. (The
  *status-bar ideas* — live tok/s, elapsed, a frozen run summary — could inspire small web-UI
  touches, but that's UI polish, not an architectural adoption.)

- **`reasoning_content` / thinking handling.** Already correct in agentic-coder
  (`llm/client.py::_chunk_parts` reads `reasoning_content`/`reasoning`, `ThinkSplitter` handles
  inline `<think>`). codehamr's `delta.reasoning` is the same idea. **No change needed** —
  noted only to record that this was checked and is already aligned.

- **`X-Context-Window` / live-budget headers.** These are specific to codehamr's hosted
  hamrpass proxy. Ollama doesn't send them (codehamr itself falls back to config there), so
  there's nothing to harvest in agentic-coder's local-only setup. Skip.

---

## 4. Suggested sequencing

Grouped so each step is independently shippable and testable:

1. **Conversation safety (closes the silent-overflow hole):** #1 packing + #2 tool-output cap
   + #10 headroom. One coherent change to the ephemeral loops; biggest correctness win.
2. **`edit_file` (#3)** — isolated new tool; unlocks #4's "small change" advice and de-risks
   the FIX loop.
3. **Prompt discipline (#4 anti-truncation + #5 verification honesty)** — pure prompt/rules
   text; cheap, reversible, high quality impact. Ship alongside #3.
4. **Runtime backstops (#6 nudges)** — needs the failure classifier; thresholds want tuning,
   so land it after the above are stable.
5. **Polish (#7 idle watchdog, #8 leak diagnostic, #9 strict config)** — independent, do as
   capacity allows.

Each item lists its exact target files above. Nothing here changes the pipeline shape, the
event schema, the sandbox safety model, or the no-git constraint.

---

*End of plan. Awaiting approval before implementing.*
