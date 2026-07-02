# Suggestions, gaps & future work — AIForge

Observations from a full read of `agentic-coder/`. Grouped by priority. Each item notes the
**gap**, **why it matters**, and a concrete **direction**.

> **✅ Implemented (2026-06-30):** A1 model eviction (the `{"error":"EOF"}` fix), A2 dead-`tester`
> removal (the implementer owns tests), B2 the coding/planning `thinking:` toggle, the tok/s part of
> B3 (emitted on `llm_complete`; shown live in CLI + web UI), C1 the `tests/` suite (functional +
> opt-in small-model e2e), and D1 the spec-drift / config-precedence notes. Plus a UI fix unrelated to
> this list: the Thinking panel's auto-scroll is now *sticky* (it no longer yanks to the bottom on
> every token, so you can scroll up and collapse it). Items below marked **✅ DONE** are finished; the
> rest remain backlog.

---

## A. Correctness / directly tied to the current failure

### A1. Model lifecycle management — no eviction on model switch (root cause of the EOF bug) — ✅ DONE
- **Was:** `LLMClient` passed `keep_alive` verbatim with **no unload/evict logic**, so on a 16GB-VRAM
  box the planner's `qwen3.6:27b` stayed warm while `qwen3-coder:30b` loaded → OOM → `{"error":"EOF"}`
  at the first IMPLEMENT call, every run.
- **Done:** `LLMClient` now tracks the resident model and unloads it (Ollama `/api/generate
  {keep_alive:0}`, guarded by `/api/ps`) before a **different** model loads; a same model re-called
  stays warm; `Orchestrator.run()` unloads at the end. Gated by `evict_on_model_switch` (default
  true). Tests in `tests/test_client_eviction.py`.
- **Still open (optional):** a startup **fit check** that estimates each model's footprint at
  `num_ctx` vs detected VRAM/RAM and warns when two adjacent phases can't co-reside.

### A2. `tester` phase is dead configuration — ✅ DONE
- **Was:** `tester` was declared in `config.yaml` / `config.py:DEFAULT_MODELS` / `context_budget.yaml`
  but never invoked.
- **Done:** removed the `tester` key from all three; the implementer owns test-writing in its
  IMPLEMENT+WRITE_TESTS conversation (it has the context of what it just built), as the user wanted.

### A3. No final "whole-suite green" gate
- **Gap:** The reviewer runs tool calls and writes `review.md`, but `pipeline_complete` is reported
  from task counts, not from a final full test-suite pass. A run can end "done" with subtasks that
  individually passed but don't integrate.
- **Direction:** Add an explicit final `run` of the stack's full test command in the reviewer (or a
  post-review gate), surface its pass/fail in the completion summary, and append failures to
  `blocked.md`.

---

## B. Performance / UX on this hardware (16GB VRAM AMD + 32GB RAM)

### B1. Model selection is mismatched to the box — and swaps are the enemy
- **Gap:** The config picks the two largest models that *almost* fit, guaranteeing a heavy reload on
  every PLAN↔IMPLEMENT boundary (and, per A1, OOM). Upstream `/think` reasoning calls take **10–45
  min each** here (see `run.log`: 615s, 873s, 1417s, 2701s).
- **Direction:** Offer **named profiles** (`fast` / `balanced` / `max`) selectable in config, and
  prefer a profile that **minimizes model swaps** — e.g. one reasoner + one coder that can both stay
  resident, or even a single mid-size model family for the whole pipeline. Smaller models that *stay
  warm* will out-throughput larger models that reload every call.

### B2. Control thinking for coding vs planning — ✅ DONE
- **Was:** `detect_thinking_mode` forced `/think` on anything starting with `qwen3` with no easy way
  to control it per role.
- **Done:** added a `thinking:` block to `config.yaml` (`coding:` / `planning:` = true/false/null)
  resolved with precedence `model_options > thinking-toggle > auto-detect`
  (`config.thinking_mode_for` + `_toggle_thinking`). Shipped on for both (quality over speed, as the
  user asked) but now flippable. Tests in `tests/test_thinking.py`.

### B3. tok/s throughput (✅ DONE) — full ETA / wall-clock budget still open
- **Done (tok/s):** the client emits `tokens_per_second` on `llm_complete`; the CLI renderer and the
  web UI (HeaderBar + per-block in the Thinking panel) show **live + final tok/s**.
- **Still open:** a run-level **ETA** and an optional **global wall-clock budget** that pauses/blocks
  gracefully when exceeded (a 2.6-hour run still gives no time estimate up front).

---

## C. Robustness / maintainability

### C1. Test suite for the tool's own fragile parsers — ✅ DONE
- **Done:** added `agentic-coder/tests/` (pytest, 51 fast tests, no LLM) covering tool-call salvage on
  malformed JSON, the sandbox denylist (every `git` form) + path-jail, tasks.json scheduling/blocked-
  skipping, compressor over-budget summarization + `compression_failure`, conversation packing,
  thinking resolution, **model eviction**, throughput, and test-command extraction — plus an opt-in
  **small-model** end-to-end test (`test_e2e_small.py`, `AIFORGE_E2E=1`) that never touches the 27B/30B.
  `pytest.ini` sets the import root; `pytest` added to `requirements.txt`.
- **Still open (optional):** wire the e2e into a make/CI target with a model-availability gate.

### C2. Stale `in_progress` on crash requires manual `--resume`
- **Gap:** The crashed run left `T001`/`T001.1` as `in_progress`; only `--resume` resets them. A
  fresh `/start` on the same slug, or a second crash, can leave confusing state.
- **Direction:** On fresh start against an existing slug with `in_progress` subtasks, detect the
  stale state and offer/auto-do the reset; or write a clean-shutdown marker and reconcile on boot.

### C3. Single global run / single project at a time — ⛔ WON'T DO (de-scoped by user)
- The user builds **one app at a time**, so the single-run design is correct for this tool. Left as
  the documented, intended scope; no multi-project work planned.

### C4. Token-estimate drift at `num_ctx 32768`
- **Gap:** Budgets use tiktoken `cl100k` as a proxy for qwen/llama tokenizers. Mitigated with 5%
  headroom, but code-heavy windows can still undercount near the ceiling, risking Ollama's silent
  front-truncation.
- **Direction:** Calibrate against real usage — read `prompt_eval_count` from Ollama responses and
  adjust the headroom factor adaptively, or query the model's tokenizer when available.

### C5. Network policy is classification-only (acknowledged)
- **Gap:** Non-install steps only get proxy env scrubbed, not a real socket jail (no-Docker
  constraint). A malicious/buggy generated command could still reach the network.
- **Direction:** On Linux, optionally wrap non-install `run` steps in a network namespace
  (`unshare -n` / `ip netns`) for a genuine offline jail where available; keep the scrub as fallback
  and document the residual risk.

---

## D. Minor / docs

- **D1. Spec drift — ✅ DONE:** added a callout at the top of `agentic-coder.md` §4 (and a precedence
  banner in `config.yaml` + notes in README/CLAUDE.md) stating that the live `config.yaml` is the
  single source of truth and the spec's example config is outdated/illustrative only.
- **D2. `run.json` vs `run.log`:** there is no `run.json`; the durable log is `run.log` (JSONL) and
  the task state is `tasks.json`. Consider writing a small `run.json` end-of-run **summary snapshot**
  (result, counts, elapsed, blocked ids) since users reach for that name.
- **D3. Test-prompt filename:** the shipped prompt file is `testpropmt.txt` (misspelled); README
  examples say `prompt.txt`/`testprompt.txt`. Rename the file or fix the references to avoid a
  "prompt file not found" stumble.
- **D4. UI build is manual & unverified:** add a convenience script/Make target for
  `npm install && npm run build`, and a one-liner dev note for `npm run dev` (proxy to :8765).
