# MASTER BUILD PROMPT — "AIForge" Local Autonomous Coding Pipeline

## 0. Mission

Build a local, cost-free autonomous software-engineering pipeline — a Claude-Code-style "prompt and walk away" tool that turns a single natural-language prompt into a fully implemented, tested, working application on disk. It orchestrates **local LLMs (via Ollama, through LiteLLM)** as a **state machine**, where each phase uses a configurable model. It generates a suite of spec/steering documents (SDD), plans tasks and subtasks, then implements and tests each subtask in an isolated conversation loop until the whole app is built and passing.

This build covers the **backend pipeline + a streaming event layer + a CLI renderer only**. Do NOT build a frontend/UI now, but architect the event layer so a VSCode-style web UI can be added later with zero backend changes.

---

## 1. HARD CONSTRAINTS (read first, never violate)

- **NEVER use git.** Do not run `git init`, `add`, `commit`, `push`, `status`, or any git subcommand. Do not create `.gitignore` for the tool itself. Add `git` to the sandbox command denylist so the LLM-driven phases can never invoke it either. The user is not version-controlling this project right now.
- **NEVER operate outside the resolved project directory.** All file writes and all shell commands for a generated app must occur inside the project dir (see §5). Reject any path or command that escapes it.
- **Do NOT build any frontend/UI** in this phase. Build the backend, the SSE event layer, and the terminal CLI renderer. Everything must be structured so a frontend can subscribe to the same event stream later.
- **The model requests, the orchestrator executes.** LLMs never run shell commands directly. They emit tool-call requests; the Python orchestrator validates and runs them, then feeds results back. This is non-negotiable for safety, determinism, and token cost.
- **Durable state lives on disk; conversations are ephemeral.** See §6.

---

## 2. Tool's Own Tech Stack

- **Language:** Python 3.11+
- **LLM abstraction:** LiteLLM (gives a unified OpenAI-style interface to Ollama and lets the user later swap in LM Studio / remote providers by changing a model string).
- **Server / event streaming:** FastAPI + Server-Sent Events (SSE) via `sse-starlette` or manual `StreamingResponse`.
- **Templating:** Jinja2 for all prompt templates.
- **Shell execution:** Python `subprocess` (NOT Docker) with the safety layers in §12.
- **Config:** PyYAML.
- **CLI rendering:** `rich` for live terminal output.
- Keep dependencies minimal; list them all in `requirements.txt`.

---

## 3. Target Project Structure

Create this layout (adapt only where clearly sensible). `aiforge/` is the tool root where `main.py` lives.

```
agentic-coder/
├── main.py                       # Entry: loads config, starts FastAPI server + orchestrator, launches CLI renderer
├── config.yaml                   # User config (models per phase, project_dir, limits)
├── context_budget.yaml           # Per-phase token budgets
├── requirements.txt
├── README.md                     # How to install Ollama, pull models, configure, run
│
├── server/
│   ├── app.py                    # FastAPI app; SSE endpoint /events; control endpoints (start, resume, cancel)
│   └── events.py                 # Event schema, EventBus (async pub/sub), emit helpers
│
├── orchestrator/
│   ├── orchestrator.py           # Top-level pipeline state machine driver
│   ├── states.py                 # State enum + legal transitions
│   └── subtask_loop.py           # The implement → test → fix → escalate loop (the heart)
│
├── stages/
│   ├── intake.py                 # prompt → project_brief.md
│   ├── requirements.py           # brief → requirements.md
│   ├── stack_decider.py          # requirements → stack.md
│   ├── architect.py              # stack + requirements → architecture.md (STACK-SPECIFIC)
│   ├── sdd_generator.py          # → sdd.md + steering.md
│   ├── task_planner.py           # → tasks.json (tasks + subtasks)
│   ├── planner.py                # LARGE-model per-subtask implementation planner
│   ├── implementer.py            # SMALL-model implement + write-tests (owns the ephemeral convo)
│   └── reviewer.py               # Final full-project review pass
│
├── context/
│   ├── builder.py                # Assembles fresh context per LLM call from disk
│   ├── loader.py                 # Reads SDD/steering docs + source files
│   ├── manifest.py               # Maintains file_manifest.md (annotated) + file-directory.txt (raw)
│   └── compressor.py             # Token-budget trimming / summarization with logged decisions
│
├── llm/
│   ├── client.py                 # LiteLLM wrapper; streams tokens to EventBus; per-phase model selection
│   └── tool_parser.py            # Tolerant tool-call parser (fenced JSON, tags, trailing commas, prose)
│
├── tools/
│   ├── registry.py               # Tool definitions + dispatch: read_file, write_file, run
│   ├── sandbox.py                # subprocess executor + all safety layers
│   └── process_manager.py        # Long-running process harness (dev servers, watchers)
│
├── prompts/                      # Jinja2 templates, one per phase
│   ├── intake.j2
│   ├── requirements.j2
│   ├── stack_decider.j2
│   ├── architect.j2
│   ├── sdd.j2
│   ├── steering.j2
│   ├── task_plan.j2
│   ├── planner.j2
│   ├── implementer.j2
│   ├── fix.j2
│   └── reviewer.j2
│
├── cli/
│   └── renderer.py               # Subscribes to SSE, renders live output with rich
│
└── sandbox/                      # DEFAULT project output root (used only if project_dir is blank)
    └── <project_name>/
        ├── .agent/             # All generated control/spec documents
        │   ├── project_brief.md
        │   ├── requirements.md
        │   ├── stack.md
        │   ├── architecture.md
        │   ├── sdd.md
        │   ├── steering.md
        │   ├── tasks.json
        │   ├── file_manifest.md
        │   ├── file-directory.txt
        │   ├── blocked.md
        │   ├── run.log           # JSONL of every event (also feeds future frontend / debugging)
        │   └── llm_calls/        # Optional: dumped prompt+response per call for debugging
        └── (the generated application code)
```

---

## 4. Configuration — `config.yaml`

> ⚠️ **This section is the original spec and its example below is OUTDATED.** The
> **shipped `config.yaml` is the single source of truth** and overrides everything:
> it uses `qwen3.6:27b` (reasoning/planning) and `qwen3-coder:30b` (coding/review),
> `num_ctx: 32768`, and adds knobs not shown here — `evict_on_model_switch` (keep at
> most one model resident, preventing the 27B↔30B swap OOM) and a `thinking:`
> coding/planning toggle. There is **no `tester` phase** (the implementer writes its
> own tests). Trust `config.yaml` + `config.py` over the example below.

The user must have full control over which model runs each phase. Any phase omitted falls back to a documented default. Model strings are LiteLLM-format.

```yaml
# Leave project_dir blank to use the default ./sandbox/<project_name>/
project_dir: ""

ollama_base_url: "http://localhost:11434"

models:
    # ── ONE-TIME PHASES (slow is fine, quality is everything) ──────────────────
    # ~40GB at Q4_K_M: 16GB VRAM + ~24GB RAM. Tight but works. 3-6 tok/s.
    # Fallback if RAM is too constrained: deepseek-r1:32b
    architect: "ollama/deepseek-r1:32b"
    sdd_generator: "ollama/deepseek-r1:32b"

    # ── REASONING PHASES (run once or a handful of times) ──────────────────────
    # ~20GB at Q4_K_M: 16GB VRAM + ~4GB RAM. Good speed.
    requirements: "ollama/qwen3:32b"
    stack_decider: "ollama/qwen3:32b"
    task_planner: "ollama/qwen3:32b"

    # ── PER-SUBTASK PLANNER (runs every loop — needs to be faster than 70B) ────
    # deepseek-r1:32b: strong reasoning, ~20GB, acceptable speed per subtask
    planner: "ollama/deepseek-r1:32b"
    escalation: "ollama/deepseek-r1:32b"

    # ── CODING PHASES (runs most frequently — must be fast) ────────────────────
    # 9GB, fits fully in VRAM. Best coding model at this size.
    implementer: "ollama/qwen2.5-coder:14b"
    tester: "ollama/qwen2.5-coder:14b"

    # ── TOOL CALLING (structured output, fast dispatch) ────────────────────────
    # qwen2.5:7b has the best tool-call reliability of any sub-10B local model.
    # Used by the orchestrator's re-ask loop when the implementer emits a malformed call.
    tool_caller: "ollama/qwen2.5:7b"

    # ── INTAKE (trivial classification task) ───────────────────────────────────
    intake: "ollama/llama3.2:3b"

    # ── FINAL REVIEW (sees the whole codebase — needs coding depth + size) ─────
    reviewer: "ollama/qwen2.5-coder:32b"

model_options:
    # ── DEEPSEEK-R1 PHASES (thinking always on, no toggle needed) ─────────────
    # The model automatically wraps reasoning in <think>...</think> blocks.
    # The llm/client.py MUST detect these tags and emit them as
    # `llm_thinking_token` events (separate from `llm_token`) so the CLI
    # and future frontend can render thinking vs output differently.
    # Do NOT strip thinking tokens — they are valuable debug output.
    deepseek_thinking_always_on:
        - architect # deepseek-r1:32b
        - sdd_generator # deepseek-r1:32b
        - planner # deepseek-r1:32b
        - escalation # deepseek-r1:32b

    # ── QWEN3 PHASES — append /think to the user message ──────────────────────
    # Use for all reasoning-heavy one-time phases. Slower but best output.
    # Time is not a concern for these phases.
    qwen3_thinking:
        - requirements # qwen3:32b
        - stack_decider # qwen3:32b
        - task_planner # qwen3:32b

    # ── QWEN3 PHASES — append /no_think to the user message ───────────────────
    # None currently — qwen3 is only used in reasoning phases.
    # If qwen3 is ever assigned to implementer/tester, add it here.
    qwen3_no_thinking: []

    # ── NO THINKING AVAILABLE ─────────────────────────────────────────────────
    # These models have no thinking mode. Run as-is.
    no_thinking:
        - intake # llama3.2:3b
        - implementer # qwen2.5-coder:14b
        - tester # qwen2.5-coder:14b
        - tool_caller # qwen2.5:7b
        - reviewer # qwen2.5-coder:32b

limits:
    sandbox_timeout: 120 # seconds per command
    max_fix_retries: 3 # small-model fix attempts per subtask
    max_escalations: 2 # large-model re-plans per subtask before BLOCK
    long_process_timeout: 30 # seconds to wait for a background server to become healthy

server:
    host: "127.0.0.1"
    port: 8765
```

## 4b. Configuration — `context_budget.yaml`

Each phase declares a max token budget so the compressor knows how aggressively to trim. Never silently truncate — log every compression decision as an event.

```yaml
# All models support 128K (131072 tokens).
# Ollama MUST have num_ctx set to 131072 — see setup instructions.
# Budgets below are input token limits. reserve_for_output is always subtracted
# before a call is made, leaving headroom for the model's response.
# The compressor logs every decision when it trims below the phase budget.

reserve_for_output:
    8192 # increased from 2000 — large reasoning models
    # produce verbose output, especially with thinking blocks

budgets:
    # ── ONE-TIME ARCHITECTURE PHASES ──────────────────────────────────────────
    # deepseek-r1:32b. Gets everything: brief, requirements, stack, plus
    # full thinking budget. These are the most important calls in the pipeline.
    architect: 122880 # 128K - 8K reserve. Feed it everything.
    sdd_generator: 122880 # Same. SDD needs to see architecture.md in full.

    # ── REASONING / PLANNING PHASES ───────────────────────────────────────────
    # qwen3:32b with /think. Rich context but output is structured JSON/markdown.
    requirements: 122880 # Sees only the project brief — headroom to spare.
    stack_decider: 122880 # Sees brief + requirements.md.
    task_planner:
        122880 # Sees brief + requirements + architecture + sdd.
        # This is the heaviest upstream call.

    # ── PER-SUBTASK PLANNER ───────────────────────────────────────────────────
    # deepseek-r1:32b. Runs every loop iteration. Gets steering + sdd +
    # manifest + relevant source files. Generous budget since codebase grows.
    planner: 122880

    # ── ESCALATION ────────────────────────────────────────────────────────────
    # deepseek-r1:32b. Gets everything the planner gets PLUS full failure
    # history (all stderr/stdout from failed attempts). Needs the full window.
    escalation: 122880

    # ── CODING PHASES ─────────────────────────────────────────────────────────
    # qwen2.5-coder:14b. Gets the plan + minimal necessary files.
    # The planner's job is to make this context small and precise —
    # but budget generously for large files and growing codebases.
    implementer: 122880
    tester: 122880 # Same model, same needs.

    # ── TOOL CALLER ───────────────────────────────────────────────────────────
    # qwen2.5:7b. Only ever sees a short re-ask prompt with the malformed
    # tool call and the expected format. Small budget is correct here.
    tool_caller: 8192

    # ── INTAKE ────────────────────────────────────────────────────────────────
    # llama3.2:3b. Only sees the user's raw prompt. Tiny input.
    intake: 8192

    # ── FINAL REVIEW ──────────────────────────────────────────────────────────
    # qwen2.5-coder:32b. Needs to see the full codebase + all SDD docs.
    # This is the heaviest single call in the whole pipeline.
    reviewer: 122880

# ── COMPRESSION POLICY ────────────────────────────────────────────────────
# When assembled context exceeds the phase budget, the compressor must:
# 1. Always keep verbatim: steering.md, the current task/subtask spec, the plan.
# 2. Always keep verbatim: files the current subtask directly creates/edits.
# 3. Summarize to one-liners (using file_manifest.md descriptions): completed
#    files from distant/unrelated features.
# 4. Summarize to a short paragraph: completed task blocks in tasks.json.
# 5. Emit a `compression` event listing every file that was summarized,
#    with original token count vs. summary token count.
# Never truncate mid-file. Never silently drop content.
# If even after full compression the context still exceeds budget,
# emit a `compression_failure` event and halt the subtask with a clear error.

# ── THINKING TOKEN BUDGET NOTE ────────────────────────────────────────────
# deepseek-r1 and qwen3 thinking blocks consume input+output tokens.
# The thinking tokens count against the model's total context window
# but are NOT part of your input — they come out of reserve_for_output.
# 8192 reserve is intentionally large to accommodate verbose thinking chains.
# If deepseek-r1 consistently hits the output limit mid-thought,
# increase reserve_for_output to 12288 and reduce phase budgets accordingly.
```

---

## 5. Project Directory Resolution

On startup:

1. If `config.yaml`'s `project_dir` is set, use it. Create it if missing.
2. If blank, derive a slug from the project name (the LLM proposes one during intake) and use `./sandbox/<slug>/`.
3. All generated app files and ALL shell commands run inside this directory. Store the absolute resolved path once and pass it everywhere. Every file/command path is validated against it (§12).
4. Create the `.agent/` subdir inside it for all control documents.

---

## 6. Core Architectural Principle — Two-Level Context Model

This is the most important concept; implement it exactly:

- **Pipeline level = stateless across tasks.** Each pipeline stage and each _new subtask_ reconstructs its context from disk every time (steering.md, sdd.md, file_manifest.md, and the specific source files it needs). There is no shared chat memory between subtasks.
- **Subtask level = one accumulating conversation.** The implement → write-tests → run → fix cycle for a _single subtask_ is one growing conversation (messages keep appending) so the small model never loses what it just wrote. **This conversation is discarded the moment the subtask passes** — its only durable outputs are the files it wrote and the updated manifest/task status on disk.

When subtask T002.3 begins, it does NOT inherit T002.2's conversation. It reads disk state (which already reflects everything T002.2 produced) and starts a clean conversation.

---

## 7. The Pipeline Stages (run in order; each emits stage_start/stage_end events)

The tool is **stack-agnostic at intake** and becomes **stack-specific once the stack is decided** — and critically, **`architecture.md` and everything after it must be stack-specific**. It must say "implement the UI in React + TypeScript", never "implement a UI". The stack decision happens _before_ architecture.

1. **INTAKE** (`intake.py`) — Interpret the user's raw prompt. Clarify goals, infer scope, propose a project name/slug. Output `project_brief.md`. Stack-agnostic.
2. **REQUIREMENTS** (`requirements.py`) — Derive functional + non-functional requirements, core entities, user flows. Output `requirements.md`. Stack-agnostic.
3. **STACK DECISION** (`stack_decider.py`) — Choose a concrete, coherent stack (languages, frameworks, DB, test runner, package manager, run commands) justified against the requirements. Output `stack.md`. **This locks the stack** so later phases cannot drift or hallucinate technologies.
4. **ARCHITECT** (`architect.py`) — Produce `architecture.md`, **fully stack-specific**, referencing the chosen technologies by name: directory layout, module boundaries, data model in the chosen DB, API surface, how tests are run for this stack, etc.
5. **SDD GENERATION** (`sdd_generator.py`) — Produce:
    - `sdd.md` — the detailed software design (components, contracts, data shapes, error handling), stack-specific.
    - `steering.md` — coding conventions, naming rules, patterns, "always/never" rules. This file is injected into every implement/fix call to keep 50+ stateless calls coherent.
6. **TASK PLANNING** (`task_planner.py`) — Produce `tasks.json` (schema in §8) broken into **tasks → subtasks**, dependency-ordered, each with a goal/intent and a test strategy.
7. **SUBTASK LOOP** (`subtask_loop.py`) — Iterate every pending subtask through the implement/test/fix/escalate state machine in §9 until all are done or blocked.
8. **REVIEW PASS** (`reviewer.py`) — With the full project assembled, run a final review: check cross-cutting concerns, run the whole test suite once more, surface and optionally fix integration issues. Record anything unresolved in `blocked.md`.

---

## 8. SDD Document Suite — exact contents

Generate all of these under `.agent/`:

- **project_brief.md** — interpreted prompt, clarified goals, success criteria, project name.
- **requirements.md** — functional + non-functional requirements, entities, user flows.
- **stack.md** — chosen stack with versions where relevant, package manager, test runner, the exact commands to install deps / run tests / run the app.
- **architecture.md** — stack-specific system design, directory plan, module map, data model, API surface.
- **sdd.md** — detailed design: per-module responsibilities, function/endpoint contracts, data shapes, error handling, edge cases.
- **steering.md** — conventions and rules injected into every code-gen/fix call.
- **tasks.json** — task/subtask breakdown with live status. Schema:

```json
{
    "project": "todo-app",
    "tasks": [
        {
            "id": "T001",
            "title": "Project scaffold & tooling",
            "goal": "A runnable skeleton with the chosen stack, deps installed, test runner working.",
            "status": "pending",
            "subtasks": [
                {
                    "id": "T001.1",
                    "title": "Initialize project & install dependencies",
                    "intent": "Establish the runnable base every later subtask builds on",
                    "files": ["package.json"],
                    "depends_on": [],
                    "test_strategy": "Dependency install exits 0; test runner executes an empty suite successfully",
                    "status": "pending"
                }
            ]
        }
    ]
}
```

Valid `status` values: `pending`, `in_progress`, `done`, `blocked`.

- **file_manifest.md** — annotated file tree (see §10).
- **file-directory.txt** — raw `ls -R`-style listing (see §10).
- **blocked.md** — any subtask that exhausted retries+escalations, with a short failure summary so the user (or a later run) can intervene.

---

## 9. The Subtask Loop — detailed state machine (`subtask_loop.py`)

For each pending subtask, in dependency order:

```
SELECT_SUBTASK
  - Read tasks.json; pick the next pending subtask whose depends_on are all done.
  - Mark it in_progress; persist tasks.json. Emit subtask_start.

PLAN  (LARGE model = config.models.planner; fresh single call, no convo)
  - Context = steering.md + sdd.md + relevant architecture section
            + file_manifest.md + file-directory.txt
            + cat() of the specific files this subtask touches/depends on
            + parent task `goal` + this subtask `intent` + `test_strategy`.
  - Output: a thorough, unambiguous implementation plan — exact files to create/edit,
    exact changes, what already exists (named functions/imports) so the small model
    does NOT reinvent or hallucinate, the tests to write, and the exact command to run them.
  - This plan is good enough that a small coding model can execute it with minimal extra context.

>>> begin ephemeral conversation (lives only for this subtask) <

IMPLEMENT  (SMALL model = config.models.implementer)
  - Input: the plan + minimal necessary files.
  - Model emits write_file / read_file tool calls. Orchestrator executes them,
    updates the manifest + file-directory.txt, emits file_written events.

WRITE_TESTS  (same conversation, same small model)
  - Model writes tests for exactly what it just implemented, per the plan's test list.
  - Test cases were specified by the PLANNER, not invented freely, to avoid trivial/passing-by-default tests.

RUN  (ORCHESTRATOR executes — model only requested the command)
  - Run the test command in the sandbox (§12). Capture {exit_code, stdout, stderr}.
  - For long-running targets (dev servers), use the process harness (§13): start in
    background, wait for health, run smoke checks, then kill. Reduce to pass/fail.
  - Emit test_run event with result.

  IF exit_code == 0:
     UPDATE_STATE: mark subtask done; persist tasks.json; refresh manifest+file-directory.txt.
     Discard the conversation. Emit subtask_done. → next subtask.

  ELSE (failure):
     IF fix_retries_remaining > 0:
        FIX (same conversation, small model): feed it exit_code + stderr + stdout.
            Ask for a targeted fix via tool calls. Decrement retries. → RUN again.
     ELSE IF escalations_remaining > 0:
        ESCALATE: discard the small-model conversation. Call the LARGE escalation model
            with full context + the entire failure history. It produces a NEW plan.
            Start a fresh ephemeral conversation and re-enter IMPLEMENT with the new plan.
            Reset fix_retries; decrement escalations. Emit escalation event.
     ELSE:
        BLOCK: mark subtask blocked; append a failure summary to blocked.md;
            persist tasks.json. Emit blocked event. Skip to the next independent subtask
            (skip any subtasks that depend on the blocked one).
```

Never allow an unbounded loop. Caps come from `config.limits`.

---

## 10. Context System

- **`context/builder.py`** assembles a fresh prompt for each call: always include `steering.md` (+ `sdd.md` where relevant), then the manifest, then `cat()`'d source files the task actually needs, then the task-specific instruction. Apply the phase's token budget from `context_budget.yaml`, reserving `reserve_for_output`.
- **`context/manifest.py`** maintains two files, regenerated after every write/pass:
    - **`file-directory.txt`** — a raw recursive listing of the project (the `ls -R` the user wants), but **filtered**: exclude `node_modules/`, `.git/`, `dist/`, `build/`, `__pycache__/`, `.venv/`, lockfiles' noise, etc. This is the ground-truth file list so the model always knows what exists and can `read_file` anything it needs.
    - **`file_manifest.md`** — an **annotated** tree where each generated file has a one-line description recorded at creation time (e.g. `src/routes/auth.ts — POST /register, POST /login, JWT issuance`). This gives the planner semantic awareness of the whole codebase without having to `read_file` everything; `read_file` remains the fallback when a one-liner isn't enough. Keep both in sync every pass.
- **`context/compressor.py`** — when context exceeds budget, summarize the least-relevant/most-distant completed files into short descriptions (drawing on the manifest) while keeping the active feature's files verbatim. Emit a compression event documenting what was summarized — never truncate silently.

---

## 11. Tool-Calling Protocol (`llm/tool_parser.py`, `tools/registry.py`)

Local models have unreliable native function-calling, so use a robust text protocol the parser tolerates.

- **Expected model output** for a tool call (accept any of these forms):

```
  <tool_call>
  {"tool": "run", "args": {"cmd": "npm test -- tests/test_auth.ts"}}
  </tool_call>
```

Also accept ```json fenced blocks, raw JSON objects, trailing commas, and surrounding prose. Extract the first valid tool call.

- **Three tools** (defined in `registry.py`):
    - `read_file` — `{ "path": "..." }` → returns file contents.
    - `write_file` — `{ "path": "...", "content": "...", "summary": "one-line description" }` → writes (path-validated), records the `summary` in the manifest, emits `file_written`.
    - `run` — `{ "cmd": "...", "background": false, "timeout": 120 }` → executed by the sandbox; returns `{exit_code, stdout, stderr}`.
- If parsing fails, **re-ask** the model with a short corrective instruction (show it the exact expected format). Cap re-asks (e.g. 3); on exhaustion treat as a subtask failure and let the escalation ladder handle it.

---

## 12. Sandbox & Safety (`tools/sandbox.py`) — subprocess, NOT Docker

Every command requested by the model passes through these layers before execution:

1. **Working-directory jail:** run with `cwd` = resolved project dir. Reject commands containing absolute paths outside the project root, `..` traversal that escapes it, or `~`. Reject `write_file`/`read_file` paths that resolve outside the project root.
2. **Command denylist (reject outright):** `rm -rf /`, `rm -rf ~`, `sudo`, `dd`, `mkfs`, `:(){ :|:& };:` and other fork bombs, `shutdown`/`reboot`, `chmod -R 777 /`, raw `curl`/`wget` piped into a shell (`| sh`, `| bash`), and **all `git` commands**.
3. **Timeout:** every command gets `config.limits.sandbox_timeout`; kill on expiry and return a timeout failure (kills hung processes/infinite loops).
4. **Network:** disabled by default. Allow it only for explicitly-tagged dependency-install steps (e.g. `npm install`, `pip install`), then treat normal command steps as offline.
5. Capture and return `{exit_code, stdout, stderr, duration}`; emit a `tool_result` event for each.

---

## 13. Long-Running Process Harness (`tools/process_manager.py`)

Servers/watchers don't fit a single pass/fail exit code. When `run` is called with `background: true`:

1. Start the process detached, capturing its output.
2. Poll for health up to `config.limits.long_process_timeout` (a health endpoint, an expected log line, or a port becoming open — choose per the stack from `stack.md`).
3. Run any smoke commands the plan specified against it.
4. **Kill the process** and reduce the whole thing to a pass/fail result returned to the loop. Always guarantee cleanup (no orphaned servers) even on error/timeout.

---

## 14. SSE Event Layer (`server/app.py`, `server/events.py`)

A FastAPI server exposes an SSE stream so the CLI now — and a frontend later — can render the full thought process and output live, VSCode-style.

- **Endpoints:**
    - `GET /events` — SSE stream of all events.
    - `POST /start` — body: the user prompt (+ optional overrides). Kicks off the pipeline.
    - `POST /resume` — resume an existing project (§16).
    - `POST /cancel` — cooperatively cancel the current run.
    - `GET /status` — current state snapshot.
- **EventBus:** an async pub/sub the orchestrator publishes to and the SSE endpoint subscribes to. Also append every event as one JSON line to `.agent/run.log`.
- **Event schema** — every event: `{ "type": str, "timestamp": iso8601, "phase": str, "data": {...} }`. Emit at least these types:
    - `stage_start`, `stage_end`
    - `llm_request` `{model, phase, prompt_token_estimate}`
    - `llm_token` `{phase, token}` (stream the model's output token-by-token — this is the "thinking/output" the user wants visualized)
    - `llm_complete` `{phase, total_tokens}`
    - `tool_call` `{tool, args}`
    - `tool_result` `{tool, exit_code?, stdout, stderr, duration}`
    - `file_written` `{path, action}` (`create`|`edit`)
    - `test_run` `{cmd, exit_code, passed}`
    - `subtask_start`, `subtask_done`, `subtask_failed`, `escalation`, `blocked`
    - `compression` `{summarized_files}`
    - `pipeline_complete`
    - `error` `{message, context}`
- Stream LLM tokens from `llm/client.py` (LiteLLM streaming) straight onto the bus as `llm_token` events.

---

## 15. CLI Renderer (`cli/renderer.py`)

Subscribe to `/events` and render live with `rich`: current stage, streaming LLM tokens, tool calls and their results, test outcomes, subtask progress, escalations, and a final summary. This is purely a consumer of the event stream — it holds no orchestration logic, proving the event layer is sufficient for the future frontend.

---

## 16. Resumability

A run can take a long time. On startup, if the resolved project dir already contains a `tasks.json`, offer to **resume**: reset any `in_progress` subtask to `pending`, then continue from the first runnable pending subtask. Completed work is trusted from disk. Provide this via `main.py` (a `--resume` flag) and the `POST /resume` endpoint.

---

## 17. Logging & Debugging Artifacts

- Append every event to `.agent/run.log` (JSONL).
- Optionally dump each LLM call's full prompt + response under `.agent/llm_calls/<phase>_<timestamp>.md` for debugging local-model behavior. Make this toggleable.

---

## 18. Build Order (implement in this sequence)

1. Config loading + project-dir resolution + the `.agent/` scaffold.
2. `llm/client.py` (LiteLLM wrapper with streaming) + `tools/sandbox.py` (with all safety layers) + `tools/registry.py` + `llm/tool_parser.py`.
3. **The subtask loop in isolation** — hardcode one subtask and one plan, and get IMPLEMENT → WRITE_TESTS → RUN → FIX cycling reliably against the sandbox. This is the riskiest piece; make it solid before anything upstream.
4. `context/` (builder, manifest, compressor).
5. The upstream stages (intake → requirements → stack → architect → sdd → task_planner) and the prompt templates.
6. `server/` event layer + `cli/renderer.py`.
7. `orchestrator/orchestrator.py` wiring the whole state machine together + resumability.
8. `reviewer.py` final pass.
9. `README.md`.

---

## 19. Deliverables / Acceptance

- A runnable tool: `python main.py "<prompt>"` (optionally `--config`, `--project-dir`, `--resume`) starts the server, runs the pipeline, and streams everything to the terminal.
- Per-phase model selection works via `config.yaml`.
- Given a simple prompt (e.g. "a todo app with a REST API and a small web UI"), it generates the full SDD suite, plans tasks/subtasks, and autonomously implements + tests them, producing a working app inside the project dir.
- No git is ever invoked. Nothing is written or executed outside the project dir.
- No frontend is built, but the SSE event layer fully exposes the pipeline's state and the LLM's streaming output, ready for a future UI.

Build it thoroughly and idiomatically. Prioritize a rock-solid subtask loop and a clean, well-documented event schema, since those are what everything else depends on.
