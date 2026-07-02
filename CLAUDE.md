# CLAUDE.md — AgenticCoder-6 / "AIForge"

Single-source guide for working on this project in any session. Read this first.

> **Project memory also applies** (loaded automatically): never run git here; Ollama
> thinking arrives via `reasoning_content`; tool lives in `agentic-coder/`, sandbox + `.venv`
> at the repo root; hardware is 16GB-VRAM AMD + 32GB RAM. This file expands on all of that.

---

## 1. What this is

**AIForge** (the code/README name; the directory is `agentic-coder/`) is a **local, cost-free,
"prompt-and-walk-away" autonomous coding pipeline**. It turns one natural-language prompt into a
fully implemented, tested application on disk, orchestrating **local LLMs via Ollama (through
LiteLLM)** as a **state machine**. Each phase uses a configurable model. It writes a full
spec/steering document suite (SDD), plans tasks→subtasks, then implements + tests each subtask in
an isolated conversation loop until everything is built and passing.

The whole product runs as **one FastAPI server** (`127.0.0.1:8765`) that serves the SSE event
stream, control endpoints, **and** the built web IDE (`ui/dist/`). The terminal CLI renderer and
the browser IDE are both pure consumers of `/events`.

### Hard constraints (never violate)
- **NO GIT, EVER.** The tool runs no git command; `git` is on the sandbox denylist so LLM phases
  can't either. Don't add `.gitignore` for the tool. (`is_git_repository: false` is intentional.)
- **Never operate outside the resolved project dir.** All generated-app file writes and shell
  commands are jailed to `sandbox/<slug>/`. `~`, `..` escapes, and absolute writes outside root
  are rejected (`workspace.py`, `tools/sandbox.py`).
- **The model requests; the orchestrator executes.** LLMs emit text tool calls; Python validates
  and runs them, then feeds results back. Models never run shell commands directly.
- **Durable state lives on disk under `<project>/.agent/`; conversations are ephemeral.**

### ⚠️ Do not edit generated code to "fix the tool"
Everything under `sandbox/<slug>/` (outside `.agent/`) is **LLM-generated app output**. Tool/pipeline
bugs are fixed in `agentic-coder/` only. When asked to fix "the error," fix the pipeline, not the
generated app.

---

## 2. Layout & where things live

```
AgenticCoder-6/                 ← repo root / launch dir / primary CWD (this CLAUDE.md is here)
├── .venv/                      ← Python env (shipped, deps installed)
├── testpropmt.txt              ← the test prompt (NOTE the misspelling: "propmt"); a Notes app
├── adoption_plan.md            ← design notes ported from a sibling tool ("codehamr")
├── sandbox/<slug>/             ← generated projects; <slug>/.agent/ = control dir, rest = app code
└── agentic-coder/              ← ALL TOOL SOURCE
    ├── main.py                 entry: starts server thread, waits /healthz, connects renderer, POSTs /start
    ├── config.py / config.yaml resolved AppConfig + the live config (models, num_ctx, keep_alive, limits)
    ├── context_budget.yaml     per-phase input-token budgets + reserve_for_output
    ├── workspace.py            project-dir resolution + path jail (PathEscapeError)
    ├── services.py             Services container (DI) + Progress + clean_doc(); PipelineCancelled
    ├── taskstore.py            tasks.json model: next_runnable(), status roll-up, resume reset
    ├── promptlib.py / tokens.py Jinja2 render + tiktoken(cl100k) estimation
    ├── prompts/*.j2            one template per phase (intake, requirements, …, planner, implementer, fix, reviewer)
    ├── server/                 events.py (EventBus + schema) · app.py (FastAPI/SSE + UI mount)
    ├── orchestrator/           states.py (enums/transitions) · orchestrator.py (driver) · subtask_loop.py (the heart)
    ├── stages/                 intake, requirements, stack_decider, architect, sdd_generator, task_planner,
    │                           planner (per-subtask), implementer (ephemeral convo), reviewer, common.py
    ├── context/                builder.py · compressor.py · manifest.py · loader.py · conversation.py (convo packing)
    ├── llm/                    client.py (LiteLLM streaming) · tool_parser.py (tolerant) · tool_router.py (salvage)
    ├── tools/                  registry.py (read/write/edit/run) · sandbox.py (safety) · process_manager.py (servers)
    ├── cli/renderer.py         live terminal UI (rich), pure /events consumer
    └── ui/                     React+TS+Vite IDE → build to ui/dist/ (served at /). Manual `npm run build`.
```

> **`agentic-coder.md` is the original build spec and is partly STALE.** Its example `config.yaml`
> shows `deepseek-r1:32b` / `qwen2.5-coder:14b`, `num_ctx 131072`, budgets `122880`. The **shipped**
> `config.yaml` + `config.py` defaults are different (see §4). Trust `config.yaml`/`config.py` over
> the spec for anything concrete. The spec is still the authority on *intent* and architecture.

---

## 3. Run / resume / debug

```bash
source .venv/bin/activate
python agentic-coder/main.py "a todo app with a REST API"     # inline prompt
python agentic-coder/main.py -f testpropmt.txt                # file prompt (multi-line; -f wins over inline)
python agentic-coder/main.py "…" --project-dir ./sandbox/foo  # pin output dir
python agentic-coder/main.py --resume --project-dir ./sandbox/foo   # resume (needs known project_dir)
python agentic-coder/main.py … --no-dump                      # disable .agent/llm_calls/ dumps
```
Paths are anchored on `main.py`'s location, so launch from the repo root **or** `agentic-coder/`;
output always lands in `AgenticCoder-6/sandbox/<slug>/`. Open `http://localhost:8765` for the IDE
(needs `ui/dist/`; the terminal renderer works without it). Requires `ollama serve` on
`localhost:11434` with the configured models pulled.

**Debug artifacts** under `<project>/.agent/`: `run.log` (JSONL of every event — the first place to
look), `llm_calls/<phase>_<ts>.md` (full prompt+response per call), `blocked.md` (exhausted subtasks),
plus the SDD suite + `tasks.json` + `manifest.json`.

---

## 4. Configuration (the part that actually bites)

### Models per phase (`config.yaml`, defaults mirrored in `config.py:DEFAULT_MODELS`)
| Phase(s) | Model | Size | Thinking (auto) |
|---|---|---|---|
| intake | `ollama/llama3.2:3b` | ~2GB | none |
| requirements, stack_decider, architect, sdd_generator, task_planner, **planner**, escalation | `ollama/qwen3.6:27b` | **~17GB** | `/think` |
| **implementer**, reviewer | `ollama/qwen3-coder:30b` | **~18GB** | `/think` (starts with "qwen3"!) |
| tool_caller (salvage/router) | `ollama/qwen2.5-coder:14b` | ~9GB | none |

> **`config.yaml` is the single source of truth for models** and overrides `config.py:DEFAULT_MODELS`,
> the README, and the (older) example in `agentic-coder.md`. There is **no `tester` phase** — the
> implementer writes its own tests in the same conversation (it has the implementation context).

- **Thinking** is auto-detected from the model name (`config.detect_thinking_mode`): `deepseek-r1`→
  auto `<think>`; name starts with `qwen3`/`qwq`→`/think`; else none. A coarse **`thinking:` block in
  config.yaml** (`coding:` / `planning:` = true/false/null) overrides that per group; a per-phase
  `model_options` entry overrides the toggle. Precedence: **model_options > thinking-toggle >
  auto-detect**. Shipped: `coding: true, planning: true` (quality over speed). Coding phases =
  {implementer, reviewer}; planning = {requirements, stack_decider, architect, sdd_generator,
  task_planner, planner, escalation}.
- **`num_ctx: 32768`** — sizes the **KV cache Ollama allocates at load time** → the dominant memory
  cost. Set the Ollama server's `OLLAMA_CONTEXT_LENGTH` ≥ this or it's silently capped.
- **`keep_alive: "5m"`** — how long Ollama keeps a model resident after a call; warms a *same-model*
  loop (the implement/fix loop) so it doesn't reload every step.
- **`evict_on_model_switch: true`** (the §9 fix) — never keep two *different* models resident: the
  client unloads the warm model before a different one loads (so 27B planner + 30B coder can't stack
  and OOM). A same model re-called stays warm. Set `false` only on a big-VRAM box.
- **tok/s** — the client emits `tokens_per_second` on `llm_complete`; the CLI renderer and the web UI
  (HeaderBar + per-block in the Thinking panel) show live + final throughput.
- `limits`: `sandbox_timeout 120`, `max_fix_retries 3`, `max_escalations 2`, `long_process_timeout 30`,
  `stream_idle_timeout 600`.
- `context_budget.yaml`: per-phase **input** budgets (all `32768` except `tool_caller 24576`,
  `intake 16384`), `reserve_for_output: 8192`. Budgets only split the window input/output; they do
  **not** change loaded memory (that's `num_ctx` + model size).

### Hardware reality (this box): AMD RX **6950XT 16GB VRAM** (ROCm) + Ryzen **5800X3D** + **32GB** RAM
No `nvidia-smi` because the GPU is AMD/ROCm. **Two of the ~17–18GB models cannot be co-resident**
(neither fits with another in 16GB VRAM; together their weights ~35GB approach the 48GB VRAM+RAM
ceiling once KV caches + ~8GB OS are added). Upstream reasoning calls take **10–45 min each** on
this box (see run.log durations) because they run `/think` and largely spill off the 16GB GPU.

---

## 5. Pipeline flow (`orchestrator/orchestrator.py`)

`main.py` → `POST /start` → `Orchestrator.start_async` runs `run()` in a **daemon worker thread**:

```
intake → requirements → stack_decider → architect → sdd_generator → task_planner
       → subtask_loop → reviewer
```
Each stage is a `with self._stage(...)` block emitting `stage_start`/`stage_end`, honoring
cooperative pause/cancel at boundaries. Stages 1–6 are single shaped LLM calls writing one `.agent/`
doc each (via `stages/common.generate_doc`, except intake/task_planner which parse specially).
`PipelineState` enum + `LEGAL_TRANSITIONS` in `states.py`. Project slug comes from **intake** when
`project_dir` is blank, so intake may run before the workspace exists (orchestrator writes
`project_brief.md` after).

- **Stack-agnostic until stack_decider; stack-specific after.** `stack.md` locks the stack so later
  phases can't drift/hallucinate technologies.
- **Resume** (`--resume`): requires a known `project_dir` with a `tasks.json`; resets any
  `in_progress` subtask → `pending`, jumps straight to subtask_loop.

---

## 6. The subtask loop (`orchestrator/subtask_loop.py`) — the heart

Two-level context model: **stateless across subtasks** (each rebuilds context from disk),
**one growing conversation within a subtask** (discarded the moment it passes). For each pending
subtask whose deps are all `done` (`TaskStore.next_runnable`, which also skips subtasks transitively
depending on a blocked one):

```
PLAN      planner.run() — LARGE model (planner=qwen3.6:27b), one fresh call, no convo.
            Context via ContextBuilder: steering (verbatim) + sdd/architecture + manifest
            + cat() of files this subtask touches/depends on. Output = exact plan (files,
            signatures, tests, exact test command) — NOT full code.
IMPLEMENT  Implementer(...).implement_and_write_tests() — small/coding model (implementer=
+WRITE      qwen3-coder:30b) drives an EPHEMERAL conversation, emitting write_file/edit_file/
TESTS       read_file/run tool calls. Writes files AND tests, then replies "DONE".
RUN        orchestrator runs the extracted test command in the sandbox → pass/fail.
on fail →  FIX (same convo, same model) × max_fix_retries(3)
            → ESCALATE (discard convo; escalation=qwen3.6:27b re-plans with FULL failure
              history; fresh convo; reset fixes) × max_escalations(2)
              → BLOCK (record to blocked.md; skip dependents)
```

Key implementer internals (`stages/implementer.py`): `_MAX_STEPS=48`, `_MAX_REASKS=3`,
`_MAX_CALLS_PER_REPLY=16`; `HARD_RULES` injected every call; **`pack_conversation`** trims the
growing convo newest-first to the phase budget (keeps the system prompt pinned so Ollama doesn't
front-truncate it); `DONE` only counts on a standalone line in a no-tool-call message (a file
containing "DONE" can't end the turn); if the reply carries tool intent in the wrong shape, it's
routed to `tool_caller` to salvage real calls (§8); soft "runaway"/"repeated-failure" nudges.

RUN details: `extract_test_command` looks in plan → `test_strategy` → `stack.md`; bare `pytest` is
rewritten to `python -m pytest` (importable root); pytest exit **5** ("no tests collected") is
treated as **pass** for scaffolds; when no test command exists, `_fallback_verification` installs a
declared dependency manifest (`pip install -r …` / `npm install`) or accepts on declared-file
presence — so non-testable scaffold subtasks don't burn the whole ladder.

---

## 7. Context system (`context/`)

- **builder.py** — assembles ordered `Block`s (steering → design docs → manifest → source files →
  the TASK instruction) into a system+user message pair, run through the compressor.
- **compressor.py** — priority bands (`P_CRITICAL=0` never compressed → `P_DISTANT_SRC=40`). When
  over budget, swaps the most-distant/largest compressible blocks for their one-line manifest
  summaries (never truncates mid-file), emits a `compression` event; raises `CompressionError` +
  `compression_failure` if it still won't fit.
- **manifest.py** — keeps `file_manifest.md` (annotated: `path — summary`) and `file-directory.txt`
  (filtered `ls -R`) in sync after every write; summaries persisted in `.agent/manifest.json`
  (survive resume). Excludes `node_modules/.git/.venv/dist/...`.
- **loader.py** — jailed reads of `.agent/` docs + source files.
- **conversation.py** — `pack_conversation` (newest-first budget packing for ephemeral convos) +
  `cap_tool_output` (collapse huge tool results to head+tail with a recovery marker). `PACK_HEADROOM
  0.95` because tiktoken under-counts code/JSON.

---

## 8. LLM + tools layer

- **`llm/client.py`** — `LLMClient.complete(phase, messages)`: picks model+thinking, appends
  `/think`·`/no_think`, calls `litellm.completion(..., stream=True, num_ctx, keep_alive, api_base,
  api_key="ollama")`. Splits **thinking vs output**: reads `reasoning_content`/`reasoning` from the
  delta (Path A) AND inline `<think>…</think>` via `ThinkSplitter` (Path B) → `llm_thinking_token`
  vs `llm_token` events. Streamed via a daemon thread + bounded queue with an **inter-frame idle
  timeout** (`stream_idle_timeout`, reset per chunk — tolerates slow prefill, trips only on a truly
  silent stream). Raises **`LLMError`** on any provider failure (this is where the §9 EOF surfaces:
  `client.py:152`). Optional dump to `.agent/llm_calls/`.
- **`llm/tool_parser.py`** — tolerant: `<tool_call>` tags, ```json fences, raw JSON in prose,
  trailing commas, **unescaped newlines/quotes inside `content`** (schema-aware `_salvage_tool_call`,
  which sets `salvaged=True` → registry attaches a "args may be truncated, rebuild with heredoc" note).
  `extract_all_tool_calls` returns every call in order. `extract_json` for tasks.json.
- **`llm/tool_router.py`** — `looks_like_tool_content()` heuristic + `salvage_calls()` which hands
  mis-shaped replies to the `tool_caller` model to re-express as clean tool calls (recovers narrated
  edits / multi-file dumps instead of losing the work).
- **`tools/registry.py`** — 4 tools: `read_file`, `write_file{path,content,summary}`,
  `edit_file{path,old_string(unique),new_string}`, `run{cmd,background?,timeout?,smoke?}`. Every
  dispatch emits `tool_call`+`tool_result`; records summaries to the manifest; caps display text.
- **`tools/sandbox.py`** — validate→run: working-dir jail; denylist (`rm -rf /`, `sudo`, `dd`,
  `mkfs`, fork bombs, `shutdown`, `chmod 777 /`, `curl|sh`, **all `git`**); per-command timeout
  (kills whole process group); network allowed **only** for recognized installs (else proxy env
  scrubbed — classification, not a hard packet filter). 60k-char output cap.
- **`tools/process_manager.py`** — `background:true` → start detached → health-check (port open /
  ready-log / survives grace) → optional smoke cmds → **always kill the group** → pass/fail.

---

## 9. ✅ RESOLVED — Ollama `{"error":"EOF"}` at the first IMPLEMENT call (model-stacking OOM)

**Was:** every run, all upstream phases succeeded and the first subtask `T001.1` planned fine
(`planner`=`qwen3.6:27b`), then the **implementer** (`qwen3-coder:30b`) died after ~27s with
`litellm.APIConnectionError: OllamaException - {"error":"EOF"}` (`client.py`).

**Root cause:** `keep_alive="5m"` kept the planner's 27B (~17GB) **resident** while the loop
immediately loaded the implementer's 30B (~18GB) — `subtask_loop._process_subtask` calls
`planner.run()` then `Implementer(...)` with no eviction between, and the client passed `keep_alive`
verbatim (there was **no unload logic at all**). On this **16GB-VRAM (AMD RX 6950XT / ROCm) + 32GB-RAM**
box two ~17–18GB models can't coexist; loading the second while the first was warm OOM-killed the
ROCm runner mid-load → the HTTP stream EOFs.

**Fix (implemented):** `LLMClient` now enforces **at most one model resident at a time**. It tracks
the warm model (`_resident_model`); when a call needs a *different* model it first unloads the warm
one via Ollama (`POST /api/generate {keep_alive:0}`, guarded by `/api/ps` so it never reloads-to-
unload), then loads the new one — so the 27B and 30B never stack. A model re-called consecutively
(the implement/fix loop) stays warm. `Orchestrator.run()` calls `client.unload_all()` in a `finally`
so nothing lingers after a run. Gated by **`evict_on_model_switch: true`** in `config.yaml` (set
`false` on a big-VRAM box to allow stacking). The models in config.yaml are unchanged — the same
27B/30B run, they just no longer stack. Covered by `tests/test_client_eviction.py`.

---

## 10. Server / events (`server/`)

- **EventBus** (`events.py`): orchestrator (worker thread) calls `emit()` synchronously; SSE endpoint
  (asyncio loop) consumes via per-subscriber `asyncio.Queue`; bridged with `call_soon_threadsafe`.
  Every event also appended to `run.log` as one JSON line **except** `llm_token`/`llm_thinking_token`
  (stream-only, to keep the log bounded); oversized values trimmed in the durable copy. Event schema:
  `{type, phase, data, timestamp}`. Types incl. stage_start/end, llm_request/token/thinking_token/
  complete, tool_call/result, file_written, test_run, subtask_start/done/failed, escalation, blocked,
  compression(_failure), pipeline_paused/resumed/complete, error, log.
- **Endpoints** (`app.py`): `GET /events` (SSE), `POST /start|/pause|/resume|/cancel`, `GET /status`,
  `GET /project/state` (rich snapshot for UI), `GET /project/manifest`, `GET /file?path=`,
  `GET /healthz`, `/` (UI). `/resume` is polymorphic: unpause if paused, else disk-resume. UI mounted
  last so API routes win.
- **`cli/renderer.py`** — rich `Live` view; pure `/events` consumer (holds no orchestration logic).

---

## 11. Conventions when editing

- Match the surrounding style: dense, purposeful module docstrings citing the spec section
  (`spec §N`) and, where relevant, the sibling tool `codehamr` whose patterns several modules port
  (`pack_conversation`, `cap_tool_output`, edit-whitespace hints, truncated-args recovery).
- Keep dependencies minimal (`requirements.txt`: litellm, fastapi, uvicorn, sse-starlette, jinja2,
  pyyaml, rich, httpx, tiktoken, jsonschema).
- Phases/budgets/limits are config-driven — add a new phase by adding its model + budget + template,
  not by hardcoding. Any omitted phase falls back to a documented default in `config.py`.
- After changing `ui/src/*`, rebuild (`cd agentic-coder/ui && npm run build`); `main.py` warns if
  `ui/dist/` is missing/stale but never auto-builds. Type-check with `node_modules/.bin/tsc --noEmit`
  (the build uses esbuild and won't fail on type errors).
- **Tests** live in `agentic-coder/tests/` (pytest). Run `cd agentic-coder && python -m pytest -q`
  for the fast functional suite (no LLM: parsers, sandbox, taskstore, compressor, conversation,
  thinking resolution, model eviction, throughput, test-command extraction). The end-to-end test
  (`test_e2e_small.py`) is opt-in — `AIFORGE_E2E=1` with `ollama serve` up — and uses **small models
  only** (never the 27B/30B). Add a test alongside any change to the fragile heuristics.
- Don't reintroduce git anywhere.
