# AIForge — Local Autonomous Coding Pipeline

AIForge turns a single natural-language prompt into a fully implemented, tested,
working application on disk — a local, cost-free, "prompt and walk away" tool. It
orchestrates **local LLMs (via Ollama, through LiteLLM)** as a **state machine**:
each phase uses a configurable model, it generates a full spec/steering document
suite (SDD), plans tasks and subtasks, then implements and tests each subtask in
an isolated conversation loop until the whole app is built and passing.

This build is the **backend pipeline + a streaming SSE event layer + a terminal
CLI renderer + a VSCode-style web IDE**. Every piece of pipeline state is exposed
on the event stream, and the web UI is purely a consumer of `/events` plus a few
read endpoints — the same stream the CLI renderer uses.

The whole product runs as **one server**: FastAPI serves the API/SSE *and* the
built web UI (`ui/dist/`) at `http://localhost:8765`. Open that in a browser to
watch the pipeline live — file explorer, streaming file writes with syntax
highlighting, model thinking, tool calls, test results, and an event log.

> **No git, ever.** The tool never runs any git command, and `git` is on the
> sandbox denylist so the LLM-driven phases can't invoke it either.

> **Layout.** All tool source lives in `agentic-coder/`. Generated projects are
> written to `AgenticCoder-6/sandbox/<slug>/` (a sibling of `agentic-coder/`,
> never nested inside it). Paths are anchored on `main.py`'s own location, so the
> tool works no matter which directory you launch it from.

---

## How it works

```
intake → requirements → stack decision → architect → SDD+steering
       → task planning → [ subtask loop ] → final review
```

* **Two-level context model.** The pipeline is *stateless across subtasks*: each
  new subtask rebuilds its context from disk (steering, sdd, manifest, the source
  files it needs). A *single subtask* runs as one accumulating conversation
  (implement → write tests → run → fix) that is **discarded the moment it passes**
  — its only durable outputs are the files on disk and the updated manifest/task
  status.
* **The model requests, the orchestrator executes.** LLMs never run shell
  commands. They emit tool calls (`read_file` / `write_file` / `edit_file` /
  `run`); the Python orchestrator validates and runs them, then feeds results back.
* **Wrong format ≠ lost work.** When a coding model produces the right work in the
  wrong shape (narrated edits, several files dumped in code blocks, malformed tool
  JSON), a fast **tool-call router** (`llm/tool_router.py`) detects the tool intent
  and hands the raw text to the `tool_caller` model, which re-expresses it as clean
  tool calls the orchestrator executes — several files in one turn included.
* **Durable state lives on disk** under `<project>/.agent/` (the SDD suite,
  `tasks.json`, `file_manifest.md`, `run.log`, …). Conversations are ephemeral.

### The subtask loop (the heart)

For each pending subtask whose dependencies are done:

1. **PLAN** — the planner model writes an exact implementation plan: it describes
   the files, signatures, and tests to build (not the full code — that's step 2).
2. **IMPLEMENT + WRITE_TESTS** — the coding model explores as needed, then writes
   files and tests via tool calls (one growing conversation).
3. **RUN** — the orchestrator runs the test command and captures the result. A
   subtask that ships no test command is verified instead by installing its
   dependency manifest (e.g. `pip install -r requirements.txt`) or by confirming
   its declared files exist — so non-testable artifacts don't churn the ladder.
4. On failure: **FIX** (small model, same conversation) up to `max_fix_retries`,
   then **ESCALATE** (large model re-plans with the full failure history) up to
   `max_escalations`, then **BLOCK** (recorded to `blocked.md`, dependents skipped).

Every cap comes from `config.yaml` — the loop is never unbounded.

---

## Install

> All commands below assume you start from the repo root, `AgenticCoder-6/`.

### 1. Python deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r agentic-coder/requirements.txt
```

### 1b. Web UI (build once)

The browser IDE is a manual build step. You need Node 18+ and npm.

```bash
cd agentic-coder/ui
npm install
npm run build          # outputs agentic-coder/ui/dist/ (served by the backend)
cd ../..               # back to AgenticCoder-6/
```

The API/SSE server runs fine without this — the terminal renderer still works —
but `http://localhost:8765` will be empty until `ui/dist/` exists. `main.py`
prints a hint if the build is missing or stale. For UI development with
hot-reload run `npm run dev` (port 5173); it proxies the API to port 8765.

### 2. Ollama + models

Install Ollama from <https://ollama.com>, then pull the models named in
`config.yaml`. The shipped config targets a **16GB-VRAM / 32GB-RAM machine** with
the strongest models that fit (no 70b):

```bash
ollama pull qwen3.6:27b        # reasoning / planning phases (~17GB)
ollama pull qwen3-coder:30b    # coding + final review (~18GB)
ollama pull qwen2.5-coder:14b  # tool-call router / salvage (~9GB)
ollama pull llama3.2:3b        # intake (~2GB)
```

`num_ctx` is **32768** here (see `config.yaml`). `num_ctx` sizes the KV cache Ollama
allocates *at load time*, so it is the dominant memory cost — too large a window is
what makes a big model report *"model requires more system memory than is
available."* Make sure your Ollama server actually allows 32K so it isn't capped
lower (Ollama Desktop may default to 4K): set `OLLAMA_CONTEXT_LENGTH=32768` when you
start `ollama serve`, or bake it into a Modelfile (`PARAMETER num_ctx 32768`).

Make sure `ollama serve` is running (default `http://localhost:11434`).

### Memory / speed tuning (scale up or down)

Peak memory ≈ **one model + its KV cache at a time**, because the pipeline never
keeps two *different* models resident:

* **`evict_on_model_switch`** — `true` (default, right for ≤16GB VRAM) makes the
  tool **unload the warm model before a different one loads**, so a big swap (the
  27B planner → the 30B coder) can't stack and OOM-kill the Ollama runner (which
  surfaced as `OllamaException {"error":"EOF"}`). A model re-called consecutively
  (the implement/fix loop) stays warm; the load between *different* models is paid
  deliberately — fine for "code and walk away". Set `false` only if you upgrade to
  enough VRAM/RAM to hold two models at once.
* **`keep_alive`** — `"5m"` (default) warms a same-model loop so the implement/fix
  cycle doesn't reload weights every step. (With eviction on, this only affects the
  *same* model; different models are unloaded on switch regardless.)
* **`num_ctx`** — raise it (and the budgets in `context_budget.yaml`) for more
  context if you have memory; lower both together to shrink the KV cache.
* **model sizes** — for a **smaller box (8–16GB RAM)**, drop the reasoning model to
  `qwen3:8b`, the coder to `qwen2.5-coder:7b`, and `num_ctx` to 8192–16384. For
  **more VRAM/RAM**, raise `num_ctx` further.

Two server-side Ollama env vars cut KV-cache memory further (set them where you run
`ollama serve`): `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` (the
latter roughly halves the cache, letting you push `num_ctx` higher).

---

## Run

Paths are anchored on `main.py`'s own location (never the CWD), so you can launch
from **either** the repo root or the tool directory:

```bash
# from the repo root (AgenticCoder-6/)
source .venv/bin/activate
python agentic-coder/main.py "a todo app with a REST API and a small web UI"

# …or from the tool directory (agentic-coder/)
cd agentic-coder
source ../.venv/bin/activate
python main.py "a todo app with a REST API and a small web UI"
```

Either way the build output goes to `AgenticCoder-6/sandbox/<slug>/`.

```bash
# long / multi-line prompt: put it in a file and pass -f (avoids shell-quoting
# pain and lets you paste a big spec without it getting mangled)
python agentic-coder/main.py -f prompt.txt

# pin the output directory
python agentic-coder/main.py "a CLI calculator in Python" --project-dir ./sandbox/calc

# resume a run that was interrupted (needs a known project_dir)
python agentic-coder/main.py --resume --project-dir ./sandbox/calc
```

> `-f, --prompt-file PATH` reads the entire file (multi-line preserved) as the
> build prompt. The path is relative to your current directory; if both `-f` and
> an inline prompt are given, the file wins.

Then open **http://localhost:8765** in a browser for the full IDE view.

`main.py` starts the server (which also serves the web UI), connects the terminal
renderer to the SSE stream, then kicks off the pipeline — you'll see stages,
streaming model tokens (thinking vs output), **live tok/s throughput**, tool calls,
file writes, test results, and subtask progress live in both the terminal and the
browser.

### HTTP API (also drives the same pipeline + feeds the web UI)

| Method | Path                | Body                          | Purpose                                            |
| ------ | ------------------- | ----------------------------- | -------------------------------------------------- |
| GET    | `/events`           | —                             | SSE stream of all events                           |
| POST   | `/start`            | `{ "prompt", "project_dir?" }`| start a fresh pipeline run                         |
| POST   | `/pause`            | —                             | cooperatively pause (holds at the next boundary)   |
| POST   | `/resume`           | `{ "project_dir?" }`          | unpause a paused run, or disk-resume when idle     |
| POST   | `/cancel`           | —                             | cooperatively cancel the run                       |
| GET    | `/status`           | —                             | legacy state snapshot                              |
| GET    | `/project/state`    | —                             | rich snapshot (phase, progress, counts, timers)    |
| GET    | `/project/manifest` | —                             | current `file_manifest.md` (plain text)            |
| GET    | `/file?path=`       | —                             | content of a file inside the project root          |
| GET    | `/`                 | —                             | the built web UI (`ui/dist/`)                      |

**Pause / resume.** `/pause` sets a cooperative flag the worker honours at its
next atomic boundary — it finishes the current tool call or model stream, then
holds. `/resume` is polymorphic: if a run is paused it unpauses it (emitting
`pipeline_resumed`); if nothing is running it disk-resumes an existing project.
Pausing emits `pipeline_paused`; both flow over `/events`.

---

## Configuration

> **`config.yaml` is the single source of truth.** The models (and every setting)
> chosen there **override** everything else — `config.py`'s `DEFAULT_MODELS`, any
> example in this README, and the older example config in `agentic-coder.md`. Those
> are fallbacks/illustrations only. Change a phase's model **here**.

* **`config.yaml`** — the model for each phase (LiteLLM strings like
  `ollama/qwen3-coder:30b`), the `thinking:` coding/planning toggle + optional
  per-phase `model_options` overrides (thinking is auto-detected per model
  otherwise), `limits`, `server`, the `project_dir` (blank →
  `AgenticCoder-6/sandbox/<slug>/`, resolved relative to `main.py`'s location), and
  the memory knobs: `num_ctx` (KV-cache / context size), `keep_alive` (how long
  Ollama keeps a model warm), and **`evict_on_model_switch`** (keep at most one model
  resident — see below). Any phase you omit falls back to a documented default.
  There is **no `tester` phase** — the implementer writes its own tests in the same
  conversation (it has the implementation context).
* **`context_budget.yaml`** — per-phase input-token budgets and the
  `reserve_for_output`. When assembled context exceeds a budget, the compressor
  summarizes the most distant completed files to one-liners (from the manifest),
  emits a `compression` event documenting every decision, and never truncates
  mid-file.

### Thinking models (auto-detected)

Each phase's thinking mode is inferred from its model — you don't maintain a list:

* **qwen3 family** (qwen3, qwen3.5/3.6, qwen3-coder, qwq) get `/think` appended.
* **deepseek-r1** thinks automatically (`<think>…</think>`, no suffix).
* **everything else** (qwen2.5, llama3.x, …) runs with no thinking.

Change a model in `config.yaml` and its thinking follows automatically.

**Control thinking for coding vs planning** with the `thinking:` block in
`config.yaml`:

```yaml
thinking:
    coding: true     # implementer + reviewer (the coder phases)
    planning: true   # requirements/stack/architect/sdd/task_planner/planner/escalation
```

Each is `true` (force on where supported → qwen3 `/think`), `false` (force off →
qwen3 `/no_think`), or `null` (auto-detect). It ships **on for both** — quality over
speed. Precedence: a specific **`model_options`** entry > this **`thinking`** toggle
> **auto-detect**. So the coarse toggle is the easy knob; `model_options` is the
per-phase override when you need one.

The client detects `<think>` tags in the stream (even when split across chunks)
and emits them as separate `llm_thinking_token` events so thinking and output can
be rendered differently — they are never stripped.

---

## Safety (`tools/sandbox.py`)

Every model-requested command passes through, in order:

1. **Working-directory jail** — `cwd` is the resolved project root; `~`, `..`
   escapes, and absolute write targets outside the root are rejected.
2. **Command denylist** — `rm -rf /`, `sudo`, `dd`, `mkfs`, fork bombs,
   `shutdown`/`reboot`, `chmod -R 777 /`, `curl|wget | sh`, and **all `git`**.
3. **Timeout** — every command is killed (whole process group) after
   `limits.sandbox_timeout`.
4. **Network policy** — only commands recognized as dependency installs
   (`npm install`, `pip install`, …) are permitted to need the network; other
   steps run with proxy env scrubbed. *Note:* airtight socket isolation requires
   containers/namespaces, which the no-Docker constraint forbids — so this is a
   classification-and-scrub policy, not a hard packet filter.

Long-running targets (dev servers) go through the process harness: started in the
background, health-checked (port/ready-log/liveness), optionally smoke-tested,
then **always killed** (no orphans) and reduced to pass/fail.

---

## Project layout

```
AgenticCoder-6/                 ← repo root / launch dir (nothing tool-related here)
├── agentic-coder/              ← ALL tool source lives here
│   ├── main.py                 entry: server + orchestrator + renderer + UI host
│   ├── config.py / workspace.py  config loading + path jail
│   ├── taskstore.py            tasks.json model + scheduling
│   ├── services.py             shared dependency container (+ Progress, pause)
│   ├── promptlib.py / tokens.py  prompt rendering + token estimation
│   ├── server/                 events.py (EventBus) + app.py (FastAPI/SSE + UI mount)
│   ├── orchestrator/           states.py, orchestrator.py, subtask_loop.py
│   ├── stages/                 intake … task_planner, planner, implementer, reviewer
│   ├── context/                builder, loader, manifest, compressor
│   ├── llm/                    client.py (streaming) + tool_parser.py
│   ├── tools/                  registry.py, sandbox.py, process_manager.py
│   ├── prompts/                one Jinja2 template per phase
│   ├── cli/renderer.py         live SSE consumer (terminal)
│   └── ui/                     React + TS + Vite web IDE (build → ui/dist/)
│       ├── src/                store (Zustand), useEventStream hook, components
│       └── dist/               built bundle, served by the backend at /
└── sandbox/                    default output root (sibling of agentic-coder/)
    └── <slug>/.agent/          per-project control dir + generated app code
```

The per-project control directory is `.agent/` (tasks.json, the SDD suite,
`file_manifest.md`, `run.log`, `llm_calls/`, `blocked.md`).

---

## Tests

Functional tests of the tool itself live in `agentic-coder/tests/` (pytest):

```bash
cd agentic-coder
python -m pytest -q          # fast suite, no LLM required
```

They cover the fragile heuristics — the tolerant tool-call parser, the sandbox
denylist/path-jail, tasks.json scheduling, the context compressor, conversation
packing, thinking-mode resolution, the model-eviction policy, throughput, and
test-command extraction. The **end-to-end** test (`tests/test_e2e_small.py`) drives
the *whole* pipeline but is opt-in and uses **small models only** (never the 27B/30B):

```bash
AIFORGE_E2E=1 python -m pytest -q tests/test_e2e_small.py   # needs `ollama serve` + small models
```

## Debugging

* `<project>/.agent/run.log` — every event as one JSON line (also feeds a future
  frontend).
* `<project>/.agent/llm_calls/` — full prompt+response per call (toggle with
  `--no-dump` or `dump_llm_calls` in config).
* `<project>/.agent/blocked.md` — any subtask that exhausted fixes + escalations.
