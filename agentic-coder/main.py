"""AIForge entry point (spec §19).

``python main.py "<prompt>"`` loads config, starts the FastAPI server (with the
SSE event layer + orchestrator) in a background thread, connects the CLI renderer
to the event stream, kicks off the pipeline, and streams everything to the
terminal until the run completes.

All paths are anchored on this file's location (never the CWD), so you can launch
it either way:

    python agentic-coder/main.py "<prompt>"     # from the repo root
    cd agentic-coder && python main.py "<prompt>"   # from the tool dir

For long, multi-line prompts that are awkward to paste/quote in a shell, put the
prompt in a file and pass ``-f``:

    python main.py -f prompt.txt

Flags:
  -f, --prompt-file PATH   read the build prompt from a file (multi-line OK)
  --config PATH            alternate config.yaml
  --budget PATH            deprecated (context budgets are derived from the
                           resolved model windows now); accepted and ignored
  --project-dir PATH       override the output project directory
  --resume                 resume an existing run (requires a known project_dir)
  --no-dump                disable per-call prompt/response dumps under .agent/llm_calls/
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from config import load_config
from server.app import create_app


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="aiforge", description="Local autonomous coding pipeline")
    p.add_argument("prompt", nargs="*", help="the natural-language build prompt")
    p.add_argument(
        "-f",
        "--prompt-file",
        dest="prompt_file",
        default=None,
        metavar="PATH",
        help="read the build prompt from a file (best for long, multi-line prompts)",
    )
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--budget", default=None, help="deprecated and ignored (budgets derive from resolved model windows)")
    p.add_argument("--project-dir", default=None, help="override the output project directory")
    p.add_argument("--resume", action="store_true", help="resume an existing run from disk")
    p.add_argument("--no-dump", action="store_true", help="disable LLM prompt/response dumps")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    prompt = " ".join(args.prompt).strip()
    if args.prompt_file:
        # A prompt file (relative to the CWD) wins over any inline words — it's the
        # way to feed a long multi-line prompt that can't be pasted/quoted sanely.
        fpath = Path(args.prompt_file).expanduser()
        if not fpath.is_file():
            print(f"error: prompt file not found: {fpath}", file=sys.stderr)
            return 2
        try:
            file_prompt = fpath.read_text(encoding="utf-8").strip()
        except OSError as exc:
            print(f"error: could not read prompt file {fpath}: {exc}", file=sys.stderr)
            return 2
        if not file_prompt:
            print(f"error: prompt file is empty: {fpath}", file=sys.stderr)
            return 2
        if prompt:
            print(f"note: ignoring inline prompt; using --prompt-file {fpath}", file=sys.stderr)
        prompt = file_prompt

    if not prompt and not args.resume:
        print("error: a prompt is required (pass a quoted string, -f <file>, or --resume).\n", file=sys.stderr)
        print("examples:", file=sys.stderr)
        print('  python main.py "a todo app with a REST API and a small web UI"', file=sys.stderr)
        print("  python main.py -f prompt.txt", file=sys.stderr)
        return 2

    if args.budget:
        print("note: --budget is deprecated and ignored (budgets derive from the resolved model windows)", file=sys.stderr)

    config = load_config(
        args.config,
        project_dir_override=args.project_dir,
        dump_llm_calls=False if args.no_dump else None,
    )

    _check_ui_build(config.tool_root)

    app = create_app(config)
    host, port = config.server.host, config.server.port
    connect_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    base_url = f"http://{connect_host}:{port}"

    server = _ServerThread(app, host, port)
    server.start()
    if not _wait_healthy(base_url):
        print(f"error: server did not come up at {base_url}", file=sys.stderr)
        server.shutdown()
        return 1

    # Import here so a missing optional dep doesn't break --help.
    from cli.renderer import Renderer

    renderer = Renderer(base_url)

    def on_connected() -> None:
        # Posted only after the SSE stream is established, so no events are missed.
        endpoint = "/resume" if args.resume else "/start"
        body = {"project_dir": args.project_dir} if args.resume else {"prompt": prompt}
        if args.project_dir and not args.resume:
            body["project_dir"] = args.project_dir
        try:
            r = httpx.post(f"{base_url}{endpoint}", json=body, timeout=30)
            if r.status_code >= 400:
                print(f"\nfailed to {endpoint}: {r.text}", file=sys.stderr)
        except httpx.HTTPError as exc:
            print(f"\nfailed to reach server: {exc}", file=sys.stderr)

    result: dict = {}
    try:
        result = renderer.run(on_connected=on_connected)
    except KeyboardInterrupt:
        print("\ncancelling run…")
        try:
            httpx.post(f"{base_url}/cancel", timeout=10)
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    finally:
        server.shutdown()

    # Only a clean pipeline.complete(result=done) is success; a cancelled run
    # (pipeline.cancelled carries no "result" the renderer records) and an
    # error both exit non-zero.
    return 0 if result.get("result") == "done" else 1


class _ServerThread(threading.Thread):
    """Runs uvicorn in a daemon thread with quiet logging."""

    def __init__(self, app, host: str, port: int):
        super().__init__(daemon=True, name="aiforge-server")
        cfg = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
        self._server = uvicorn.Server(cfg)

    def run(self) -> None:
        self._server.run()

    def shutdown(self) -> None:
        self._server.should_exit = True
        self.join(timeout=5)


def _check_ui_build(tool_root: Path) -> None:
    """Warn (don't auto-build) if the web UI is missing or looks stale.

    The IDE frontend is a manual ``npm run build`` step (spec §Phase2). The
    API/SSE server runs fine without it — the terminal renderer still works — so
    this only prints guidance; it never blocks startup or builds anything.
    """
    ui = Path(tool_root) / "ui"
    dist_index = ui / "dist" / "index.html"
    src = ui / "src"
    if not dist_index.is_file():
        print(
            "note: web UI not built — the browser IDE at http://localhost:8765 "
            "will be unavailable.\n"
            "      build it once with:  cd agentic-coder/ui && npm install && npm run build\n"
            "      (the terminal renderer below works regardless.)",
            file=sys.stderr,
        )
        return
    if src.is_dir():
        built = dist_index.stat().st_mtime
        newest_src = max((p.stat().st_mtime for p in src.rglob("*") if p.is_file()), default=0.0)
        if newest_src > built:
            print(
                "note: web UI sources changed since the last build — it may be stale.\n"
                "      rebuild with:  cd agentic-coder/ui && npm run build",
                file=sys.stderr,
            )


def _wait_healthy(base_url: str, timeout: float = 25.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/healthz", timeout=2)
            if r.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    return False


if __name__ == "__main__":
    raise SystemExit(main())
