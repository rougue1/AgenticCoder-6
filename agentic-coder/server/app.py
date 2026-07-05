"""FastAPI app: SSE event stream + control endpoints + UI host (spec §14, Phase 2).

Control / data endpoints:
* ``GET  /events``           — Server-Sent Events stream of every pipeline event.
* ``POST /start``            — body ``{prompt, project_dir?}`` — kick off a run.
* ``POST /resume``           — unpause a paused run, OR (when idle) resume an
                               existing on-disk project. Body ``{project_dir?}``.
* ``POST /pause``            — cooperatively pause the running pipeline.
* ``POST /cancel``           — cooperatively cancel the current run.
* ``GET  /status``           — legacy state snapshot.
* ``GET  /project/state``    — rich snapshot for the IDE frontend.
* ``GET  /project/manifest`` — the annotated Worker-file manifest (plain text).
* ``GET  /file?path=``       — content of a file inside the project root.
* ``GET  /healthz``          — liveness probe (used by main.py to wait for boot).

The built single-page UI (``ui/dist/``) is mounted at ``/`` *after* every API
route, so one server hosts the whole product. The CLI renderer and the web UI
both consume ``/events``; all orchestration state is exposed through the stream.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import AppConfig
from context.manifest import Manifest
from orchestrator.orchestrator import Orchestrator
from server.events import Event, EventBus
from workspace import PathEscapeError

_KEEPALIVE_SECONDS = 15


class StartRequest(BaseModel):
    prompt: str
    project_dir: str | None = None


class ResumeRequest(BaseModel):
    project_dir: str | None = None


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="AIForge", version="1.0")
    bus = EventBus()
    app.state.config = config
    app.state.bus = bus
    app.state.orchestrator = Orchestrator(config, bus)

    @app.on_event("startup")
    async def _bind_loop() -> None:
        bus.bind_loop(asyncio.get_running_loop())

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.get("/status")
    async def status() -> dict:
        return app.state.orchestrator.status()

    @app.post("/start")
    async def start(req: StartRequest) -> JSONResponse:
        orch: Orchestrator = app.state.orchestrator
        if orch.is_running():
            return JSONResponse({"ok": False, "error": "a run is already in progress"}, status_code=409)
        if req.project_dir is not None:
            config.project_dir = req.project_dir
        try:
            orch.start_async(req.prompt, resume=False)
        except RuntimeError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
        return JSONResponse({"ok": True})

    @app.post("/resume")
    async def resume(req: ResumeRequest) -> JSONResponse:
        """Polymorphic resume.

        * If the pipeline is running and paused -> **unpause** it.
        * If nothing is running -> **disk-resume** an existing project.
        * If running but not paused -> 409 (nothing to do).
        """
        orch: Orchestrator = app.state.orchestrator
        if orch.is_running():
            if orch.resume_pause():
                return JSONResponse({"ok": True, "action": "unpaused"})
            return JSONResponse(
                {"ok": False, "error": "a run is already in progress and not paused"},
                status_code=409,
            )
        if req.project_dir is not None:
            config.project_dir = req.project_dir
        try:
            orch.start_async("", resume=True)
        except RuntimeError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
        return JSONResponse({"ok": True, "action": "resumed"})

    @app.post("/pause")
    async def pause() -> JSONResponse:
        """Cooperatively pause the running pipeline (holds at the next boundary)."""
        orch: Orchestrator = app.state.orchestrator
        if not orch.is_running():
            return JSONResponse({"ok": False, "error": "no run in progress"}, status_code=409)
        if orch.pause():
            return JSONResponse({"ok": True, "action": "paused"})
        return JSONResponse({"ok": True, "action": "already_paused"})

    @app.post("/cancel")
    async def cancel() -> dict:
        app.state.orchestrator.cancel()
        return {"ok": True}

    @app.get("/project/state")
    async def project_state() -> dict:
        return app.state.orchestrator.project_state()

    @app.get("/project/manifest")
    async def project_manifest() -> PlainTextResponse:
        orch: Orchestrator = app.state.orchestrator
        ws = orch.active_workspace()
        if ws is None:
            return PlainTextResponse("")
        # Prefer the live manifest (in-memory + sidecar); fall back to a fresh
        # load for a not-yet-attached on-disk project.
        manifest = orch.services.manifest if orch.services.manifest is not None else Manifest(ws)
        return PlainTextResponse(manifest.render_markdown())

    @app.get("/file")
    async def get_file(path: str = Query(..., description="project-relative file path")) -> JSONResponse:
        orch: Orchestrator = app.state.orchestrator
        ws = orch.active_workspace()
        if ws is None:
            return JSONResponse({"content": "", "exists": False, "path": path})
        try:
            target = ws.resolve_in_root(path)
        except PathEscapeError as exc:
            return JSONResponse({"content": "", "exists": False, "error": str(exc)}, status_code=400)
        if not target.exists() or not target.is_file():
            return JSONResponse({"content": "", "exists": False, "path": path})
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return JSONResponse({"content": "", "exists": True, "error": str(exc)}, status_code=500)
        return JSONResponse({"content": content, "exists": True, "path": ws.relative(path)})

    @app.get("/events")
    async def events(request: Request) -> StreamingResponse:
        return StreamingResponse(
            _event_stream(bus, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    _mount_ui(app, config)
    return app


def _mount_ui(app: FastAPI, config: AppConfig) -> None:
    """Serve the built single-page UI from ``<tool_root>/ui/dist`` at ``/``.

    Mounted last so every API route above takes precedence. The directory is
    resolved relative to the tool root (never the CWD); if it hasn't been built
    yet the mount is skipped so the API/SSE server still boots — main.py prints a
    "run npm run build" hint in that case.
    """
    dist = Path(config.tool_root) / "ui" / "dist"
    if dist.is_dir() and (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="ui")


async def _event_stream(bus: EventBus, request: Request):
    queue = bus.subscribe()
    try:
        yield ": connected\n\n"
        while True:
            if await request.is_disconnected():
                break
            try:
                event: Event = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            yield _format_sse(event)
    finally:
        bus.unsubscribe(queue)


def _format_sse(event: Event) -> str:
    payload = json.dumps(event.to_dict(), default=str)
    return f"event: {event.type}\ndata: {payload}\n\n"
