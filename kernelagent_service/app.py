"""FastAPI application for the single-node KernelAgent task service."""

from __future__ import annotations

import mimetypes
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, Response

from kernelagent_service.config import ServiceSettings
from kernelagent_service.manager import (
    NoGpuWorkersError,
    QueueCapacityError,
    TaskManager,
    TaskNotFoundError,
)
from kernelagent_service.models import (
    CreateTaskRequest,
    CreateTaskResponse,
    HealthResponse,
    KernelGenerationRequest,
    RunnerBackend,
    TaskEvent,
    TaskRecord,
    TaskStatus,
)
from kernelagent_service.runner import AgentRunner, create_runner
from kernelagent_service.storage import TaskStore
from kernelagent_service.ui import render_task_ui


def create_app(
    settings: ServiceSettings | None = None,
    runner: AgentRunner | Mapping[RunnerBackend, AgentRunner] | None = None,
) -> FastAPI:
    settings = settings or ServiceSettings.from_env()
    backends: tuple[RunnerBackend, ...] = ("claude", "pi", "codex")
    if runner is None:
        runners = {
            backend: create_runner(replace(settings, agent=backend))
            for backend in backends
        }
    elif isinstance(runner, Mapping):
        runners = dict(runner)
    else:
        # A single injected runner remains useful for tests and embedding.
        runners = {backend: runner for backend in backends}
    missing_runners = set(backends) - runners.keys()
    if missing_runners:
        raise ValueError(
            f"missing task runners: {', '.join(sorted(missing_runners))}"
        )
    store = TaskStore(
        settings.runs_dir,
        settings.skills_dir,
        max_artifact_bytes=settings.max_artifact_bytes,
        max_artifacts=settings.max_artifacts,
    )
    manager = TaskManager(settings, runners, store)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()

    app = FastAPI(
        title="KernelAgent Task Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.task_manager = manager

    @app.get("/v1/ui", response_class=HTMLResponse, include_in_schema=False)
    async def task_ui() -> HTMLResponse:
        return HTMLResponse(
            content=render_task_ui(),
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'self'; "
                    "style-src 'self' 'unsafe-inline'; "
                    "script-src 'self' 'unsafe-inline'; "
                    "connect-src 'self'; "
                    "img-src 'self' data:"
                ),
            },
        )

    @app.get("/healthz", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if settings.gpu_ids else "degraded",
            runner_backends=list(backends),
            queue_size=manager.queue.qsize(),
            queue_capacity=settings.queue_capacity,
            gpu_workers=list(settings.gpu_ids),
            running_tasks=manager.running_count,
        )

    @app.post(
        "/v1/tasks",
        response_model=CreateTaskResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_task(
        payload: KernelGenerationRequest, request: Request
    ) -> CreateTaskResponse:
        try:
            record = await manager.create(payload.to_task_request())
        except QueueCapacityError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except NoGpuWorkersError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        status_url = str(request.url_for("get_task", task_id=record.id))
        return CreateTaskResponse(
            task_id=record.id,
            status=record.status,
            status_url=status_url,
        )

    @app.get("/v1/tasks", response_model=list[TaskRecord])
    async def list_tasks(
        task_statuses: list[TaskStatus] | None = Query(default=None, alias="status"),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1_000),
    ) -> list[TaskRecord]:
        statuses = set(task_statuses) if task_statuses else None
        return await manager.list(statuses=statuses, offset=offset, limit=limit)

    @app.get("/v1/tasks/{task_id}", response_model=TaskRecord, name="get_task")
    async def get_task(task_id: str) -> TaskRecord:
        try:
            return await manager.get(task_id)
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.post("/v1/tasks/{task_id}/cancel", response_model=TaskRecord)
    async def cancel_task(task_id: str) -> TaskRecord:
        try:
            return await manager.cancel(task_id)
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.get("/v1/tasks/{task_id}/events", response_model=list[TaskEvent])
    async def get_task_events(
        task_id: str,
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1_000),
    ) -> list[TaskEvent]:
        try:
            return await manager.read_events(task_id, after=after, limit=limit)
        except TaskNotFoundError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.get("/v1/tasks/{task_id}/artifacts/{artifact_id}")
    async def download_artifact(task_id: str, artifact_id: str) -> Response:
        try:
            artifact, path = await manager.artifact_path(task_id, artifact_id)
        except (TaskNotFoundError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="artifact not found") from exc
        media_type = (
            mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
        )
        return Response(
            content=path.read_bytes(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(artifact.name)}",
            },
        )

    return app


app = create_app()
