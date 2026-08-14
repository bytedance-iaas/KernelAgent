"""FastAPI application for the single-node KernelAgent task service."""

from __future__ import annotations

import mimetypes
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import Response

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
    TaskEvent,
    TaskRecord,
    TaskStatus,
)
from kernelagent_service.runner import AgentRunner, ClaudeCodeRunner, PiRunner
from kernelagent_service.storage import TaskStore


def create_app(
    settings: ServiceSettings | None = None,
    runner: AgentRunner | None = None,
) -> FastAPI:
    settings = settings or ServiceSettings.from_env()
    if runner is None:
        runner = PiRunner(settings) if settings.agent == "pi" else ClaudeCodeRunner(settings)
    store = TaskStore(
        settings.runs_dir,
        settings.skills_dir,
        max_artifact_bytes=settings.max_artifact_bytes,
        max_artifacts=settings.max_artifacts,
    )
    manager = TaskManager(settings, runner, store)

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

    @app.get("/healthz", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if settings.gpu_ids else "degraded",
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
        media_type = mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
        return Response(
            content=path.read_bytes(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(artifact.name)}",
            },
        )

    return app


app = create_app()
