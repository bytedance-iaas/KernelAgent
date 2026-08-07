"""In-memory task queue and one-worker-per-GPU scheduler."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from kernelagent_service.config import ServiceSettings
from kernelagent_service.models import (
    TERMINAL_STATUSES,
    Artifact,
    CreateTaskRequest,
    TaskEvent,
    TaskRecord,
    TaskStatus,
    utc_now,
)
from kernelagent_service.runner import AgentRunner, RunnerResult
from kernelagent_service.storage import TaskStore


class QueueCapacityError(RuntimeError):
    pass


class NoGpuWorkersError(RuntimeError):
    pass


class TaskNotFoundError(KeyError):
    pass


class TaskManager:
    def __init__(
        self,
        settings: ServiceSettings,
        runner: AgentRunner,
        store: TaskStore,
    ) -> None:
        self.settings = settings
        self.runner = runner
        self.store = store
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=settings.queue_capacity)
        self.records: dict[str, TaskRecord] = {}
        self.requests: dict[str, CreateTaskRequest] = {}
        self.worker_tasks: list[asyncio.Task[None]] = []
        self._current_by_gpu: dict[str, str] = {}
        self._started = False
        self._lock = asyncio.Lock()
        self._submit_lock = asyncio.Lock()

    @property
    def running_count(self) -> int:
        return len(self._current_by_gpu)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        await self._recover()
        self.worker_tasks = [
            asyncio.create_task(self._gpu_worker(gpu_id), name=f"kernel-agent-gpu-{gpu_id}")
            for gpu_id in self.settings.gpu_ids
        ]

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        for worker in self.worker_tasks:
            worker.cancel()
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        await self.runner.close()

    async def create(self, request: CreateTaskRequest) -> TaskRecord:
        if not self.settings.gpu_ids:
            raise NoGpuWorkersError("no GPUs configured; set KERNEL_AGENT_GPU_IDS")
        input_bytes = len((request.problem or "").encode("utf-8")) + sum(
            len(item.content.encode("utf-8")) for item in request.files
        )
        if input_bytes > self.settings.max_input_bytes:
            raise ValueError(
                f"task input is {input_bytes} bytes; limit is {self.settings.max_input_bytes}"
            )
        async with self._submit_lock:
            if self.queue.full():
                raise QueueCapacityError("task queue is full")

            record = TaskRecord(id=str(uuid4()), operation=request.operation)
            self.store.create(record, request)
            async with self._lock:
                self.records[record.id] = record
                self.requests[record.id] = request
            self.queue.put_nowait(record.id)
        return record.model_copy(deep=True)

    async def get(self, task_id: str) -> TaskRecord:
        async with self._lock:
            record = self.records.get(task_id)
            if record is None:
                raise TaskNotFoundError(task_id)
            return record.model_copy(deep=True)

    async def list(
        self,
        *,
        statuses: set[TaskStatus] | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[TaskRecord]:
        async with self._lock:
            records = (
                record
                for record in self.records.values()
                if statuses is None or record.status in statuses
            )
            ordered = sorted(records, key=lambda record: record.created_at, reverse=True)
            return [record.model_copy(deep=True) for record in ordered[offset : offset + limit]]

    async def cancel(self, task_id: str) -> TaskRecord:
        async with self._lock:
            record = self.records.get(task_id)
            if record is None:
                raise TaskNotFoundError(task_id)
            if record.status in TERMINAL_STATUSES:
                return record.model_copy(deep=True)
            record.status = TaskStatus.CANCELED
            record.stage = "canceled"
            record.finished_at = utc_now()
            record.error = "task canceled by user"
            self._touch(record)
            result = record.model_copy(deep=True)
        await self.runner.cancel(task_id)
        return result

    async def read_events(self, task_id: str, *, after: int, limit: int) -> list[TaskEvent]:
        await self.get(task_id)
        return self.store.read_events(task_id, after=after, limit=limit)

    async def artifact_path(self, task_id: str, artifact_id: str) -> tuple[Artifact, Path]:
        record = await self.get(task_id)
        artifact = next((item for item in record.artifacts if item.id == artifact_id), None)
        if artifact is None:
            raise FileNotFoundError(artifact_id)
        return artifact, self.store.artifact_path(task_id, artifact)

    async def _recover(self) -> None:
        for record, request in self.store.load_all():
            if record.status == TaskStatus.QUEUED:
                self.records[record.id] = record
                self.requests[record.id] = request
                try:
                    self.queue.put_nowait(record.id)
                except asyncio.QueueFull:
                    record.status = TaskStatus.FAILED
                    record.error = "queue capacity exceeded during service recovery"
                    record.finished_at = utc_now()
                    self._touch(record)
            elif record.status == TaskStatus.RUNNING:
                record.status = TaskStatus.LOST
                record.stage = "recovery"
                record.error = "service restarted while task was running"
                record.finished_at = utc_now()
                self._touch(record)
                self.records[record.id] = record
                self.requests[record.id] = request
            else:
                self.records[record.id] = record
                self.requests[record.id] = request

    async def _gpu_worker(self, gpu_id: str) -> None:
        while True:
            task_id = await self.queue.get()
            try:
                async with self._lock:
                    record = self.records[task_id]
                    if record.status == TaskStatus.CANCELED:
                        continue
                    record.status = TaskStatus.RUNNING
                    record.stage = "starting"
                    record.gpu_id = gpu_id
                    record.started_at = utc_now()
                    record.error = None
                    self._current_by_gpu[gpu_id] = task_id
                    self._touch(record)
                    request = self.requests[task_id]

                timeout = request.options.timeout_seconds or self.settings.default_timeout_seconds
                result = await self.runner.run(
                    task_id=task_id,
                    request=request,
                    workspace=self.store.workspace(task_id),
                    gpu_id=gpu_id,
                    timeout_seconds=timeout,
                    on_event=lambda event: self._record_event(task_id, event),
                    on_stderr=lambda text: self._record_stderr(task_id, text),
                )
                await self._complete(task_id, result)
            except asyncio.CancelledError:
                await self.runner.cancel(task_id)
                async with self._lock:
                    record = self.records.get(task_id)
                    if record is not None and record.status == TaskStatus.RUNNING:
                        record.status = TaskStatus.LOST
                        record.stage = "shutdown"
                        record.error = "service stopped while task was running"
                        record.finished_at = utc_now()
                        self._touch(record)
                raise
            except Exception as exc:
                async with self._lock:
                    record = self.records.get(task_id)
                    if record is not None and record.status != TaskStatus.CANCELED:
                        record.status = TaskStatus.FAILED
                        record.stage = "failed"
                        record.error = f"worker error: {exc}"
                        record.finished_at = utc_now()
                        self._touch(record)
            finally:
                self._current_by_gpu.pop(gpu_id, None)
                self.queue.task_done()

    async def _complete(self, task_id: str, result: RunnerResult) -> None:
        artifacts = self.store.discover_artifacts(task_id)
        async with self._lock:
            record = self.records[task_id]
            if record.status == TaskStatus.CANCELED:
                return
            record.artifacts = artifacts
            record.result = {
                "success": result.success,
                "exit_code": result.exit_code,
                "session_id": result.session_id,
                "output": result.output,
            }
            record.error = result.error
            record.finished_at = utc_now()
            if result.timed_out:
                record.status = TaskStatus.TIMED_OUT
                record.stage = "timed_out"
            elif result.success:
                record.status = TaskStatus.SUCCEEDED
                record.stage = "completed"
            else:
                record.status = TaskStatus.FAILED
                record.stage = "failed"
            self._touch(record)

    async def _record_event(self, task_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            record = self.records[task_id]
            record.event_count += 1
            event_type = str(payload.get("type", "event"))
            if event_type in {"assistant", "tool_use", "tool_result", "result"}:
                record.stage = event_type
            event = TaskEvent(
                sequence=record.event_count,
                type=event_type,
                payload=payload,
            )
            self.store.append_event(task_id, event)
            self._touch(record)

    async def _record_stderr(self, task_id: str, text: str) -> None:
        self.store.append_stderr(task_id, text)

    def _touch(self, record: TaskRecord) -> None:
        record.updated_at = utc_now()
        self.store.save_record(record)
