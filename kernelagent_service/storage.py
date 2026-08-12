"""Filesystem persistence for task metadata, events, inputs, and artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from kernelagent_service.models import (
    Artifact,
    CreateTaskRequest,
    TaskEvent,
    TaskRecord,
)


class TaskStore:
    def __init__(
        self,
        root: Path,
        skills_dir: Path,
        *,
        max_artifact_bytes: int,
        max_artifacts: int,
    ) -> None:
        self.root = root.resolve()
        self.skills_dir = skills_dir.resolve()
        self.max_artifact_bytes = max_artifact_bytes
        self.max_artifacts = max_artifacts
        self.root.mkdir(parents=True, exist_ok=True)

    def task_dir(self, task_id: str) -> Path:
        return self.root / task_id

    def workspace(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "workspace"

    def create(self, record: TaskRecord, request: CreateTaskRequest) -> Path:
        task_dir = self.task_dir(record.id)
        workspace = task_dir / "workspace"
        input_dir = workspace / "input"
        task_dir.mkdir(parents=True, exist_ok=False)
        input_dir.mkdir(parents=True)
        (task_dir / "logs").mkdir()
        (workspace / ".claude-runtime").mkdir()

        project_claude_dir = workspace / ".claude"
        project_claude_dir.mkdir()
        skills_link = project_claude_dir / "skills"
        skills_link.symlink_to(self.skills_dir, target_is_directory=True)

        input_root = input_dir.resolve()
        for item in request.files:
            destination = input_dir / item.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            resolved = destination.resolve()
            if not resolved.is_relative_to(input_root):
                raise ValueError(f"input path escapes workspace: {item.path}")
            destination.write_text(item.content, encoding="utf-8")

        self._atomic_json(task_dir / "request.json", request.model_dump(mode="json"))
        self.save_record(record)
        return workspace

    def save_record(self, record: TaskRecord) -> None:
        self._atomic_json(
            self.task_dir(record.id) / "task.json",
            record.model_dump(mode="json"),
        )

    def load_all(self) -> list[tuple[TaskRecord, CreateTaskRequest]]:
        recovered: list[tuple[TaskRecord, CreateTaskRequest]] = []
        for task_file in sorted(self.root.glob("*/task.json")):
            request_file = task_file.parent / "request.json"
            if not request_file.is_file():
                continue
            try:
                record = TaskRecord.model_validate_json(task_file.read_text(encoding="utf-8"))
                request = CreateTaskRequest.model_validate_json(
                    request_file.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            recovered.append((record, request))
        return recovered

    def append_event(self, task_id: str, event: TaskEvent) -> None:
        event_file = self.task_dir(task_id) / "logs" / "events.jsonl"
        with event_file.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json())
            handle.write("\n")

    def append_stderr(self, task_id: str, text: str) -> None:
        stderr_file = self.task_dir(task_id) / "logs" / "stderr.log"
        with stderr_file.open("a", encoding="utf-8") as handle:
            handle.write(text)

    def read_events(self, task_id: str, *, after: int, limit: int) -> list[TaskEvent]:
        event_file = self.task_dir(task_id) / "logs" / "events.jsonl"
        if not event_file.is_file():
            return []
        events: list[TaskEvent] = []
        with event_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = TaskEvent.model_validate_json(line)
                except ValueError:
                    continue
                if event.sequence <= after:
                    continue
                events.append(event)
                if len(events) >= limit:
                    break
        return events

    def discover_artifacts(self, task_id: str) -> list[Artifact]:
        workspace = self.workspace(task_id).resolve()
        excluded_dirs = {".claude", ".claude-runtime", ".git", "__pycache__"}
        source_files = self._source_file_paths(task_id)
        artifacts: list[Artifact] = []
        for current, directories, files in os.walk(workspace, followlinks=False):
            directories[:] = [name for name in directories if name not in excluded_dirs]
            for name in sorted(files):
                if len(artifacts) >= self.max_artifacts:
                    return artifacts
                path = Path(current) / name
                if path.is_symlink() or not path.is_file():
                    continue
                size = path.stat().st_size
                if size > self.max_artifact_bytes:
                    continue
                relative = path.relative_to(workspace).as_posix()
                if relative in source_files:
                    continue
                artifacts.append(
                    Artifact(
                        id=uuid4().hex,
                        name=name,
                        relative_path=relative,
                        size_bytes=size,
                        sha256=self._sha256(path),
                    )
                )
        return artifacts

    def _source_file_paths(self, task_id: str) -> set[str]:
        """Return uploaded paths so generated files under input/ remain artifacts."""
        request_file = self.task_dir(task_id) / "request.json"
        try:
            request = CreateTaskRequest.model_validate_json(
                request_file.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return set()
        return {f"input/{item.path}" for item in request.files}

    def artifact_path(self, task_id: str, artifact: Artifact) -> Path:
        workspace = self.workspace(task_id).resolve()
        path = (workspace / artifact.relative_path).resolve()
        if not path.is_relative_to(workspace) or not path.is_file() or path.is_symlink():
            raise FileNotFoundError(artifact.relative_path)
        return path

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
