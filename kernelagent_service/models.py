"""API and persistence models for the single-node task service."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

RunnerBackend = Literal["claude", "pi", "codex"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Operation(str, Enum):
    PARSE = "parse"
    GENERATE = "generate"
    PROFILE = "profile"
    DIAGNOSE = "diagnose"
    OPTIMIZE = "optimize"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    TIMED_OUT = "timed_out"
    LOST = "lost"


TERMINAL_STATUSES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELED,
    TaskStatus.TIMED_OUT,
    TaskStatus.LOST,
}


class InputFile(BaseModel):
    """One UTF-8 input file materialized below ``workspace/input``."""

    path: str = Field(min_length=1, max_length=512)
    content: str

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("input file paths must use '/' separators")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("input file path must be a normalized relative path")
        return path.as_posix()


class TaskOptions(BaseModel):
    kernel_language: Literal["triton", "tilelang", "cutedsl"] = "triton"
    target_platform: Literal["cuda", "xpu"] = "cuda"
    max_rounds: int = Field(default=5, ge=1, le=100)
    timeout_seconds: int | None = Field(default=None, ge=30, le=86_400)
    extra_instructions: str | None = Field(default=None, max_length=8_000)


class KernelGenerationRequest(BaseModel):
    """Public request for compiling a PyTorch reference into a GPU kernel."""

    pytorch_code: str = Field(
        min_length=1,
        max_length=500_000,
        description=(
            "KernelBench-style Python containing class Model and get_inputs()."
        ),
    )
    runner_backend: RunnerBackend = "claude"
    kernel_language: Literal["triton", "cutedsl"] = "triton"
    test_code: str | None = Field(
        default=None,
        max_length=500_000,
        description="Optional additional Python correctness test.",
    )
    max_rounds: int = Field(default=5, ge=1, le=100)
    timeout_seconds: int | None = Field(default=None, ge=30, le=86_400)
    extra_instructions: str | None = Field(default=None, max_length=8_000)

    @field_validator("pytorch_code")
    @classmethod
    def validate_pytorch_problem(cls, value: str) -> str:
        tree = cls._parse_python(value, "pytorch_code")
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        functions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing: list[str] = []
        if "Model" not in classes:
            missing.append("class Model")
        if "get_inputs" not in functions:
            missing.append("get_inputs()")
        if missing:
            raise ValueError(
                "pytorch_code must be KernelBench-style Python containing "
                + " and ".join(missing)
            )
        return value

    @field_validator("test_code")
    @classmethod
    def validate_test_code(cls, value: str | None) -> str | None:
        if value is not None:
            cls._parse_python(value, "test_code")
        return value

    def to_task_request(self) -> "CreateTaskRequest":
        files = [InputFile(path="problem.py", content=self.pytorch_code)]
        if self.test_code:
            files.append(InputFile(path="custom_test.py", content=self.test_code))
        return CreateTaskRequest(
            operation=Operation.GENERATE,
            runner_backend=self.runner_backend,
            problem=(
                "Generate a verified, high-performance GPU implementation of the "
                "uploaded PyTorch reference. Benchmark and refine it; do not use "
                "PyTorch as a fallback in the generated kernel."
            ),
            files=files,
            entrypoint="problem.py",
            options=TaskOptions(
                kernel_language=self.kernel_language,
                target_platform="cuda",
                max_rounds=self.max_rounds,
                timeout_seconds=self.timeout_seconds,
                extra_instructions=self.extra_instructions,
            ),
        )

    @staticmethod
    def _parse_python(value: str, field_name: str) -> ast.Module:
        try:
            return ast.parse(value)
        except SyntaxError as exc:
            location = f" at line {exc.lineno}" if exc.lineno else ""
            raise ValueError(
                f"{field_name} is not valid Python{location}: {exc.msg}"
            ) from exc


class CreateTaskRequest(BaseModel):
    operation: Operation = Operation.GENERATE
    runner_backend: RunnerBackend = "claude"
    problem: str | None = Field(
        default=None,
        max_length=200_000,
        description="Plain-text problem description, primarily for generate mode.",
    )
    files: list[InputFile] = Field(default_factory=list, max_length=64)
    entrypoint: str | None = Field(
        default=None,
        max_length=512,
        description="Path of the primary file relative to the input directory.",
    )
    options: TaskOptions = Field(default_factory=TaskOptions)

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return InputFile.validate_relative_path(value)

    @model_validator(mode="after")
    def validate_operation_inputs(self) -> "CreateTaskRequest":
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("input file paths must be unique")
        for path in paths:
            if any(other.startswith(f"{path}/") for other in paths):
                raise ValueError(f"input file path conflicts with a directory: {path}")
        if self.entrypoint is not None and self.entrypoint not in paths:
            raise ValueError("entrypoint must name one of the uploaded files")
        if not self.files and not self.problem:
            raise ValueError(
                "at least one input file or a problem description is required"
            )
        if self.operation == Operation.PARSE and not self.files:
            raise ValueError("parse requires at least one kernel source file")
        if self.operation in {
            Operation.PROFILE,
            Operation.DIAGNOSE,
            Operation.OPTIMIZE,
        }:
            required = {"input.py", "problem.py", "test.py"}
            top_level = {path for path in paths if "/" not in path}
            missing = required - top_level
            if missing:
                raise ValueError(
                    f"{self.operation.value} requires input.py, problem.py, and test.py; "
                    f"missing: {', '.join(sorted(missing))}"
                )
        return self


class Artifact(BaseModel):
    id: str
    name: str
    relative_path: str
    size_bytes: int
    sha256: str


class TaskRecord(BaseModel):
    id: str
    operation: Operation
    runner_backend: RunnerBackend = "claude"
    status: TaskStatus = TaskStatus.QUEUED
    stage: str | None = None
    gpu_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    event_count: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    artifacts: list[Artifact] = Field(default_factory=list)


class CreateTaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    status_url: str


class TaskEvent(BaseModel):
    sequence: int
    type: str
    payload: dict[str, Any]
    created_at: datetime = Field(default_factory=utc_now)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    runner_backends: list[RunnerBackend]
    queue_size: int
    queue_capacity: int
    gpu_workers: list[str]
    running_tasks: int
