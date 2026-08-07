"""Claude Code subprocess runner and runner interface used by the scheduler."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from kernelagent_service.config import ServiceSettings
from kernelagent_service.models import CreateTaskRequest, Operation


EventCallback = Callable[[dict[str, Any]], Awaitable[None]]
StderrCallback = Callable[[str], Awaitable[None]]


@dataclass
class RunnerResult:
    success: bool
    exit_code: int | None
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    timed_out: bool = False
    session_id: str | None = None


class AgentRunner(Protocol):
    async def run(
        self,
        *,
        task_id: str,
        request: CreateTaskRequest,
        workspace: Path,
        gpu_id: str,
        timeout_seconds: int,
        on_event: EventCallback,
        on_stderr: StderrCallback,
    ) -> RunnerResult: ...

    async def cancel(self, task_id: str) -> bool: ...

    async def close(self) -> None: ...


_SKILL_BY_OPERATION = {
    Operation.PARSE: "ka-kernel-parser",
    Operation.GENERATE: "ka-kernel-gen",
    Operation.PROFILE: "ka-kernel-opt",
    Operation.DIAGNOSE: "ka-kernel-opt",
    Operation.OPTIMIZE: "ka-kernel-opt",
}


def build_skill_prompt(request: CreateTaskRequest) -> str:
    skill = _SKILL_BY_OPERATION[request.operation]
    uploaded = {item.path for item in request.files}

    primary = request.entrypoint
    if primary is None and request.operation == Operation.GENERATE and "problem.py" in uploaded:
        primary = "problem.py"
    if primary is None and len(request.files) == 1:
        primary = request.files[0].path

    if request.operation == Operation.PARSE:
        target = f"input/{primary}" if primary else "input"
        invocation = f"/{skill} {target}"
    elif request.operation == Operation.GENERATE:
        if primary:
            invocation = f"/{skill} input/{primary}"
        else:
            invocation = f"/{skill} {request.problem}"
    elif request.operation == Operation.PROFILE:
        invocation = f"/{skill} profile input"
    elif request.operation == Operation.DIAGNOSE:
        invocation = f"/{skill} diagnose input"
    else:
        invocation = f"/{skill} input"

    options = request.options
    details = [
        invocation,
        "",
        "Service execution constraints:",
        "- Work only inside the current task workspace.",
        "- Do not ask the user interactive questions; make safe, documented assumptions.",
        f"- Kernel language: {options.kernel_language}.",
        f"- Target platform: {options.target_platform}.",
        f"- Maximum rounds/iterations: {options.max_rounds}.",
        "- Verify correctness before reporting success.",
        "- Optimize and benchmark the generated kernel; correctness alone is not enough.",
        "- The final GPU kernel must not call PyTorch as a fallback implementation.",
        "- Finish with a concise structured summary and paths relative to this workspace.",
    ]
    if "custom_test.py" in uploaded:
        details.append(
            "- Run input/custom_test.py as an additional correctness test when applicable."
        )
    if request.problem and primary:
        details.extend(["", "Additional problem description:", request.problem])
    if options.extra_instructions:
        details.extend(["", "Additional user instructions:", options.extra_instructions])
    return "\n".join(details)


_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "summary": {"type": "string"},
        "skill": {"type": "string"},
        "route": {"type": ["string", "null"]},
        "kernel_path": {"type": ["string", "null"]},
        "rounds": {"type": ["integer", "null"]},
        "metrics": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["success", "summary", "skill"],
    "additionalProperties": True,
}


class ClaudeCodeRunner:
    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._lock = asyncio.Lock()

    async def run(
        self,
        *,
        task_id: str,
        request: CreateTaskRequest,
        workspace: Path,
        gpu_id: str,
        timeout_seconds: int,
        on_event: EventCallback,
        on_stderr: StderrCallback,
    ) -> RunnerResult:
        env = self._build_environment(workspace, gpu_id)
        command = [
            self.settings.claude_command,
            "-p",
            "--verbose",
            "--output-format",
            "stream-json",
            "--permission-mode",
            "dontAsk",
            "--tools",
            "Read,Write,Edit,Grep,Glob,Bash,Skill",
            "--allowedTools",
            "Read,Write,Edit,Grep,Glob,Bash",
            "--no-session-persistence",
            "--json-schema",
            json.dumps(_RESULT_SCHEMA, separators=(",", ":")),
            build_skill_prompt(request),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=workspace,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except (FileNotFoundError, OSError) as exc:
            return RunnerResult(success=False, exit_code=None, error=str(exc))

        async with self._lock:
            self._processes[task_id] = process

        last_result: dict[str, Any] | None = None
        stderr_tail = ""

        async def drain_stderr() -> None:
            nonlocal stderr_tail
            assert process.stderr is not None
            while line := await process.stderr.readline():
                text = line.decode("utf-8", errors="replace")
                stderr_tail = (stderr_tail + text)[-20_000:]
                await on_stderr(text)

        async def consume_stdout() -> int:
            nonlocal last_result
            assert process.stdout is not None
            while line := await process.stdout.readline():
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                try:
                    event = json.loads(text)
                    if not isinstance(event, dict):
                        event = {"type": "stdout", "value": event}
                except json.JSONDecodeError:
                    event = {"type": "stdout", "text": text}
                await on_event(event)
                if event.get("type") == "result":
                    last_result = event
            exit_code = await process.wait()
            await stderr_task
            return exit_code

        stderr_task = asyncio.create_task(drain_stderr())
        try:
            exit_code = await asyncio.wait_for(consume_stdout(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            await self._terminate(process)
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
            return RunnerResult(
                success=False,
                exit_code=process.returncode,
                error=f"Claude Code exceeded the {timeout_seconds}s task timeout",
                timed_out=True,
                session_id=(last_result or {}).get("session_id"),
            )
        except asyncio.CancelledError:
            await self._terminate(process)
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
            raise
        finally:
            async with self._lock:
                self._processes.pop(task_id, None)

        if last_result is None:
            return RunnerResult(
                success=False,
                exit_code=exit_code,
                error=stderr_tail.strip() or "Claude Code exited without a result event",
            )

        structured = last_result.get("structured_output")
        if not isinstance(structured, dict):
            structured = {}
        result_subtype = str(last_result.get("subtype", ""))
        semantic_success = structured.get("success", True) is True
        success = (
            exit_code == 0
            and not bool(last_result.get("is_error"))
            and not result_subtype.startswith("error")
            and result_subtype != "failure"
            and semantic_success
        )
        error = None
        if not success:
            error = (
                str(last_result.get("error") or last_result.get("result") or "").strip()
                or stderr_tail.strip()
                or f"Claude Code failed with exit code {exit_code}"
            )
        return RunnerResult(
            success=success,
            exit_code=exit_code,
            output=structured or {"result": last_result.get("result")},
            error=error,
            session_id=last_result.get("session_id"),
        )

    async def cancel(self, task_id: str) -> bool:
        async with self._lock:
            process = self._processes.get(task_id)
        if process is None:
            return False
        await self._terminate(process)
        return True

    async def close(self) -> None:
        async with self._lock:
            processes = list(self._processes.values())
        await asyncio.gather(*(self._terminate(process) for process in processes))

    def _build_environment(self, workspace: Path, gpu_id: str) -> dict[str, str]:
        env = os.environ.copy()
        for name in (
            "ANTHROPIC_API_KEY",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_VERTEX",
            "CLAUDE_CODE_USE_FOUNDRY",
        ):
            env.pop(name, None)
        env.update(
            {
                "ANTHROPIC_BASE_URL": self.settings.model_base_url,
                "ANTHROPIC_AUTH_TOKEN": self.settings.model_auth_token,
                "ANTHROPIC_MODEL": self.settings.model_name,
                "CUDA_VISIBLE_DEVICES": gpu_id,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CLAUDE_CONFIG_DIR": str(workspace / ".claude-runtime"),
                "PYTHONUNBUFFERED": "1",
            }
        )
        return env

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(
                process.wait(), timeout=self.settings.shutdown_grace_seconds
            )
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            await process.wait()
