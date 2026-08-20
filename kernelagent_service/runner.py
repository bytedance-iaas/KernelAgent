"""Claude Code, pi, and Codex subprocess runners used by the scheduler."""

from __future__ import annotations

import asyncio
import json
import os
import re
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


# asyncio.StreamReader defaults to a 64KB line-length limit; both CLIs can
# emit single JSON lines far larger than that (e.g. inlined SKILL.md content
# or long tool output), which would otherwise raise LimitOverrunError.
_STREAM_LIMIT = 10 * 1024 * 1024

_SKILL_BY_OPERATION = {
    Operation.PARSE: "ka-kernel-parser",
    Operation.GENERATE: "ka-kernel-gen",
    Operation.PROFILE: "ka-kernel-opt",
    Operation.DIAGNOSE: "ka-kernel-opt",
    Operation.OPTIMIZE: "ka-kernel-opt",
}


def build_skill_prompt(request: CreateTaskRequest, *, command_prefix: str = "/") -> str:
    """Build the initial prompt that invokes the right skill for the operation.

    ``command_prefix`` selects the skill syntax for the target agent: Claude
    Code uses ``/skill-name``, pi uses ``/skill:skill-name``, and Codex uses
    ``$skill-name``.
    """
    skill = _SKILL_BY_OPERATION[request.operation]
    uploaded = {item.path for item in request.files}

    primary = request.entrypoint
    if (
        primary is None
        and request.operation == Operation.GENERATE
        and "problem.py" in uploaded
    ):
        primary = "problem.py"
    if primary is None and len(request.files) == 1:
        primary = request.files[0].path

    if request.operation == Operation.PARSE:
        target = f"input/{primary}" if primary else "input"
        invocation = f"{command_prefix}{skill} {target}"
    elif request.operation == Operation.GENERATE:
        if primary:
            invocation = f"{command_prefix}{skill} input/{primary}"
        else:
            invocation = f"{command_prefix}{skill} {request.problem}"
    elif request.operation == Operation.PROFILE:
        invocation = f"{command_prefix}{skill} profile input"
    elif request.operation == Operation.DIAGNOSE:
        invocation = f"{command_prefix}{skill} diagnose input"
    else:
        invocation = f"{command_prefix}{skill} input"

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
        details.extend(
            ["", "Additional user instructions:", options.extra_instructions]
        )
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


class _SubprocessAgentRunner:
    """Shared process bookkeeping for CLI-based agent runners.

    Both ``ClaudeCodeRunner`` and ``PiRunner`` spawn one detached process
    group per task and need identical cancel/close/terminate handling; only
    command construction, environment setup, and event-stream parsing differ
    between the two CLIs.
    """

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._lock = asyncio.Lock()

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


class ClaudeCodeRunner(_SubprocessAgentRunner):
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
            "Read,Write,Edit,Grep,Glob,Bash,Skill",
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
                limit=_STREAM_LIMIT,
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
            exit_code = await asyncio.wait_for(
                consume_stdout(), timeout=timeout_seconds
            )
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
                error=stderr_tail.strip()
                or "Claude Code exited without a result event",
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


class CodexRunner(_SubprocessAgentRunner):
    """Run repository skills through ``codex exec`` and a Responses provider."""

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
        runtime_dir = workspace / ".codex-runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        schema_path = runtime_dir / "result-schema.json"
        output_path = runtime_dir / "final-output.json"
        schema_path.write_text(
            json.dumps(_RESULT_SCHEMA, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_config(runtime_dir)

        command = [
            self.settings.codex_command,
            "--ask-for-approval",
            "never",
            "exec",
            "--json",
            "--ephemeral",
            "--sandbox",
            self.settings.codex_sandbox,
            "--skip-git-repo-check",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "--cd",
            str(workspace),
            build_skill_prompt(request, command_prefix="$"),
        ]
        env = self._build_environment(workspace, request, gpu_id)

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=workspace,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                limit=_STREAM_LIMIT,
            )
        except (FileNotFoundError, OSError) as exc:
            return RunnerResult(success=False, exit_code=None, error=str(exc))

        async with self._lock:
            self._processes[task_id] = process

        session_id: str | None = None
        event_error: str | None = None
        stderr_tail = ""

        async def drain_stderr() -> None:
            nonlocal stderr_tail
            assert process.stderr is not None
            while line := await process.stderr.readline():
                text = line.decode("utf-8", errors="replace")
                stderr_tail = (stderr_tail + text)[-20_000:]
                await on_stderr(text)

        async def consume_stdout() -> int:
            nonlocal event_error, session_id
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
                if event.get("type") == "thread.started":
                    value = event.get("thread_id")
                    if isinstance(value, str):
                        session_id = value
                if event.get("type") in {"error", "turn.failed"}:
                    event_error = self._event_error(event)
            exit_code = await process.wait()
            await stderr_task
            return exit_code

        stderr_task = asyncio.create_task(drain_stderr())
        try:
            exit_code = await asyncio.wait_for(
                consume_stdout(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            await self._terminate(process)
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
            return RunnerResult(
                success=False,
                exit_code=process.returncode,
                error=f"Codex exceeded the {timeout_seconds}s task timeout",
                timed_out=True,
                session_id=session_id,
            )
        except asyncio.CancelledError:
            await self._terminate(process)
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
            raise
        finally:
            async with self._lock:
                self._processes.pop(task_id, None)

        output = self._read_output(output_path)
        semantic_success = output.get("success") is True
        success = exit_code == 0 and event_error is None and semantic_success
        error = None
        if not success:
            error = (
                event_error
                or stderr_tail.strip()
                or str(output.get("summary") or output.get("result") or "").strip()
                or f"Codex failed with exit code {exit_code}"
            )
        return RunnerResult(
            success=success,
            exit_code=exit_code,
            output=output,
            error=error,
            session_id=session_id,
        )

    def _write_config(self, runtime_dir: Path) -> None:
        base_url = self.settings.codex_model_base_url
        if base_url is None:
            base_url = self.settings.model_base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
        config = "\n".join(
            [
                f"model = {json.dumps(self.settings.model_name)}",
                'model_provider = "kernel_agent_sglang"',
                "model_supports_reasoning_summaries = false",
                "",
                "[model_providers.kernel_agent_sglang]",
                'name = "KernelAgent SGLang"',
                f"base_url = {json.dumps(base_url)}",
                'wire_api = "responses"',
                "requires_openai_auth = false",
                "",
            ]
        )
        (runtime_dir / "config.toml").write_text(config, encoding="utf-8")

    def _build_environment(
        self, workspace: Path, request: CreateTaskRequest, gpu_id: str
    ) -> dict[str, str]:
        env = os.environ.copy()
        for name in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CODEX_API_KEY",
            "OPENAI_API_KEY",
        ):
            env.pop(name, None)
        skill = _SKILL_BY_OPERATION[request.operation]
        runtime_dir = workspace / ".codex-runtime"
        env.update(
            {
                "CLAUDE_SKILL_DIR": str((self.settings.skills_dir / skill).resolve()),
                "CODEX_HOME": str(runtime_dir),
                "CUDA_VISIBLE_DEVICES": gpu_id,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_CACHE_PATH": str(runtime_dir / "cache" / "cuda"),
                "TRITON_CACHE_DIR": str(runtime_dir / "cache" / "triton"),
                "PYTHONUNBUFFERED": "1",
            }
        )
        return env

    @staticmethod
    def _read_output(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {"result": value}

    @staticmethod
    def _event_error(event: dict[str, Any]) -> str:
        value = event.get("error") or event.get("message") or event
        if isinstance(value, dict):
            message = value.get("message") or value.get("code")
            if message:
                return str(message)
            return json.dumps(value, ensure_ascii=False)
        return str(value)


_PI_PROVIDER_ID = "kernelagent-gateway"
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _pi_result_instructions() -> str:
    """Instruct pi to emit the structured result as a fenced JSON block.

    pi has no equivalent of Claude Code's ``--json-schema`` flag, so the
    schema is requested in-prompt and the reply is parsed for it afterward.
    """
    schema_json = json.dumps(_RESULT_SCHEMA, indent=2)
    return (
        "\n\nWhen you are done, end your final message with a fenced "
        "```json code block (and nothing after it) containing a single "
        "JSON object matching this schema:\n```json\n" + schema_json + "\n```"
    )


def _extract_result_json(text: str) -> dict[str, Any] | None:
    """Return the last fenced block in ``text`` that parses as a JSON object."""
    for candidate in reversed(_JSON_FENCE_RE.findall(text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _pi_assistant_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


class PiRunner(_SubprocessAgentRunner):
    """Runs tasks through the pi coding agent (``@earendil-works/pi-coding-agent``).

    pi speaks the Agent Skills standard and can load this repo's
    ``.claude/skills`` directly, so the same skills used by Claude Code
    drive the pi runs too. The model gateway is registered as a custom
    ``anthropic-messages`` provider via a per-task ``models.json`` (pi has
    no ``--base-url`` flag for arbitrary endpoints).
    """

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
        config_dir = self._write_pi_config(workspace)
        env = self._build_environment(workspace, gpu_id, config_dir)
        prompt = (
            build_skill_prompt(request, command_prefix="/skill:")
            + _pi_result_instructions()
        )
        command = [
            self.settings.pi_command,
            "--print",
            "--mode",
            "json",
            "--no-session",
            "--no-context-files",
            "--no-skills",
            "--skill",
            str(self.settings.skills_dir),
            "--tools",
            "read,write,edit,grep,find,bash",
            "--provider",
            _PI_PROVIDER_ID,
            "--model",
            self.settings.model_name,
            prompt,
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
                limit=_STREAM_LIMIT,
            )
        except (FileNotFoundError, OSError) as exc:
            return RunnerResult(success=False, exit_code=None, error=str(exc))

        async with self._lock:
            self._processes[task_id] = process

        last_assistant_message: dict[str, Any] | None = None
        agent_ended = False
        stderr_tail = ""

        async def drain_stderr() -> None:
            nonlocal stderr_tail
            assert process.stderr is not None
            while line := await process.stderr.readline():
                text = line.decode("utf-8", errors="replace")
                stderr_tail = (stderr_tail + text)[-20_000:]
                await on_stderr(text)

        async def consume_stdout() -> int:
            nonlocal last_assistant_message, agent_ended
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
                event_type = event.get("type")
                if event_type == "message_end":
                    message = event.get("message")
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        last_assistant_message = message
                elif event_type == "agent_end":
                    agent_ended = True
            exit_code = await process.wait()
            await stderr_task
            return exit_code

        stderr_task = asyncio.create_task(drain_stderr())
        try:
            exit_code = await asyncio.wait_for(
                consume_stdout(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            await self._terminate(process)
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
            return RunnerResult(
                success=False,
                exit_code=process.returncode,
                error=f"pi exceeded the {timeout_seconds}s task timeout",
                timed_out=True,
            )
        except asyncio.CancelledError:
            await self._terminate(process)
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
            raise
        finally:
            async with self._lock:
                self._processes.pop(task_id, None)

        if last_assistant_message is None:
            return RunnerResult(
                success=False,
                exit_code=exit_code,
                error=stderr_tail.strip() or "pi exited without an assistant message",
            )

        assistant_text = _pi_assistant_text(last_assistant_message)
        structured = _extract_result_json(assistant_text)
        stop_reason = last_assistant_message.get("stopReason")
        error_message = last_assistant_message.get("errorMessage")
        semantic_success = bool(structured) and structured.get("success", True) is True
        success = (
            exit_code == 0
            and agent_ended
            and stop_reason not in {"error", "aborted"}
            and not error_message
            and structured is not None
            and semantic_success
        )
        error = None
        if not success:
            error = (
                error_message
                or stderr_tail.strip()
                or (
                    "pi finished without producing the requested structured result"
                    if structured is None
                    else None
                )
                or f"pi failed with exit code {exit_code}"
            )
        return RunnerResult(
            success=success,
            exit_code=exit_code,
            output=structured or {"result": assistant_text},
            error=error,
        )

    def _write_pi_config(self, workspace: Path) -> Path:
        config_dir = workspace / ".pi-runtime"
        config_dir.mkdir(parents=True, exist_ok=True)
        models_config = {
            "providers": {
                _PI_PROVIDER_ID: {
                    "baseUrl": self.settings.model_base_url,
                    "api": "anthropic-messages",
                    "apiKey": self.settings.model_auth_token,
                    "models": [
                        {
                            "id": self.settings.model_name,
                            "name": self.settings.model_name,
                            "reasoning": True,
                            "input": ["text"],
                            "contextWindow": self.settings.pi_context_window,
                            "maxTokens": self.settings.pi_max_output_tokens,
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                        }
                    ],
                }
            }
        }
        (config_dir / "models.json").write_text(
            json.dumps(models_config), encoding="utf-8"
        )
        return config_dir

    def _build_environment(
        self, workspace: Path, gpu_id: str, config_dir: Path
    ) -> dict[str, str]:
        env = os.environ.copy()
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN", "OPENAI_API_KEY"):
            env.pop(name, None)
        env.update(
            {
                "PI_CODING_AGENT_DIR": str(config_dir),
                "PI_OFFLINE": "1",
                "PI_TELEMETRY": "0",
                "CUDA_VISIBLE_DEVICES": gpu_id,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTHONUNBUFFERED": "1",
            }
        )
        return env


def create_runner(settings: ServiceSettings) -> AgentRunner:
    if settings.agent == "claude":
        return ClaudeCodeRunner(settings)
    if settings.agent == "pi":
        return PiRunner(settings)
    if settings.agent == "codex":
        return CodexRunner(settings)
    raise ValueError(f"unsupported agent: {settings.agent}")
