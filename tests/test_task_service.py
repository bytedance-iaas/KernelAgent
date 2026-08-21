"""Tests for the single-process task service without a GPU or model server."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx

from kernelagent_service.app import create_app
from kernelagent_service.config import ServiceSettings
from kernelagent_service.models import CreateTaskRequest, TaskRecord, TaskStatus
from kernelagent_service.runner import (
    ClaudeCodeRunner,
    CodexRunner,
    PiRunner,
    RunnerResult,
    build_skill_prompt,
    create_runner,
)
from kernelagent_service.storage import TaskStore

PYTORCH_CODE = """\
import torch
from torch import nn


class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)


def get_inputs():
    return [torch.randn(1024, device="cuda")]


def get_init_inputs():
    return []
"""


class FakeRunner:
    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.running = 0
        self.max_running = 0
        self.calls: list[tuple[str, str]] = []
        self.requests: list[CreateTaskRequest] = []
        self.canceled: set[str] = set()

    async def run(self, **kwargs) -> RunnerResult:
        task_id = kwargs["task_id"]
        gpu_id = kwargs["gpu_id"]
        workspace = kwargs["workspace"]
        self.calls.append((task_id, gpu_id))
        self.requests.append(kwargs["request"])
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            await kwargs["on_event"]({"type": "assistant", "message": "working"})
            await asyncio.sleep(self.delay)
            output = workspace / "output"
            output.mkdir(exist_ok=True)
            (output / "kernel.py").write_text(
                "def kernel_function(): pass\n", encoding="utf-8"
            )
            return RunnerResult(
                success=True,
                exit_code=0,
                output={"success": True, "summary": "done", "skill": "fake"},
            )
        finally:
            self.running -= 1

    async def cancel(self, task_id: str) -> bool:
        self.canceled.add(task_id)
        return True

    async def close(self) -> None:
        return None


class InputArtifactRunner(FakeRunner):
    async def run(self, **kwargs) -> RunnerResult:
        workspace = kwargs["workspace"]
        (workspace / "input" / "optimized_kernel.py").write_text(
            "def kernel_function(): pass\n", encoding="utf-8"
        )
        return RunnerResult(
            success=True,
            exit_code=0,
            output={"success": True, "summary": "optimized", "skill": "fake"},
        )


class PiRuntimeConfigRunner(FakeRunner):
    """Mimics PiRunner writing its per-task models.json (with the auth token)."""

    async def run(self, **kwargs) -> RunnerResult:
        workspace = kwargs["workspace"]
        pi_runtime = workspace / ".pi-runtime"
        pi_runtime.mkdir(exist_ok=True)
        (pi_runtime / "models.json").write_text(
            '{"providers": {"kernelagent-gateway": {"apiKey": "secret-token"}}}',
            encoding="utf-8",
        )
        return RunnerResult(
            success=True,
            exit_code=0,
            output={"success": True, "summary": "done", "skill": "fake"},
        )


def make_settings(
    tmp_path: Path, gpu_ids: tuple[str, ...] = ("GPU-test",)
) -> ServiceSettings:
    repo_root = Path(__file__).resolve().parent.parent
    return ServiceSettings(
        repo_root=repo_root,
        runs_dir=tmp_path / "runs",
        skills_dir=repo_root / ".claude" / "skills",
        gpu_ids=gpu_ids,
        queue_capacity=10,
        default_timeout_seconds=30,
        authentication_enabled=False,
        users_file=tmp_path / "users.json",
    )


def make_auth_settings(tmp_path: Path) -> ServiceSettings:
    return replace(
        make_settings(tmp_path),
        authentication_enabled=True,
        users_file=tmp_path / "users.json",
        admin_username="test_admin",
        admin_password="test-admin-password",
    )


async def submit(
    client: httpx.AsyncClient,
    label: str = "test",
    kernel_language: str = "triton",
    runner_backend: str = "claude",
) -> str:
    response = await client.post(
        "/v1/tasks",
        json={
            "pytorch_code": f"{PYTORCH_CODE}\n# {label}\n",
            "kernel_language": kernel_language,
            "runner_backend": runner_backend,
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["task_id"]


async def wait_for_terminal(
    client: httpx.AsyncClient, task_id: str, timeout: float = 3
) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        payload = (await client.get(f"/v1/tasks/{task_id}")).json()
        if payload["status"] in {
            "succeeded",
            "failed",
            "canceled",
            "timed_out",
            "lost",
        }:
            return payload
        await asyncio.sleep(0.01)
    raise AssertionError(f"task {task_id} did not finish")


def test_submit_run_query_and_download_artifact(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = FakeRunner()
        app = create_app(make_settings(tmp_path), runner)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                task_id = await submit(client)
                task = await wait_for_terminal(client, task_id)

                assert task["status"] == "succeeded"
                assert task["runner_backend"] == "claude"
                assert task["gpu_id"] == "GPU-test"
                assert task["result"]["output"]["summary"] == "done"
                assert task["event_count"] == 1
                assert len(task["artifacts"]) == 1
                assert runner.requests[0].entrypoint == "problem.py"
                assert runner.requests[0].options.kernel_language == "triton"
                assert runner.requests[0].files[0].content.startswith("import torch")

                events = (await client.get(f"/v1/tasks/{task_id}/events")).json()
                assert events[0]["type"] == "assistant"

                artifact = task["artifacts"][0]
                download = await client.get(
                    f"/v1/tasks/{task_id}/artifacts/{artifact['id']}"
                )
                assert download.status_code == 200
                assert b"kernel_function" in download.content

    asyncio.run(scenario())


def test_each_task_selects_its_harness_runner(tmp_path: Path) -> None:
    async def scenario() -> None:
        runners = {
            "claude": FakeRunner(),
            "pi": FakeRunner(),
            "codex": FakeRunner(),
        }
        app = create_app(make_settings(tmp_path), runners)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                task_ids = {
                    backend: await submit(client, backend, runner_backend=backend)
                    for backend in runners
                }
                for backend, task_id in task_ids.items():
                    task = await wait_for_terminal(client, task_id)
                    assert task["status"] == "succeeded"
                    assert task["runner_backend"] == backend

        assert len(runners["claude"].calls) == 1
        assert len(runners["pi"].calls) == 1
        assert len(runners["codex"].calls) == 1

    asyncio.run(scenario())


def test_task_ui_is_served_with_dashboard_and_security_headers(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = create_app(make_settings(tmp_path), FakeRunner())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/v1/ui")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert response.headers["cache-control"] == "no-store"
        assert "default-src 'self'" in response.headers["content-security-policy"]
        assert "<title>Anvil</title>" in response.text
        assert '<div class="brand-name">Anvil</div>' in response.text
        assert "创建 Kernel 任务" in response.text
        assert 'id="runner"' in response.text
        assert "runner_backend: $('runner').value" in response.text
        assert "fetch(path" in response.text
        assert "/v1/tasks" in response.text

    asyncio.run(scenario())


def test_auth_signup_login_and_role_access(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = create_app(make_auth_settings(tmp_path), FakeRunner())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as anonymous:
            ui = await anonymous.get("/v1/ui", follow_redirects=False)
            assert ui.status_code == 303
            assert ui.headers["location"].startswith("/v1/auth")
            assert (await anonymous.get("/v1/tasks")).status_code == 401

            admin_login = await anonymous.post(
                "/v1/auth/login",
                json={"username": "test_admin", "password": "test-admin-password"},
            )
            assert admin_login.status_code == 200
            assert admin_login.json()["role"] == "admin"
            assert (await anonymous.get("/v1/ui")).status_code == 200

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as general:
            second = await general.post(
                "/v1/auth/signup", json={"username": "general_user", "password": "password-456"}
            )
            assert second.status_code == 200
            assert second.json()["role"] == "general"
            assert (await general.get("/v1/ui")).status_code == 200
            console = await general.get("/v1/console")
            assert console.status_code == 403
            assert console.headers["content-type"].startswith("text/html")
            assert "Only admin users can access the console" in console.text
            assert 'href="/v1/ui"' in console.text

            logout = await general.post("/v1/auth/logout")
            assert logout.status_code == 200
            assert (await general.get("/v1/ui", follow_redirects=False)).status_code == 303

            bad_login = await general.post(
                "/v1/auth/login", json={"username": "general_user", "password": "wrong-password"}
            )
            assert bad_login.status_code == 401
            login = await general.post(
                "/v1/auth/login", json={"username": "general_user", "password": "password-456"}
            )
            assert login.status_code == 200

        stored = json.loads((tmp_path / "users.json").read_text(encoding="utf-8"))
        assert {item["role"] for item in stored["users"]} == {"admin", "general"}
        contents = (tmp_path / "users.json").read_text(encoding="utf-8")
        assert "test-admin-password" not in contents
        assert "password-456" not in contents

    asyncio.run(scenario())


def test_auth_rejects_duplicate_and_invalid_signup(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = create_app(make_auth_settings(tmp_path), FakeRunner())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            weak = await client.post(
                "/v1/auth/signup", json={"username": "valid_user", "password": "short"}
            )
            assert weak.status_code == 400
            assert (await client.post(
                "/v1/auth/signup", json={"username": "valid_user", "password": "long-enough"}
            )).status_code == 200
            duplicate = await client.post(
                "/v1/auth/signup", json={"username": "VALID_USER", "password": "another-pass"}
            )
            assert duplicate.status_code == 400

    asyncio.run(scenario())


def test_first_self_signup_is_general_with_provisioned_admin(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = create_app(make_auth_settings(tmp_path), FakeRunner())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            signup = await client.post(
                "/v1/auth/signup",
                json={"username": "first_user", "password": "password-123"},
            )
            assert signup.status_code == 200
            assert signup.json()["role"] == "general"
            assert (await client.get("/v1/console")).status_code == 403

    asyncio.run(scenario())


def test_default_admin_pair_is_created_automatically(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = replace(
            make_settings(tmp_path),
            authentication_enabled=True,
            users_file=tmp_path / "users.json",
        )
        app = create_app(settings, FakeRunner())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            login = await client.post(
                "/v1/auth/login",
                json={"username": "admin", "password": "kernelagent-admin"},
            )
            assert login.status_code == 200
            assert login.json()["role"] == "admin"
            assert (await client.get("/v1/ui")).status_code == 200

        contents = (tmp_path / "users.json").read_text(encoding="utf-8")
        assert '"username": "admin"' in contents
        assert "kernelagent-admin" not in contents

    asyncio.run(scenario())


def test_one_worker_serializes_tasks_on_one_gpu(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = FakeRunner(delay=0.08)
        app = create_app(make_settings(tmp_path), runner)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                first = await submit(client, "first")
                second = await submit(client, "second")
                assert (await wait_for_terminal(client, first))["status"] == "succeeded"
                assert (await wait_for_terminal(client, second))[
                    "status"
                ] == "succeeded"
        assert runner.max_running == 1
        assert [gpu for _, gpu in runner.calls] == ["GPU-test", "GPU-test"]

    asyncio.run(scenario())


def test_list_tasks_filters_status_and_paginates(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = FakeRunner(delay=0.15)
        app = create_app(make_settings(tmp_path), runner)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                first = await submit(client, "first")
                for _ in range(50):
                    if (await client.get(f"/v1/tasks/{first}")).json()[
                        "status"
                    ] == "running":
                        break
                    await asyncio.sleep(0.01)
                else:
                    raise AssertionError("first task did not start")
                second = await submit(client, "second")
                await client.post(f"/v1/tasks/{second}/cancel")

                canceled = (
                    await client.get("/v1/tasks", params={"status": "canceled"})
                ).json()
                assert [task["id"] for task in canceled] == [second]

                active_and_canceled = (
                    await client.get(
                        "/v1/tasks",
                        params=[("status", "running"), ("status", "canceled")],
                    )
                ).json()
                assert {task["id"] for task in active_and_canceled} == {first, second}

                page = (
                    await client.get("/v1/tasks", params={"offset": 1, "limit": 1})
                ).json()
                assert len(page) == 1
                assert page[0]["id"] == first

    asyncio.run(scenario())


def test_cancel_queued_task_is_skipped(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = FakeRunner(delay=0.15)
        app = create_app(make_settings(tmp_path), runner)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                first = await submit(client, "first")
                second = await submit(client, "second")
                canceled = await client.post(f"/v1/tasks/{second}/cancel")
                assert canceled.status_code == 200
                assert canceled.json()["status"] == "canceled"
                assert (await wait_for_terminal(client, first))["status"] == "succeeded"
                assert (await wait_for_terminal(client, second))["status"] == "canceled"
        assert [task_id for task_id, _ in runner.calls] == [first]

    asyncio.run(scenario())


def test_rejects_invalid_pytorch_problem_and_missing_gpu(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = FakeRunner()
        app = create_app(make_settings(tmp_path), runner)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                invalid = await client.post(
                    "/v1/tasks",
                    json={"pytorch_code": "def forward(:\n    pass"},
                )
                assert invalid.status_code == 422

                missing_contract = await client.post(
                    "/v1/tasks",
                    json={"pytorch_code": "import torch\ndef kernel(x): return x"},
                )
                assert missing_contract.status_code == 422

                unsupported_language = await client.post(
                    "/v1/tasks",
                    json={"pytorch_code": PYTORCH_CODE, "kernel_language": "tilelang"},
                )
                assert unsupported_language.status_code == 422

                unsupported_runner = await client.post(
                    "/v1/tasks",
                    json={"pytorch_code": PYTORCH_CODE, "runner_backend": "unknown"},
                )
                assert unsupported_runner.status_code == 422

        no_gpu_app = create_app(make_settings(tmp_path / "none", ()), runner)
        async with no_gpu_app.router.lifespan_context(no_gpu_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=no_gpu_app),
                base_url="http://testserver",
            ) as client:
                health = (await client.get("/healthz")).json()
                assert health["status"] == "degraded"
                assert health["runner_backends"] == ["claude", "pi", "codex"]
                unavailable = await client.post(
                    "/v1/tasks",
                    json={"pytorch_code": PYTORCH_CODE},
                )
                assert unavailable.status_code == 503

    asyncio.run(scenario())


def test_generated_files_beside_inputs_are_artifacts(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = create_app(make_settings(tmp_path), InputArtifactRunner())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/v1/tasks",
                    json={
                        "pytorch_code": PYTORCH_CODE,
                        "kernel_language": "cutedsl",
                    },
                )
                task = await wait_for_terminal(client, response.json()["task_id"])

        paths = {artifact["relative_path"] for artifact in task["artifacts"]}
        assert "input/optimized_kernel.py" in paths
        assert "input/problem.py" not in paths

    asyncio.run(scenario())


def test_pi_runtime_config_is_not_exposed_as_artifact(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = create_app(make_settings(tmp_path), PiRuntimeConfigRunner())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/v1/tasks", json={"pytorch_code": PYTORCH_CODE}
                )
                task = await wait_for_terminal(client, response.json()["task_id"])

        paths = {artifact["relative_path"] for artifact in task["artifacts"]}
        assert not any(path.startswith(".pi-runtime") for path in paths)

    asyncio.run(scenario())


def test_restart_recovers_queued_and_marks_running_lost(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, ())
    store = TaskStore(
        settings.runs_dir,
        settings.skills_dir,
        max_artifact_bytes=settings.max_artifact_bytes,
        max_artifacts=settings.max_artifacts,
    )
    request = CreateTaskRequest(operation="generate", problem="vector addition")
    store.create(TaskRecord(id="queued-task", operation="generate"), request)
    store.create(
        TaskRecord(
            id="running-task",
            operation="generate",
            status=TaskStatus.RUNNING,
        ),
        request,
    )

    async def scenario() -> None:
        app = create_app(settings, FakeRunner())
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                queued = (await client.get("/v1/tasks/queued-task")).json()
                lost = (await client.get("/v1/tasks/running-task")).json()
                assert queued["status"] == "queued"
                assert lost["status"] == "lost"
                assert "restarted" in lost["error"]

    asyncio.run(scenario())


def test_prompt_routes_operations_to_explicit_skills() -> None:
    parse = CreateTaskRequest(
        operation="parse",
        entrypoint="kernel.py",
        files=[{"path": "kernel.py", "content": "# kernel"}],
    )
    optimize = CreateTaskRequest(
        operation="optimize",
        files=[
            {"path": "input.py", "content": "# kernel"},
            {"path": "problem.py", "content": "# problem"},
            {"path": "test.py", "content": "# test"},
        ],
    )
    assert build_skill_prompt(parse).startswith("/ka-kernel-parser input/kernel.py")
    assert build_skill_prompt(optimize).startswith("/ka-kernel-opt input")
    assert build_skill_prompt(parse, command_prefix="$").startswith(
        "$ka-kernel-parser input/kernel.py"
    )


def test_runner_rejects_claude_error_result_with_zero_exit(tmp_path: Path) -> None:
    executable = tmp_path / "fake-claude"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({'type': 'result', 'subtype': 'error_max_turns', "
        "'is_error': True, 'result': 'turn limit'}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".claude-runtime").mkdir()
    settings = replace(make_settings(tmp_path), claude_command=str(executable))
    runner = ClaudeCodeRunner(settings)

    async def _append(target: list[dict], item: dict) -> None:
        target.append(item)

    async def _ignore(_: str) -> None:
        return None

    async def scenario() -> RunnerResult:
        events: list[dict] = []
        return await runner.run(
            task_id="runner-error",
            request=CreateTaskRequest(problem="test"),
            workspace=workspace,
            gpu_id="GPU-test",
            timeout_seconds=5,
            on_event=lambda event: _append(events, event),
            on_stderr=lambda text: _ignore(text),
        )

    result = asyncio.run(scenario())
    assert result.success is False
    assert result.exit_code == 0
    assert result.error == "turn limit"


def test_claude_runner_handles_json_lines_over_64kb(tmp_path: Path) -> None:
    """asyncio.StreamReader defaults to a 64KB line limit; a single stream-json
    event (e.g. large tool input/output) can exceed that and must not crash
    the task with LimitOverrunError."""
    executable = tmp_path / "fake-claude-large-line"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "huge_event = {'type': 'tool_use', 'name': 'Read', "
        "'input': {'blob': 'x' * 200_000}}\n"
        "print(json.dumps(huge_event))\n"
        "print(json.dumps({'type': 'result', 'subtype': 'success', "
        "'is_error': False, 'result': 'done', 'structured_output': "
        "{'success': True, 'summary': 'ok', 'skill': 'ka-kernel-gen'}}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".claude-runtime").mkdir()
    settings = replace(make_settings(tmp_path), claude_command=str(executable))
    runner = ClaudeCodeRunner(settings)

    async def _ignore_event(_: dict) -> None:
        return None

    async def _ignore_stderr(_: str) -> None:
        return None

    async def scenario() -> RunnerResult:
        return await runner.run(
            task_id="claude-large-line",
            request=CreateTaskRequest(problem="test"),
            workspace=workspace,
            gpu_id="GPU-test",
            timeout_seconds=5,
            on_event=_ignore_event,
            on_stderr=_ignore_stderr,
        )

    result = asyncio.run(scenario())
    assert result.success is True
    assert result.output["summary"] == "ok"


def test_pi_runner_handles_json_lines_over_64kb(tmp_path: Path) -> None:
    """Same 64KB StreamReader limit applies to pi's --mode json output, which
    can inline large content (e.g. a full SKILL.md) in a single event line."""
    executable = tmp_path / "fake-pi-large-line"
    result_json = json.dumps(
        {"success": True, "summary": "ok", "skill": "ka-kernel-gen"}
    )
    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": f"```json\n{result_json}\n```"}],
        "stopReason": "stop",
    }
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "huge_event = {'type': 'tool_execution_start', 'toolCallId': 't1', "
        "'toolName': 'read', 'args': {'blob': 'x' * 200_000}}\n"
        "print(json.dumps(huge_event))\n"
        f"message = {assistant_message!r}\n"
        "print(json.dumps({'type': 'message_start', 'message': message}))\n"
        "print(json.dumps({'type': 'message_end', 'message': message}))\n"
        "print(json.dumps({'type': 'agent_end', 'messages': [message], "
        "'willRetry': False}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    result = _run_pi_runner(tmp_path, executable)
    assert result.success is True
    assert result.output["summary"] == "ok"


def test_prompt_uses_pi_skill_command_syntax() -> None:
    parse = CreateTaskRequest(
        operation="parse",
        entrypoint="kernel.py",
        files=[{"path": "kernel.py", "content": "# kernel"}],
    )
    prompt = build_skill_prompt(parse, command_prefix="/skill:")
    assert prompt.startswith("/skill:ka-kernel-parser input/kernel.py")


def _write_fake_pi(path: Path, assistant_message: dict) -> None:
    script = (
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"message = {assistant_message!r}\n"
        "print(json.dumps({'type': 'session', 'version': 3, 'id': 't', "
        "'timestamp': 't', 'cwd': '.'}))\n"
        "print(json.dumps({'type': 'agent_start'}))\n"
        "print(json.dumps({'type': 'turn_start'}))\n"
        "print(json.dumps({'type': 'message_start', 'message': message}))\n"
        "print(json.dumps({'type': 'message_end', 'message': message}))\n"
        "print(json.dumps({'type': 'turn_end', 'message': message, "
        "'toolResults': []}))\n"
        "print(json.dumps({'type': 'agent_end', 'messages': [message], "
        "'willRetry': False}))\n"
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def _run_pi_runner(tmp_path: Path, executable: Path) -> RunnerResult:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    settings = replace(make_settings(tmp_path), pi_command=str(executable))
    runner = PiRunner(settings)

    async def _ignore_event(_: dict) -> None:
        return None

    async def _ignore_stderr(_: str) -> None:
        return None

    async def scenario() -> RunnerResult:
        return await runner.run(
            task_id="pi-runner-test",
            request=CreateTaskRequest(problem="test"),
            workspace=workspace,
            gpu_id="GPU-test",
            timeout_seconds=5,
            on_event=_ignore_event,
            on_stderr=_ignore_stderr,
        )

    return asyncio.run(scenario())


def test_pi_runner_parses_structured_result(tmp_path: Path) -> None:
    executable = tmp_path / "fake-pi"
    result_json = json.dumps(
        {"success": True, "summary": "did the thing", "skill": "ka-kernel-gen"}
    )
    _write_fake_pi(
        executable,
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"Done.\n\n```json\n{result_json}\n```"}
            ],
            "stopReason": "stop",
        },
    )
    result = _run_pi_runner(tmp_path, executable)
    assert result.success is True
    assert result.exit_code == 0
    assert result.output["summary"] == "did the thing"
    assert (tmp_path / "workspace" / ".pi-runtime" / "models.json").is_file()


def test_pi_runner_rejects_missing_structured_result(tmp_path: Path) -> None:
    executable = tmp_path / "fake-pi"
    _write_fake_pi(
        executable,
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I finished but forgot the JSON block."}
            ],
            "stopReason": "stop",
        },
    )
    result = _run_pi_runner(tmp_path, executable)
    assert result.success is False
    assert "structured result" in result.error


def test_pi_runner_rejects_error_stop_reason_despite_zero_exit(tmp_path: Path) -> None:
    executable = tmp_path / "fake-pi"
    _write_fake_pi(
        executable,
        {
            "role": "assistant",
            "content": [],
            "stopReason": "error",
            "errorMessage": "Connection error.",
        },
    )
    result = _run_pi_runner(tmp_path, executable)
    assert result.success is False
    assert result.exit_code == 0
    assert result.error == "Connection error."


def test_create_app_registers_all_task_runners(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path))
    runners = app.state.task_manager.runners

    assert isinstance(runners["claude"], ClaudeCodeRunner)
    assert isinstance(runners["pi"], PiRunner)
    assert isinstance(runners["codex"], CodexRunner)
    assert isinstance(
        create_runner(replace(make_settings(tmp_path / "pi"), agent="pi")), PiRunner
    )


def test_codex_runner_uses_project_skill_and_responses_provider(tmp_path: Path) -> None:
    executable = tmp_path / "fake-codex"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "output = pathlib.Path(args[args.index('--output-last-message') + 1])\n"
        "payload = {\n"
        "    'success': True,\n"
        "    'summary': 'codex done',\n"
        "    'skill': 'ka-kernel-gen',\n"
        "    'route': 'kernelagent',\n"
        "    'kernel_path': 'output/kernel.py',\n"
        "    'rounds': 2,\n"
        "    'metrics': {},\n"
        "    'warnings': [],\n"
        "}\n"
        "output.write_text(json.dumps(payload))\n"
        "print(json.dumps({'type': 'thread.started', 'thread_id': 'codex-thread'}))\n"
        "print(json.dumps({'type': 'turn.started'}))\n"
        "print(json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message'}}))\n"
        "print(json.dumps({'type': 'turn.completed'}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill_dir = tmp_path / "skills" / "ka-kernel-gen"
    skill_dir.mkdir(parents=True)
    settings = replace(
        make_settings(tmp_path),
        agent="codex",
        codex_command=str(executable),
        skills_dir=tmp_path / "skills",
        model_base_url="http://127.0.0.1:30000",
        model_name="local-kernel-model",
    )
    runner = CodexRunner(settings)

    async def scenario() -> tuple[RunnerResult, list[dict]]:
        events: list[dict] = []

        async def on_event(event: dict) -> None:
            events.append(event)

        async def on_stderr(_: str) -> None:
            return None

        result = await runner.run(
            task_id="codex-success",
            request=CreateTaskRequest(problem="test"),
            workspace=workspace,
            gpu_id="GPU-test",
            timeout_seconds=5,
            on_event=on_event,
            on_stderr=on_stderr,
        )
        return result, events

    result, events = asyncio.run(scenario())
    config = (workspace / ".codex-runtime" / "config.toml").read_text(encoding="utf-8")

    assert result.success is True
    assert result.session_id == "codex-thread"
    assert result.output["summary"] == "codex done"
    assert [event["type"] for event in events] == [
        "thread.started",
        "turn.started",
        "item.completed",
        "turn.completed",
    ]
    assert 'model = "local-kernel-model"' in config
    assert 'base_url = "http://127.0.0.1:30000/v1"' in config
    assert 'wire_api = "responses"' in config
    assert "requires_openai_auth = false" in config


def test_task_workspace_exposes_skills_to_claude_and_codex(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    store = TaskStore(
        settings.runs_dir,
        settings.skills_dir,
        max_artifact_bytes=settings.max_artifact_bytes,
        max_artifacts=settings.max_artifacts,
    )
    record = TaskRecord(id="skill-links", operation="generate")
    workspace = store.create(record, CreateTaskRequest(problem="vector addition"))

    assert (workspace / ".claude" / "skills").resolve() == settings.skills_dir.resolve()
    assert (workspace / ".agents" / "skills").resolve() == settings.skills_dir.resolve()
