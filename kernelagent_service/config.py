"""Environment-backed settings for the service."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _detect_gpu_ids() -> tuple[str, ...]:
    configured = os.getenv("KERNEL_AGENT_GPU_IDS")
    if configured is not None:
        return tuple(item.strip() for item in configured.split(",") if item.strip())

    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible and visible not in {"-1", "none", "None"}:
        return tuple(item.strip() for item in visible.split(",") if item.strip())

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ()
    if result.returncode != 0:
        return ()
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


@dataclass(frozen=True)
class ServiceSettings:
    repo_root: Path
    runs_dir: Path
    skills_dir: Path
    gpu_ids: tuple[str, ...]
    queue_capacity: int = 100
    claude_command: str = "claude"
    model_base_url: str = "http://127.0.0.1:30000"
    model_auth_token: str = "dummy"
    model_name: str = "default"
    default_timeout_seconds: int = 7_200
    shutdown_grace_seconds: int = 10
    max_input_bytes: int = 10 * 1024 * 1024
    max_artifact_bytes: int = 50 * 1024 * 1024
    max_artifacts: int = 500
    host: str = "127.0.0.1"
    port: int = 8080

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        repo_root = Path(__file__).resolve().parent.parent
        runs_dir = Path(
            os.getenv("KERNEL_AGENT_RUNS_DIR", str(repo_root / ".kernel_agent_service" / "runs"))
        ).expanduser()
        skills_dir = Path(
            os.getenv("KERNEL_AGENT_SKILLS_DIR", str(repo_root / ".claude" / "skills"))
        ).expanduser()
        return cls(
            repo_root=repo_root,
            runs_dir=runs_dir,
            skills_dir=skills_dir,
            gpu_ids=_detect_gpu_ids(),
            queue_capacity=_env_int("KERNEL_AGENT_QUEUE_CAPACITY", 100),
            claude_command=os.getenv("KERNEL_AGENT_CLAUDE_COMMAND", "claude"),
            model_base_url=os.getenv(
                "KERNEL_AGENT_MODEL_BASE_URL",
                os.getenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:30000"),
            ),
            model_auth_token=os.getenv(
                "KERNEL_AGENT_MODEL_AUTH_TOKEN",
                os.getenv("ANTHROPIC_AUTH_TOKEN", "dummy"),
            ),
            model_name=os.getenv(
                "KERNEL_AGENT_MODEL",
                os.getenv("ANTHROPIC_MODEL", "default"),
            ),
            default_timeout_seconds=_env_int("KERNEL_AGENT_TASK_TIMEOUT_SECONDS", 7_200),
            shutdown_grace_seconds=_env_int("KERNEL_AGENT_SHUTDOWN_GRACE_SECONDS", 10),
            max_input_bytes=_env_int("KERNEL_AGENT_MAX_INPUT_BYTES", 10 * 1024 * 1024),
            max_artifact_bytes=_env_int(
                "KERNEL_AGENT_MAX_ARTIFACT_BYTES", 50 * 1024 * 1024
            ),
            max_artifacts=_env_int("KERNEL_AGENT_MAX_ARTIFACTS", 500),
            host=os.getenv("KERNEL_AGENT_SERVICE_HOST", "127.0.0.1"),
            port=_env_int("KERNEL_AGENT_SERVICE_PORT", 8080),
        )
