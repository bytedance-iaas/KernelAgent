"""Run the service with ``python -m kernelagent_service``."""

from __future__ import annotations

import argparse
from dataclasses import replace

import uvicorn

from kernelagent_service.app import create_app
from kernelagent_service.config import ServiceSettings


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kernel-agent-service",
        description="KernelAgent single-node task service.",
    )
    parser.add_argument(
        "-p",
        "--pi",
        action="store_true",
        help=(
            "Use the pi coding agent instead of Claude Code (default) to "
            "execute tasks. Equivalent to KERNEL_AGENT_AGENT=pi."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    settings = ServiceSettings.from_env()
    if args.pi:
        settings = replace(settings, agent="pi")
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
