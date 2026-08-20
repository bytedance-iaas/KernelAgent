"""Run the service with ``python -m kernelagent_service``."""

from __future__ import annotations

import uvicorn

from kernelagent_service.app import create_app
from kernelagent_service.config import ServiceSettings


def main() -> None:
    settings = ServiceSettings.from_env()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
