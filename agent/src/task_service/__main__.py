"""Run the unified task API and web console."""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app
from .settings import TaskServiceSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified Task0-Task3 service")
    parser.add_argument("--config", default="config/runtime.production.yaml")
    args = parser.parse_args()
    settings = TaskServiceSettings.load(args.config)
    uvicorn.run(
        create_app(settings),
        host=settings.server.host,
        port=settings.server.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
