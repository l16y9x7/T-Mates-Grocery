"""Run the standalone external API mock."""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app
from .settings import MockExternalSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete-flow external task API mock")
    parser.add_argument("--config", default="config/runtime.mock.yaml")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    settings = MockExternalSettings.load(args.config)
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
