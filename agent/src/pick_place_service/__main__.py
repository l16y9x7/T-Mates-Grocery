"""启动 8086 取放服务：python -m pick_place_service."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from pick_place_service.app import create_app
from pick_place_service.models import PickPlaceSettings
from runtime_config import load_runtime_document


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the pick/place orchestration service")
    parser.add_argument("--config", default="config/runtime.production.yaml")
    args = parser.parse_args()
    document = load_runtime_document(args.config)
    server = document.servers.pick_place
    uvicorn.run(
        create_app(PickPlaceSettings.from_runtime_document(document)),
        host=server.host,
        port=server.port,
    )


if __name__ == "__main__":
    main()
