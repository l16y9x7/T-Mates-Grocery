"""启动 8086 取放服务：python -m pick_place_service."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from pick_place_service.app import create_app
from pick_place_service.models import PickPlaceSettings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the pick/place orchestration service")
    parser.add_argument("--config", default="config/pick-place.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8086)
    args = parser.parse_args()
    uvicorn.run(create_app(PickPlaceSettings.load(args.config)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
