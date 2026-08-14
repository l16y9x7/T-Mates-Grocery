"""启动任务二独立服务。"""

from __future__ import annotations

import argparse

import uvicorn

from task2_service.app import create_app
from task2_service.models import Task2Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run task 2 shelf replenishment service")
    parser.add_argument("--config", default="config/task2.production.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8109)
    args = parser.parse_args()
    uvicorn.run(create_app(Task2Settings.load(args.config)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
