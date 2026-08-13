"""启动任务一独立服务。"""

from __future__ import annotations

import argparse

import uvicorn

from task1_service.app import create_app
from task1_service.models import Task1Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run task 1 receipt-to-pick service")
    parser.add_argument("--config", default="config/task1.production.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8108)
    args = parser.parse_args()
    uvicorn.run(create_app(Task1Settings.load(args.config)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
