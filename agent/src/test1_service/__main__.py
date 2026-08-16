"""Run one complete Test1 collection batch."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from test1_service.client import Test1Client
from test1_service.models import Test1ServiceError, Test1Settings
from test1_service.service import Test1Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Test1 RGB-D collection")
    parser.add_argument("--config", default="config/runtime.production.yaml")
    parser.add_argument("--operation-key")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        settings = Test1Settings.load(args.config)
        result = asyncio.run(_run(settings, args.operation_key))
    except (RuntimeError, ValueError, Test1ServiceError, OSError) as exc:
        payload = {
            "status": "FAILED",
            "error_code": getattr(exc, "code", type(exc).__name__),
            "step": getattr(exc, "step", None),
            "message": str(exc),
        }
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


async def _run(
    settings: Test1Settings, operation_key: str | None
):
    async with Test1Client(settings) as client:
        return await Test1Orchestrator(settings, client).run(operation_key)


if __name__ == "__main__":
    main()
