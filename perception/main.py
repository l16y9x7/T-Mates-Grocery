"""Unified entry point for perception APIs served on port 8083."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

if __package__:
    from .config import SERVICE_BIND_HOST
    from .inspect import main as inspect_api
    from .parse_receipt import server as receipt_api
    from .pick.check import server as check_api
    from .pick.locate import main as locate_api
    from .place.check import server as place_check_api
    from .place.locate import main as place_locate_api
else:
    from config import SERVICE_BIND_HOST

    # ``inspect`` is also a Python standard-library module.  Load this local
    # namespace package by file path when the gateway is run as a script.
    inspect_path = Path(__file__).resolve().parent / "inspect" / "main.py"
    inspect_spec = importlib.util.spec_from_file_location(
        "perception_inspect_api",
        inspect_path,
    )
    if inspect_spec is None or inspect_spec.loader is None:
        raise RuntimeError(f"cannot load inspection API from {inspect_path}")
    inspect_api = importlib.util.module_from_spec(inspect_spec)
    sys.modules[inspect_spec.name] = inspect_api
    inspect_spec.loader.exec_module(inspect_api)
    from parse_receipt import server as receipt_api
    from pick.check import server as check_api
    from pick.locate import main as locate_api
    from place.check import server as place_check_api
    from place.locate import main as place_locate_api


app = FastAPI(title="Perception API", version="1.0.0")
app.include_router(locate_api.router)
app.include_router(check_api.router)
app.include_router(place_check_api.router)
app.include_router(place_locate_api.router)
app.include_router(receipt_api.router)
app.include_router(inspect_api.router)
app.add_exception_handler(
    receipt_api.ServiceError,
    receipt_api.handle_service_error,
)


@app.get("/perception/health")
def perception_health() -> dict[str, str]:
    """Report that the unified perception gateway is ready to serve requests."""
    return {"status": "READY"}


if __name__ == "__main__":
    uvicorn.run(app, host=SERVICE_BIND_HOST, port=8083)
