"""Unified entry point for perception APIs served on port 8083."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

if __package__:
    from .parse_receipt import server as receipt_api
    from .pick.check import server as check_api
    from .pick.locate import main as locate_api
    from .place.check import server as place_check_api
else:
    from parse_receipt import server as receipt_api
    from pick.check import server as check_api
    from pick.locate import main as locate_api
    from place.check import server as place_check_api


app = FastAPI(title="Perception API", version="1.0.0")
app.include_router(locate_api.router)
app.include_router(check_api.router)
app.include_router(place_check_api.router)
app.include_router(receipt_api.router)
app.add_exception_handler(
    receipt_api.ServiceError,
    receipt_api.handle_service_error,
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083)
