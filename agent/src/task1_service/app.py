"""任务一服务的 FastAPI 应用。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from task1_service.client import Task1Client
from task1_service.models import (
    HealthResponse,
    Task1Request,
    Task1Result,
    Task1ServiceError,
    Task1Settings,
)
from task1_service.service import Task1Orchestrator


def create_app(
    settings: Task1Settings,
    *,
    orchestrator: Task1Orchestrator | None = None,
) -> FastAPI:
    client: Task1Client | None = None
    run_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal client, orchestrator
        if orchestrator is None:
            client = Task1Client(settings)
            orchestrator = Task1Orchestrator(settings, client)
        yield
        if client is not None:
            await client.aclose()

    app = FastAPI(title="Task 1 Receipt-to-Pick Service", lifespan=lifespan)

    @app.exception_handler(Task1ServiceError)
    async def service_error_handler(_: Request, exc: Task1ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error_code": exc.code, "message": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error_code": "INVALID_REQUEST", "message": str(exc.errors())},
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse | JSONResponse:
        assert orchestrator is not None
        if not await orchestrator.client.health_ready():
            return JSONResponse(status_code=503, content={"status": "ERROR"})
        return HealthResponse(status="READY")

    @app.post("/task1/run", response_model=Task1Result)
    async def run_task(
        request: Task1Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Task1Result:
        assert orchestrator is not None
        if run_lock.locked():
            raise Task1ServiceError("TASK_IN_PROGRESS", "another task is already running", status_code=409)
        async with run_lock:
            return await orchestrator.run(request, idempotency_key)

    return app
