"""任务二服务的 FastAPI 应用。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from task2_service.client import Task2Client
from task2_service.models import (
    HealthResponse,
    Task2Request,
    Task2Result,
    Task2ServiceError,
    Task2Settings,
)
from task2_service.service import Task2Orchestrator


def create_app(
    settings: Task2Settings,
    *,
    orchestrator: Task2Orchestrator | None = None,
) -> FastAPI:
    client: Task2Client | None = None
    run_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal client, orchestrator
        if orchestrator is None:
            client = Task2Client(settings)
            orchestrator = Task2Orchestrator(settings, client)
        yield
        if client is not None:
            await client.aclose()

    app = FastAPI(title="Task 2 Shelf Replenishment Service", lifespan=lifespan)

    @app.exception_handler(Task2ServiceError)
    async def service_error_handler(_: Request, exc: Task2ServiceError) -> JSONResponse:
        content = {"error_code": exc.code, "message": exc.message}
        if exc.step:
            content["failed_step"] = exc.step
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
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

    @app.post("/task2/run", response_model=Task2Result)
    async def run_task(
        request: Task2Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Task2Result:
        assert orchestrator is not None
        if run_lock.locked():
            raise Task2ServiceError(
                "TASK_IN_PROGRESS", "another task is already running", status_code=409
            )
        async with run_lock:
            return await orchestrator.run(request, idempotency_key)

    return app
