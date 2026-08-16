"""FastAPI 入口：8086 独立取放编排服务。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from pick_place_service.models import HealthResponse, PickPlaceRequest, PickPlaceSettings, ServiceError, StatusResponse
from pick_place_service.service import CameraFrameProvider, OperationCache, PickPlaceOrchestrator, SubagentClient


def create_app(
    settings: PickPlaceSettings,
    *,
    orchestrator: PickPlaceOrchestrator | None = None,
) -> FastAPI:
    """创建可测试的应用实例；未注入 orchestrator 时连接真实下游服务。"""

    http_client: httpx.AsyncClient | None = None
    cache = OperationCache()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal http_client, orchestrator
        if orchestrator is None:
            http_client = httpx.AsyncClient()
            subagents = SubagentClient(settings, http_client)
            frames = CameraFrameProvider(settings, http_client)
            orchestrator = PickPlaceOrchestrator(settings, subagents, frames)
        yield
        if http_client is not None:
            await http_client.aclose()

    app = FastAPI(title="Pick/Place Orchestrator", lifespan=lifespan)

    @app.exception_handler(ServiceError)
    async def service_error_handler(_: Request, exc: ServiceError) -> JSONResponse:
        content = {"error_code": exc.code, "message": exc.message}
        if exc.failed_interface:
            content["failed_interface"] = exc.failed_interface
        if exc.url:
            content["url"] = exc.url
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse | JSONResponse:
        assert orchestrator is not None
        if isinstance(orchestrator.subagents, SubagentClient) and not await orchestrator.subagents.health():
            return JSONResponse(status_code=503, content={"status": "ERROR"})
        return HealthResponse(status="READY")

    @app.get("/status")
    async def status() -> dict[str, int | str]:
        active_operations = await cache.active_count()
        return {
            "status": "BUSY" if active_operations else "READY",
            "active_operations": active_operations,
        }

    async def run_operation(
        request: PickPlaceRequest,
        kind: str,
        idempotency_key: str | None,
    ) -> StatusResponse:
        if not idempotency_key or not idempotency_key.strip():
            raise ServiceError("MISSING_IDEMPOTENCY_KEY", "Idempotency-Key header is required", status_code=400)
        assert orchestrator is not None
        return await cache.run(
            idempotency_key,
            request,
            lambda: orchestrator.run(request, kind, idempotency_key),
        )

    @app.post("/pick", response_model=StatusResponse)
    async def pick(
        request: PickPlaceRequest,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> StatusResponse:
        return await run_operation(request, "pick", idempotency_key)

    @app.post("/place", response_model=StatusResponse)
    async def place(
        request: PickPlaceRequest,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> StatusResponse:
        return await run_operation(request, "place", idempotency_key)

    return app
