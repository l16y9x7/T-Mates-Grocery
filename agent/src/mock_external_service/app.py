"""FastAPI application for the standalone external API mock."""

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from task_service.external import Task0TriggerRequest, Task1TriggerRequest, Task2TriggerRequest

from .service import MockExternalService, MockTaskError
from .settings import MockExternalSettings


def create_app(settings: MockExternalSettings, *, service: MockExternalService | None = None) -> FastAPI:
    mock_service = service or MockExternalService(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await mock_service.close()

    app = FastAPI(title="External Task API Mock", version="1.0", lifespan=lifespan)

    async def error_handler(request: Request, exc: MockTaskError) -> JSONResponse:
        request_id = request.headers.get("X-Request-Id") or f"request-{uuid4().hex}"
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "schema_version": "1.0",
                "request_id": request_id,
                "error_code": exc.code,
                "message": exc.message,
                "retryable": exc.retryable,
            },
        )

    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "schema_version": "1.0",
                "request_id": request.headers.get("X-Request-Id") or f"request-{uuid4().hex}",
                "error_code": "INVALID_REQUEST",
                "message": str(exc.errors()),
                "retryable": False,
            },
        )

    app.add_exception_handler(MockTaskError, error_handler)
    app.add_exception_handler(RequestValidationError, validation_handler)
    router = APIRouter(prefix="/api/external/v1")

    def request_id(value: str | None) -> str:
        return value.strip() if value and value.strip() else f"request-{uuid4().hex}"

    def authorize(authorization: str | None) -> None:
        if settings.access_token and authorization != f"Bearer {settings.access_token}":
            raise MockTaskError("UNAUTHORIZED", "缺少或无效的访问令牌", 401)

    @router.get("/health")
    async def health(authorization: str | None = Header(default=None), x_request_id: str | None = Header(default=None, alias="X-Request-Id")) -> JSONResponse:
        authorize(authorization)
        return JSONResponse(await mock_service.health(request_id(x_request_id)))

    @router.post("/tasks/0/runs", status_code=202)
    async def task0(body: Task0TriggerRequest, authorization: str | None = Header(default=None), idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"), x_request_id: str | None = Header(default=None, alias="X-Request-Id")) -> JSONResponse:
        return await submit("0", body, authorization, idempotency_key, x_request_id)

    @router.post("/task1/orders", status_code=202)
    async def task1(body: Task1TriggerRequest, authorization: str | None = Header(default=None), idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"), x_request_id: str | None = Header(default=None, alias="X-Request-Id")) -> JSONResponse:
        return await submit("1", body, authorization, idempotency_key, x_request_id)

    @router.post("/tasks/2/runs", status_code=202)
    async def task2(body: Task2TriggerRequest, authorization: str | None = Header(default=None), idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"), x_request_id: str | None = Header(default=None, alias="X-Request-Id")) -> JSONResponse:
        return await submit("2", body, authorization, idempotency_key, x_request_id)

    @router.get("/tasks/{task_run_id}/status")
    async def status(task_run_id: str, authorization: str | None = Header(default=None)) -> JSONResponse:
        authorize(authorization)
        return JSONResponse(await mock_service.get_status(task_run_id))

    async def submit(task_number: str, body: Task0TriggerRequest | Task1TriggerRequest | Task2TriggerRequest, authorization: str | None, idempotency_key: str | None, x_request_id: str | None) -> JSONResponse:
        authorize(authorization)
        if not idempotency_key or not idempotency_key.strip():
            raise MockTaskError("INVALID_REQUEST", "缺少 Idempotency-Key", 400)
        response, duplicate = await mock_service.submit(task_number, body, idempotency_key.strip(), request_id(x_request_id))
        return JSONResponse(response, status_code=200 if duplicate else 202)

    app.include_router(router)
    return app
