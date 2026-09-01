"""FastAPI application combining Task0-Task3 and the web console."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from task0_service.client import Task0Client
from task0_service.models import Task0Result, Task0ServiceError
from task0_service.service import Task0Orchestrator
from task1_service.client import Task1Client
from task1_service.models import Task1Result, Task1ServiceError
from task1_service.service import Task1Orchestrator
from task2_service.client import Task2Client
from task2_service.models import Task2Result, Task2ServiceError
from task2_service.service import Task2Orchestrator
from task3_service.client import Task3Client
from task3_service.models import Task3Result, Task3ServiceError
from task3_service.service import Task3Orchestrator

from .coordinator import TaskBinding, TaskCoordinator, TaskServiceError
from .settings import TaskServiceSettings


TaskResult = Task0Result | Task1Result | Task2Result | Task3Result
DomainError = Task0ServiceError | Task1ServiceError | Task2ServiceError | Task3ServiceError


def _bindings(orchestrators: dict[str, Any]) -> dict[str, TaskBinding]:
    return {
        "0": TaskBinding(orchestrators["0"], orchestrators["0"].client.health_ready),
        "1": TaskBinding(orchestrators["1"], orchestrators["1"].client.health_ready),
        "2": TaskBinding(orchestrators["2"], orchestrators["2"].ready),
        "3": TaskBinding(orchestrators["3"], orchestrators["3"].ready),
    }


def create_app(
    settings: TaskServiceSettings,
    *,
    orchestrators: dict[str, Any] | None = None,
) -> FastAPI:
    coordinator: TaskCoordinator | None = (
        TaskCoordinator(_bindings(orchestrators)) if orchestrators is not None else None
    )
    clients: list[Any] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal coordinator
        if coordinator is None:
            task0_settings = settings.tasks.task0
            task1_settings = settings.tasks.task1
            task2_settings = settings.tasks.task2
            task3_settings = settings.tasks.task3
            task0_client = Task0Client(task0_settings)
            task1_client = Task1Client(task1_settings)
            task2_client = Task2Client(task2_settings)
            task3_client = Task3Client(task3_settings)
            clients.extend((task0_client, task1_client, task2_client, task3_client))
            coordinator = TaskCoordinator(
                _bindings(
                    {
                        "0": Task0Orchestrator(task0_settings, task0_client),
                        "1": Task1Orchestrator(task1_settings, task1_client),
                        "2": Task2Orchestrator(task2_settings, task2_client),
                        "3": Task3Orchestrator(task3_settings, task3_client),
                    }
                )
            )
        app.state.task_coordinator = coordinator
        yield
        for client in clients:
            await client.aclose()

    app = FastAPI(
        title="Robot Retail Task Service",
        version="1.0",
        lifespan=lifespan,
    )

    def get_coordinator() -> TaskCoordinator:
        if coordinator is None:
            raise RuntimeError("task service lifespan has not started")
        return coordinator

    async def unified_error_handler(_: Request, exc: TaskServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error_code": exc.code, "message": exc.message},
        )

    async def domain_error_handler(_: Request, exc: DomainError) -> JSONResponse:
        content = {"error_code": exc.code, "message": exc.message}
        if exc.step:
            content["failed_step"] = exc.step
        if getattr(exc, "failed_interface", None):
            content["failed_interface"] = exc.failed_interface
        if getattr(exc, "url", None):
            content["url"] = exc.url
        interface_metrics = getattr(exc, "interface_metrics", None)
        if interface_metrics:
            content["interface_metrics"] = [
                metric.model_dump(mode="json")
                if isinstance(metric, BaseModel)
                else metric
                for metric in interface_metrics
            ]
        return JSONResponse(status_code=exc.status_code, content=content)

    async def validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error_code": "INVALID_REQUEST", "message": str(exc.errors())},
        )

    app.add_exception_handler(TaskServiceError, unified_error_handler)
    for error_type in (
        Task0ServiceError,
        Task1ServiceError,
        Task2ServiceError,
        Task3ServiceError,
    ):
        app.add_exception_handler(error_type, domain_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    @app.get("/health")
    async def health() -> JSONResponse:
        tasks = await get_coordinator().health()
        ready = all(status == "READY" for status in tasks.values())
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "READY" if ready else "ERROR", "tasks": tasks},
        )

    @app.post("/tasks/{task_id}/run", response_model=TaskResult)
    async def run_task(
        task_id: str,
        payload: dict[str, object] = Body(default_factory=dict),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> object:
        return await get_coordinator().run(task_id, payload, idempotency_key)

    from web.app import app as web_app, configure_runtime

    configure_runtime(settings, get_coordinator)
    app.mount("/", web_app)
    return app
