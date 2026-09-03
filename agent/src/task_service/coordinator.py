"""Shared dispatch and concurrency control for all robot tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

from task0_service.models import Task0Request
from task1_service.models import Task1Request
from task2_service.models import Task2Request
from task3_service.models import Task3Request


TASK_IDS = ("0", "1", "2", "3")
REQUEST_TYPES: dict[str, type[BaseModel]] = {
    "0": Task0Request,
    "1": Task1Request,
    "2": Task2Request,
    "3": Task3Request,
}


class TaskServiceError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class TaskBinding:
    orchestrator: Any
    health_check: Callable[[], Awaitable[bool]]


class TaskCoordinator:
    """Dispatch tasks while reserving one global robot execution slot."""

    def __init__(self, bindings: dict[str, TaskBinding]) -> None:
        missing = set(TASK_IDS) - bindings.keys()
        if missing:
            raise ValueError(f"missing task bindings: {sorted(missing)}")
        self.bindings = bindings
        self._state_lock = asyncio.Lock()
        self._active_task_id: str | None = None

    @property
    def active_task_id(self) -> str | None:
        return self._active_task_id

    def validate_task_id(self, task_id: str) -> str:
        if task_id not in self.bindings:
            raise TaskServiceError(
                "TASK_NOT_FOUND", f"unknown task id: {task_id}", status_code=404
            )
        return task_id

    def build_request(self, task_id: str, payload: dict[str, object]) -> BaseModel:
        self.validate_task_id(task_id)
        try:
            return REQUEST_TYPES[task_id].model_validate(payload)
        except ValidationError as exc:
            raise TaskServiceError(
                "INVALID_REQUEST", str(exc.errors()), status_code=422
            ) from exc

    async def _reserve(self, task_id: str) -> None:
        self.validate_task_id(task_id)
        async with self._state_lock:
            if self._active_task_id is not None:
                raise TaskServiceError(
                    "TASK_IN_PROGRESS",
                    f"task {self._active_task_id} is already running",
                    status_code=409,
                )
            self._active_task_id = task_id

    async def _execute_reserved(
        self, task_id: str, request: BaseModel, operation_key: str | None
    ) -> object:
        try:
            return await self.bindings[task_id].orchestrator.run(request, operation_key)
        finally:
            async with self._state_lock:
                self._active_task_id = None

    async def run(
        self, task_id: str, payload: dict[str, object], operation_key: str | None
    ) -> object:
        request = self.build_request(task_id, payload)
        await self._reserve(task_id)
        return await self._execute_reserved(task_id, request, operation_key)

    async def start_background(
        self, task_id: str, payload: dict[str, object], operation_key: str
    ) -> asyncio.Task[object]:
        request = self.build_request(task_id, payload)
        await self._reserve(task_id)
        started = asyncio.Event()

        async def execute() -> object:
            started.set()
            return await self._execute_reserved(task_id, request, operation_key)

        task = asyncio.create_task(
            execute(),
            name=f"task-{task_id}-{operation_key}",
        )
        await started.wait()
        return task

    async def health(self) -> dict[str, str]:
        results = await asyncio.gather(
            *(self.bindings[task_id].health_check() for task_id in TASK_IDS),
            return_exceptions=True,
        )
        return {
            task_id: "READY" if result is True else "ERROR"
            for task_id, result in zip(TASK_IDS, results)
        }

    async def task_ready(self, task_id: str) -> bool:
        """Check one task without reserving the global execution slot."""

        self.validate_task_id(task_id)
        return await self.bindings[task_id].health_check()
