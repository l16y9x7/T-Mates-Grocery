"""External asynchronous task API and callback status adapter."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from task1_service.models import Task1ServiceError

from .coordinator import TaskCoordinator, TaskServiceError
from .settings import ExternalServiceSettings


LOGGER = logging.getLogger(__name__)


class CallbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status_callback_url: str | None = Field(default=None, max_length=2000)


class Task0TriggerRequest(CallbackRequest):
    external_task_id: str = Field(min_length=1, max_length=200)

    @field_validator("external_task_id")
    @classmethod
    def trim_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("external_task_id must not be empty")
        return value


class ExternalOrderItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, max_length=200)
    sku_id: str = Field(min_length=1, max_length=100)
    quantity: Literal[1] = 1

    @field_validator("item_id", "sku_id")
    @classmethod
    def trim_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class Task1TriggerRequest(CallbackRequest):
    external_task_id: str = Field(min_length=1, max_length=200)
    external_order_id: str = Field(min_length=1, max_length=200)
    items: list[ExternalOrderItem]

    @field_validator("external_task_id", "external_order_id")
    @classmethod
    def trim_ids(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("order IDs must not be empty")
        return value

    @field_validator("items")
    @classmethod
    def validate_items(cls, value: list[ExternalOrderItem]) -> list[ExternalOrderItem]:
        if len(value) != 2:
            raise ValueError("items must contain exactly two products")
        if len({item.sku_id.upper() for item in value}) != 2:
            raise ValueError("items must contain two distinct sku_id values")
        if len({item.item_id for item in value}) != 2:
            raise ValueError("item_id values must be distinct")
        return value


class Task2TriggerRequest(CallbackRequest):
    external_task_id: str = Field(min_length=1, max_length=200)

    @field_validator("external_task_id")
    @classmethod
    def trim_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("external_task_id must not be empty")
        return value


@dataclass
class _CallbackWorker:
    url: str
    access_token: str | None
    settings: ExternalServiceSettings
    queue: asyncio.Queue[dict[str, Any] | None] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name="external-status-callback")

    async def enqueue(self, payload: dict[str, Any]) -> None:
        await self.queue.put(payload)

    async def close(self) -> None:
        await self.queue.put(None)
        if self.task is not None:
            await self.task

    async def _run(self) -> None:
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            while True:
                payload = await self.queue.get()
                if payload is None:
                    return
                await self._send(client, payload)

    async def _send(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> None:
        headers = {
            "Content-Type": "application/json",
            "X-Event-Id": payload["event_id"],
            "X-Task-Run-Id": payload["task_run_id"],
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        attempts = self.settings.max_retries + 1
        for attempt in range(attempts):
            try:
                response = await client.post(self.url, json=payload, headers=headers)
                if 200 <= response.status_code < 300:
                    return
                error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                error = str(exc)
            if attempt + 1 < attempts:
                delay = self.settings.retry_backoff_seconds * (2**attempt)
                if delay:
                    await asyncio.sleep(delay)
            else:
                LOGGER.error(
                    "external status callback failed event_id=%s attempts=%s error=%s",
                    payload["event_id"],
                    attempts,
                    error,
                )


@dataclass
class _ExternalTaskRecord:
    task_id: str
    task_number: str
    task_type: str
    task_name: str
    external_task_id: str
    external_order_id: str | None
    request: BaseModel
    fingerprint: str
    response: dict[str, Any]
    callback_url: str | None
    callback_worker: _CallbackWorker | None
    sequence: int = 0
    last_status: dict[str, Any] | None = None
    monitor: asyncio.Task[None] | None = None
    sku_names: dict[str, str] = field(default_factory=dict)


class ExternalTaskService:
    """Translate external contracts into the existing Task0-Task2 services."""

    def __init__(
        self,
        coordinator: TaskCoordinator | None,
        settings: ExternalServiceSettings,
    ) -> None:
        self.coordinator = coordinator
        self.settings = settings
        self._lock = asyncio.Lock()
        self._by_idempotency: dict[str, _ExternalTaskRecord] = {}
        self._by_external_id: dict[str, _ExternalTaskRecord] = {}
        self._records: dict[str, _ExternalTaskRecord] = {}

    def bind_coordinator(self, coordinator: TaskCoordinator) -> None:
        self.coordinator = coordinator

    def _coordinator(self) -> TaskCoordinator:
        if self.coordinator is None:
            raise RuntimeError("task service lifespan has not started")
        return self.coordinator

    async def close(self) -> None:
        monitors = [record.monitor for record in self._records.values()]
        for monitor in monitors:
            if monitor is not None and not monitor.done():
                monitor.cancel()
        if monitors:
            await asyncio.gather(*monitors, return_exceptions=True)
        workers = [record.callback_worker for record in self._records.values()]
        for worker in workers:
            if worker is not None:
                await worker.close()

    async def health(self) -> dict[str, Any]:
        task_status = {}
        for task_number in ("0", "1", "2"):
            task_status[task_number] = "READY" if await self._coordinator().task_ready(task_number) else "ERROR"
        active = self._coordinator().active_task_id
        return {
            "status": "BUSY" if active is not None else ("READY" if all(value == "READY" for value in task_status.values()) else "NOT_READY"),
            "accepting_tasks": active is None and any(value == "READY" for value in task_status.values()),
            "ready_for_task0": task_status["0"] == "READY",
            "ready_for_task1": task_status["1"] == "READY",
            "ready_for_task2": task_status["2"] == "READY",
            "active_task": active,
        }

    async def submit(
        self,
        task_number: str,
        request: Task0TriggerRequest | Task1TriggerRequest | Task2TriggerRequest,
        idempotency_key: str,
        request_id: str,
    ) -> tuple[dict[str, Any], bool]:
        callback_url = self._callback_url(request.status_callback_url)
        fingerprint = self._fingerprint(request)
        external_task_id = request.external_task_id
        async with self._lock:
            duplicate = self._by_idempotency.get(idempotency_key)
            if duplicate is not None:
                self._check_duplicate(duplicate, task_number, fingerprint)
                return dict(
                    duplicate.response,
                    duplicate=True,
                    request_id=request_id,
                    status=(duplicate.last_status or {}).get(
                        "status", duplicate.response["status"]
                    ),
                ), True
            duplicate = self._by_external_id.get(external_task_id)
            if duplicate is not None:
                self._check_duplicate(duplicate, task_number, fingerprint)
                return dict(
                    duplicate.response,
                    duplicate=True,
                    request_id=request_id,
                    status=(duplicate.last_status or {}).get(
                        "status", duplicate.response["status"]
                    ),
                ), True
            if self._coordinator().active_task_id is not None:
                raise TaskServiceError("TASK_BUSY", "当前有任务正在执行，请稍后重试", status_code=409)
            if not await self._coordinator().task_ready(task_number):
                code = "BASELINE_NOT_READY" if task_number == "2" else "SYSTEM_NOT_READY"
                raise TaskServiceError(code, "当前任务依赖未就绪，请稍后重试", status_code=503)

            task_id = self._new_task_id(task_number)
            task_type, task_name = self._task_identity(task_number)
            response = {
                "schema_version": "1.0",
                "request_id": request_id,
                "external_task_id": external_task_id,
                "task_run_id": task_id,
                "task_type": task_type,
                "task_name": task_name,
                "status": "ACCEPTED",
                "accepted_at": _now(),
                "status_callback_enabled": callback_url is not None,
            }
            if isinstance(request, Task1TriggerRequest):
                response["external_order_id"] = request.external_order_id
            record = _ExternalTaskRecord(
                task_id=task_id,
                task_number=task_number,
                task_type=task_type,
                task_name=task_name,
                external_task_id=external_task_id,
                external_order_id=getattr(request, "external_order_id", None),
                request=request,
                fingerprint=fingerprint,
                response=response,
                callback_url=callback_url,
                callback_worker=self._new_worker(callback_url),
            )
            self._records[task_id] = record
            self._by_idempotency[idempotency_key] = record
            self._by_external_id[external_task_id] = record
            try:
                internal_payload = await self._internal_payload(task_number, request, record)
                execution = await self._coordinator().start_background(task_number, internal_payload, task_id)
            except Exception:
                self._records.pop(task_id, None)
                self._by_idempotency.pop(idempotency_key, None)
                self._by_external_id.pop(external_task_id, None)
                raise
            if record.callback_worker is not None:
                record.callback_worker.start()
            await self._publish(record, "TASK_ACCEPTED", "ACCEPTED", self._accepted_status(record))
            await self._publish(record, "TASK_PROGRESS", "RUNNING", self._running_status(record))
            record.monitor = asyncio.create_task(
                self._monitor(record, execution), name=f"external-monitor-{task_id}"
            )
            return response, False

    async def get_status(self, task_run_id: str) -> dict[str, Any]:
        record = self._records.get(task_run_id)
        if record is None:
            raise TaskServiceError("TASK_NOT_FOUND", "task_run_id not found", status_code=404)
        return record.last_status or self._accepted_status(record)

    def router(self) -> APIRouter:
        router = APIRouter(prefix="/api/external/v1")

        @router.get("/health")
        async def external_health(
            authorization: str | None = Header(default=None),
        ) -> JSONResponse:
            self._authorize(authorization)
            return JSONResponse(await self.health())

        @router.post("/tasks/0/runs", status_code=202)
        async def task0(
            body: Task0TriggerRequest,
            authorization: str | None = Header(default=None),
            idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
            request_id: str | None = Header(default=None, alias="X-Request-Id"),
        ) -> JSONResponse:
            self._authorize(authorization)
            return await self._submit_response("0", body, idempotency_key, request_id)

        @router.post("/task1/orders", status_code=202)
        async def task1(
            body: Task1TriggerRequest,
            authorization: str | None = Header(default=None),
            idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
            request_id: str | None = Header(default=None, alias="X-Request-Id"),
        ) -> JSONResponse:
            self._authorize(authorization)
            return await self._submit_response("1", body, idempotency_key, request_id)

        @router.post("/tasks/2/runs", status_code=202)
        async def task2(
            body: Task2TriggerRequest,
            authorization: str | None = Header(default=None),
            idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
            request_id: str | None = Header(default=None, alias="X-Request-Id"),
        ) -> JSONResponse:
            self._authorize(authorization)
            return await self._submit_response("2", body, idempotency_key, request_id)

        @router.get("/tasks/{task_run_id}/status")
        async def status(
            task_run_id: str,
            authorization: str | None = Header(default=None),
        ) -> JSONResponse:
            self._authorize(authorization)
            return JSONResponse(await self.get_status(task_run_id))

        return router

    async def _submit_response(
        self,
        task_number: str,
        request: BaseModel,
        idempotency_key: str | None,
        request_id: str | None,
    ) -> JSONResponse:
        if not idempotency_key or not idempotency_key.strip():
            raise TaskServiceError("INVALID_REQUEST", "缺少 Idempotency-Key", status_code=400)
        response, duplicate = await self.submit(
            task_number,
            request,  # type: ignore[arg-type]
            idempotency_key.strip(),
            request_id or f"request-{uuid4().hex}",
        )
        return JSONResponse(response, status_code=200 if duplicate else 202)

    def _new_worker(self, callback_url: str | None) -> _CallbackWorker | None:
        if callback_url is None:
            return None
        return _CallbackWorker(callback_url, self.settings.callback_access_token, self.settings)

    async def _monitor(self, record: _ExternalTaskRecord, execution: asyncio.Task[Any]) -> None:
        while not execution.done():
            try:
                await asyncio.wait_for(asyncio.shield(execution), timeout=self.settings.heartbeat_seconds)
            except asyncio.TimeoutError:
                if record.callback_worker is not None:
                    await self._publish(record, "TASK_HEARTBEAT", "RUNNING", self._running_status(record))
            except Exception as exc:
                await self._publish_failure(record, exc)
                return
        try:
            result = execution.result()
        except Exception as exc:
            await self._publish_failure(record, exc)
            return
        await self._publish_final(record, result)

    async def _publish_failure(self, record: _ExternalTaskRecord, error: Exception) -> None:
        payload = self._failed_status(record, error)
        await self._publish(record, "TASK_FAILED", "FAILED", payload)

    async def _publish_final(self, record: _ExternalTaskRecord, result: Any) -> None:
        payload = self._result_status(record, result)
        await self._publish(
            record,
            "TASK_COMPLETED",
            payload.get("status", "FAILED"),
            payload,
        )

    async def _publish(
        self,
        record: _ExternalTaskRecord,
        event_type: str,
        status: str,
        payload: dict[str, Any],
    ) -> None:
        record.sequence += 1
        payload = dict(payload)
        payload.update(
            schema_version="1.0",
            event_id=f"evt-{record.task_id}-{record.sequence}-{uuid4().hex[:8]}",
            sequence=record.sequence,
            event_type=event_type,
            occurred_at=_now(),
            task_run_id=record.task_id,
            task_type=record.task_type,
            task_name=record.task_name,
            status=status,
        )
        record.last_status = payload
        if record.callback_worker is not None:
            await record.callback_worker.enqueue(payload)

    def _accepted_status(self, record: _ExternalTaskRecord) -> dict[str, Any]:
        return self._status(
            record,
            "ACCEPTED",
            "已接收任务",
            f"已接收{record.task_name}任务",
            {"code": "ACCEPTED", "label": "已接收任务", "progress_percent": 0, "message": "任务准备启动"},
            {"code": "UNKNOWN", "label": "等待启动"},
            self._initial_summary(record),
            {"level": "INFO", "code": "TASK_ACCEPTED", "message": f"已接收{record.task_name}任务"},
        )

    def _running_status(self, record: _ExternalTaskRecord) -> dict[str, Any]:
        labels = {"0": ("正在巡检并记录货架状态", "正在检查设备"), "1": ("正在为您取货", "正在检查设备"), "2": ("正在检查货架并准备补货", "正在检查设备")}
        title, label = labels[record.task_number]
        return self._status(
            record,
            "RUNNING",
            title,
            f"{label}，请稍候",
            {"code": "HEALTH_CHECKING", "label": label, "progress_percent": 5, "message": "正在检查任务所需设备"},
            {"code": "UNKNOWN", "label": "继续执行任务"},
            self._initial_summary(record),
            {"level": "INFO", "code": "TASK_IN_PROGRESS", "message": f"{record.task_name}正在进行中，请稍候"},
        )

    def _result_status(self, record: _ExternalTaskRecord, result: Any) -> dict[str, Any]:
        data = result.model_dump(mode="json") if isinstance(result, BaseModel) else result
        if record.task_number == "0":
            captures = [{"target_id": item["target_id"], "target_label": item["target_id"], "view": item["pose_type"].replace("SHELF_VIEW_", ""), "status": "COMPLETED", "status_label": "已记录", "message": "货架信息已记录"} for item in data.get("captures", [])]
            total = (len(captures) + 1) // 2
            summary = {"inspection_points_total": total, "inspection_points_completed": total, "captures_total": len(captures), "captures_completed": len(captures), "captures_failed": 0}
            return self._status(record, "SUCCEEDED", "理货完成", "货架信息已全部记录", {"code": "SUCCEEDED", "label": "理货完成", "progress_percent": 100, "message": "所有货架区域记录完成"}, None, summary, {"level": "SUCCESS", "code": "TASK_SUCCEEDED", "message": "理货已完成"}, captures=captures)
        if record.task_number == "1":
            request = record.request
            items = []
            for requested in request.items:  # type: ignore[union-attr]
                product_name = record.sku_names[requested.sku_id.upper()]
                target = next((item for item in data.get("target_items", []) if item["product_name"] == product_name), None)
                item_status = "PLACED" if target and target.get("placed") else "PICKED" if target and target.get("picked") else "PENDING"
                items.append({"item_id": requested.item_id, "sku_id": requested.sku_id, "product_name": product_name, "status": item_status, "status_label": {"PLACED": "已完成", "PICKED": "已取到", "PENDING": "等待处理"}[item_status], "picked": bool(target and target.get("picked")), "placed": bool(target and target.get("placed")), "message": {"PLACED": "商品已放到交付台", "PICKED": "商品已取到，等待交付", "PENDING": "商品未完成处理"}[item_status]})
            placed = sum(item["placed"] for item in items)
            status = "SUCCEEDED" if placed == len(items) else "PARTIAL_SUCCESS" if placed else "FAILED"
            title = "取货完成" if status == "SUCCEEDED" else "取货部分完成" if status == "PARTIAL_SUCCESS" else "取货失败"
            return self._status(record, status, title, f"已完成 {placed}/{len(items)} 件商品", {"code": status, "label": title, "progress_percent": 100, "message": "订单处理已结束"}, None, {"total_items": len(items), "items_completed": placed, "items_in_progress": 0, "items_failed": len(items) - placed, "items_held": sum(item["picked"] and not item["placed"] for item in items)}, {"level": "SUCCESS" if status == "SUCCEEDED" else "WARNING" if status == "PARTIAL_SUCCESS" else "ERROR", "code": "TASK_COMPLETED", "message": title}, items=items)
        target_items = data.get("target_items", [])
        items = [{"item_id": f"{record.task_id}-item-{index}", "product_name": item["product_name"], "status": "REPLENISHED" if item.get("placed") else "PICKED" if item.get("picked") else "PENDING", "status_label": "已完成补货" if item.get("placed") else "已取到" if item.get("picked") else "等待处理", "picked": item.get("picked", False), "placed": item.get("placed", False), "message": "商品已补回货架" if item.get("placed") else "等待补货"} for index, item in enumerate(target_items, 1)]
        placed = sum(item["placed"] for item in items)
        if not items:
            status = "SUCCEEDED"
            title = "货架检查完成"
            message = "本次检查未发现需要补货的商品"
        else:
            status = "SUCCEEDED" if placed == len(items) else "PARTIAL_SUCCESS" if placed else "FAILED"
            title = "补货完成" if status == "SUCCEEDED" else "补货部分完成" if status == "PARTIAL_SUCCESS" else "补货失败"
            message = f"已完成 {placed}/{len(items)} 件补货"
        return self._status(record, status, title, message, {"code": status, "label": title, "progress_percent": 100, "message": "补货流程已结束"}, None, {"shortage_items_found": len(items), "replenishment_items_placed": placed, "held_items": sum(item["picked"] and not item["placed"] for item in items)}, {"level": "SUCCESS" if status == "SUCCEEDED" else "WARNING" if status == "PARTIAL_SUCCESS" else "ERROR", "code": "TASK_COMPLETED", "message": title}, items=items)

    def _failed_status(self, record: _ExternalTaskRecord, error: Exception) -> dict[str, Any]:
        message = getattr(error, "message", str(error))
        return self._status(record, "FAILED", f"{record.task_name}失败", message, {"code": "FAILED", "label": f"{record.task_name}失败", "progress_percent": 0, "message": message}, None, self._initial_summary(record), {"level": "ERROR", "code": "TASK_FAILED", "message": "任务未能完成，请稍后重试或联系工作人员"}, error={"error_code": getattr(error, "code", type(error).__name__), "message": message, "step": getattr(error, "step", None)})

    def _status(self, record: _ExternalTaskRecord, status: str, title: str, message: str, current_step: dict[str, Any], next_step: dict[str, Any] | None, summary: dict[str, Any], notice: dict[str, Any], **details: Any) -> dict[str, Any]:
        payload = {"status": status, "display_title": title, "display_message": message, "current_step": current_step, "next_step": next_step, "location": {"code": "UNKNOWN", "label": "机器人任务区域"}, "summary": summary, "user_notice": notice, "last_updated_at": _now()}
        payload.update(details)
        return payload

    def _initial_summary(self, record: _ExternalTaskRecord) -> dict[str, Any]:
        if record.task_number == "0":
            return {"inspection_points_total": 0, "inspection_points_completed": 0, "captures_total": 0, "captures_completed": 0}
        if record.task_number == "1":
            return {"total_items": 2, "items_completed": 0, "items_in_progress": 0, "items_failed": 0, "items_held": 0}
        return {"inspection_points_total": 0, "inspection_points_completed": 0, "shortage_items_found": 0, "replenishment_items_placed": 0, "held_items": 0}

    async def _internal_payload(
        self,
        task_number: str,
        request: BaseModel,
        record: _ExternalTaskRecord,
    ) -> dict[str, Any]:
        if task_number == "1":
            body = request  # type: ignore[assignment]
            product_names = []
            for item in body.items:
                try:
                    sku = await self._coordinator().bindings["1"].orchestrator.client.search_by_sku(item.sku_id)
                except Task1ServiceError as exc:
                    if exc.code == "SKU_NOT_FOUND":
                        raise TaskServiceError(
                            "PRODUCT_NOT_FOUND",
                            f"SKU 不存在: {item.sku_id}",
                            status_code=422,
                        ) from exc
                    raise
                record.sku_names[item.sku_id.upper()] = sku.name
                product_names.append(sku.name)
            return {
                "order_source": "mock_random",
                "order_id": body.external_order_id,
                "product_names": product_names,
            }
        return {}

    def _callback_url(self, requested: str | None) -> str | None:
        value = (requested or self.settings.callback_url or "").strip()
        if not value:
            return None
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise TaskServiceError("INVALID_CALLBACK_URL", "status_callback_url 必须是合法的 HTTP(S) 地址", status_code=422)
        allowed = {host.lower() for host in self.settings.callback_allowed_hosts}
        if requested and not allowed:
            raise TaskServiceError("CALLBACK_URL_NOT_ALLOWED", "按请求传入的回调地址必须配置白名单", status_code=422)
        if allowed and parsed.hostname.lower() not in allowed:
            raise TaskServiceError("CALLBACK_URL_NOT_ALLOWED", "status_callback_url 不在允许的回调地址白名单中", status_code=422)
        return value

    def _authorize(self, authorization: str | None) -> None:
        expected = self.settings.access_token
        if expected is None:
            return
        if authorization != f"Bearer {expected}":
            raise TaskServiceError("UNAUTHORIZED", "缺少或无效的访问令牌", status_code=401)

    @staticmethod
    def _fingerprint(request: BaseModel) -> str:
        content = json.dumps(request.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _check_duplicate(record: _ExternalTaskRecord, task_number: str, fingerprint: str) -> None:
        if record.task_number != task_number or record.fingerprint != fingerprint:
            raise TaskServiceError("TASK_CONFLICT", "相同任务号或幂等键对应的请求内容不一致", status_code=409)

    @staticmethod
    def _task_identity(task_number: str) -> tuple[str, str]:
        return {"0": ("TASK0_INVENTORY", "理货"), "1": ("TASK1_PICKUP", "取货"), "2": ("TASK2_REPLENISHMENT", "补货")} [task_number]

    @staticmethod
    def _new_task_id(task_number: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"task{task_number}-{timestamp}-{uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")
