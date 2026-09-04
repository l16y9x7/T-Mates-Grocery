"""Complete-flow mock implementation of the external task API."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from pydantic import BaseModel

from task_service.external import (
    Task0TriggerRequest,
    Task1TriggerRequest,
    Task2TriggerRequest,
)

from .settings import MockExternalSettings


LOGGER = logging.getLogger(__name__)
CallbackSender = Callable[[str, dict[str, Any], dict[str, str]], Awaitable[None]]


class MockTaskError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class _CallbackWorker:
    url: str
    access_token: str | None
    settings: MockExternalSettings
    sender: CallbackSender | None = None
    queue: asyncio.Queue[dict[str, Any] | None] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name="mock-external-callback")

    async def enqueue(self, payload: dict[str, Any]) -> None:
        await self.queue.put(payload)

    async def close(self) -> None:
        await self.queue.put(None)
        if self.task is not None:
            await self.task

    async def _run(self) -> None:
        if self.sender is not None:
            while True:
                payload = await self.queue.get()
                if payload is None:
                    return
                await self._send_with_retry(payload, None)

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            while True:
                payload = await self.queue.get()
                if payload is None:
                    return
                await self._send_with_retry(payload, client)

    async def _send_with_retry(self, payload: dict[str, Any], client: httpx.AsyncClient | None) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        timestamp = payload["occurred_at"]
        headers = {
            "Content-Type": "application/json",
            "X-Event-Id": payload["event_id"],
            "X-Task-Run-Id": payload["task_run_id"],
            "X-Signature-Timestamp": timestamp,
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
            signature = hmac.new(self.access_token.encode("utf-8"), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()
            headers["X-Signature"] = f"sha256={signature}"

        attempts = self.settings.max_retries + 1
        for attempt in range(attempts):
            try:
                if self.sender is not None:
                    await self.sender(self.url, payload, headers)
                else:
                    assert client is not None
                    response = await client.post(self.url, content=body, headers=headers)
                    if not 200 <= response.status_code < 300:
                        raise RuntimeError(f"HTTP {response.status_code}")
                return
            except Exception as exc:
                if attempt + 1 < attempts:
                    delay = self.settings.retry_backoff_seconds * (2**attempt)
                    if delay:
                        await asyncio.sleep(delay)
                else:
                    LOGGER.error("mock callback failed event_id=%s attempts=%s error=%s", payload["event_id"], attempts, exc)


@dataclass
class _TaskRecord:
    task_run_id: str
    task_number: str
    task_type: str
    task_name: str
    external_task_id: str
    external_order_id: str | None
    request: BaseModel
    fingerprint: str
    response: dict[str, Any]
    callback_worker: _CallbackWorker | None
    sequence: int = 0
    last_status: dict[str, Any] | None = None
    job: asyncio.Task[None] | None = None


class MockExternalService:
    def __init__(self, settings: MockExternalSettings, callback_sender: CallbackSender | None = None) -> None:
        self.settings = settings
        self.callback_sender = callback_sender
        self._lock = asyncio.Lock()
        self._records: dict[str, _TaskRecord] = {}
        self._by_idempotency: dict[str, _TaskRecord] = {}
        self._by_external_id: dict[str, _TaskRecord] = {}
        self._active_task_id: str | None = None
        self._task0_succeeded = False

    async def close(self) -> None:
        jobs = [record.job for record in self._records.values() if record.job is not None]
        for job in jobs:
            if not job.done():
                job.cancel()
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)
        workers = [record.callback_worker for record in self._records.values() if record.callback_worker is not None]
        for worker in workers:
            await worker.close()

    async def health(self, request_id: str) -> dict[str, Any]:
        active = self._active_task_id
        ready_for_task2 = self._task0_succeeded
        status = "BUSY" if active else "READY"
        active_task = None
        dependencies = [
            {"name": "task_orchestrator", "status": "READY", "latency_ms": 1},
            {"name": "navigation", "status": "READY", "latency_ms": 1},
            {"name": "pose", "status": "READY", "latency_ms": 1},
            {"name": "pick_place", "status": "READY", "latency_ms": 1},
            {"name": "sku", "status": "READY", "latency_ms": 1},
        ]
        if active:
            record = self._records[active]
            active_task = {
                "task_type": record.task_type,
                "task_name": record.task_name,
                "task_run_id": record.task_run_id,
            }
            dependencies = []
        return {
            "schema_version": "1.0",
            "request_id": request_id,
            "checked_at": _now(),
            "status": status,
            "ready_for_task0": active is None,
            "ready_for_task1": active is None,
            "ready_for_task2": active is None and ready_for_task2,
            "accepting_tasks": active is None,
            "active_task": active_task,
            "dependencies": dependencies,
        }

    async def submit(
        self,
        task_number: str,
        request: Task0TriggerRequest | Task1TriggerRequest | Task2TriggerRequest,
        idempotency_key: str,
        request_id: str,
    ) -> tuple[dict[str, Any], bool]:
        callback_url = _callback_url(request.status_callback_url or self.settings.callback_url)
        fingerprint = _fingerprint(request)
        async with self._lock:
            duplicate = self._by_idempotency.get(idempotency_key) or self._by_external_id.get(request.external_task_id)
            if duplicate is not None:
                self._check_duplicate(duplicate, task_number, fingerprint)
                response = dict(duplicate.response)
                response.update({"request_id": request_id, "duplicate": True, "status": (duplicate.last_status or {}).get("status", "ACCEPTED")})
                return response, True
            if self._active_task_id is not None:
                raise MockTaskError("TASK_BUSY", "当前有任务正在执行，请稍后重试", 409, True)
            if task_number == "2" and not self._task0_succeeded:
                raise MockTaskError("BASELINE_NOT_READY", "Task0 理货尚未成功完成，请先完成理货任务", 503, True)

            task_type, task_name = _task_identity(task_number)
            task_run_id = _new_task_id(task_number)
            callback_worker = _CallbackWorker(
                callback_url,
                self.settings.callback_access_token,
                self.settings,
                self.callback_sender,
            ) if callback_url else None
            response = {
                "schema_version": "1.0",
                "request_id": request_id,
                "external_task_id": request.external_task_id,
                "task_run_id": task_run_id,
                "task_type": task_type,
                "task_name": task_name,
                "status": "ACCEPTED",
                "accepted_at": _now(),
                "status_callback_enabled": callback_worker is not None,
            }
            if isinstance(request, Task1TriggerRequest):
                response["external_order_id"] = request.external_order_id
            record = _TaskRecord(
                task_run_id=task_run_id,
                task_number=task_number,
                task_type=task_type,
                task_name=task_name,
                external_task_id=request.external_task_id,
                external_order_id=getattr(request, "external_order_id", None),
                request=request,
                fingerprint=fingerprint,
                response=response,
                callback_worker=callback_worker,
            )
            self._records[task_run_id] = record
            self._by_idempotency[idempotency_key] = record
            self._by_external_id[request.external_task_id] = record
            self._active_task_id = task_run_id
            if callback_worker:
                callback_worker.start()
            record.job = asyncio.create_task(self._run(record), name=f"mock-external-{task_number}-{task_run_id}")
            return response, False

    async def get_status(self, task_run_id: str) -> dict[str, Any]:
        record = self._records.get(task_run_id)
        if record is None:
            raise MockTaskError("TASK_NOT_FOUND", "task_run_id not found", 404)
        return record.last_status or self._accepted_status(record)

    async def _run(self, record: _TaskRecord) -> None:
        try:
            await self._publish(record, "ACCEPTED", "已接收任务", "任务已经受理，准备启动", "ACCEPTED", 0, "UNKNOWN", "等待启动", self._initial_summary(record))
            await self._delay()
            await self._publish(record, "RUNNING", "正在检查设备", "正在检查任务所需设备", "HEALTH_CHECKING", 5, "UNKNOWN", "继续执行任务", self._initial_summary(record))
            await self._delay()
            if record.task_number == "0":
                await self._run_task0(record)
            elif record.task_number == "1":
                await self._run_task1(record)
            else:
                await self._run_task2(record)
            if record.task_number == "0":
                self._task0_succeeded = True
        finally:
            if self._active_task_id == record.task_run_id:
                self._active_task_id = None

    async def _run_task0(self, record: _TaskRecord) -> None:
        points = self.settings.inspection_points
        captures = [_capture(point, view, "PENDING") for point in points for view in ("UPPER", "LOWER")]
        for index, point in enumerate(points):
            if index == 0:
                await self._publish(record, "RUNNING", "正在巡检并记录货架状态", "正在前往第 1 个货架区域", "NAVIGATING_TO_START", 10, "START", "检查货架区域", _task0_summary(points, captures), captures=captures)
                await self._delay()
            await self._publish(record, "RUNNING", "正在巡检并记录货架状态", f"已到达第 {index + 1} 个货架区域，共 {len(points)} 个区域", "INSPECTING", 10 + index * 15, "SHELF", "记录货架信息", _task0_summary(points, captures), captures=captures)
            await self._delay()
            for view in ("UPPER", "LOWER"):
                current = next(item for item in captures if item["target_id"] == point and item["view"] == view)
                current.update(status="IN_PROGRESS", status_label="正在记录", message=f"{_target_label(point)}{_view_label(view)}信息正在记录")
                await self._publish(record, "RUNNING", "正在巡检并记录货架状态", f"正在记录第 {index + 1} 个货架区域，共 {len(points)} 个区域", "CAPTURING", 15 + int((index * 2 + (1 if view == "UPPER" else 2)) * 80 / (len(points) * 2)), "SHELF", "继续记录下一个货架区域", _task0_summary(points, captures), captures=captures)
                await self._delay()
                current.update(status="COMPLETED", status_label="已记录", message=f"{_target_label(point)}{_view_label(view)}信息已记录")
                await self._publish(record, "RUNNING", "正在巡检并记录货架状态", f"已记录第 {index + 1} 个货架区域，共 {len(points)} 个区域", "CAPTURING", 15 + int((index * 2 + (1 if view == "UPPER" else 2)) * 80 / (len(points) * 2)), "SHELF", "继续记录下一个货架区域", _task0_summary(points, captures), captures=captures)
                await self._delay()
        await self._publish(record, "RUNNING", "正在完成理货任务", "货架信息已全部记录，正在返回起点", "RETURNING_TO_START", 95, "START", "完成理货任务", _task0_summary(points, captures), captures=captures)
        await self._delay()
        await self._publish(record, "SUCCEEDED", "理货完成", "货架信息已全部记录", "SUCCEEDED", 100, "START", None, _task0_summary(points, captures), event_type="TASK_COMPLETED", captures=captures)

    async def _run_task1(self, record: _TaskRecord) -> None:
        request = record.request
        assert isinstance(request, Task1TriggerRequest)
        items = [_task1_item(item.sku_id, index) for index, item in enumerate(request.items)]
        await self._publish(record, "RUNNING", "正在为您取货", "订单已确认，准备处理两件商品", "ORDER_CONFIRMED", 10, "UNKNOWN", "解析商品货位", _task1_summary(items), items=items)
        await self._delay()
        for item in items:
            item.update(status="LOCATING", status_label="正在寻找商品", message="正在前往商品所在货架")
        await self._publish(record, "RUNNING", "正在为您取货", "正在确认订单商品和所在货位", "RESOLVING_PRODUCTS", 20, "UNKNOWN", "规划取货路线", _task1_summary(items), items=items)
        await self._delay()
        await self._publish(record, "RUNNING", "正在为您取货", "正在规划两件商品的取货顺序", "PLANNING", 25, "UNKNOWN", "前往商品货架", _task1_summary(items), items=items)
        await self._delay()
        for index, item in enumerate(items):
            item.update(status="LOCATING", status_label="正在寻找商品", message="正在前往商品所在货架")
            await self._publish(record, "RUNNING", "正在为您取货", f"正在前往第 {index + 1} 件商品所在货架", "NAVIGATING_TO_SHELF", 30 + index * 15, "SHELF", "开始取商品", _task1_summary(items), items=items)
            await self._delay()
            item.update(status="PICKING", status_label="正在取货", message=f"正在取{item['product_name']}")
            await self._publish(record, "RUNNING", "正在为您取货", f"正在处理第 {index + 1} 件商品", "PICKING", 35 + index * 15, "SHELF", "确认商品已取到", _task1_summary(items), items=items)
            await self._delay()
            item.update(status="PICKED", status_label="已取到", picked=True, message="商品已取到，准备送到交付台")
            await self._publish(record, "RUNNING", "正在为您取货", f"已取到第 {index + 1} 件商品", "PICKING", 45 + index * 15, "SHELF", "继续处理订单", _task1_summary(items), items=items)
            await self._delay()
        await self._publish(record, "RUNNING", "正在为您取货", "两件商品已取到，正在前往交付台", "NAVIGATING_TO_DELIVERY", 65, "DELIVERY_TABLE", "交付商品", _task1_summary(items), items=items)
        await self._delay()
        for index, item in enumerate(items):
            item.update(status="PLACING", status_label="正在交付", message="正在把商品放到交付台")
            await self._publish(record, "RUNNING", "正在交付商品", f"正在交付{item['product_name']}", "PLACING", 70 + index * 10, "DELIVERY_TABLE", "完成商品交付", _task1_summary(items), items=items)
            await self._delay()
            item.update(status="PLACED", status_label="已完成", placed=True, message="商品已放到交付台")
            await self._publish(record, "RUNNING", "正在交付商品", f"{item['product_name']}已完成交付", "PLACING", 75 + index * 15, "DELIVERY_TABLE", "前往任务判定区", _task1_summary(items), items=items)
            await self._delay()
        await self._publish(record, "RUNNING", "正在完成取货任务", "商品已交付，正在前往任务判定区", "FINISHING", 95, "TASK_BOUNDARY", "完成取货任务", _task1_summary(items), items=items)
        await self._delay()
        await self._publish(record, "SUCCEEDED", "取货完成", "已完成 2/2 件商品", "SUCCEEDED", 100, "TASK_BOUNDARY", None, _task1_summary(items), event_type="TASK_COMPLETED", items=items)

    async def _run_task2(self, record: _TaskRecord) -> None:
        points = self.settings.inspection_points
        items: list[dict[str, Any]] = []
        for index, point in enumerate(points):
            await self._publish(record, "RUNNING", "正在检查货架并准备补货", f"正在检查第 {index + 1} 个货架区域，共 {len(points)} 个区域", "INSPECTING_SHELF", 10 + int(index * 20 / len(points)), "SHELF", "继续检查下一个货架区域", _task2_summary(points, index, items), items=items)
            await self._delay()
        items = [_task2_item("SKU_001", "可口可乐罐装"), _task2_item("SKU_002", "百事可乐瓶装")]
        await self._publish(record, "RUNNING", "正在检查货架并准备补货", "已发现 2 件商品需要补货", "IDENTIFYING_SHORTAGE", 35, "SHELF", "规划补货路线", _task2_summary(points, len(points), items), items=items)
        await self._delay()
        await self._publish(record, "RUNNING", "正在检查货架并准备补货", "正在规划补货商品和目标货位", "PLANNING_REPLENISHMENT", 45, "UNKNOWN", "前往补货台", _task2_summary(points, len(points), items), items=items)
        await self._delay()
        await self._publish(record, "RUNNING", "正在执行补货", "正在前往补货台取商品", "NAVIGATING_TO_REPLENISHMENT", 55, "UNKNOWN", "取补货商品", _task2_summary(points, len(points), items), items=items)
        await self._delay()
        for index, item in enumerate(items):
            item.update(status="PICKING", status_label="正在取补货商品", message="正在从补货台取出商品")
            await self._publish(record, "RUNNING", "正在执行补货", f"正在取{item['product_name']}", "PICKING_REPLENISHMENT", 57 + index * 5, "UNKNOWN", "返回目标货架", _task2_summary(points, len(points), items), items=items)
            await self._delay()
            item.update(status="PICKED", status_label="已取到", picked=True, message="补货商品已取到，准备送回货架")
            await self._publish(record, "RUNNING", "正在执行补货", f"已取到{item['product_name']}", "PICKING_REPLENISHMENT", 60 + index * 5, "UNKNOWN", "返回目标货架", _task2_summary(points, len(points), items), items=items)
            await self._delay()
        await self._publish(record, "RUNNING", "正在执行补货", "正在把补货商品送回目标货架", "NAVIGATING_TO_SHELF", 75, "SHELF", "放置补货商品", _task2_summary(points, len(points), items), items=items)
        await self._delay()
        for index, item in enumerate(items):
            item.update(status="PLACING", status_label="正在补回货架", message="正在把商品放回正确位置")
            await self._publish(record, "RUNNING", "正在执行补货", f"正在补回{item['product_name']}", "PLACING_REPLENISHMENT", 78 + index * 8, "SHELF", "完成补货", _task2_summary(points, len(points), items), items=items)
            await self._delay()
            item.update(status="REPLENISHED", status_label="已完成补货", placed=True, message="商品已补回货架")
            await self._publish(record, "RUNNING", "正在执行补货", f"{item['product_name']}已补回货架", "PLACING_REPLENISHMENT", 82 + index * 8, "SHELF", "前往任务判定区", _task2_summary(points, len(points), items), items=items)
            await self._delay()
        await self._publish(record, "RUNNING", "正在完成补货任务", "补货已完成，正在前往任务判定区", "FINISHING", 95, "TASK_BOUNDARY", "完成补货任务", _task2_summary(points, len(points), items), items=items)
        await self._delay()
        await self._publish(record, "SUCCEEDED", "补货完成", "已完成 2/2 件补货", "SUCCEEDED", 100, "TASK_BOUNDARY", None, _task2_summary(points, len(points), items), event_type="TASK_COMPLETED", items=items)

    async def _publish(self, record: _TaskRecord, status: str, title: str, message: str, code: str, progress: int, location_code: str, next_label: str | None, summary: dict[str, Any], *, event_type: str | None = None, captures: list[dict[str, Any]] | None = None, items: list[dict[str, Any]] | None = None) -> None:
        occurred_at = _now()
        record.sequence += 1
        payload = {
            "schema_version": "1.0",
            "event_id": f"evt-{record.task_run_id}-{record.sequence}-{uuid4().hex[:8]}",
            "sequence": record.sequence,
            "event_type": event_type or ("TASK_ACCEPTED" if status == "ACCEPTED" else "TASK_PROGRESS"),
            "occurred_at": occurred_at,
            "external_task_id": record.external_task_id,
            "external_order_id": record.external_order_id,
            "task_run_id": record.task_run_id,
            "task_type": record.task_type,
            "task_name": record.task_name,
            "status": status,
            "display_title": title,
            "display_message": message,
            "current_step": {"code": code, "label": _step_label(code, record.task_name), "progress_percent": progress, "message": message},
            "location": {"code": location_code, "label": _location_label(location_code)},
            "next_step": {"code": "UNKNOWN", "label": next_label} if next_label else None,
            "estimated_remaining_seconds": max(0, int((100 - progress) * max(self.settings.stage_delay_seconds, 1))),
            "summary": summary,
            "user_notice": {"level": "SUCCESS" if status == "SUCCEEDED" else "INFO", "code": "TASK_SUCCEEDED" if status == "SUCCEEDED" else "TASK_ACCEPTED" if status == "ACCEPTED" else "TASK_IN_PROGRESS", "message": message},
            "last_updated_at": occurred_at,
            "error": None,
        }
        if record.task_number == "0":
            payload["captures"] = captures or []
        else:
            payload["items"] = items or []
        record.last_status = payload
        if record.callback_worker:
            await record.callback_worker.enqueue(payload)

    def _accepted_status(self, record: _TaskRecord) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "event_id": f"evt-{record.task_run_id}-0",
            "sequence": 0,
            "event_type": "TASK_ACCEPTED",
            "occurred_at": record.response["accepted_at"],
            "external_task_id": record.external_task_id,
            "external_order_id": record.external_order_id,
            "task_run_id": record.task_run_id,
            "task_type": record.task_type,
            "task_name": record.task_name,
            "status": "ACCEPTED",
            "display_title": f"已接收{record.task_name}任务",
            "display_message": f"已接收{record.task_name}任务，准备启动",
            "current_step": {"code": "ACCEPTED", "label": "已接收任务", "progress_percent": 0, "message": "任务准备启动"},
            "location": {"code": "UNKNOWN", "label": "位置未知"},
            "next_step": {"code": "HEALTH_CHECKING", "label": "检查设备"},
            "estimated_remaining_seconds": 0,
            "summary": self._initial_summary(record),
            "user_notice": {"level": "INFO", "code": "TASK_ACCEPTED", "message": f"已接收{record.task_name}任务"},
            "last_updated_at": record.response["accepted_at"],
            "error": None,
            "captures": [] if record.task_number == "0" else None,
            "items": [] if record.task_number != "0" else None,
        }

    def _initial_summary(self, record: _TaskRecord) -> dict[str, int]:
        if record.task_number == "0":
            total = len(self.settings.inspection_points)
            return {"inspection_points_total": total, "inspection_points_completed": 0, "captures_total": total * 2, "captures_completed": 0, "captures_failed": 0}
        if record.task_number == "1":
            return {"total_items": 2, "items_completed": 0, "items_in_progress": 0, "items_failed": 0, "items_held": 0}
        return {"inspection_points_total": len(self.settings.inspection_points), "inspection_points_completed": 0, "shortage_items_found": 0, "replenishment_items_picked": 0, "replenishment_items_placed": 0, "held_items": 0}

    async def _delay(self) -> None:
        if self.settings.stage_delay_seconds:
            await asyncio.sleep(self.settings.stage_delay_seconds)

    @staticmethod
    def _check_duplicate(record: _TaskRecord, task_number: str, fingerprint: str) -> None:
        if record.task_number != task_number or record.fingerprint != fingerprint:
            raise MockTaskError("TASK_CONFLICT", "相同任务号或幂等键对应的请求内容不一致", 409)


def _callback_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise MockTaskError("INVALID_CALLBACK_URL", "status_callback_url 必须是合法的 HTTP(S) 地址", 422)
    return value.strip()


def _fingerprint(request: BaseModel) -> str:
    content = json.dumps(request.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _task_identity(task_number: str) -> tuple[str, str]:
    return {"0": ("TASK0_INVENTORY", "理货"), "1": ("TASK1_PICKUP", "取货"), "2": ("TASK2_REPLENISHMENT", "补货")}[task_number]


def _new_task_id(task_number: str) -> str:
    return f"task{task_number}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _target_label(target_id: str) -> str:
    return {"H1_INSPECT": "1 号货架", "H12_INSPECT": "1-2 号货架", "H2_INSPECT": "2 号货架", "H23_INSPECT": "2-3 号货架", "H3_INSPECT": "3 号货架"}.get(target_id, target_id)


def _view_label(view: str) -> str:
    return "上层" if view == "UPPER" else "下层"


def _capture(target_id: str, view: str, status: str) -> dict[str, Any]:
    labels = {"PENDING": "待记录", "IN_PROGRESS": "正在记录", "COMPLETED": "已记录"}
    return {"target_id": target_id, "target_label": _target_label(target_id), "view": view, "status": status, "status_label": labels[status], "message": f"{_target_label(target_id)}{_view_label(view)}信息{labels[status]}"}


def _task0_summary(points: list[str], captures: list[dict[str, Any]]) -> dict[str, int]:
    completed = sum(item["status"] == "COMPLETED" for item in captures)
    return {"inspection_points_total": len(points), "inspection_points_completed": completed // 2, "captures_total": len(captures), "captures_completed": completed, "captures_failed": 0}


def _task1_item(sku_id: str, index: int) -> dict[str, Any]:
    names = {"SKU_001": "可口可乐罐装", "SKU_002": "百事可乐瓶装"}
    product_name = names.get(sku_id.upper(), f"Mock 商品 {index + 1}")
    return {"sku_id": sku_id, "product_name": product_name, "status": "PENDING", "status_label": "等待处理", "picked": False, "placed": False, "message": "等待处理"}


def _task1_summary(items: list[dict[str, Any]]) -> dict[str, int]:
    return {"total_items": len(items), "items_completed": sum(item["placed"] for item in items), "items_in_progress": sum(item["status"] not in {"PENDING", "PLACED"} for item in items), "items_failed": 0, "items_held": sum(item["picked"] and not item["placed"] for item in items)}


def _task2_item(sku_id: str, product_name: str) -> dict[str, Any]:
    return {"sku_id": sku_id, "product_name": product_name, "status": "SHORTAGE_FOUND", "status_label": "发现缺货", "picked": False, "placed": False, "message": f"发现{product_name}缺货，准备补货"}


def _task2_summary(points: list[str], completed_points: int, items: list[dict[str, Any]]) -> dict[str, int]:
    return {"inspection_points_total": len(points), "inspection_points_completed": completed_points, "shortage_items_found": len(items), "replenishment_items_picked": sum(item["picked"] for item in items), "replenishment_items_placed": sum(item["placed"] for item in items), "held_items": sum(item["picked"] and not item["placed"] for item in items)}


def _step_label(code: str, task_name: str) -> str:
    labels = {
        "ACCEPTED": "已接收任务",
        "HEALTH_CHECKING": "正在检查设备",
        "NAVIGATING_TO_START": "正在前往起点",
        "INSPECTING": "正在巡检货架区域",
        "CAPTURING": "正在记录货架信息",
        "RETURNING_TO_START": "正在返回起点",
        "ORDER_CONFIRMED": "订单已确认",
        "RESOLVING_PRODUCTS": "正在定位商品",
        "PLANNING": "正在规划取货",
        "NAVIGATING_TO_SHELF": "正在前往货架",
        "PICKING": "正在取商品",
        "NAVIGATING_TO_DELIVERY": "正在前往交付台",
        "PLACING": "正在交付商品",
        "FINISHING": "正在完成任务",
        "INSPECTING_SHELF": "正在检查货架",
        "IDENTIFYING_SHORTAGE": "正在确认缺货商品",
        "PLANNING_REPLENISHMENT": "正在规划补货",
        "NAVIGATING_TO_REPLENISHMENT": "正在前往补货台",
        "PICKING_REPLENISHMENT": "正在取补货商品",
        "PLACING_REPLENISHMENT": "正在放置补货商品",
        "SUCCEEDED": f"{task_name}完成",
    }
    return labels.get(code, code)


def _location_label(code: str) -> str:
    return {"START": "起点", "SHELF": "货架区域", "DELIVERY_TABLE": "交付台", "TASK_BOUNDARY": "任务判定区", "UNKNOWN": "位置未知"}.get(code, code)
