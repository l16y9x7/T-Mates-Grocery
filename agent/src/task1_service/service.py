"""任务一的串行编排流程。"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from task1_service.client import Task1Client
from task1_service.models import (
    Hand,
    PRODUCT_SLOT_PATTERN,
    TargetItem,
    Task1Request,
    Task1Result,
    Task1ServiceError,
    Task1Settings,
)


LOGGER = logging.getLogger(__name__)


def shelf_level(slot_id: str) -> str:
    return slot_id.split("_")[2]


class Task1Orchestrator:
    def __init__(self, settings: Task1Settings, client: Task1Client) -> None:
        self.settings = settings
        self.client = client

    async def run(self, request: Task1Request, operation_key: str | None = None) -> Task1Result:
        task_run_id = operation_key or uuid4().hex
        logger = _Task1Log(self.settings, task_run_id, request)
        step = "健康检查"
        try:
            logger.event("operation", "started", pick_count=request.pick_count)
            logger.event("健康检查", "started")
            await self.client.check_all_health()
            logger.event("健康检查", "succeeded")

            step = "小票点导航"
            logger.event("小票点导航", "started", target_id=self.settings.receipt_viewpoint)
            await self.client.navigate(self.settings.receipt_viewpoint, f"{task_run_id}:task1.receipt.navigate")
            logger.event("小票点导航", "succeeded", target_id=self.settings.receipt_viewpoint)

            step = "小票拍摄位姿"
            logger.event("小票拍摄位姿", "started", pose_type="RECEIPT_VIEW")
            await self.client.prepare_pose("RECEIPT_VIEW", f"{task_run_id}:task1.receipt.pose")
            logger.event("小票拍摄位姿", "succeeded", pose_type="RECEIPT_VIEW")

            step = "小票识别"
            logger.event("小票识别", "started")
            product_names = await self.client.parse_receipt()
            logger.event("小票识别", "succeeded", product_names=product_names)

            step = "SKU货位转换"
            targets: list[TargetItem] = []
            for index, name in enumerate(product_names):
                logger.event("SKU货位转换", "started", product_name=name)
                sku = await self.client.search_by_name(name)
                if len(sku.locations) != 1:
                    raise Task1ServiceError(
                        "AMBIGUOUS_PRODUCT_SLOT",
                        f"SKU {name} must resolve to exactly one location",
                        status_code=422,
                    )
                slot_id = sku.locations[0]
                if not PRODUCT_SLOT_PATTERN.fullmatch(slot_id):
                    raise Task1ServiceError(
                        "INVALID_PRODUCT_SLOT", f"invalid product slot: {slot_id}", status_code=422
                    )
                target = TargetItem(
                    product_name=name,
                    product_slot_id=slot_id,
                    shelf_level=shelf_level(slot_id),
                    hand=Hand.LEFT if index == 0 else Hand.RIGHT,
                )
                targets.append(target)
                logger.event(
                    "SKU货位转换",
                    "succeeded",
                    product_name=name,
                    sku_id=sku.sku_id,
                    product_slot_id=slot_id,
                    shelf_level=target.shelf_level,
                )
            if len({target.product_slot_id for target in targets}) != 2:
                raise Task1ServiceError(
                    "INVALID_RECEIPT", "products resolve to the same location", status_code=422
                )

            for index, target in enumerate(targets[: request.pick_count]):
                step = f"商品{index + 1}导航"
                logger.event("商品导航", "started", product_name=target.product_name, target_id=target.product_slot_id)
                await self.client.navigate(target.product_slot_id, f"{task_run_id}:task1.pick.{index}.navigate")
                logger.event("商品导航", "succeeded", product_name=target.product_name, target_id=target.product_slot_id)

                step = f"商品{index + 1}抓取位姿"
                logger.event(
                    "抓取位姿",
                    "started",
                    product_name=target.product_name,
                    pose_type="SHELF_PICK_READY",
                    shelf_level=target.shelf_level,
                )
                await self.client.prepare_pose(
                    "SHELF_PICK_READY",
                    f"{task_run_id}:task1.pick.{index}.pose",
                    shelf_level=target.shelf_level,
                )
                logger.event("抓取位姿", "succeeded", product_name=target.product_name, shelf_level=target.shelf_level)

                step = f"商品{index + 1}抓取"
                logger.event("抓取", "started", product_name=target.product_name, hand=target.hand.value)
                await self.client.pick(target.product_name, target.hand, f"{task_run_id}:task1.pick.{index}.pick")
                target.picked = True
                logger.event("抓取", "succeeded", product_name=target.product_name, hand=target.hand.value)

            held_items = {target.hand: target.product_name for target in targets if target.picked}
            result = Task1Result(
                task_run_id=task_run_id,
                task_type="SORTING",
                requested_pick_count=request.pick_count,
                status="SUCCEEDED",
                product_names=product_names,
                target_items=targets,
                held_items=held_items,
            )
            logger.event("operation", "succeeded", picked_count=request.pick_count)
            return result
        except Exception as exc:
            logger.event(
                "operation",
                "failed",
                step=step,
                error_code=getattr(exc, "code", type(exc).__name__),
                message=str(exc),
            )
            LOGGER.exception("任务一流程失败 step=%s key=%s", step, task_run_id)
            raise


class _Task1Log:
    """写入与 8086 相同的 operation.json/events.jsonl 结构。"""

    def __init__(self, settings: Task1Settings, operation_key: str, request: Task1Request) -> None:
        root = Path(settings.log_dir)
        root.mkdir(parents=True, exist_ok=True)
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", operation_key).strip("._") or "operation"
        self.directory = root / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{safe_key}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self._write_json(
            "operation.json",
            {
                "kind": "task1",
                "operation_key": operation_key,
                "request": request.model_dump(mode="json"),
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
            },
        )

    def _write_json(self, relative_path: str, payload: object) -> None:
        path = self.directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    def event(self, name: str, status: str, **details: object) -> None:
        try:
            path = self.directory / "events.jsonl"
            record = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "event": name,
                "status": status,
                **details,
            }
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except (OSError, TypeError, ValueError):
            LOGGER.exception("任务一日志写入失败 event=%s status=%s", name, status)
