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


_INSPECTION_LEFT_MAX = {
    "H1_F": (3, 6, 5, 3, 3),
    "H1_B": (3, 3, 2, 2, 1),
    "H2_F": (3, 2, 4, 6, 5),
    "H2_B": (3, 4, 5, 2, 1),
}


def default_inspection_target_id(slot_id: str) -> str:
    """Return the inspection point for legacy in-memory settings without the map."""

    shelf, face, level, column = slot_id.split("_")
    shelf_face = f"{shelf}_{face}"
    left_max = _INSPECTION_LEFT_MAX[shelf_face][int(level[1:]) - 1]
    side = "L" if int(column[1:]) <= left_max else "R"
    return f"{shelf_face}_{side}_INSPECT"


class Task1Orchestrator:
    def __init__(self, settings: Task1Settings, client: Task1Client) -> None:
        self.settings = settings
        self.client = client

    async def run(self, request: Task1Request, operation_key: str | None = None) -> Task1Result:
        task_run_id = operation_key or uuid4().hex
        logger = _Task1Log(self.settings, task_run_id, request)
        self.client.set_trace_callback(logger.interface_event)
        navigation_state: dict[str, str | None] = {"target_id": None}
        step = "健康检查"
        try:
            logger.event("operation", "started")
            logger.event("健康检查", "started")
            await self.client.check_all_health()
            logger.event("健康检查", "succeeded")

            step = "小票点导航"
            logger.event("小票点导航", "started", target_id=self.settings.receipt_viewpoint)
            await self._navigate(
                self.settings.receipt_viewpoint,
                f"{task_run_id}:task1.receipt.navigate",
                logger,
                navigation_state,
            )
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
            for name in product_names:
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
                target_id = self._target_id(slot_id)
                targets.append(TargetItem(
                    product_name=name,
                    product_slot_id=slot_id,
                    target_id=target_id,
                    shelf_level=shelf_level(slot_id),
                    hand=Hand.LEFT,
                ))
                logger.event(
                    "SKU货位转换",
                    "succeeded",
                    product_name=name,
                    sku_id=sku.sku_id,
                    product_slot_id=slot_id,
                    target_id=target_id,
                    shelf_level=shelf_level(slot_id),
                )
            if len({target.product_slot_id for target in targets}) != 2:
                raise Task1ServiceError(
                    "INVALID_RECEIPT", "products resolve to the same location", status_code=422
                )

            hands = self._assign_hands([target.product_slot_id for target in targets])
            for target, hand in zip(targets, hands):
                target.hand = hand
                logger.event(
                    "抓取手分配", "succeeded", product_name=target.product_name,
                    product_slot_id=target.product_slot_id, hand=hand.value,
                    allowed_hands=[item.value if isinstance(item, Hand) else str(item) for item in self._allowed_hands(target.product_slot_id)],
                )

            if targets[0].hand != targets[1].hand:
                for index, target in enumerate(targets):
                    await self._pick_target(target, index, task_run_id, logger, navigation_state)
                await self._prepare_delivery(task_run_id, logger, navigation_state=navigation_state)
                for index, target in enumerate(targets):
                    await self._place_target(target, index, task_run_id, logger)
            else:
                # 单手能力受限时，始终保持该手一次只持有一件商品。
                for index, target in enumerate(targets):
                    await self._pick_target(target, index, task_run_id, logger, navigation_state)
                    await self._prepare_delivery(task_run_id, logger, index, navigation_state)
                    await self._place_target(target, index, task_run_id, logger)

            step = "任务判定区导航"
            logger.event("任务判定区导航", "started", target_id=self.settings.task_boundary)
            await self._navigate(
                self.settings.task_boundary,
                f"{task_run_id}:task1.finish.navigate",
                logger,
                navigation_state,
            )
            logger.event("任务判定区导航", "succeeded", target_id=self.settings.task_boundary)

            held_items: dict[Hand, str] = {}
            result = Task1Result(
                task_run_id=task_run_id,
                task_type="SORTING",
                status="SUCCEEDED",
                product_names=product_names,
                target_items=targets,
                held_items=held_items,
            )
            logger.event("operation", "succeeded", picked_count=2, placed_count=2)
            return result
        except Exception as exc:
            if isinstance(exc, Task1ServiceError):
                exc.step = step
            logger.event(
                "operation",
                "failed",
                step=step,
                error_code=getattr(exc, "code", type(exc).__name__),
                message=str(exc),
            )
            LOGGER.exception("任务一流程失败 step=%s key=%s", step, task_run_id)
            raise
        finally:
            self.client.set_trace_callback(None)

    def _allowed_hands(self, slot_id: str) -> list[Hand]:
        return self.settings.product_hand_options.get(slot_id, [Hand.LEFT, Hand.RIGHT])

    def _target_id(self, slot_id: str) -> str:
        return self.settings.product_target_ids.get(slot_id) or default_inspection_target_id(slot_id)

    def _assign_hands(self, slots: list[str]) -> tuple[Hand, Hand]:
        options = [self._allowed_hands(slot) for slot in slots]
        candidates = [
            (Hand.LEFT, Hand.RIGHT),
            (Hand.RIGHT, Hand.LEFT),
            (Hand.LEFT, Hand.LEFT),
            (Hand.RIGHT, Hand.RIGHT),
        ]
        for candidate in candidates:
            if candidate[0] in options[0] and candidate[1] in options[1]:
                return candidate
        raise Task1ServiceError(
            "NO_FEASIBLE_HAND_ASSIGNMENT",
            f"no feasible hand assignment for product slots: {', '.join(slots)}",
            status_code=422,
        )

    async def _pick_target(
        self,
        target: TargetItem,
        index: int,
        task_run_id: str,
        logger: _Task1Log,
        navigation_state: dict[str, str | None],
    ) -> None:
        logger.event("商品导航", "started", product_name=target.product_name, target_id=target.target_id)
        await self._navigate(
            target.target_id,
            f"{task_run_id}:task1.pick.{index}.navigate",
            logger,
            navigation_state,
        )
        logger.event("商品导航", "succeeded", product_name=target.product_name, target_id=target.target_id)
        logger.event("抓取位姿", "started", product_name=target.product_name, pose_type="SHELF_PICK_READY", shelf_level=target.shelf_level)
        await self.client.prepare_pose("SHELF_PICK_READY", f"{task_run_id}:task1.pick.{index}.pose", shelf_level=target.shelf_level)
        logger.event("抓取位姿", "succeeded", product_name=target.product_name, shelf_level=target.shelf_level)
        logger.event("抓取", "started", product_name=target.product_name, hand=target.hand.value)
        await self.client.pick(target.product_name, target.hand, f"{task_run_id}:task1.pick.{index}.pick")
        target.picked = True
        logger.event("抓取", "succeeded", product_name=target.product_name, hand=target.hand.value)

    async def _prepare_delivery(
        self,
        task_run_id: str,
        logger: _Task1Log,
        cycle: int | None = None,
        navigation_state: dict[str, str | None] | None = None,
    ) -> None:
        if navigation_state is None:
            navigation_state = {"target_id": None}
        suffix = "delivery" if cycle is None else f"delivery.{cycle}"
        logger.event("交付台导航", "started", target_id=self.settings.delivery_place)
        await self._navigate(
            self.settings.delivery_place,
            f"{task_run_id}:task1.{suffix}.navigate",
            logger,
            navigation_state,
        )
        logger.event("交付台导航", "succeeded", target_id=self.settings.delivery_place)
        logger.event("放置位姿", "started", pose_type="DELIVERY_TABLE_PLACE_READY")
        await self.client.prepare_pose("DELIVERY_TABLE_PLACE_READY", f"{task_run_id}:task1.{suffix}.pose")
        logger.event("放置位姿", "succeeded", pose_type="DELIVERY_TABLE_PLACE_READY")

    async def _place_target(self, target: TargetItem, index: int, task_run_id: str, logger: _Task1Log) -> None:
        logger.event("放置", "started", product_name=target.product_name, hand=target.hand.value)
        await self.client.place(target.product_name, target.hand, f"{task_run_id}:task1.place.{index}.place")
        target.placed = True
        logger.event("放置", "succeeded", product_name=target.product_name, hand=target.hand.value)

    async def _navigate(
        self,
        target_id: str,
        idempotency_key: str,
        logger: _Task1Log,
        navigation_state: dict[str, str | None],
    ) -> None:
        """收回机器人姿态后移动；已在目标点时复用当前位置。"""

        if navigation_state["target_id"] == target_id:
            logger.event("移动前复位", "skipped", target_id=target_id, reason="already_at_target")
            return

        logger.event("移动前复位", "started", pose_type="START_POSITION", target_id=target_id)
        await self.client.prepare_pose("START_POSITION", f"{idempotency_key}.reset_pose")
        logger.event("移动前复位", "succeeded", pose_type="START_POSITION", target_id=target_id)
        await self.client.navigate(target_id, idempotency_key)
        navigation_state["target_id"] = target_id


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

    def interface_event(self, trace: dict[str, object]) -> None:
        status_code = trace.get("status_code")
        status = "succeeded" if isinstance(status_code, int) and status_code < 400 else "failed"
        self.event(
            "接口调用",
            status,
            interface=trace.get("interface"),
            service=trace.get("service"),
            method=trace.get("method"),
            url=trace.get("url"),
            request={
                "headers": trace.get("headers") or {},
                "query": trace.get("query") or {},
                "body": trace.get("body"),
            },
            attempt=trace.get("attempt"),
            response={
                "status_code": status_code,
                "headers": trace.get("response_headers") or {},
                "body": trace.get("response_body"),
                "error": trace.get("error"),
            },
        )
