"""任务二的串行巡检与补货流程。"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from task2_service.client import Task2Client
from task2_service.models import (
    Hand,
    InspectionPose,
    TargetItem,
    Task2Request,
    Task2Result,
    Task2ServiceError,
    Task2Settings,
)


LOGGER = logging.getLogger(__name__)
POSE_LEVELS = {
    InspectionPose.UPPER: {"L1", "L2"},
    InspectionPose.LOWER: {"L3", "L4", "L5"},
}


@dataclass(frozen=True)
class FindingContext:
    product_name: str
    inspection_target_id: str
    inspection_pose_type: InspectionPose


class Task2Orchestrator:
    def __init__(self, settings: Task2Settings, client: Task2Client) -> None:
        self.settings = settings
        self.client = client

    async def ready(self) -> bool:
        return self.baselines_ready() and await self.client.health_ready()

    def baselines_ready(self) -> bool:
        return all(
            self._baseline_path(target_id, pose).is_file()
            and self._baseline_path(target_id, pose).stat().st_size > 0
            for target_id in self.settings.inspection_points
            for pose in (InspectionPose.UPPER, InspectionPose.LOWER)
        )

    async def run(
        self, request: Task2Request, operation_key: str | None = None
    ) -> Task2Result:
        task_run_id = operation_key or uuid4().hex
        logger = _Task2Log(self.settings, task_run_id, request)
        self.client.set_trace_callback(logger.interface_event)
        navigation_state: dict[str, str | None] = {"target_id": None}
        held_items: dict[Hand, str] = {}
        action_failures: list[dict[str, str]] = []
        step = "健康检查"
        try:
            logger.event("operation", "started")
            logger.event("健康检查", "started")
            await self.client.check_all_health()
            self._require_baselines()
            logger.event("健康检查", "succeeded")

            step = "货架巡检"
            findings, inspection_pass = await self._inspect_until_two(
                task_run_id, logger, navigation_state
            )
            hands = self._assign_hands(findings)
            targets = [
                TargetItem(
                    product_name=finding.product_name,
                    inspection_target_id=finding.inspection_target_id,
                    inspection_pose_type=finding.inspection_pose_type,
                    hand=hand,
                )
                for finding, hand in zip(findings, hands)
            ]
            for target in targets:
                logger.event(
                    "抓取手分配",
                    "succeeded",
                    product_name=target.product_name,
                    inspection_target_id=target.inspection_target_id,
                    inspection_pose_type=target.inspection_pose_type.value,
                    hand=target.hand.value,
                    allowed_hands=[
                        hand.value for hand in self._allowed_hands(
                            FindingContext(
                                target.product_name,
                                target.inspection_target_id,
                                target.inspection_pose_type,
                            )
                        )
                    ],
                )

            if targets[0].hand != targets[1].hand:
                step = "补货台准备"
                await self._prepare_replenishment(
                    task_run_id, logger, navigation_state
                )
                for index, target in enumerate(targets):
                    step = "补货商品抓取"
                    await self._pick_target(
                        target,
                        index,
                        task_run_id,
                        logger,
                        held_items,
                        action_failures,
                    )
                for index, target in enumerate(targets):
                    step = "货架商品放置"
                    await self._place_target(
                        target,
                        index,
                        task_run_id,
                        logger,
                        navigation_state,
                        held_items,
                        action_failures,
                    )
            else:
                for index, target in enumerate(targets):
                    if target.hand in held_items:
                        failure = {
                            "operation": "pick",
                            "product_name": target.product_name,
                            "hand": target.hand.value,
                            "error_code": "HAND_OCCUPIED",
                            "message": f"{target.hand.value} hand already holds an item",
                        }
                        action_failures.append(failure)
                        logger.event("补货商品抓取", "skipped", **failure)
                        logger.event(
                            "货架商品放置",
                            "skipped",
                            product_name=target.product_name,
                            hand=target.hand.value,
                            reason="pick_not_succeeded",
                        )
                        continue
                    step = "补货台准备"
                    await self._prepare_replenishment(
                        task_run_id, logger, navigation_state, cycle=index
                    )
                    step = "补货商品抓取"
                    picked = await self._pick_target(
                        target,
                        index,
                        task_run_id,
                        logger,
                        held_items,
                        action_failures,
                    )
                    if not picked:
                        logger.event(
                            "货架商品放置",
                            "skipped",
                            product_name=target.product_name,
                            hand=target.hand.value,
                            reason="pick_not_succeeded",
                        )
                        continue
                    step = "货架商品放置"
                    await self._place_target(
                        target,
                        index,
                        task_run_id,
                        logger,
                        navigation_state,
                        held_items,
                        action_failures,
                    )

            if action_failures:
                step = "抓放失败汇总"
                raise Task2ServiceError(
                    "TASK_ACTIONS_FAILED",
                    f"{len(action_failures)} pick/place actions failed after nudge recovery",
                )
            step = "任务判定区导航"
            logger.event(
                "任务判定区导航", "started", target_id=self.settings.task_boundary
            )
            await self._navigate(
                self.settings.task_boundary,
                f"{task_run_id}:task2.finish.navigate",
                logger,
                navigation_state,
            )
            logger.event(
                "任务判定区导航", "succeeded", target_id=self.settings.task_boundary
            )

            result = Task2Result(
                task_run_id=task_run_id,
                task_type="SHORTAGE",
                status="SUCCEEDED",
                inspection_pass=inspection_pass,
                product_names=[target.product_name for target in targets],
                target_items=targets,
                held_items=held_items,
            )
            logger.event("operation", "succeeded", picked_count=2, placed_count=2)
            return result
        except Exception as exc:
            original_step = step
            if isinstance(exc, Task2ServiceError):
                exc.step = original_step
            try:
                navigation_state["target_id"] = None
                logger.event(
                    "失败回开始点",
                    "started",
                    target_id=self.settings.start_target_id,
                    original_step=original_step,
                    original_error_code=getattr(exc, "code", type(exc).__name__),
                )
                await self._navigate(
                    self.settings.start_target_id,
                    f"{task_run_id}:task2.failure.start.navigate",
                    logger,
                    navigation_state,
                )
                logger.event(
                    "失败回开始点",
                    "succeeded",
                    target_id=self.settings.start_target_id,
                    original_step=original_step,
                )
            except Exception as recovery_exc:
                logger.event(
                    "失败回开始点",
                    "failed",
                    target_id=self.settings.start_target_id,
                    original_step=original_step,
                    error_code=getattr(recovery_exc, "code", type(recovery_exc).__name__),
                    message=str(recovery_exc),
                )
                recovery_error = Task2ServiceError(
                    "FAILURE_RECOVERY_FAILED",
                    f"task failed at {original_step}; navigation back to start also failed: {recovery_exc}",
                )
                recovery_error.step = "失败回开始点"
                logger.event(
                    "operation",
                    "failed",
                    step=recovery_error.step,
                    error_code=recovery_error.code,
                    message=recovery_error.message,
                    original_step=original_step,
                    original_error_code=getattr(exc, "code", type(exc).__name__),
                    original_message=str(exc),
                )
                LOGGER.exception(
                    "任务二失败且无法返回开始点 step=%s key=%s",
                    original_step,
                    task_run_id,
                )
                raise recovery_error from recovery_exc
            logger.event(
                "operation",
                "failed",
                step=original_step,
                error_code=getattr(exc, "code", type(exc).__name__),
                message=str(exc),
            )
            LOGGER.exception("任务二流程失败 step=%s key=%s", original_step, task_run_id)
            raise
        finally:
            self.client.set_trace_callback(None)

    async def _inspect_until_two(
        self,
        task_run_id: str,
        logger: "_Task2Log",
        navigation_state: dict[str, str | None],
    ) -> tuple[list[FindingContext], int]:
        findings: list[FindingContext] = []
        seen: set[tuple[str, str, InspectionPose]] = set()
        index = 0
        direction = 1
        inspection_pass = 1

        while len(findings) < 2:
            target_id = self.settings.inspection_points[index]
            location_id = self.settings.location_id_for_target(target_id)
            logger.event(
                "巡检点导航",
                "started",
                inspection_pass=inspection_pass,
                target_id=target_id,
            )
            await self._navigate(
                target_id,
                f"{task_run_id}:task2.inspect.{inspection_pass}.{index}.navigate",
                logger,
                navigation_state,
            )
            logger.event(
                "巡检点导航",
                "succeeded",
                inspection_pass=inspection_pass,
                target_id=target_id,
            )

            for pose in (InspectionPose.UPPER, InspectionPose.LOWER):
                action_prefix = (
                    f"{task_run_id}:task2.inspect.{inspection_pass}.{index}."
                    f"{pose.value.lower()}"
                )
                logger.event(
                    "巡检观察位姿",
                    "started",
                    target_id=target_id,
                    pose_type=pose.value,
                )
                await self.client.prepare_pose(pose.value, f"{action_prefix}.pose")
                logger.event(
                    "巡检观察位姿",
                    "succeeded",
                    target_id=target_id,
                    pose_type=pose.value,
                )
                baseline_path = self._baseline_path(target_id, pose)
                logger.event(
                    "缺货识别",
                    "started",
                    target_id=target_id,
                    location_id=location_id,
                    pose_type=pose.value,
                    baseline_path=str(baseline_path),
                )
                names = await self.client.inspect(
                    location_id,
                    pose.value,
                )
                logger.event(
                    "缺货识别",
                    "succeeded",
                    target_id=target_id,
                    location_id=location_id,
                    pose_type=pose.value,
                    baseline_path=str(baseline_path),
                    findings=names,
                )
                for product_name in names:
                    key = (product_name, target_id, pose)
                    if key in seen:
                        continue
                    finding = FindingContext(product_name, target_id, pose)
                    self._allowed_hands(finding)
                    seen.add(key)
                    findings.append(finding)
                    logger.event(
                        "缺货记录",
                        "succeeded",
                        product_name=product_name,
                        target_id=target_id,
                        pose_type=pose.value,
                        accumulated_count=len(findings),
                    )
                    if len(findings) > 2:
                        raise Task2ServiceError(
                            "INVALID_FINDINGS",
                            "accumulated inspection findings exceed two products",
                            status_code=422,
                        )
                if len(findings) == 2:
                    return findings, inspection_pass

            next_index = index + direction
            if 0 <= next_index < len(self.settings.inspection_points):
                index = next_index
            else:
                inspection_pass += 1
                direction *= -1

        raise AssertionError("unreachable inspection state")

    def _allowed_hands(self, finding: FindingContext) -> list[Hand]:
        visible_levels = POSE_LEVELS[finding.inspection_pose_type]
        candidates = [
            option.hands
            for slot_id, option in self.settings.product_hand_options.items()
            if option.product_name == finding.product_name
            and option.target_id == finding.inspection_target_id
            and slot_id.split("_")[2] in visible_levels
        ]
        if not candidates:
            raise Task2ServiceError(
                "UNKNOWN_PRODUCT_HAND_OPTIONS",
                "no hand configuration matches "
                f"{finding.product_name} at {finding.inspection_target_id} "
                f"in {finding.inspection_pose_type.value}",
                status_code=422,
            )
        allowed = [
            hand
            for hand in (Hand.LEFT, Hand.RIGHT)
            if all(hand in candidate for candidate in candidates)
        ]
        if not allowed:
            raise Task2ServiceError(
                "NO_SAFE_HAND_OPTION",
                "matching product slots do not share a safe hand option for "
                f"{finding.product_name}",
                status_code=422,
            )
        return allowed

    def _assign_hands(
        self, findings: list[FindingContext]
    ) -> tuple[Hand, Hand]:
        options = [self._allowed_hands(finding) for finding in findings]
        candidates = (
            (Hand.LEFT, Hand.RIGHT),
            (Hand.RIGHT, Hand.LEFT),
            (Hand.LEFT, Hand.LEFT),
            (Hand.RIGHT, Hand.RIGHT),
        )
        for candidate in candidates:
            if candidate[0] in options[0] and candidate[1] in options[1]:
                return candidate
        raise Task2ServiceError(
            "NO_FEASIBLE_HAND_ASSIGNMENT",
            "no feasible hand assignment for detected shortage products",
            status_code=422,
        )

    async def _prepare_replenishment(
        self,
        task_run_id: str,
        logger: "_Task2Log",
        navigation_state: dict[str, str | None],
        *,
        cycle: int | None = None,
    ) -> None:
        suffix = "replenishment" if cycle is None else f"replenishment.{cycle}"
        logger.event(
            "补货台导航", "started", target_id=self.settings.replenishment_pickup
        )
        await self._navigate(
            self.settings.replenishment_pickup,
            f"{task_run_id}:task2.{suffix}.navigate",
            logger,
            navigation_state,
        )
        logger.event(
            "补货台导航", "succeeded", target_id=self.settings.replenishment_pickup
        )
        logger.event(
            "补货台抓取位姿",
            "started",
            pose_type="REPLENISHMENT_TABLE_PICK_READY",
        )
        await self.client.prepare_pose(
            "REPLENISHMENT_TABLE_PICK_READY",
            f"{task_run_id}:task2.{suffix}.pose",
        )
        logger.event(
            "补货台抓取位姿",
            "succeeded",
            pose_type="REPLENISHMENT_TABLE_PICK_READY",
        )

    async def _pick_target(
        self,
        target: TargetItem,
        index: int,
        task_run_id: str,
        logger: "_Task2Log",
        held_items: dict[Hand, str],
        action_failures: list[dict[str, str]],
    ) -> bool:
        if target.hand in held_items:
            failure = {
                "operation": "pick",
                "product_name": target.product_name,
                "hand": target.hand.value,
                "error_code": "HAND_OCCUPIED",
                "message": f"{target.hand.value} hand already holds an item",
            }
            action_failures.append(failure)
            logger.event("补货商品抓取", "skipped", **failure)
            return False
        succeeded = await self._run_action_with_recovery(
            operation="pick",
            event_name="补货商品抓取",
            product_name=target.product_name,
            hand=target.hand,
            action_key=f"{task_run_id}:task2.pick.{index}",
            action=lambda key: self.client.pick(target.product_name, target.hand, key),
            logger=logger,
            action_failures=action_failures,
        )
        if succeeded:
            target.picked = True
            held_items[target.hand] = target.product_name
        return succeeded

    async def _place_target(
        self,
        target: TargetItem,
        index: int,
        task_run_id: str,
        logger: "_Task2Log",
        navigation_state: dict[str, str | None],
        held_items: dict[Hand, str],
        action_failures: list[dict[str, str]],
    ) -> bool:
        if held_items.get(target.hand) != target.product_name:
            logger.event(
                "货架商品放置",
                "skipped",
                product_name=target.product_name,
                hand=target.hand.value,
                reason="pick_not_succeeded",
            )
            return False
        logger.event(
            "补货位置导航",
            "started",
            product_name=target.product_name,
            target_id=target.inspection_target_id,
        )
        await self._navigate(
            target.inspection_target_id,
            f"{task_run_id}:task2.place.{index}.navigate",
            logger,
            navigation_state,
        )
        logger.event(
            "补货位置导航",
            "succeeded",
            product_name=target.product_name,
            target_id=target.inspection_target_id,
        )
        logger.event(
            "补货观察位姿恢复",
            "started",
            product_name=target.product_name,
            pose_type=target.inspection_pose_type.value,
        )
        await self.client.prepare_pose(
            target.inspection_pose_type.value,
            f"{task_run_id}:task2.place.{index}.pose",
        )
        logger.event(
            "补货观察位姿恢复",
            "succeeded",
            product_name=target.product_name,
            pose_type=target.inspection_pose_type.value,
        )
        succeeded = await self._run_action_with_recovery(
            operation="place",
            event_name="货架商品放置",
            product_name=target.product_name,
            hand=target.hand,
            action_key=f"{task_run_id}:task2.place.{index}",
            action=lambda key: self.client.place(target.product_name, target.hand, key),
            logger=logger,
            action_failures=action_failures,
        )
        if succeeded:
            target.placed = True
            held_items.pop(target.hand, None)
        return succeeded

    async def _run_action_with_recovery(
        self,
        *,
        operation: str,
        event_name: str,
        product_name: str,
        hand: Hand,
        action_key: str,
        action: Callable[[str], Awaitable[None]],
        logger: "_Task2Log",
        action_failures: list[dict[str, str]],
    ) -> bool:
        logger.event(
            event_name,
            "started",
            product_name=product_name,
            hand=hand.value,
            attempt=1,
        )
        initial_error: Task2ServiceError | None = None
        try:
            await action(action_key)
        except Task2ServiceError as exc:
            initial_error = exc
            logger.event(
                event_name,
                "failed",
                product_name=product_name,
                hand=hand.value,
                attempt=1,
                error_code=exc.code,
                message=exc.message,
            )
        else:
            logger.event(
                event_name,
                "succeeded",
                product_name=product_name,
                hand=hand.value,
                attempt=1,
            )
            return True

        assert initial_error is not None
        final_error = initial_error
        retry_succeeded = False
        logger.event(
            "抓放失败微调",
            "started",
            operation=operation,
            product_name=product_name,
            hand=hand.value,
            direction="back",
        )
        try:
            await self.client.nudge_back(f"{action_key}:recovery.approach")
        except Task2ServiceError as nudge_error:
            final_error = nudge_error
            logger.event(
                "抓放失败微调",
                "failed",
                operation=operation,
                product_name=product_name,
                hand=hand.value,
                direction="back",
                error_code=nudge_error.code,
                message=nudge_error.message,
            )
        else:
            logger.event(
                "抓放失败微调",
                "succeeded",
                operation=operation,
                product_name=product_name,
                hand=hand.value,
                direction="back",
            )
            logger.event(
                event_name,
                "started",
                product_name=product_name,
                hand=hand.value,
                attempt=2,
            )
            try:
                await action(f"{action_key}:recovery.retry")
            except Task2ServiceError as retry_error:
                final_error = retry_error
                logger.event(
                    event_name,
                    "failed",
                    product_name=product_name,
                    hand=hand.value,
                    attempt=2,
                    error_code=retry_error.code,
                    message=retry_error.message,
                )
            else:
                retry_succeeded = True
                logger.event(
                    event_name,
                    "succeeded",
                    product_name=product_name,
                    hand=hand.value,
                    attempt=2,
                    recovered=True,
                )

        await self._return_from_nudge(
            operation, product_name, hand, action_key, logger
        )
        if retry_succeeded:
            return True
        action_failures.append(
            {
                "operation": operation,
                "product_name": product_name,
                "hand": hand.value,
                "error_code": final_error.code,
                "message": final_error.message,
            }
        )
        return False

    async def _return_from_nudge(
        self,
        operation: str,
        product_name: str,
        hand: Hand,
        action_key: str,
        logger: "_Task2Log",
    ) -> None:
        last_error: Task2ServiceError | None = None
        for attempt in (1, 2):
            logger.event(
                "微调回原点",
                "started",
                operation=operation,
                product_name=product_name,
                hand=hand.value,
                attempt=attempt,
            )
            try:
                await self.client.nudge_return(
                    f"{action_key}:recovery.return.{attempt}"
                )
            except Task2ServiceError as exc:
                last_error = exc
                logger.event(
                    "微调回原点",
                    "failed",
                    operation=operation,
                    product_name=product_name,
                    hand=hand.value,
                    attempt=attempt,
                    error_code=exc.code,
                    message=exc.message,
                )
                continue
            logger.event(
                "微调回原点",
                "succeeded",
                operation=operation,
                product_name=product_name,
                hand=hand.value,
                attempt=attempt,
            )
            return
        assert last_error is not None
        raise Task2ServiceError(
            "NUDGE_RETURN_FAILED",
            f"navigation nudge return failed twice; robot position is unknown: {last_error.message}",
        ) from last_error

    async def _navigate(
        self,
        target_id: str,
        idempotency_key: str,
        logger: "_Task2Log",
        navigation_state: dict[str, str | None],
    ) -> None:
        if navigation_state["target_id"] == target_id:
            logger.event(
                "移动前复位", "skipped", target_id=target_id, reason="already_at_target"
            )
            return
        logger.event(
            "移动前复位", "started", pose_type="START_POSITION", target_id=target_id
        )
        await self.client.prepare_pose(
            "START_POSITION", f"{idempotency_key}.reset_pose"
        )
        logger.event(
            "移动前复位", "succeeded", pose_type="START_POSITION", target_id=target_id
        )
        await self.client.navigate(target_id, idempotency_key)
        navigation_state["target_id"] = target_id

    def _baseline_path(self, target_id: str, pose: InspectionPose) -> Path:
        return (
            Path(self.settings.baseline_dir)
            / f"{target_id}_{pose.directory_suffix}"
            / "rgb.jpg"
        )

    def _require_baselines(self) -> None:
        missing = [
            str(self._baseline_path(target_id, pose))
            for target_id in self.settings.inspection_points
            for pose in (InspectionPose.UPPER, InspectionPose.LOWER)
            if not self._baseline_path(target_id, pose).is_file()
            or self._baseline_path(target_id, pose).stat().st_size == 0
        ]
        if missing:
            raise Task2ServiceError(
                "BASELINE_NOT_READY",
                "Task0 baseline images are missing or empty: " + ", ".join(missing),
                status_code=503,
            )


class _Task2Log:
    def __init__(
        self, settings: Task2Settings, operation_key: str, request: Task2Request
    ) -> None:
        root = Path(settings.log_dir)
        root.mkdir(parents=True, exist_ok=True)
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", operation_key).strip("._") or "operation"
        self.directory = root / (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{safe_key}"
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        self._write_json(
            "operation.json",
            {
                "kind": "task2",
                "operation_key": operation_key,
                "request": request.model_dump(mode="json"),
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
            },
        )

    def _write_json(self, relative_path: str, payload: object) -> None:
        path = self.directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

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
            LOGGER.exception("任务二日志写入失败 event=%s status=%s", name, status)

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
