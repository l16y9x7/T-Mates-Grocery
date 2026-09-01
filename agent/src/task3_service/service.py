"""Task 3 shelf inspection and misplaced-product swap orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from task3_service.client import Task3Client
from task3_service.models import (
    FindingContext,
    Hand,
    InspectionPoint,
    InspectionPose,
    MisplacedFinding,
    PRODUCT_SLOT_PATTERN,
    SkuResponse,
    SwapItem,
    Task3Request,
    Task3Result,
    Task3ServiceError,
    Task3Settings,
)
from manipulation_policy import initial_shelf_nudge_direction


LOGGER = logging.getLogger(__name__)
POSE_LEVELS = {
    InspectionPose.UPPER: {"L1", "L2"},
    InspectionPose.LOWER: {"L3", "L4", "L5"},
}


def _recovery_direction(error: Task3ServiceError, operation: str) -> str | None:
    expected_interface = (
        "manipulation_grasp" if operation == "pick" else "manipulation_release"
    )
    if (
        error.failed_interface != expected_interface
        or error.code in {"ACTION_RESULT_UNKNOWN", "NETWORK_ERROR"}
        or error.pose is None
        or len(error.pose) != 6
    ):
        return None
    if error.pose[0] < 0:
        return "left"
    if error.pose[0] > 0:
        return "right"
    return None


def shelf_level(slot_id: str) -> str:
    match = PRODUCT_SLOT_PATTERN.fullmatch(slot_id)
    if match is None:
        raise Task3ServiceError(
            "INVALID_PRODUCT_SLOT", f"invalid product slot: {slot_id}", status_code=422
        )
    level = slot_id.split("_")[-2]
    return f"L{int(level[1:])}"


def shelf_view_pose(level: str) -> InspectionPose:
    return InspectionPose.UPPER if level in {"L1", "L2"} else InspectionPose.LOWER


class Task3Orchestrator:
    def __init__(self, settings: Task3Settings, client: Task3Client) -> None:
        self.settings = settings
        self.client = client

    async def ready(self) -> bool:
        return self.baselines_ready() and await self.client.health_ready()

    def baselines_ready(self) -> bool:
        return all(
            self._baseline_path(point, pose).is_file()
            and self._baseline_path(point, pose).stat().st_size > 0
            for point in self.settings.inspection_points
            for pose in (InspectionPose.UPPER, InspectionPose.LOWER)
        )

    async def run(
        self, request: Task3Request, operation_key: str | None = None
    ) -> Task3Result:
        task_run_id = operation_key or uuid4().hex
        logger = _Task3Log(self.settings, task_run_id, request)
        self.client.set_trace_callback(logger.interface_event)
        navigation_state: dict[str, str | None] = {"target_id": None}
        held_items: dict[Hand, str] = {}
        action_failures: list[dict[str, str]] = []
        step = "健康检查"
        try:
            logger.event("健康检查", "started")
            await self.client.check_all_health()
            self._require_baselines()
            logger.event("健康检查", "succeeded")

            step = "货架乱放巡检"
            finding, inspection_pass = await self._inspect_until_found(
                task_run_id, logger, navigation_state
            )

            step = "SKU货位转换"
            items = await self._build_swap_items(finding, logger)

            for index, item in enumerate(items):
                step = f"抓取 {item.product_name}"
                await self._pick_item(
                    item,
                    index,
                    task_run_id,
                    logger,
                    navigation_state,
                    held_items,
                    action_failures,
                )

            for index, item in enumerate(items):
                step = f"放置 {item.product_name}"
                await self._place_item(
                    item,
                    index,
                    task_run_id,
                    logger,
                    navigation_state,
                    held_items,
                    action_failures,
                )

            if action_failures:
                step = "抓放失败汇总"
                raise Task3ServiceError(
                    "TASK_ACTIONS_FAILED",
                    f"{len(action_failures)} pick/place actions failed",
                )

            step = "任务判定区导航"
            logger.event(
                "任务判定区导航", "started", target_id=self.settings.task_boundary
            )
            await self._navigate(
                self.settings.task_boundary,
                f"{task_run_id}:task3.finish.navigate",
                logger,
                navigation_state,
            )
            logger.event(
                "任务判定区导航", "succeeded", target_id=self.settings.task_boundary
            )

            result = Task3Result(
                task_run_id=task_run_id,
                task_type="MISPLACED",
                status="SUCCEEDED",
                inspection_pass=inspection_pass,
                finding=finding,
                product_names=[item.product_name for item in items],
                target_items=items,
                held_items=held_items,
            )
            logger.event("operation", "succeeded", picked_count=2, placed_count=2)
            return result
        except Exception as exc:
            original_step = step
            if isinstance(exc, Task3ServiceError):
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
                    f"{task_run_id}:task3.failure.start.navigate",
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
                recovery_error = Task3ServiceError(
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
                    "任务三失败且无法返回开始点 step=%s key=%s",
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
            LOGGER.exception("任务三流程失败 step=%s key=%s", original_step, task_run_id)
            raise
        finally:
            self.client.set_trace_callback(None)

    async def _inspect_until_found(
        self,
        task_run_id: str,
        logger: "_Task3Log",
        navigation_state: dict[str, str | None],
    ) -> tuple[FindingContext, int]:
        index = 0
        direction = 1
        inspection_pass = 1

        while True:
            point = self.settings.inspection_points[index]
            logger.event(
                "巡检点导航",
                "started",
                inspection_pass=inspection_pass,
                target_id=point.target_id,
            )
            await self._navigate(
                point.target_id,
                f"{task_run_id}:task3.inspect.{inspection_pass}.{index}.navigate",
                logger,
                navigation_state,
            )
            logger.event(
                "巡检点导航",
                "succeeded",
                inspection_pass=inspection_pass,
                target_id=point.target_id,
            )

            for pose in (InspectionPose.UPPER, InspectionPose.LOWER):
                action_prefix = (
                    f"{task_run_id}:task3.inspect.{inspection_pass}.{index}."
                    f"{pose.directory_suffix.lower()}"
                )
                logger.event(
                    "巡检观察位姿",
                    "started",
                    target_id=point.target_id,
                    pose_type=pose.value,
                )
                await self.client.prepare_pose(pose.value, f"{action_prefix}.pose")
                logger.event(
                    "巡检观察位姿",
                    "succeeded",
                    target_id=point.target_id,
                    pose_type=pose.value,
                )
                logger.event(
                    "乱放识别",
                    "started",
                    target_id=point.target_id,
                    location_id=point.target_id,
                    pose_type=pose.value,
                    baseline_path=str(self._baseline_path(point, pose)),
                )
                findings = await self.client.inspect(point, pose)
                logger.event(
                    "乱放识别",
                    "succeeded",
                    target_id=point.target_id,
                    pose_type=pose.value,
                    findings=[finding.model_dump(mode="json") for finding in findings],
                )
                if len(findings) > 1:
                    raise Task3ServiceError(
                        "INVALID_FINDINGS",
                        "inspection response contains more than one misplaced pair",
                        status_code=422,
                    )
                if findings:
                    return self._finding_context(findings[0], point, pose), inspection_pass

            next_index = index + direction
            if 0 <= next_index < len(self.settings.inspection_points):
                index = next_index
            else:
                inspection_pass += 1
                direction *= -1

    @staticmethod
    def _finding_context(
        finding: MisplacedFinding,
        point: InspectionPoint,
        pose: InspectionPose,
    ) -> FindingContext:
        misplaced_name = finding.misplaced_product_name.strip()
        gt_name = finding.gt_product_name.strip()
        if not misplaced_name or not gt_name or misplaced_name == gt_name:
            raise Task3ServiceError(
                "INVALID_FINDINGS",
                "misplaced and expected product names must be different and non-empty",
                status_code=422,
            )
        return FindingContext(
            misplaced_product_name=misplaced_name,
            gt_product_name=gt_name,
            inspection_target_id=point.target_id,
            inspection_location_id=point.location_id,
            inspection_pose_type=pose,
        )

    async def _build_swap_items(
        self, finding: FindingContext, logger: "_Task3Log"
    ) -> list[SwapItem]:
        logger.event(
            "SKU货位转换",
            "started",
            misplaced_product_name=finding.misplaced_product_name,
            gt_product_name=finding.gt_product_name,
        )
        misplaced_sku, gt_sku = await asyncio.gather(
            self.client.search_by_name(finding.misplaced_product_name),
            self.client.search_by_name(finding.gt_product_name),
        )
        p1 = self._resolve_detected_slot(gt_sku, finding)
        p2 = self._resolve_unique_slot(misplaced_sku)
        if p1 == p2:
            raise Task3ServiceError(
                "INVALID_FINDINGS",
                "misplaced products resolve to the same slot",
                status_code=422,
            )

        p1_option = self.settings.product_hand_options[p1]
        p2_option = self.settings.product_hand_options[p2]
        if p1_option.product_name != finding.gt_product_name:
            raise Task3ServiceError(
                "STALE_PRODUCT_CONFIGURATION",
                f"configured product for {p1} does not match {finding.gt_product_name}",
                status_code=422,
            )
        if p2_option.product_name != finding.misplaced_product_name:
            raise Task3ServiceError(
                "STALE_PRODUCT_CONFIGURATION",
                f"configured product for {p2} does not match {finding.misplaced_product_name}",
                status_code=422,
            )

        first_source, first_destination, second_source, second_destination = (
            self._plan_swap_grasps(p1, p2)
        )
        items = [
            SwapItem(
                product_name=finding.misplaced_product_name,
                source_slot_id=p1,
                destination_slot_id=p2,
                source_target_id=first_source[0],
                destination_target_id=first_destination[0],
                hand=first_source[1],
            ),
            SwapItem(
                product_name=finding.gt_product_name,
                source_slot_id=p2,
                destination_slot_id=p1,
                source_target_id=second_source[0],
                destination_target_id=second_destination[0],
                hand=second_source[1],
            ),
        ]
        logger.event(
            "SKU货位转换",
            "succeeded",
            p1=p1,
            p2=p2,
            jobs=[item.model_dump(mode="json") for item in items],
        )
        return items

    def _resolve_detected_slot(
        self, sku: SkuResponse, finding: FindingContext
    ) -> str:
        visible_levels = POSE_LEVELS[finding.inspection_pose_type]
        candidates = [
            slot
            for slot in sku.locations
            if PRODUCT_SLOT_PATTERN.fullmatch(slot)
            and slot in self.settings.product_hand_options
            and any(
                grasp.target_id == finding.inspection_target_id
                for grasp in self.settings.product_hand_options[slot].grasp_options
            )
            and shelf_level(slot) in visible_levels
        ]
        return self._select_candidate(sku.name, candidates, "detected misplaced location")

    def _resolve_unique_slot(self, sku: SkuResponse) -> str:
        candidates = [
            slot
            for slot in sku.locations
            if PRODUCT_SLOT_PATTERN.fullmatch(slot)
            and slot in self.settings.product_hand_options
        ]
        return self._select_candidate(sku.name, candidates, "standard location")

    @staticmethod
    def _select_candidate(
        product_name: str, candidates: list[str], purpose: str
    ) -> str:
        unique = sorted(set(candidates))
        if not unique:
            raise Task3ServiceError(
                "UNKNOWN_PRODUCT_SLOT",
                f"SKU {product_name} has no configured {purpose}",
                status_code=422,
            )
        return unique[0]

    @staticmethod
    def _require_single_candidate(
        product_name: str, candidates: list[str], purpose: str
    ) -> str:
        unique = list(dict.fromkeys(candidates))
        if len(unique) != 1:
            raise Task3ServiceError(
                "AMBIGUOUS_PRODUCT_SLOT",
                f"SKU {product_name} must resolve to exactly one {purpose}",
                status_code=422,
            )
        return unique[0]

    def _assign_hands(self, p1: str, p2: str) -> tuple[Hand, Hand]:
        p1_hands = self.settings.product_hand_options[p1].hands
        p2_hands = self.settings.product_hand_options[p2].hands
        for first, second in ((Hand.LEFT, Hand.RIGHT), (Hand.RIGHT, Hand.LEFT)):
            if (
                first in p1_hands
                and first in p2_hands
                and second in p2_hands
                and second in p1_hands
            ):
                return first, second
        raise Task3ServiceError(
            "NO_FEASIBLE_HAND_ASSIGNMENT",
            f"no two-hand swap assignment can reach both {p1} and {p2}",
            status_code=422,
        )

    def _slot_grasps(self, slot: str, hand: Hand) -> list[tuple[str, Hand]]:
        return [
            (grasp.target_id, hand)
            for grasp in self.settings.product_hand_options[slot].grasp_options
            if hand in grasp.hands
        ]

    def _plan_swap_grasps(
        self, p1: str, p2: str
    ) -> tuple[
        tuple[str, Hand], tuple[str, Hand], tuple[str, Hand], tuple[str, Hand]
    ]:
        """Choose fixed hands for both items; hands never exchange held products."""
        for first_hand, second_hand in (
            (Hand.LEFT, Hand.RIGHT),
            (Hand.RIGHT, Hand.LEFT),
        ):
            first_sources = self._slot_grasps(p1, first_hand)
            first_destinations = self._slot_grasps(p2, first_hand)
            second_sources = self._slot_grasps(p2, second_hand)
            second_destinations = self._slot_grasps(p1, second_hand)
            if all(
                (first_sources, first_destinations, second_sources, second_destinations)
            ):
                return (
                    first_sources[0],
                    first_destinations[0],
                    second_sources[0],
                    second_destinations[0],
                )
        raise Task3ServiceError(
            "NO_FEASIBLE_HAND_ASSIGNMENT",
            f"no fixed left/right hand assignment can reach both {p1} and {p2}",
            status_code=422,
        )

    async def _pick_item(
        self,
        item: SwapItem,
        index: int,
        task_run_id: str,
        logger: "_Task3Log",
        navigation_state: dict[str, str | None],
        held_items: dict[Hand, str],
        action_failures: list[dict[str, str]],
    ) -> bool:
        if item.hand in held_items:
            failure = {
                "operation": "pick",
                "product_name": item.product_name,
                "hand": item.hand.value,
                "error_code": "HAND_OCCUPIED",
                "message": f"{item.hand.value} hand already holds an item",
            }
            action_failures.append(failure)
            logger.event("乱放商品抓取", "skipped", **failure)
            return False
        logger.event(
            "乱放商品抓取导航",
            "started",
            product_name=item.product_name,
            target_id=item.source_target_id,
        )
        await self._navigate(
            item.source_target_id,
            f"{task_run_id}:task3.pick.{index}.navigate",
            logger,
            navigation_state,
        )
        logger.event(
            "乱放商品抓取导航",
            "succeeded",
            product_name=item.product_name,
            target_id=item.source_target_id,
        )
        source_level = shelf_level(item.source_slot_id)
        await self.client.prepare_pose(
            "SHELF_PICK_READY",
            f"{task_run_id}:task3.pick.{index}.pose",
            shelf_level=source_level,
        )
        succeeded = await self._run_action_with_recovery(
            operation="pick",
            event_name="乱放商品抓取",
            product_name=item.product_name,
            hand=item.hand,
            action_key=f"{task_run_id}:task3.pick.{index}.pick",
            action=lambda key: self.client.pick(
                item.product_name, item.hand, source_level, key
            ),
            logger=logger,
            action_failures=action_failures,
            event_details={"source_slot_id": item.source_slot_id},
            initial_nudge_direction=initial_shelf_nudge_direction(
                item.product_name, item.hand.value
            ),
        )
        if succeeded:
            item.picked = True
            held_items[item.hand] = item.product_name
        return succeeded

    async def _place_item(
        self,
        item: SwapItem,
        index: int,
        task_run_id: str,
        logger: "_Task3Log",
        navigation_state: dict[str, str | None],
        held_items: dict[Hand, str],
        action_failures: list[dict[str, str]],
    ) -> bool:
        if held_items.get(item.hand) != item.product_name:
            logger.event(
                "乱放商品放置",
                "skipped",
                product_name=item.product_name,
                hand=item.hand.value,
                reason="pick_not_succeeded",
            )
            return False
        logger.event(
            "乱放商品放置导航",
            "started",
            product_name=item.product_name,
            target_id=item.destination_target_id,
        )
        await self._navigate(
            item.destination_target_id,
            f"{task_run_id}:task3.place.{index}.navigate",
            logger,
            navigation_state,
        )
        logger.event(
            "乱放商品放置导航",
            "succeeded",
            product_name=item.product_name,
            target_id=item.destination_target_id,
        )
        destination_level = shelf_level(item.destination_slot_id)
        destination_view_pose = shelf_view_pose(destination_level)
        await self.client.prepare_pose(
            destination_view_pose.value,
            f"{task_run_id}:task3.place.{index}.pose",
        )
        succeeded = await self._run_action_with_recovery(
            operation="place",
            event_name="乱放��品放置",
            product_name=item.product_name,
            hand=item.hand,
            action_key=f"{task_run_id}:task3.place.{index}.place",
            action=lambda key: self.client.place(
                item.product_name,
                item.hand,
                item.destination_target_id,
                destination_view_pose.value,
                key,
            ),
            logger=logger,
            action_failures=action_failures,
            event_details={"destination_slot_id": item.destination_slot_id},
            recover_after_failure=False,
        )
        if succeeded:
            item.placed = True
            held_items.pop(item.hand, None)
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
        logger: "_Task3Log",
        action_failures: list[dict[str, str]],
        event_details: dict[str, str] | None = None,
        initial_nudge_direction: str | None = None,
        recover_after_failure: bool = True,
    ) -> bool:
        details = event_details or {}
        if initial_nudge_direction is not None:
            logger.event(
                "货架抓放前微调",
                "started",
                operation=operation,
                product_name=product_name,
                hand=hand.value,
                direction=initial_nudge_direction,
                **details,
            )
            try:
                await self.client.nudge(
                    initial_nudge_direction, f"{action_key}:initial.approach"
                )
            except Task3ServiceError as nudge_error:
                logger.event(
                    "货架抓放前微调",
                    "failed",
                    operation=operation,
                    product_name=product_name,
                    hand=hand.value,
                    direction=initial_nudge_direction,
                    error_code=nudge_error.code,
                    message=nudge_error.message,
                    **details,
                )
                await self._return_from_nudge(
                    operation, product_name, hand, action_key, logger
                )
                action_failures.append(
                    {
                        "operation": operation,
                        "product_name": product_name,
                        "hand": hand.value,
                        "error_code": nudge_error.code,
                        "message": nudge_error.message,
                    }
                )
                return False
            logger.event(
                "货架抓放前微调",
                "succeeded",
                operation=operation,
                product_name=product_name,
                hand=hand.value,
                direction=initial_nudge_direction,
                **details,
            )
        logger.event(
            event_name,
            "started",
            product_name=product_name,
            hand=hand.value,
            attempt=1,
            **details,
        )
        initial_error: Task3ServiceError | None = None
        try:
            await action(action_key)
        except Task3ServiceError as exc:
            initial_error = exc
            logger.event(
                event_name,
                "failed",
                product_name=product_name,
                hand=hand.value,
                attempt=1,
                error_code=exc.code,
                message=exc.message,
                **details,
            )
        else:
            logger.event(
                event_name,
                "succeeded",
                product_name=product_name,
                hand=hand.value,
                attempt=1,
                **details,
            )
            if initial_nudge_direction is not None:
                await self._return_from_nudge(
                    operation, product_name, hand, action_key, logger
                )
            return True

        assert initial_error is not None
        direction = (
            _recovery_direction(initial_error, operation)
            if recover_after_failure
            else None
        )
        if direction is None:
            if initial_nudge_direction is not None:
                await self._return_from_nudge(
                    operation, product_name, hand, action_key, logger
                )
            action_failures.append(
                {
                    "operation": operation,
                    "product_name": product_name,
                    "hand": hand.value,
                    "error_code": initial_error.code,
                    "message": initial_error.message,
                }
            )
            return False
        final_error = initial_error
        retry_succeeded = False
        logger.event(
            "抓放失败微调",
            "started",
            operation=operation,
            product_name=product_name,
            hand=hand.value,
            direction=direction,
        )
        try:
            await self.client.nudge(direction, f"{action_key}:recovery.approach")
        except Task3ServiceError as nudge_error:
            final_error = nudge_error
            logger.event(
                "抓放失败微调",
                "failed",
                operation=operation,
                product_name=product_name,
                hand=hand.value,
                direction=direction,
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
                direction=direction,
            )
            logger.event(
                event_name,
                "started",
                product_name=product_name,
                hand=hand.value,
                attempt=2,
                **details,
            )
            try:
                await action(f"{action_key}:recovery.retry")
            except Task3ServiceError as retry_error:
                final_error = retry_error
                logger.event(
                    event_name,
                    "failed",
                    product_name=product_name,
                    hand=hand.value,
                    attempt=2,
                    error_code=retry_error.code,
                    message=retry_error.message,
                    **details,
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
                    **details,
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
        logger: "_Task3Log",
    ) -> None:
        last_error: Task3ServiceError | None = None
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
            except Task3ServiceError as exc:
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
        raise Task3ServiceError(
            "NUDGE_RETURN_FAILED",
            f"navigation nudge return failed twice; robot position is unknown: {last_error.message}",
        ) from last_error

    async def _navigate(
        self,
        target_id: str,
        idempotency_key: str,
        logger: "_Task3Log",
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

    def _baseline_path(
        self, point: InspectionPoint, pose: InspectionPose
    ) -> Path:
        return (
            Path(self.settings.baseline_dir)
            / f"{point.target_id}_{pose.directory_suffix}"
            / "rgb.jpg"
        )

    def _require_baselines(self) -> None:
        missing = [
            str(self._baseline_path(point, pose))
            for point in self.settings.inspection_points
            for pose in (InspectionPose.UPPER, InspectionPose.LOWER)
            if not self._baseline_path(point, pose).is_file()
            or self._baseline_path(point, pose).stat().st_size == 0
        ]
        if missing:
            raise Task3ServiceError(
                "BASELINE_NOT_READY",
                "Task0 baseline images are missing or empty: " + ", ".join(missing),
                status_code=503,
            )


class _Task3Log:
    def __init__(
        self, settings: Task3Settings, operation_key: str, request: Task3Request
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
                "kind": "task3",
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
            LOGGER.exception("任务三日志写入失败 event=%s status=%s", name, status)

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
