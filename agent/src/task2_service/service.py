"""任务二的巡检与双手优先补货流程。"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pick_place_service.models import normalize_product_name
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


def _recovery_direction(error: Task2ServiceError, operation: str) -> str | None:
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


@dataclass(frozen=True)
class FindingContext:
    product_name: str
    inspection_target_id: str
    inspection_pose_type: InspectionPose
    product_slot_id: str | None = None


class Task2Orchestrator:
    def __init__(self, settings: Task2Settings, client: Task2Client) -> None:
        self.settings = settings
        self.client = client

    async def ready(self) -> bool:
        return self.baselines_ready() and await self.client.health_ready()

    def baselines_ready(self) -> bool:
        return all(
            path.is_file() and path.stat().st_size > 0
            for target_id in self.settings.inspection_points
            for pose in (InspectionPose.UPPER, InspectionPose.LOWER)
            for path in self._baseline_files(target_id, pose)
        )

    async def run(
        self, request: Task2Request, operation_key: str | None = None
    ) -> Task2Result:
        task_run_id = operation_key or uuid4().hex
        try:
            logger = _Task2Log(self.settings, task_run_id, request)
        except (OSError, TypeError, ValueError):
            LOGGER.exception("任务二日志初始化失败，继续执行任务 operation_key=%s", task_run_id)
            logger = _NullTaskLog()
        self.client.set_trace_callback(logger.interface_event)
        navigation_state: dict[str, str | None] = {"target_id": None}
        held_items: dict[Hand, str] = {}
        uncertain_hands: set[Hand] = set()
        action_failures: list[dict[str, str]] = []
        step = "健康检查"
        try:
            logger.event("operation", "started")
            logger.event("健康检查", "started")
            await self.client.check_all_health()
            self._require_baselines()
            logger.event("健康检查", "succeeded")

            targets: list[TargetItem] = []
            successful_placements = 0
            inspection_pass = 1
            for face_index, inspection_points in enumerate(self._inspection_groups()):
                step = "货架巡检"
                findings = await self._inspect_face(
                    task_run_id,
                    logger,
                    navigation_state,
                    face_index,
                    inspection_points,
                    action_failures,
                )
                candidate_sets: list[list[TargetItem]] = []
                seen_slots: set[str] = set()
                for finding in findings:
                    if finding.product_slot_id is not None:
                        if finding.product_slot_id in seen_slots:
                            logger.event(
                                "缺货记录",
                                "skipped",
                                product_name=finding.product_name,
                                product_slot_id=finding.product_slot_id,
                                reason="duplicate_slot_finding",
                            )
                            continue
                        seen_slots.add(finding.product_slot_id)
                    try:
                        choices = self._target_choices(finding)
                    except Task2ServiceError as hand_error:
                        logger.event(
                            "抓取手分配",
                            "skipped",
                            product_name=finding.product_name,
                            product_slot_id=finding.product_slot_id,
                            inspection_target_id=finding.inspection_target_id,
                            inspection_pose_type=finding.inspection_pose_type.value,
                            error_code=hand_error.code,
                            message=hand_error.message,
                        )
                        continue
                    candidate_sets.append(choices)

                for batch in self._plan_target_batches(candidate_sets):
                    if successful_placements >= 2:
                        break
                    pick_capacity = 2 - successful_placements
                    step = "补货台准备"
                    index = len(targets)
                    try:
                        await self._prepare_replenishment(
                            task_run_id, logger, navigation_state, cycle=index
                        )
                    except Task2ServiceError as exc:
                        navigation_state["target_id"] = None
                        failure = self._failure(
                            "replenishment_prepare", batch[0].product_name, None, exc
                        )
                        action_failures.append(failure)
                        logger.event(
                            "补货台准备",
                            "skipped",
                            **failure,
                            fallback="continue_next_batch",
                        )
                        continue
                    if held_items:
                        step = "补货台弃置"
                        await self._discard_held_items(
                            task_run_id,
                            index,
                            logger,
                            held_items,
                            uncertain_hands,
                            action_failures,
                        )

                    picked_targets: list[tuple[TargetItem, int]] = []
                    for target in batch:
                        if len(picked_targets) >= pick_capacity:
                            break
                        if (
                            target.hand in held_items
                            or target.hand in uncertain_hands
                        ):
                            logger.event(
                                "抓取手分配",
                                "skipped",
                                product_name=target.product_name,
                                product_slot_id=target.product_slot_id,
                                inspection_target_id=target.inspection_target_id,
                                inspection_pose_type=target.inspection_pose_type.value,
                                hand=target.hand.value,
                                error_code="NO_AVAILABLE_SAFE_HAND",
                                message="safe hand is occupied or has unknown state",
                            )
                            continue
                        targets.append(target)
                        target_index = len(targets) - 1
                        logger.event(
                            "抓取手分配",
                            "succeeded",
                            product_name=target.product_name,
                            product_slot_id=target.product_slot_id,
                            inspection_target_id=target.inspection_target_id,
                            inspection_pose_type=target.inspection_pose_type.value,
                            hand=target.hand.value,
                        )
                        step = "补货商品抓取"
                        picked = await self._pick_target(
                            target,
                            target_index,
                            task_run_id,
                            logger,
                            navigation_state,
                            held_items,
                            uncertain_hands,
                            action_failures,
                        )
                        if picked:
                            picked_targets.append((target, target_index))
                        else:
                            logger.event(
                                "货架商品放置",
                                "skipped",
                                product_name=target.product_name,
                                product_slot_id=target.product_slot_id,
                                hand=target.hand.value,
                                reason="pick_not_succeeded",
                            )

                    for target, target_index in picked_targets:
                        step = "货架商品放置"
                        try:
                            placed = await self._place_target(
                                target,
                                target_index,
                                task_run_id,
                                logger,
                                navigation_state,
                                held_items,
                                uncertain_hands,
                                action_failures,
                            )
                        except Task2ServiceError as exc:
                            navigation_state["target_id"] = None
                            failure = self._failure(
                                "place_prerequisite",
                                target.product_name,
                                target.hand,
                                exc,
                            )
                            action_failures.append(failure)
                            logger.event(
                                "货架商品放置",
                                "skipped",
                                **failure,
                                fallback="continue_next_target",
                            )
                            placed = False
                        if placed:
                            successful_placements += 1
                            if successful_placements == 2:
                                break
                if successful_placements == 2:
                    break

            step = "任务判定区导航"
            await self._finish_navigation(task_run_id, logger, navigation_state)

            result = Task2Result(
                task_run_id=task_run_id,
                task_type="SHORTAGE",
                status="SUCCEEDED",
                inspection_pass=inspection_pass,
                product_names=[target.product_name for target in targets],
                target_items=targets,
                held_items=held_items,
            )
            logger.event(
                "operation",
                "succeeded",
                picked_count=sum(target.picked for target in targets),
                placed_count=successful_placements,
                failed_attempt_count=len(action_failures),
                partial=successful_placements < 2,
                uncertain_hands=[hand.value for hand in sorted(uncertain_hands)],
            )
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

    def _inspection_groups(self) -> list[list[str]]:
        if self.settings.inspection_points and all(
            re.fullmatch(r"H(?:1|12|2|23|3)_INSPECT", target_id)
            for target_id in self.settings.inspection_points
        ):
            # The five new points observe one continuous three-shelf face.
            # Inspect all of them before planning so a two-hand replenishment
            # pair can be selected across shelf boundaries.
            return [list(self.settings.inspection_points)]
        groups: list[list[str]] = []
        current_face: str | None = None
        for target_id in self.settings.inspection_points:
            face = target_id.rsplit("_", 2)[0]
            if face != current_face:
                groups.append([])
                current_face = face
            groups[-1].append(target_id)
        return groups

    async def _inspect_face(
        self,
        task_run_id: str,
        logger: "_Task2Log",
        navigation_state: dict[str, str | None],
        face_index: int,
        inspection_points: list[str],
        action_failures: list[dict[str, str]],
    ) -> list[FindingContext]:
        findings: list[FindingContext] = []
        for point_index, target_id in enumerate(inspection_points):
            logger.event(
                "巡检点导航",
                "started",
                inspection_pass=1,
                inspection_face=face_index + 1,
                target_id=target_id,
            )
            try:
                await self._navigate(
                    target_id,
                    f"{task_run_id}:task2.inspect.{face_index}.{point_index}.navigate",
                    logger,
                    navigation_state,
                )
            except Task2ServiceError as exc:
                navigation_state["target_id"] = None
                failure = self._failure("inspect_navigate", None, None, exc)
                action_failures.append(failure)
                logger.event(
                    "巡检点导航",
                    "skipped",
                    inspection_pass=1,
                    inspection_face=face_index + 1,
                    target_id=target_id,
                    **failure,
                    fallback="continue_next_inspection_point",
                )
                continue
            logger.event(
                "巡检点导航",
                "succeeded",
                inspection_pass=1,
                inspection_face=face_index + 1,
                target_id=target_id,
            )

            for pose in (InspectionPose.UPPER, InspectionPose.LOWER):
                action_prefix = (
                    f"{task_run_id}:task2.inspect.{face_index}.{point_index}."
                    f"{pose.value.lower()}"
                )
                logger.event(
                    "巡检观察位姿",
                    "started",
                    target_id=target_id,
                    pose_type=pose.value,
                )
                try:
                    await self.client.prepare_pose(
                        pose.value, f"{action_prefix}.pose"
                    )
                except Task2ServiceError as exc:
                    failure = self._failure("inspect_pose", None, None, exc)
                    action_failures.append(failure)
                    logger.event(
                        "巡检观察位姿",
                        "skipped",
                        target_id=target_id,
                        pose_type=pose.value,
                        **failure,
                        fallback="continue_next_inspection_pose",
                    )
                    continue
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
                    location_id=target_id,
                    pose_type=pose.value,
                    baseline_path=str(baseline_path),
                )
                try:
                    detected_findings = await self.client.inspect(
                        target_id, pose.value
                    )
                except Task2ServiceError as exc:
                    failure = self._failure("inspect", None, None, exc)
                    action_failures.append(failure)
                    logger.event(
                        "缺货识别",
                        "skipped",
                        target_id=target_id,
                        location_id=target_id,
                        pose_type=pose.value,
                        baseline_path=str(baseline_path),
                        **failure,
                        fallback="continue_next_inspection_pose",
                    )
                    continue
                logger.event(
                    "缺货识别",
                    "succeeded",
                    target_id=target_id,
                    location_id=target_id,
                    pose_type=pose.value,
                    baseline_path=str(baseline_path),
                    findings=[
                        finding.model_dump(mode="json")
                        for finding in detected_findings
                    ],
                )
                for detected in detected_findings:
                    finding = FindingContext(
                        detected.shortage_product_name,
                        target_id,
                        pose,
                        detected.slot_id,
                    )
                    findings.append(finding)
                    logger.event(
                        "缺货记录",
                        "succeeded",
                        product_name=finding.product_name,
                        product_slot_id=finding.product_slot_id,
                        target_id=target_id,
                        pose_type=pose.value,
                        accumulated_count=len(findings),
                    )
        return findings

    def _target_choices(self, finding: FindingContext) -> list[TargetItem]:
        """Resolve one exact shortage slot into its valid target/hand choices."""

        if finding.product_slot_id is None:
            if self.settings.product_hand_options_schema_version == "2.0":
                raise Task2ServiceError(
                    "MISSING_SHORTAGE_SLOT_ID",
                    "schema 2.0 shortage findings must include an exact slot_id",
                    status_code=422,
                )
            return [
                TargetItem(
                    product_name=finding.product_name,
                    product_slot_id=None,
                    inspection_target_id=finding.inspection_target_id,
                    inspection_pose_type=finding.inspection_pose_type,
                    hand=hand,
                )
                for hand in self._allowed_hands(finding)
            ]

        option = self.settings.product_hand_options.get(finding.product_slot_id)
        if option is None:
            raise Task2ServiceError(
                "UNKNOWN_PRODUCT_HAND_OPTIONS",
                f"no hand configuration matches slot {finding.product_slot_id}",
                status_code=422,
            )
        if normalize_product_name(option.product_name) != normalize_product_name(
            finding.product_name
        ):
            raise Task2ServiceError(
                "SLOT_PRODUCT_MISMATCH",
                f"slot {finding.product_slot_id} is configured for "
                f"{option.product_name}, not {finding.product_name}",
                status_code=422,
            )
        level_number = int(finding.product_slot_id.split("_")[-2][1:])
        pose = (
            InspectionPose.UPPER
            if level_number <= 2
            else InspectionPose.LOWER
        )
        if pose != finding.inspection_pose_type:
            raise Task2ServiceError(
                "SLOT_POSE_MISMATCH",
                f"slot {finding.product_slot_id} is not visible in "
                f"{finding.inspection_pose_type.value}",
                status_code=422,
            )
        choices = [
            TargetItem(
                product_name=finding.product_name,
                product_slot_id=finding.product_slot_id,
                inspection_target_id=grasp.target_id,
                inspection_pose_type=pose,
                hand=hand,
            )
            for grasp in option.grasp_options
            for hand in grasp.hands
        ]
        if not choices:
            raise Task2ServiceError(
                "NO_SAFE_HAND_OPTION",
                f"slot {finding.product_slot_id} has no configured grasp choice",
                status_code=422,
            )
        return choices

    @staticmethod
    def _plan_target_batches(
        candidate_sets: list[list[TargetItem]],
    ) -> list[list[TargetItem]]:
        """Prefer two distinct hands per replenishment-table visit."""

        pending = [list(candidates) for candidates in candidate_sets if candidates]
        batches: list[list[TargetItem]] = []
        while pending:
            best: tuple[int, int, int, TargetItem, TargetItem] | None = None
            for first_index, first_choices in enumerate(pending):
                for second_index in range(first_index + 1, len(pending)):
                    for first in first_choices:
                        for second in pending[second_index]:
                            if first.hand == second.hand:
                                continue
                            if (
                                first.product_slot_id is not None
                                and first.product_slot_id == second.product_slot_id
                            ):
                                continue
                            score = (
                                0
                                if first.inspection_target_id
                                == second.inspection_target_id
                                else 1
                            )
                            candidate = (
                                score,
                                first_index,
                                second_index,
                                first,
                                second,
                            )
                            if best is None or candidate[:3] < best[:3]:
                                best = candidate
            if best is None:
                batches.append([pending.pop(0)[0]])
                continue
            _, first_index, second_index, first, second = best
            batches.append([first, second])
            pending.pop(second_index)
            pending.pop(first_index)
        return batches

    def _allowed_hands(self, finding: FindingContext) -> list[Hand]:
        if finding.product_slot_id is not None:
            option = self.settings.product_hand_options.get(finding.product_slot_id)
            if option is None or normalize_product_name(
                option.product_name
            ) != normalize_product_name(finding.product_name):
                raise Task2ServiceError(
                    "UNKNOWN_PRODUCT_HAND_OPTIONS",
                    f"no hand configuration matches slot {finding.product_slot_id}",
                    status_code=422,
                )
            exact_candidates = [
                grasp.hands
                for grasp in option.grasp_options
                if grasp.target_id == finding.inspection_target_id
            ]
            allowed = list(
                dict.fromkeys(
                    hand for hands in exact_candidates for hand in hands
                )
            )
            if allowed:
                return allowed
            raise Task2ServiceError(
                "NO_SAFE_HAND_OPTION",
                f"slot {finding.product_slot_id} has no grasp option at "
                f"{finding.inspection_target_id}",
                status_code=422,
            )
        visible_levels = POSE_LEVELS[finding.inspection_pose_type]
        candidates = [
            grasp.hands
            for slot_id, option in self.settings.product_hand_options.items()
            for grasp in option.grasp_options
            if normalize_product_name(option.product_name)
            == normalize_product_name(finding.product_name)
            and grasp.target_id == finding.inspection_target_id
            and f"L{int(slot_id.split('_')[-2][1:])}" in visible_levels
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

    @staticmethod
    def _failure(
        operation: str,
        product_name: str | None,
        hand: Hand | None,
        error: Task2ServiceError,
    ) -> dict[str, str]:
        return {
            "operation": operation,
            "product_name": product_name or "",
            "hand": hand.value if hand is not None else "",
            "error_code": error.code,
            "message": error.message,
        }

    async def _finish_navigation(
        self,
        task_run_id: str,
        logger: "_Task2Log",
        navigation_state: dict[str, str | None],
    ) -> None:
        attempts = (
            (self.settings.task_boundary, "boundary.1"),
            (self.settings.task_boundary, "boundary.2"),
            (self.settings.start_target_id, "start_fallback"),
        )
        last_error: Task2ServiceError | None = None
        for target_id, suffix in attempts:
            navigation_state["target_id"] = None
            logger.event("任务判定区导航", "started", target_id=target_id)
            try:
                await self._navigate(
                    target_id,
                    f"{task_run_id}:task2.finish.{suffix}.navigate",
                    logger,
                    navigation_state,
                )
            except Task2ServiceError as exc:
                last_error = exc
                logger.event(
                    "任务判定区导航",
                    "failed",
                    target_id=target_id,
                    error_code=exc.code,
                    message=exc.message,
                )
                continue
            logger.event("任务判定区导航", "succeeded", target_id=target_id)
            return
        assert last_error is not None
        raise last_error

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

    async def _discard_held_items(
        self,
        task_run_id: str,
        index: int,
        logger: "_Task2Log",
        held_items: dict[Hand, str],
        uncertain_hands: set[Hand],
        action_failures: list[dict[str, str]],
    ) -> None:
        for hand, product_name in tuple(held_items.items()):
            logger.event(
                "补货台弃置",
                "started",
                product_name=product_name,
                hand=hand.value,
            )
            error: Task2ServiceError | None = None
            for attempt in (1, 2):
                try:
                    await self.client.open_gripper(
                        hand,
                        f"{task_run_id}:task2.discard.{index}."
                        f"{hand.value.lower()}.{attempt}",
                    )
                except Task2ServiceError as exc:
                    error = exc
                    logger.event(
                        "补货台弃置",
                        "failed",
                        product_name=product_name,
                        hand=hand.value,
                        attempt=attempt,
                        error_code=exc.code,
                        message=exc.message,
                    )
                    if exc.code in {
                        "ACTION_RESULT_UNKNOWN",
                        "NETWORK_ERROR",
                        "INVALID_RESPONSE",
                    }:
                        uncertain_hands.add(hand)
                        break
                else:
                    held_items.pop(hand)
                    error = None
                    break
            if error is not None:
                action_failures.append(
                    self._failure("discard", product_name, hand, error)
                )
                continue
            logger.event(
                "补货台弃置",
                "succeeded",
                product_name=product_name,
                hand=hand.value,
            )

    async def _pick_target(
        self,
        target: TargetItem,
        index: int,
        task_run_id: str,
        logger: "_Task2Log",
        navigation_state: dict[str, str | None],
        held_items: dict[Hand, str],
        uncertain_hands: set[Hand],
        action_failures: list[dict[str, str]],
    ) -> bool:
        if target.hand in held_items or target.hand in uncertain_hands:
            failure = {
                "operation": "pick",
                "product_name": target.product_name,
                "hand": target.hand.value,
                "error_code": "HAND_UNAVAILABLE",
                "message": f"{target.hand.value} hand is occupied or its state is unknown",
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
            uncertain_hands=uncertain_hands,
            navigation_state=navigation_state,
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
        uncertain_hands: set[Hand],
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
            action=lambda key: self.client.place(
                target.product_name,
                target.hand,
                target.inspection_target_id,
                target.inspection_pose_type.value,
                key,
                target.product_slot_id,
            ),
            logger=logger,
            action_failures=action_failures,
            uncertain_hands=uncertain_hands,
            navigation_state=navigation_state,
            recover_after_failure=False,
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
        uncertain_hands: set[Hand],
        navigation_state: dict[str, str | None] | None = None,
        initial_nudge_direction: str | None = None,
        recover_after_failure: bool = True,
    ) -> bool:
        if initial_nudge_direction is not None:
            logger.event(
                "货架抓放前微调",
                "started",
                operation=operation,
                product_name=product_name,
                hand=hand.value,
                direction=initial_nudge_direction,
            )
            try:
                await self.client.nudge(
                    initial_nudge_direction, f"{action_key}:initial.approach"
                )
            except Task2ServiceError as nudge_error:
                logger.event(
                    "货架抓放前微调",
                    "failed",
                    operation=operation,
                    product_name=product_name,
                    hand=hand.value,
                    direction=initial_nudge_direction,
                    error_code=nudge_error.code,
                    message=nudge_error.message,
                )
                await self._return_from_nudge(
                    operation,
                    product_name,
                    hand,
                    action_key,
                    logger,
                    navigation_state,
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
            )
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
            if initial_nudge_direction is not None:
                await self._return_from_nudge(
                    operation,
                    product_name,
                    hand,
                    action_key,
                    logger,
                    navigation_state,
                )
            return True

        assert initial_error is not None
        if initial_error.code in {
            "ACTION_RESULT_UNKNOWN",
            "NETWORK_ERROR",
            "INVALID_RESPONSE",
        }:
            uncertain_hands.add(hand)
            action_failures.append(
                self._failure(operation, product_name, hand, initial_error)
            )
            return False
        direction = (
            _recovery_direction(initial_error, operation)
            if recover_after_failure
            else None
        )
        if direction is None:
            retry_error = initial_error
            logger.event(
                event_name,
                "started",
                product_name=product_name,
                hand=hand.value,
                attempt=2,
            )
            try:
                await action(f"{action_key}:recovery.retry")
            except Task2ServiceError as exc:
                retry_error = exc
                logger.event(
                    event_name,
                    "failed",
                    product_name=product_name,
                    hand=hand.value,
                    attempt=2,
                    error_code=exc.code,
                    message=exc.message,
                )
            else:
                logger.event(
                    event_name,
                    "succeeded",
                    product_name=product_name,
                    hand=hand.value,
                    attempt=2,
                    recovered=True,
                )
                if initial_nudge_direction is not None:
                    await self._return_from_nudge(
                        operation,
                        product_name,
                        hand,
                        action_key,
                        logger,
                        navigation_state,
                    )
                return True
            if retry_error.code in {
                "ACTION_RESULT_UNKNOWN",
                "NETWORK_ERROR",
                "INVALID_RESPONSE",
            }:
                uncertain_hands.add(hand)
            if initial_nudge_direction is not None:
                await self._return_from_nudge(
                    operation,
                    product_name,
                    hand,
                    action_key,
                    logger,
                    navigation_state,
                )
            action_failures.append(
                self._failure(operation, product_name, hand, retry_error)
            )
            return False
        final_error = initial_error
        retry_succeeded = False
        if operation == "pick":
            # A failed replenishment grasp can leave the arm away from its
            # observation pose. Restore that pose before applying the lateral
            # recovery nudge and retrying the grasp.
            logger.event(
                "补货台观察位恢复",
                "started",
                product_name=product_name,
                hand=hand.value,
            )
            try:
                await self.client.prepare_pose(
                    "REPLENISHMENT_TABLE_PICK_READY",
                    f"{action_key}:recovery.pose",
                )
            except Task2ServiceError as pose_error:
                final_error = pose_error
                logger.event(
                    "补货台观察位恢复",
                    "failed",
                    product_name=product_name,
                    hand=hand.value,
                    error_code=pose_error.code,
                    message=pose_error.message,
                )
                if pose_error.code in {
                    "ACTION_RESULT_UNKNOWN",
                    "NETWORK_ERROR",
                    "INVALID_RESPONSE",
                }:
                    uncertain_hands.add(hand)
                action_failures.append(
                    self._failure(operation, product_name, hand, final_error)
                )
                return False
            logger.event(
                "补货台观察位恢复",
                "succeeded",
                product_name=product_name,
                hand=hand.value,
            )
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
        except Task2ServiceError as nudge_error:
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
            operation,
            product_name,
            hand,
            action_key,
            logger,
            navigation_state,
        )
        if retry_succeeded:
            return True
        if final_error.code in {
            "ACTION_RESULT_UNKNOWN",
            "NETWORK_ERROR",
            "INVALID_RESPONSE",
        }:
            uncertain_hands.add(hand)
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
        navigation_state: dict[str, str | None] | None = None,
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
        if navigation_state is not None:
            navigation_state["target_id"] = None
        logger.event(
            "微调回原点",
            "exhausted",
            operation=operation,
            product_name=product_name,
            hand=hand.value,
            error_code="NUDGE_RETURN_FAILED",
            message=last_error.message,
            fallback="continue_with_absolute_navigation",
        )

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

    def _baseline_files(
        self, target_id: str, pose: InspectionPose
    ) -> tuple[Path, Path, Path]:
        directory = self._baseline_path(target_id, pose).parent
        return (
            directory / "rgb.jpg",
            directory / "depth_mm.npy",
            directory / "meta.json",
        )

    def _require_baselines(self) -> None:
        missing = [
            str(path)
            for target_id in self.settings.inspection_points
            for pose in (InspectionPose.UPPER, InspectionPose.LOWER)
            for path in self._baseline_files(target_id, pose)
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            raise Task2ServiceError(
                "BASELINE_NOT_READY",
                "Task0 baseline RGB-D files are missing or empty: "
                + ", ".join(missing),
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


class _NullTaskLog:
    def event(self, name: str, status: str, **details: object) -> None:
        pass

    def interface_event(self, trace: dict[str, object]) -> None:
        pass
