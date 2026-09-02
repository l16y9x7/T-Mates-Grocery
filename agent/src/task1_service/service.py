"""任务一的串行编排流程。"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from task1_service.client import Task1Client
from task1_service.mock_order import MockOrder, MockOrderSystem
from task1_service.models import (
    Hand,
    PRODUCT_SLOT_PATTERN,
    TargetItem,
    Task1Request,
    Task1Result,
    Task1ServiceError,
    Task1Settings,
)
from manipulation_policy import initial_shelf_nudge_direction


LOGGER = logging.getLogger(__name__)
SHELF_LEVEL_PRIORITY = {
    level: rank
    for rank, level in enumerate(("L3", "L2", "L4", "L1", "L5"))
}


def _recovery_direction(error: Task1ServiceError, operation: str) -> str | None:
    if (
        operation != "pick"
        or error.failed_interface != "manipulation_grasp"
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
    level = slot_id.split("_")[-2]
    return f"L{int(level[1:])}"


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


def _initial_nudge_direction(target: TargetItem) -> str | None:
    """Return any configured approach nudge for this task1 shelf target."""

    if (
        target.product_slot_id == "H2_B_L1_C01"
        and target.product_name == "舒肤佳香皂纯白清香型"
        and target.hand is Hand.LEFT
        and target.target_id == "H2_B_L_INSPECT"
    ):
        return "left"
    return initial_shelf_nudge_direction(target.product_name, target.hand.value)


class Task1Orchestrator:
    def __init__(self, settings: Task1Settings, client: Task1Client) -> None:
        self.settings = settings
        self.client = client
        self.mock_order_system = MockOrderSystem(self._available_order_products)

    async def _available_order_products(self) -> list[str]:
        names = await self.client.list_product_names()
        excluded = set(self.settings.skip_product_names)
        return [name for name in names if name not in excluded]

    async def create_mock_order(
        self,
        product_names: list[str] | None = None,
        order_id: str | None = None,
    ) -> MockOrder:
        """Generate or revalidate the mock order selected by the web console."""

        return await self.mock_order_system.create_order(product_names, order_id)

    async def run(self, request: Task1Request, operation_key: str | None = None) -> Task1Result:
        task_run_id = operation_key or uuid4().hex
        try:
            logger = _Task1Log(self.settings, task_run_id, request)
        except (OSError, TypeError, ValueError):
            LOGGER.exception("任务一日志初始化失败，继续执行任务 operation_key=%s", task_run_id)
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
            logger.event("健康检查", "succeeded")

            step = "模拟点单"
            logger.event("模拟点单", "started", order_source=request.order_source)
            order = await self.create_mock_order(
                request.product_names,
                request.order_id,
            )
            product_names = list(order.product_names)
            logger.event(
                "模拟点单",
                "succeeded",
                order_id=order.order_id,
                order_source=order.source,
                catalog_size=order.catalog_size,
                product_names=product_names,
            )

            step = "SKU货位转换"
            targets: list[TargetItem] = []
            target_slot_candidates: list[list[str]] = []
            selected_product_names = self._select_products(product_names, logger)
            for name in selected_product_names:
                logger.event("SKU货位转换", "started", product_name=name)
                try:
                    sku = await self.client.search_by_name(name)
                    if not sku.locations:
                        raise Task1ServiceError(
                            "AMBIGUOUS_PRODUCT_SLOT",
                            f"SKU {name} must resolve to at least one location",
                            status_code=422,
                        )
                    invalid_slot = next(
                        (
                            location
                            for location in sku.locations
                            if not PRODUCT_SLOT_PATTERN.fullmatch(location)
                        ),
                        None,
                    )
                    if invalid_slot is not None:
                        raise Task1ServiceError(
                            "INVALID_PRODUCT_SLOT",
                            f"invalid product slot: {invalid_slot}",
                            status_code=422,
                        )
                    candidate_slots = sorted(
                        sku.locations,
                        key=lambda location: (
                            SHELF_LEVEL_PRIORITY[shelf_level(location)],
                            location,
                        ),
                    )
                    slot_id = candidate_slots[0]
                    target = TargetItem(
                        product_name=name,
                        product_slot_id=slot_id,
                        target_id="",
                        shelf_level=shelf_level(slot_id),
                        hand=Hand.LEFT,
                    )
                except Task1ServiceError as exc:
                    action_failures.append(self._failure("sku", name, None, exc))
                    logger.event(
                        "SKU货位转换",
                        "skipped",
                        product_name=name,
                        error_code=exc.code,
                        message=exc.message,
                    )
                    continue
                targets.append(target)
                target_slot_candidates.append(candidate_slots)
                logger.event(
                    "SKU货位转换",
                    "succeeded",
                    product_name=name,
                    sku_id=sku.sku_id,
                    product_slot_id=slot_id,
                    target_id="",
                    shelf_level=shelf_level(slot_id),
                )
            try:
                planned = self._plan_grasps(targets, target_slot_candidates)
            except Task1ServiceError as exc:
                action_failures.append(self._failure("hand_assignment", None, None, exc))
                targets.clear()
                planned = []
            for target, (target_id, hand) in zip(targets, planned):
                target.target_id = target_id
                target.hand = hand
                logger.event(
                    "抓取手分配", "succeeded", product_name=target.product_name,
                    product_slot_id=target.product_slot_id, hand=hand.value,
                    target_id=target_id,
                    allowed_choices=[
                        {"target_id": choice_target, "hand": choice_hand.value}
                        for choice_target, choice_hand in self._grasp_choices(target.product_slot_id)
                    ],
                )

            if len(targets) == 2 and targets[0].hand != targets[1].hand:
                for index, target in enumerate(targets):
                    step = "商品抓取"
                    await self._pick_target_best_effort(
                        target,
                        index,
                        task_run_id,
                        logger,
                        navigation_state,
                        held_items,
                        uncertain_hands,
                        action_failures,
                    )
                if held_items:
                    step = "交付台准备"
                    delivery_ready = await self._prepare_delivery_best_effort(
                        task_run_id,
                        logger,
                        navigation_state,
                        action_failures,
                    )
                    if delivery_ready and set(held_items) == {Hand.LEFT, Hand.RIGHT}:
                        step = "商品放置"
                        placed_both = await self._place_both_targets(
                            targets,
                            task_run_id,
                            logger,
                            held_items,
                            uncertain_hands,
                            action_failures,
                        )
                        if not placed_both and not uncertain_hands:
                            for index, target in enumerate(targets):
                                await self._place_target(
                                    target,
                                    index,
                                    task_run_id,
                                    logger,
                                    held_items,
                                    uncertain_hands,
                                    action_failures,
                                )
                    elif delivery_ready:
                        for index, target in enumerate(targets):
                            step = "商品放置"
                            await self._place_target(
                                target,
                                index,
                                task_run_id,
                                logger,
                                held_items,
                                uncertain_hands,
                                action_failures,
                            )
                else:
                    logger.event(
                        "交付台准备",
                        "skipped",
                        reason="no_picked_items",
                    )
            else:
                # 单手能力受限时，始终保持该手一次只持有一件商品。
                for index, target in enumerate(targets):
                    step = "商品抓取"
                    picked = await self._pick_target_best_effort(
                        target,
                        index,
                        task_run_id,
                        logger,
                        navigation_state,
                        held_items,
                        uncertain_hands,
                        action_failures,
                    )
                    if not picked:
                        logger.event(
                            "交付台准备",
                            "skipped",
                            product_name=target.product_name,
                            reason="pick_not_succeeded",
                        )
                        logger.event(
                            "放置",
                            "skipped",
                            product_name=target.product_name,
                            hand=target.hand.value,
                            reason="pick_not_succeeded",
                        )
                        continue
                    step = "交付台准备"
                    delivery_ready = await self._prepare_delivery_best_effort(
                        task_run_id,
                        logger,
                        navigation_state,
                        action_failures,
                        cycle=index,
                    )
                    if not delivery_ready:
                        continue
                    step = "商品放置"
                    await self._place_target(
                        target,
                        index,
                        task_run_id,
                        logger,
                        held_items,
                        uncertain_hands,
                        action_failures,
                    )

            step = "任务判定区导航"
            await self._finish_navigation(
                task_run_id, logger, navigation_state
            )

            result = Task1Result(
                task_run_id=task_run_id,
                task_type="SORTING",
                status="SUCCEEDED",
                product_names=product_names,
                target_items=targets,
                held_items=held_items,
                order={
                    "order_id": order.order_id,
                    "source": order.source,
                    "catalog_size": order.catalog_size,
                    "product_names": order.product_names,
                },
                interface_metrics=self.client.interface_metrics(),
            )
            picked_count = sum(target.picked for target in targets)
            placed_count = sum(target.placed for target in targets)
            logger.event(
                "operation",
                "succeeded",
                picked_count=picked_count,
                placed_count=placed_count,
                failed_attempt_count=len(action_failures),
                partial=placed_count < 2,
                uncertain_hands=[hand.value for hand in sorted(uncertain_hands)],
                interface_metrics=[
                    metric.model_dump(mode="json")
                    for metric in result.interface_metrics
                ],
            )
            return result
        except Exception as exc:
            original_step = step
            if isinstance(exc, Task1ServiceError):
                exc.step = original_step
                exc.interface_metrics = self.client.interface_metrics()
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
                    f"{task_run_id}:task1.failure.start.navigate",
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
                recovery_error = Task1ServiceError(
                    "FAILURE_RECOVERY_FAILED",
                    f"task failed at {original_step}; navigation back to start also failed: {recovery_exc}",
                    failed_interface=getattr(recovery_exc, "failed_interface", None),
                    url=getattr(recovery_exc, "url", None),
                    interface_metrics=self.client.interface_metrics(),
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
                    interface_metrics=[
                        metric.model_dump(mode="json")
                        for metric in self.client.interface_metrics()
                    ],
                )
                LOGGER.exception(
                    "任务一失败且无法返回开始点 step=%s key=%s",
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
                failed_interface=getattr(exc, "failed_interface", None),
                url=getattr(exc, "url", None),
                interface_metrics=[
                    metric.model_dump(mode="json")
                    for metric in self.client.interface_metrics()
                ],
            )
            if isinstance(exc, Task1ServiceError):
                # Include the recovery navigation calls in the error returned
                # to the web layer as well as in the persisted event log.
                exc.interface_metrics = self.client.interface_metrics()
            LOGGER.exception("任务一流程失败 step=%s key=%s", original_step, task_run_id)
            raise
        finally:
            self.client.set_trace_callback(None)

    def _allowed_hands(self, slot_id: str) -> list[Hand]:
        return list(dict.fromkeys(hand for _, hand in self._grasp_choices(slot_id)))

    def _grasp_choices(self, slot_id: str) -> list[tuple[str, Hand]]:
        configured = self.settings.product_grasp_options.get(slot_id)
        if configured:
            return [
                (option.target_id, hand)
                for option in configured
                for hand in option.hands
            ]
        # The legacy fallback derives H1_F/H1_B/H2_F/H2_B inspection points
        # from a four-part slot id. New three-part shelf slots must be explicitly
        # mapped so an unknown catalog location cannot silently gain both hands
        # or crash while being parsed by default_inspection_target_id().
        if len(slot_id.split("_")) == 3:
            return []
        target_id = self._target_id(slot_id)
        hands = self.settings.product_hand_options.get(
            slot_id, [Hand.LEFT, Hand.RIGHT]
        )
        return [(target_id, Hand(hand)) for hand in hands]

    def _plan_grasps(
        self,
        targets: list[TargetItem],
        slot_candidates: list[list[str]] | None = None,
    ) -> list[tuple[str, Hand]]:
        candidates = slot_candidates or [
            [target.product_slot_id] for target in targets
        ]
        choices = [
            [
                (slot, target_id, hand)
                for slot in slots
                for target_id, hand in self._grasp_choices(slot)
            ]
            for slots in candidates
        ]
        if any(not item for item in choices):
            raise Task1ServiceError(
                "NO_FEASIBLE_HAND_ASSIGNMENT",
                "one or more products have no grasp option",
                status_code=422,
            )
        if len(choices) == 2:
            # First prefer one navigation point where both hands can pick one item.
            # If that is impossible, still prefer carrying both items in one trip,
            # even when the robot must visit two shelf points.
            for require_same_target in (True, False):
                for first in choices[0]:
                    for second in choices[1]:
                        if first[0] == second[0] or first[2] == second[2]:
                            continue
                        if require_same_target and first[1] != second[1]:
                            continue
                        targets[0].product_slot_id = first[0]
                        targets[0].shelf_level = shelf_level(first[0])
                        targets[1].product_slot_id = second[0]
                        targets[1].shelf_level = shelf_level(second[0])
                        return [(first[1], first[2]), (second[1], second[2])]
            # No distinct left/right pair exists. A serial two-trip plan is still
            # valid, but the two requested products cannot refer to one physical
            # slot even when the SKU service returns overlapping candidates.
            selected_pair = next(
                (
                    (first, second)
                    for first in choices[0]
                    for second in choices[1]
                    if first[0] != second[0]
                ),
                None,
            )
            if selected_pair is None:
                raise Task1ServiceError(
                    "NO_FEASIBLE_HAND_ASSIGNMENT",
                    "two products have no distinct physical slot assignment",
                    status_code=422,
                )
            selected = list(selected_pair)
        else:
            selected = [item[0] for item in choices]
        # No left/right pair exists: execute the one-item-per-trip path.
        for target, (slot, _, _) in zip(targets, selected):
            target.product_slot_id = slot
            target.shelf_level = shelf_level(slot)
        return [(target_id, hand) for _, target_id, hand in selected]

    def _select_products(
        self, product_names: list[str], logger: _Task1Log | _NullTaskLog
    ) -> list[str]:
        skipped = set(self.settings.skip_product_names)
        deferred = set(self.settings.defer_product_names)
        selected: list[str] = []
        deferred_products: list[str] = []
        for product_name in product_names:
            if product_name in skipped:
                logger.event(
                    "商品抓取策略",
                    "skipped",
                    product_name=product_name,
                    policy="skip",
                    reason="configured_product_skip",
                )
            elif product_name in deferred:
                deferred_products.append(product_name)
                logger.event(
                    "商品抓取策略",
                    "succeeded",
                    product_name=product_name,
                    policy="defer",
                )
            else:
                selected.append(product_name)
        return selected + deferred_products

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

    @staticmethod
    def _failure(
        operation: str,
        product_name: str | None,
        hand: Hand | None,
        error: Task1ServiceError,
    ) -> dict[str, str]:
        return {
            "operation": operation,
            "product_name": product_name or "",
            "hand": hand.value if hand is not None else "",
            "error_code": error.code,
            "message": error.message,
        }

    async def _pick_target_best_effort(
        self,
        target: TargetItem,
        index: int,
        task_run_id: str,
        logger: "_Task1Log",
        navigation_state: dict[str, str | None],
        held_items: dict[Hand, str],
        uncertain_hands: set[Hand],
        action_failures: list[dict[str, str]],
    ) -> bool:
        try:
            return await self._pick_target(
                target,
                index,
                task_run_id,
                logger,
                navigation_state,
                held_items,
                uncertain_hands,
                action_failures,
            )
        except Task1ServiceError as exc:
            navigation_state["target_id"] = None
            failure = self._failure("pick_prerequisite", target.product_name, target.hand, exc)
            action_failures.append(failure)
            logger.event(
                "商品处理",
                "skipped",
                **failure,
                fallback="continue_next_product",
            )
            return False

    async def _prepare_delivery_best_effort(
        self,
        task_run_id: str,
        logger: "_Task1Log",
        navigation_state: dict[str, str | None],
        action_failures: list[dict[str, str]],
        *,
        cycle: int | None = None,
    ) -> bool:
        try:
            await self._prepare_delivery(
                task_run_id, logger, cycle, navigation_state
            )
        except Task1ServiceError as exc:
            navigation_state["target_id"] = None
            failure = self._failure("delivery_prepare", None, None, exc)
            action_failures.append(failure)
            logger.event(
                "交付台准备",
                "failed",
                **failure,
                fallback="continue_to_finish",
            )
            return False
        return True

    async def _finish_navigation(
        self,
        task_run_id: str,
        logger: "_Task1Log",
        navigation_state: dict[str, str | None],
    ) -> None:
        attempts = (
            (self.settings.task_boundary, "boundary.1"),
            (self.settings.task_boundary, "boundary.2"),
            (self.settings.start_target_id, "start_fallback"),
        )
        last_error: Task1ServiceError | None = None
        for target_id, suffix in attempts:
            navigation_state["target_id"] = None
            logger.event("任务判定区导航", "started", target_id=target_id)
            try:
                await self._navigate(
                    target_id,
                    f"{task_run_id}:task1.finish.{suffix}.navigate",
                    logger,
                    navigation_state,
                )
            except Task1ServiceError as exc:
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

    async def _pick_target(
        self,
        target: TargetItem,
        index: int,
        task_run_id: str,
        logger: _Task1Log,
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
            logger.event("抓取", "skipped", **failure)
            return False
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
        succeeded = await self._run_action_with_recovery(
            operation="pick",
            event_name="抓取",
            product_name=target.product_name,
            hand=target.hand,
            action_key=f"{task_run_id}:task1.pick.{index}.pick",
            action=lambda key: self.client.pick(
                target.product_name,
                target.hand,
                target.shelf_level,
                key,
                slot_id=target.product_slot_id,
                target_id=target.target_id,
            ),
            logger=logger,
            action_failures=action_failures,
            uncertain_hands=uncertain_hands,
            navigation_state=navigation_state,
            initial_nudge_direction=_initial_nudge_direction(target),
            before_retry=lambda key: self.client.prepare_pose(
                "SHELF_PICK_READY",
                key,
                shelf_level=target.shelf_level,
            ),
        )
        if succeeded:
            target.picked = True
            held_items[target.hand] = target.product_name
        return succeeded

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

    async def _place_target(
        self,
        target: TargetItem,
        index: int,
        task_run_id: str,
        logger: _Task1Log,
        held_items: dict[Hand, str],
        uncertain_hands: set[Hand],
        action_failures: list[dict[str, str]],
    ) -> bool:
        if not target.picked or held_items.get(target.hand) != target.product_name:
            logger.event(
                "放置",
                "skipped",
                product_name=target.product_name,
                hand=target.hand.value,
                reason="pick_not_succeeded",
            )
            return False
        succeeded = await self._run_action_with_recovery(
            operation="place",
            event_name="放置",
            product_name=target.product_name,
            hand=target.hand,
            action_key=f"{task_run_id}:task1.place.{index}.place",
            action=lambda key: self.client.place(target.product_name, target.hand, key),
            logger=logger,
            action_failures=action_failures,
            uncertain_hands=uncertain_hands,
        )
        if succeeded:
            target.placed = True
            held_items.pop(target.hand, None)
        return succeeded

    async def _place_both_targets(
        self,
        targets: list[TargetItem],
        task_run_id: str,
        logger: _Task1Log,
        held_items: dict[Hand, str],
        uncertain_hands: set[Hand],
        action_failures: list[dict[str, str]],
    ) -> bool:
        left_product = held_items[Hand.LEFT]
        right_product = held_items[Hand.RIGHT]
        details = {
            "left_product_name": left_product,
            "right_product_name": right_product,
            "attempt": 1,
        }
        logger.event("双手放置", "started", **details)
        try:
            await self.client.place_both(
                left_product,
                right_product,
                f"{task_run_id}:task1.place.both",
            )
        except Task1ServiceError as exc:
            if exc.code in {
                "ACTION_RESULT_UNKNOWN",
                "NETWORK_ERROR",
                "INVALID_RESPONSE",
            }:
                uncertain_hands.update((Hand.LEFT, Hand.RIGHT))
            logger.event(
                "双手放置",
                "failed",
                **details,
                error_code=exc.code,
                message=exc.message,
                failed_interface=exc.failed_interface,
                url=exc.url,
            )
            action_failures.append(
                {
                    "operation": "place_both",
                    "product_name": f"{left_product}, {right_product}",
                    "hand": "BOTH",
                    "error_code": exc.code,
                    "message": exc.message,
                }
            )
            return False

        logger.event("双手放置", "succeeded", **details)
        for target in targets:
            if held_items.get(target.hand) == target.product_name:
                target.placed = True
        held_items.clear()
        return True

    async def _run_action_with_recovery(
        self,
        *,
        operation: str,
        event_name: str,
        product_name: str,
        hand: Hand,
        action_key: str,
        action: Callable[[str], Awaitable[None]],
        logger: _Task1Log,
        action_failures: list[dict[str, str]],
        uncertain_hands: set[Hand],
        navigation_state: dict[str, str | None] | None = None,
        initial_nudge_direction: str | None = None,
        before_retry: Callable[[str], Awaitable[None]] | None = None,
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
            except Task1ServiceError as nudge_error:
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
        initial_error: Task1ServiceError | None = None
        try:
            await action(action_key)
        except Task1ServiceError as exc:
            initial_error = exc
            logger.event(
                event_name,
                "failed",
                product_name=product_name,
                hand=hand.value,
                attempt=1,
                error_code=exc.code,
                message=exc.message,
                failed_interface=exc.failed_interface,
                url=exc.url,
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
                self._failure(operation, product_name, hand, initial_error)
            )
            return False
        direction = _recovery_direction(initial_error, operation)
        if direction is None:
            retry_error = initial_error
            retry_ready = True
            if before_retry is not None:
                try:
                    await before_retry(f"{action_key}:recovery.pose")
                except Task1ServiceError as exc:
                    retry_ready = False
                    retry_error = exc
            if retry_ready:
                logger.event(
                    event_name,
                    "started",
                    product_name=product_name,
                    hand=hand.value,
                    attempt=2,
                )
                try:
                    await action(f"{action_key}:recovery.retry")
                except Task1ServiceError as exc:
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
        except Task1ServiceError as nudge_error:
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
            retry_ready = True
            if before_retry is not None:
                logger.event(
                    "抓取位姿恢复",
                    "started",
                    product_name=product_name,
                    hand=hand.value,
                    attempt=2,
                )
                try:
                    await before_retry(f"{action_key}:recovery.pose")
                except Task1ServiceError as pose_error:
                    retry_ready = False
                    final_error = pose_error
                    logger.event(
                        "抓取位姿恢复",
                        "failed",
                        product_name=product_name,
                        hand=hand.value,
                        attempt=2,
                        error_code=pose_error.code,
                        message=pose_error.message,
                        failed_interface=pose_error.failed_interface,
                        url=pose_error.url,
                    )
                else:
                    logger.event(
                        "抓取位姿恢复",
                        "succeeded",
                        product_name=product_name,
                        hand=hand.value,
                        attempt=2,
                    )
            if retry_ready:
                logger.event(
                    event_name,
                    "started",
                    product_name=product_name,
                    hand=hand.value,
                    attempt=2,
                )
                try:
                    await action(f"{action_key}:recovery.retry")
                except Task1ServiceError as retry_error:
                    final_error = retry_error
                    logger.event(
                        event_name,
                        "failed",
                        product_name=product_name,
                        hand=hand.value,
                        attempt=2,
                        error_code=retry_error.code,
                        message=retry_error.message,
                        failed_interface=retry_error.failed_interface,
                        url=retry_error.url,
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
        logger: _Task1Log,
        navigation_state: dict[str, str | None] | None = None,
    ) -> None:
        last_error: Task1ServiceError | None = None
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
            except Task1ServiceError as exc:
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
        status = (
            "succeeded"
            if isinstance(status_code, int) and 200 <= status_code < 300
            else "failed"
        )
        self.event(
            "接口调用",
            status,
            call_id=trace.get("call_id"),
            interface=trace.get("interface"),
            service=trace.get("service"),
            method=trace.get("method"),
            url=trace.get("url"),
            duration_ms=trace.get("duration_ms"),
            call_count=trace.get("call_count"),
            success_count=trace.get("success_count"),
            failure_count=trace.get("failure_count"),
            total_duration_ms=trace.get("total_duration_ms"),
            average_duration_ms=trace.get("average_duration_ms"),
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


class _NullTaskLog:
    def event(self, name: str, status: str, **details: object) -> None:
        pass

    def interface_event(self, trace: dict[str, object]) -> None:
        pass
