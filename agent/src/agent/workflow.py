"""基于 LangGraph 的三类零售服务比赛任务编排。

本模块是项目的业务核心：它把能力服务提供的导航、感知、位姿和抓放动作组合为
商品拣选、货架补货、乱放归位三张状态图。这里不实现机器人算法，只维护任务状态、
校验比赛规则、生成搬运作业，并决定每一步成功或失败后应流向哪个节点。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from agent.client import CapabilityClient
from agent.models import (
    ActionStatus,
    AgentError,
    AgentSettings,
    Hand,
    Job,
    RuleError,
    TaskType,
    WorkflowState,
    WorkflowStatus,
    shelf_level,
    validate_slot_id,
)


LOGGER = logging.getLogger(__name__)
PREPARATION_DESCRIPTIONS = {
    "RECEIPT_VIEW": "前往小票观察点并准备拍摄姿态",
    "SHELF_VIEW_UPPER": "前往巡检点并准备货架上半部观察姿态",
    "SHELF_VIEW_LOWER": "准备货架下半部观察姿态",
    "SHELF_PICK_READY": "前往商品货位并准备货架抓取姿态",
    "DELIVERY_TABLE_PLACE_READY": "前往交付台并准备放置姿态",
    "REPLENISHMENT_TABLE_PICK_READY": "前往补货台并准备抓取姿态",
    "SHELF_PLACE_READY": "前往目标货位并准备货架放置姿态",
}


def initial_state(settings: AgentSettings, task_type: TaskType) -> WorkflowState:
    """为一次独立任务创建全新的 LangGraph 状态。

    列表和字典均在每次调用时重新创建，防止不同任务共享可变对象。巡检从第 1 轮、
    第 0 个点开始；物理动作尚未发生，因此最近动作状态为 ``IDLE``。
    """

    return {
        "task_run_id": uuid4().hex,
        "task_type": task_type.value,
        "status": WorkflowStatus.RUNNING.value,
        "inspection_points": list(settings.inspection_points),
        "inspection_index": 0,
        "inspection_pass": 1,
        "target_items": [],
        "findings": [],
        "jobs": [],
        "current_job_index": 0,
        "held_items": {},
        "current_action_id": None,
        "current_action_status": ActionStatus.IDLE.value,
        "error_code": None,
        "error_message": None,
    }


class WorkflowBuilder:
    """把任务节点、业务规则和条件边组装成可执行的 LangGraph 图。

    每个异步节点接收完整 ``WorkflowState``，只返回需要更新的字段。节点不会原地
    修改状态中的列表或字典，从而符合 LangGraph 的状态合并模型。
    """

    def __init__(self, settings: AgentSettings, client: CapabilityClient) -> None:
        """绑定经过校验的配置和能力服务客户端。"""

        self.settings = settings
        self.client = client

    def build(self, task_type: TaskType):
        """按任务类型构建并编译对应状态图。"""

        LOGGER.debug("构建工作流图 | task_type=%s", task_type.value)
        if task_type is TaskType.SORTING:
            return self._build_sorting_graph()
        if task_type is TaskType.SHORTAGE:
            return self._build_shortage_graph()
        if task_type is TaskType.MISPLACED:
            return self._build_misplaced_graph()
        raise ValueError(f"unsupported task type: {task_type}")

    async def check_health(self, state: WorkflowState) -> dict[str, Any]:
        """启动前检查全部能力模块，避免任务中途才发现模块不可用。"""

        LOGGER.info("开始步骤 | 内容=检查主流程依赖的5个服务是否全部就绪")
        try:
            await self.client.check_all_health()
        except AgentError as exc:
            return self._failure(exc)
        LOGGER.info("步骤完成 | 内容=导航、视觉、姿态、商品库和取放编排服务均已就绪")
        return {}

    async def prepare_receipt(self, state: WorkflowState) -> dict[str, Any]:
        """商品拣选：前往交付台的小票观察点，同时切换到拍摄位姿。"""

        return await self._prepare(
            state,
            "receipt_viewpoint",
            "RECEIPT_VIEW",
            None,
            "sorting.receipt.prepare",
        )

    async def parse_receipt(self, state: WorkflowState) -> dict[str, Any]:
        """商品拣选：识别商品名，并通过商品库解析唯一标准货位。"""

        LOGGER.info("开始步骤 | 内容=识别小票中的两个商品")
        try:
            receipt_items = await self.client.parse_receipt()
            # 当前接口返回 product_names；旧版联调 Mock 返回货位数组。两者都在
            # 这里归一化为后续作业所需的 (商品名, 标准货位) 对。
            if all(item.startswith("H") for item in receipt_items):
                slots = receipt_items
                self._validate_distinct_slots(slots)
                names = await asyncio.gather(*(self.client.sku_name(slot) for slot in slots))
            else:
                names = receipt_items
                resolved = await asyncio.gather(
                    *(self.client.sku_search_by_name(name) for name in names)
                )
                slots = []
                for name, result in zip(names, resolved):
                    if len(result.locations) != 1:
                        raise RuleError(
                            "AMBIGUOUS_PRODUCT_SLOT",
                            f"商品 {name} 必须解析为唯一标准货位",
                        )
                    slot = result.locations[0]
                    validate_slot_id(self.settings, slot)
                    slots.append(slot)
                self._validate_distinct_slots(slots)
        except AgentError as exc:
            return self._failure(exc)
        except RuleError as exc:
            return self._failure(exc)
        LOGGER.info("步骤完成 | 内容=小票识别 | 商品=%s | 货位=%s", names, slots)
        return {"target_items": names, "findings": slots}

    async def build_sorting_jobs(self, state: WorkflowState) -> dict[str, Any]:
        """商品拣选：将两个清单货位转换成按路线排序的搬运作业。

        路线靠前的商品交给左手，第二件交给右手；两件商品最终都送到交付台。
        虽然小票可能给出两个货位，但比赛要求的是两种商品，因此还要拒绝两个货位
        映射到同一商品的情况。
        """

        try:
            # route_order 来自场地标定配置，用于减少机器人在货架之间往返。
            slots = sorted(
                state["findings"],
                key=lambda slot_id: self.settings.slot(slot_id).route_order,
            )
            names_by_slot = dict(zip(state.get("findings", []), state.get("target_items", [])))
            names = [
                names_by_slot[slot_id]
                if slot_id in names_by_slot
                else await self.client.sku_name(slot_id)
                for slot_id in slots
            ]
            if len(set(names)) != 2:
                raise RuleError("INVALID_RECEIPT", "receipt must identify two different products")
            jobs = [
                self._job(index, names[index], slots[index], "delivery_place", self._hand(index))
                for index in range(2)
            ]
        except AgentError as exc:
            return self._failure(exc)
        LOGGER.info("生成搬运作业 | 任务=商品拣选 | %s", self._describe_jobs(jobs))
        return {"jobs": jobs, "current_job_index": 0}

    async def prepare_sorting_pick(self, state: WorkflowState) -> dict[str, Any]:
        """商品拣选：导航到当前作业货位，并准备对应层的抓取姿态。"""

        job = self._current_job(state)
        return await self._prepare(
            state,
            job["source"],
            "SHELF_PICK_READY",
            shelf_level(job["source"]),
            f"sorting.{job['job_id']}.prepare_pick",
        )

    async def pick_current(self, state: WorkflowState) -> dict[str, Any]:
        """抓取当前作业商品；成功后推进作业游标，供条件边判断是否继续。"""

        index = state["current_job_index"]
        result = await self._pick_job(state, index)
        if result.get("status") == WorkflowStatus.FAILED.value:
            return result
        result["current_job_index"] = index + 1
        return result

    async def prepare_delivery(self, state: WorkflowState) -> dict[str, Any]:
        """商品拣选：两件商品都抓取后前往交付台，并重置放置循环游标。"""

        result = await self._prepare(
            state,
            "delivery_place",
            "DELIVERY_TABLE_PLACE_READY",
            None,
            "sorting.delivery.prepare",
        )
        if result.get("status") != WorkflowStatus.FAILED.value:
            result["current_job_index"] = 0
        return result

    async def place_current(self, state: WorkflowState) -> dict[str, Any]:
        """放置当前作业商品；成功后推进作业游标。"""

        index = state["current_job_index"]
        result = await self._place_job(state, index)
        if result.get("status") == WorkflowStatus.FAILED.value:
            return result
        result["current_job_index"] = index + 1
        return result

    async def prepare_inspection(self, state: WorkflowState) -> dict[str, Any]:
        """补货/归位：前往当前巡检点，并准备上半部货架观察位姿。

        奇数轮按配置顺序访问，偶数轮反向访问。新一轮的第一个点就是上一轮的最后
        一个点，机器人位置不变，只需切换到上半部观察位姿。
        """

        target = state["inspection_points"][state["inspection_index"]]
        if self._is_repeated_pass_start(state):
            LOGGER.info(
                "无需移动 | 内容=机器人仍在上一轮最后一个巡检点，切换到上半部拍摄姿态 "
                "| 巡检轮次=%d | 巡检点=%s",
                state["inspection_pass"],
                target,
            )
            return await self._prepare_pose_only(
                state,
                "SHELF_VIEW_UPPER",
                None,
                f"{state['task_type'].lower()}.inspect.{state['inspection_pass']}."
                f"{state['inspection_index']}.upper",
            )
        action_prefix = (
            f"{state['task_type'].lower()}.inspect.{state['inspection_pass']}."
            f"{state['inspection_index']}"
        )
        return await self._prepare(
            state,
            target,
            "SHELF_VIEW_UPPER",
            None,
            f"{action_prefix}.upper",
        )

    async def inspect_shortage(self, state: WorkflowState) -> dict[str, Any]:
        """货架补货：巡检当前货架面，并跨点位有序去重、累计缺货位。

        比赛要求补齐两处。单次或累计超过两处说明感知结果不符合本任务场景；不足
        两处则持续正反向巡检，直到满足条件或外部接口失败。
        """

        inspection_point = state["inspection_points"][state["inspection_index"]]
        LOGGER.info(
            "开始步骤 | 内容=识别当前货架上的缺货位置 | 巡检轮次=%d | 巡检点=%s",
            state["inspection_pass"],
            inspection_point,
        )
        try:
            upper_findings = await self.client.inspect(TaskType.SHORTAGE)
            lower_pose = await self._prepare_pose_only(
                state,
                "SHELF_VIEW_LOWER",
                None,
                f"shortage.inspect.{state['inspection_pass']}."
                f"{state['inspection_index']}.lower",
            )
            if lower_pose.get("status") == WorkflowStatus.FAILED.value:
                return lower_pose
            lower_findings = await self.client.inspect(TaskType.SHORTAGE)
            new_findings = self._merge_findings(upper_findings, lower_findings)
            if len(new_findings) > 2:
                raise RuleError("INVALID_FINDINGS", "shortage response contains more than two slots")
            for slot_id in new_findings:
                validate_slot_id(self.settings, slot_id)

            # 复制旧列表后再合并，避免原地修改 LangGraph 传入的状态。
            merged = list(state["findings"])
            for slot_id in new_findings:
                if slot_id not in merged:
                    merged.append(slot_id)
            if len(merged) > 2:
                raise RuleError("INVALID_FINDINGS", "accumulated shortage findings exceed two slots")
            if len(merged) == 2:
                LOGGER.info(
                    "步骤完成 | 内容=已找到两处缺货 | 本次发现=%s | 累计结果=%s",
                    new_findings,
                    merged,
                )
                return {"findings": merged}
            next_position = self._advance_inspection(state)
            LOGGER.info(
                "继续巡检 | 原因=尚未找到两处缺货 | 本次发现=%s | 累计结果=%s "
                "| 下一轮次=%d | 下一点序号=%d",
                new_findings,
                merged,
                next_position.get("inspection_pass", state["inspection_pass"]),
                next_position["inspection_index"],
            )
            return {"findings": merged, **next_position}
        except AgentError as exc:
            return self._failure(exc)

    async def build_shortage_jobs(self, state: WorkflowState) -> dict[str, Any]:
        """货架补货：为每个缺货位创建“补货台 -> 目标货位”的作业。

        商品名由目标货位的标准摆放表反查，发现顺序决定左右手分配。
        """

        try:
            names = await asyncio.gather(
                *(self.client.sku_name(slot_id) for slot_id in state["findings"])
            )
            jobs = [
                self._job(
                    index,
                    names[index],
                    "replenishment_pickup",
                    slot_id,
                    self._hand(index),
                )
                for index, slot_id in enumerate(state["findings"])
            ]
        except AgentError as exc:
            return self._failure(exc)
        LOGGER.info("生成搬运作业 | 任务=货架补货 | %s", self._describe_jobs(jobs))
        return {"jobs": jobs, "current_job_index": 0}

    async def prepare_replenishment(self, state: WorkflowState) -> dict[str, Any]:
        """货架补货：前往补货台，并切换到补货箱抓取预备位姿。"""

        return await self._prepare(
            state,
            "replenishment_pickup",
            "REPLENISHMENT_TABLE_PICK_READY",
            None,
            "shortage.replenishment.prepare",
        )

    async def prepare_shortage_place(self, state: WorkflowState) -> dict[str, Any]:
        """货架补货：为当前商品准备目标货位和对应层的放置姿态。

        抓取循环结束时游标等于作业数量，因此进入放置阶段前需回到第 0 项；之后
        每完成一项，条件边会带着递增后的游标再次进入本节点。
        """

        index = state["current_job_index"]
        if index >= len(state["jobs"]):
            index = 0
        job = state["jobs"][index]
        result = await self._prepare(
            state,
            job["destination"],
            "SHELF_PLACE_READY",
            shelf_level(job["destination"]),
            f"shortage.{job['job_id']}.prepare_place",
        )
        if result.get("status") != WorkflowStatus.FAILED.value:
            result["current_job_index"] = index
        return result

    async def inspect_misplaced(self, state: WorkflowState) -> dict[str, Any]:
        """乱放归位：查找一对互换货位，无结果时继续下一巡检点。

        感知结果固定为 ``[P1, P2]``：P1 是商品当前所在的错误货位，P2 是该商品
        的标准货位。两个编号都必须有效且不同。
        """

        inspection_point = state["inspection_points"][state["inspection_index"]]
        LOGGER.info(
            "开始步骤 | 内容=识别当前货架上的乱放商品 | 巡检轮次=%d | 巡检点=%s",
            state["inspection_pass"],
            inspection_point,
        )
        try:
            upper_findings = await self.client.inspect(TaskType.MISPLACED)
            lower_pose = await self._prepare_pose_only(
                state,
                "SHELF_VIEW_LOWER",
                None,
                f"misplaced.inspect.{state['inspection_pass']}."
                f"{state['inspection_index']}.lower",
            )
            if lower_pose.get("status") == WorkflowStatus.FAILED.value:
                return lower_pose
            lower_findings = await self.client.inspect(TaskType.MISPLACED)
            findings = self._merge_findings(upper_findings, lower_findings)
            if findings:
                self._validate_distinct_slots(findings)
                LOGGER.info(
                    "步骤完成 | 内容=找到需要交换归位的两个货位 | 错误货位=%s | 标准货位=%s",
                    findings[0],
                    findings[1],
                )
                return {"findings": findings}
            next_position = self._advance_inspection(state)
            LOGGER.info(
                "继续巡检 | 原因=当前货架未发现乱放商品 | 下一轮次=%d | 下一点序号=%d",
                next_position.get("inspection_pass", state["inspection_pass"]),
                next_position["inspection_index"],
            )
            return next_position
        except AgentError as exc:
            return self._failure(exc)

    async def build_misplaced_jobs(self, state: WorkflowState) -> dict[str, Any]:
        """乱放归位：利用“两件商品互换位置”的前提生成两个固定作业。

        P1 位置实际放着 P2 对应商品，先由左手从 P1 取出并送往 P2；P2 位置实际
        放着 P1 对应商品，由右手取出并送回 P1。先全部取出再放置，避免目标货位
        仍被另一件商品占用。
        """

        try:
            current_slot, standard_slot = state["findings"]
            current_name, standard_name = await asyncio.gather(
                self.client.sku_name(current_slot),
                self.client.sku_name(standard_slot),
            )
            jobs = [
                self._job(
                    0,
                    standard_name,
                    current_slot,
                    standard_slot,
                    "LEFT",
                ),
                self._job(
                    1,
                    current_name,
                    standard_slot,
                    current_slot,
                    "RIGHT",
                ),
            ]
        except (AgentError, ValueError) as exc:
            error = exc if isinstance(exc, AgentError) else RuleError("INVALID_FINDINGS", str(exc))
            return self._failure(error)
        LOGGER.info("生成搬运作业 | 任务=乱放归位 | %s", self._describe_jobs(jobs))
        return {"jobs": jobs, "current_job_index": 0}

    async def prepare_misplaced_pick_left(self, state: WorkflowState) -> dict[str, Any]:
        """乱放归位：准备在 P1 用左手抓取第一件商品。"""

        return await self._prepare_fixed_pick(state, 0)

    async def pick_misplaced_left(self, state: WorkflowState) -> dict[str, Any]:
        """乱放归位：从 P1 抓取应当放到 P2 的商品。"""

        return await self._pick_job(state, 0)

    async def prepare_misplaced_pick_right(self, state: WorkflowState) -> dict[str, Any]:
        """乱放归位：准备在 P2 用右手抓取第二件商品。"""

        return await self._prepare_fixed_pick(state, 1)

    async def pick_misplaced_right(self, state: WorkflowState) -> dict[str, Any]:
        """乱放归位：从 P2 抓取应当放回 P1 的商品。"""

        return await self._pick_job(state, 1)

    async def prepare_misplaced_place_left(self, state: WorkflowState) -> dict[str, Any]:
        """乱放归位：机器人已在 P2，只准备左手放置位姿。"""

        job = state["jobs"][0]
        return await self._prepare_pose_only(
            state,
            "SHELF_PLACE_READY",
            shelf_level(job["destination"]),
            f"misplaced.{job['job_id']}.prepare_place",
        )

    async def place_misplaced_left(self, state: WorkflowState) -> dict[str, Any]:
        """乱放归位：左手将第一件商品放入其标准货位 P2。"""

        return await self._place_job(state, 0)

    async def prepare_misplaced_place_right(self, state: WorkflowState) -> dict[str, Any]:
        """乱放归位：返回 P1，准备右手放置。"""

        return await self._prepare_fixed_place(state, 1)

    async def place_misplaced_right(self, state: WorkflowState) -> dict[str, Any]:
        """乱放归位：右手将第二件商品放回其标准货位 P1。"""

        return await self._place_job(state, 1)

    async def finish(self, state: WorkflowState) -> dict[str, Any]:
        """核验全部搬运已完成且双手为空，然后导航至任务判定区。

        根据比赛规则，机器人完全进入判定区才算本环节结束，因此该导航本身也是
        必须成功的物理动作，不能只在本地把任务标记为成功。
        """

        LOGGER.info(
            "开始步骤 | 内容=确认商品全部放置且双手为空，然后返回任务判定区 | 当前持物=%s "
            "| 作业数=%d",
            self._describe_held_items(state["held_items"]),
            len(state["jobs"]),
        )
        if state["held_items"] or any(not job["placed"] for job in state["jobs"]):
            return self._failure(
                RuleError("PRECONDITION_FAILED", "cannot finish with incomplete jobs or held items")
            )
        result = await self._physical(
            state,
            f"{state['task_type'].lower()}.finish.navigation",
            lambda: self.client.navigate(
                "task_boundary",
                state["task_run_id"],
                f"{state['task_type'].lower()}.finish.navigation",
            ),
        )
        if result.get("status") != WorkflowStatus.FAILED.value:
            LOGGER.info("步骤完成 | 内容=机器人已返回任务判定区")
        return result

    async def success(self, state: WorkflowState) -> dict[str, Any]:
        """统一成功终点：将整个任务状态置为 ``SUCCEEDED``。"""

        LOGGER.info("工作流完成 | 结果=成功 | 已完成作业=%d", len(state["jobs"]))
        return {"status": WorkflowStatus.SUCCEEDED.value}

    async def fail(self, state: WorkflowState) -> dict[str, Any]:
        """统一失败终点：保留前序节点写入的错误详情，并停止状态图。"""

        LOGGER.debug("工作流已进入失败终点 | 错误码=%s", state["error_code"])
        return {"status": WorkflowStatus.FAILED.value}

    async def _prepare(
        self,
        state: WorkflowState,
        target_id: str,
        pose_type: str,
        level: str | None,
        action_prefix: str,
    ) -> dict[str, Any]:
        """先导航到目标点，导航成功后再准备姿态。"""

        description = PREPARATION_DESCRIPTIONS.get(pose_type, "导航到目标点并准备机器人姿态")
        LOGGER.info(
            "开始步骤 | 内容=%s | 目标=%s | 姿态=%s | 货架层=%s",
            description,
            target_id,
            pose_type,
            level or "无",
        )
        navigation = await self._physical(
            state,
            f"{action_prefix}.navigation",
            lambda: self.client.navigate(
                target_id,
                state["task_run_id"],
                f"{action_prefix}.navigation",
            ),
        )
        if navigation.get("status") == WorkflowStatus.FAILED.value:
            return navigation
        pose = await self._physical(
            state,
            f"{action_prefix}.pose",
            lambda: self.client.prepare_pose(
                pose_type,
                level,
                state["task_run_id"],
                f"{action_prefix}.pose",
            ),
        )
        if pose.get("status") == WorkflowStatus.FAILED.value:
            return pose
        LOGGER.info(
            "步骤完成 | 内容=%s | 目标=%s",
            description,
            target_id,
        )
        return {
            "current_action_id": action_prefix,
            "current_action_status": ActionStatus.SUCCEEDED.value,
        }

    async def _prepare_pose_only(
        self,
        state: WorkflowState,
        pose_type: str,
        level: str | None,
        action_prefix: str,
    ) -> dict[str, Any]:
        """机器人已经位于目标点时，只执行位姿准备，不重复发送导航。"""

        pose_description = {
            "SHELF_VIEW_UPPER": "机器人已在巡检点，仅准备货架上半部拍摄姿态",
            "SHELF_VIEW_LOWER": "机器人已在巡检点，仅准备货架下半部拍摄姿态",
        }.get(pose_type, "机器人已在目标点，仅准备货架放置姿态")
        LOGGER.info(
            "开始步骤 | 内容=%s | 姿态=%s | 货架层=%s",
            pose_description,
            pose_type,
            level or "无",
        )
        result = await self._physical(
            state,
            action_prefix,
            lambda: self.client.prepare_pose(
                pose_type,
                level,
                state["task_run_id"],
                f"{action_prefix}.pose",
            ),
        )
        if result.get("status") != WorkflowStatus.FAILED.value:
            LOGGER.info(
                "步骤完成 | 内容=%s | 姿态=%s",
                pose_description,
                pose_type,
            )
        return result

    async def _pick_job(self, state: WorkflowState, index: int) -> dict[str, Any]:
        """执行一项抓取，并仅在服务确认成功后更新作业和手部占用状态。"""

        job = state["jobs"][index]
        hand_name = self._hand_name(job["hand"])
        LOGGER.info(
            "开始步骤 | 内容=使用%s抓取商品 | 商品=%s | 取货位置=%s",
            hand_name,
            job["product_name"],
            job["source"],
        )
        # 同一只手不能同时持有两件商品，这是下发抓取动作前的本地安全前置条件。
        if job["hand"] in state["held_items"]:
            return self._failure(
                RuleError("PRECONDITION_FAILED", f"{job['hand']} hand is not empty")
            )
        action_id = f"{state['task_type'].lower()}.{job['job_id']}.pick"
        result = await self._physical(
            state,
            action_id,
            lambda: self.client.pick(
                TaskType(state["task_type"]),
                job["product_name"],
                job["hand"],
                state["task_run_id"],
                action_id,
            ),
        )
        if result.get("status") == WorkflowStatus.FAILED.value:
            return result

        # 使用副本提交局部状态更新，不直接篡改当前节点收到的 State。
        jobs = [dict(item) for item in state["jobs"]]
        jobs[index]["picked"] = True
        held_items = dict(state["held_items"])
        held_items[job["hand"]] = job["product_name"]
        LOGGER.info(
            "步骤完成 | 内容=%s已抓取%s | 当前持物=%s",
            hand_name,
            job["product_name"],
            self._describe_held_items(held_items),
        )
        return {**result, "jobs": jobs, "held_items": held_items}

    async def _place_job(self, state: WorkflowState, index: int) -> dict[str, Any]:
        """执行一项放置，并确认指定手确实持有该作业对应的商品。"""

        job = state["jobs"][index]
        hand_name = self._hand_name(job["hand"])
        LOGGER.info(
            "开始步骤 | 内容=使用%s放置商品 | 商品=%s | 放置位置=%s",
            hand_name,
            job["product_name"],
            job["destination"],
        )
        if state["held_items"].get(job["hand"]) != job["product_name"]:
            return self._failure(
                RuleError("PRECONDITION_FAILED", f"{job['hand']} hand does not hold the job item")
            )
        action_id = f"{state['task_type'].lower()}.{job['job_id']}.place"
        result = await self._physical(
            state,
            action_id,
            lambda: self.client.place(
                TaskType(state["task_type"]),
                job["product_name"],
                job["hand"],
                state["task_run_id"],
                action_id,
            ),
        )
        if result.get("status") == WorkflowStatus.FAILED.value:
            return result

        # 收到明确成功响应后，才把作业标为完成并释放对应手。
        jobs = [dict(item) for item in state["jobs"]]
        jobs[index]["placed"] = True
        held_items = dict(state["held_items"])
        held_items.pop(job["hand"])
        LOGGER.info(
            "步骤完成 | 内容=%s已放置%s | 当前持物=%s",
            hand_name,
            job["product_name"],
            self._describe_held_items(held_items),
        )
        return {**result, "jobs": jobs, "held_items": held_items}

    async def _physical(
        self,
        state: WorkflowState,
        action_id: str,
        call: Callable[[], Awaitable[None]],
    ) -> dict[str, Any]:
        """包装单个物理动作，把 ``AgentError`` 转换为 LangGraph 状态更新。"""

        try:
            await call()
        except AgentError as exc:
            return self._failure(exc, action_id)
        return {
            "current_action_id": action_id,
            "current_action_status": ActionStatus.SUCCEEDED.value,
        }

    def _advance_inspection(self, state: WorkflowState) -> dict[str, Any]:
        """奇数轮正向、偶数轮反向推进；换轮时停留在当前边界点。"""

        direction = 1 if state["inspection_pass"] % 2 == 1 else -1
        next_index = state["inspection_index"] + direction
        if 0 <= next_index < len(state["inspection_points"]):
            return {"inspection_index": next_index}
        return {
            # 下一轮反向访问，首点就是当前轮的末点，不需要移动机器人。
            "inspection_index": state["inspection_index"],
            "inspection_pass": state["inspection_pass"] + 1,
        }

    @staticmethod
    def _is_repeated_pass_start(state: WorkflowState) -> bool:
        """判断当前点是否为第 2 轮及以后与上一轮重合的首个巡检点。"""

        if state["inspection_pass"] <= 1:
            return False
        boundary_index = (
            len(state["inspection_points"]) - 1
            if state["inspection_pass"] % 2 == 0
            else 0
        )
        return state["inspection_index"] == boundary_index

    def _validate_distinct_slots(self, slots: list[str]) -> None:
        """校验结果恰好包含两个不同且存在于摆放表中的货位。"""

        if len(slots) != 2 or len(set(slots)) != 2:
            raise RuleError("INVALID_FINDINGS", "expected two different product slots")
        for slot_id in slots:
            validate_slot_id(self.settings, slot_id)

    @staticmethod
    def _merge_findings(*groups: list[str]) -> list[str]:
        """按上下拍摄顺序合并识别结果，并保留首次出现顺序。"""

        merged: list[str] = []
        for group in groups:
            for slot_id in group:
                if slot_id not in merged:
                    merged.append(slot_id)
        return merged

    @staticmethod
    def _job(
        index: int,
        product_name: str,
        source: str,
        destination: str,
        hand: Hand,
    ) -> Job:
        """创建尚未执行的标准搬运作业。"""

        return {
            "job_id": f"job_{index}",
            "product_name": product_name,
            "source": source,
            "destination": destination,
            "hand": hand,
            "picked": False,
            "placed": False,
        }

    @staticmethod
    def _hand(index: int) -> Hand:
        """按作业顺序固定分配双手：第一件左手，第二件右手。"""

        return "LEFT" if index == 0 else "RIGHT"

    @staticmethod
    def _hand_name(hand: Hand | str) -> str:
        """返回日志使用的中文手部名称。"""

        return "左手" if hand == "LEFT" else "右手"

    @classmethod
    def _describe_jobs(cls, jobs: list[Job]) -> str:
        """把内部 Job 字典压缩为面向操作人员的搬运说明。"""

        return "；".join(
            f"{cls._hand_name(job['hand'])}从{job['source']}取{job['product_name']}，"
            f"放到{job['destination']}"
            for job in jobs
        )

    @classmethod
    def _describe_held_items(cls, held_items: dict[str, str]) -> str:
        """把持物字典转换为简短中文描述。"""

        if not held_items:
            return "双手为空"
        return "，".join(
            f"{cls._hand_name(hand)}={product}" for hand, product in held_items.items()
        )

    @staticmethod
    def _current_job(state: WorkflowState) -> Job:
        """取得作业游标当前指向的作业。"""

        return state["jobs"][state["current_job_index"]]

    @staticmethod
    def _failure(error: AgentError, action_id: str | None = None) -> dict[str, Any]:
        """将异常规范化为失败状态，并准确标记最近物理动作结果。

        没有关联物理动作的规则/健康检查错误保持 ``IDLE``；明确动作失败标记为
        ``FAILED``；两次传输失败无法确认执行结果时标记为 ``UNKNOWN``。
        """

        if action_id is None:
            action_status = ActionStatus.IDLE.value
        elif error.result_unknown:
            action_status = ActionStatus.UNKNOWN.value
        else:
            action_status = ActionStatus.FAILED.value
        LOGGER.error(
            "步骤失败 | 动作结果=%s | 错误码=%s | 原因=%s",
            action_status,
            error.code,
            error.message,
        )
        return {
            "status": WorkflowStatus.FAILED.value,
            "error_code": error.code,
            "error_message": error.message,
            "current_action_id": action_id,
            "current_action_status": action_status,
        }

    @staticmethod
    async def _route_after_step(state: WorkflowState) -> str:
        """普通节点路由：失败进入统一终点，否则继续下一业务节点。"""

        return "failed" if state["status"] == WorkflowStatus.FAILED.value else "ok"

    @staticmethod
    async def _route_job_loop(state: WorkflowState) -> str:
        """搬运循环路由：区分失败、仍有作业和全部完成三种情况。"""

        if state["status"] == WorkflowStatus.FAILED.value:
            return "failed"
        return "more" if state["current_job_index"] < len(state["jobs"]) else "done"

    @staticmethod
    async def _route_inspection(state: WorkflowState) -> str:
        """巡检路由：累计两个结果后建作业，否则继续巡检。"""

        if state["status"] == WorkflowStatus.FAILED.value:
            return "failed"
        return "found" if len(state["findings"]) == 2 else "continue"

    def _add_ok_edge(self, graph: StateGraph, source: str, target: str) -> None:
        """为普通节点添加“成功继续、失败终止”的公共条件边。"""

        graph.add_conditional_edges(
            source,
            self._route_after_step,
            {"ok": target, "failed": "fail"},
        )

    def _add_finish_nodes(self, graph: StateGraph) -> None:
        """向任务图挂载共用的返判定区、成功和失败终点。"""

        graph.add_node("finish", self.finish)
        graph.add_node("success", self.success)
        graph.add_node("fail", self.fail)
        self._add_ok_edge(graph, "finish", "success")
        graph.add_edge("success", END)
        graph.add_edge("fail", END)

    def _build_sorting_graph(self):
        """构建商品拣选图：读小票、按路线抓两件、到交付台逐件放置。"""

        graph = StateGraph(WorkflowState)
        # 节点只注册处理函数；实际先后关系由下方普通边和条件边定义。
        graph.add_node("check_health", self.check_health)
        graph.add_node("prepare_receipt", self.prepare_receipt)
        graph.add_node("parse_receipt", self.parse_receipt)
        graph.add_node("build_jobs", self.build_sorting_jobs)
        graph.add_node("prepare_pick", self.prepare_sorting_pick)
        graph.add_node("pick", self.pick_current)
        graph.add_node("prepare_delivery", self.prepare_delivery)
        graph.add_node("place", self.place_current)
        self._add_finish_nodes(graph)

        # 前置步骤任一失败都会立即进入 fail，不再发起新的物理动作。
        graph.add_edge(START, "check_health")
        self._add_ok_edge(graph, "check_health", "prepare_receipt")
        self._add_ok_edge(graph, "prepare_receipt", "parse_receipt")
        self._add_ok_edge(graph, "parse_receipt", "build_jobs")
        self._add_ok_edge(graph, "build_jobs", "prepare_pick")
        self._add_ok_edge(graph, "prepare_pick", "pick")
        graph.add_conditional_edges(
            "pick",
            self._route_job_loop,
            {"more": "prepare_pick", "done": "prepare_delivery", "failed": "fail"},
        )
        self._add_ok_edge(graph, "prepare_delivery", "place")
        graph.add_conditional_edges(
            "place",
            self._route_job_loop,
            # 两件商品在同一个交付台放置，无需每件都重复导航和准备姿态。
            {"more": "place", "done": "finish", "failed": "fail"},
        )
        return graph.compile()

    def _build_shortage_graph(self):
        """构建货架补货图：循环巡检、补货台抓两件、逐货位放置。"""

        graph = StateGraph(WorkflowState)
        graph.add_node("check_health", self.check_health)
        graph.add_node("prepare_inspection", self.prepare_inspection)
        graph.add_node("inspect", self.inspect_shortage)
        graph.add_node("build_jobs", self.build_shortage_jobs)
        graph.add_node("prepare_replenishment", self.prepare_replenishment)
        graph.add_node("pick", self.pick_current)
        graph.add_node("prepare_place", self.prepare_shortage_place)
        graph.add_node("place", self.place_current)
        self._add_finish_nodes(graph)

        graph.add_edge(START, "check_health")
        self._add_ok_edge(graph, "check_health", "prepare_inspection")
        self._add_ok_edge(graph, "prepare_inspection", "inspect")
        graph.add_conditional_edges(
            "inspect",
            self._route_inspection,
            # 未收集满两处时，状态中的巡检游标已经由 inspect 节点推进。
            {"continue": "prepare_inspection", "found": "build_jobs", "failed": "fail"},
        )
        self._add_ok_edge(graph, "build_jobs", "prepare_replenishment")
        self._add_ok_edge(graph, "prepare_replenishment", "pick")
        graph.add_conditional_edges(
            "pick",
            self._route_job_loop,
            # 两件补货商品来自同一补货台，所以连续抓取不重复准备点位。
            {"more": "pick", "done": "prepare_place", "failed": "fail"},
        )
        self._add_ok_edge(graph, "prepare_place", "place")
        graph.add_conditional_edges(
            "place",
            self._route_job_loop,
            # 两个目标货位可能不同，每次放置前都需重新导航并准备对应层姿态。
            {"more": "prepare_place", "done": "finish", "failed": "fail"},
        )
        return graph.compile()

    def _build_misplaced_graph(self):
        """构建乱放归位图，严格执行左抓、右抓、左放、右放的交换顺序。"""

        graph = StateGraph(WorkflowState)
        graph.add_node("check_health", self.check_health)
        graph.add_node("prepare_inspection", self.prepare_inspection)
        graph.add_node("inspect", self.inspect_misplaced)
        graph.add_node("build_jobs", self.build_misplaced_jobs)
        graph.add_node("prepare_pick_left", self.prepare_misplaced_pick_left)
        graph.add_node("pick_left", self.pick_misplaced_left)
        graph.add_node("prepare_pick_right", self.prepare_misplaced_pick_right)
        graph.add_node("pick_right", self.pick_misplaced_right)
        graph.add_node("prepare_place_left", self.prepare_misplaced_place_left)
        graph.add_node("place_left", self.place_misplaced_left)
        graph.add_node("prepare_place_right", self.prepare_misplaced_place_right)
        graph.add_node("place_right", self.place_misplaced_right)
        self._add_finish_nodes(graph)

        graph.add_edge(START, "check_health")
        self._add_ok_edge(graph, "check_health", "prepare_inspection")
        self._add_ok_edge(graph, "prepare_inspection", "inspect")
        graph.add_conditional_edges(
            "inspect",
            self._route_inspection,
            {"continue": "prepare_inspection", "found": "build_jobs", "failed": "fail"},
        )
        # 归位流程不是同构循环：两件商品的手和先后顺序承载互换规则，显式链条更清楚。
        sequence = (
            ("build_jobs", "prepare_pick_left"),
            ("prepare_pick_left", "pick_left"),
            ("pick_left", "prepare_pick_right"),
            ("prepare_pick_right", "pick_right"),
            ("pick_right", "prepare_place_left"),
            ("prepare_place_left", "place_left"),
            ("place_left", "prepare_place_right"),
            ("prepare_place_right", "place_right"),
            ("place_right", "finish"),
        )
        for source, target in sequence:
            self._add_ok_edge(graph, source, target)
        return graph.compile()

    async def _prepare_fixed_pick(
        self, state: WorkflowState, index: int
    ) -> dict[str, Any]:
        """按固定作业下标准备乱放商品的货架抓取。"""

        job = state["jobs"][index]
        return await self._prepare(
            state,
            job["source"],
            "SHELF_PICK_READY",
            shelf_level(job["source"]),
            f"misplaced.{job['job_id']}.prepare_pick",
        )

    async def _prepare_fixed_place(
        self, state: WorkflowState, index: int
    ) -> dict[str, Any]:
        """按固定作业下标准备乱放商品的货架放置。"""

        job = state["jobs"][index]
        return await self._prepare(
            state,
            job["destination"],
            "SHELF_PLACE_READY",
            shelf_level(job["destination"]),
            f"misplaced.{job['job_id']}.prepare_place",
        )
