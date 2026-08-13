"""Agent 的领域模型、运行状态和配置校验。

本模块不执行机器人动作，只负责定义各模块共同使用的数据契约。把货位格式、
配置完整性和错误类型集中在这里，可以让工作流层只关心业务步骤，并在启动时尽早
发现错误配置，而不是等机器人运动到一半才失败。
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# 商品货位编码格式：H=货架编号，F/B=正面/背面，L=层号，C=列号。
# 例如 H1_F_L2_C01 表示 1 号货架正面第 2 层第 01 列。
PRODUCT_SLOT_PATTERN = re.compile(r"^H[12]_[FB]_L[1-5]_C\d{2}$")

class TaskType(StrEnum):
    """工作流支持的三类比赛任务。"""

    SORTING = "SORTING"
    SHORTAGE = "SHORTAGE"
    MISPLACED = "MISPLACED"


class WorkflowStatus(StrEnum):
    """整个任务的生命周期状态。"""

    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class ActionStatus(StrEnum):
    """最近一次物理动作的执行状态。

    ``UNKNOWN`` 与 ``FAILED`` 不同：前者表示请求超时后无法确认机器人是否已经执行，
    此时继续下发动作可能造成重复抓取等危险，因此工作流必须立即停止。
    """

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


# 操作模块只接受左右手两个稳定值，使用 Literal 可在静态检查和运行校验中防止拼写错误。
Hand = Literal["LEFT", "RIGHT"]


class Job(TypedDict):
    """一次商品搬运作业。

    作业把“从哪里取、送到哪里、由哪只手操作”固定下来；``picked`` 和
    ``placed`` 用于记录执行进度，供结束节点检查任务是否真正完成。
    """

    job_id: str
    product_name: str
    source: str
    destination: str
    hand: Hand
    picked: bool
    placed: bool


class WorkflowState(TypedDict):
    """LangGraph 在所有节点之间传递和合并的完整状态。

    字段按用途可分为四组：任务身份与总状态、巡检游标与发现结果、搬运作业与
    双手占用情况、最近动作与错误信息。节点只返回需要更新的字段，LangGraph 会
    将它们合并回此状态。
    """

    # 每次运行的唯一标识，也是物理动作幂等键的一部分。
    task_run_id: str
    task_type: str
    status: str
    # 巡检点列表、当前位置和当前轮次共同组成可恢复的巡检游标。
    inspection_points: list[str]
    inspection_index: int
    inspection_pass: int
    target_items: list[str]
    findings: list[str]
    # jobs 保存规划后的搬运步骤，current_job_index 指向下一项待处理作业。
    jobs: list[Job]
    current_job_index: int
    held_items: dict[str, str]
    # 最近物理动作的信息用于诊断失败，尤其用于标记 ACTION_RESULT_UNKNOWN。
    current_action_id: str | None
    current_action_status: str
    error_code: str | None
    error_message: str | None


class TimeoutSettings(BaseModel):
    """各类外部调用的超时秒数。

    每个值必须大于零；连接超时与具体动作的读取超时分开配置，便于适配耗时较长
    的导航、抓取动作和较短的健康检查。
    """

    model_config = ConfigDict(extra="forbid")

    connect_seconds: float = Field(gt=0)
    health_seconds: float = Field(gt=0)
    sku_seconds: float = Field(gt=0, default=10)
    receipt_seconds: float = Field(gt=0)
    inspection_seconds: float = Field(gt=0)
    navigation_seconds: float = Field(gt=0)
    pose_seconds: float = Field(gt=0)
    pick_seconds: float = Field(gt=0)
    place_seconds: float = Field(gt=0)


class ServiceSettings(BaseModel):
    """Agent 直接依赖的能力服务基础 URL。"""

    model_config = ConfigDict(extra="forbid")

    navigation: str = Field(min_length=1)
    perception: str = Field(min_length=1)
    pose: str = Field(min_length=1)
    manipulation: str = Field(min_length=1)
    pick_place: str = Field(min_length=1)
    sku: str = Field(min_length=1)


class ProductSlot(BaseModel):
    """一个标准货位对应的商品及其最优访问顺序。"""

    model_config = ConfigDict(extra="forbid")

    product_name: str = Field(min_length=1)
    route_order: int = Field(ge=0)


class AgentSettings(BaseModel):
    """Agent 的顶层配置模型。

    ``extra='forbid'`` 会拒绝未声明字段，避免 YAML 中的拼写错误被静默忽略。
    """

    model_config = ConfigDict(extra="forbid")

    services: ServiceSettings
    inspection_points: list[str]
    timeouts: TimeoutSettings
    product_slots: dict[str, ProductSlot]

    @model_validator(mode="after")
    def validate_agent_config(self) -> AgentSettings:
        """执行涉及多个字段的配置一致性校验。"""

        if not self.inspection_points:
            raise ValueError("inspection_points must contain at least one point")
        if any(not point.strip() for point in self.inspection_points):
            raise ValueError("inspection_points must not contain empty values")
        if len(set(self.inspection_points)) != len(self.inspection_points):
            raise ValueError("inspection_points must not contain duplicates")

        # 所有配置货位都必须遵循比赛约定的结构化编码。
        invalid_slots = [
            slot_id for slot_id in self.product_slots if not PRODUCT_SLOT_PATTERN.fullmatch(slot_id)
        ]
        if invalid_slots:
            raise ValueError(f"invalid product slot ids: {invalid_slots}")

        # 路线序号决定分拣任务的取货顺序，重复值会导致顺序不确定。
        route_orders = [slot.route_order for slot in self.product_slots.values()]
        if len(route_orders) != len(set(route_orders)):
            raise ValueError("product slot route_order values must be unique")
        return self

    @classmethod
    def load(cls, path: str | Path) -> AgentSettings:
        """从 UTF-8 YAML 文件加载配置，并立即完成全部 Pydantic 校验。"""

        with Path(path).open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
        return cls.model_validate(raw_config)

    def slot(self, slot_id: str) -> ProductSlot:
        """查找货位配置，并把普通 KeyError 转换为工作流可识别的规则错误。"""

        try:
            return self.product_slots[slot_id]
        except KeyError as exc:
            raise RuleError("UNKNOWN_PRODUCT_SLOT", f"unknown product slot: {slot_id}") from exc


class AgentError(Exception):
    """可安全写入工作流状态的统一业务异常。

    ``result_unknown`` 专门描述物理动作请求超时后的不确定性。调用方据此将动作状态
    标为 ``UNKNOWN``，而不是错误地断言动作已经失败。
    """

    def __init__(self, code: str, message: str, *, result_unknown: bool = False) -> None:
        """保存稳定错误码、可读信息，以及物理动作结果是否无法确认。"""

        super().__init__(message)
        self.code = code
        self.message = message
        self.result_unknown = result_unknown


class RuleError(AgentError):
    """输入或状态违反业务规则时抛出的异常。"""

    pass


def validate_slot_id(settings: AgentSettings, slot_id: str) -> None:
    """同时验证货位编码格式以及该货位是否存在于当前配置。"""

    if not PRODUCT_SLOT_PATTERN.fullmatch(slot_id):
        raise RuleError("INVALID_PRODUCT_SLOT", f"invalid product slot: {slot_id}")
    settings.slot(slot_id)


def shelf_level(slot_id: str) -> str:
    """从结构化货位编码中提取层号，如 ``H1_F_L2_C01`` 得到 ``L2``。"""

    return slot_id.split("_")[2]
