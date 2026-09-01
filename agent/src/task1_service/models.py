"""任务一服务的数据契约和配置模型。"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PRODUCT_SLOT_PATTERN = re.compile(r"^(?:H[12]_[FB]_L[1-5]|H[1-3]_L0[1-5])_C\d{2}$")


class TaskType(StrEnum):
    SORTING = "SORTING"


class Hand(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class GraspOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hands: list[Hand]
    target_id: str = Field(min_length=1)

    @field_validator("hands")
    @classmethod
    def valid_hands(cls, value: list[Hand]) -> list[Hand]:
        if not value or len(set(value)) != len(value):
            raise ValueError("hands must be non-empty and unique")
        return value


class ProductHandOption(BaseModel):
    """一个货位在一个或多个导航点的抓取能力。"""

    model_config = ConfigDict(extra="forbid")

    product_name: str = Field(min_length=1)
    grasp_options: list[GraspOption]

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_shape(cls, value: Any) -> Any:
        if isinstance(value, dict) and "grasp_options" not in value:
            value = dict(value)
            hands = value.pop("hands", None)
            target_id = value.pop("target_id", None)
            value["grasp_options"] = [{"hands": hands, "target_id": target_id}]
        elif isinstance(value, dict) and isinstance(value.get("grasp_options"), dict):
            value = dict(value)
            value["grasp_options"] = [value["grasp_options"]]
        return value

    @field_validator("grasp_options")
    @classmethod
    def valid_options(cls, value: list[GraspOption]) -> list[GraspOption]:
        if not value:
            raise ValueError("grasp_options must not be empty")
        return value

    @property
    def hands(self) -> list[Hand]:
        return list(dict.fromkeys(hand for option in self.grasp_options for hand in option.hands))

    @property
    def target_id(self) -> str:
        return self.grasp_options[0].target_id


class ProductHandOptionsFile(BaseModel):
    """独立商品货位手能力和巡检点配置文件。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "2.0"]
    source_catalog_version: str = Field(min_length=1)
    product_hand_options: dict[str, ProductHandOption]

    @field_validator("product_hand_options")
    @classmethod
    def valid_slots(
        cls, value: dict[str, ProductHandOption]
    ) -> dict[str, ProductHandOption]:
        if not value:
            raise ValueError("product_hand_options must not be empty")
        invalid = [slot for slot in value if not PRODUCT_SLOT_PATTERN.fullmatch(slot)]
        if invalid:
            raise ValueError(f"invalid product slot ids in product_hand_options: {invalid}")
        return value


class Task1Services(BaseModel):
    model_config = ConfigDict(extra="forbid")

    navigation: str = Field(min_length=1)
    perception: str = Field(min_length=1)
    pose: str = Field(min_length=1)
    pick_place: str = Field(min_length=1)
    sku: str = Field(min_length=1)
    camera: str = Field(min_length=1)


class Task1Timeouts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connect_seconds: float = Field(gt=0, default=3)
    health_seconds: float = Field(gt=0, default=5)
    receipt_seconds: float = Field(gt=0, default=120)
    resolution_seconds: float = Field(gt=0, default=60)
    sku_seconds: float = Field(gt=0, default=10)
    navigation_seconds: float = Field(gt=0, default=600)
    pose_seconds: float = Field(gt=0, default=300)
    pick_seconds: float = Field(gt=0, default=600)
    place_seconds: float = Field(gt=0, default=600)


class Task1Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    services: Task1Services
    timeouts: Task1Timeouts = Field(default_factory=Task1Timeouts)
    receipt_viewpoint: str = Field(min_length=1, default="receipt_viewpoint")
    delivery_place: str = Field(min_length=1, default="delivery_place")
    task_boundary: str = Field(min_length=1, default="task_boundary")
    start_target_id: str = Field(min_length=1, default="start")
    product_hand_options_file: str | None = Field(default=None, min_length=1)
    product_hand_options: dict[str, list[Hand]] = Field(default_factory=dict)
    product_target_ids: dict[str, str] = Field(default_factory=dict)
    product_grasp_options: dict[str, list[GraspOption]] = Field(default_factory=dict)
    skip_product_names: list[str] = Field(default_factory=list)
    defer_product_names: list[str] = Field(default_factory=list)
    log_dir: str = Field(min_length=1, default="log")

    @field_validator("skip_product_names", "defer_product_names")
    @classmethod
    def valid_policy_product_names(cls, value: list[str]) -> list[str]:
        normalized = [name.strip() for name in value]
        if any(not name for name in normalized):
            raise ValueError("product policy names must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("product policy names must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def disjoint_product_policies(self) -> "Task1Settings":
        overlap = set(self.skip_product_names) & set(self.defer_product_names)
        if overlap:
            raise ValueError(
                "skip_product_names and defer_product_names must not overlap: "
                f"{sorted(overlap)}"
            )
        return self

    @field_validator("product_hand_options")
    @classmethod
    def valid_hand_options(cls, value: dict[str, list[Hand]]) -> dict[str, list[Hand]]:
        invalid = [slot for slot in value if not PRODUCT_SLOT_PATTERN.fullmatch(slot)]
        if invalid:
            raise ValueError(f"invalid product slot ids in product_hand_options: {invalid}")
        for slot, hands in value.items():
            if not hands:
                raise ValueError(f"product_hand_options[{slot}] must not be empty")
            if len(set(hands)) != len(hands):
                raise ValueError(f"product_hand_options[{slot}] contains duplicate hands")
        return value

    @field_validator("product_target_ids")
    @classmethod
    def valid_target_ids(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [slot for slot in value if not PRODUCT_SLOT_PATTERN.fullmatch(slot)]
        if invalid:
            raise ValueError(f"invalid product slot ids in product_target_ids: {invalid}")
        if any(not target_id.strip() for target_id in value.values()):
            raise ValueError("product_target_ids must contain non-empty target ids")
        return value

    @classmethod
    def load(cls, path: str | Path) -> "Task1Settings":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)

        return cls.from_mapping(raw_config, config_path.parent)

    @classmethod
    def from_mapping(
        cls, raw_config: dict[str, Any], base_dir: str | Path
    ) -> "Task1Settings":
        raw_config = dict(raw_config)

        options_file = raw_config.get("product_hand_options_file")
        if options_file:
            if raw_config.get("product_hand_options"):
                raise ValueError(
                    "product_hand_options and product_hand_options_file must not both be configured"
                )
            options_path = Path(options_file)
            if not options_path.is_absolute():
                options_path = Path(base_dir) / options_path
            with options_path.open("r", encoding="utf-8") as options_stream:
                options_config = ProductHandOptionsFile.model_validate(
                    yaml.safe_load(options_stream)
                )
            # product_name 是人工可读标签，不进入任务运行数据。
            raw_config["product_hand_options"] = {
                slot: option.hands
                for slot, option in options_config.product_hand_options.items()
            }
            raw_config["product_target_ids"] = {
                slot: option.target_id
                for slot, option in options_config.product_hand_options.items()
            }
            raw_config["product_grasp_options"] = {
                slot: option.grasp_options
                for slot, option in options_config.product_hand_options.items()
            }
        return cls.model_validate(raw_config)


class Task1Request(BaseModel):
    """任务入口请求；任务一固定完成小票上的两件商品。"""

    model_config = ConfigDict(extra="forbid")

    pass


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["STARTING", "READY", "ERROR"]


class ParseReceiptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_names: list[str]


class SkuResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    images: list[str]
    locations: list[str]


class ActionResponse(BaseModel):
    # Physical services may return execution metadata alongside the required status.
    model_config = ConfigDict(extra="ignore")

    status: Literal["SUCCEEDED"]


class TargetItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str
    product_slot_id: str
    target_id: str
    shelf_level: str
    hand: Hand
    picked: bool = False
    placed: bool = False

    @field_validator("product_slot_id")
    @classmethod
    def valid_slot(cls, value: str) -> str:
        if not PRODUCT_SLOT_PATTERN.fullmatch(value):
            raise ValueError("invalid product_slot_id")
        return value


class Task1Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_run_id: str
    task_type: Literal["SORTING"]
    status: Literal["SUCCEEDED"]
    product_names: list[str]
    target_items: list[TargetItem]
    held_items: dict[Hand, str]


class Task1ServiceError(Exception):
    """统一转换为 HTTP 错误响应的业务异常。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 502,
        step: str | None = None,
        failed_interface: str | None = None,
        url: str | None = None,
        pose: list[float] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.step = step
        self.failed_interface = failed_interface
        self.url = url
        self.pose = pose
