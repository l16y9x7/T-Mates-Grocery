"""任务一服务的数据契约和配置模型。"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


PRODUCT_SLOT_PATTERN = re.compile(r"^H[12]_[FB]_L[1-5]_C\d{2}$")


class TaskType(StrEnum):
    SORTING = "SORTING"


class Hand(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class Task1Services(BaseModel):
    model_config = ConfigDict(extra="forbid")

    navigation: str = Field(min_length=1)
    perception: str = Field(min_length=1)
    pose: str = Field(min_length=1)
    pick_place: str = Field(min_length=1)
    sku: str = Field(min_length=1)


class Task1Timeouts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connect_seconds: float = Field(gt=0, default=3)
    health_seconds: float = Field(gt=0, default=5)
    receipt_seconds: float = Field(gt=0, default=120)
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
    product_hand_options: dict[str, list[Hand]] = Field(default_factory=dict)
    log_dir: str = Field(min_length=1, default="log")

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

    @classmethod
    def load(cls, path: str | Path) -> "Task1Settings":
        with Path(path).open("r", encoding="utf-8") as config_file:
            return cls.model_validate(yaml.safe_load(config_file))


class Task1Request(BaseModel):
    """任务入口请求；任务一固定完成小票上的两件商品。"""

    model_config = ConfigDict(extra="forbid")

    pass


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    status: Literal["SUCCEEDED"]


class TargetItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str
    product_slot_id: str
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

    def __init__(self, code: str, message: str, *, status_code: int = 502, step: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.step = step
