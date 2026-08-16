"""任务二服务的数据契约和配置模型。"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PRODUCT_SLOT_PATTERN = re.compile(r"^H[12]_[FB]_L[1-5]_C\d{2}$")


class TaskType(StrEnum):
    SHORTAGE = "SHORTAGE"


class Hand(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class InspectionPose(StrEnum):
    UPPER = "SHELF_VIEW_UPPER"
    LOWER = "SHELF_VIEW_LOWER"

    @property
    def directory_suffix(self) -> str:
        return "UPPER" if self is InspectionPose.UPPER else "LOWER"


class ProductHandOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str = Field(min_length=1)
    hands: list[Hand]
    target_id: str = Field(min_length=1)

    @field_validator("hands")
    @classmethod
    def valid_hands(cls, value: list[Hand]) -> list[Hand]:
        if not value:
            raise ValueError("hands must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("hands contains duplicate values")
        return value


class ProductHandOptionsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
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


class Task2Services(BaseModel):
    model_config = ConfigDict(extra="forbid")

    navigation: str = Field(min_length=1)
    perception: str = Field(min_length=1)
    pose: str = Field(min_length=1)
    pick_place: str = Field(min_length=1)
    camera: str = Field(min_length=1)


class Task2Timeouts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connect_seconds: float = Field(gt=0, default=3)
    health_seconds: float = Field(gt=0, default=5)
    inspection_seconds: float = Field(gt=0, default=180)
    camera_seconds: float = Field(gt=0, default=30)
    navigation_seconds: float = Field(gt=0, default=600)
    pose_seconds: float = Field(gt=0, default=300)
    pick_seconds: float = Field(gt=0, default=600)
    place_seconds: float = Field(gt=0, default=600)


class Task2Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    services: Task2Services
    timeouts: Task2Timeouts = Field(default_factory=Task2Timeouts)
    inspection_points: list[str]
    camera: Literal["head"] = "head"
    baseline_dir: str = Field(min_length=1, default="output/task0")
    replenishment_pickup: str = Field(min_length=1, default="replenishment_pickup")
    task_boundary: str = Field(min_length=1, default="task_boundary")
    start_target_id: str = Field(min_length=1, default="start")
    product_hand_options_file: str = Field(min_length=1)
    product_hand_options: dict[str, ProductHandOption] = Field(default_factory=dict)
    log_dir: str = Field(min_length=1, default="log")

    @model_validator(mode="after")
    def valid_configuration(self) -> "Task2Settings":
        if not self.inspection_points:
            raise ValueError("inspection_points must not be empty")
        if any(not point.strip() for point in self.inspection_points):
            raise ValueError("inspection_points must not contain empty values")
        if len(set(self.inspection_points)) != len(self.inspection_points):
            raise ValueError("inspection_points must not contain duplicates")
        invalid_targets = sorted(
            {
                option.target_id
                for option in self.product_hand_options.values()
                if option.target_id not in self.inspection_points
            }
        )
        if invalid_targets:
            raise ValueError(
                "product hand options contain unknown inspection targets: "
                + ", ".join(invalid_targets)
            )
        mapped_targets = {
            option.target_id for option in self.product_hand_options.values()
        }
        missing_targets = sorted(set(self.inspection_points) - mapped_targets)
        if missing_targets:
            raise ValueError(
                "inspection points have no product location mapping: "
                + ", ".join(missing_targets)
            )
        return self

    def location_id_for_target(self, target_id: str) -> str:
        locations = sorted(
            slot_id
            for slot_id, option in self.product_hand_options.items()
            if option.target_id == target_id
        )
        if not locations:
            raise ValueError(f"inspection target has no product location: {target_id}")
        return locations[0]

    @classmethod
    def load(cls, path: str | Path) -> "Task2Settings":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)

        return cls.from_mapping(raw_config, config_path.parent)

    @classmethod
    def from_mapping(
        cls, raw_config: dict[str, Any], base_dir: str | Path
    ) -> "Task2Settings":
        raw_config = dict(raw_config)

        options_path = Path(raw_config["product_hand_options_file"])
        if not options_path.is_absolute():
            options_path = Path(base_dir) / options_path
        with options_path.open("r", encoding="utf-8") as options_stream:
            options = ProductHandOptionsFile.model_validate(yaml.safe_load(options_stream))
        raw_config["product_hand_options"] = options.product_hand_options
        return cls.model_validate(raw_config)


class Task2Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pass


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["STARTING", "READY", "ERROR"]


class ShortageProductFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shortage_product_name: str


class InspectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[ShortageProductFinding]


class CameraStreamState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = Field(min_length=1)
    online: bool


class CameraState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    online: bool
    streams: list[CameraStreamState]


class CameraListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cameras: list[CameraState]


class ActionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["SUCCEEDED"]


class TargetItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str
    inspection_target_id: str
    inspection_pose_type: InspectionPose
    hand: Hand
    picked: bool = False
    placed: bool = False


class Task2Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_run_id: str
    task_type: Literal["SHORTAGE"]
    status: Literal["SUCCEEDED"]
    inspection_pass: int
    product_names: list[str]
    target_items: list[TargetItem]
    held_items: dict[Hand, str]


class Task2ServiceError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 502,
        step: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.step = step
