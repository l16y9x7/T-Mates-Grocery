"""任务二服务的数据契约和配置模型。"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PRODUCT_SLOT_PATTERN = re.compile(r"^(?:H[12]_[FB]_L[1-5]|H[1-3]_L0[1-5])_C\d{2}$")


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


class GraspOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

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


class ProductHandOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str = Field(min_length=1)
    grasp_options: list[GraspOption]

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_shape(cls, value: Any) -> Any:
        if isinstance(value, dict) and "grasp_options" not in value:
            value = dict(value)
            value["grasp_options"] = [
                {
                    "hands": value.pop("hands", None),
                    "target_id": value.pop("target_id", None),
                }
            ]
        elif isinstance(value, dict) and isinstance(value.get("grasp_options"), dict):
            value = dict(value)
            value["grasp_options"] = [value["grasp_options"]]
        return value

    @field_validator("grasp_options")
    @classmethod
    def valid_grasp_options(cls, value: list[GraspOption]) -> list[GraspOption]:
        if not value:
            raise ValueError("grasp_options must not be empty")
        return value

    @property
    def hands(self) -> list[Hand]:
        return list(
            dict.fromkeys(
                hand for option in self.grasp_options for hand in option.hands
            )
        )

    @property
    def target_id(self) -> str:
        return self.grasp_options[0].target_id


class ProductHandOptionsFile(BaseModel):
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
    product_hand_options_schema_version: Literal["1.0", "2.0"] = "1.0"
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
        # 巡检点与抓取点是两套能力：巡检只走正对三个货架的点位，
        # 商品抓放仍可使用 H12/H23 等连接处点位，不能要求二者完全相同。
        mapped_targets = {
            grasp.target_id
            for option in self.product_hand_options.values()
            for grasp in option.grasp_options
        }
        missing_targets = sorted(set(self.inspection_points) - mapped_targets)
        if missing_targets:
            raise ValueError(
                "inspection points have no product location mapping: "
                + ", ".join(missing_targets)
            )
        return self

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
        raw_config["product_hand_options_schema_version"] = options.schema_version
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
    slot_id: str | None = None

    @field_validator("slot_id")
    @classmethod
    def valid_optional_slot_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not PRODUCT_SLOT_PATTERN.fullmatch(normalized):
            raise ValueError("invalid shortage slot_id")
        return normalized


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
    product_slot_id: str | None = None
    inspection_target_id: str
    inspection_pose_type: InspectionPose
    hand: Hand
    picked: bool = False
    placed: bool = False

    @field_validator("product_slot_id")
    @classmethod
    def valid_optional_product_slot_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not PRODUCT_SLOT_PATTERN.fullmatch(normalized):
            raise ValueError("invalid product_slot_id")
        return normalized


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
