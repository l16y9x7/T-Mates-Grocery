"""Task 3 public contracts and production configuration."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PRODUCT_SLOT_PATTERN = re.compile(r"^(?:H[12]_[FB]_L[1-5]|H[1-3]_L0[1-5])_C\d{2}$")


class TaskType(StrEnum):
    MISPLACED = "MISPLACED"


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
            value["grasp_options"] = [{"hands": value.pop("hands", None), "target_id": value.pop("target_id", None)}]
        elif isinstance(value, dict) and isinstance(value.get("grasp_options"), dict):
            value = dict(value); value["grasp_options"] = [value["grasp_options"]]
        return value

    @property
    def hands(self) -> list[Hand]:
        return list(dict.fromkeys(hand for option in self.grasp_options for hand in option.hands))

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


class InspectionPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(min_length=1)
    location_id: str = Field(min_length=1)

    @field_validator("location_id")
    @classmethod
    def valid_location(cls, value: str) -> str:
        if not PRODUCT_SLOT_PATTERN.fullmatch(value):
            raise ValueError("inspection location_id must be a product slot id")
        return value


class Task3Services(BaseModel):
    model_config = ConfigDict(extra="forbid")

    navigation: str = Field(min_length=1)
    perception: str = Field(min_length=1)
    pose: str = Field(min_length=1)
    pick_place: str = Field(min_length=1)
    sku: str = Field(min_length=1)
    camera: str = Field(min_length=1)


class Task3Timeouts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connect_seconds: float = Field(gt=0, default=3)
    health_seconds: float = Field(gt=0, default=5)
    inspection_seconds: float = Field(gt=0, default=180)
    camera_seconds: float = Field(gt=0, default=30)
    sku_seconds: float = Field(gt=0, default=10)
    navigation_seconds: float = Field(gt=0, default=600)
    pose_seconds: float = Field(gt=0, default=300)
    pick_seconds: float = Field(gt=0, default=600)
    place_seconds: float = Field(gt=0, default=600)


class Task3Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    services: Task3Services
    timeouts: Task3Timeouts = Field(default_factory=Task3Timeouts)
    inspection_points: list[InspectionPoint]
    camera: Literal["head"] = "head"
    baseline_dir: str = Field(min_length=1, default="output/task0")
    task_boundary: str = Field(min_length=1, default="task_boundary")
    start_target_id: str = Field(min_length=1, default="start")
    product_hand_options_file: str = Field(min_length=1)
    product_hand_options: dict[str, ProductHandOption] = Field(default_factory=dict)
    log_dir: str = Field(min_length=1, default="log")

    @model_validator(mode="after")
    def valid_configuration(self) -> "Task3Settings":
        if not self.inspection_points:
            raise ValueError("inspection_points must not be empty")
        target_ids = [point.target_id for point in self.inspection_points]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("inspection point target_id values must not contain duplicates")
        for point in self.inspection_points:
            option = self.product_hand_options.get(point.location_id)
            if option is None:
                raise ValueError(
                    f"inspection location is absent from product hand options: {point.location_id}"
                )
            if not any(grasp.target_id == point.target_id for grasp in option.grasp_options):
                raise ValueError(
                    f"inspection location {point.location_id} does not map to "
                    f"{point.target_id}"
                )
        return self

    @classmethod
    def load(cls, path: str | Path) -> "Task3Settings":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)

        return cls.from_mapping(raw_config, config_path.parent)

    @classmethod
    def from_mapping(
        cls, raw_config: dict[str, Any], base_dir: str | Path
    ) -> "Task3Settings":
        raw_config = dict(raw_config)

        options_path = Path(raw_config["product_hand_options_file"])
        if not options_path.is_absolute():
            options_path = Path(base_dir) / options_path
        with options_path.open("r", encoding="utf-8") as options_stream:
            options = ProductHandOptionsFile.model_validate(yaml.safe_load(options_stream))
        raw_config["product_hand_options"] = options.product_hand_options

        inspection_points = raw_config.get("inspection_points")
        if isinstance(inspection_points, list) and all(
            isinstance(target_id, str) for target_id in inspection_points
        ):
            locations_by_target: dict[str, list[str]] = {}
            for slot_id, option in options.product_hand_options.items():
                for grasp in option.grasp_options:
                    locations_by_target.setdefault(grasp.target_id, []).append(slot_id)
            missing_targets = sorted(
                target_id
                for target_id in inspection_points
                if target_id not in locations_by_target
            )
            if missing_targets:
                raise ValueError(
                    "inspection points have no product location mapping: "
                    + ", ".join(missing_targets)
                )
            raw_config["inspection_points"] = [
                {
                    "target_id": target_id,
                    "location_id": sorted(locations_by_target[target_id])[0],
                }
                for target_id in inspection_points
            ]
        return cls.model_validate(raw_config)


class Task3Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["STARTING", "READY", "ERROR"]


class ActionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["SUCCEEDED"]


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


class MisplacedFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    misplaced_product_name: str = Field(min_length=1)
    gt_product_name: str = Field(min_length=1)


class InspectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[MisplacedFinding]


class SkuResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    images: list[str]
    locations: list[str]


class FindingContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    misplaced_product_name: str
    gt_product_name: str
    inspection_target_id: str
    inspection_location_id: str
    inspection_pose_type: InspectionPose


class SwapItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_name: str
    source_slot_id: str
    destination_slot_id: str
    source_target_id: str
    destination_target_id: str
    hand: Hand
    picked: bool = False
    placed: bool = False


class Task3Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_run_id: str
    task_type: Literal["MISPLACED"]
    status: Literal["SUCCEEDED"]
    inspection_pass: int
    finding: FindingContext
    product_names: list[str]
    target_items: list[SwapItem]
    held_items: dict[Hand, str]


class Task3ServiceError(Exception):
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
