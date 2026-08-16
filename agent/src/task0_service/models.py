"""Task 0 public contracts and production configuration."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SAFE_TARGET_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class InspectionPose(StrEnum):
    UPPER = "SHELF_VIEW_UPPER"
    LOWER = "SHELF_VIEW_LOWER"

    @property
    def directory_suffix(self) -> str:
        return "UPPER" if self is InspectionPose.UPPER else "LOWER"


class Task0Services(BaseModel):
    model_config = ConfigDict(extra="forbid")

    navigation: str = Field(min_length=1)
    pose: str = Field(min_length=1)
    camera: str = Field(min_length=1)


class Task0Timeouts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connect_seconds: float = Field(gt=0, default=3)
    health_seconds: float = Field(gt=0, default=5)
    navigation_seconds: float = Field(gt=0, default=600)
    pose_seconds: float = Field(gt=0, default=300)
    camera_seconds: float = Field(gt=0, default=30)


class Task0Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    services: Task0Services
    timeouts: Task0Timeouts = Field(default_factory=Task0Timeouts)
    inspection_points: list[str]
    start_target_id: str = Field(min_length=1, default="start")
    capture_settle_seconds: float = Field(ge=0, default=2)
    camera: str = Field(min_length=1, default="head")
    output_dir: str = Field(min_length=1, default="output/task0")
    log_dir: str = Field(min_length=1, default="log")

    @field_validator("inspection_points")
    @classmethod
    def valid_inspection_points(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("inspection_points must not be empty")
        normalized = [point.strip() for point in value]
        if any(not point for point in normalized):
            raise ValueError("inspection_points must not contain empty values")
        invalid = [
            point for point in normalized if not SAFE_TARGET_ID.fullmatch(point)
        ]
        if invalid:
            raise ValueError(
                "inspection_points contain values unsafe for directory names: "
                + ", ".join(invalid)
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("inspection_points must not contain duplicates")
        return normalized

    @field_validator("start_target_id")
    @classmethod
    def valid_start_target_id(cls, value: str) -> str:
        value = value.strip()
        if not SAFE_TARGET_ID.fullmatch(value):
            raise ValueError("start_target_id contains unsafe characters")
        return value

    @model_validator(mode="after")
    def require_head_camera(self) -> "Task0Settings":
        if self.camera.strip() != "head":
            raise ValueError("task0 camera must be head")
        self.camera = "head"
        return self

    @classmethod
    def load(cls, path: str | Path) -> "Task0Settings":
        with Path(path).open("r", encoding="utf-8") as config_file:
            return cls.from_mapping(yaml.safe_load(config_file))

    @classmethod
    def from_mapping(cls, raw_config: dict[str, Any]) -> "Task0Settings":
        return cls.model_validate(raw_config)


class Task0Request(BaseModel):
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


class CaptureResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str
    pose_type: InspectionPose
    directory: str
    rgb_path: str
    depth_path: str
    meta_path: str


class Task0Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_run_id: str
    task_type: Literal["PREPARATION"]
    status: Literal["SUCCEEDED"]
    inspection_points: list[str]
    captures: list[CaptureResult]


class Task0ServiceError(Exception):
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
