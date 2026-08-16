"""Test1 configuration and result contracts."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from runtime_config import load_runtime_document


SAFE_TARGET_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
SHELF_LEVELS = ("L1", "L2", "L3", "L4", "L5")


class Hand(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    @property
    def camera(self) -> str:
        return "left_wrist" if self is Hand.LEFT else "right_wrist"


class Test1Services(BaseModel):
    model_config = ConfigDict(extra="forbid")

    navigation: str = Field(min_length=1)
    pose: str = Field(min_length=1)
    camera: str = Field(min_length=1)


class Test1Timeouts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connect_seconds: float = Field(gt=0, default=3)
    health_seconds: float = Field(gt=0, default=5)
    navigation_seconds: float = Field(gt=0, default=600)
    pose_seconds: float = Field(gt=0, default=300)
    camera_seconds: float = Field(gt=0, default=30)


class Test1Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    services: Test1Services
    timeouts: Test1Timeouts = Field(default_factory=Test1Timeouts)
    inspection_points: list[str]
    start_target_id: str = Field(min_length=1, default="start")
    capture_settle_seconds: float = Field(ge=0, default=2)
    output_dir: str = Field(min_length=1, default="output/test1")
    log_dir: str = Field(min_length=1, default="log")

    @field_validator("inspection_points")
    @classmethod
    def valid_inspection_points(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("inspection_points must not be empty")
        normalized = [point.strip() for point in value]
        if any(not SAFE_TARGET_ID.fullmatch(point) for point in normalized):
            raise ValueError("inspection_points contain unsafe values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("inspection_points must not contain duplicates")
        return normalized

    @field_validator("start_target_id")
    @classmethod
    def valid_start_target_id(cls, value: str) -> str:
        value = value.strip()
        if not SAFE_TARGET_ID.fullmatch(value):
            raise ValueError("start_target_id contains unsafe characters")
        return value

    @classmethod
    def load(cls, path: str | Path) -> "Test1Settings":
        document = load_runtime_document(path)
        values = document.task_section("test1", cls)
        values["services"] = {
            "navigation": document.robot.navigation_url,
            "pose": document.robot.pose_url,
            "camera": document.robot.camera_url,
        }
        for field_name in ("output_dir", "log_dir"):
            if field_name in values:
                values[field_name] = document.resolve(str(values[field_name]))
        return cls.model_validate(values)


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


class CameraFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    rgb: bytes
    rgb_suffix: Literal[".jpg", ".png"]
    depth: bytes


class CaptureResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str
    shelf_level: Literal["L1", "L2", "L3", "L4", "L5"]
    hand: Hand
    camera: str
    directory: str
    rgb_path: str
    depth_path: str


class Test1Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_run_id: str
    task_type: Literal["TEST1"] = "TEST1"
    status: Literal["SUCCEEDED"] = "SUCCEEDED"
    batch_directory: str
    captures: list[CaptureResult]


class Test1ServiceError(Exception):
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
