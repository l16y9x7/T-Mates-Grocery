"""取放服务的公共和内部数据模型。"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskType(StrEnum):
    SORTING = "SORTING"
    SHORTAGE = "SHORTAGE"
    MISPLACED = "MISPLACED"


class PickPlaceRequest(BaseModel):
    """8086 对外的取放请求。"""

    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    product_name: str = Field(min_length=1)
    hand: Literal["left", "right", "LEFT", "RIGHT"]
    product_type: str | int | None = None

    @field_validator("product_name")
    @classmethod
    def product_name_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("product_name must not be blank")
        return value

    @property
    def normalized_hand(self) -> str:
        return self.hand.lower()


class StatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["SUCCEEDED"]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["STARTING", "READY", "ERROR"]


class LocateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_name: str
    bbox: list[int | float]
    mask: str | None = None
    image_path: str | None = None


class PoseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pose: list[float]
    corners_mm: list[list[float]] | None = None
    frame: str | None = None
    pose_unit: str | None = None
    rotation_order: str | None = None


class ServiceTimeouts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    connect_seconds: float = Field(gt=0, default=3)
    locate_seconds: float = Field(gt=0, default=120)
    camera_seconds: float = Field(gt=0, default=15)
    pose_seconds: float = Field(gt=0, default=300)
    action_seconds: float = Field(gt=0, default=600)
    check_seconds: float = Field(gt=0, default=120)
    health_seconds: float = Field(gt=0, default=5)


class PickPlaceSettings(BaseModel):
    """8086 独立服务配置。"""

    model_config = ConfigDict(extra="forbid")
    perception_url: str = Field(min_length=1)
    locate_url: str | None = Field(default=None, min_length=1)
    pose_estimation_url: str | None = Field(default=None, min_length=1)
    manipulation_url: str = Field(min_length=1)
    camera_url: str = Field(min_length=1)
    pick_camera: str = Field(min_length=1, default="head")
    pick_cameras: dict[Literal["left", "right"], str] = Field(default_factory=dict)
    place_camera: str = Field(min_length=1, default="head")
    # 正式配置按相机 ID 选择标定；calibration_file 仅兼容旧配置和单元测试。
    calibration_files: dict[str, str] = Field(default_factory=dict)
    calibration_file: str | None = Field(default=None, min_length=1)
    log_dir: str = Field(min_length=1, default="log")
    temp_dir: str = Field(min_length=1, default="/tmp/pick-place")
    timeouts: ServiceTimeouts = Field(default_factory=ServiceTimeouts)

    @classmethod
    def load(cls, path: str | Path) -> PickPlaceSettings:
        with Path(path).open("r", encoding="utf-8") as config_file:
            return cls.model_validate(yaml.safe_load(config_file))

    def calibration_for(self, camera: str) -> str:
        """返回指定相机的标定文件，旧配置下回退到单一标定文件。"""

        calibration = self.calibration_files.get(camera) or self.calibration_file
        if not calibration:
            raise ValueError(f"未配置相机 {camera} 的标定文件")
        return calibration

    def camera_for(self, operation: str, hand: str) -> str:
        """按操作和手臂选择相机；旧配置继续回退到单一 pick_camera。"""

        if operation == "pick":
            return self.pick_cameras.get(hand.lower(), self.pick_camera)
        return self.place_camera


class FrameBundle(BaseModel):
    """位姿估计所需的文件引用。"""

    model_config = ConfigDict(extra="forbid")
    rgb: str
    depth: str
    camera: str
    mask: str
    cleanup_path: str | None = None


class ServiceError(Exception):
    def __init__(self, code: str, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def action_payload(request: PickPlaceRequest, pose: PoseResponse) -> dict[str, Any]:
    """按 8084 grasp/release 契约生成执行请求。"""

    payload: dict[str, Any] = {
        "task_type": request.task_type.value,
        "pose": pose.pose,
        "hand": request.normalized_hand,
        # 位姿服务可能返回这些元数据；缺省时使用 8084 文档约定值。
        "frame": pose.frame or "camera",
        "pose_unit": pose.pose_unit or "mm_rad",
        "rotation_order": pose.rotation_order or "zyx",
    }
    if request.product_type is not None:
        payload["product_type"] = request.product_type
    return payload
