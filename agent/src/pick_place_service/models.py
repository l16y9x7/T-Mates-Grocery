"""取放服务的公共和内部数据模型。"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from runtime_config import RuntimeDocument, load_runtime_document


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
    level: Literal["L1", "L2", "L3", "L4", "L5"] | None = None
    product_type: str | int | None = None
    location_id: str | None = Field(default=None, min_length=1)
    pose_type: Literal["SHELF_VIEW_UPPER", "SHELF_VIEW_LOWER"] | None = None

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


def normalize_product_name(value: str) -> str:
    """去掉空格和符号后再比较商品名。

    感知定位可能返回 ``Lay's乐事薯片...``，而请求侧配置是 ``Lays乐事薯片...``。
    """

    return "".join(ch for ch in value if ch.isalnum())


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


class PlaceLocateResponse(BaseModel):
    """Reference-image inputs returned by the shelf place locator."""

    model_config = ConfigDict(extra="forbid")

    product_name: str
    bbox: list[int]
    mask: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    current_image_path: str | None = Field(default=None, min_length=1)
    rotate_matrix: list[list[float]]
    level: Literal["L1", "L2", "L3", "L4", "L5"]


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
    pick_cameras: dict[Literal["left", "right"], str]
    shortage_pick_camera: str = Field(min_length=1, default="head")
    place_camera: str = Field(min_length=1, default="head")
    # 正式配置按相机 ID 选择标定；calibration_file 仅兼容旧配置和单元测试。
    calibration_files: dict[str, str] = Field(default_factory=dict)
    calibration_file: str | None = Field(default=None, min_length=1)
    log_dir: str = Field(min_length=1, default="log")
    temp_dir: str = Field(min_length=1, default="/tmp/pick-place")
    timeouts: ServiceTimeouts = Field(default_factory=ServiceTimeouts)

    @classmethod
    def load(cls, path: str | Path) -> PickPlaceSettings:
        return cls.from_runtime_document(load_runtime_document(path))

    @classmethod
    def from_runtime_document(cls, document: RuntimeDocument) -> PickPlaceSettings:
        raw = document.section("pick_place")
        raw.update(
            {
                "perception_url": document.local_services.perception,
                "pose_estimation_url": document.local_services.pose_estimation,
                "manipulation_url": document.robot.pose_url,
                "camera_url": document.robot.camera_url,
            }
        )
        calibration_files = raw.get("calibration_files")
        if isinstance(calibration_files, dict):
            raw["calibration_files"] = {
                camera: document.resolve(str(calibration))
                for camera, calibration in calibration_files.items()
            }
        if raw.get("calibration_file"):
            raw["calibration_file"] = document.resolve(str(raw["calibration_file"]))
        if raw.get("log_dir"):
            raw["log_dir"] = document.resolve(str(raw["log_dir"]))
        return cls.model_validate(raw)

    @field_validator("pick_cameras")
    @classmethod
    def require_both_pick_cameras(
        cls, value: dict[Literal["left", "right"], str]
    ) -> dict[Literal["left", "right"], str]:
        missing = {"left", "right"} - value.keys()
        if missing:
            raise ValueError(f"pick_cameras 缺少配置: {', '.join(sorted(missing))}")
        blank = [hand for hand, camera in value.items() if not camera.strip()]
        if blank:
            raise ValueError(f"pick_cameras 相机 ID 不能为空: {', '.join(sorted(blank))}")
        return value

    def calibration_for(self, camera: str) -> str:
        """返回指定相机的标定文件，旧配置下回退到单一标定文件。"""

        calibration = self.calibration_files.get(camera) or self.calibration_file
        if not calibration:
            raise ValueError(f"未配置相机 {camera} 的标定文件")
        return calibration

    def camera_for(self, operation: str, hand: str, task_type: TaskType) -> str:
        """按任务类型、操作和手臂选择相机。"""

        if operation == "pick":
            if task_type is TaskType.SHORTAGE:
                return self.shortage_pick_camera
            return self.pick_cameras[hand.lower()]
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
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 502,
        failed_interface: str | None = None,
        url: str | None = None,
        pose: list[float] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.failed_interface = failed_interface
        self.url = url
        self.pose = pose


def action_payload(request: PickPlaceRequest, pose: PoseResponse) -> dict[str, Any]:
    """按 8084 grasp/release 契约生成执行请求。"""

    payload: dict[str, Any] = {
        "task_type": request.task_type.value,
        "product_name": request.product_name,
        "pose": pose.pose,
        "hand": request.normalized_hand,
        # 位姿服务可能返回这些元数据；缺省时使用 8084 文档约定值。
        "frame": pose.frame or "camera",
        "pose_unit": pose.pose_unit or "mm_rad",
        "rotation_order": pose.rotation_order or "zyx",
    }
    return payload
