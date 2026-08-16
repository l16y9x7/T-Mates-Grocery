"""Prepare Task0 product inputs and reference-to-current camera registration.

Place Locate does not estimate an object pose.  It recovers the actual product
mask in the fixed Task0 image and estimates the 4x4 ``current_from_reference``
camera transform.  A downstream pose estimator consumes those values, estimates
the object pose in the Task0 camera frame, and applies the supplied transform.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

if __package__ and __package__.startswith("perception."):
    from ...camera_capture import (
        CameraCaptureError,
        capture_head_rgbd,
        inspection_temporary_directory,
    )
    from ...initial_scan import InitialScanError, load_initial_scan
    from ...row_detection import (
        PoseType,
        RowDetectionConfig,
        ShelfRow,
        detect_rows,
    )
else:
    from camera_capture import (
        CameraCaptureError,
        capture_head_rgbd,
        inspection_temporary_directory,
    )
    from initial_scan import InitialScanError, load_initial_scan
    from row_detection import (
        PoseType,
        RowDetectionConfig,
        ShelfRow,
        detect_rows,
    )

from .pose_transfer import (
    PoseTransferError,
    as_rigid_transform,
)
from .registration import (
    CameraIntrinsics,
    RGBDRegistrationResult,
    register_rgbd_images,
)
from .reference_mask import (
    ReferenceMaskError,
    ReferenceMaskResult,
    generate_reference_mask,
)


TaskType = Literal["SHORTAGE", "MISPLACED"]
ShelfLevel = Literal["L1", "L2", "L3", "L4", "L5"]
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_DEPTH_BYTES = 40 * 1024 * 1024
DEFAULT_CURRENT_IMAGE_NAME = "current_rgb.jpg"
HEAD_CAMERA_FRAME_ID = "head_color_optical_frame"
HEAD_CAMERA_CALIBRATION_WIDTH = 1280
HEAD_CAMERA_CALIBRATION_HEIGHT = 720
HEAD_CAMERA_FX = 910.744324
HEAD_CAMERA_FY = 910.395020
HEAD_CAMERA_CX = 650.132690
HEAD_CAMERA_CY = 381.874634
HEAD_CAMERA_DISTORTION_MODEL = "plumb_bob"
HEAD_CAMERA_DISTORTION = (0.0, 0.0, 0.0, 0.0, 0.0)
MAX_REGISTRATION_RMSE_MM = 20.0
MIN_REGISTRATION_INLIER_RATIO = 0.5
MAX_RGB_REPROJECTION_RMSE_PX = 4.0
DEFAULT_ARTIFACT_ROOT = Path(
    os.getenv(
        "PLACE_LOCATE_DEBUG_DIR",
        str(Path(__file__).resolve().parent / "debug"),
    )
)

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Place Locate",
    version="2.0.0",
    description=(
        "Return a Task0 product bbox/mask and the reference-to-current 4x4 "
        "camera transform for downstream pose estimation."
    ),
)
router = APIRouter()


class PlaceReferenceMaskRequest(BaseModel):
    """Inputs needed to recover a product mask in the fixed Task0 image."""

    model_config = ConfigDict(extra="forbid")

    task_type: TaskType = "SHORTAGE"
    product_name: str = Field(min_length=1)
    location_id: str = Field(min_length=1)
    current_image_base64: str = Field(min_length=1)
    current_depth_image_base64: str = Field(min_length=1)
    pose_type: PoseType = ""
    current_image_name: str = DEFAULT_CURRENT_IMAGE_NAME
    depth_is_bigendian: bool = False
    depth_unit_mm: float = Field(default=1.0, gt=0)
    region_index: int = Field(default=1, ge=1)
    reference_bbox: list[int] | None = None

    @field_validator("product_name", "location_id")
    @classmethod
    def normalize_nonempty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("current_image_name")
    @classmethod
    def validate_image_name(cls, value: str) -> str:
        normalized = value.strip() or DEFAULT_CURRENT_IMAGE_NAME
        if Path(normalized).name != normalized:
            raise ValueError("current_image_name must not contain a path")
        return normalized

    @field_validator("reference_bbox")
    @classmethod
    def validate_reference_bbox(
        cls,
        value: list[int] | None,
    ) -> list[int] | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("reference_bbox must be [x, y, width, height]")
        if value[2] <= 0 or value[3] <= 0:
            raise ValueError("reference_bbox width and height must be positive")
        return value


class PlaceLocateRequest(BaseModel):
    """A place target whose current RGB-D input is captured at runtime."""

    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    location_id: str = Field(min_length=1)
    pose_type: PoseType
    reference_item_area: float | None = Field(default=None, gt=0)
    product_name: str = Field(min_length=1)

    @field_validator("product_name", "location_id")
    @classmethod
    def normalize_nonempty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


class PlaceLocateDebugRequest(PlaceReferenceMaskRequest):
    """Explicit RGB-D inputs retained for offline tests and diagnostics."""


class PlaceLocateResponse(BaseModel):
    """Inputs consumed by downstream reference-image pose estimation."""

    product_name: str
    bbox: list[int]
    mask: str
    image_path: str
    rotate_matrix: list[list[float]]
    level: ShelfLevel

    @field_validator("rotate_matrix")
    @classmethod
    def validate_rotate_matrix(
        cls,
        value: list[list[float]],
    ) -> list[list[float]]:
        try:
            return as_rigid_transform(
                value,
                name="rotate_matrix",
            ).tolist()
        except PoseTransferError as error:
            raise ValueError(str(error)) from error


class RegistrationMetrics(BaseModel):
    current_from_reference: list[list[float]]
    rmse_mm: float
    depth_correspondence_count: int
    inlier_count: int
    inlier_ratio: float
    reprojection_rmse_px: float


class PlaceReferenceMaskDebugResponse(BaseModel):
    """Reference-mask pipeline output before pose transfer and reprojection."""

    task_type: TaskType
    product_name: str
    location_id: str
    inspection_target_id: str
    pose_type: PoseType
    baseline_path: str
    region_index: int
    reference_image_size: list[int]
    current_image_size: list[int]
    reference_image_base64: str
    change_mask_reference: str
    reference_component_mask: str
    reference_bbox: list[int]
    row_index: int | None = None
    row_bbox: list[int] | None = None
    reference_mask: str
    reference_mask_source: Literal["sam3", "depth_change"]
    reference_sam3_prompt: str | None = None
    reference_sam3_crop_box: list[int] | None = None
    reference_sam3_bbox: list[float] | None = None
    reference_sam3_score: float | None = None
    reference_sam3_candidate_count: int | None = None
    registration: RegistrationMetrics


class PlaceLocateDebugResponse(PlaceLocateResponse):
    task_type: TaskType
    location_id: str
    inspection_target_id: str
    baseline_path: str
    region_index: int
    image_size: list[int]
    current_image_size: list[int]
    reference_bbox: list[int]
    row_index: int | None = None
    row_bbox: list[int] | None = None
    reference_mask: str
    reference_mask_source: Literal["sam3", "depth_change"] = "depth_change"
    reference_sam3_prompt: str | None = None
    reference_sam3_crop_box: list[int] | None = None
    reference_sam3_bbox: list[float] | None = None
    reference_sam3_score: float | None = None
    reference_sam3_candidate_count: int | None = None
    registration: RegistrationMetrics


def _decode_base64(value: str, field_name: str, maximum_bytes: int) -> bytes:
    encoded = value.strip().split(",", 1)[-1]
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is not valid base64",
        ) from error
    if not payload:
        raise HTTPException(status_code=400, detail=f"{field_name} must not be empty")
    if len(payload) > maximum_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} exceeds {maximum_bytes} decoded bytes",
        )
    return payload


def decode_color_image(value: str, field_name: str) -> np.ndarray:
    payload = _decode_base64(value, field_name, MAX_IMAGE_BYTES)
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail=f"{field_name} is not a JPG/PNG image")
    return image


def decode_depth_image(
    value: str,
    field_name: str,
    expected_shape: tuple[int, int],
    *,
    is_bigendian: bool = False,
    depth_unit_mm: float = 1.0,
) -> np.ndarray:
    """Decode NPY, 16-bit PNG/TIFF, or headerless 16UC1 depth into mm."""

    payload = _decode_base64(value, field_name, MAX_DEPTH_BYTES)
    depth: np.ndarray | None = None
    if payload.startswith(b"\x93NUMPY"):
        try:
            loaded = np.load(io.BytesIO(payload), allow_pickle=False)
        except (OSError, ValueError, TypeError) as error:
            raise HTTPException(
                status_code=400,
                detail=f"{field_name} contains an invalid NPY array: {error}",
            ) from error
        if isinstance(loaded, np.ndarray):
            depth = loaded
    else:
        depth = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if depth is None:
            expected_bytes = expected_shape[0] * expected_shape[1] * 2
            if len(payload) == expected_bytes:
                byte_order = ">u2" if is_bigendian else "<u2"
                depth = np.frombuffer(payload, dtype=byte_order).reshape(expected_shape)

    if depth is None or depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a two-dimensional numeric depth image",
        )
    if depth.shape != expected_shape:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{field_name} must be aligned to its RGB image: "
                f"expected={expected_shape}, actual={depth.shape}"
            ),
        )
    depth_mm = depth.astype(np.float32) * float(depth_unit_mm)
    if not np.isfinite(depth_mm).all() or np.any(depth_mm < 0):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} contains invalid depth values",
        )
    return depth_mm


def head_camera_intrinsics(image_width: int, image_height: int) -> CameraIntrinsics:
    """Scale the fixed 1280x720 head-camera calibration to an RGB resolution."""

    scale_x = image_width / HEAD_CAMERA_CALIBRATION_WIDTH
    scale_y = image_height / HEAD_CAMERA_CALIBRATION_HEIGHT
    return CameraIntrinsics(
        fx=HEAD_CAMERA_FX * scale_x,
        fy=HEAD_CAMERA_FY * scale_y,
        cx=HEAD_CAMERA_CX * scale_x,
        cy=HEAD_CAMERA_CY * scale_y,
        width=image_width,
        height=image_height,
    )


def resolve_intrinsics(
    reference_image: np.ndarray,
    current_image: np.ndarray,
) -> tuple[CameraIntrinsics, CameraIntrinsics]:
    reference_height, reference_width = reference_image.shape[:2]
    current_height, current_width = current_image.shape[:2]
    return (
        head_camera_intrinsics(reference_width, reference_height),
        head_camera_intrinsics(current_width, current_height),
    )


def _bbox_iou_xywh(first: Sequence[int], second: Sequence[int]) -> float:
    first_x1, first_y1, first_width, first_height = first
    second_x1, second_y1, second_width, second_height = second
    first_x2, first_y2 = first_x1 + first_width, first_y1 + first_height
    second_x2, second_y2 = second_x1 + second_width, second_y1 + second_height
    intersection_width = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_width * intersection_height
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union > 0 else 0.0


def extract_change_regions(change_mask: np.ndarray) -> list[tuple[list[int], np.ndarray]]:
    """Return stable top-to-bottom/left-to-right non-border change regions."""

    mask = np.where(np.asarray(change_mask) > 0, 255, 0).astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError("change_mask must be two-dimensional")
    height, width = mask.shape
    border = max(2, round(min(height, width) * 0.02))
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_area = height * width
    minimum_area = max(64, round(image_area * 0.0005))
    maximum_area = round(image_area * 0.35)
    regions: list[tuple[list[int], np.ndarray]] = []
    for label in range(1, count):
        x, y, region_width, region_height, area = stats[label].tolist()
        if area < minimum_area or area > maximum_area:
            continue
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        regions.append(([x, y, region_width, region_height], component))
    regions.sort(key=lambda item: (item[0][1], item[0][0]))
    return regions


def select_reference_region(
    change_mask: np.ndarray,
    *,
    region_index: int,
    reference_bbox: Sequence[int] | None,
) -> tuple[list[int], np.ndarray, int]:
    regions = extract_change_regions(change_mask)
    if not regions:
        raise HTTPException(status_code=404, detail="no reliable changed product region found")
    if reference_bbox is not None:
        best_index, selected = max(
            enumerate(regions),
            key=lambda item: _bbox_iou_xywh(item[1][0], reference_bbox),
        )
        if _bbox_iou_xywh(selected[0], reference_bbox) <= 0:
            raise HTTPException(
                status_code=404,
                detail="reference_bbox does not overlap a reliable changed region",
            )
        return selected[0], selected[1], best_index + 1
    if region_index > len(regions):
        raise HTTPException(
            status_code=404,
            detail=f"region_index={region_index} exceeds detected region count={len(regions)}",
        )
    bbox, component = regions[region_index - 1]
    return bbox, component, region_index


def refine_reference_mask_with_depth(
    component_mask: np.ndarray,
    reference_depth_mm: np.ndarray,
    bbox: Sequence[int],
) -> np.ndarray:
    """Expand a photometric change component over its same-depth object surface."""

    depth = np.asarray(reference_depth_mm, dtype=np.float32)
    component = np.asarray(component_mask) > 0
    valid_seed = component & np.isfinite(depth) & (depth > 0)
    if int(valid_seed.sum()) < 20:
        raise PoseTransferError("changed reference region has too few valid depth pixels")
    seed_depths = depth[valid_seed]
    median_depth = float(np.median(seed_depths))
    median_deviation = float(np.median(np.abs(seed_depths - median_depth)))
    tolerance_mm = min(180.0, max(40.0, median_deviation * 4.0))

    x, y, width, height = [int(value) for value in bbox]
    padding_x = max(2, round(width * 0.08))
    padding_y = max(2, round(height * 0.08))
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(depth.shape[1], x + width + padding_x)
    y2 = min(depth.shape[0], y + height + padding_y)
    roi = np.zeros_like(component, dtype=bool)
    roi[y1:y2, x1:x2] = True
    depth_gate = (
        roi
        & np.isfinite(depth)
        & (depth > 0)
        & (np.abs(depth - median_depth) <= tolerance_mm)
    )
    nearby = cv2.dilate(
        component.astype(np.uint8),
        np.ones((11, 11), dtype=np.uint8),
        iterations=2,
    ) > 0
    candidate = np.where(depth_gate & nearby, 255, 0).astype(np.uint8)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
    )
    count, labels, _, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if count <= 1:
        return np.where(component, 255, 0).astype(np.uint8)
    best_label = max(
        range(1, count),
        key=lambda label: int(np.count_nonzero((labels == label) & component)),
    )
    refined = np.where(labels == best_label, 255, 0).astype(np.uint8)
    if np.count_nonzero(refined & component.astype(np.uint8)) == 0:
        return np.where(component, 255, 0).astype(np.uint8)
    return refined


def constrain_mask_to_shelf_row(
    component_mask: np.ndarray,
    row: ShelfRow | None,
) -> np.ndarray:
    """Remove cross-row changes while preserving a no-row-detection fallback."""

    component = np.where(np.asarray(component_mask) > 0, 255, 0).astype(np.uint8)
    if row is None:
        return component
    x, y, width, height = row.bbox
    allowed = np.zeros_like(component)
    allowed[y : y + height, x : x + width] = 255
    constrained = cv2.bitwise_and(component, allowed)
    return constrained if np.count_nonzero(constrained) else component


def encode_png_base64(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", np.asarray(image))
    if not success:
        raise RuntimeError("failed to encode PNG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def encode_jpeg_base64(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(
        ".jpg",
        np.asarray(image),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    if not success:
        raise RuntimeError("failed to encode JPEG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def mask_bbox_xyxy(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(np.asarray(mask) > 0)
    if not len(xs):
        raise PoseTransferError("reference product mask is empty")
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def create_place_locate_artifact_directory(
    request: PlaceLocateDebugRequest,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    """Create one inspect-style directory for a formal Place Locate request."""

    root = Path(artifact_root) if artifact_root is not None else DEFAULT_ARTIFACT_ROOT
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    safe_location = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.location_id.strip())
    safe_product = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.product_name.strip())
    directory = root / (
        f"{timestamp}_{safe_location}_{request.task_type}_{safe_product}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _write_artifact_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_artifact_image(path: Path, image: np.ndarray) -> None:
    parameters = (
        [cv2.IMWRITE_JPEG_QUALITY, 95]
        if path.suffix.lower() in {".jpg", ".jpeg"}
        else []
    )
    success, encoded = cv2.imencode(path.suffix, np.asarray(image), parameters)
    if not success:
        raise OSError(f"cannot encode Place Locate artifact image: {path}")
    encoded.tofile(path)


def _write_artifact_depth(path: Path, depth_mm: np.ndarray) -> None:
    depth = np.asarray(depth_mm)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
        raise OSError(f"Place Locate artifact depth is invalid: {path}")
    np.save(path, depth, allow_pickle=False)


def _clamped_crop_box(
    box: Sequence[int | float],
    image_shape: Sequence[int],
) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise OSError("Place Locate bbox crop must contain four values")
    height, width = int(image_shape[0]), int(image_shape[1])
    x1 = max(0, min(width, int(round(float(box[0])))))
    y1 = max(0, min(height, int(round(float(box[1])))))
    x2 = max(0, min(width, int(round(float(box[2])))))
    y2 = max(0, min(height, int(round(float(box[3])))))
    if x2 <= x1 or y2 <= y1:
        raise OSError(f"Place Locate bbox crop is empty: {[x1, y1, x2, y2]}")
    return x1, y1, x2, y2


def save_place_locate_artifacts(
    directory: Path,
    *,
    request_payload: dict[str, Any],
    response: PlaceLocateDebugResponse,
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    prepared: PreparedReferenceMask,
) -> None:
    """Persist the formal response and every RGB-D/mask input needed to audit it."""

    directory.mkdir(parents=True, exist_ok=True)
    baseline_rgb_name = "baseline_rgb.jpg"
    baseline_depth_name = "baseline_depth_mm.npy"
    current_rgb_name = "current_rgb.jpg"
    current_depth_name = "current_depth_mm.npy"
    change_mask_name = "change_mask_reference.png"
    component_mask_name = "reference_component_mask.png"
    mask_name = "sam3_mask.png" if prepared.sam3 is not None else "reference_mask.png"
    bbox_crop_name = "bbox_crop.jpg"
    sam3_crop_name = "sam3_crop.jpg" if prepared.sam3 is not None else None

    crop_box = _clamped_crop_box(response.bbox, reference_image.shape)
    x1, y1, x2, y2 = crop_box
    sam3_crop_box = (
        _clamped_crop_box(prepared.sam3.crop_box, reference_image.shape)
        if prepared.sam3 is not None
        else None
    )

    _write_artifact_json(directory / "request.json", request_payload)
    _write_artifact_image(directory / baseline_rgb_name, reference_image)
    _write_artifact_depth(directory / baseline_depth_name, reference_depth_mm)
    _write_artifact_image(directory / current_rgb_name, current_image)
    _write_artifact_depth(directory / current_depth_name, current_depth_mm)
    _write_artifact_image(
        directory / change_mask_name,
        prepared.registration.rgb.change_mask_reference,
    )
    _write_artifact_image(directory / component_mask_name, prepared.component_mask)
    _write_artifact_image(directory / mask_name, prepared.reference_mask)
    _write_artifact_image(
        directory / bbox_crop_name,
        np.asarray(reference_image)[y1:y2, x1:x2],
    )
    if sam3_crop_name is not None and sam3_crop_box is not None:
        sam_x1, sam_y1, sam_x2, sam_y2 = sam3_crop_box
        _write_artifact_image(
            directory / sam3_crop_name,
            np.asarray(reference_image)[sam_y1:sam_y2, sam_x1:sam_x2],
        )

    rgbd_manifest = {
        "coordinate_system": "Task0 reference RGB",
        "baseline": {
            "rgb": baseline_rgb_name,
            "depth": baseline_depth_name,
            "width": int(reference_image.shape[1]),
            "height": int(reference_image.shape[0]),
            "depth_dtype": str(np.asarray(reference_depth_mm).dtype),
            "valid_depth_pixels": int(
                np.count_nonzero(
                    np.isfinite(reference_depth_mm) & (reference_depth_mm > 0)
                )
            ),
        },
        "current": {
            "rgb": current_rgb_name,
            "depth": current_depth_name,
            "width": int(current_image.shape[1]),
            "height": int(current_image.shape[0]),
            "depth_dtype": str(np.asarray(current_depth_mm).dtype),
            "valid_depth_pixels": int(
                np.count_nonzero(
                    np.isfinite(current_depth_mm) & (current_depth_mm > 0)
                )
            ),
        },
    }
    _write_artifact_json(directory / "rgbd.json", rgbd_manifest)

    artifacts = {
        "baseline_rgb": baseline_rgb_name,
        "baseline_depth_mm": baseline_depth_name,
        "current_rgb": current_rgb_name,
        "current_depth_mm": current_depth_name,
        "change_mask_reference": change_mask_name,
        "reference_component_mask": component_mask_name,
        "reference_mask": mask_name,
        "sam3_mask": mask_name if prepared.sam3 is not None else None,
        "bbox_crop": bbox_crop_name,
        "bbox_crop_box": list(crop_box),
        "bbox_crop_format": ["x1", "y1", "x2", "y2"],
        "bbox_crop_source": "bbox",
        "sam3_crop": sam3_crop_name,
        "sam3_crop_box": list(sam3_crop_box) if sam3_crop_box is not None else None,
        "mask_coordinate_system": "baseline_rgb",
    }
    saved_result = response.model_dump(mode="json")
    saved_result["artifacts"] = artifacts
    _write_artifact_json(directory / "result.json", saved_result)
    logger.info(
        "Place Locate artifacts saved: directory=%s product_name=%s level=%s",
        directory,
        response.product_name,
        response.level,
    )


def validate_registration_quality(registration: RGBDRegistrationResult) -> None:
    """Reject a geometrically weak transform before it can produce a robot mask."""

    failures: list[str] = []
    if registration.rmse_mm > MAX_REGISTRATION_RMSE_MM:
        failures.append(
            f"3D RMSE {registration.rmse_mm:.2f} mm > {MAX_REGISTRATION_RMSE_MM:.2f} mm"
        )
    if registration.inlier_ratio < MIN_REGISTRATION_INLIER_RATIO:
        failures.append(
            "3D inlier ratio "
            f"{registration.inlier_ratio:.3f} < {MIN_REGISTRATION_INLIER_RATIO:.3f}"
        )
    if registration.rgb.reprojection_rmse_px > MAX_RGB_REPROJECTION_RMSE_PX:
        failures.append(
            "RGB reprojection RMSE "
            f"{registration.rgb.reprojection_rmse_px:.2f} px > "
            f"{MAX_RGB_REPROJECTION_RMSE_PX:.2f} px"
        )
    if failures:
        raise PoseTransferError("registration quality check failed: " + "; ".join(failures))


def registration_metrics(registration: RGBDRegistrationResult) -> RegistrationMetrics:
    return RegistrationMetrics(
        current_from_reference=registration.current_from_reference.tolist(),
        rmse_mm=registration.rmse_mm,
        depth_correspondence_count=registration.depth_correspondence_count,
        inlier_count=registration.inlier_count,
        inlier_ratio=registration.inlier_ratio,
        reprojection_rmse_px=registration.rgb.reprojection_rmse_px,
    )


@dataclass(frozen=True)
class PreparedReferenceMask:
    registration: RGBDRegistrationResult
    reference_bbox: list[int]
    component_mask: np.ndarray
    selected_region_index: int
    matched_row: ShelfRow | None
    reference_mask: np.ndarray
    sam3: ReferenceMaskResult | None


def resolve_shelf_level(
    pose_type: PoseType,
    row: ShelfRow | None,
    location_id: str,
) -> ShelfLevel:
    """Map the detected row in an UPPER/LOWER view to physical level L1-L5."""

    location_match = re.search(r"(?:^|_)L([1-5])(?:_|$)", location_id.upper())
    if location_match is not None:
        level_number = int(location_match.group(1))
        if (
            pose_type == "SHELF_VIEW_UPPER"
            and level_number in {1, 2}
        ) or (
            pose_type == "SHELF_VIEW_LOWER"
            and level_number in {3, 4, 5}
        ):
            return cast(ShelfLevel, f"L{level_number}")

    if row is not None:
        if pose_type == "SHELF_VIEW_UPPER" and row.index in {1, 2}:
            return cast(ShelfLevel, f"L{row.index}")
        if pose_type == "SHELF_VIEW_LOWER" and row.index in {1, 2, 3}:
            return cast(ShelfLevel, f"L{row.index + 2}")

    raise PoseTransferError(
        "cannot determine physical shelf level from pose_type and detected row"
    )


def prepare_reference_mask(
    request: PlaceReferenceMaskRequest,
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    reference_intrinsics: CameraIntrinsics,
    current_intrinsics: CameraIntrinsics,
) -> PreparedReferenceMask:
    """Run the shared pipeline through reference-mask generation only."""

    registration = register_rgbd_images(
        reference_image,
        current_image,
        reference_depth_mm,
        current_depth_mm,
        reference_intrinsics,
        current_intrinsics,
    )
    validate_registration_quality(registration)
    reference_bbox, component, selected_index = select_reference_region(
        registration.rgb.change_mask_reference,
        region_index=request.region_index,
        reference_bbox=request.reference_bbox,
    )
    row_detection = detect_rows(
        reference_image,
        RowDetectionConfig(target_size=None, pose_type=request.pose_type),
    )
    matched_row = row_detection.row_for_bbox(reference_bbox)
    component = constrain_mask_to_shelf_row(component, matched_row)
    reference_sam3: ReferenceMaskResult | None = None
    if request.task_type == "SHORTAGE":
        reference_sam3 = generate_reference_mask(
            reference_image,
            reference_bbox,
            request.product_name,
            component_mask=component,
            row_bbox=(matched_row.bbox if matched_row is not None else None),
        )
        reference_mask = reference_sam3.mask
    else:
        reference_mask = refine_reference_mask_with_depth(
            component,
            reference_depth_mm,
            reference_bbox,
        )
    return PreparedReferenceMask(
        registration=registration,
        reference_bbox=reference_bbox,
        component_mask=component,
        selected_region_index=selected_index,
        matched_row=matched_row,
        reference_mask=reference_mask,
        sam3=reference_sam3,
    )


def build_debug_response(
    request: PlaceLocateDebugRequest,
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    reference_intrinsics: CameraIntrinsics,
    current_intrinsics: CameraIntrinsics,
    *,
    inspection_target_id: str,
    baseline_path: str,
    artifact_directory: Path | None = None,
    artifact_request: dict[str, Any] | None = None,
) -> PlaceLocateDebugResponse:
    prepared = prepare_reference_mask(
        request,
        reference_image,
        current_image,
        reference_depth_mm,
        current_depth_mm,
        reference_intrinsics,
        current_intrinsics,
    )
    registration = prepared.registration
    reference_bbox = prepared.reference_bbox
    matched_row = prepared.matched_row
    reference_mask = prepared.reference_mask
    reference_sam3 = prepared.sam3
    reference_product_bbox = mask_bbox_xyxy(reference_mask)
    level = resolve_shelf_level(
        request.pose_type,
        matched_row,
        request.location_id,
    )
    reference_height, reference_width = reference_image.shape[:2]
    current_height, current_width = current_image.shape[:2]
    response = PlaceLocateDebugResponse(
        product_name=request.product_name,
        bbox=reference_product_bbox,
        mask=encode_png_base64(reference_mask),
        image_path=baseline_path,
        rotate_matrix=registration.current_from_reference.tolist(),
        level=level,
        task_type=request.task_type,
        location_id=request.location_id,
        inspection_target_id=inspection_target_id,
        baseline_path=baseline_path,
        region_index=prepared.selected_region_index,
        image_size=[reference_width, reference_height],
        current_image_size=[current_width, current_height],
        reference_bbox=reference_bbox,
        row_index=matched_row.index if matched_row is not None else None,
        row_bbox=list(matched_row.bbox) if matched_row is not None else None,
        reference_mask=encode_png_base64(reference_mask),
        reference_mask_source="sam3" if reference_sam3 is not None else "depth_change",
        reference_sam3_prompt=(
            reference_sam3.sam_prompt if reference_sam3 is not None else None
        ),
        reference_sam3_crop_box=(
            list(reference_sam3.crop_box) if reference_sam3 is not None else None
        ),
        reference_sam3_bbox=(
            list(reference_sam3.selected_bbox)
            if reference_sam3 is not None
            else None
        ),
        reference_sam3_score=(
            reference_sam3.selected_score if reference_sam3 is not None else None
        ),
        reference_sam3_candidate_count=(
            reference_sam3.candidate_count if reference_sam3 is not None else None
        ),
        registration=registration_metrics(registration),
    )
    if artifact_directory is not None:
        save_place_locate_artifacts(
            artifact_directory,
            request_payload=(artifact_request or request.model_dump(mode="json")),
            response=response,
            reference_image=reference_image,
            current_image=current_image,
            reference_depth_mm=reference_depth_mm,
            current_depth_mm=current_depth_mm,
            prepared=prepared,
        )
    return response


def build_reference_mask_debug_response(
    request: PlaceReferenceMaskRequest,
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    reference_intrinsics: CameraIntrinsics,
    current_intrinsics: CameraIntrinsics,
    *,
    inspection_target_id: str,
    baseline_path: str,
) -> PlaceReferenceMaskDebugResponse:
    prepared = prepare_reference_mask(
        request,
        reference_image,
        current_image,
        reference_depth_mm,
        current_depth_mm,
        reference_intrinsics,
        current_intrinsics,
    )
    reference_height, reference_width = reference_image.shape[:2]
    current_height, current_width = current_image.shape[:2]
    sam3 = prepared.sam3
    return PlaceReferenceMaskDebugResponse(
        task_type=request.task_type,
        product_name=request.product_name,
        location_id=request.location_id,
        inspection_target_id=inspection_target_id,
        pose_type=request.pose_type,
        baseline_path=baseline_path,
        region_index=prepared.selected_region_index,
        reference_image_size=[reference_width, reference_height],
        current_image_size=[current_width, current_height],
        reference_image_base64=encode_jpeg_base64(reference_image),
        change_mask_reference=encode_png_base64(
            prepared.registration.rgb.change_mask_reference
        ),
        reference_component_mask=encode_png_base64(prepared.component_mask),
        reference_bbox=prepared.reference_bbox,
        row_index=(
            prepared.matched_row.index if prepared.matched_row is not None else None
        ),
        row_bbox=(
            list(prepared.matched_row.bbox)
            if prepared.matched_row is not None
            else None
        ),
        reference_mask=encode_png_base64(prepared.reference_mask),
        reference_mask_source="sam3" if sam3 is not None else "depth_change",
        reference_sam3_prompt=(sam3.sam_prompt if sam3 is not None else None),
        reference_sam3_crop_box=(
            list(sam3.crop_box) if sam3 is not None else None
        ),
        reference_sam3_bbox=(
            list(sam3.selected_bbox) if sam3 is not None else None
        ),
        reference_sam3_score=(sam3.selected_score if sam3 is not None else None),
        reference_sam3_candidate_count=(
            sam3.candidate_count if sam3 is not None else None
        ),
        registration=registration_metrics(prepared.registration),
    )


def locate_reference_mask_debug(
    request: PlaceReferenceMaskRequest,
) -> PlaceReferenceMaskDebugResponse:
    """Generate the Task0 reference mask without requiring a 6D object pose."""

    try:
        initial_scan = load_initial_scan(request.location_id, request.pose_type)
    except InitialScanError as error:
        raise HTTPException(
            status_code=422,
            detail={"type": "initial_scan_error", "message": str(error)},
        ) from error
    runtime_request = request.model_copy(update={"pose_type": initial_scan.pose_type})
    reference_image = initial_scan.rgb
    current_image = decode_color_image(
        request.current_image_base64,
        "current_image_base64",
    )
    reference_depth = initial_scan.depth_mm
    current_depth = decode_depth_image(
        request.current_depth_image_base64,
        "current_depth_image_base64",
        current_image.shape[:2],
        is_bigendian=request.depth_is_bigendian,
        depth_unit_mm=request.depth_unit_mm,
    )
    reference_intrinsics, current_intrinsics = resolve_intrinsics(
        reference_image,
        current_image,
    )
    try:
        return build_reference_mask_debug_response(
            runtime_request,
            reference_image,
            current_image,
            reference_depth,
            current_depth,
            reference_intrinsics,
            current_intrinsics,
            inspection_target_id=initial_scan.inspection_target_id,
            baseline_path=str(initial_scan.rgb_path),
        )
    except HTTPException:
        raise
    except ReferenceMaskError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail={"type": "reference_mask_failed", "message": str(error)},
        ) from error
    except (PoseTransferError, ValueError, cv2.error) as error:
        raise HTTPException(
            status_code=422,
            detail={"type": "place_registration_failed", "message": str(error)},
        ) from error


def locate_place_debug(
    request: PlaceLocateDebugRequest,
    *,
    persist_artifacts: bool = False,
    artifact_root: str | Path | None = None,
    artifact_request: dict[str, Any] | None = None,
) -> PlaceLocateDebugResponse:
    try:
        initial_scan = load_initial_scan(request.location_id, request.pose_type)
    except InitialScanError as error:
        raise HTTPException(
            status_code=422,
            detail={"type": "initial_scan_error", "message": str(error)},
        ) from error
    runtime_request = request.model_copy(update={"pose_type": initial_scan.pose_type})
    reference_image = initial_scan.rgb
    current_image = decode_color_image(
        request.current_image_base64,
        "current_image_base64",
    )
    reference_depth = initial_scan.depth_mm
    current_depth = decode_depth_image(
        request.current_depth_image_base64,
        "current_depth_image_base64",
        current_image.shape[:2],
        is_bigendian=request.depth_is_bigendian,
        depth_unit_mm=request.depth_unit_mm,
    )
    reference_intrinsics, current_intrinsics = resolve_intrinsics(
        reference_image,
        current_image,
    )
    artifact_directory = (
        create_place_locate_artifact_directory(
            runtime_request,
            artifact_root=artifact_root,
        )
        if persist_artifacts
        else None
    )
    persisted_request = artifact_request
    if persisted_request is None and artifact_directory is not None:
        persisted_request = runtime_request.model_dump(
            mode="json",
            exclude={"current_image_base64", "current_depth_image_base64"},
        )
    try:
        return build_debug_response(
            runtime_request,
            reference_image,
            current_image,
            reference_depth,
            current_depth,
            reference_intrinsics,
            current_intrinsics,
            inspection_target_id=initial_scan.inspection_target_id,
            baseline_path=str(initial_scan.rgb_path),
            artifact_directory=artifact_directory,
            artifact_request=persisted_request,
        )
    except HTTPException:
        raise
    except ReferenceMaskError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail={"type": "reference_mask_failed", "message": str(error)},
        ) from error
    except (PoseTransferError, ValueError, cv2.error) as error:
        raise HTTPException(
            status_code=422,
            detail={"type": "place_registration_failed", "message": str(error)},
        ) from error


@router.post("/perception/place/locate", response_model=PlaceLocateResponse)
def locate_place(request: PlaceLocateRequest) -> PlaceLocateResponse:
    try:
        with inspection_temporary_directory() as temporary_directory:
            current = capture_head_rgbd(temporary_directory)
            debug_request = PlaceLocateDebugRequest(
                task_type=request.task_type,
                product_name=request.product_name,
                location_id=request.location_id,
                pose_type=request.pose_type,
                current_image_name=current.rgb_path.name,
                current_image_base64=base64.b64encode(
                    current.rgb_path.read_bytes()
                ).decode("ascii"),
                current_depth_image_base64=base64.b64encode(
                    current.depth_path.read_bytes()
                ).decode("ascii"),
            )
            debug = locate_place_debug(
                debug_request,
                persist_artifacts=True,
                artifact_request=request.model_dump(mode="json"),
            )
    except CameraCaptureError as error:
        raise HTTPException(
            status_code=502,
            detail=f"获取当前 head camera RGB-D 失败: {error}",
        ) from error
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"创建、读取或保存定位目录失败: {error}",
        ) from error
    return PlaceLocateResponse(
        product_name=debug.product_name,
        bbox=debug.bbox,
        mask=debug.mask,
        image_path=debug.image_path,
        rotate_matrix=debug.rotate_matrix,
        level=debug.level,
    )


@router.post(
    "/perception/place/locate/debug",
    response_model=PlaceLocateDebugResponse,
)
def locate_place_debug_api(
    request: PlaceLocateDebugRequest,
) -> PlaceLocateDebugResponse:
    return locate_place_debug(request)


@router.post(
    "/perception/place/locate/reference-mask/debug",
    response_model=PlaceReferenceMaskDebugResponse,
)
def locate_reference_mask_debug_api(
    request: PlaceReferenceMaskRequest,
) -> PlaceReferenceMaskDebugResponse:
    return locate_reference_mask_debug(request)


app.include_router(router)
