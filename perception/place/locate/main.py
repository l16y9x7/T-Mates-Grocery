"""Locate current-image reference objects for product placement.

The fixed Task0 scan is still used to find the target slot. That slot is
registered into the current head-camera image, where Pick Locate's existing
Qwen/SAM3 detector supplies the neighboring reference masks.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import re
import sys
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
    from ...inspect.sam_shortage_pipeline import (
        SamShortageError,
        ShortageAnalysis,
        analysis_as_dict,
        analyze_shortage,
        full_image_mask,
        save_shelf_preprocessing_artifacts,
        select_place_references,
    )
    from ...row_detection import (
        PoseType,
        RowDetectionConfig,
        ShelfRow,
        detect_rows,
    )
else:
    PERCEPTION_ROOT = Path(__file__).resolve().parents[2]
    INSPECT_ROOT = PERCEPTION_ROOT / "inspect"
    for module_root in (PERCEPTION_ROOT, INSPECT_ROOT):
        if str(module_root) not in sys.path:
            sys.path.insert(0, str(module_root))
    from camera_capture import (
        CameraCaptureError,
        capture_head_rgbd,
        inspection_temporary_directory,
    )
    from initial_scan import InitialScanError, load_initial_scan
    from sam_shortage_pipeline import (
        SamShortageError,
        ShortageAnalysis,
        analysis_as_dict,
        analyze_shortage,
        full_image_mask,
        save_shelf_preprocessing_artifacts,
        select_place_references,
    )
    from row_detection import (
        PoseType,
        RowDetectionConfig,
        ShelfRow,
        detect_rows,
    )

from .pose_transfer import PoseTransferError
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

if __package__ and __package__.startswith("perception."):
    from ...pick.locate.main import (
        LocateRequest as PickLocateRequest,
        LocatedInstance as PickLocatedInstance,
        locate_product_debug as locate_pick_product_debug,
        uses_upper_confidence_pick,
    )
else:
    from pick.locate.main import (
        LocateRequest as PickLocateRequest,
        LocatedInstance as PickLocatedInstance,
        locate_product_debug as locate_pick_product_debug,
        uses_upper_confidence_pick,
    )


TaskType = Literal["SHORTAGE", "MISPLACED"]
ShelfLevel = Literal["L1", "L2", "L3", "L4", "L5"]
ReferenceDirection = Literal["left", "right", "both", "up"]
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
    version="3.0.0",
    description=(
        "Return current-image neighboring product bbox/masks for downstream "
        "placement pose estimation."
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
    """Current-image references consumed by downstream place pose estimation."""

    name: str
    bbox: list[list[int]]
    mask: list[str]
    direction: ReferenceDirection
    image_path: str
    current_image_path: str
    level: ShelfLevel


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
    target_bbox_current: list[int]
    current_candidate_count: int
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
    request: PlaceLocateDebugRequest | PlaceLocateRequest,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    """Create one inspect-style directory for a formal Place Locate request."""

    root = Path(artifact_root) if artifact_root is not None else DEFAULT_ARTIFACT_ROOT
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    safe_location = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.location_id.strip())
    product_name = request.product_name
    safe_product = re.sub(r"[^A-Za-z0-9_.-]+", "_", product_name.strip())
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
    reference_mask_name = (
        "reference_sam3_mask.png"
        if prepared.sam3 is not None
        else "reference_mask.png"
    )
    sam3_crop_name = "sam3_crop.jpg" if prepared.sam3 is not None else None

    crop_boxes = [
        _clamped_crop_box(box, current_image.shape)
        for box in response.bbox
    ]
    current_mask_names = [
        f"current_reference_mask_{index:02d}.png"
        for index in range(1, len(response.mask) + 1)
    ]
    current_crop_names = [
        f"current_reference_crop_{index:02d}.jpg"
        for index in range(1, len(response.bbox) + 1)
    ]
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
    _write_artifact_image(directory / reference_mask_name, prepared.reference_mask)
    for mask_name, crop_name, encoded_mask, crop_box in zip(
        current_mask_names,
        current_crop_names,
        response.mask,
        crop_boxes,
        strict=True,
    ):
        try:
            mask_payload = base64.b64decode(
                encoded_mask.strip().split(",", 1)[-1],
                validate=True,
            )
        except (ValueError, binascii.Error) as error:
            raise OSError("Place Locate current reference mask is invalid base64") from error
        decoded_mask = cv2.imdecode(
            np.frombuffer(mask_payload, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        if decoded_mask is None:
            raise OSError("Place Locate current reference mask is not a PNG")
        if decoded_mask.shape != current_image.shape[:2]:
            decoded_mask = cv2.resize(
                decoded_mask,
                (current_image.shape[1], current_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        _write_artifact_image(directory / mask_name, decoded_mask)
        x1, y1, x2, y2 = crop_box
        _write_artifact_image(
            directory / crop_name,
            np.asarray(current_image)[y1:y2, x1:x2],
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
        "reference_mask": reference_mask_name,
        "sam3_mask": current_mask_names,
        "bbox_crop": current_crop_names,
        "bbox_crop_box": [list(box) for box in crop_boxes],
        "bbox_crop_format": ["x1", "y1", "x2", "y2"],
        "bbox_crop_source": "current_rgb/bbox",
        "sam3_crop": sam3_crop_name,
        "sam3_crop_box": list(sam3_crop_box) if sam3_crop_box is not None else None,
        "mask_coordinate_system": "current_rgb",
    }
    saved_result = response.model_dump(mode="json")
    saved_result["artifacts"] = artifacts
    _write_artifact_json(directory / "result.json", saved_result)
    logger.info(
        "Place Locate artifacts saved: directory=%s product_name=%s level=%s",
        directory,
        response.name,
        response.level,
    )


def save_sam_shortage_place_artifacts(
    directory: Path,
    *,
    request: PlaceLocateRequest,
    response: PlaceLocateResponse,
    baseline_rgb: np.ndarray,
    baseline_depth_mm: np.ndarray,
    current_rgb: np.ndarray,
    current_depth_mm: np.ndarray,
    reference_masks: Sequence[np.ndarray],
    analysis: ShortageAnalysis,
) -> None:
    """Persist the formal SAM3 shortage locate result in inspect-style form."""

    directory.mkdir(parents=True, exist_ok=True)
    _write_artifact_json(directory / "request.json", request.model_dump(mode="json"))
    _write_artifact_image(directory / "baseline_rgb.jpg", baseline_rgb)
    _write_artifact_depth(directory / "baseline_depth_mm.npy", baseline_depth_mm)
    _write_artifact_image(directory / "current_rgb.jpg", current_rgb)
    _write_artifact_depth(directory / "current_depth_mm.npy", current_depth_mm)
    shelf_artifacts = save_shelf_preprocessing_artifacts(directory, analysis)
    _write_artifact_json(
        directory / "rgbd.json",
        {
            "schema_version": 1,
            "depth_unit": "millimeter",
            "depth_aligned_to": "matching_rgb",
            "baseline": {
                "rgb": "baseline_rgb.jpg",
                "depth": "baseline_depth_mm.npy",
                "image_size": [
                    int(baseline_rgb.shape[1]),
                    int(baseline_rgb.shape[0]),
                ],
            },
            "current": {
                "rgb": "current_rgb.jpg",
                "depth": "current_depth_mm.npy",
                "image_size": [
                    int(current_rgb.shape[1]),
                    int(current_rgb.shape[0]),
                ],
            },
        },
    )

    crop_names: list[str] = []
    mask_names: list[str] = []
    for index, (bbox, mask) in enumerate(
        zip(response.bbox, reference_masks, strict=True),
        start=1,
    ):
        x1, y1, x2, y2 = _clamped_crop_box(bbox, current_rgb.shape)
        crop_name = f"current_reference_crop_{index:02d}.jpg"
        mask_name = f"current_reference_mask_{index:02d}.png"
        _write_artifact_image(
            directory / crop_name,
            np.asarray(current_rgb)[y1:y2, x1:x2],
        )
        _write_artifact_image(directory / mask_name, mask)
        crop_names.append(crop_name)
        mask_names.append(mask_name)

    saved_result = response.model_dump(mode="json")
    saved_result["artifacts"] = {
        "baseline_rgb": "baseline_rgb.jpg",
        "baseline_depth_mm": "baseline_depth_mm.npy",
        "current_rgb": "current_rgb.jpg",
        "current_depth_mm": "current_depth_mm.npy",
        "rgbd": "rgbd.json",
        "bbox_crop": crop_names,
        "sam3_mask": mask_names,
        "mask_coordinate_system": "current_rgb",
        "bbox_format": ["x1", "y1", "x2", "y2"],
        "shelf_preprocessing": shelf_artifacts,
    }
    saved_result["shortage_analysis"] = analysis_as_dict(analysis)
    _write_artifact_json(directory / "result.json", saved_result)
    logger.info(
        "SAM3 shortage Place Locate artifacts saved: directory=%s product=%s",
        directory,
        response.name,
    )


def save_sam_shortage_place_failure_artifacts(
    directory: Path,
    *,
    request: PlaceLocateRequest,
    baseline_rgb: np.ndarray,
    baseline_depth_mm: np.ndarray,
    current_rgb: np.ndarray,
    current_depth_mm: np.ndarray,
    baseline_path: Path,
    error_type: str,
    error_message: str,
    status_code: int,
    analysis: dict[str, Any] | None = None,
) -> None:
    """Persist RGB-D inputs when shortage placement cannot select a slot."""

    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "baseline_rgb": "baseline_rgb.jpg",
        "baseline_depth_mm": "baseline_depth_mm.npy",
        "current_rgb": "current_rgb.jpg",
        "current_depth_mm": "current_depth_mm.npy",
        "rgbd": "rgbd.json",
    }
    _write_artifact_json(directory / "request.json", request.model_dump(mode="json"))
    _write_artifact_image(directory / artifacts["baseline_rgb"], baseline_rgb)
    _write_artifact_depth(
        directory / artifacts["baseline_depth_mm"],
        baseline_depth_mm,
    )
    _write_artifact_image(directory / artifacts["current_rgb"], current_rgb)
    _write_artifact_depth(
        directory / artifacts["current_depth_mm"],
        current_depth_mm,
    )
    _write_artifact_json(
        directory / artifacts["rgbd"],
        {
            "schema_version": 1,
            "depth_unit": "millimeter",
            "depth_aligned_to": "matching_rgb",
            "baseline": {
                "rgb": artifacts["baseline_rgb"],
                "depth": artifacts["baseline_depth_mm"],
                "image_size": [
                    int(baseline_rgb.shape[1]),
                    int(baseline_rgb.shape[0]),
                ],
            },
            "current": {
                "rgb": artifacts["current_rgb"],
                "depth": artifacts["current_depth_mm"],
                "image_size": [
                    int(current_rgb.shape[1]),
                    int(current_rgb.shape[0]),
                ],
            },
        },
    )

    error = {
        "type": error_type,
        "message": error_message,
        "status_code": status_code,
    }
    saved_result: dict[str, Any] = {
        "status": "error",
        "name": request.product_name,
        "task_type": request.task_type,
        "location_id": request.location_id,
        "pose_type": request.pose_type,
        "image_path": str(baseline_path.resolve()),
        "current_image_path": str((directory / artifacts["current_rgb"]).resolve()),
        "error": error,
        "artifacts": artifacts,
    }
    if analysis is not None:
        saved_result["shortage_analysis"] = analysis
    _write_artifact_json(directory / "error.json", error)
    _write_artifact_json(directory / "result.json", saved_result)
    logger.info(
        "Failed SAM3 shortage Place Locate artifacts saved: directory=%s product=%s error=%s",
        directory,
        request.product_name,
        error_message,
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


def project_reference_bbox_to_current(
    bbox_xywh: Sequence[int | float],
    homography: np.ndarray,
    current_shape: Sequence[int],
) -> list[int]:
    """Project a reference ``xywh`` box into current-image ``xyxy`` pixels."""

    if len(bbox_xywh) != 4 or len(current_shape) < 2:
        raise PoseTransferError("cannot project an invalid reference bbox")
    x, y, width, height = [float(value) for value in bbox_xywh]
    if width <= 0 or height <= 0:
        raise PoseTransferError("reference bbox width and height must be positive")
    corners = np.asarray(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float32,
    ).reshape(1, -1, 2)
    projected = cv2.perspectiveTransform(
        corners,
        np.asarray(homography, dtype=np.float64),
    ).reshape(-1, 2)
    if not np.all(np.isfinite(projected)):
        raise PoseTransferError("projected target bbox contains non-finite values")
    current_height, current_width = int(current_shape[0]), int(current_shape[1])
    x1 = max(0, min(current_width, int(np.floor(projected[:, 0].min()))))
    y1 = max(0, min(current_height, int(np.floor(projected[:, 1].min()))))
    x2 = max(0, min(current_width, int(np.ceil(projected[:, 0].max()))))
    y2 = max(0, min(current_height, int(np.ceil(projected[:, 1].max()))))
    if x2 <= x1 or y2 <= y1:
        raise PoseTransferError("projected target bbox is outside the current image")
    return [x1, y1, x2, y2]


def locate_current_product_instances(
    current_image: np.ndarray,
    product_name: str,
    task_type: TaskType,
    level: ShelfLevel,
) -> list[PickLocatedInstance]:
    """Reuse Pick Locate's current-image Qwen/SAM3 product detector."""

    request = PickLocateRequest(
        task_type=task_type,
        product_name=product_name,
        level=level,
        hand="left",
        image_name=DEFAULT_CURRENT_IMAGE_NAME,
        image_base64=encode_jpeg_base64(current_image),
    )
    result = locate_pick_product_debug(request)
    if result.error:
        raise HTTPException(
            status_code=result.error_status_code or 422,
            detail=f"当前图商品实例识别失败: {result.error}",
        )
    if not result.instances:
        raise HTTPException(status_code=404, detail="当前图没有找到同名参照商品")
    return list(result.instances)


def _instance_bbox_pixels(
    instance: PickLocatedInstance,
    image_shape: Sequence[int],
) -> list[int]:
    height, width = int(image_shape[0]), int(image_shape[1])
    x1 = max(0, min(width, int(np.floor(float(instance.bbox[0])))))
    y1 = max(0, min(height, int(np.floor(float(instance.bbox[1])))))
    x2 = max(0, min(width, int(np.ceil(float(instance.bbox[2])))))
    y2 = max(0, min(height, int(np.ceil(float(instance.bbox[3])))))
    if x2 <= x1 or y2 <= y1:
        raise PoseTransferError("Pick Locate returned an empty current-image bbox")
    return [x1, y1, x2, y2]


def _filter_instances_to_target_row(
    instances: list[PickLocatedInstance],
    row_bbox_current: list[int] | None,
) -> list[PickLocatedInstance]:
    if row_bbox_current is None:
        return instances
    _, row_y1, _, row_y2 = row_bbox_current
    tolerance = max(8.0, (row_y2 - row_y1) * 0.20)
    filtered = [
        instance
        for instance in instances
        if row_y1 - tolerance
        <= (float(instance.bbox[1]) + float(instance.bbox[3])) / 2.0
        <= row_y2 + tolerance
    ]
    return filtered or instances


def select_place_reference_instances(
    instances: list[PickLocatedInstance],
    target_bbox_current: Sequence[int | float],
    *,
    place_on_top: bool,
) -> tuple[list[PickLocatedInstance], ReferenceDirection]:
    """Select current references relative to the registered target slot.

    Horizontal placement returns one instance on each side when possible.  At
    an edge it returns the two nearest instances from the available side.
    Vertical placement returns the nearest horizontally aligned instance below
    the target slot.
    """

    if len(target_bbox_current) != 4:
        raise PoseTransferError("target_bbox_current must be xyxy")
    target_x1, target_y1, target_x2, target_y2 = [
        float(value) for value in target_bbox_current
    ]
    target_width = max(1.0, target_x2 - target_x1)
    target_height = max(1.0, target_y2 - target_y1)
    target_cx = (target_x1 + target_x2) / 2.0
    target_cy = (target_y1 + target_y2) / 2.0

    def center(instance: PickLocatedInstance) -> tuple[float, float]:
        return (
            (float(instance.bbox[0]) + float(instance.bbox[2])) / 2.0,
            (float(instance.bbox[1]) + float(instance.bbox[3])) / 2.0,
        )

    if place_on_top:
        below: list[tuple[tuple[float, float], PickLocatedInstance]] = []
        for instance in instances:
            center_x, center_y = center(instance)
            vertical_offset = center_y - target_cy
            horizontal_overlap = max(
                0.0,
                min(target_x2, float(instance.bbox[2]))
                - max(target_x1, float(instance.bbox[0])),
            )
            instance_width = max(
                1.0,
                float(instance.bbox[2]) - float(instance.bbox[0]),
            )
            overlap_ratio = horizontal_overlap / min(target_width, instance_width)
            if vertical_offset <= target_height * 0.10:
                continue
            if overlap_ratio < 0.20 and abs(center_x - target_cx) > target_width:
                continue
            below.append(
                (
                    (
                        abs(center_x - target_cx) / target_width,
                        vertical_offset / target_height,
                    ),
                    instance,
                )
            )
        if not below:
            raise HTTPException(
                status_code=404,
                detail="上下放置商品的目标槽位下方没有可用 bbox/mask",
            )
        return [min(below, key=lambda item: item[0])[1]], "up"

    horizontal_tolerance = target_width * 0.08
    left = sorted(
        (
            instance
            for instance in instances
            if center(instance)[0] < target_cx - horizontal_tolerance
        ),
        key=lambda instance: target_cx - center(instance)[0],
    )
    right = sorted(
        (
            instance
            for instance in instances
            if center(instance)[0] > target_cx + horizontal_tolerance
        ),
        key=lambda instance: center(instance)[0] - target_cx,
    )
    if left and right:
        selected = [left[0], right[0]]
        direction: ReferenceDirection = "both"
    elif len(left) >= 2:
        selected = left[:2]
        direction = "left"
    elif len(right) >= 2:
        selected = right[:2]
        direction = "right"
    else:
        raise HTTPException(
            status_code=404,
            detail="当前货架行不足两个可用的左右参照 bbox/mask",
        )
    selected.sort(key=lambda instance: center(instance)[0])
    return selected, direction


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
    level = resolve_shelf_level(
        request.pose_type,
        matched_row,
        request.location_id,
    )
    reference_height, reference_width = reference_image.shape[:2]
    current_height, current_width = current_image.shape[:2]
    current_image_path = (
        str((artifact_directory / "current_rgb.jpg").resolve())
        if artifact_directory is not None
        else request.current_image_name
    )
    target_bbox_current = project_reference_bbox_to_current(
        reference_bbox,
        registration.rgb.reference_to_current_homography,
        current_image.shape,
    )
    row_bbox_current = (
        project_reference_bbox_to_current(
            matched_row.bbox,
            registration.rgb.reference_to_current_homography,
            current_image.shape,
        )
        if matched_row is not None
        else None
    )
    current_candidates = _filter_instances_to_target_row(
        locate_current_product_instances(
            current_image,
            request.product_name,
            request.task_type,
            level,
        ),
        row_bbox_current,
    )
    selected_references, direction = select_place_reference_instances(
        current_candidates,
        target_bbox_current,
        place_on_top=uses_upper_confidence_pick(
            request.product_name,
            "SORTING",
        ),
    )
    response = PlaceLocateDebugResponse(
        name=request.product_name,
        bbox=[
            _instance_bbox_pixels(instance, current_image.shape)
            for instance in selected_references
        ],
        mask=[instance.mask for instance in selected_references],
        direction=direction,
        image_path=baseline_path,
        current_image_path=current_image_path,
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
        target_bbox_current=target_bbox_current,
        current_candidate_count=len(current_candidates),
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


def locate_shortage_place_from_rgbd(
    request: PlaceLocateRequest,
    *,
    current_rgb: np.ndarray,
    current_depth_mm: np.ndarray,
    artifact_root: str | Path | None = None,
) -> PlaceLocateResponse:
    """Use the production SAM3 slot comparison to locate placement references."""

    try:
        initial_scan = load_initial_scan(request.location_id, request.pose_type)
    except InitialScanError as error:
        raise HTTPException(
            status_code=422,
            detail={"type": "initial_scan_error", "message": str(error)},
        ) from error

    artifact_directory = create_place_locate_artifact_directory(
        request,
        artifact_root=artifact_root,
    )
    analysis = None
    try:
        analysis = analyze_shortage(
            location_id=initial_scan.inspection_target_id,
            pose_type=cast(PoseType, initial_scan.pose_type),
            baseline_rgb=initial_scan.rgb,
            baseline_depth_mm=initial_scan.depth_mm,
            current_rgb=current_rgb,
            current_depth_mm=current_depth_mm,
            product_name_filter=request.product_name,
        )
        selection = select_place_references(analysis, request.product_name)
    except SamShortageError as error:
        message = str(error)
        status_code = 502 if message.startswith("SAM3 ") else 422
        try:
            save_sam_shortage_place_failure_artifacts(
                artifact_directory,
                request=request,
                baseline_rgb=initial_scan.rgb,
                baseline_depth_mm=initial_scan.depth_mm,
                current_rgb=current_rgb,
                current_depth_mm=current_depth_mm,
                baseline_path=initial_scan.rgb_path,
                error_type="sam_shortage_place_failed",
                error_message=message,
                status_code=status_code,
                analysis=(
                    analysis_as_dict(analysis)
                    if analysis is not None
                    else None
                ),
            )
        except (OSError, TypeError, ValueError, cv2.error):
            logger.exception(
                "Failed to persist SAM3 shortage Place Locate error artifacts: directory=%s",
                artifact_directory,
            )
        raise HTTPException(
            status_code=status_code,
            detail={"type": "sam_shortage_place_failed", "message": message},
        ) from error

    image_height, image_width = current_rgb.shape[:2]
    bboxes: list[list[int]] = []
    full_masks: list[np.ndarray] = []
    for instance in selection.references:
        bbox = [
            max(0, min(image_width, int(round(instance.bbox_original_xyxy[0])))),
            max(0, min(image_height, int(round(instance.bbox_original_xyxy[1])))),
            max(0, min(image_width, int(round(instance.bbox_original_xyxy[2])))),
            max(0, min(image_height, int(round(instance.bbox_original_xyxy[3])))),
        ]
        _clamped_crop_box(bbox, current_rgb.shape)
        bboxes.append(bbox)
        full_masks.append(
            full_image_mask(instance, selection.current_row, current_rgb.shape)
        )

    current_image_path = (artifact_directory / "current_rgb.jpg").resolve()
    response = PlaceLocateResponse(
        name=request.product_name,
        bbox=bboxes,
        mask=[encode_png_base64(mask) for mask in full_masks],
        direction=selection.direction,
        image_path=str(initial_scan.rgb_path.resolve()),
        current_image_path=str(current_image_path),
        level=cast(ShelfLevel, selection.level),
    )
    try:
        save_sam_shortage_place_artifacts(
            artifact_directory,
            request=request,
            response=response,
            baseline_rgb=initial_scan.rgb,
            baseline_depth_mm=initial_scan.depth_mm,
            current_rgb=current_rgb,
            current_depth_mm=current_depth_mm,
            reference_masks=full_masks,
            analysis=analysis,
        )
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"保存 Place Locate SAM3 结果失败: {error}",
        ) from error
    return response


@router.post("/perception/place/locate", response_model=PlaceLocateResponse)
def locate_place(request: PlaceLocateRequest) -> PlaceLocateResponse:
    try:
        with inspection_temporary_directory() as temporary_directory:
            current = capture_head_rgbd(temporary_directory)
            if (
                request.task_type == "SHORTAGE"
                and hasattr(current, "rgb")
                and hasattr(current, "depth_mm")
            ):
                return locate_shortage_place_from_rgbd(
                    request,
                    current_rgb=current.rgb,
                    current_depth_mm=current.depth_mm,
                )
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
        name=debug.name,
        bbox=debug.bbox,
        mask=debug.mask,
        direction=debug.direction,
        image_path=debug.image_path,
        current_image_path=debug.current_image_path,
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
