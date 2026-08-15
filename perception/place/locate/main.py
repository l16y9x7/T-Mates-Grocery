"""HTTP API for transferring a reference product mask into the current view.

The inspection endpoint remains an RGB-only semantic stage.  This endpoint is
the RGB-D geometry stage: it registers a normal reference scene to the current
shortage/misplaced scene, reconstructs the changed reference object from its
reference depth, and projects that object into the current head-camera image.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

if __package__ and __package__.startswith("perception."):
    from ...initial_scan import InitialScanError, load_initial_scan
    from ...row_detection import (
        PoseType,
        RowDetectionConfig,
        ShelfRow,
        detect_rows,
    )
else:
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
    transfer_reference_pose,
)
from .registration import (
    CameraIntrinsics,
    RGBDRegistrationResult,
    ReprojectedMask,
    register_rgbd_images,
    reproject_reference_mask,
)


TaskType = Literal["SHORTAGE", "MISPLACED"]
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_DEPTH_BYTES = 40 * 1024 * 1024
DEFAULT_CURRENT_IMAGE_NAME = "current_rgb.jpg"
INTRINSICS_ENVIRONMENT_NAME = "PLACE_HEAD_CAMERA_INTRINSICS"
MAX_REGISTRATION_RMSE_MM = 20.0
MIN_REGISTRATION_INLIER_RATIO = 0.5
MAX_RGB_REPROJECTION_RMSE_PX = 4.0

app = FastAPI(
    title="Place Locate",
    version="1.0.0",
    description="Transfer a reference RGB-D product mask into the current view.",
)
router = APIRouter()


class CameraIntrinsicsInput(BaseModel):
    """Pinhole calibration; width/height describe its calibration resolution."""

    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)

    def resolve(self, image_width: int, image_height: int) -> CameraIntrinsics:
        calibration_width = self.width or image_width
        calibration_height = self.height or image_height
        scale_x = image_width / calibration_width
        scale_y = image_height / calibration_height
        return CameraIntrinsics(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            width=image_width,
            height=image_height,
        )


class PoseInput(BaseModel):
    """Object pose in the normal reference head-camera frame."""

    frame_id: Literal["reference_head_camera"] = "reference_head_camera"
    unit: Literal["millimeter"] = "millimeter"
    matrix: list[list[float]]

    @field_validator("matrix")
    @classmethod
    def validate_matrix(cls, value: list[list[float]]) -> list[list[float]]:
        try:
            return as_rigid_transform(value, name="reference_pose.matrix").tolist()
        except PoseTransferError as error:
            raise ValueError(str(error)) from error


class PoseOutput(BaseModel):
    frame_id: Literal["current_head_camera"] = "current_head_camera"
    unit: Literal["millimeter"] = "millimeter"
    matrix: list[list[float]]


class PlaceLocateRequest(BaseModel):
    """Current RGB-D plus the identity of a fixed task0 reference scan."""

    model_config = ConfigDict(extra="forbid")

    task_type: TaskType = "SHORTAGE"
    product_name: str = Field(min_length=1)
    location_id: str = Field(min_length=1)
    current_image_base64: str = Field(min_length=1)
    current_depth_image_base64: str = Field(min_length=1)
    reference_pose: PoseInput
    pose_type: PoseType = ""
    current_image_name: str = DEFAULT_CURRENT_IMAGE_NAME
    camera_intrinsics: CameraIntrinsicsInput | None = None
    current_camera_intrinsics: CameraIntrinsicsInput | None = None
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


class PlaceLocateResponse(BaseModel):
    """Transferred pose plus Pick Locate-compatible mask fields."""

    product_name: str
    bbox: list[int]
    mask: str
    image_path: str
    target_pose: PoseOutput


class RegistrationMetrics(BaseModel):
    current_from_reference: list[list[float]]
    rmse_mm: float
    depth_correspondence_count: int
    inlier_count: int
    inlier_ratio: float
    reprojection_rmse_px: float


class PlaceLocateDebugResponse(PlaceLocateResponse):
    task_type: TaskType
    location_id: str
    inspection_target_id: str
    baseline_path: str
    region_index: int
    image_size: list[int]
    reference_bbox: list[int]
    row_index: int | None = None
    row_bbox: list[int] | None = None
    target_bbox_pixels: list[int]
    reference_mask: str
    visible_mask: str
    expected_depth_npy_base64: str
    projected_point_count: int
    visible_point_count: int
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


def _environment_intrinsics() -> CameraIntrinsicsInput | None:
    encoded = os.getenv(INTRINSICS_ENVIRONMENT_NAME, "").strip()
    if not encoded:
        return None
    try:
        payload = json.loads(encoded)
        return CameraIntrinsicsInput.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"{INTRINSICS_ENVIRONMENT_NAME} is invalid: {error}",
        ) from error


def resolve_intrinsics(
    request: PlaceLocateRequest,
    reference_image: np.ndarray,
    current_image: np.ndarray,
) -> tuple[CameraIntrinsics, CameraIntrinsics]:
    reference_input = request.camera_intrinsics or _environment_intrinsics()
    if reference_input is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "camera_intrinsics is required unless "
                f"{INTRINSICS_ENVIRONMENT_NAME} is configured"
            ),
        )
    reference_height, reference_width = reference_image.shape[:2]
    current_height, current_width = current_image.shape[:2]
    reference = reference_input.resolve(reference_width, reference_height)
    current_input = request.current_camera_intrinsics or reference_input
    current = current_input.resolve(current_width, current_height)
    return reference, current


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


def encode_npy_base64(array: np.ndarray) -> str:
    output = io.BytesIO()
    np.save(output, np.asarray(array), allow_pickle=False)
    return base64.b64encode(output.getvalue()).decode("ascii")


def mask_bbox_xyxy(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(np.asarray(mask) > 0)
    if not len(xs):
        raise PoseTransferError("projected target mask is empty")
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def normalize_bbox_to_1_1000(bbox: Sequence[int], image_size: Sequence[int]) -> list[int]:
    width, height = image_size
    scales = (width, height, width, height)
    return [
        max(1, min(1000, round(1 + max(0, min(scale, value)) / scale * 999)))
        for value, scale in zip(bbox, scales)
    ]


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


def build_debug_response(
    request: PlaceLocateRequest,
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    reference_intrinsics: CameraIntrinsics,
    current_intrinsics: CameraIntrinsics,
    *,
    inspection_target_id: str,
    baseline_path: str,
) -> PlaceLocateDebugResponse:
    registration: RGBDRegistrationResult = register_rgbd_images(
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
    reference_mask = refine_reference_mask_with_depth(
        component,
        reference_depth_mm,
        reference_bbox,
    )
    projected: ReprojectedMask = reproject_reference_mask(
        reference_mask,
        reference_depth_mm,
        registration.current_from_reference,
        reference_intrinsics,
        current_intrinsics,
        current_depth_mm=current_depth_mm,
    )
    target_bbox = mask_bbox_xyxy(projected.full_mask)
    target_pose_matrix = transfer_reference_pose(
        registration.current_from_reference,
        request.reference_pose.matrix,
    )
    current_height, current_width = current_image.shape[:2]
    return PlaceLocateDebugResponse(
        product_name=request.product_name,
        bbox=normalize_bbox_to_1_1000(
            target_bbox,
            [current_width, current_height],
        ),
        mask=encode_png_base64(projected.full_mask),
        image_path=request.current_image_name,
        target_pose=PoseOutput(
            frame_id="current_head_camera",
            matrix=target_pose_matrix.tolist(),
        ),
        task_type=request.task_type,
        location_id=request.location_id,
        inspection_target_id=inspection_target_id,
        baseline_path=baseline_path,
        region_index=selected_index,
        image_size=[current_width, current_height],
        reference_bbox=reference_bbox,
        row_index=matched_row.index if matched_row is not None else None,
        row_bbox=list(matched_row.bbox) if matched_row is not None else None,
        target_bbox_pixels=target_bbox,
        reference_mask=encode_png_base64(reference_mask),
        visible_mask=encode_png_base64(projected.visible_mask),
        expected_depth_npy_base64=encode_npy_base64(projected.expected_depth_mm),
        projected_point_count=projected.projected_point_count,
        visible_point_count=projected.visible_point_count,
        registration=RegistrationMetrics(
            current_from_reference=registration.current_from_reference.tolist(),
            rmse_mm=registration.rmse_mm,
            depth_correspondence_count=registration.depth_correspondence_count,
            inlier_count=registration.inlier_count,
            inlier_ratio=registration.inlier_ratio,
            reprojection_rmse_px=registration.rgb.reprojection_rmse_px,
        ),
    )


def locate_place_debug(request: PlaceLocateRequest) -> PlaceLocateDebugResponse:
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
        runtime_request,
        reference_image,
        current_image,
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
        )
    except HTTPException:
        raise
    except (PoseTransferError, ValueError, cv2.error) as error:
        raise HTTPException(
            status_code=422,
            detail={"type": "place_registration_failed", "message": str(error)},
        ) from error


@router.post("/perception/place/locate", response_model=PlaceLocateResponse)
def locate_place(request: PlaceLocateRequest) -> PlaceLocateResponse:
    debug = locate_place_debug(request)
    return PlaceLocateResponse(
        product_name=debug.product_name,
        bbox=debug.bbox,
        mask=debug.mask,
        image_path=debug.image_path,
        target_pose=debug.target_pose,
    )


@router.post(
    "/perception/place/locate/debug",
    response_model=PlaceLocateDebugResponse,
)
def locate_place_debug_api(request: PlaceLocateRequest) -> PlaceLocateDebugResponse:
    return locate_place_debug(request)


app.include_router(router)
