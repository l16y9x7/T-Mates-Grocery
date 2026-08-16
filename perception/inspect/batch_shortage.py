"""Batch-run the current shortage inspection pipeline on grouped RGB-D records."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np


INSPECT_ROOT = Path(__file__).resolve().parent
PERCEPTION_ROOT = INSPECT_ROOT.parent
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from initial_scan import InitialScan, load_initial_scan  # noqa: E402


DEFAULT_DATA_ROOT = (
    PERCEPTION_ROOT / "test_data" / "2026-08-16-self-collect-shortage-grouped"
)
DEFAULT_SUMMARY_NAME = "shortage_inspection_batch_results.json"
RESULT_DIRECTORY_NAME = "shortage_inspection"
GROUP_PATTERN = re.compile(
    r"^(?P<target>H[12]_[FB]_[LR]_INSPECT)_(?P<pose>UPPER|LOWER)$"
)
RECORD_PATTERN = re.compile(r"^record_\d{8}_\d{6}_\d{6}$")
DEPTH_CHANGE_THRESHOLD_MM = 60.0
MIN_DEPTH_VALID_PIXELS = 50
MIN_DEPTH_VALID_RATIO = 0.02
MIN_DEPTH_FARTHER_RATIO = 0.20
MIN_DEPTH_PROMOTION_AREA_RATIO = 0.001
MAX_DEPTH_PROMOTION_AREA_RATIO = 0.20
MIN_DEPTH_PROMOTION_RGB_PIXELS = 10
MIN_DEPTH_PROMOTION_RGB_RATIO = 0.20
MIN_DEPTH_PROMOTION_NEAR_RATIO = 0.15
MIN_DEPTH_PROMOTION_ROW_OVERLAP = 0.80
DEPTH_PROMOTION_BORDER_RATIO = 0.015
DEPTH_PROMOTION_RGB_DILATION_RATIO = 0.011
DEPTH_PROMOTION_MAX_ASPECT_RATIO = 5.0
DEPTH_PROMOTION_DUPLICATE_IOU = 0.20
DEPTH_PROMOTION_INTERIOR_INSET_RATIO = 0.25
MIN_OPEN_ROW_INTERIOR_FILL_RATIO = 0.20
SIGNED_DEPTH_CLOSE_RATIO = 0.0125
MIN_SIGNED_DEPTH_COMPONENT_AREA_RATIO = 0.001
MIN_SIGNED_DEPTH_RGB_OVERLAP_RATIO = 0.02
MAX_SIGNED_DEPTH_COMPONENT_ASPECT_RATIO = 8.0
MAX_SIGNED_DEPTH_COMPONENTS_PER_FINDING = 3
MIN_DOMINANT_DEPTH_PROMOTION_AREA_RATIO = 0.003
MIN_DOMINANT_DEPTH_PROMOTION_RGB_RATIO = 0.22
MIN_LOW_CONTRAST_PROMOTION_AREA_RATIO = 0.0025
MIN_LOW_CONTRAST_DIFFERENCE = 15.0
MAX_LOW_CONTRAST_DIFFERENCE = 60.0
MIN_LOW_CONTRAST_NEARER_COMPANION_RATIO = 0.10
MAX_RGB_FALLBACK_CHROMA_DOMINANCE_RATIO = 0.15
MIN_BALANCED_DEPTH_CHANGE_RATIO = 0.03
MIN_OUTWARD_SHELF_OVERLAP_RATIO = 0.75
MIN_CANDIDATE_ROW_OVERLAP_RATIO = 0.50
UPPER_SIDE_CROP_RATIO = 0.10
LOWER_IMAGE_SIDE_CROP_RATIO = 0.10
LOWER_SHELF_SIDE_INSET_RATIO = 0.025
MIN_OBJECT_WINDOW_WIDTH_RATIO = 0.24
MIN_OBJECT_WINDOW_HEIGHT_RATIO = 0.45
MIN_OPEN_ROW_FARTHER_WINDOW_RATIO = 0.76
MAX_OPEN_ROW_MOVEMENT_BALANCE_RATIO = 0.55
MIN_FLOOR_SIMILARITY_RATIO = 0.50
MAX_FLOOR_LIKE_FARTHER_WINDOW_RATIO = 0.85
MIN_BACK_LEFT_CANDIDATE_WIDTH_RATIO = 0.135
MIN_BACK_LEFT_CANDIDATE_HEIGHT_RATIO = 0.18
MAX_BACK_LEFT_MOVEMENT_BALANCE_RATIO = 0.68
MIN_STACKED_COLUMN_HEIGHT_RATIO = 0.65
MIN_STACKED_COLUMN_WIDTH_RATIO = 0.25
MIN_STACKED_COLUMN_FARTHER_RATIO = 0.25
MAX_FRAGMENT_VERTICAL_GAP_RATIO = 0.15
MIN_FRAGMENT_HORIZONTAL_OVERLAP_RATIO = 0.45
MAX_FRONT_RIGHT_TOP_FRAGMENT_WIDTH_RATIO = 0.14
MAX_FRONT_RIGHT_TOP_FRAGMENT_HEIGHT_RATIO = 0.22
MIN_MOVEMENT_OBJECT_WINDOW_FARTHER_RATIO = 0.20
MIN_DEPTH_SUPPORTED_OBJECT_WINDOW_RATIO = 0.02
RECOVERY_CLOSE_RATIO = 0.05
MIN_RECOVERY_COMPONENT_AREA_RATIO = 0.0025
MIN_RECOVERY_WIDTH_TO_ROW_HEIGHT_RATIO = 0.15
MAX_RECOVERY_MOVEMENT_BALANCE_RATIO = 0.55
MIN_RECOVERY_VISUAL_DIFFERENCE = 25.0
MAX_RECOVERY_VISUAL_DIFFERENCE = 85.0
RECOVERY_LEFT_EDGE_CLEARANCE_RATIO = 0.04
RECOVERY_RIGHT_EDGE_CLEARANCE_RATIO = 0.15
MAX_FRONT_RGB_FALLBACK_CHROMA_DOMINANCE_RATIO = 0.80
FIXED_LAYOUT_REFERENCE_SIZE = (1280, 720)
H2_BACK_LEFT_UPPER_SLOT_DEPTH_THRESHOLD_MM = 20.0
MIN_H2_BACK_LEFT_UPPER_SLOT_FARTHER_RATIO = 0.25
MIN_H2_FRONT_LEFT_UPPER_SLOT_SCORE_MM = 10.0
MIN_H2_FRONT_LEFT_UPPER_SLOT_RGB_RATIO = 0.02
DEFAULT_WORKERS = 4


# Detection ROI, output bbox.  These are fixed inspection poses; coordinates
# are scaled from the 1280x720 task0 scan so the rule also survives resizing.
H2_BACK_LEFT_UPPER_DEPTH_SLOTS = (
    ((120, 145, 195, 105), (90, 125, 235, 145)),
    ((350, 145, 175, 105), (335, 125, 210, 145)),
    ((540, 145, 170, 105), (525, 125, 205, 145)),
    ((175, 350, 190, 270), (175, 335, 190, 300)),
    ((400, 350, 145, 270), (390, 335, 170, 300)),
    ((600, 350, 140, 270), (580, 335, 180, 300)),
)
H2_FRONT_LEFT_UPPER_DEPTH_SLOTS = (
    ((60, 0, 180, 240), -80.0),
    ((240, 0, 190, 240), 30.0),
    ((430, 0, 195, 240), 30.0),
    ((625, 0, 205, 240), 30.0),
    ((830, 0, 200, 240), 30.0),
    ((1030, 0, 180, 240), 30.0),
)


def load_inspect_api() -> ModuleType:
    module_name = "perception_shortage_batch_inspect_api"
    existing = sys.modules.get(module_name)
    if isinstance(existing, ModuleType):
        return existing
    if str(INSPECT_ROOT) not in sys.path:
        sys.path.insert(0, str(INSPECT_ROOT))
    spec = importlib.util.spec_from_file_location(module_name, INSPECT_ROOT / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载巡检入口: {INSPECT_ROOT / 'main.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


INSPECT_API = load_inspect_api()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"读取 JSON 失败 {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON 必须是对象: {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_image(path: Path) -> np.ndarray:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise RuntimeError(f"读取 RGB 失败 {path}: {error}") from error
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"RGB 文件无效: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".png", image)
    if not success:
        raise RuntimeError(f"无法编码图像: {path}")
    encoded.tofile(path)


def validate_depth(
    record_directory: Path,
    image_shape: tuple[int, int],
) -> tuple[Path, int, np.ndarray]:
    depth_path = record_directory / "depth_mm.npy"
    try:
        depth = np.load(depth_path, allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"读取深度失败 {depth_path}: {error}") from error
    if depth.ndim != 2 or depth.shape != image_shape:
        raise RuntimeError(
            f"RGB/深度尺寸不一致: rgb={image_shape}, depth={depth.shape}"
        )
    if not np.issubdtype(depth.dtype, np.number):
        raise RuntimeError(f"深度必须是数值数组: {depth_path}")
    depth_mm = depth.astype(np.float32)
    valid_pixels = int(np.count_nonzero(np.isfinite(depth_mm) & (depth_mm > 0)))
    return depth_path, valid_pixels, depth_mm


def parse_group_name(group_name: str) -> tuple[str, str]:
    match = GROUP_PATTERN.fullmatch(group_name)
    if match is None:
        raise RuntimeError(f"巡检分组名称不合法: {group_name}")
    return match.group("target"), f"SHELF_VIEW_{match.group('pose')}"


def discover_records(
    data_root: Path,
    *,
    groups: set[str] | None = None,
    record_name: str | None = None,
) -> list[dict[str, Any]]:
    if not data_root.is_dir():
        raise RuntimeError(f"批测数据目录不存在: {data_root}")
    records: list[dict[str, Any]] = []
    for group_directory in sorted(data_root.iterdir(), key=lambda path: path.name):
        if not group_directory.is_dir() or GROUP_PATTERN.fullmatch(group_directory.name) is None:
            continue
        if groups is not None and group_directory.name not in groups:
            continue
        target_id, pose_type = parse_group_name(group_directory.name)
        for record_directory in sorted(group_directory.iterdir(), key=lambda path: path.name):
            if not record_directory.is_dir() or RECORD_PATTERN.fullmatch(record_directory.name) is None:
                continue
            if record_name is not None and record_directory.name != record_name:
                continue
            records.append(
                {
                    "group": group_directory.name,
                    "record": record_directory.name,
                    "record_directory": record_directory,
                    "inspection_target_id": target_id,
                    "location_id": target_id,
                    "pose_type": pose_type,
                }
            )
    return records


def relative_path(path: Path, data_root: Path) -> str:
    return path.resolve().relative_to(data_root.resolve()).as_posix()


def clipped_region_mask(mask: np.ndarray, bbox: list[int]) -> np.ndarray:
    output = np.zeros(mask.shape, dtype=np.uint8)
    x, y, width, height = (int(value) for value in bbox)
    x0 = max(0, min(mask.shape[1], x))
    y0 = max(0, min(mask.shape[0], y))
    x1 = max(x0, min(mask.shape[1], x + width))
    y1 = max(y0, min(mask.shape[0], y + height))
    output[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return output


def align_depth_to_review(
    depth_mm: np.ndarray,
    review_shape: tuple[int, int],
    homography: np.ndarray | None = None,
) -> np.ndarray:
    """Resize and optionally warp RGB-aligned depth into review coordinates."""

    height, width = review_shape
    resized = cv2.resize(
        np.asarray(depth_mm, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    if homography is None:
        return resized
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise RuntimeError("RGB 配准矩阵不是有效的 3x3 homography")
    return cv2.warpPerspective(
        resized,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def depth_support_for_finding(
    photometric_mask: np.ndarray,
    bbox: list[int],
    baseline_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
) -> dict[str, Any]:
    """Measure whether an RGB change is supported by a farther current surface."""

    region_mask = clipped_region_mask(photometric_mask, bbox) > 0
    changed_pixels = int(np.count_nonzero(region_mask))
    valid = (
        region_mask
        & np.isfinite(baseline_depth_mm)
        & np.isfinite(current_depth_mm)
        & (baseline_depth_mm > 0)
        & (current_depth_mm > 0)
    )
    valid_pixels = int(np.count_nonzero(valid))
    required_pixels = max(
        MIN_DEPTH_VALID_PIXELS,
        round(changed_pixels * MIN_DEPTH_VALID_RATIO),
    )
    applicable = valid_pixels >= required_pixels
    deltas = current_depth_mm[valid] - baseline_depth_mm[valid]
    farther_pixels = int(np.count_nonzero(deltas > DEPTH_CHANGE_THRESHOLD_MM))
    farther_ratio = farther_pixels / valid_pixels if valid_pixels else 0.0
    return {
        "applicable": applicable,
        "accepted": (not applicable) or farther_ratio >= MIN_DEPTH_FARTHER_RATIO,
        "changed_pixels": changed_pixels,
        "valid_pixels": valid_pixels,
        "required_valid_pixels": required_pixels,
        "farther_pixels": farther_pixels,
        "farther_ratio": round(farther_ratio, 4),
        "median_delta_mm": (
            round(float(np.median(deltas)), 1) if deltas.size else None
        ),
        "threshold_mm": DEPTH_CHANGE_THRESHOLD_MM,
    }


def _bbox_iou(first: list[int], second: list[int]) -> float:
    ax, ay, aw, ah = (int(value) for value in first)
    bx, by, bw, bh = (int(value) for value in second)
    intersection_width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _maximum_row_overlap_ratio(
    bbox: list[int],
    row_bboxes: list[list[int]],
) -> float:
    _, y, _, height = bbox
    if height <= 0:
        return 0.0
    bottom = y + height
    return max(
        (
            max(0, min(bottom, row_y + row_height) - max(y, row_y)) / height
            for _, row_y, _, row_height in row_bboxes
        ),
        default=0.0,
    )


def _component_interior_fill_ratio(
    component: np.ndarray,
    bbox: list[int],
) -> float:
    """Measure whether a component fills its center instead of only its edges."""

    x, y, width, height = bbox
    inset_x = round(width * DEPTH_PROMOTION_INTERIOR_INSET_RATIO)
    inset_y = round(height * DEPTH_PROMOTION_INTERIOR_INSET_RATIO)
    left = x + inset_x
    top = y + inset_y
    right = x + width - inset_x
    bottom = y + height - inset_y
    if right <= left or bottom <= top:
        return 0.0
    interior = component[top:bottom, left:right]
    return float(np.count_nonzero(interior)) / interior.size


def refine_findings_with_signed_depth(
    photometric_mask: np.ndarray,
    depth_change_mask: np.ndarray,
    baseline_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    findings: list[Any],
) -> tuple[list[tuple[Any, dict[str, Any], np.ndarray]], int]:
    """Turn coarse RGB changes into baseline-side shortage depth components."""

    height, width = depth_change_mask.shape
    image_area = height * width
    minimum_area = max(
        50,
        round(image_area * MIN_SIGNED_DEPTH_COMPONENT_AREA_RATIO),
    )
    close_radius = max(
        1,
        round(min(height, width) * SIGNED_DEPTH_CLOSE_RATIO / 2),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (close_radius * 2 + 1, close_radius * 2 + 1),
    )
    rgb_binary = np.asarray(photometric_mask) > 0
    positive_depth = np.asarray(depth_change_mask) > 0
    delta = current_depth_mm - baseline_depth_mm
    refined: list[tuple[Any, dict[str, Any], np.ndarray]] = []
    accepted_input_count = 0
    occupied: list[list[int]] = []

    for finding in findings:
        x, y, box_width, box_height = (int(value) for value in finding.bbox)
        x0 = max(0, min(width, x))
        y0 = max(0, min(height, y))
        x1 = max(x0, min(width, x + box_width))
        y1 = max(y0, min(height, y + box_height))
        if x1 <= x0 or y1 <= y0:
            continue

        roi = positive_depth[y0:y1, x0:x1].astype(np.uint8)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, close_kernel)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            roi,
            8,
        )
        candidates: list[tuple[int, int, list[int], float, int]] = []
        for label in range(1, component_count):
            local_x, local_y, component_width, component_height, area = (
                int(value) for value in stats[label]
            )
            if area < minimum_area:
                continue
            aspect_ratio = max(
                component_width / max(1, component_height),
                component_height / max(1, component_width),
            )
            if aspect_ratio > MAX_SIGNED_DEPTH_COMPONENT_ASPECT_RATIO:
                continue
            component = labels == label
            rgb_pixels = int(
                np.count_nonzero(
                    component
                    & rgb_binary[y0:y1, x0:x1]
                )
            )
            required_rgb_pixels = max(
                10,
                round(area * MIN_SIGNED_DEPTH_RGB_OVERLAP_RATIO),
            )
            if rgb_pixels < required_rgb_pixels:
                continue
            component_bbox = [
                x0 + local_x,
                y0 + local_y,
                component_width,
                component_height,
            ]
            candidates.append(
                (area, label, component_bbox, aspect_ratio, rgb_pixels)
            )

        candidates.sort(reverse=True, key=lambda candidate: candidate[0])
        accepted_this_input = False
        for area, label, component_bbox, aspect_ratio, rgb_pixels in candidates[
            :MAX_SIGNED_DEPTH_COMPONENTS_PER_FINDING
        ]:
            output_bbox = component_bbox
            original_aspect_ratio = max(
                box_width / max(1, box_height),
                box_height / max(1, box_width),
            )
            # A narrow farther-depth seed often exposes only one side of the
            # removed item. Preserve the RGB coarse box for Qwen while keeping
            # the signed-depth component as the output mask.
            if aspect_ratio > 3.0 and original_aspect_ratio <= 3.0:
                output_bbox = [x0, y0, x1 - x0, y1 - y0]
            elif component_height < round((y1 - y0) * 0.65):
                # Signed depth often sees only the exposed shelf/backboard at
                # the bottom of a removed product.  Keep its precise x span,
                # but restore the RGB candidate's full product-height slot so
                # the downstream SKU image is not just a strip of shelf floor.
                output_bbox = [
                    component_bbox[0],
                    y0,
                    component_bbox[2],
                    y1 - y0,
                ]
            if any(
                _bbox_iou(output_bbox, existing_bbox)
                >= DEPTH_PROMOTION_DUPLICATE_IOU
                for existing_bbox in occupied
            ):
                continue

            local_component = labels == label
            component_mask = np.zeros((height, width), dtype=np.uint8)
            component_mask[y0:y1, x0:x1] = np.where(
                local_component,
                255,
                0,
            ).astype(np.uint8)
            component_deltas = delta[component_mask > 0]
            support = {
                "applicable": True,
                "accepted": True,
                "refined": True,
                "reason": "farther signed-depth component inside RGB candidate",
                "original_bbox": [x0, y0, x1 - x0, y1 - y0],
                "changed_pixels": rgb_pixels,
                "valid_pixels": area,
                "required_valid_pixels": minimum_area,
                "farther_pixels": area,
                "farther_ratio": 1.0,
                "median_delta_mm": (
                    round(float(np.median(component_deltas)), 1)
                    if component_deltas.size
                    else None
                ),
                "threshold_mm": DEPTH_CHANGE_THRESHOLD_MM,
                "depth_component_pixels": area,
                "required_depth_component_pixels": minimum_area,
                "rgb_evidence_pixels": rgb_pixels,
                "required_rgb_evidence_pixels": required_rgb_pixels,
                "component_aspect_ratio": round(aspect_ratio, 4),
            }
            refined_finding = INSPECT_API.Finding(
                bbox=output_bbox,
                center=[
                    output_bbox[0] + output_bbox[2] // 2,
                    output_bbox[1] + output_bbox[3] // 2,
                ],
                sources=list(finding.sources),
                votes=finding.votes,
            )
            refined.append((refined_finding, support, component_mask))
            occupied.append(output_bbox)
            accepted_this_input = True
        if accepted_this_input:
            accepted_input_count += 1

    return refined, accepted_input_count


def promote_depth_components(
    photometric_mask: np.ndarray,
    depth_change_mask: np.ndarray,
    baseline_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    row_bboxes: list[list[int]],
    open_ended_row_bboxes: list[list[int]],
    existing_findings: list[Any],
) -> list[tuple[Any, dict[str, Any], np.ndarray]]:
    """Promote large depth holes that retain nearby fragmented RGB evidence."""

    height, width = depth_change_mask.shape
    image_area = height * width
    minimum_area = max(50, round(image_area * MIN_DEPTH_PROMOTION_AREA_RATIO))
    maximum_area = max(minimum_area, round(image_area * MAX_DEPTH_PROMOTION_AREA_RATIO))
    border_clearance = max(
        2,
        round(min(height, width) * DEPTH_PROMOTION_BORDER_RATIO),
    )
    dilation_radius = max(
        2,
        round(min(height, width) * DEPTH_PROMOTION_RGB_DILATION_RATIO),
    )
    kernel_size = dilation_radius * 2 + 1
    rgb_binary = np.asarray(photometric_mask) > 0
    rgb_near = cv2.dilate(
        rgb_binary.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        ),
    ) > 0
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (depth_change_mask > 0).astype(np.uint8),
        8,
    )
    promoted: list[tuple[Any, dict[str, Any], np.ndarray]] = []
    occupied = [list(finding.bbox) for finding in existing_findings]
    delta = current_depth_mm - baseline_depth_mm
    for label in range(1, component_count):
        x, y, box_width, box_height, area = (
            int(value) for value in stats[label]
        )
        if not minimum_area <= area <= maximum_area:
            continue
        # The first visible shelf row is commonly cropped by the image top,
        # especially for LOWER views.  Reject left/right/bottom warp borders,
        # but do not discard a real product merely because it reaches y=0.
        if min(
            x,
            width - (x + box_width),
            height - (y + box_height),
        ) < border_clearance:
            continue
        aspect_ratio = max(
            box_width / max(1, box_height),
            box_height / max(1, box_width),
        )
        if aspect_ratio > DEPTH_PROMOTION_MAX_ASPECT_RATIO:
            continue
        bbox = [x, y, box_width, box_height]
        row_overlap = _maximum_row_overlap_ratio(bbox, row_bboxes)
        if row_overlap < MIN_DEPTH_PROMOTION_ROW_OVERLAP:
            continue
        component = labels == label
        bbox_fill_ratio = area / (box_width * box_height)
        interior_fill_ratio = _component_interior_fill_ratio(component, bbox)
        in_open_ended_row = (
            _maximum_row_overlap_ratio(bbox, open_ended_row_bboxes)
            >= MIN_DEPTH_PROMOTION_ROW_OVERLAP
        )
        if (
            in_open_ended_row
            and interior_fill_ratio < MIN_OPEN_ROW_INTERIOR_FILL_RATIO
        ):
            continue
        rgb_pixels = int(np.count_nonzero(component & rgb_binary))
        rgb_evidence_ratio = rgb_pixels / area
        required_rgb_pixels = max(
            MIN_DEPTH_PROMOTION_RGB_PIXELS,
            round(area * MIN_DEPTH_PROMOTION_RGB_RATIO),
        )
        nearby_rgb_pixels = int(np.count_nonzero(component & rgb_near))
        nearby_rgb_ratio = nearby_rgb_pixels / area
        if (
            rgb_pixels < required_rgb_pixels
            or nearby_rgb_ratio < MIN_DEPTH_PROMOTION_NEAR_RATIO
        ):
            continue
        if any(
            _bbox_iou(bbox, existing_bbox) >= DEPTH_PROMOTION_DUPLICATE_IOU
            for existing_bbox in occupied
        ):
            continue
        component_mask = np.where(component, 255, 0).astype(np.uint8)
        component_deltas = delta[component]
        support = {
            "applicable": True,
            "accepted": True,
            "promoted": True,
            "reason": "large depth hole with nearby RGB difference fragments",
            "changed_pixels": rgb_pixels,
            "valid_pixels": area,
            "required_valid_pixels": minimum_area,
            "farther_pixels": area,
            "farther_ratio": 1.0,
            "median_delta_mm": (
                round(float(np.median(component_deltas)), 1)
                if component_deltas.size
                else None
            ),
            "threshold_mm": DEPTH_CHANGE_THRESHOLD_MM,
            "depth_component_pixels": area,
            "required_depth_component_pixels": minimum_area,
            "rgb_evidence_pixels": rgb_pixels,
            "rgb_evidence_ratio": round(rgb_evidence_ratio, 4),
            "required_rgb_evidence_pixels": required_rgb_pixels,
            "nearby_rgb_evidence_pixels": nearby_rgb_pixels,
            "nearby_rgb_evidence_ratio": round(nearby_rgb_ratio, 4),
            "row_overlap_ratio": round(row_overlap, 4),
            "bbox_fill_ratio": round(bbox_fill_ratio, 4),
            "interior_fill_ratio": round(interior_fill_ratio, 4),
            "open_ended_row": in_open_ended_row,
        }
        finding = INSPECT_API.Finding(
            bbox=bbox,
            center=[x + box_width // 2, y + box_height // 2],
            sources=["depth_rgb_fusion"],
            votes=1,
        )
        promoted.append((finding, support, component_mask))
        occupied.append(bbox)
    return promoted


def _algorithm_finding_metadata(
    execution: Any,
    bbox: list[int],
) -> Any | None:
    """Find the detector metadata corresponding to one fused RGB bbox."""

    best: Any | None = None
    best_iou = 0.0
    for algorithm in execution.response.algorithms:
        if algorithm.name != "comparison_based" or not algorithm.success:
            continue
        for candidate in algorithm.findings:
            overlap = _bbox_iou(bbox, list(candidate.bbox))
            if overlap > best_iou:
                best = candidate
                best_iou = overlap
    return best if best_iou >= 0.5 else None


def select_rgb_fallback_finding(
    execution: Any,
    baseline_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    rows: list[Any] | None = None,
    rails: list[Any] | None = None,
) -> tuple[Any, dict[str, Any], np.ndarray] | None:
    """Keep one product-like RGB change when depth cannot expose the gap.

    Removing the front item can reveal an identical item behind it.  That item
    may lean forward, so requiring a farther current surface would erase the
    real shortage.  Geometry is checked against the cropped shelf rows before
    ranking; front views may contain legitimate high-chroma package removals.
    """

    height, width = execution.review_mask.shape
    candidates: list[
        tuple[float, float, float, Any, dict[str, Any], np.ndarray]
    ] = []
    available_rows = list(rows or [])
    available_rails = list(rails or [])
    location_id = str(execution.response.location_id).upper()
    pose_type = str(execution.response.pose_type)
    front_view = "_F_" in location_id
    maximum_chroma = (
        MAX_FRONT_RGB_FALLBACK_CHROMA_DOMINANCE_RATIO
        if front_view
        else MAX_RGB_FALLBACK_CHROMA_DOMINANCE_RATIO
    )
    valid_depth = (
        np.isfinite(baseline_depth_mm)
        & np.isfinite(current_depth_mm)
        & (baseline_depth_mm > 0)
        & (current_depth_mm > 0)
    )
    delta = current_depth_mm - baseline_depth_mm
    for finding in execution.response.findings:
        metadata = _algorithm_finding_metadata(execution, list(finding.bbox))
        if metadata is None:
            continue
        chroma_dominance = float(metadata.chroma_dominance_ratio)
        if chroma_dominance > maximum_chroma:
            continue

        x, y, box_width, box_height = (int(value) for value in finding.bbox)
        x0 = max(0, min(width, x))
        y0 = max(0, min(height, y))
        x1 = max(x0, min(width, x + box_width))
        y1 = max(y0, min(height, y + box_height))
        if x1 <= x0 or y1 <= y0:
            continue
        row = _find_candidate_row([x0, y0, x1 - x0, y1 - y0], available_rows)
        if row is not None:
            row_overlap = _candidate_row_overlap_ratio(
                [x0, y0, x1 - x0, y1 - y0],
                row,
            )
            if row_overlap < MIN_CANDIDATE_ROW_OVERLAP_RATIO:
                continue
            shelf_span = _effective_row_shelf_span(
                row,
                available_rails,
                image_width=width,
                pose_type=pose_type,
            )
            shelf_overlap = _horizontal_overlap_ratio(
                [x0, y0, x1 - x0, y1 - y0],
                shelf_span,
            )
            if (
                shelf_overlap is not None
                and shelf_overlap < MIN_OUTWARD_SHELF_OVERLAP_RATIO
            ):
                continue
            row_height = max(1, int(row.bbox[3]))
            width_ratio = (x1 - x0) / row_height
            height_ratio = (y1 - y0) / row_height
            maximum_width_ratio = (
                1.25 if pose_type.upper().endswith("LOWER") else 0.75
            )
            if width_ratio > maximum_width_ratio or height_ratio < 0.16:
                continue
        else:
            row_overlap = 1.0
            shelf_overlap = None
            width_ratio = (x1 - x0) / max(1, height)
            height_ratio = (y1 - y0) / max(1, height)
        valid = valid_depth[y0:y1, x0:x1]
        valid_pixels = int(np.count_nonzero(valid))
        region_delta = delta[y0:y1, x0:x1]
        farther_pixels = int(
            np.count_nonzero(valid & (region_delta > DEPTH_CHANGE_THRESHOLD_MM))
        )
        nearer_pixels = int(
            np.count_nonzero(valid & (region_delta < -DEPTH_CHANGE_THRESHOLD_MM))
        )
        farther_ratio = farther_pixels / valid_pixels if valid_pixels else 0.0
        nearer_ratio = nearer_pixels / valid_pixels if valid_pixels else 0.0
        region_mask = clipped_region_mask(
            execution.review_mask,
            list(finding.bbox),
        )
        changed_pixels = int(np.count_nonzero(region_mask))
        if changed_pixels == 0:
            continue
        signed_change_ratio = farther_ratio + nearer_ratio
        support = {
            "applicable": False,
            "accepted": True,
            "rgb_fallback": True,
            "reason": (
                "row-valid RGB product change retained when no reliable "
                "farther-depth component was available"
            ),
            "changed_pixels": changed_pixels,
            "valid_pixels": valid_pixels,
            "farther_pixels": farther_pixels,
            "nearer_pixels": nearer_pixels,
            "farther_ratio": round(farther_ratio, 4),
            "nearer_ratio": round(nearer_ratio, 4),
            "signed_change_ratio": round(signed_change_ratio, 4),
            "chroma_dominance_ratio": round(chroma_dominance, 4),
            "row_overlap_ratio": round(row_overlap, 4),
            "shelf_overlap_ratio": (
                round(shelf_overlap, 4) if shelf_overlap is not None else None
            ),
            "bbox_width_to_row_height_ratio": round(width_ratio, 4),
            "bbox_height_to_row_height_ratio": round(height_ratio, 4),
            "threshold_mm": DEPTH_CHANGE_THRESHOLD_MM,
        }
        product_shape_score = min(1.0, height_ratio / 0.55)
        candidate_score = float(metadata.changed_pixels) * product_shape_score
        candidates.append(
            (
                -candidate_score,
                chroma_dominance,
                signed_change_ratio,
                finding,
                support,
                region_mask,
            )
        )

    if not candidates:
        return None
    merged = merge_fragmented_candidates(
        [
            (finding, support, region_mask)
            for _, _, _, finding, support, region_mask in candidates
        ],
        available_rows,
    )
    return max(
        merged,
        key=lambda pair: (
            int(np.count_nonzero(pair[2]))
            * min(1.0, int(pair[0].bbox[3]) / max(1, height * 0.35))
        ),
    )


def promote_low_contrast_depth_component(
    baseline_image: np.ndarray,
    current_aligned_image: np.ndarray,
    depth_change_mask: np.ndarray,
    baseline_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    row_bboxes: list[list[int]],
    open_ended_row_bboxes: list[list[int]],
) -> tuple[Any, dict[str, Any], np.ndarray] | None:
    """Recover an identical rear item that slid into the removed item's slot."""

    height, width = depth_change_mask.shape
    baseline_resized = cv2.resize(
        baseline_image,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    baseline_lab = cv2.cvtColor(baseline_resized, cv2.COLOR_BGR2LAB).astype(
        np.float32
    )
    current_lab = cv2.cvtColor(
        current_aligned_image,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)
    luminance_difference = np.abs(current_lab[:, :, 0] - baseline_lab[:, :, 0])
    chroma_difference = np.sqrt(
        np.sum(
            (current_lab[:, :, 1:] - baseline_lab[:, :, 1:]) ** 2,
            axis=2,
        )
    )
    visual_difference = np.maximum(luminance_difference, chroma_difference)

    image_area = height * width
    minimum_area = max(
        50,
        round(image_area * MIN_LOW_CONTRAST_PROMOTION_AREA_RATIO),
    )
    maximum_area = max(
        minimum_area,
        round(image_area * MAX_DEPTH_PROMOTION_AREA_RATIO),
    )
    border_clearance = max(
        2,
        round(min(height, width) * DEPTH_PROMOTION_BORDER_RATIO),
    )
    positive = depth_change_mask > 0
    valid = (
        np.isfinite(baseline_depth_mm)
        & np.isfinite(current_depth_mm)
        & (baseline_depth_mm > 0)
        & (current_depth_mm > 0)
    )
    delta = current_depth_mm - baseline_depth_mm
    negative = valid & (delta < -DEPTH_CHANGE_THRESHOLD_MM)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        positive.astype(np.uint8),
        8,
    )
    candidates: list[
        tuple[float, int, Any, dict[str, Any], np.ndarray]
    ] = []
    companion_padding = max(8, round(min(height, width) * 0.03))
    for label in range(1, component_count):
        x, y, box_width, box_height, area = (
            int(value) for value in stats[label]
        )
        if not minimum_area <= area <= maximum_area:
            continue
        if min(
            x,
            width - (x + box_width),
            height - (y + box_height),
        ) < border_clearance:
            continue
        aspect_ratio = max(
            box_width / max(1, box_height),
            box_height / max(1, box_width),
        )
        if aspect_ratio > DEPTH_PROMOTION_MAX_ASPECT_RATIO:
            continue
        bbox = [x, y, box_width, box_height]
        if (
            _maximum_row_overlap_ratio(bbox, row_bboxes)
            < MIN_DEPTH_PROMOTION_ROW_OVERLAP
        ):
            continue
        component = labels == label
        if (
            _maximum_row_overlap_ratio(bbox, open_ended_row_bboxes)
            >= MIN_DEPTH_PROMOTION_ROW_OVERLAP
            and _component_interior_fill_ratio(component, bbox)
            < MIN_OPEN_ROW_INTERIOR_FILL_RATIO
        ):
            continue
        mean_difference = float(np.mean(visual_difference[component]))
        if not (
            MIN_LOW_CONTRAST_DIFFERENCE
            <= mean_difference
            <= MAX_LOW_CONTRAST_DIFFERENCE
        ):
            continue
        x0 = max(0, x - companion_padding)
        y0 = max(0, y - companion_padding)
        x1 = min(width, x + box_width + companion_padding)
        y1 = min(height, y + box_height + companion_padding)
        nearer_pixels = int(np.count_nonzero(negative[y0:y1, x0:x1]))
        nearer_companion_ratio = nearer_pixels / area
        if (
            nearer_companion_ratio
            < MIN_LOW_CONTRAST_NEARER_COMPANION_RATIO
        ):
            continue

        component_mask = np.where(component, 255, 0).astype(np.uint8)
        nearby_negative_y, nearby_negative_x = np.where(
            negative[y0:y1, x0:x1]
        )
        negative_center_x = (
            x0 + float(np.mean(nearby_negative_x))
            if nearby_negative_x.size
            else x + box_width / 2
        )
        horizontal_expansion = box_width * 3
        if negative_center_x >= x + box_width / 2:
            output_x = x - round(box_width * 0.15)
        else:
            output_x = x + box_width - horizontal_expansion + round(
                box_width * 0.15
            )
        output_x = max(0, min(width - 1, output_x))
        output_width = min(width - output_x, horizontal_expansion)
        vertical_padding = round(box_height * 0.20)
        output_y = max(0, y - vertical_padding)
        output_bottom = min(height, y + box_height + vertical_padding)
        output_bbox = [
            output_x,
            output_y,
            output_width,
            output_bottom - output_y,
        ]
        support = {
            "applicable": True,
            "accepted": True,
            "promoted": True,
            "low_contrast_promotion": True,
            "reason": (
                "low-contrast farther component with a nearby nearer-depth "
                "companion, consistent with a rear item sliding forward"
            ),
            "valid_pixels": area,
            "farther_pixels": area,
            "farther_ratio": 1.0,
            "nearer_companion_pixels": nearer_pixels,
            "nearer_companion_ratio": round(nearer_companion_ratio, 4),
            "mean_visual_difference": round(mean_difference, 2),
            "threshold_mm": DEPTH_CHANGE_THRESHOLD_MM,
            "depth_component_pixels": area,
            "required_depth_component_pixels": minimum_area,
            "component_aspect_ratio": round(aspect_ratio, 4),
            "component_bbox": bbox,
        }
        finding = INSPECT_API.Finding(
            bbox=output_bbox,
            center=[
                output_bbox[0] + output_bbox[2] // 2,
                output_bbox[1] + output_bbox[3] // 2,
            ],
            sources=["depth_rgb_low_contrast"],
            votes=1,
        )
        candidates.append(
            (
                abs(mean_difference - 30.0),
                -area,
                finding,
                support,
                component_mask,
            )
        )

    if not candidates:
        return None
    candidates.sort(key=lambda candidate: candidate[:2])
    _, _, finding, support, component_mask = candidates[0]
    return finding, support, component_mask


def _find_candidate_row(
    bbox: list[int],
    rows: list[Any],
) -> Any | None:
    if not rows or bbox[3] <= 0:
        return None
    y, height = int(bbox[1]), int(bbox[3])
    bottom = y + height
    return max(
        rows,
        key=lambda row: max(
            0,
            min(bottom, int(row.bbox[1]) + int(row.bbox[3]))
            - max(y, int(row.bbox[1])),
        ),
    )


def _candidate_row_overlap_ratio(bbox: list[int], row: Any) -> float:
    _, y, _, height = (int(value) for value in bbox)
    if height <= 0:
        return 0.0
    row_y = int(row.bbox[1])
    row_bottom = row_y + int(row.bbox[3])
    overlap = max(0, min(y + height, row_bottom) - max(y, row_y))
    return overlap / height


def _row_shelf_span(
    row: Any,
    rails: list[Any],
) -> tuple[int, int] | None:
    if not rails:
        return None
    rail_index = getattr(row, "lower_rail_index", None)
    if rail_index is None:
        row_top = int(row.bbox[1])
        candidates = [
            (index, abs(int(rail.y_center) - row_top))
            for index, rail in enumerate(rails)
            if int(rail.y_center) <= row_top + 8
        ]
        if not candidates:
            return None
        rail_index = min(candidates, key=lambda candidate: candidate[1])[0]
    if not 0 <= int(rail_index) < len(rails):
        return None
    x1, _, x2, _ = (int(value) for value in rails[int(rail_index)].line)
    return min(x1, x2), max(x1, x2) + 1


def _effective_row_shelf_span(
    row: Any,
    rails: list[Any],
    *,
    image_width: int,
    pose_type: str,
) -> tuple[int, int]:
    """Return the usable shelf span after removing perspective side clutter."""

    detected = _row_shelf_span(row, rails) or (0, image_width)
    pose = str(pose_type).upper()
    if pose.endswith("LOWER"):
        # LOWER views form a trapezoid.  Inset each detected rail rather than
        # applying a large image crop, so deeper rows naturally narrow with
        # the detected shelf front.
        inset = round(image_width * LOWER_SHELF_SIDE_INSET_RATIO)
        absolute_inset = round(image_width * LOWER_IMAGE_SIDE_CROP_RATIO)
        left = max(detected[0] + inset, absolute_inset)
        right = min(detected[1] - inset, image_width - absolute_inset)
    else:
        # UPPER views still include the neighbouring bay at both image edges.
        # A fixed inner frame removes those objects and the vertical uprights.
        absolute_inset = round(image_width * UPPER_SIDE_CROP_RATIO)
        left = max(detected[0], absolute_inset)
        right = min(detected[1], image_width - absolute_inset)
    if right <= left:
        return max(0, detected[0]), min(image_width, detected[1])
    return max(0, left), min(image_width, right)


def _build_shelf_roi_mask(
    shape: tuple[int, int],
    rows: list[Any],
    rails: list[Any],
    pose_type: str,
) -> np.ndarray:
    """Build the row-wise trapezoid used before RGB/depth candidate extraction."""

    height, width = shape
    if not rows:
        return np.full((height, width), 255, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    for row in rows:
        row_y = max(0, int(row.bbox[1]))
        row_bottom = min(height, row_y + int(row.bbox[3]))
        left, right = _effective_row_shelf_span(
            row,
            rails,
            image_width=width,
            pose_type=pose_type,
        )
        if row_bottom > row_y and right > left:
            mask[row_y:row_bottom, left:right] = 255
    return mask


def _horizontal_overlap_ratio(
    bbox: list[int],
    span: tuple[int, int] | None,
) -> float | None:
    if span is None:
        return None
    x, _, width, _ = bbox
    left, right = span
    overlap = max(0, min(x + width, right) - max(x, left))
    return overlap / max(1, width)


def _baseline_floor_similarity_ratio(
    baseline_image: np.ndarray | None,
    bbox: list[int],
    row: Any,
    shelf_span: tuple[int, int] | None,
) -> float | None:
    """Estimate how much the baseline candidate already resembles bare shelf."""

    if baseline_image is None or shelf_span is None:
        return None
    image_height, image_width = baseline_image.shape[:2]
    x, y, width, height = bbox
    x0 = max(0, min(image_width, x))
    y0 = max(0, min(image_height, y))
    x1 = max(x0, min(image_width, x + width))
    y1 = max(y0, min(image_height, y + height))
    if x1 <= x0 or y1 <= y0:
        return None

    row_y = max(0, int(row.bbox[1]))
    row_bottom = min(image_height, row_y + int(row.bbox[3]))
    band_height = max(8, min(30, round(int(row.bbox[3]) * 0.12)))
    band_y0 = max(row_y, row_bottom - band_height - 4)
    band_y1 = max(band_y0 + 1, row_bottom - 4)
    span_left = max(0, min(image_width, shelf_span[0]))
    span_right = max(span_left, min(image_width, shelf_span[1]))
    reference = baseline_image[band_y0:band_y1, span_left:span_right]
    candidate = baseline_image[y0:y1, x0:x1]
    if reference.size == 0 or candidate.size == 0:
        return None

    reference_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    reference_edges = cv2.Canny(reference_gray, 35, 90)
    neutral_smooth = (
        (reference_hsv[:, :, 1] < 70)
        & (reference_hsv[:, :, 2] > 35)
        & (reference_edges == 0)
    )
    if int(np.count_nonzero(neutral_smooth)) < 50:
        return None
    shelf_color = np.median(reference_lab[neutral_smooth], axis=0)
    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB).astype(np.float32)
    distance = np.linalg.norm(candidate_lab - shelf_color, axis=2)
    return float(np.count_nonzero(distance < 18.0)) / distance.size


def merge_fragmented_candidates(
    pairs: list[tuple[Any, dict[str, Any], np.ndarray]],
    rows: list[Any],
) -> list[tuple[Any, dict[str, Any], np.ndarray]]:
    """Join vertically split depth pieces belonging to the same product."""

    merged = list(pairs)
    changed = True
    while changed:
        changed = False
        for first_index in range(len(merged)):
            first_finding, first_support, first_mask = merged[first_index]
            first_bbox = [int(value) for value in first_finding.bbox]
            first_row = _find_candidate_row(first_bbox, rows)
            if first_row is None:
                continue
            for second_index in range(first_index + 1, len(merged)):
                second_finding, second_support, second_mask = merged[second_index]
                second_bbox = [int(value) for value in second_finding.bbox]
                second_row = _find_candidate_row(second_bbox, rows)
                if second_row is not first_row:
                    continue
                first_x, first_y, first_width, first_height = first_bbox
                second_x, second_y, second_width, second_height = second_bbox
                horizontal_overlap = max(
                    0,
                    min(first_x + first_width, second_x + second_width)
                    - max(first_x, second_x),
                )
                horizontal_overlap_ratio = horizontal_overlap / max(
                    1,
                    min(first_width, second_width),
                )
                vertical_gap = max(
                    0,
                    max(first_y, second_y)
                    - min(first_y + first_height, second_y + second_height),
                )
                maximum_gap = round(
                    int(first_row.bbox[3]) * MAX_FRAGMENT_VERTICAL_GAP_RATIO
                )
                if (
                    horizontal_overlap_ratio
                    < MIN_FRAGMENT_HORIZONTAL_OVERLAP_RATIO
                    or vertical_gap > maximum_gap
                ):
                    continue

                left = min(first_x, second_x)
                top = min(first_y, second_y)
                right = max(first_x + first_width, second_x + second_width)
                bottom = max(first_y + first_height, second_y + second_height)
                bbox = [left, top, right - left, bottom - top]
                sources = list(
                    dict.fromkeys(
                        [*first_finding.sources, *second_finding.sources]
                    )
                )
                finding = INSPECT_API.Finding(
                    bbox=bbox,
                    center=[left + (right - left) // 2, top + (bottom - top) // 2],
                    sources=sources,
                    votes=max(first_finding.votes, second_finding.votes),
                )
                support = {
                    **first_support,
                    "merged_fragments": [first_bbox, second_bbox],
                    "merged_fragment_support": second_support,
                }
                region_mask = cv2.bitwise_or(first_mask, second_mask)
                merged[first_index] = (finding, support, region_mask)
                merged.pop(second_index)
                changed = True
                break
            if changed:
                break
    return merged


def recover_closed_depth_candidate(
    *,
    execution: Any,
    baseline_image: np.ndarray | None,
    depth_change_mask: np.ndarray,
    baseline_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    rows: list[Any],
    rails: list[Any],
) -> tuple[Any, dict[str, Any], np.ndarray] | None:
    """Recover one fragmented shelf-edge hole after all regular candidates fail."""

    if baseline_image is None or not rows:
        return None
    height, width = depth_change_mask.shape
    radius = max(2, round(min(height, width) * RECOVERY_CLOSE_RATIO / 2))
    closed = cv2.morphologyEx(
        (depth_change_mask > 0).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        ),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed,
        8,
    )
    valid = (
        np.isfinite(baseline_depth_mm)
        & np.isfinite(current_depth_mm)
        & (baseline_depth_mm > 0)
        & (current_depth_mm > 0)
    )
    delta = current_depth_mm - baseline_depth_mm
    minimum_area = max(
        50,
        round(depth_change_mask.size * MIN_RECOVERY_COMPONENT_AREA_RATIO),
    )
    candidates: list[
        tuple[float, Any, dict[str, Any], np.ndarray]
    ] = []
    for label in range(1, component_count):
        x, y, box_width, box_height, area = (
            int(value) for value in stats[label]
        )
        if area < minimum_area:
            continue
        bbox = [x, y, box_width, box_height]
        row = _find_candidate_row(bbox, rows)
        if row is None:
            continue
        row_y = int(row.bbox[1])
        row_height = max(1, int(row.bbox[3]))
        row_bottom = row_y + row_height
        vertical_overlap = max(
            0,
            min(y + box_height, row_bottom) - max(y, row_y),
        ) / max(1, box_height)
        if vertical_overlap < MIN_DEPTH_PROMOTION_ROW_OVERLAP:
            continue
        shelf_span = _row_shelf_span(row, rails)
        shelf_overlap = _horizontal_overlap_ratio(bbox, shelf_span)
        if (
            shelf_overlap is not None
            and shelf_overlap < MIN_OUTWARD_SHELF_OVERLAP_RATIO
        ):
            continue
        if box_width / row_height < MIN_RECOVERY_WIDTH_TO_ROW_HEIGHT_RATIO:
            continue
        aspect_ratio = max(
            box_width / max(1, box_height),
            box_height / max(1, box_width),
        )
        if aspect_ratio > DEPTH_PROMOTION_MAX_ASPECT_RATIO:
            continue
        local_x0 = max(0, x - round(box_width * 1.5))
        local_x1 = min(width, x + box_width + round(box_width * 1.5))
        local_y0 = max(row_y, y - round(box_height * 0.15))
        local_y1 = min(row_bottom, y + box_height + round(box_height * 0.15))
        local_valid = valid[local_y0:local_y1, local_x0:local_x1]
        local_delta = delta[local_y0:local_y1, local_x0:local_x1]
        farther_pixels = int(
            np.count_nonzero(local_valid & (local_delta > DEPTH_CHANGE_THRESHOLD_MM))
        )
        nearer_pixels = int(
            np.count_nonzero(local_valid & (local_delta < -DEPTH_CHANGE_THRESHOLD_MM))
        )
        movement_balance = min(farther_pixels, nearer_pixels) / max(
            1,
            max(farther_pixels, nearer_pixels),
        )
        baseline_crop = baseline_image[y : y + box_height, x : x + box_width]
        current_crop = execution.review_image[y : y + box_height, x : x + box_width]
        if baseline_crop.size == 0 or current_crop.size == 0:
            continue
        visual_difference = float(
            np.mean(cv2.absdiff(baseline_crop, current_crop))
        )
        if not (
            MIN_RECOVERY_VISUAL_DIFFERENCE
            <= visual_difference
            <= MAX_RECOVERY_VISUAL_DIFFERENCE
        ):
            continue

        component = labels == label
        region_mask = np.where(component, 255, 0).astype(np.uint8)
        support = {
            "applicable": True,
            "accepted": True,
            "promoted": True,
            "closed_depth_recovery": True,
            "reason": "fragmented farther-depth silhouette inside shelf span",
            "depth_component_pixels": area,
            "movement_balance_ratio": round(movement_balance, 4),
            "mean_visual_difference": round(visual_difference, 2),
            "shelf_overlap_ratio": (
                round(shelf_overlap, 4) if shelf_overlap is not None else None
            ),
        }
        finding = INSPECT_API.Finding(
            bbox=bbox,
            center=[x + box_width // 2, y + box_height // 2],
            sources=["depth_closed_recovery"],
            votes=1,
        )
        score = (
            area
            * (1.0 - movement_balance)
            * min(2.0, visual_difference / 45.0)
        )
        candidates.append((score, finding, support, region_mask))
    if not candidates:
        return None
    _, finding, support, region_mask = max(candidates, key=lambda item: item[0])
    return finding, support, region_mask


def filter_shelf_interference_candidates(
    pairs: list[tuple[Any, dict[str, Any], np.ndarray]],
    *,
    execution: Any,
    baseline_image: np.ndarray | None,
    baseline_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    rows: list[Any],
    rails: list[Any],
) -> tuple[
    list[tuple[Any, dict[str, Any], np.ndarray]],
    list[dict[str, Any]],
]:
    """Reject shelf-edge, bare-floor, and moved-item depth artifacts."""

    if not pairs or not rows:
        return pairs, []
    image_height, image_width = baseline_depth_mm.shape
    valid = (
        np.isfinite(baseline_depth_mm)
        & np.isfinite(current_depth_mm)
        & (baseline_depth_mm > 0)
        & (current_depth_mm > 0)
    )
    delta = current_depth_mm - baseline_depth_mm
    location_id = str(execution.response.location_id).upper()
    pose_type = str(execution.response.pose_type)
    is_back_left_inspection = "_B_L_INSPECT" in location_id
    is_back_right_inspection = "_B_R_INSPECT" in location_id
    is_front_left_inspection = "_F_L_INSPECT" in location_id
    is_front_right_inspection = "_F_R_INSPECT" in location_id
    is_h2_front_right_inspection = "H2_F_R_INSPECT" in location_id
    lower_view = pose_type.upper().endswith("LOWER")
    kept: list[tuple[Any, dict[str, Any], np.ndarray]] = []
    rejected: list[dict[str, Any]] = []

    for finding, support, region_mask in pairs:
        bbox = [int(value) for value in finding.bbox]
        x, y, width, height = bbox
        row = _find_candidate_row(bbox, rows)
        if row is None:
            kept.append((finding, support, region_mask))
            continue
        row_y = int(row.bbox[1])
        row_height = max(1, int(row.bbox[3]))
        row_bottom = min(image_height, row_y + row_height)
        row_index = int(getattr(row, "index", rows.index(row) + 1))
        row_overlap = _candidate_row_overlap_ratio(bbox, row)
        shelf_span = _effective_row_shelf_span(
            row,
            rails,
            image_width=image_width,
            pose_type=pose_type,
        )
        shelf_overlap = _horizontal_overlap_ratio(bbox, shelf_span)
        open_ended_row = getattr(row, "lower_rail_index", None) is None

        local_x0 = max(0, x - round(width * 1.5))
        local_x1 = min(image_width, x + width + round(width * 1.5))
        local_y0 = max(row_y, y - round(height * 0.15))
        local_y1 = min(row_bottom, y + height + round(height * 0.15))
        local_valid = valid[local_y0:local_y1, local_x0:local_x1]
        local_delta = delta[local_y0:local_y1, local_x0:local_x1]
        farther_pixels = int(
            np.count_nonzero(local_valid & (local_delta > DEPTH_CHANGE_THRESHOLD_MM))
        )
        nearer_pixels = int(
            np.count_nonzero(local_valid & (local_delta < -DEPTH_CHANGE_THRESHOLD_MM))
        )
        movement_balance = min(farther_pixels, nearer_pixels) / max(
            1,
            max(farther_pixels, nearer_pixels),
        )

        expected_width = max(width, round(row_height * MIN_OBJECT_WINDOW_WIDTH_RATIO))
        expected_height = max(
            height,
            round(row_height * MIN_OBJECT_WINDOW_HEIGHT_RATIO),
        )
        center_x = x + width / 2
        center_y = y + height / 2
        window_x0 = max(0, round(center_x - expected_width / 2))
        window_x1 = min(image_width, window_x0 + expected_width)
        window_y0 = max(row_y, round(center_y - expected_height / 2))
        window_y1 = min(row_bottom, window_y0 + expected_height)
        window_valid = valid[window_y0:window_y1, window_x0:window_x1]
        window_delta = delta[window_y0:window_y1, window_x0:window_x1]
        window_valid_pixels = int(np.count_nonzero(window_valid))
        window_farther_pixels = int(
            np.count_nonzero(
                window_valid & (window_delta > DEPTH_CHANGE_THRESHOLD_MM)
            )
        )
        window_farther_ratio = window_farther_pixels / max(1, window_valid_pixels)
        floor_similarity = _baseline_floor_similarity_ratio(
            baseline_image,
            bbox,
            row,
            shelf_span,
        )
        width_ratio = width / row_height
        height_ratio = height / row_height
        stacked_column_depth_case = (
            is_back_left_inspection
            and lower_view
            and row_index >= 2
            and height_ratio >= MIN_STACKED_COLUMN_HEIGHT_RATIO
            and width_ratio >= MIN_STACKED_COLUMN_WIDTH_RATIO
            and window_farther_ratio >= MIN_STACKED_COLUMN_FARTHER_RATIO
        )

        reasons: list[str] = []
        if row_overlap < MIN_CANDIDATE_ROW_OVERLAP_RATIO:
            reasons.append("candidate is mostly on a shelf rail or outside its row")
        if (
            shelf_overlap is not None
            and shelf_overlap < MIN_OUTWARD_SHELF_OVERLAP_RATIO
        ):
            reasons.append("candidate lies outside the cropped shelf interior")
        if is_back_left_inspection:
            if width_ratio < MIN_BACK_LEFT_CANDIDATE_WIDTH_RATIO:
                reasons.append("too narrow to contain a baseline product")
            if (
                height_ratio < MIN_BACK_LEFT_CANDIDATE_HEIGHT_RATIO
                and movement_balance > 0.20
            ):
                reasons.append("shallow package-edge displacement")
            if (
                movement_balance > MAX_BACK_LEFT_MOVEMENT_BALANCE_RATIO
                and not stacked_column_depth_case
            ):
                reasons.append("balanced signed depth indicates item movement")
            if (
                bool(support.get("refined"))
                and height_ratio < 0.35
                and movement_balance > 0.25
                and window_farther_ratio < 0.36
            ):
                reasons.append("shallow back-left parallax fragment")
            if (
                lower_view
                and row_index >= 2
                and window_farther_ratio < 0.12
                and nearer_pixels > farther_pixels * 4
            ):
                reasons.append("back-left lower-row foreground edge")
        if (
            is_back_right_inspection
            and x + width >= shelf_span[1] - round(image_width * 0.03)
            and width_ratio < 0.20
        ):
            reasons.append("narrow fragment at the back-right shelf edge")
        if (
            is_front_right_inspection
            and row_index == 1
            and width_ratio < MAX_FRONT_RIGHT_TOP_FRAGMENT_WIDTH_RATIO
            and height_ratio < MAX_FRONT_RIGHT_TOP_FRAGMENT_HEIGHT_RATIO
        ):
            reasons.append("small split edge above the complete product scale")
        if is_front_left_inspection:
            if (
                shelf_overlap is not None
                and shelf_overlap < MIN_OUTWARD_SHELF_OVERLAP_RATIO
            ):
                reasons.append("candidate extends beyond the detected shelf span")
            if (
                len(pairs) > 1
                and width_ratio < 0.20
                and movement_balance > 0.30
            ):
                reasons.append("narrow fragment beside a stronger shelf candidate")
            if (
                len(pairs) > 1
                and bool(support.get("promoted"))
                and movement_balance > 0.60
            ):
                reasons.append("promoted fragment is dominated by object motion")
            if (
                lower_view
                and bool(support.get("rgb_fallback"))
                and window_farther_ratio < 0.06
            ):
                reasons.append("front-left RGB shift has no depth-hole support")
            if (
                not lower_view
                and row_index >= 2
                and width_ratio < 0.15
                and height_ratio < 0.35
                and floor_similarity is not None
                and floor_similarity >= 0.10
            ):
                reasons.append("narrow lower-row shelf-floor fragment")
        if is_h2_front_right_inspection and lower_view and row_index >= 3:
            if width_ratio < 0.30:
                reasons.append("narrow H2 front-right bottle-edge fragment")
            if (
                movement_balance > 0.55
                and window_farther_ratio < 0.25
                and floor_similarity is not None
                and floor_similarity >= 0.10
            ):
                reasons.append("H2 front-right bottle displacement")

        bottom_rows_need_floor_filter = (
            row_index >= 2
            and (
                open_ended_row
                or (is_front_right_inspection and row_index >= 2)
            )
            and not bool(support.get("low_contrast_promotion"))
        )
        if bottom_rows_need_floor_filter:
            if (
                floor_similarity is not None
                and floor_similarity >= MIN_FLOOR_SIMILARITY_RATIO
                and window_farther_ratio < MAX_FLOOR_LIKE_FARTHER_WINDOW_RATIO
            ):
                reasons.append("baseline bbox is dominated by shelf-floor pixels")
        if (
            is_back_right_inspection
            and open_ended_row
            and floor_similarity is not None
            and floor_similarity >= 0.10
            and window_farther_ratio < 0.60
        ):
            reasons.append("back-view open-row candidate resembles shelf floor")

        metrics = {
            "accepted": not reasons,
            "row_index": row_index,
            "row_overlap_ratio": round(row_overlap, 4),
            "open_ended_row": open_ended_row,
            "shelf_span": list(shelf_span) if shelf_span is not None else None,
            "shelf_overlap_ratio": (
                round(shelf_overlap, 4) if shelf_overlap is not None else None
            ),
            "bbox_width_to_row_height_ratio": round(width_ratio, 4),
            "bbox_height_to_row_height_ratio": round(height_ratio, 4),
            "nearer_pixels": nearer_pixels,
            "farther_pixels": farther_pixels,
            "movement_balance_ratio": round(movement_balance, 4),
            "object_window_farther_ratio": round(window_farther_ratio, 4),
            "stacked_column_depth_case": stacked_column_depth_case,
            "baseline_floor_similarity_ratio": (
                round(floor_similarity, 4)
                if floor_similarity is not None
                else None
            ),
            "reasons": reasons,
        }
        updated_support = {**support, "shelf_interference_filter": metrics}
        if reasons:
            rejected.append({"bbox": bbox, **metrics})
        else:
            clipped_left = max(x, shelf_span[0])
            clipped_top = max(y, row_y)
            clipped_right = min(x + width, shelf_span[1])
            clipped_bottom = min(y + height, row_bottom)
            clipped_bbox = [
                clipped_left,
                clipped_top,
                max(0, clipped_right - clipped_left),
                max(0, clipped_bottom - clipped_top),
            ]
            if clipped_bbox[2] <= 0 or clipped_bbox[3] <= 0:
                rejected.append(
                    {
                        "bbox": bbox,
                        **metrics,
                        "accepted": False,
                        "reasons": ["candidate is empty after shelf-row clipping"],
                    }
                )
                continue
            clipped_mask = clipped_region_mask(region_mask, clipped_bbox)
            clipped_finding = finding.model_copy(
                update={
                    "bbox": clipped_bbox,
                    "center": [
                        clipped_left + clipped_bbox[2] // 2,
                        clipped_top + clipped_bbox[3] // 2,
                    ],
                }
            )
            if clipped_bbox != bbox:
                updated_support = {
                    **updated_support,
                    "shelf_row_clip": {
                        "original_bbox": bbox,
                        "clipped_bbox": clipped_bbox,
                    },
                }
            kept.append((clipped_finding, updated_support, clipped_mask))
    return kept, rejected


def _scale_fixed_layout_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> list[int]:
    image_height, image_width = image_shape
    reference_width, reference_height = FIXED_LAYOUT_REFERENCE_SIZE
    x, y, width, height = bbox
    left = max(0, min(image_width, round(x * image_width / reference_width)))
    top = max(0, min(image_height, round(y * image_height / reference_height)))
    right = max(
        left,
        min(image_width, round((x + width) * image_width / reference_width)),
    )
    bottom = max(
        top,
        min(image_height, round((y + height) * image_height / reference_height)),
    )
    return [left, top, right - left, bottom - top]


def promote_fixed_layout_depth_slot(
    *,
    execution: Any,
    photometric_mask: np.ndarray,
    baseline_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
) -> tuple[Any, dict[str, Any], np.ndarray] | None:
    """Recover count shortages hidden behind the front item in fixed poses.

    These two task0 views contain product groups where removing a rear item
    does not reveal an empty RGB rectangle.  A slot-level depth statistic is
    much more stable than lowering the component threshold for every shelf.
    """

    location_id = str(execution.response.location_id).upper()
    pose_type = str(execution.response.pose_type).upper()
    if not pose_type.endswith("UPPER"):
        return None
    image_shape = baseline_depth_mm.shape
    valid_depth = (
        np.isfinite(baseline_depth_mm)
        & np.isfinite(current_depth_mm)
        & (baseline_depth_mm > 0)
        & (current_depth_mm > 0)
    )
    delta = current_depth_mm - baseline_depth_mm
    candidates: list[
        tuple[float, int, list[int], np.ndarray, dict[str, Any]]
    ] = []

    if location_id == "H2_B_L_INSPECT":
        for slot_index, (detection_roi, output_roi) in enumerate(
            H2_BACK_LEFT_UPPER_DEPTH_SLOTS,
            start=1,
        ):
            x, y, width, height = _scale_fixed_layout_bbox(
                detection_roi,
                image_shape,
            )
            if width <= 0 or height <= 0:
                continue
            region_valid = valid_depth[y : y + height, x : x + width]
            valid_pixels = int(np.count_nonzero(region_valid))
            if valid_pixels < MIN_DEPTH_VALID_PIXELS:
                continue
            region_delta = delta[y : y + height, x : x + width]
            farther = region_valid & (
                region_delta > H2_BACK_LEFT_UPPER_SLOT_DEPTH_THRESHOLD_MM
            )
            farther_pixels = int(np.count_nonzero(farther))
            farther_ratio = farther_pixels / valid_pixels
            if farther_ratio < MIN_H2_BACK_LEFT_UPPER_SLOT_FARTHER_RATIO:
                continue

            output_bbox = _scale_fixed_layout_bbox(output_roi, image_shape)
            region_mask = np.zeros(image_shape, dtype=np.uint8)
            region_mask[y : y + height, x : x + width][farther] = 255
            region_mask = cv2.bitwise_or(
                region_mask,
                clipped_region_mask(photometric_mask, output_bbox),
            )
            support = {
                "applicable": True,
                "accepted": True,
                "promoted": True,
                "fixed_layout_depth_promotion": True,
                "fixed_layout": "H2_B_L_INSPECT_UPPER",
                "slot_index": slot_index,
                "detection_roi": [x, y, width, height],
                "valid_pixels": valid_pixels,
                "farther_pixels": farther_pixels,
                "slot_farther_ratio": round(farther_ratio, 4),
                "threshold_mm": H2_BACK_LEFT_UPPER_SLOT_DEPTH_THRESHOLD_MM,
                "reason": "fixed-slot farther-depth ratio indicates a hidden count shortage",
            }
            candidates.append(
                (farther_ratio, slot_index, output_bbox, region_mask, support)
            )

    elif location_id == "H2_F_L_INSPECT":
        for slot_index, (slot_roi, calibrated_threshold) in enumerate(
            H2_FRONT_LEFT_UPPER_DEPTH_SLOTS,
            start=1,
        ):
            x, y, width, height = _scale_fixed_layout_bbox(
                slot_roi,
                image_shape,
            )
            if width <= 0 or height <= 0:
                continue
            region_valid = valid_depth[y : y + height, x : x + width]
            valid_pixels = int(np.count_nonzero(region_valid))
            if valid_pixels < MIN_DEPTH_VALID_PIXELS:
                continue
            baseline_region = baseline_depth_mm[y : y + height, x : x + width]
            current_region = current_depth_mm[y : y + height, x : x + width]
            baseline_near = float(np.percentile(baseline_region[region_valid], 5))
            current_near = float(np.percentile(current_region[region_valid], 5))

            strip_valid = valid_depth[:, x : x + width]
            strip_delta = delta[:, x : x + width][strip_valid]
            stable_delta = strip_delta[
                (strip_delta > -200.0) & (strip_delta < 200.0)
            ]
            if stable_delta.size < MIN_DEPTH_VALID_PIXELS:
                continue
            local_depth_offset = float(np.median(stable_delta))
            near_depth_residual = (
                current_near - baseline_near - local_depth_offset
            )
            slot_score = near_depth_residual - calibrated_threshold
            rgb_pixels = int(
                np.count_nonzero(photometric_mask[y : y + height, x : x + width])
            )
            rgb_ratio = rgb_pixels / max(1, width * height)
            if (
                slot_score < MIN_H2_FRONT_LEFT_UPPER_SLOT_SCORE_MM
                or rgb_ratio < MIN_H2_FRONT_LEFT_UPPER_SLOT_RGB_RATIO
            ):
                continue

            normalized_depth_hole = region_valid & (
                current_region - baseline_region - local_depth_offset > 30.0
            )
            region_mask = clipped_region_mask(
                photometric_mask,
                [x, y, width, height],
            )
            depth_mask = np.zeros(image_shape, dtype=np.uint8)
            depth_mask[y : y + height, x : x + width][normalized_depth_hole] = 255
            region_mask = cv2.bitwise_or(region_mask, depth_mask)
            region_mask = cv2.morphologyEx(
                region_mask,
                cv2.MORPH_CLOSE,
                np.ones((5, 5), dtype=np.uint8),
            )
            output_bbox = [x, y, width, height]
            support = {
                "applicable": True,
                "accepted": True,
                "promoted": True,
                "fixed_layout_depth_promotion": True,
                "fixed_layout": "H2_F_L_INSPECT_UPPER",
                "slot_index": slot_index,
                "detection_roi": output_bbox,
                "valid_pixels": valid_pixels,
                "rgb_pixels": rgb_pixels,
                "rgb_ratio": round(rgb_ratio, 4),
                "baseline_near_depth_mm": round(baseline_near, 2),
                "current_near_depth_mm": round(current_near, 2),
                "local_depth_offset_mm": round(local_depth_offset, 2),
                "near_depth_residual_mm": round(near_depth_residual, 2),
                "calibrated_threshold_mm": calibrated_threshold,
                "slot_score_mm": round(slot_score, 2),
                "reason": "fixed-slot near-depth distribution indicates a hidden rear-item shortage",
            }
            candidates.append(
                (slot_score, slot_index, output_bbox, region_mask, support)
            )

    if not candidates:
        return None
    _, _, bbox, region_mask, support = max(candidates, key=lambda item: item[0])
    x, y, width, height = bbox
    finding = INSPECT_API.Finding(
        bbox=bbox,
        center=[x + width // 2, y + height // 2],
        sources=["fixed_layout_depth_fusion"],
        votes=1,
    )
    return finding, support, region_mask


def filter_execution_with_depth(
    execution: Any,
    baseline_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    baseline_image: np.ndarray | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any], np.ndarray]:
    """Reject illumination-only RGB regions when aligned depth contradicts shortage."""

    review_shape = execution.review_image.shape[:2]
    baseline_aligned = align_depth_to_review(baseline_depth_mm, review_shape)
    homography = getattr(execution, "review_homography", None)
    if homography is None:
        return (
            execution,
            [
                {
                    "applicable": False,
                    "accepted": True,
                    "reason": "RGB alignment homography unavailable",
                }
                for _ in execution.response.findings
            ],
            {
                "applied": False,
                "input_findings": len(execution.response.findings),
                "kept_findings": len(execution.response.findings),
                "rejected_findings": 0,
                "reason": "RGB alignment homography unavailable",
            },
            np.zeros(review_shape, dtype=np.uint8),
        )
    current_aligned = align_depth_to_review(
        current_depth_mm,
        review_shape,
        homography,
    )
    raw_depth_change_mask = np.where(
        np.isfinite(baseline_aligned)
        & np.isfinite(current_aligned)
        & (baseline_aligned > 0)
        & (current_aligned > 0)
        & ((current_aligned - baseline_aligned) > DEPTH_CHANGE_THRESHOLD_MM),
        255,
        0,
    ).astype(np.uint8)
    promotion_error: str | None = None
    row_bboxes: list[list[int]] = []
    open_ended_row_bboxes: list[list[int]] = []
    detected_rows: list[Any] = []
    detected_rails: list[Any] = []
    try:
        row_detection = INSPECT_API.detect_rows(
            execution.review_image,
            INSPECT_API.RowDetectionConfig(
                target_size=None,
                pose_type=execution.response.pose_type,
            ),
        )
        detected_rows = list(row_detection.rows)
        detected_rails = list(getattr(row_detection, "rails", []))
        for row in detected_rows:
            row_bbox = list(row.bbox)
            row_bboxes.append(row_bbox)
            if getattr(row, "lower_rail_index", None) is None:
                open_ended_row_bboxes.append(row_bbox)
    except (ValueError, cv2.error) as error:
        promotion_error = f"{type(error).__name__}: {error}"
    shelf_roi_mask = _build_shelf_roi_mask(
        review_shape,
        detected_rows,
        detected_rails,
        str(execution.response.pose_type),
    )
    photometric_mask = cv2.bitwise_and(execution.review_mask, shelf_roi_mask)
    depth_change_mask = cv2.bitwise_and(raw_depth_change_mask, shelf_roi_mask)
    working_execution = replace(execution, review_mask=photometric_mask)
    kept_pairs, accepted_input_count = refine_findings_with_signed_depth(
        photometric_mask,
        depth_change_mask,
        baseline_aligned,
        current_aligned,
        list(execution.response.findings),
    )
    # A large, RGB-supported depth hole can be the real shortage even when the
    # global RGB detector was distracted by a different shelf row.  Only let a
    # dominant hole override those coarse candidates; weaker depth edges stay
    # out of the result.
    promotion_candidates = promote_depth_components(
        photometric_mask,
        depth_change_mask,
        baseline_aligned,
        current_aligned,
        row_bboxes,
        open_ended_row_bboxes,
        [finding for finding, _, _ in kept_pairs],
    )
    minimum_dominant_area = max(
        50,
        round(
            depth_change_mask.size
            * MIN_DOMINANT_DEPTH_PROMOTION_AREA_RATIO
        ),
    )
    promotions = [
        promotion
        for promotion in promotion_candidates
        if int(promotion[1].get("depth_component_pixels", 0))
        >= minimum_dominant_area
        and float(promotion[1].get("rgb_evidence_ratio", 0.0))
        >= MIN_DOMINANT_DEPTH_PROMOTION_RGB_RATIO
    ]
    kept_pairs.extend(promotions)
    low_contrast_promotion_count = 0
    if (
        not kept_pairs
        and not execution.response.findings
        and baseline_image is not None
    ):
        low_contrast_promotion = promote_low_contrast_depth_component(
            baseline_image,
            execution.review_image,
            depth_change_mask,
            baseline_aligned,
            current_aligned,
            row_bboxes,
            open_ended_row_bboxes,
        )
        if low_contrast_promotion is not None:
            kept_pairs.append(low_contrast_promotion)
            promotions.append(low_contrast_promotion)
            low_contrast_promotion_count = 1
    had_depth_candidates_before_shelf_filter = bool(kept_pairs)
    kept_pairs = merge_fragmented_candidates(kept_pairs, detected_rows)
    kept_pairs, shelf_interference_rejections = (
        filter_shelf_interference_candidates(
            kept_pairs,
            execution=execution,
            baseline_image=baseline_image,
            baseline_depth_mm=baseline_aligned,
            current_depth_mm=current_aligned,
            rows=detected_rows,
            rails=detected_rails,
        )
    )
    fixed_layout_depth_promotion_count = 0
    fixed_layout_promotion = promote_fixed_layout_depth_slot(
        execution=working_execution,
        photometric_mask=photometric_mask,
        baseline_depth_mm=baseline_aligned,
        current_depth_mm=current_aligned,
    )
    if fixed_layout_promotion is not None:
        kept_pairs = [fixed_layout_promotion]
        fixed_layout_depth_promotion_count = 1
    rgb_fallback_count = 0
    location_id = str(execution.response.location_id).upper()
    allow_rgb_fallback = (
        "_F_" in location_id or "_INSPECT" not in location_id
    )
    if (
        not kept_pairs
        and allow_rgb_fallback
        and not had_depth_candidates_before_shelf_filter
    ):
        rgb_fallback = select_rgb_fallback_finding(
            working_execution,
            baseline_aligned,
            current_aligned,
            detected_rows,
            detected_rails,
        )
        if rgb_fallback is not None:
            fallback_pairs, fallback_rejections = (
                filter_shelf_interference_candidates(
                    [rgb_fallback],
                    execution=working_execution,
                    baseline_image=baseline_image,
                    baseline_depth_mm=baseline_aligned,
                    current_depth_mm=current_aligned,
                    rows=detected_rows,
                    rails=detected_rails,
                )
            )
            kept_pairs.extend(fallback_pairs)
            shelf_interference_rejections.extend(fallback_rejections)
            if fallback_pairs:
                accepted_input_count = 1
                rgb_fallback_count = 1
    closed_depth_recovery_count = 0
    if (
        not kept_pairs
        and "_F_" in location_id
        and not execution.response.findings
    ):
        recovered = recover_closed_depth_candidate(
            execution=working_execution,
            baseline_image=baseline_image,
            depth_change_mask=depth_change_mask,
            baseline_depth_mm=baseline_aligned,
            current_depth_mm=current_aligned,
            rows=detected_rows,
            rails=detected_rails,
        )
        if recovered is not None:
            recovery_pairs, recovery_rejections = (
                filter_shelf_interference_candidates(
                    [recovered],
                    execution=working_execution,
                    baseline_image=baseline_image,
                    baseline_depth_mm=baseline_aligned,
                    current_depth_mm=current_aligned,
                    rows=detected_rows,
                    rails=detected_rails,
                )
            )
            kept_pairs.extend(recovery_pairs)
            shelf_interference_rejections.extend(recovery_rejections)
            if recovery_pairs:
                closed_depth_recovery_count = 1
    accepted_refined_count = sum(
        1 for _, support, _ in kept_pairs if bool(support.get("refined"))
    )
    accepted_promotion_count = sum(
        1 for _, support, _ in kept_pairs if bool(support.get("promoted"))
    )
    accepted_low_contrast_count = sum(
        1
        for _, support, _ in kept_pairs
        if bool(support.get("low_contrast_promotion"))
    )
    accepted_rgb_fallback_count = sum(
        1 for _, support, _ in kept_pairs if bool(support.get("rgb_fallback"))
    )
    kept_pairs.sort(key=lambda pair: (pair[0].bbox[1], pair[0].bbox[0]))
    kept_findings = [finding for finding, _, _ in kept_pairs]
    kept_supports = [support for _, support, _ in kept_pairs]
    filtered_mask = np.zeros(execution.review_mask.shape, dtype=np.uint8)
    for _, _, region_mask in kept_pairs:
        filtered_mask = cv2.bitwise_or(
            filtered_mask,
            region_mask,
        )
    response = execution.response.model_copy(
        update={
            "findings": kept_findings,
            "has_anomaly": bool(kept_findings),
        }
    )
    filtered = replace(
        execution,
        response=response,
        review_mask=filtered_mask,
    )
    return (
        filtered,
        kept_supports,
        {
            "applied": True,
            "input_findings": len(execution.response.findings),
            "kept_findings": len(kept_findings),
            "rejected_findings": (
                len(execution.response.findings) - accepted_input_count
            ),
            "refined_findings": (
                accepted_refined_count
            ),
            "promoted_findings": accepted_promotion_count,
            "low_contrast_promoted_findings": (
                accepted_low_contrast_count
            ),
            "rgb_fallback_findings": accepted_rgb_fallback_count,
            "shelf_interference_rejected_findings": len(
                shelf_interference_rejections
            ),
            "shelf_interference_rejections": shelf_interference_rejections,
            "closed_depth_recovery_findings": closed_depth_recovery_count,
            "fixed_layout_depth_promoted_findings": (
                fixed_layout_depth_promotion_count
            ),
            "promotion_row_count": len(row_bboxes),
            "promotion_error": promotion_error,
            "depth_change_threshold_mm": DEPTH_CHANGE_THRESHOLD_MM,
            "minimum_farther_ratio": MIN_DEPTH_FARTHER_RATIO,
            "minimum_promotion_area_ratio": MIN_DEPTH_PROMOTION_AREA_RATIO,
            "minimum_promotion_rgb_ratio": MIN_DEPTH_PROMOTION_RGB_RATIO,
            "minimum_signed_depth_component_area_ratio": (
                MIN_SIGNED_DEPTH_COMPONENT_AREA_RATIO
            ),
            "minimum_dominant_depth_promotion_area_ratio": (
                MIN_DOMINANT_DEPTH_PROMOTION_AREA_RATIO
            ),
            "minimum_dominant_depth_promotion_rgb_ratio": (
                MIN_DOMINANT_DEPTH_PROMOTION_RGB_RATIO
            ),
            "minimum_low_contrast_promotion_area_ratio": (
                MIN_LOW_CONTRAST_PROMOTION_AREA_RATIO
            ),
            "maximum_rgb_fallback_chroma_dominance_ratio": (
                MAX_RGB_FALLBACK_CHROMA_DOMINANCE_RATIO
            ),
            "minimum_open_row_interior_fill_ratio": (
                MIN_OPEN_ROW_INTERIOR_FILL_RATIO
            ),
            "minimum_outward_shelf_overlap_ratio": (
                MIN_OUTWARD_SHELF_OVERLAP_RATIO
            ),
            "minimum_open_row_farther_window_ratio": (
                MIN_OPEN_ROW_FARTHER_WINDOW_RATIO
            ),
            "maximum_open_row_movement_balance_ratio": (
                MAX_OPEN_ROW_MOVEMENT_BALANCE_RATIO
            ),
        },
        depth_change_mask,
    )


def build_overlay(
    image: np.ndarray,
    combined_mask: np.ndarray,
    findings: list[dict[str, Any]],
) -> np.ndarray:
    canvas = image.copy()
    tint = canvas.copy()
    tint[combined_mask > 0] = (40, 40, 245)
    canvas = cv2.addWeighted(canvas, 0.72, tint, 0.28, 0.0)
    line_width = max(2, round(canvas.shape[1] / 420))
    for finding in findings:
        x, y, width, height = finding["bbox"]
        cv2.rectangle(
            canvas,
            (x, y),
            (x + width - 1, y + height - 1),
            (0, 255, 255),
            line_width,
        )
        cv2.putText(
            canvas,
            f"REGION {finding['region_index']}",
            (x + 3, max(24, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def existing_result_is_reusable(result: dict[str, Any], detection_only: bool) -> bool:
    status = result.get("status")
    if detection_only:
        return status in {
            "success",
            "partial",
            "unrecognized",
            "no_anomaly",
            "detection_only",
            "recognition_error",
        }
    return status in {"success", "partial", "unrecognized", "no_anomaly"}


def run_record(
    entry: dict[str, Any],
    *,
    data_root: Path,
    initial_scan: InitialScan,
    reviewer: Any | None,
    detection_only: bool,
    overwrite: bool,
) -> dict[str, Any]:
    record_directory: Path = entry["record_directory"]
    output_directory = record_directory / RESULT_DIRECTORY_NAME
    result_path = output_directory / "result.json"
    if not overwrite and result_path.is_file():
        existing = read_json(result_path)
        if existing_result_is_reusable(existing, detection_only):
            return existing

    started_at = time.perf_counter()
    rgb_path = record_directory / "rgb.jpg"
    base_result: dict[str, Any] = {
        "schema_version": 1,
        "task_type": "SHORTAGE",
        "group": entry["group"],
        "record": entry["record"],
        "inspection_target_id": entry["inspection_target_id"],
        "location_id": entry["location_id"],
        "pose_type": entry["pose_type"],
        "source_rgb": relative_path(rgb_path, data_root),
        "baseline_rgb": str(initial_scan.rgb_path),
        "findings": [],
    }
    try:
        current = read_image(rgb_path)
        depth_path, valid_depth_pixels, current_depth_mm = validate_depth(
            record_directory,
            current.shape[:2],
        )
        base_result["source_depth"] = relative_path(depth_path, data_root)
        base_result["valid_depth_pixels"] = valid_depth_pixels
        execution = INSPECT_API.inspect_images_with_artifacts(
            "SHORTAGE",
            initial_scan.rgb,
            current,
            location_id=entry["location_id"],
            pose_type=entry["pose_type"],
        )
        raw_finding_count = len(execution.response.findings)
        execution, depth_supports, depth_filter, depth_change_mask = (
            filter_execution_with_depth(
                execution,
                initial_scan.depth_mm,
                current_depth_mm,
                initial_scan.rgb,
            )
        )
        response = execution.response
        reviewed_by_region: dict[int, Any] = {}
        prompt_by_region: dict[int, str] = {}
        recognition_error: dict[str, str] | None = None
        if response.findings and not detection_only:
            try:
                review = INSPECT_API.review_inspection_execution(
                    execution,
                    task_type="SHORTAGE",
                    location_id=entry["location_id"],
                    pose_type=entry["pose_type"],
                    baseline=initial_scan.rgb,
                    current_source=current,
                    baseline_depth_mm=initial_scan.depth_mm,
                    current_depth_mm=current_depth_mm,
                    reviewer=reviewer,
                )
                reviewed_by_region = {
                    finding.region_index: finding for finding in review.findings
                }
                prompt_by_region = {
                    region_index: prompt
                    for region_index, prompt in enumerate(review.prompts, start=1)
                }
            except INSPECT_API.QwenReviewError as error:
                recognition_error = {
                    "stage": error.stage,
                    "message": str(error),
                }

        combined_mask = np.zeros(execution.review_mask.shape, dtype=np.uint8)
        findings: list[dict[str, Any]] = []
        for region_index, (finding, depth_support) in enumerate(
            zip(response.findings, depth_supports),
            start=1,
        ):
            region_mask = clipped_region_mask(execution.review_mask, finding.bbox)
            combined_mask = cv2.bitwise_or(combined_mask, region_mask)
            mask_path = output_directory / f"region_{region_index:02d}_mask.png"
            write_image(mask_path, region_mask)
            reviewed = reviewed_by_region.get(region_index)
            product_name = (
                reviewed.shortage_product_name
                if reviewed is not None
                else None
            )
            findings.append(
                {
                    "region_index": region_index,
                    "bbox": finding.bbox,
                    "center": finding.center,
                    "sources": finding.sources,
                    "votes": finding.votes,
                    "mask": relative_path(mask_path, data_root),
                    "mask_pixels": int(np.count_nonzero(region_mask)),
                    "depth_support": depth_support,
                    "product_name": product_name,
                    "qwen_prompt": prompt_by_region.get(region_index),
                    "confidence": (
                        reviewed.confidence if reviewed is not None else None
                    ),
                }
            )

        output_directory.mkdir(parents=True, exist_ok=True)
        aligned_path = output_directory / "aligned_current.jpg"
        combined_mask_path = output_directory / "combined_mask.png"
        depth_change_mask_path = output_directory / "depth_change_mask.png"
        overlay_path = output_directory / "overlay.jpg"
        row_detection_path = output_directory / "row_detection.jpg"
        write_image(aligned_path, execution.review_image)
        write_image(combined_mask_path, combined_mask)
        write_image(depth_change_mask_path, depth_change_mask)
        write_image(
            overlay_path,
            build_overlay(execution.review_image, combined_mask, findings),
        )
        row_detection_data: dict[str, Any] | None = None
        row_detection_error: str | None = None
        try:
            row_detection = INSPECT_API.detect_rows(
                execution.review_image,
                INSPECT_API.RowDetectionConfig(pose_type=entry["pose_type"]),
            )
            row_detection_data = row_detection.as_dict()
            write_image(row_detection_path, row_detection.draw())
        except (ValueError, cv2.error) as error:
            row_detection_error = f"{type(error).__name__}: {error}"

        recognized_count = sum(
            1 for finding in findings if finding.get("product_name")
        )
        if not findings:
            status = "no_anomaly"
        elif detection_only:
            status = "detection_only"
        elif recognition_error is not None:
            status = "recognition_error"
        elif recognized_count == len(findings):
            status = "success"
        elif recognized_count:
            status = "partial"
        else:
            status = "unrecognized"
        base_result.update(
            {
                "status": status,
                "has_anomaly": bool(findings),
                "image_size": response.image_size,
                "bbox_format": response.bbox_format,
                "alignment_success": next(
                    (
                        algorithm.alignment_success
                        for algorithm in response.algorithms
                        if algorithm.name == "comparison_based"
                    ),
                    None,
                ),
                "raw_finding_count": raw_finding_count,
                "depth_filter": depth_filter,
                "row_detection": row_detection_data,
                "row_detection_error": row_detection_error,
                "findings": findings,
                "recognized_count": recognized_count,
                "recognition_error": recognition_error,
                "artifacts": {
                    "aligned_current": relative_path(aligned_path, data_root),
                    "combined_mask": relative_path(combined_mask_path, data_root),
                    "depth_change_mask": relative_path(
                        depth_change_mask_path,
                        data_root,
                    ),
                    "overlay": relative_path(overlay_path, data_root),
                    "row_detection": (
                        relative_path(row_detection_path, data_root)
                        if row_detection_path.is_file()
                        else None
                    ),
                },
            }
        )
    except Exception as error:
        base_result.update(
            {
                "status": "error",
                "has_anomaly": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
    base_result["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 1)
    base_result["completed_at"] = datetime.now(UTC).isoformat()
    write_json_atomic(result_path, base_result)
    return base_result


def collect_results(data_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(data_root.glob(f"*/record_*/{RESULT_DIRECTORY_NAME}/result.json")):
        try:
            results.append(read_json(path))
        except RuntimeError:
            continue
    return results


def build_summary(
    data_root: Path,
    results: list[dict[str, Any]],
    *,
    total_records: int,
    detection_only: bool,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return {
        "schema_version": 1,
        "task_type": "SHORTAGE",
        "data_root": str(data_root.resolve()),
        "generated_at": datetime.now(UTC).isoformat(),
        "detection_only": detection_only,
        "total_records": total_records,
        "completed_records": len(results),
        "status_counts": counts,
        "results": results,
    }


def run_records_concurrently(
    records: list[dict[str, Any]],
    *,
    data_root: Path,
    scans: dict[str, InitialScan],
    reviewer_kwargs: dict[str, Any] | None,
    detection_only: bool,
    overwrite: bool,
    workers: int,
    on_result: Callable[[int, int, dict[str, Any], dict[str, Any]], None]
    | None = None,
) -> list[dict[str, Any]]:
    """Run independent records in parallel with one Qwen session per worker."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    if not records:
        return []

    worker_state = threading.local()

    def execute(entry: dict[str, Any]) -> dict[str, Any]:
        reviewer = None
        if not detection_only:
            reviewer = getattr(worker_state, "reviewer", None)
            if reviewer is None:
                configured = dict(reviewer_kwargs or {})
                # SHORTAGE does not use the misplaced full-catalog retriever.
                # Avoid loading one model per worker while keeping independent
                # requests.Session instances for concurrent SKU/Qwen calls.
                configured.setdefault("visual_retriever", None)
                reviewer = INSPECT_API.QwenReviewer(**configured)
                worker_state.reviewer = reviewer
        return run_record(
            entry,
            data_root=data_root,
            initial_scan=scans[entry["group"]],
            reviewer=reviewer,
            detection_only=detection_only,
            overwrite=overwrite,
        )

    ordered_results: list[dict[str, Any] | None] = [None] * len(records)
    completed = 0
    worker_count = min(workers, len(records))
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="shortage-batch",
    ) as executor:
        futures = {
            executor.submit(execute, entry): (index, entry)
            for index, entry in enumerate(records)
        }
        for future in as_completed(futures):
            index, entry = futures[future]
            result = future.result()
            ordered_results[index] = result
            completed += 1
            if on_result is not None:
                on_result(completed, len(records), entry, result)

    return [result for result in ordered_results if result is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--group",
        action="append",
        help="只运行指定分组，可重复传入",
    )
    parser.add_argument("--record", help="只运行指定 record")
    parser.add_argument("--limit", type=int, help="最多运行多少条")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="只生成 bbox/mask，不调用 SKU/Qwen 商品识别",
    )
    parser.add_argument("--qwen-url")
    parser.add_argument("--sku-base-url")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并发 record 数，默认 {DEFAULT_WORKERS}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    selected_groups = set(args.group) if args.group else None
    records = discover_records(
        data_root,
        groups=selected_groups,
        record_name=args.record,
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise RuntimeError("--limit 必须为正整数")
        records = records[: args.limit]
    if not records:
        raise RuntimeError("没有找到匹配的 shortage record")
    if args.workers <= 0:
        raise RuntimeError("--workers 必须为正整数")

    reviewer_kwargs: dict[str, Any] | None = None
    if not args.detection_only:
        reviewer_kwargs = {
            "debug_root": data_root / "qwen_debug",
        }
        if args.qwen_url:
            reviewer_kwargs["qwen_url"] = args.qwen_url
        if args.sku_base_url:
            reviewer_kwargs["sku_base_url"] = args.sku_base_url

    scans: dict[str, InitialScan] = {}
    for entry in records:
        group = entry["group"]
        if group not in scans:
            scans[group] = load_initial_scan(
                entry["inspection_target_id"],
                entry["pose_type"],
            )

    summary_path = data_root / DEFAULT_SUMMARY_NAME
    total_available_records = len(discover_records(data_root))

    def report_result(
        completed: int,
        total: int,
        entry: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        group = entry["group"]
        print(
            f"[{completed}/{total}] {group}/{entry['record']}: "
            f"{result.get('status')} findings={len(result.get('findings', []))} "
            f"elapsed={result.get('elapsed_ms', 0)}ms",
            flush=True,
        )
        all_results = collect_results(data_root)
        write_json_atomic(
            summary_path,
            build_summary(
                data_root,
                all_results,
                total_records=total_available_records,
                detection_only=args.detection_only,
            ),
        )

    run_records_concurrently(
        records,
        data_root=data_root,
        scans=scans,
        reviewer_kwargs=reviewer_kwargs,
        detection_only=args.detection_only,
        overwrite=args.overwrite,
        workers=args.workers,
        on_result=report_result,
    )

    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
