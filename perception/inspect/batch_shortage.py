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
MIN_DEPTH_PROMOTION_AREA_RATIO = 0.005
MAX_DEPTH_PROMOTION_AREA_RATIO = 0.20
MIN_DEPTH_PROMOTION_RGB_PIXELS = 10
MIN_DEPTH_PROMOTION_RGB_RATIO = 0.05
MIN_DEPTH_PROMOTION_NEAR_RATIO = 0.15
MIN_DEPTH_PROMOTION_ROW_OVERLAP = 0.80
DEPTH_PROMOTION_BORDER_RATIO = 0.015
DEPTH_PROMOTION_RGB_DILATION_RATIO = 0.011
DEPTH_PROMOTION_MAX_ASPECT_RATIO = 5.0
DEPTH_PROMOTION_DUPLICATE_IOU = 0.20
DEPTH_PROMOTION_INTERIOR_INSET_RATIO = 0.25
MIN_OPEN_ROW_INTERIOR_FILL_RATIO = 0.20
DEFAULT_WORKERS = 4


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
        if min(
            x,
            y,
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


def filter_execution_with_depth(
    execution: Any,
    baseline_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
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
    depth_change_mask = np.where(
        np.isfinite(baseline_aligned)
        & np.isfinite(current_aligned)
        & (baseline_aligned > 0)
        & (current_aligned > 0)
        & ((current_aligned - baseline_aligned) > DEPTH_CHANGE_THRESHOLD_MM),
        255,
        0,
    ).astype(np.uint8)
    supports = [
        depth_support_for_finding(
            execution.review_mask,
            finding.bbox,
            baseline_aligned,
            current_aligned,
        )
        for finding in execution.response.findings
    ]
    kept_pairs = [
        (
            finding,
            support,
            clipped_region_mask(execution.review_mask, finding.bbox),
        )
        for finding, support in zip(execution.response.findings, supports)
        if support["accepted"]
    ]
    original_kept_count = len(kept_pairs)
    promotion_error: str | None = None
    row_bboxes: list[list[int]] = []
    open_ended_row_bboxes: list[list[int]] = []
    try:
        row_detection = INSPECT_API.detect_rows(
            execution.review_image,
            INSPECT_API.RowDetectionConfig(
                target_size=None,
                pose_type=execution.response.pose_type,
            ),
        )
        for row in row_detection.rows:
            row_bbox = list(row.bbox)
            row_bboxes.append(row_bbox)
            if getattr(row, "lower_rail_index", None) is None:
                open_ended_row_bboxes.append(row_bbox)
    except (ValueError, cv2.error) as error:
        promotion_error = f"{type(error).__name__}: {error}"
    promotions = promote_depth_components(
        execution.review_mask,
        depth_change_mask,
        baseline_aligned,
        current_aligned,
        row_bboxes,
        open_ended_row_bboxes,
        [finding for finding, _, _ in kept_pairs],
    )
    kept_pairs.extend(promotions)
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
            "input_findings": len(supports),
            "kept_findings": len(kept_findings),
            "rejected_findings": len(supports) - original_kept_count,
            "promoted_findings": len(promotions),
            "promotion_row_count": len(row_bboxes),
            "promotion_error": promotion_error,
            "depth_change_threshold_mm": DEPTH_CHANGE_THRESHOLD_MM,
            "minimum_farther_ratio": MIN_DEPTH_FARTHER_RATIO,
            "minimum_promotion_area_ratio": MIN_DEPTH_PROMOTION_AREA_RATIO,
            "minimum_promotion_rgb_ratio": MIN_DEPTH_PROMOTION_RGB_RATIO,
            "minimum_open_row_interior_fill_ratio": (
                MIN_OPEN_ROW_INTERIOR_FILL_RATIO
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
