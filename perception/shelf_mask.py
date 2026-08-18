"""Shared SAM3 shelf-region masking for inspection and place locate.

The same function is used for baseline and current row crops.  Narrow
components at (or close to) the horizontal image edges are removed first,
the remaining components are merged, and the result is accepted when a
connected component spans more than 60 percent of the row width and the
derived retained ROI covers at least 60 percent of the row image.  Pixels
outside the retained main-shelf region are set to pure black.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import requests

try:
    from .config import SAM3_URL
except ImportError:
    from config import SAM3_URL


DEFAULT_PROMPT = "shelf"
DEFAULT_DETECTION_THRESHOLDS = (0.5, 0.25)
DEFAULT_MASK_THRESHOLD = 0.35
DEFAULT_MAX_EDGE_BOTTOM_WIDTH_RATIO = 0.30
DEFAULT_EDGE_TOUCH_TOLERANCE_RATIO = 0.02
DEFAULT_EDGE_TOUCH_MIN_TOLERANCE_PX = 10
DEFAULT_MIN_SPANNING_WIDTH_RATIO = 0.60
DEFAULT_RETAINED_EXPANSION_PX = 10

ShelfSam3Caller = Callable[[np.ndarray, str, float, float], dict[str, Any]]


class ShelfMaskError(RuntimeError):
    """Raised when a shelf mask request or response is invalid."""


@dataclass
class ShelfMaskResult:
    shelf_mask: np.ndarray
    retained_mask: np.ndarray
    filtered_rgb: np.ndarray
    fallback_to_full_image: bool
    attempts: list[dict[str, Any]]
    components: list[dict[str, Any]]
    kept_components: list[dict[str, Any]]
    selected_component: dict[str, Any] | None
    parameters: dict[str, Any]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "fallback_to_full_image": self.fallback_to_full_image,
            "status": (
                "fallback_full_image"
                if self.fallback_to_full_image
                else "success"
            ),
            "attempts": self.attempts,
            "components": self.components,
            "kept_components": self.kept_components,
            "selected_component": self.selected_component,
            "parameters": self.parameters,
        }


def call_sam3_shelf(
    image: np.ndarray,
    prompt: str,
    threshold: float,
    mask_threshold: float,
) -> dict[str, Any]:
    success, encoded = cv2.imencode(
        ".jpg",
        np.asarray(image),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not success:
        raise ShelfMaskError("无法编码 shelf SAM3 行图片")
    try:
        response = requests.post(
            SAM3_URL,
            files={"image": ("shelf_row.jpg", encoded.tobytes(), "image/jpeg")},
            data={
                "prompt": prompt,
                "threshold": threshold,
                "mask_threshold": mask_threshold,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        raise ShelfMaskError(f"SAM3 shelf 请求失败: {error}") from error
    except ValueError as error:
        raise ShelfMaskError(f"SAM3 shelf 返回格式错误: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("instances"), list):
        raise ShelfMaskError("SAM3 shelf 响应缺少 instances")
    return payload


def decode_mask(value: object, image_shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, str) or not value.strip():
        raise ShelfMaskError("SAM3 shelf 实例缺少 mask_png_base64")
    try:
        payload = base64.b64decode(value.split(",", 1)[-1], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ShelfMaskError("SAM3 shelf mask Base64 无效") from error
    mask = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != image_shape:
        raise ShelfMaskError(
            f"SAM3 shelf mask 尺寸错误: "
            f"mask={None if mask is None else mask.shape}, image={image_shape}"
        )
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def build_retained_roi_mask(
    image_shape: tuple[int, int],
    removed_edge_components: Sequence[dict[str, Any]],
    selected_component: dict[str, Any],
    *,
    expansion_px: int = DEFAULT_RETAINED_EXPANSION_PX,
) -> np.ndarray:
    """Keep a full-height shelf band without changing the image dimensions."""

    height, width = image_shape
    selected_bbox = selected_component.get("bbox_xywh") or [0, 0, width, height]
    selected_left = max(0, min(width, int(selected_bbox[0])))
    selected_right = max(
        selected_left,
        min(width, selected_left + int(selected_bbox[2])),
    )
    removed_left: list[int] = []
    removed_right: list[int] = []
    for candidate in removed_edge_components:
        edge_range = candidate.get("bottom_edge_x_range") or [0, width]
        left = max(0, min(width, int(edge_range[0])))
        right = max(left, min(width, int(edge_range[1])))
        if candidate.get("touches_left_edge"):
            removed_left.append(right)
        if candidate.get("touches_right_edge"):
            removed_right.append(left)

    # Rejected edge masks define the inner deletion boundary.  When a side
    # has no rejected mask, use the main shelf component and preserve a small
    # outward margin.  The output remains the original HxW image.
    keep_left = (
        max(removed_left)
        if removed_left
        else max(0, selected_left - expansion_px)
    )
    keep_right = (
        min(removed_right)
        if removed_right
        else min(width, selected_right + expansion_px)
    )
    retained = np.zeros((height, width), dtype=np.uint8)
    if keep_left < keep_right:
        retained[:, keep_left:keep_right] = 255
    else:
        retained[:, :] = 255
    return retained


def component_candidates(
    sam_result: dict[str, Any],
    rgb: np.ndarray,
    *,
    max_edge_component_width_ratio: float = (
        DEFAULT_MAX_EDGE_BOTTOM_WIDTH_RATIO
    ),
    edge_touch_tolerance_ratio: float = DEFAULT_EDGE_TOUCH_TOLERANCE_RATIO,
    edge_touch_min_tolerance_px: int = DEFAULT_EDGE_TOUCH_MIN_TOLERANCE_PX,
) -> list[dict[str, Any]]:
    height, width = np.asarray(rgb).shape[:2]
    tolerance_px = max(
        int(edge_touch_min_tolerance_px),
        int(round(float(width) * edge_touch_tolerance_ratio)),
    )
    candidates: list[dict[str, Any]] = []
    for instance_index, raw in enumerate(sam_result.get("instances", []), start=1):
        if not isinstance(raw, dict):
            continue
        mask = decode_mask(raw.get("mask_png_base64"), (height, width))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.where(mask > 0, 1, 0).astype(np.uint8),
            connectivity=8,
        )
        for component_index in range(1, count):
            x = int(stats[component_index, cv2.CC_STAT_LEFT])
            y = int(stats[component_index, cv2.CC_STAT_TOP])
            component_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            component = labels == component_index
            right = x + component_width
            # The mask's lower edge is the per-column bottom envelope, not the
            # single globally lowest scanline.  A wide shelf mask can contain
            # one or two narrow downward protrusions; measuring only their last
            # row incorrectly classifies the whole shelf as an edge sliver.
            bottom_x = np.flatnonzero(np.any(component, axis=0))
            bottom_y_by_x = np.asarray(
                [int(np.flatnonzero(component[:, column]).max()) for column in bottom_x],
                dtype=np.int32,
            )
            bottom_left = int(bottom_x.min())
            bottom_right = int(bottom_x.max()) + 1
            bottom_width = int(bottom_x.size)
            bottom_width_ratio = float(bottom_width) / max(1.0, float(width))
            left_gap = max(0, x)
            right_gap = max(0, width - right)
            touches_left = left_gap <= tolerance_px
            touches_right = right_gap <= tolerance_px
            removed = (
                (touches_left or touches_right)
                and bottom_width_ratio < max_edge_component_width_ratio
            )
            candidates.append(
                {
                    "instance_index": instance_index,
                    "component_index": component_index,
                    "score": (
                        round(float(raw["score"]), 6)
                        if isinstance(raw.get("score"), (int, float))
                        else None
                    ),
                    "area_px": area,
                    "bbox_xywh": [x, y, component_width, component_height],
                    "width_ratio": round(component_width / float(width), 6),
                    "bottom_edge_y_range": [
                        int(bottom_y_by_x.min()),
                        int(bottom_y_by_x.max()),
                    ],
                    "bottom_edge_y_median": round(
                        float(np.median(bottom_y_by_x)),
                        3,
                    ),
                    "bottom_edge_x_range": [bottom_left, bottom_right],
                    "bottom_edge_width_px": bottom_width,
                    "bottom_edge_width_ratio": round(bottom_width_ratio, 6),
                    "edge_touch_tolerance_px": tolerance_px,
                    "left_edge_gap_px": left_gap,
                    "right_edge_gap_px": right_gap,
                    "touches_left_edge": touches_left,
                    "touches_right_edge": touches_right,
                    "removed_edge_sliver": removed,
                    "kept": not removed,
                    "mask": component,
                }
            )
    return candidates


def spanning_components(
    mask: np.ndarray,
    *,
    min_width_ratio: float = DEFAULT_MIN_SPANNING_WIDTH_RATIO,
) -> list[dict[str, Any]]:
    binary = np.where(np.asarray(mask) > 0, 1, 0).astype(np.uint8)
    _, width = binary.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    results: list[dict[str, Any]] = []
    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        component_width = int(stats[index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        ratio = float(component_width) / max(1.0, float(width))
        if ratio > min_width_ratio:
            results.append(
                {
                    "component_index": index,
                    "bbox_xywh": [x, y, component_width, component_height],
                    "area_px": area,
                    "width_ratio": round(ratio, 6),
                }
            )
    return sorted(results, key=lambda item: int(item["area_px"]), reverse=True)


def _serializable(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if key != "mask"}


def apply_shelf_mask(
    rgb: np.ndarray,
    *,
    prompt: str = DEFAULT_PROMPT,
    detection_thresholds: Sequence[float] = DEFAULT_DETECTION_THRESHOLDS,
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
    max_edge_component_width_ratio: float = (
        DEFAULT_MAX_EDGE_BOTTOM_WIDTH_RATIO
    ),
    edge_touch_tolerance_ratio: float = DEFAULT_EDGE_TOUCH_TOLERANCE_RATIO,
    edge_touch_min_tolerance_px: int = DEFAULT_EDGE_TOUCH_MIN_TOLERANCE_PX,
    min_spanning_component_width_ratio: float = (
        DEFAULT_MIN_SPANNING_WIDTH_RATIO
    ),
    expansion_px: int = DEFAULT_RETAINED_EXPANSION_PX,
    sam3_caller: ShelfSam3Caller = call_sam3_shelf,
) -> ShelfMaskResult:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ShelfMaskError("shelf mask 输入必须是 BGR 彩色图")

    attempts: list[dict[str, Any]] = []
    final_candidates: list[dict[str, Any]] = []
    final_kept: list[dict[str, Any]] = []
    final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    selected: dict[str, Any] | None = None
    for threshold in detection_thresholds:
        try:
            payload = sam3_caller(image, prompt, float(threshold), mask_threshold)
            candidates = component_candidates(
                payload,
                image,
                max_edge_component_width_ratio=max_edge_component_width_ratio,
                edge_touch_tolerance_ratio=edge_touch_tolerance_ratio,
                edge_touch_min_tolerance_px=edge_touch_min_tolerance_px,
            )
            kept = [candidate for candidate in candidates if candidate["kept"]]
            merged = np.zeros(image.shape[:2], dtype=np.uint8)
            for candidate in kept:
                merged[candidate["mask"]] = 255
            spanning = spanning_components(
                merged,
                min_width_ratio=min_spanning_component_width_ratio,
            )
            attempts.append(
                {
                    "threshold": float(threshold),
                    "instance_count": len(payload.get("instances", [])),
                    "component_count": len(candidates),
                    "removed_edge_sliver_count": len(candidates) - len(kept),
                    "kept_component_count": len(kept),
                    "spanning_component_count": len(spanning),
                }
            )
            final_candidates = candidates
            if spanning:
                selected = spanning[0]
                final_kept = kept
                final_mask = merged
                break
        except (ShelfMaskError, ValueError, cv2.error) as error:
            attempts.append(
                {
                    "threshold": float(threshold),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    fallback = selected is None
    if fallback:
        shelf_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        retained_mask = shelf_mask.copy()
        filtered = image.copy()
    else:
        shelf_mask = final_mask
        retained_mask = build_retained_roi_mask(
            image.shape[:2],
            [item for item in final_candidates if item["removed_edge_sliver"]],
            selected,
            expansion_px=expansion_px,
        )
        # A rejected left/right cross-section is a full-height exclusion.
        # Do not let another overlapping SAM instance paint that column back
        # into the merged shelf mask.
        shelf_mask = shelf_mask.copy()
        shelf_mask[retained_mask == 0] = 0
        filtered = np.zeros_like(image)
        filtered[retained_mask > 0] = image[retained_mask > 0]

    return ShelfMaskResult(
        shelf_mask=shelf_mask,
        retained_mask=retained_mask,
        filtered_rgb=filtered,
        fallback_to_full_image=fallback,
        attempts=attempts,
        components=[_serializable(item) for item in final_candidates],
        kept_components=[_serializable(item) for item in final_kept],
        selected_component=dict(selected) if selected is not None else None,
        parameters={
            "prompt": prompt,
            "detection_thresholds": [float(value) for value in detection_thresholds],
            "mask_threshold": mask_threshold,
            "max_edge_bottom_width_ratio_exclusive": (
                max_edge_component_width_ratio
            ),
            "edge_touch_tolerance_ratio": edge_touch_tolerance_ratio,
            "edge_touch_min_tolerance_px": edge_touch_min_tolerance_px,
            "min_spanning_width_ratio_exclusive": (
                min_spanning_component_width_ratio
            ),
            "main_shelf_side_outward_margin_px": expansion_px,
        },
    )
