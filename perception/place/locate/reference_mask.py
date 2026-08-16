"""Generate a shortage product mask in the fixed Task0 reference image.

The shortage comparison already expresses its changed regions in reference-image
coordinates.  This module crops that known region from the complete Task0 RGB,
runs SAM3 with the product's SHORTAGE prompt, and maps the selected instance
back to a full-resolution mask aligned with the Task0 depth image.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import requests

try:
    from ...config import SAM3_URL
except ImportError:  # ``python main.py`` from the perception directory.
    from config import SAM3_URL


PERCEPTION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHORTAGE_PROMPT_MAPPING_PATH = (
    PERCEPTION_ROOT / "pick" / "locate" / "qwen_sam_prompt_mapping_shortage.json"
)
SHORTAGE_PROMPT_MAPPING_ENVIRONMENT = "PLACE_SHORTAGE_PROMPT_MAPPING_PATH"
SAM3_THRESHOLD = 0.5
SAM3_MASK_THRESHOLD = 0.5
SAM3_TIMEOUT_SECONDS = 120
REFERENCE_CROP_X_PADDING_RATIO = 0.3
REFERENCE_CROP_Y_PADDING_RATIO = 1.5
REFERENCE_CROP_MAX_Y_PADDING = 100
REFERENCE_CROP_ROW_CONTEXT = 12


class ReferenceMaskError(RuntimeError):
    """A caller-facing failure while producing the reference product mask."""

    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ReferenceMaskResult:
    mask: np.ndarray
    sam_prompt: str
    crop_box: tuple[int, int, int, int]
    selected_bbox: tuple[float, float, float, float]
    selected_score: float | None
    candidate_count: int


Sam3Client = Callable[[str, np.ndarray], list[dict[str, Any]]]


def shortage_prompt_mapping_path() -> Path:
    configured = os.getenv(SHORTAGE_PROMPT_MAPPING_ENVIRONMENT, "").strip()
    return (
        Path(configured).expanduser()
        if configured
        else DEFAULT_SHORTAGE_PROMPT_MAPPING_PATH
    )


def _normalized_product_name(value: str) -> str:
    return "".join(character for character in value if character.isalnum()).casefold()


def load_shortage_sam_prompt(
    product_name: str,
    *,
    mapping_path: str | Path | None = None,
) -> str:
    """Load only the SAM3 half of the existing SHORTAGE prompt mapping."""

    path = Path(mapping_path) if mapping_path is not None else shortage_prompt_mapping_path()
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReferenceMaskError(
            f"SHORTAGE Prompt 配对文件不存在: {path}",
            status_code=500,
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise ReferenceMaskError(
            f"读取 SHORTAGE Prompt 配对失败: {error}",
            status_code=500,
        ) from error
    if not isinstance(mapping, dict):
        raise ReferenceMaskError(
            "SHORTAGE Prompt 配对文件必须是 JSON object",
            status_code=500,
        )

    pair = mapping.get(product_name.strip())
    if pair is None:
        normalized_target = _normalized_product_name(product_name)
        matches = [
            value
            for name, value in mapping.items()
            if isinstance(name, str)
            and _normalized_product_name(name) == normalized_target
        ]
        if len(matches) == 1:
            pair = matches[0]
    if not isinstance(pair, dict):
        raise ReferenceMaskError(
            f"SHORTAGE 尚未配置商品 SAM3 Prompt: {product_name}",
            status_code=400,
        )
    prompt = pair.get("sam3_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ReferenceMaskError(
            f"商品缺少 SHORTAGE SAM3 Prompt: {product_name}",
            status_code=500,
        )
    return prompt.strip()


def reference_crop_box(
    image_shape: Sequence[int],
    bbox: Sequence[int | float],
    *,
    row_bbox: Sequence[int | float] | None = None,
) -> tuple[int, int, int, int]:
    """Expand an ``xywh`` shortage bbox while staying near its shelf row."""

    if len(image_shape) < 2 or len(bbox) != 4:
        raise ReferenceMaskError("reference image shape or shortage bbox is invalid")
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    x, y, width, height = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ReferenceMaskError("shortage bbox contains non-finite values")
    if width <= 0 or height <= 0:
        raise ReferenceMaskError("shortage bbox width and height must be positive")

    x_padding = round(width * REFERENCE_CROP_X_PADDING_RATIO)
    y_padding = min(
        round(height * REFERENCE_CROP_Y_PADDING_RATIO),
        REFERENCE_CROP_MAX_Y_PADDING,
    )
    left = max(0, math.floor(x - x_padding))
    top = max(0, math.floor(y - y_padding))
    right = min(image_width, math.ceil(x + width + x_padding))
    bottom = min(image_height, math.ceil(y + height + y_padding))

    if row_bbox is not None:
        if len(row_bbox) != 4:
            raise ReferenceMaskError("row_bbox must be [x, y, width, height]")
        _, row_y, _, row_height = [float(value) for value in row_bbox]
        if row_height <= 0:
            raise ReferenceMaskError("row_bbox height must be positive")
        top = max(top, math.floor(row_y - REFERENCE_CROP_ROW_CONTEXT))
        bottom = min(
            bottom,
            math.ceil(row_y + row_height + REFERENCE_CROP_ROW_CONTEXT),
        )
    if right <= left or bottom <= top:
        raise ReferenceMaskError("shortage bbox does not define a valid reference crop")
    return left, top, right, bottom


def call_sam3(prompt: str, crop_image: np.ndarray) -> list[dict[str, Any]]:
    """Call the shared SAM3 service with a BGR reference-image crop."""

    success, encoded = cv2.imencode(
        ".jpg",
        np.asarray(crop_image),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not success:
        raise ReferenceMaskError("无法编码 Task0 reference crop", status_code=500)
    try:
        response = requests.post(
            SAM3_URL,
            files={"image": ("reference_crop.jpg", encoded.tobytes(), "image/jpeg")},
            data={
                "prompt": prompt,
                "threshold": SAM3_THRESHOLD,
                "mask_threshold": SAM3_MASK_THRESHOLD,
            },
            timeout=SAM3_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        raise ReferenceMaskError(
            f"SAM3 请求失败: {error}",
            status_code=502,
        ) from error
    except ValueError as error:
        raise ReferenceMaskError("SAM3 响应不是有效 JSON", status_code=502) from error
    instances = payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(instances, list):
        raise ReferenceMaskError("SAM3 响应缺少 instances 数组", status_code=502)
    return [instance for instance in instances if isinstance(instance, dict)]


def _decode_sam_mask(value: object, target_shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, str) or not value:
        raise ReferenceMaskError("SAM3 实例缺少 mask_png_base64", status_code=502)
    encoded = value.split(",", 1)[-1]
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ReferenceMaskError("SAM3 mask base64 无效", status_code=502) from error
    mask = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ReferenceMaskError("SAM3 mask PNG 无效", status_code=502)
    if mask.shape != target_shape:
        mask = cv2.resize(
            mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ReferenceMaskError("SAM3 返回了空 mask", status_code=502)
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def _xywh_mask(shape: tuple[int, int], bbox: Sequence[int | float]) -> np.ndarray:
    x, y, width, height = [float(value) for value in bbox]
    left = max(0, math.floor(x))
    top = max(0, math.floor(y))
    right = min(shape[1], math.ceil(x + width))
    bottom = min(shape[0], math.ceil(y + height))
    output = np.zeros(shape, dtype=bool)
    if right > left and bottom > top:
        output[top:bottom, left:right] = True
    return output


def generate_reference_mask(
    reference_image: np.ndarray,
    shortage_bbox: Sequence[int | float],
    product_name: str,
    *,
    component_mask: np.ndarray | None = None,
    row_bbox: Sequence[int | float] | None = None,
    mapping_path: str | Path | None = None,
    sam3_client: Sam3Client | None = None,
) -> ReferenceMaskResult:
    """Return the one SAM3 instance matching a shortage in Task0 coordinates."""

    image = np.asarray(reference_image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ReferenceMaskError("reference_image must be a uint8 BGR image")
    if component_mask is not None and np.asarray(component_mask).shape != image.shape[:2]:
        raise ReferenceMaskError("component_mask must match the reference image")

    crop_box = reference_crop_box(image.shape, shortage_bbox, row_bbox=row_bbox)
    left, top, right, bottom = crop_box
    crop = image[top:bottom, left:right]
    prompt = load_shortage_sam_prompt(product_name, mapping_path=mapping_path)
    instances = (sam3_client or call_sam3)(prompt, crop)
    if not instances:
        raise ReferenceMaskError(
            "SAM3 没有在 Task0 完整图缺货区域找到目标商品",
            status_code=404,
        )

    crop_shape = crop.shape[:2]
    bbox_target = _xywh_mask(image.shape[:2], shortage_bbox)[top:bottom, left:right]
    if component_mask is not None:
        component_target = np.asarray(component_mask)[top:bottom, left:right] > 0
    else:
        component_target = bbox_target

    candidates: list[
        tuple[
            tuple[float, float, float, float, float, float],
            np.ndarray,
            tuple[float, float, float, float],
            float | None,
        ]
    ] = []
    bbox_center_x = float(shortage_bbox[0]) + float(shortage_bbox[2]) / 2 - left
    bbox_center_y = float(shortage_bbox[1]) + float(shortage_bbox[3]) / 2 - top
    for instance in instances:
        try:
            mask = _decode_sam_mask(instance.get("mask_png_base64"), crop_shape)
            local_bbox = _mask_bbox(mask)
        except ReferenceMaskError:
            continue
        candidate = mask > 0
        candidate_pixels = int(np.count_nonzero(candidate))
        if not candidate_pixels:
            continue
        component_intersection = int(np.count_nonzero(candidate & component_target))
        bbox_intersection = int(np.count_nonzero(candidate & bbox_target))
        if component_intersection == 0 and bbox_intersection == 0:
            continue
        component_pixels = max(1, int(np.count_nonzero(component_target)))
        bbox_pixels = max(1, int(np.count_nonzero(bbox_target)))
        center_x = (local_bbox[0] + local_bbox[2]) / 2
        center_y = (local_bbox[1] + local_bbox[3]) / 2
        center_distance = math.hypot(center_x - bbox_center_x, center_y - bbox_center_y)
        diagonal = max(1.0, math.hypot(crop_shape[1], crop_shape[0]))
        score_value = instance.get("score")
        score = float(score_value) if isinstance(score_value, (int, float)) else None
        ranking = (
            component_intersection / component_pixels,
            bbox_intersection / bbox_pixels,
            component_intersection / candidate_pixels,
            bbox_intersection / candidate_pixels,
            -center_distance / diagonal,
            score if score is not None and math.isfinite(score) else -1.0,
        )
        candidates.append((ranking, mask, local_bbox, score))

    if not candidates:
        raise ReferenceMaskError(
            "SAM3 候选均不与完整图缺货区域重叠",
            status_code=404,
        )
    _, selected_mask, selected_local_bbox, selected_score = max(
        candidates,
        key=lambda item: item[0],
    )
    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_mask[top:bottom, left:right] = selected_mask
    selected_bbox = (
        selected_local_bbox[0] + left,
        selected_local_bbox[1] + top,
        selected_local_bbox[2] + left,
        selected_local_bbox[3] + top,
    )
    return ReferenceMaskResult(
        mask=full_mask,
        sam_prompt=prompt,
        crop_box=crop_box,
        selected_bbox=selected_bbox,
        selected_score=selected_score,
        candidate_count=len(instances),
    )
