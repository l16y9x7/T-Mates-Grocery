"""Production baseline/current SAM3 shortage comparison pipeline.

The module contains the RGB-D logic shared by the formal inspection and place
locate endpoints.  It deliberately accepts in-memory images and has no
dependency on ``test_web`` or files under ``test_data``.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

import cv2
import numpy as np
import requests

try:
    from ..config import SAM3_URL
    from ..front_row_selection import select_front_row_instances
    from ..pick.locate.main import uses_upper_confidence_pick
    from ..row_detection import RowDetectionConfig, detect_rows
except ImportError:
    from config import SAM3_URL
    from front_row_selection import select_front_row_instances
    from pick.locate.main import uses_upper_confidence_pick
    from row_detection import RowDetectionConfig, detect_rows

try:
    from .export_sam_rows import fit_shelf_boundary_model, perspective_row_crop
except ImportError:
    from export_sam_rows import fit_shelf_boundary_model, perspective_row_crop


PoseType = Literal["SHELF_VIEW_UPPER", "SHELF_VIEW_LOWER"]
Direction = Literal["left", "right", "both", "up"]
Sam3Caller = Callable[[np.ndarray, str, float], dict[str, Any]]

MAPPING_PATH = Path(__file__).resolve().parent / "shortage_mapping_config.json"
POSE_LEVELS: dict[PoseType, tuple[str, ...]] = {
    "SHELF_VIEW_UPPER": ("L1", "L2"),
    "SHELF_VIEW_LOWER": ("L3", "L4", "L5"),
}
DEPTH_DELTA_THRESHOLD_MM = 40.0
SYSTEMATIC_DEPTH_SHIFT_MIN_MM = 30.0
SYSTEMATIC_DEPTH_SHIFT_MAX_MM = 80.0


class SamShortageError(RuntimeError):
    """Raised when the production SAM3 comparison cannot produce a result."""


@dataclass
class RowCrop:
    row_index: int
    level: str
    crop_bbox_xywh: tuple[int, int, int, int]
    rgb: np.ndarray
    depth_mm: np.ndarray


@dataclass
class SamInstance:
    instance_index: int
    score: float | None
    bbox_crop_xyxy: list[float]
    bbox_original_xyxy: list[float]
    mask_crop: np.ndarray
    front_selected: bool
    duplicate_of: int | None
    depth_reliable: bool
    stable_depth_mm: float | None
    depth_mad_mm: float | None
    selection_reason: str


@dataclass
class GroupDetection:
    prompt: str
    expected_front_count: int
    row: RowCrop
    instances: list[SamInstance]
    detection_failed: bool = False

    @property
    def front_instances(self) -> list[SamInstance]:
        return sorted(
            (instance for instance in self.instances if instance.front_selected),
            key=lambda instance: _bbox_center(instance.bbox_crop_xyxy)[0],
        )


@dataclass
class SlotComparison:
    slot_index: int
    product_name: str | None
    status: str
    baseline_instance_index: int
    current_instance_index: int | None
    baseline_bbox_xyxy: list[float]
    target_bbox_current_crop_xyxy: list[float]
    target_bbox_current_xyxy: list[int]
    current_bbox_xyxy: list[float] | None
    baseline_depth_mm: float | None
    current_depth_mm: float | None
    depth_delta_mm: float | None

    @property
    def missing(self) -> bool:
        return self.status.startswith("missing_")


@dataclass
class GroupComparison:
    level: str
    group_index: int
    prompt: str
    slot_product_names: list[str]
    expected_front_count: int
    baseline: GroupDetection
    current: GroupDetection
    slots: list[SlotComparison]
    normalized_x_shift: float
    normalized_pitch: float
    systematic_depth_shift: bool

    @property
    def missing_slots(self) -> list[SlotComparison]:
        return [slot for slot in self.slots if slot.missing]


@dataclass
class ShortageAnalysis:
    location_id: str
    pose_type: PoseType
    image_size: tuple[int, int]
    comparisons: list[GroupComparison] = field(default_factory=list)

    @property
    def missing_product_names(self) -> list[str]:
        return list(
            dict.fromkeys(
                slot.product_name
                for comparison in self.comparisons
                for slot in comparison.missing_slots
                if isinstance(slot.product_name, str) and slot.product_name.strip()
            )
        )


@dataclass
class PlaceReferenceSelection:
    product_name: str
    level: str
    direction: Direction
    target_slot: SlotComparison
    references: list[SamInstance]
    current_row: RowCrop


def load_mapping_config(
    path: str | Path | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    source = Path(path) if path is not None else MAPPING_PATH
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SamShortageError(f"读取 shortage 分组配置失败: {error}") from error
    if not isinstance(payload, dict):
        raise SamShortageError("shortage 分组配置根节点必须是对象")
    normalized: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for location_id, levels in payload.items():
        if not isinstance(location_id, str) or not isinstance(levels, dict):
            raise SamShortageError("shortage 分组配置 view 格式错误")
        normalized_levels: dict[str, list[dict[str, Any]]] = {}
        for level, groups in levels.items():
            if not isinstance(level, str) or not isinstance(groups, list):
                raise SamShortageError(f"{location_id}/{level} 必须是配置组列表")
            normalized_groups: list[dict[str, Any]] = []
            for group_index, group in enumerate(groups, start=1):
                if not isinstance(group, dict):
                    raise SamShortageError(
                        f"{location_id}/{level} 第 {group_index} 组必须是对象"
                    )
                expected = group.get("expected_front_count")
                prompt = group.get("sam3_prompt")
                names = group.get("slot_product_names", [])
                if (
                    not isinstance(expected, int)
                    or isinstance(expected, bool)
                    or expected <= 0
                    or not isinstance(prompt, str)
                    or not prompt.strip()
                ):
                    raise SamShortageError(
                        f"{location_id}/{level} 第 {group_index} 组配置无效"
                    )
                if (
                    not isinstance(names, list)
                    or len(names) != expected
                    or any(not isinstance(name, str) or not name.strip() for name in names)
                ):
                    raise SamShortageError(
                        f"{location_id}/{level} 第 {group_index} 组商品数量不匹配"
                    )
                normalized_groups.append(
                    {
                        "group_index": group_index,
                        "expected_front_count": expected,
                        "sam3_prompt": prompt.strip(),
                        "slot_product_names": [name.strip() for name in names],
                    }
                )
            normalized_levels[level.strip().upper()] = normalized_groups
        normalized[location_id.strip().upper()] = normalized_levels
    return normalized


def extract_rows(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    pose_type: PoseType,
) -> dict[str, RowCrop]:
    image = np.asarray(rgb)
    depth = np.asarray(depth_mm)
    if image.ndim != 3 or image.shape[2] != 3:
        raise SamShortageError("RGB 必须是 BGR 彩色图")
    if depth.ndim != 2 or depth.shape != image.shape[:2]:
        raise SamShortageError(
            f"RGB/深度尺寸不一致: rgb={image.shape[:2]}, depth={depth.shape}"
        )
    if pose_type not in POSE_LEVELS:
        raise SamShortageError(f"不支持的 pose_type: {pose_type}")
    try:
        detection = detect_rows(
            image,
            RowDetectionConfig(target_size=None, pose_type=pose_type),
        )
    except (ValueError, cv2.error) as error:
        raise SamShortageError(f"row_detection 失败: {error}") from error
    if not detection.rows:
        raise SamShortageError("row_detection 没有检测到货架层")
    boundary_model = fit_shelf_boundary_model(
        list(getattr(detection, "rails", [])), image_width=image.shape[1]
    )
    levels = POSE_LEVELS[pose_type]
    rows: dict[str, RowCrop] = {}
    for row_number, row in enumerate(detection.rows, start=1):
        if row_number > len(levels):
            break
        rgb_crop, depth_crop, _shelf_mask, crop_bbox = perspective_row_crop(
            image,
            depth,
            tuple(int(value) for value in row.bbox),
            boundary_model,
        )
        level = levels[row_number - 1]
        rows[level] = RowCrop(
            row_index=row_number,
            level=level,
            crop_bbox_xywh=tuple(int(value) for value in crop_bbox),
            rgb=rgb_crop,
            depth_mm=np.asarray(depth_crop),
        )
    return rows


def call_sam3(
    image: np.ndarray,
    prompt: str,
    threshold: float,
) -> dict[str, Any]:
    success, encoded = cv2.imencode(
        ".jpg", np.asarray(image), [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    if not success:
        raise SamShortageError("无法编码 SAM3 行图片")
    try:
        response = requests.post(
            SAM3_URL,
            files={"image": ("shelf_row.jpg", encoded.tobytes(), "image/jpeg")},
            data={
                "prompt": prompt,
                "threshold": threshold,
                "mask_threshold": 0.5,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        raise SamShortageError(f"SAM3 请求失败: {error}") from error
    except ValueError as error:
        raise SamShortageError(f"SAM3 返回格式错误: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("instances"), list):
        raise SamShortageError("SAM3 响应缺少 instances")
    return payload


def _decode_mask(value: object, image_shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, str) or not value.strip():
        raise SamShortageError("SAM3 实例缺少 mask_png_base64")
    try:
        payload = base64.b64decode(value.split(",", 1)[-1], validate=True)
    except (ValueError, binascii.Error) as error:
        raise SamShortageError("SAM3 mask Base64 无效") from error
    mask = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != image_shape:
        raise SamShortageError(
            f"SAM3 mask 尺寸错误: mask={None if mask is None else mask.shape}, "
            f"image={image_shape}"
        )
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def detect_group(
    row: RowCrop,
    *,
    prompt: str,
    expected_front_count: int,
    location_id: str,
    multiple_groups_on_level: bool,
    level_uses_upper_pick: bool,
    enforce_expected_count: bool,
    sam3_caller: Sam3Caller = call_sam3,
) -> GroupDetection:
    decoded: list[tuple[dict[str, Any], np.ndarray, list[float]]] = []
    for threshold in (0.5, 0.25):
        payload = sam3_caller(row.rgb, prompt, threshold)
        attempt: list[tuple[dict[str, Any], np.ndarray, list[float]]] = []
        for raw in payload.get("instances", []):
            if not isinstance(raw, dict):
                continue
            mask = _decode_mask(raw.get("mask_png_base64"), row.rgb.shape[:2])
            if not np.any(mask):
                continue
            bbox = raw.get("bbox_xyxy")
            if not (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(value, (int, float)) for value in bbox)
            ):
                ys, xs = np.where(mask > 0)
                bbox = [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max() + 1),
                    float(ys.max() + 1),
                ]
            attempt.append((raw, mask, [float(value) for value in bbox]))
        decoded = attempt
        if decoded:
            break

    selection = select_front_row_instances(
        [mask for _, mask, _ in decoded],
        row.depth_mm,
        scores=[
            float(raw["score"])
            if isinstance(raw.get("score"), (int, float))
            else None
            for raw, _, _ in decoded
        ],
        expected_front_count=expected_front_count,
        prefer_global_depth_layer=location_id.upper().startswith(
            ("H2_F_L_INSPECT", "H2_F_R_INSPECT")
        ),
        prefer_regular_columns=location_id.upper().startswith(
            ("H2_F_L_INSPECT", "H2_F_R_INSPECT")
        ),
        prefer_vertical_position_anomaly=level_uses_upper_pick,
        max_same_prompt_depth_spread_mm=(30.0 if multiple_groups_on_level else None),
        enforce_expected_count=enforce_expected_count,
        horizontal_roi=None,
    )
    crop_x, crop_y, _, _ = row.crop_bbox_xywh
    instances: list[SamInstance] = []
    for index, ((raw, mask, bbox), analysis) in enumerate(
        zip(decoded, selection["instances"], strict=True), start=1
    ):
        depth_estimate = analysis["depth_estimate"]
        instances.append(
            SamInstance(
                instance_index=index,
                score=(
                    float(raw["score"])
                    if isinstance(raw.get("score"), (int, float))
                    else None
                ),
                bbox_crop_xyxy=bbox,
                bbox_original_xyxy=[
                    bbox[0] + crop_x,
                    bbox[1] + crop_y,
                    bbox[2] + crop_x,
                    bbox[3] + crop_y,
                ],
                mask_crop=mask,
                front_selected=bool(analysis["selected"]),
                duplicate_of=analysis["duplicate_of"],
                depth_reliable=bool(depth_estimate["reliable"]),
                stable_depth_mm=(
                    float(depth_estimate["depth_mm"])
                    if isinstance(depth_estimate.get("depth_mm"), (int, float))
                    else None
                ),
                depth_mad_mm=(
                    float(depth_estimate["mad_mm"])
                    if isinstance(depth_estimate.get("mad_mm"), (int, float))
                    else None
                ),
                selection_reason=str(analysis["selection_reason"]),
            )
        )
    return GroupDetection(
        prompt=prompt,
        expected_front_count=expected_front_count,
        row=row,
        instances=instances,
        detection_failed=not bool(decoded),
    )


def _bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _normalized_center_x(instance: SamInstance, image_width: int) -> float:
    crop_x = instance.bbox_original_xyxy[0] - instance.bbox_crop_xyxy[0]
    return (_bbox_center(instance.bbox_crop_xyxy)[0] + crop_x) / max(
        1.0, float(image_width)
    )


def _normalized_pitch(
    baseline: list[SamInstance], image_width: int, expected_count: int
) -> float:
    centers = np.asarray(
        [_normalized_center_x(item, image_width) for item in baseline],
        dtype=np.float32,
    )
    if centers.size >= 2:
        positive = np.diff(centers)
        positive = positive[positive > 0.005]
        if positive.size:
            return float(np.median(positive))
    return 1.0 / max(1, expected_count)


def match_slots(
    baseline: list[SamInstance],
    current: list[SamInstance],
    *,
    baseline_image_width: int,
    current_image_width: int,
    expected_count: int,
) -> tuple[dict[int, int], float, float]:
    if not baseline or not current:
        return {}, 0.0, _normalized_pitch(
            baseline, baseline_image_width, expected_count
        )
    baseline_u = [
        _normalized_center_x(item, baseline_image_width) for item in baseline
    ]
    current_u = [_normalized_center_x(item, current_image_width) for item in current]
    pitch = _normalized_pitch(baseline, baseline_image_width, expected_count)
    if len(baseline_u) == len(current_u):
        shift = float(
            np.median(
                np.asarray(current_u, dtype=np.float32)
                - np.asarray(baseline_u, dtype=np.float32)
            )
        )
        return {index: index for index in range(len(baseline_u))}, shift, pitch

    def monotonic_alignment(shift: float) -> tuple[dict[int, int], float]:
        baseline_shorter = len(baseline_u) <= len(current_u)
        short_values = baseline_u if baseline_shorter else current_u
        long_values = current_u if baseline_shorter else baseline_u
        short_count, long_count = len(short_values), len(long_values)
        costs = np.full((short_count + 1, long_count + 1), np.inf, dtype=np.float64)
        take = np.zeros((short_count + 1, long_count + 1), dtype=np.uint8)
        costs[0, :] = 0.0
        for short_index in range(1, short_count + 1):
            for long_index in range(1, long_count + 1):
                skip_cost = costs[short_index, long_index - 1]
                if baseline_shorter:
                    base_value = short_values[short_index - 1]
                    current_value = long_values[long_index - 1]
                else:
                    base_value = long_values[long_index - 1]
                    current_value = short_values[short_index - 1]
                match_cost = costs[short_index - 1, long_index - 1] + abs(
                    base_value + shift - current_value
                )
                if np.isfinite(match_cost) and match_cost <= skip_cost:
                    costs[short_index, long_index] = match_cost
                    take[short_index, long_index] = 1
                else:
                    costs[short_index, long_index] = skip_cost
        matches: dict[int, int] = {}
        short_index, long_index = short_count, long_count
        while short_index > 0 and long_index > 0:
            if take[short_index, long_index]:
                if baseline_shorter:
                    matches[short_index - 1] = long_index - 1
                else:
                    matches[long_index - 1] = short_index - 1
                short_index -= 1
                long_index -= 1
            else:
                long_index -= 1
        return matches, float(costs[short_count, long_count])

    candidates = {0.0}
    candidates.update(
        round(current_value - baseline_value, 6)
        for baseline_value in baseline_u
        for current_value in current_u
    )
    best_matches: dict[int, int] = {}
    best_shift = 0.0
    best_rank: tuple[float, float] | None = None
    for shift in candidates:
        matches, distance = monotonic_alignment(shift)
        rank = (distance, abs(shift))
        if best_rank is None or rank < best_rank:
            best_rank, best_matches, best_shift = rank, matches, shift
    if best_matches:
        best_shift = float(
            np.median(
                np.asarray(
                    [
                        current_u[current_index] - baseline_u[baseline_index]
                        for baseline_index, current_index in best_matches.items()
                    ],
                    dtype=np.float32,
                )
            )
        )
    return best_matches, best_shift, pitch


def _map_bbox(
    bbox: Sequence[float],
    *,
    source_row: RowCrop,
    target_row: RowCrop,
    source_image_size: tuple[int, int],
    target_image_size: tuple[int, int],
    normalized_x_shift: float,
) -> list[float]:
    source_crop_x, _, _, source_height = source_row.crop_bbox_xywh
    target_crop_x, _, _, target_height = target_row.crop_bbox_xywh
    source_width, _ = source_image_size
    target_width, _ = target_image_size
    scale_y = target_height / max(1.0, float(source_height))

    def map_x(value: float) -> float:
        source_u = (float(value) + source_crop_x) / max(1.0, float(source_width))
        return (source_u + normalized_x_shift) * target_width - target_crop_x

    return [
        map_x(float(bbox[0])),
        float(bbox[1]) * scale_y,
        map_x(float(bbox[2])),
        float(bbox[3]) * scale_y,
    ]


def compare_group(
    *,
    level: str,
    group_index: int,
    slot_product_names: list[str],
    baseline: GroupDetection,
    current: GroupDetection,
    baseline_image_size: tuple[int, int],
    current_image_size: tuple[int, int],
) -> GroupComparison:
    baseline_front = baseline.front_instances
    current_front = current.front_instances
    expected = baseline.expected_front_count
    matches, shift, pitch = match_slots(
        baseline_front,
        current_front,
        baseline_image_width=baseline_image_size[0],
        current_image_width=current_image_size[0],
        expected_count=expected,
    )
    deltas: list[float] = []
    for baseline_index, current_index in matches.items():
        baseline_depth = baseline_front[baseline_index].stable_depth_mm
        current_depth = current_front[current_index].stable_depth_mm
        if baseline_depth is not None and current_depth is not None:
            deltas.append(current_depth - baseline_depth)
    systematic_shift = (
        len(baseline_front) >= 2
        and len(matches) == len(baseline_front)
        and len(deltas) == len(baseline_front)
        and min(deltas) >= SYSTEMATIC_DEPTH_SHIFT_MIN_MM
        and max(deltas) <= SYSTEMATIC_DEPTH_SHIFT_MAX_MM
    )
    baseline_complete = len(baseline_front) == expected
    slots: list[SlotComparison] = []
    crop_x, crop_y, _, _ = current.row.crop_bbox_xywh
    current_height, current_width = current_image_size[1], current_image_size[0]
    for slot_position, baseline_instance in enumerate(baseline_front, start=1):
        baseline_index = slot_position - 1
        current_index = matches.get(baseline_index)
        current_instance = (
            current_front[current_index] if current_index is not None else None
        )
        baseline_depth = baseline_instance.stable_depth_mm
        current_depth = (
            current_instance.stable_depth_mm if current_instance is not None else None
        )
        depth_delta = (
            current_depth - baseline_depth
            if current_depth is not None and baseline_depth is not None
            else None
        )
        mapped = _map_bbox(
            baseline_instance.bbox_crop_xyxy,
            source_row=baseline.row,
            target_row=current.row,
            source_image_size=baseline_image_size,
            target_image_size=current_image_size,
            normalized_x_shift=shift,
        )
        if current_instance is None:
            status = (
                "current_detection_failed"
                if current.detection_failed
                else ("missing_unmatched" if baseline_complete else "baseline_incomplete")
            )
        elif systematic_shift:
            status = "occupied_systematic_shift"
        elif depth_delta is not None and depth_delta > DEPTH_DELTA_THRESHOLD_MM:
            status = "missing_depth_delta"
        else:
            status = "occupied"
        original_target = [
            max(0, min(current_width, int(round(mapped[0] + crop_x)))),
            max(0, min(current_height, int(round(mapped[1] + crop_y)))),
            max(0, min(current_width, int(round(mapped[2] + crop_x)))),
            max(0, min(current_height, int(round(mapped[3] + crop_y)))),
        ]
        slots.append(
            SlotComparison(
                slot_index=slot_position,
                product_name=(
                    slot_product_names[slot_position - 1]
                    if slot_position <= len(slot_product_names)
                    else None
                ),
                status=status,
                baseline_instance_index=baseline_instance.instance_index,
                current_instance_index=(
                    current_instance.instance_index
                    if current_instance is not None
                    else None
                ),
                baseline_bbox_xyxy=list(baseline_instance.bbox_crop_xyxy),
                target_bbox_current_crop_xyxy=mapped,
                target_bbox_current_xyxy=original_target,
                current_bbox_xyxy=(
                    list(current_instance.bbox_crop_xyxy)
                    if current_instance is not None
                    else None
                ),
                baseline_depth_mm=baseline_depth,
                current_depth_mm=current_depth,
                depth_delta_mm=(round(depth_delta, 2) if depth_delta is not None else None),
            )
        )
    return GroupComparison(
        level=level,
        group_index=group_index,
        prompt=baseline.prompt,
        slot_product_names=slot_product_names,
        expected_front_count=expected,
        baseline=baseline,
        current=current,
        slots=slots,
        normalized_x_shift=shift,
        normalized_pitch=pitch,
        systematic_depth_shift=systematic_shift,
    )


def analyze_shortage(
    *,
    location_id: str,
    pose_type: PoseType,
    baseline_rgb: np.ndarray,
    baseline_depth_mm: np.ndarray,
    current_rgb: np.ndarray,
    current_depth_mm: np.ndarray,
    product_name_filter: str | None = None,
    mapping_path: str | Path | None = None,
    sam3_caller: Sam3Caller = call_sam3,
) -> ShortageAnalysis:
    normalized_location = location_id.strip().upper()
    mapping = load_mapping_config(mapping_path)
    level_config = mapping.get(normalized_location)
    if level_config is None:
        raise SamShortageError(f"shortage 配置不存在: {normalized_location}")
    baseline_rows = extract_rows(baseline_rgb, baseline_depth_mm, pose_type)
    current_rows = extract_rows(current_rgb, current_depth_mm, pose_type)
    image_size = (int(current_rgb.shape[1]), int(current_rgb.shape[0]))
    baseline_size = (int(baseline_rgb.shape[1]), int(baseline_rgb.shape[0]))
    analysis = ShortageAnalysis(normalized_location, pose_type, image_size)
    requested_name = (product_name_filter or "").strip()
    for level in POSE_LEVELS[pose_type]:
        baseline_row = baseline_rows.get(level)
        current_row = current_rows.get(level)
        if baseline_row is None or current_row is None:
            continue
        configs = level_config.get(level, [])
        if requested_name and not any(
            requested_name in config["slot_product_names"] for config in configs
        ):
            continue
        level_uses_upper_pick = any(
            uses_upper_confidence_pick(name, "SORTING")
            for config in configs
            for name in config["slot_product_names"]
        )
        for config in configs:
            prompt = str(config["sam3_prompt"])
            expected = int(config["expected_front_count"])
            baseline_detection = detect_group(
                baseline_row,
                prompt=prompt,
                expected_front_count=expected,
                location_id=normalized_location,
                multiple_groups_on_level=len(configs) > 1,
                level_uses_upper_pick=level_uses_upper_pick,
                enforce_expected_count=True,
                sam3_caller=sam3_caller,
            )
            current_detection = detect_group(
                current_row,
                prompt=prompt,
                expected_front_count=expected,
                location_id=normalized_location,
                multiple_groups_on_level=len(configs) > 1,
                level_uses_upper_pick=level_uses_upper_pick,
                enforce_expected_count=False,
                sam3_caller=sam3_caller,
            )
            analysis.comparisons.append(
                compare_group(
                    level=level,
                    group_index=int(config["group_index"]),
                    slot_product_names=list(config["slot_product_names"]),
                    baseline=baseline_detection,
                    current=current_detection,
                    baseline_image_size=baseline_size,
                    current_image_size=image_size,
                )
            )
    if requested_name and not analysis.comparisons:
        raise SamShortageError(
            f"当前巡检视角没有配置商品: {requested_name}"
        )
    return analysis


def _select_horizontal(
    instances: list[SamInstance], target_bbox: Sequence[float]
) -> tuple[list[SamInstance], Direction] | None:
    target_cx, _ = _bbox_center(target_bbox)
    target_width = max(1.0, float(target_bbox[2]) - float(target_bbox[0]))
    tolerance = target_width * 0.08
    left = sorted(
        [
            instance
            for instance in instances
            if _bbox_center(instance.bbox_crop_xyxy)[0] < target_cx - tolerance
        ],
        key=lambda instance: target_cx - _bbox_center(instance.bbox_crop_xyxy)[0],
    )
    right = sorted(
        [
            instance
            for instance in instances
            if _bbox_center(instance.bbox_crop_xyxy)[0] > target_cx + tolerance
        ],
        key=lambda instance: _bbox_center(instance.bbox_crop_xyxy)[0] - target_cx,
    )
    if left and right:
        selected, direction = [left[0], right[0]], "both"
    elif len(left) >= 2:
        selected, direction = left[:2], "left"
    elif len(right) >= 2:
        selected, direction = right[:2], "right"
    else:
        return None
    selected.sort(key=lambda instance: _bbox_center(instance.bbox_crop_xyxy)[0])
    return selected, direction


def _select_below(
    instances: Iterable[SamInstance], target_bbox: Sequence[float]
) -> SamInstance | None:
    target_cx, target_cy = _bbox_center(target_bbox)
    target_width = max(1.0, float(target_bbox[2]) - float(target_bbox[0]))
    target_height = max(1.0, float(target_bbox[3]) - float(target_bbox[1]))
    candidates: list[tuple[tuple[float, float], SamInstance]] = []
    for instance in instances:
        center_x, center_y = _bbox_center(instance.bbox_crop_xyxy)
        vertical_offset = center_y - target_cy
        overlap = max(
            0.0,
            min(float(target_bbox[2]), instance.bbox_crop_xyxy[2])
            - max(float(target_bbox[0]), instance.bbox_crop_xyxy[0]),
        )
        instance_width = max(
            1.0, instance.bbox_crop_xyxy[2] - instance.bbox_crop_xyxy[0]
        )
        overlap_ratio = overlap / min(target_width, instance_width)
        if vertical_offset <= target_height * 0.10:
            continue
        if overlap_ratio < 0.20 and abs(center_x - target_cx) > target_width:
            continue
        candidates.append(
            (
                (
                    abs(center_x - target_cx) / target_width,
                    vertical_offset / target_height,
                ),
                instance,
            )
        )
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    return intersection / max(1.0, first_area + second_area - intersection)


def _level_reference_candidates(
    analysis: ShortageAnalysis,
    target_comparison: GroupComparison,
    target_slot: SlotComparison,
) -> list[SamInstance]:
    """Collect de-duplicated instances from every configured group on a level."""

    candidates: list[SamInstance] = []
    for comparison in analysis.comparisons:
        if comparison.level != target_comparison.level:
            continue
        for instance in comparison.current.instances:
            if (
                comparison is target_comparison
                and instance.instance_index == target_slot.current_instance_index
            ):
                continue
            if instance.duplicate_of is not None:
                continue
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(candidates)
                    if _bbox_iou(
                        existing.bbox_crop_xyxy,
                        instance.bbox_crop_xyxy,
                    )
                    >= 0.65
                ),
                None,
            )
            if duplicate_index is None:
                candidates.append(instance)
            elif instance.front_selected and not candidates[duplicate_index].front_selected:
                candidates[duplicate_index] = instance
    return candidates


def select_place_references(
    analysis: ShortageAnalysis,
    product_name: str,
) -> PlaceReferenceSelection:
    normalized_name = product_name.strip()
    failures: list[str] = []
    for comparison in analysis.comparisons:
        for slot in comparison.missing_slots:
            if slot.product_name != normalized_name:
                continue
            target = slot.target_bbox_current_crop_xyxy
            reference_candidates = _level_reference_candidates(
                analysis,
                comparison,
                slot,
            )
            if uses_upper_confidence_pick(normalized_name, "SORTING"):
                below = _select_below(
                    (
                        instance
                        for instance in reference_candidates
                        if instance.duplicate_of is None and instance.depth_reliable
                    ),
                    target,
                )
                if below is not None:
                    return PlaceReferenceSelection(
                        normalized_name,
                        comparison.level,
                        "up",
                        slot,
                        [below],
                        comparison.current.row,
                    )
            horizontal = _select_horizontal(
                sorted(
                    (
                        instance
                        for instance in reference_candidates
                        if instance.front_selected
                    ),
                    key=lambda instance: _bbox_center(
                        instance.bbox_crop_xyxy
                    )[0],
                ),
                target,
            )
            if horizontal is not None:
                references, direction = horizontal
                return PlaceReferenceSelection(
                    normalized_name,
                    comparison.level,
                    direction,
                    slot,
                    references,
                    comparison.current.row,
                )
            failures.append(
                f"{comparison.level}/group_{comparison.group_index}/slot_{slot.slot_index}"
            )
    if failures:
        raise SamShortageError(
            "缺失槽位周围没有足够参照物: " + ", ".join(failures)
        )
    raise SamShortageError(f"当前画面没有确认商品缺失槽位: {normalized_name}")


def full_image_mask(
    instance: SamInstance,
    row: RowCrop,
    image_shape: Sequence[int],
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    full = np.zeros((height, width), dtype=np.uint8)
    crop_x, crop_y, crop_width, crop_height = row.crop_bbox_xywh
    right = min(width, crop_x + crop_width)
    bottom = min(height, crop_y + crop_height)
    source_width = max(0, right - crop_x)
    source_height = max(0, bottom - crop_y)
    if source_width and source_height:
        full[crop_y:bottom, crop_x:right] = instance.mask_crop[
            :source_height, :source_width
        ]
    return full


def analysis_as_dict(analysis: ShortageAnalysis) -> dict[str, Any]:
    """Compact JSON-safe diagnostics without embedding mask pixels."""

    return {
        "location_id": analysis.location_id,
        "pose_type": analysis.pose_type,
        "image_size": list(analysis.image_size),
        "findings": [
            {"shortage_product_name": name}
            for name in analysis.missing_product_names
        ],
        "comparisons": [
            {
                "level": comparison.level,
                "group_index": comparison.group_index,
                "prompt": comparison.prompt,
                "expected_front_count": comparison.expected_front_count,
                "baseline_front_count": len(comparison.baseline.front_instances),
                "current_front_count": len(comparison.current.front_instances),
                "systematic_depth_shift": comparison.systematic_depth_shift,
                "slots": [
                    {
                        "slot_index": slot.slot_index,
                        "product_name": slot.product_name,
                        "status": slot.status,
                        "target_bbox_current_xyxy": slot.target_bbox_current_xyxy,
                        "baseline_depth_mm": slot.baseline_depth_mm,
                        "current_depth_mm": slot.current_depth_mm,
                        "depth_delta_mm": slot.depth_delta_mm,
                    }
                    for slot in comparison.slots
                ],
            }
            for comparison in analysis.comparisons
        ],
    }
