"""Batch baseline/current front-row comparison for real shortage records.

Run from the perception directory::

    python inspect/batch_front_row_compare.py --workers 4 --overwrite

The script runs the same row_detection -> SAM3 -> front-row selection pipeline
for baseline and current RGB-D, matches current instances to baseline slots, and
writes web-ready artifacts under ``test_data/real_shortage_front_compare``.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from test_web import server as web_server  # noqa: E402


DEFAULT_DATA_ROOT = PERCEPTION_ROOT / "test_data" / "real_shortage_regression"
DEFAULT_OUTPUT_ROOT = PERCEPTION_ROOT / "test_data" / "real_shortage_front_compare"
COMPARISON_SCHEMA_VERSION = 11
DEPTH_DELTA_THRESHOLD_MM = 40.0
DEPTH_CONSISTENCY_THRESHOLD_MM = 10.0
SYSTEMATIC_DEPTH_SHIFT_MIN_MM = 30.0
SYSTEMATIC_DEPTH_SHIFT_MAX_MM = 80.0
PRINT_LOCK = threading.Lock()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".jpg", np.asarray(image))
    if not success:
        raise RuntimeError(f"无法编码图像: {path}")
    encoded.tofile(path)


def decode_data_url(value: str) -> np.ndarray:
    if not isinstance(value, str) or "," not in value:
        raise RuntimeError("结果缺少图像 data URL")
    encoded = value.split(",", 1)[1]
    image = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if image is None:
        raise RuntimeError("无法解码结果图像")
    return image


def selected_instances(result: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [item for item in result.get("instances", []) if item.get("front_selected")],
        key=lambda item: (
            float(item["bbox_crop_xyxy"][0])
            + float(item["bbox_crop_xyxy"][2])
        )
        / 2.0,
    )


def bbox_center_x(instance: dict[str, Any]) -> float:
    bbox = instance["bbox_crop_xyxy"]
    return (float(bbox[0]) + float(bbox[2])) / 2.0


def normalized_center_x(
    instance: dict[str, Any],
    *,
    crop_x: int,
    image_width: int,
) -> float:
    """Return the instance center in the shared, uncropped image coordinates."""

    return (bbox_center_x(instance) + float(crop_x)) / max(
        1.0, float(image_width)
    )


def normalized_pitch(
    baseline: list[dict[str, Any]],
    *,
    baseline_crop_x: int,
    baseline_image_width: int,
    expected_count: int,
) -> float:
    centers = np.asarray(
        [
            normalized_center_x(
                item,
                crop_x=baseline_crop_x,
                image_width=baseline_image_width,
            )
            for item in baseline
        ],
        dtype=np.float32,
    )
    if centers.size >= 2:
        gaps = np.diff(centers)
        positive = gaps[gaps > 0.005]
        if positive.size:
            return float(np.median(positive))
    return 1.0 / max(1, expected_count)


def match_slots(
    baseline: list[dict[str, Any]],
    current: list[dict[str, Any]],
    *,
    baseline_crop_x: int,
    current_crop_x: int,
    baseline_image_width: int,
    current_image_width: int,
    expected_count: int,
) -> tuple[dict[int, int], float, float]:
    """Return order-preserving matches in the shared source-image coordinates."""

    if not baseline or not current:
        return {}, 0.0, normalized_pitch(
            baseline,
            baseline_crop_x=baseline_crop_x,
            baseline_image_width=baseline_image_width,
            expected_count=expected_count,
        )
    baseline_u = [
        normalized_center_x(
            item,
            crop_x=baseline_crop_x,
            image_width=baseline_image_width,
        )
        for item in baseline
    ]
    current_u = [
        normalized_center_x(
            item,
            crop_x=current_crop_x,
            image_width=current_image_width,
        )
        for item in current
    ]
    pitch = normalized_pitch(
        baseline,
        baseline_crop_x=baseline_crop_x,
        baseline_image_width=baseline_image_width,
        expected_count=expected_count,
    )

    # With equal counts the ordinal identity is unambiguous: a camera/view
    # translation must not make the nearest-neighbour matcher skip one item and
    # invent a shortage at the opposite edge.  Match strictly left-to-right and
    # estimate the common horizontal translation from all pairs.
    if len(baseline_u) == len(current_u):
        matches = {index: index for index in range(len(baseline_u))}
        shift = float(
            np.median(
                np.asarray(current_u, dtype=np.float32)
                - np.asarray(baseline_u, dtype=np.float32)
            )
        )
        return matches, shift, pitch

    def monotonic_alignment(shift: float) -> tuple[dict[int, int], float]:
        """Match every item in the shorter sequence to an ordered subsequence."""

        baseline_is_shorter = len(baseline_u) <= len(current_u)
        short_values = baseline_u if baseline_is_shorter else current_u
        long_values = current_u if baseline_is_shorter else baseline_u
        short_count = len(short_values)
        long_count = len(long_values)
        infinity = float("inf")
        costs = np.full((short_count + 1, long_count + 1), infinity, dtype=np.float64)
        take = np.zeros((short_count + 1, long_count + 1), dtype=np.uint8)
        costs[0, :] = 0.0

        for short_index in range(1, short_count + 1):
            for long_index in range(1, long_count + 1):
                skip_cost = costs[short_index, long_index - 1]
                if baseline_is_shorter:
                    base_value = short_values[short_index - 1]
                    current_value = long_values[long_index - 1]
                else:
                    base_value = long_values[long_index - 1]
                    current_value = short_values[short_index - 1]
                match_cost = costs[short_index - 1, long_index - 1] + abs(
                    (base_value + shift) - current_value
                )
                if np.isfinite(match_cost) and match_cost <= skip_cost:
                    costs[short_index, long_index] = match_cost
                    take[short_index, long_index] = 1
                else:
                    costs[short_index, long_index] = skip_cost

        matches: dict[int, int] = {}
        short_index = short_count
        long_index = long_count
        while short_index > 0 and long_index > 0:
            if take[short_index, long_index]:
                if baseline_is_shorter:
                    matches[short_index - 1] = long_index - 1
                else:
                    matches[long_index - 1] = short_index - 1
                short_index -= 1
                long_index -= 1
            else:
                long_index -= 1
        return matches, float(costs[short_count, long_count])

    # Search common translations, but evaluate every candidate with monotonic
    # sequence alignment.  The smaller absolute shift breaks geometrically
    # equivalent edge ambiguities, preventing a whole-slot jump.
    shift_candidates = {0.0}
    shift_candidates.update(
        round(current_value - base_value, 6)
        for base_value in baseline_u
        for current_value in current_u
    )
    best_matches: dict[int, int] = {}
    best_shift = 0.0
    best_rank: tuple[float, float] | None = None
    for shift in shift_candidates:
        matches, total_distance = monotonic_alignment(shift)
        rank = (total_distance, abs(shift))
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_matches = matches
            best_shift = shift

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


def map_bbox(
    bbox: list[float],
    *,
    source_crop: tuple[int, int, int, int],
    target_crop: tuple[int, int, int, int],
    source_image_size: tuple[int, int],
    target_image_size: tuple[int, int],
    normalized_x_shift: float,
) -> list[float]:
    source_crop_x, _source_crop_y, _source_width, source_height = source_crop
    target_crop_x, _target_crop_y, _target_width, target_height = target_crop
    source_image_width, _source_image_height = source_image_size
    target_image_width, _target_image_height = target_image_size
    scale_y = target_height / max(1.0, float(source_height))

    def map_x(value: float) -> float:
        source_u = (float(value) + source_crop_x) / max(
            1.0, float(source_image_width)
        )
        return (
            (source_u + normalized_x_shift) * target_image_width
            - target_crop_x
        )

    return [
        map_x(float(bbox[0])),
        float(bbox[1]) * scale_y,
        map_x(float(bbox[2])),
        float(bbox[3]) * scale_y,
    ]


def draw_dashed_box(
    image: np.ndarray,
    bbox: list[float],
    *,
    color: tuple[int, int, int],
    label: str,
    thickness: int = 3,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(image.shape[1] - 1, x1))
    x2 = max(0, min(image.shape[1] - 1, x2))
    y1 = max(0, min(image.shape[0] - 1, y1))
    y2 = max(0, min(image.shape[0] - 1, y2))
    dash, gap = 13, 7
    for start in range(x1, x2 + 1, dash + gap):
        end = min(x2, start + dash)
        cv2.line(image, (start, y1), (end, y1), color, thickness, cv2.LINE_AA)
        cv2.line(image, (start, y2), (end, y2), color, thickness, cv2.LINE_AA)
    for start in range(y1, y2 + 1, dash + gap):
        end = min(y2, start + dash)
        cv2.line(image, (x1, start), (x1, end), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x2, start), (x2, end), color, thickness, cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in result.items()
        if not key.endswith("_data_url") and key != "instances"
    }
    compact["instances"] = [
        {
            key: value
            for key, value in item.items()
            if not key.endswith("_data_url")
        }
        for item in result.get("instances", [])
    ]
    return compact


def compare_prompt_group(
    *,
    group: str,
    record: str,
    level: str,
    baseline_row_index: int,
    current_row_index: int,
    config: dict[str, Any],
    output_directory: Path,
    baseline_metadata: dict[str, Any],
    current_metadata: dict[str, Any],
) -> dict[str, Any]:
    group_index = int(config["group_index"])
    expected_count = int(config["expected_front_count"])
    prompt = str(config["sam3_prompt"])
    slot_product_names = [
        str(name) for name in config.get("slot_product_names", [])
    ]
    baseline_result = web_server.run_sam_row_debug(
        web_server.SamRowRunRequest(
            group=group,
            record=record,
            row_index=baseline_row_index,
            config_group_index=group_index,
            source="baseline",
            comparison_full_width=True,
            enforce_expected_count=True,
        )
    )
    current_result = web_server.run_sam_row_debug(
        web_server.SamRowRunRequest(
            group=group,
            record=record,
            row_index=current_row_index,
            config_group_index=group_index,
            source="current",
            comparison_full_width=True,
            # Baseline and current must use the same candidate-completion rule.
            # Otherwise a small SAM-mask geometry change can make the baseline
            # promote a valid front instance while current silently drops it,
            # which creates an artificial missing slot before depth comparison.
            enforce_expected_count=True,
        )
    )

    baseline_row = baseline_metadata["rows"][baseline_row_index - 1]
    current_row = current_metadata["rows"][current_row_index - 1]
    baseline_crop = tuple(int(value) for value in baseline_row["crop_bbox_xywh"])
    current_crop = tuple(int(value) for value in current_row["crop_bbox_xywh"])
    baseline_image_size = tuple(
        int(value) for value in baseline_metadata["source_image_size"]
    )
    current_image_size = tuple(
        int(value) for value in current_metadata["source_image_size"]
    )
    baseline_front = selected_instances(baseline_result)
    current_front = selected_instances(current_result)
    current_detection_failed = (
        current_result.get("sam3_detection_status") == "empty_after_retry"
    )
    matches, normalized_shift, pitch = match_slots(
        baseline_front,
        current_front,
        baseline_crop_x=baseline_crop[0],
        current_crop_x=current_crop[0],
        baseline_image_width=baseline_image_size[0],
        current_image_width=current_image_size[0],
        expected_count=expected_count,
    )
    baseline_complete = len(baseline_front) == expected_count

    matched_depth_deltas: list[float] = []
    for baseline_index, current_index in matches.items():
        baseline_depth = baseline_front[baseline_index].get("stable_depth_mm")
        current_depth = current_front[current_index].get("stable_depth_mm")
        if isinstance(baseline_depth, (int, float)) and isinstance(
            current_depth, (int, float)
        ):
            matched_depth_deltas.append(float(current_depth) - float(baseline_depth))

    # A nearly identical positive delta on every occupied slot is camera/view
    # drift, not several products disappearing at once.  Keep the raw deltas in
    # the result, but suppress shortage decisions for this common-mode band.
    systematic_depth_shift = (
        len(baseline_front) >= 2
        and len(matches) == len(baseline_front)
        and len(matched_depth_deltas) == len(baseline_front)
        and min(matched_depth_deltas) >= SYSTEMATIC_DEPTH_SHIFT_MIN_MM
        and max(matched_depth_deltas) <= SYSTEMATIC_DEPTH_SHIFT_MAX_MM
    )
    systematic_shift_median = (
        float(np.median(np.asarray(matched_depth_deltas, dtype=np.float32)))
        if matched_depth_deltas
        else None
    )
    slots: list[dict[str, Any]] = []
    missing_slots: list[dict[str, Any]] = []
    for slot_position, baseline_instance in enumerate(baseline_front, start=1):
        baseline_index = slot_position - 1
        current_index = matches.get(baseline_index)
        current_instance = (
            current_front[current_index] if current_index is not None else None
        )
        baseline_depth = baseline_instance.get("stable_depth_mm")
        current_depth = (
            current_instance.get("stable_depth_mm")
            if current_instance is not None
            else None
        )
        depth_delta = (
            float(current_depth) - float(baseline_depth)
            if isinstance(current_depth, (int, float))
            and isinstance(baseline_depth, (int, float))
            else None
        )
        mapped_bbox = map_bbox(
            [float(value) for value in baseline_instance["bbox_crop_xyxy"]],
            source_crop=baseline_crop,
            target_crop=current_crop,
            source_image_size=baseline_image_size,
            target_image_size=current_image_size,
            normalized_x_shift=normalized_shift,
        )
        if current_instance is None:
            if current_detection_failed:
                status = "current_detection_failed"
            else:
                status = (
                    "missing_unmatched"
                    if baseline_complete
                    else "baseline_incomplete"
                )
        # A matched slot whose depth is effectively unchanged cannot be a
        # shortage.  Keep this as a final decision invariant so later matching
        # or geometry heuristics cannot turn a depth-consistent pair back into
        # a missing result.
        elif (
            depth_delta is not None
            and abs(depth_delta) < DEPTH_CONSISTENCY_THRESHOLD_MM
        ):
            status = "occupied_depth_consistent"
        elif systematic_depth_shift:
            status = "occupied_systematic_shift"
        elif depth_delta is not None and depth_delta > DEPTH_DELTA_THRESHOLD_MM:
            status = "missing_depth_delta"
        else:
            status = "occupied"
        slot = {
            "slot_index": slot_position,
            "product_name": (
                slot_product_names[slot_position - 1]
                if slot_position <= len(slot_product_names)
                else None
            ),
            "status": status,
            "baseline_instance_index": baseline_instance["instance_index"],
            "current_instance_index": (
                current_instance["instance_index"]
                if current_instance is not None
                else None
            ),
            "baseline_bbox_xyxy": baseline_instance["bbox_crop_xyxy"],
            "mapped_current_bbox_xyxy": [round(value, 2) for value in mapped_bbox],
            "current_bbox_xyxy": (
                current_instance["bbox_crop_xyxy"]
                if current_instance is not None
                else None
            ),
            "baseline_depth_mm": baseline_depth,
            "current_depth_mm": current_depth,
            "depth_delta_mm": round(depth_delta, 2) if depth_delta is not None else None,
        }
        slots.append(slot)
        if status.startswith("missing_"):
            missing_slots.append(slot)

    baseline_overlay = decode_data_url(baseline_result["front_overlay_data_url"])
    current_overlay = decode_data_url(current_result["front_overlay_data_url"])
    comparison_overlay = current_overlay.copy()
    for slot in slots:
        bbox = slot["mapped_current_bbox_xyxy"]
        if slot["status"].startswith("missing_"):
            draw_dashed_box(
                comparison_overlay,
                bbox,
                color=(255, 0, 255),
                label=f"SLOT {slot['slot_index']} MISSING?",
            )
        else:
            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            is_unknown = slot["status"] == "current_detection_failed"
            color = (0, 210, 255) if is_unknown else (255, 255, 0)
            cv2.rectangle(
                comparison_overlay,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )
            cv2.putText(
                comparison_overlay,
                (
                    f"SLOT {slot['slot_index']} UNKNOWN"
                    if is_unknown
                    else f"SLOT {slot['slot_index']}"
                ),
                (max(0, x1), max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv2.LINE_AA,
            )

    write_image(output_directory / "baseline_front.jpg", baseline_overlay)
    write_image(output_directory / "current_front.jpg", current_overlay)
    write_image(output_directory / "comparison.jpg", comparison_overlay)
    write_image(
        output_directory / "baseline_front_mask.png",
        decode_data_url(baseline_result["front_mask_data_url"]),
    )
    write_image(
        output_directory / "current_front_mask.png",
        decode_data_url(current_result["front_mask_data_url"]),
    )
    result = {
        "group_index": group_index,
        "prompt": prompt,
        "slot_product_names": slot_product_names,
        "expected_front_count": expected_count,
        "baseline_front_count": len(baseline_front),
        "current_front_count": len(current_front),
        "current_detection_failed": current_detection_failed,
        "resolved_slot_count": len(slots),
        "baseline_complete": baseline_complete,
        "slot_coordinate_space": "source_image",
        "normalized_x_shift": round(normalized_shift, 6),
        "normalized_pitch": round(pitch, 6),
        "slot_matching_strategy": (
            "ordinal_left_to_right"
            if len(baseline_front) == len(current_front)
            else "monotonic_sequence_alignment"
        ),
        "depth_delta_threshold_mm": DEPTH_DELTA_THRESHOLD_MM,
        "depth_consistency_threshold_mm": DEPTH_CONSISTENCY_THRESHOLD_MM,
        "systematic_depth_shift": {
            "detected": systematic_depth_shift,
            "accepted_range_mm": [
                SYSTEMATIC_DEPTH_SHIFT_MIN_MM,
                SYSTEMATIC_DEPTH_SHIFT_MAX_MM,
            ],
            "median_delta_mm": (
                round(systematic_shift_median, 2)
                if systematic_shift_median is not None
                else None
            ),
            "min_delta_mm": (
                round(min(matched_depth_deltas), 2)
                if matched_depth_deltas
                else None
            ),
            "max_delta_mm": (
                round(max(matched_depth_deltas), 2)
                if matched_depth_deltas
                else None
            ),
            "matched_slot_count": len(matched_depth_deltas),
        },
        "slots": slots,
        "missing_slots": missing_slots,
        "missing_product_names": list(
            dict.fromkeys(
                slot["product_name"]
                for slot in missing_slots
                if isinstance(slot.get("product_name"), str)
                and slot["product_name"].strip()
            )
        ),
        "artifacts": {
            "baseline_front": "baseline_front.jpg",
            "current_front": "current_front.jpg",
            "comparison": "comparison.jpg",
            "baseline_front_mask": "baseline_front_mask.png",
            "current_front_mask": "current_front_mask.png",
        },
        "baseline_result": compact_result(baseline_result),
        "current_result": compact_result(current_result),
    }
    write_json(output_directory / "result.json", result)
    return result


def process_record(
    group: str,
    record: str,
    *,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    record_output = output_root / group / record
    result_path = record_output / "result.json"
    mapping_config_mtime_ns = (
        web_server.SHORTAGE_MAPPING_CONFIG_PATH.stat().st_mtime_ns
    )
    if result_path.is_file() and not overwrite:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") == COMPARISON_SCHEMA_VERSION
            and payload.get("mapping_config_mtime_ns")
            == mapping_config_mtime_ns
        ):
            return {**payload, "cached": True}

    baseline_directory, baseline_metadata = web_server.ensure_sam_row_export(
        group, record, "baseline"
    )
    current_directory, current_metadata = web_server.ensure_sam_row_export(
        group, record, "current"
    )
    baseline_by_level = {
        row.get("level"): int(row["row_index"])
        for row in baseline_metadata.get("rows", [])
        if isinstance(row, dict) and isinstance(row.get("level"), str)
    }
    current_by_level = {
        row.get("level"): int(row["row_index"])
        for row in current_metadata.get("rows", [])
        if isinstance(row, dict) and isinstance(row.get("level"), str)
    }
    rows: list[dict[str, Any]] = []
    for level in sorted(set(baseline_by_level) & set(current_by_level)):
        prompt_groups = web_server.shortage_prompt_groups(group, level)
        if not prompt_groups:
            continue
        row_output = record_output / f"{level}"
        group_results: list[dict[str, Any]] = []
        for config in prompt_groups:
            group_index = int(config["group_index"])
            group_result = compare_prompt_group(
                group=group,
                record=record,
                level=level,
                baseline_row_index=baseline_by_level[level],
                current_row_index=current_by_level[level],
                config=config,
                output_directory=row_output / f"group_{group_index:02d}",
                baseline_metadata=baseline_metadata,
                current_metadata=current_metadata,
            )
            group_result["result_path"] = f"{level}/group_{group_index:02d}/result.json"
            group_results.append(group_result)
        rows.append(
            {
                "level": level,
                "baseline_row_index": baseline_by_level[level],
                "current_row_index": current_by_level[level],
                "prompt_groups": group_results,
            }
        )

    finding_details = [
        {
            "shortage_product_name": slot["product_name"],
            "level": row["level"],
            "group_index": prompt_group["group_index"],
            "slot_index": slot["slot_index"],
            "status": slot["status"],
        }
        for row in rows
        for prompt_group in row["prompt_groups"]
        for slot in prompt_group.get("missing_slots", [])
        if isinstance(slot.get("product_name"), str)
    ]
    unique_finding_names = list(
        dict.fromkeys(
            item["shortage_product_name"]
            for item in finding_details
            if item["shortage_product_name"].strip()
        )
    )
    result = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "mapping_config_mtime_ns": mapping_config_mtime_ns,
        "group": group,
        "record": record,
        "baseline_rows_root": str(baseline_directory.resolve()),
        "current_rows_root": str(current_directory.resolve()),
        "rows": rows,
        # The public response reports affected SKU identities, so repeated
        # facings of the same SKU are returned once.  ``finding_details`` and
        # ``missing_slot_count`` retain every physical slot for diagnostics.
        "findings": [
            {"shortage_product_name": name}
            for name in unique_finding_names
        ],
        "finding_details": finding_details,
        "missing_slot_count": sum(
            len(prompt_group.get("missing_slots", []))
            for row in rows
            for prompt_group in row["prompt_groups"]
        ),
    }
    write_json(result_path, result)
    return result


def discover_records(data_root: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for group_directory in sorted(data_root.iterdir(), key=lambda path: path.name):
        if (
            not group_directory.is_dir()
            or web_server.INITIAL_SCAN_DIRECTORY_PATTERN.fullmatch(group_directory.name)
            is None
        ):
            continue
        for record_directory in sorted(group_directory.iterdir(), key=lambda path: path.name):
            if not record_directory.is_dir():
                continue
            required = (
                record_directory / "baseline_rgb.jpg",
                record_directory / "baseline_depth_mm.npy",
                record_directory / "rgb.jpg",
                record_directory / "depth_mm.npy",
            )
            if all(path.is_file() for path in required):
                records.append((group_directory.name, record_directory.name))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--group", default="", help="只跑指定 view group")
    parser.add_argument("--record", default="", help="只跑指定 record")
    parser.add_argument("--dry-run", action="store_true", help="仅列出任务，不调用 SAM3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if data_root != DEFAULT_DATA_ROOT.resolve():
        raise RuntimeError(
            "当前网页后端固定读取 real_shortage_regression；"
            "请使用默认 --data-root，或同步修改 test_web/server.py"
        )
    records = [
        item
        for item in discover_records(data_root)
        if (not args.group or item[0] == args.group)
        and (not args.record or item[1] == args.record)
    ]
    if not records:
        raise RuntimeError("没有找到包含 baseline/current RGB-D 的 record")
    if args.dry_run:
        print(f"records: {len(records)}")
        for group, record in records:
            print(f"{group}/{record}")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    # Import the row exporter once in the main thread.  The web backend loads it
    # lazily, and doing that concurrently from several workers can race while
    # Python is registering the module.
    web_server.load_sam_row_exporter()
    completed: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_record,
                group,
                record,
                output_root=output_root,
                overwrite=args.overwrite,
            ): (group, record)
            for group, record in records
        }
        for future in as_completed(futures):
            group, record = futures[future]
            try:
                result = future.result()
                completed.append(
                    {
                        "group": group,
                        "record": record,
                        "result": f"{group}/{record}/result.json",
                        "missing_slot_count": result.get("missing_slot_count", 0),
                        "cached": bool(result.get("cached")),
                    }
                )
                with PRINT_LOCK:
                    print(f"[OK] {group}/{record}")
            except Exception as error:  # noqa: BLE001 - batch must continue
                errors.append(
                    {
                        "group": group,
                        "record": record,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                with PRINT_LOCK:
                    print(f"[ERROR] {group}/{record}: {error}")

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "workers": workers,
        "completed_records": len(completed),
        "failed_records": len(errors),
        "records": sorted(completed, key=lambda item: (item["group"], item["record"])),
        "errors": sorted(errors, key=lambda item: (item["group"], item["record"])),
    }
    write_json(output_root / "manifest.json", manifest)
    print(f"completed records: {len(completed)}")
    print(f"failed records: {len(errors)}")
    print(f"manifest: {output_root / 'manifest.json'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
