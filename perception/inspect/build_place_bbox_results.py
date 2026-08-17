"""Build place-reference bbox diagnostics from cached shortage group results.

This is intentionally a post-processing step: it reuses the current front-row
instances already written by ``batch_front_row_compare.py`` and therefore does
not call SAM3 again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from pick.locate.main import uses_upper_confidence_pick  # noqa: E402


DEFAULT_ROOT = PERCEPTION_ROOT / "test_data" / "real_shortage_front_compare"


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise RuntimeError(f"无法读取图像: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"无法编码图像: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def select_horizontal_references(
    instances: list[dict[str, Any]], target_bbox: list[float]
) -> tuple[list[dict[str, Any]], str] | tuple[list[dict[str, Any]], None]:
    target_cx, _ = bbox_center(target_bbox)
    target_width = max(1.0, target_bbox[2] - target_bbox[0])
    tolerance = target_width * 0.08
    left = sorted(
        [
            item
            for item in instances
            if bbox_center(item["bbox_crop_xyxy"])[0] < target_cx - tolerance
        ],
        key=lambda item: target_cx - bbox_center(item["bbox_crop_xyxy"])[0],
    )
    right = sorted(
        [
            item
            for item in instances
            if bbox_center(item["bbox_crop_xyxy"])[0] > target_cx + tolerance
        ],
        key=lambda item: bbox_center(item["bbox_crop_xyxy"])[0] - target_cx,
    )
    if left and right:
        selected, direction = [left[0], right[0]], "both"
    elif len(left) >= 2:
        selected, direction = left[:2], "left"
    elif len(right) >= 2:
        selected, direction = right[:2], "right"
    else:
        return [], None
    selected.sort(key=lambda item: bbox_center(item["bbox_crop_xyxy"])[0])
    return selected, direction


def select_vertical_reference(
    instances: list[dict[str, Any]], target_bbox: list[float]
) -> tuple[list[dict[str, Any]], str] | tuple[list[dict[str, Any]], None]:
    target_cx, target_cy = bbox_center(target_bbox)
    target_width = max(1.0, target_bbox[2] - target_bbox[0])
    target_height = max(1.0, target_bbox[3] - target_bbox[1])
    below: list[tuple[tuple[float, float], dict[str, Any]]] = []
    for item in instances:
        bbox = item["bbox_crop_xyxy"]
        center_x, center_y = bbox_center(bbox)
        vertical_offset = center_y - target_cy
        overlap = max(0.0, min(target_bbox[2], bbox[2]) - max(target_bbox[0], bbox[0]))
        instance_width = max(1.0, bbox[2] - bbox[0])
        overlap_ratio = overlap / min(target_width, instance_width)
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
                item,
            )
        )
    if not below:
        return [], None
    return [min(below, key=lambda candidate: candidate[0])[1]], "up"


def draw_dashed_box(
    image: np.ndarray,
    bbox: list[float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    dash = 10
    for x in range(x1, x2, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, 3)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, 3)
    for y in range(y1, y2, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, 3)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, 3)
    cv2.putText(
        image,
        label,
        (max(0, x1), max(22, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def process_prompt_group(
    prompt_group: dict[str, Any], artifact_directory: Path
) -> int:
    current_result = prompt_group.get("current_result", {})
    all_instances = [
        item
        for item in current_result.get("instances", [])
        if isinstance(item, dict)
        and isinstance(item.get("bbox_crop_xyxy"), list)
        and item.get("duplicate_of") is None
    ]
    front_instances = [item for item in all_instances if item.get("front_selected")]
    tests: list[dict[str, Any]] = []
    current_front_path = artifact_directory / str(
        prompt_group.get("artifacts", {}).get("current_front", "current_front.jpg")
    )
    if not current_front_path.is_file():
        prompt_group["place_reference_tests"] = tests
        return 0
    overlay = read_image(current_front_path)
    combined_mask_path = artifact_directory / str(
        prompt_group.get("artifacts", {}).get(
            "current_front_mask", "current_front_mask.png"
        )
    )
    combined_mask = (
        read_image(combined_mask_path, cv2.IMREAD_GRAYSCALE)
        if combined_mask_path.is_file()
        else np.zeros(overlay.shape[:2], dtype=np.uint8)
    )

    for missing in prompt_group.get("missing_slots", []):
        target = [float(value) for value in missing["mapped_current_bbox_xyxy"]]
        product_name = str(missing.get("product_name") or "")
        place_on_top = uses_upper_confidence_pick(product_name, "SORTING")
        if place_on_top:
            selected, direction = select_vertical_reference(all_instances, target)
        else:
            selected, direction = select_horizontal_references(front_instances, target)
        slot_index = int(missing["slot_index"])
        test: dict[str, Any] = {
            "slot_index": slot_index,
            "product_name": product_name or None,
            "target_bbox_xyxy": [round(value, 2) for value in target],
            "direction": direction,
            "status": "success" if direction is not None else "insufficient_references",
            "place_on_top": place_on_top,
            "references": [],
        }
        draw_dashed_box(overlay, target, (255, 0, 255), f"TARGET S{slot_index}")
        for reference_index, instance in enumerate(selected, start=1):
            bbox = [float(value) for value in instance["bbox_crop_xyxy"]]
            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(combined_mask.shape[1], x2), min(combined_mask.shape[0], y2)
            reference_mask = np.zeros_like(combined_mask)
            if x2 > x1 and y2 > y1:
                reference_mask[y1:y2, x1:x2] = combined_mask[y1:y2, x1:x2]
            mask_name = f"place_slot_{slot_index:02d}_ref_{reference_index:02d}_mask.png"
            write_image(artifact_directory / mask_name, reference_mask)
            test["references"].append(
                {
                    "instance_index": instance.get("instance_index"),
                    "bbox_xyxy": [round(value, 2) for value in bbox],
                    "bbox_original_xyxy": instance.get("bbox_original_xyxy"),
                    "mask_artifact": mask_name,
                }
            )
            color = (0, 255, 80)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                overlay,
                f"S{slot_index} REF {reference_index} {direction}",
                (max(0, x1), min(overlay.shape[0] - 8, max(22, y1 - 8))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                color,
                2,
                cv2.LINE_AA,
            )
        tests.append(test)

    prompt_group["place_reference_tests"] = tests
    if tests:
        artifact_name = "place_references.jpg"
        write_image(artifact_directory / artifact_name, overlay)
        prompt_group.setdefault("artifacts", {})["place_references"] = artifact_name
    return len(tests)


def process_record(result_path: Path) -> tuple[int, int]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    test_count = 0
    success_count = 0
    for row in payload.get("rows", []):
        level = str(row.get("level", ""))
        for prompt_group in row.get("prompt_groups", []):
            group_index = int(prompt_group["group_index"])
            artifact_directory = result_path.parent / level / f"group_{group_index:02d}"
            count = process_prompt_group(prompt_group, artifact_directory)
            test_count += count
            success_count += sum(
                item.get("status") == "success"
                for item in prompt_group.get("place_reference_tests", [])
            )
            nested_result = artifact_directory / "result.json"
            if nested_result.is_file():
                nested = json.loads(nested_result.read_text(encoding="utf-8"))
                nested["place_reference_tests"] = prompt_group.get(
                    "place_reference_tests", []
                )
                nested["artifacts"] = prompt_group.get("artifacts", {})
                write_json(nested_result, nested)
    write_json(result_path, payload)
    return test_count, success_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    record_count = test_count = success_count = 0
    for result_path in sorted(root.glob("*/*/result.json")):
        tests, successes = process_record(result_path)
        record_count += 1
        test_count += tests
        success_count += successes
    print(
        json.dumps(
            {
                "records": record_count,
                "place_reference_tests": test_count,
                "success": success_count,
                "insufficient_references": test_count - success_count,
                "root": str(root),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
