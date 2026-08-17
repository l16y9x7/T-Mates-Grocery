"""Convert saved SHORTAGE API artifacts into the existing 8082 review dataset.

Run from the perception directory:

    python inspect/convert_shortage_output_for_review.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PERCEPTION_ROOT / "test_data" / "real_shortage_output"
DEFAULT_TARGET = PERCEPTION_ROOT / "test_data" / "real_shortage_regression"
DEFAULT_FRONT_COMPARE_TARGET = (
    PERCEPTION_ROOT / "test_data" / "real_shortage_front_compare"
)
SUMMARY_FILENAME = "shortage_inspection_batch_results.json"
DEPTH_DELTA_THRESHOLD_MM = 40.0


def read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def copy_if_present(source: Path, target: Path) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def group_name(location_id: str, pose_type: str) -> str:
    pose = pose_type.removeprefix("SHELF_VIEW_") or "UNKNOWN"
    return f"{location_id}_{pose}"


def converted_findings(
    request: dict[str, Any],
    api_result: dict[str, Any],
    source_record: Path,
) -> list[dict[str, Any]]:
    bboxes = request.get("bboxes", [])
    if not isinstance(bboxes, list):
        bboxes = []
    raw_findings = api_result.get("findings", [])
    if not isinstance(raw_findings, list):
        raw_findings = []
    by_region = {
        finding.get("region_index"): finding
        for finding in raw_findings
        if isinstance(finding, dict) and isinstance(finding.get("region_index"), int)
    }

    findings: list[dict[str, Any]] = []
    for region_index, bbox in enumerate(bboxes, start=1):
        raw = by_region.get(region_index, {})
        finding = dict(raw) if isinstance(raw, dict) else {}
        finding["region_index"] = region_index
        finding["bbox"] = bbox
        finding["product_name"] = (
            finding.get("product_name") or finding.get("shortage_product_name")
        )
        finding.pop("shortage_product_name", None)
        if isinstance(bbox, list) and len(bbox) == 4:
            x, y, width, height = bbox
            finding["center"] = [x + width / 2, y + height / 2]
        prompt_path = source_record / f"region_{region_index:02d}" / "prompt.txt"
        if prompt_path.is_file():
            finding["qwen_prompt"] = prompt_path.read_text(encoding="utf-8")
        findings.append(finding)
    return findings


def copy_record_inputs(source_record: Path, target_record: Path) -> None:
    mappings = {
        "baseline_rgb.jpg": "baseline_rgb.jpg",
        "baseline_depth_mm.npy": "baseline_depth_mm.npy",
        "current_rgb.jpg": "rgb.jpg",
        "current_depth_mm.npy": "depth_mm.npy",
        "request.json": "source_request.json",
        "result.json": "source_result.json",
        "rgbd.json": "rgbd.json",
    }
    for source_name, target_name in mappings.items():
        copy_if_present(source_record / source_name, target_record / target_name)


def copy_qwen_debug(source_record: Path, debug_record: Path) -> None:
    for filename in ("request.json", "candidates.json", "result.json", "rgbd.json"):
        copy_if_present(source_record / filename, debug_record / filename)
    for region in sorted(source_record.glob("region_*")):
        if region.is_dir():
            shutil.copytree(region, debug_record / region.name, dirs_exist_ok=True)


def convert_record(source_record: Path, target_root: Path) -> dict[str, Any] | None:
    request = read_json(source_record / "request.json")
    if str(request.get("task_type", "")).upper() != "SHORTAGE":
        return None
    api_result = read_json(source_record / "result.json", required=False)
    location_id = str(request.get("location_id") or "UNKNOWN")
    pose_type = str(request.get("pose_type") or "UNKNOWN")
    group = group_name(location_id, pose_type)
    record = source_record.name
    target_record = target_root / group / record
    copy_record_inputs(source_record, target_record)
    copy_qwen_debug(source_record, target_root / "qwen_debug" / record)

    findings = converted_findings(request, api_result, source_record)
    accepted_count = sum(finding.get("accepted") is True for finding in findings)
    if not api_result:
        status = "recognition_error"
    elif accepted_count == 0:
        status = "no_anomaly"
    elif accepted_count < len(findings):
        status = "partial"
    else:
        status = "success"

    relative_record = target_record.relative_to(target_root).as_posix()
    return {
        "schema_version": 1,
        "task_type": "SHORTAGE",
        "group": group,
        "record": record,
        "inspection_target_id": location_id,
        "location_id": location_id,
        "pose_type": pose_type,
        "source_rgb": f"{relative_record}/rgb.jpg",
        "source_depth": f"{relative_record}/depth_mm.npy",
        "baseline_rgb": str((target_record / "baseline_rgb.jpg").resolve()),
        "findings": findings,
        "expected": {"findings": []},
        "status": status,
        "has_anomaly": bool(findings),
        "bbox_format": request.get("bbox_format", ["x", "y", "width", "height"]),
        "raw_finding_count": len(findings),
        "recognized_count": accepted_count,
        "artifacts": {},
        "converted_from": str(source_record.resolve()),
        "completed_at": datetime.now(UTC).isoformat(),
    }


def convert_sam_shortage_analysis(
    source_record: Path,
    *,
    group: str,
    record: str,
    front_compare_root: Path,
) -> bool:
    """Expose a saved formal SAM shortage result on the existing compare panel."""

    api_result = read_json(source_record / "result.json", required=False)
    analysis = api_result.get("sam_shortage_analysis")
    if not isinstance(analysis, dict):
        return False
    comparisons = analysis.get("comparisons")
    if not isinstance(comparisons, list):
        return False

    rows_by_level: dict[str, list[dict[str, Any]]] = {}
    finding_details: list[dict[str, Any]] = []
    for raw_comparison in comparisons:
        if not isinstance(raw_comparison, dict):
            continue
        level = str(raw_comparison.get("level") or "").strip().upper()
        group_index = raw_comparison.get("group_index")
        if not level or not isinstance(group_index, int):
            continue
        raw_slots = raw_comparison.get("slots")
        slots = [dict(item) for item in raw_slots if isinstance(item, dict)] \
            if isinstance(raw_slots, list) else []
        missing_slots = [
            item
            for item in slots
            if str(item.get("status") or "").startswith("missing_")
        ]
        for slot in slots:
            slot.setdefault("baseline_instance_index", None)
            slot.setdefault("current_instance_index", None)
        missing_names = list(
            dict.fromkeys(
                str(item.get("product_name") or "").strip()
                for item in missing_slots
                if str(item.get("product_name") or "").strip()
            )
        )
        depth_deltas = [
            float(item["depth_delta_mm"])
            for item in slots
            if isinstance(item.get("depth_delta_mm"), (int, float))
        ]
        systematic_detected = bool(
            raw_comparison.get("systematic_depth_shift")
        )
        expected = int(raw_comparison.get("expected_front_count") or 0)
        baseline_count = int(raw_comparison.get("baseline_front_count") or 0)
        current_count = int(raw_comparison.get("current_front_count") or 0)
        prompt_group = {
            "group_index": group_index,
            "prompt": str(raw_comparison.get("prompt") or ""),
            "slot_product_names": [
                str(item.get("product_name") or "") for item in slots
            ],
            "expected_front_count": expected,
            "baseline_front_count": baseline_count,
            "current_front_count": current_count,
            "current_detection_failed": any(
                item.get("status") == "current_detection_failed" for item in slots
            ),
            "resolved_slot_count": len(slots),
            "baseline_complete": baseline_count == expected,
            "slot_coordinate_space": "source_image",
            "slot_matching_strategy": (
                "ordinal_left_to_right"
                if baseline_count == current_count
                else "monotonic_sequence_alignment"
            ),
            "depth_delta_threshold_mm": DEPTH_DELTA_THRESHOLD_MM,
            "systematic_depth_shift": {
                "detected": systematic_detected,
                "accepted_range_mm": [30.0, 80.0],
                "median_delta_mm": (
                    sorted(depth_deltas)[len(depth_deltas) // 2]
                    if depth_deltas
                    else None
                ),
                "min_delta_mm": min(depth_deltas) if depth_deltas else None,
                "max_delta_mm": max(depth_deltas) if depth_deltas else None,
                "matched_slot_count": len(depth_deltas),
            },
            "slots": slots,
            "missing_slots": missing_slots,
            "missing_product_names": missing_names,
            "artifacts": {},
            "source": "formal_sam_shortage_analysis",
        }
        rows_by_level.setdefault(level, []).append(prompt_group)
        finding_details.extend(
            {
                "shortage_product_name": name,
                "level": level,
                "group_index": group_index,
                "slot_index": slot.get("slot_index"),
                "status": slot.get("status"),
            }
            for slot in missing_slots
            for name in [str(slot.get("product_name") or "").strip()]
            if name
        )

    rows = [
        {
            "level": level,
            "baseline_row_index": None,
            "current_row_index": None,
            "prompt_groups": sorted(
                prompt_groups,
                key=lambda item: int(item["group_index"]),
            ),
        }
        for level, prompt_groups in sorted(rows_by_level.items())
    ]
    if not rows:
        return False
    unique_names = list(
        dict.fromkeys(item["shortage_product_name"] for item in finding_details)
    )
    target = front_compare_root / group / record / "result.json"
    write_json(
        target,
        {
            "schema_version": 10,
            "generated_at": datetime.now(UTC).isoformat(),
            "group": group,
            "record": record,
            "rows": rows,
            "findings": [
                {"shortage_product_name": name} for name in unique_names
            ],
            "finding_details": finding_details,
            "missing_slot_count": len(finding_details),
            "converted_from": str((source_record / "result.json").resolve()),
            "source": "formal_sam_shortage_analysis",
        },
    )
    return True


def convert_all(
    source_root: Path,
    target_root: Path,
    front_compare_root: Path | None = None,
) -> tuple[int, Path]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    summary_path = target_root / SUMMARY_FILENAME
    summary = read_json(summary_path, required=False)
    existing_results = summary.get("results", [])
    if not isinstance(existing_results, list):
        existing_results = []
    indexed_results = {
        (item.get("group"), item.get("record")): item
        for item in existing_results
        if isinstance(item, dict)
    }

    converted_count = 0
    for source_record in sorted(source_root.iterdir()):
        if not source_record.is_dir() or not (source_record / "request.json").is_file():
            continue
        result = convert_record(source_record, target_root)
        if result is None:
            continue
        indexed_results[(result["group"], result["record"])] = result
        if front_compare_root is not None:
            convert_sam_shortage_analysis(
                source_record,
                group=str(result["group"]),
                record=str(result["record"]),
                front_compare_root=front_compare_root,
            )
        converted_count += 1

    results = sorted(
        indexed_results.values(),
        key=lambda item: (str(item.get("group")), str(item.get("record"))),
    )
    status_counts = Counter(str(item.get("status") or "unknown") for item in results)
    updated_summary = {
        "schema_version": 1,
        "task_type": "SHORTAGE",
        "data_root": str(target_root.resolve()),
        "generated_at": datetime.now(UTC).isoformat(),
        "detection_only": False,
        "total_records": len(results),
        "completed_records": len(results),
        "status_counts": dict(status_counts),
        "results": results,
    }
    write_json(summary_path, updated_summary)
    return converted_count, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert saved SHORTAGE API output for the existing 8082 review page."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--front-compare-target",
        type=Path,
        default=DEFAULT_FRONT_COMPARE_TARGET,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converted_count, summary_path = convert_all(
        args.source.resolve(),
        args.target.resolve(),
        args.front_compare_target.resolve(),
    )
    print(f"converted {converted_count} SHORTAGE record(s)")
    print(f"review summary: {summary_path}")
    print("open http://127.0.0.1:8082/qwen-review and select 真实数据测试")


if __name__ == "__main__":
    main()
