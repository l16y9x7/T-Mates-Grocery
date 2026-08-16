"""Batch-run the current shortage inspection pipeline on grouped RGB-D records."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
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

from initial_scan import InitialScan, load_initial_scan, load_slot_target_mapping  # noqa: E402


DEFAULT_DATA_ROOT = (
    PERCEPTION_ROOT / "test_data" / "2026-08-16-self-collect-shortage-grouped"
)
DEFAULT_SUMMARY_NAME = "shortage_inspection_batch_results.json"
RESULT_DIRECTORY_NAME = "shortage_inspection"
GROUP_PATTERN = re.compile(
    r"^(?P<target>H[12]_[FB]_[LR]_INSPECT)_(?P<pose>UPPER|LOWER)$"
)
RECORD_PATTERN = re.compile(r"^record_\d{8}_\d{6}_\d{6}$")


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


def validate_depth(record_directory: Path, image_shape: tuple[int, int]) -> tuple[Path, int]:
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
    valid_pixels = int(np.count_nonzero(np.isfinite(depth) & (depth > 0)))
    return depth_path, valid_pixels


def parse_group_name(group_name: str) -> tuple[str, str]:
    match = GROUP_PATTERN.fullmatch(group_name)
    if match is None:
        raise RuntimeError(f"巡检分组名称不合法: {group_name}")
    return match.group("target"), f"SHELF_VIEW_{match.group('pose')}"


def representative_location_id(
    inspection_target_id: str,
    pose_type: str,
    slot_mapping: dict[str, str],
) -> str:
    """Choose a real slot accepted by the SKU candidate service."""

    level = 1 if pose_type == "SHELF_VIEW_UPPER" else 3
    marker = f"_L{level}_"
    slots = sorted(
        slot
        for slot, target in slot_mapping.items()
        if target == inspection_target_id and marker in slot
    )
    if not slots:
        raise RuntimeError(
            f"{inspection_target_id} 没有可用于 {pose_type} 的 SKU 查询货位"
        )
    return slots[0]


def discover_records(
    data_root: Path,
    *,
    groups: set[str] | None = None,
    record_name: str | None = None,
) -> list[dict[str, Any]]:
    if not data_root.is_dir():
        raise RuntimeError(f"批测数据目录不存在: {data_root}")
    slot_mapping = load_slot_target_mapping()
    records: list[dict[str, Any]] = []
    for group_directory in sorted(data_root.iterdir(), key=lambda path: path.name):
        if not group_directory.is_dir() or GROUP_PATTERN.fullmatch(group_directory.name) is None:
            continue
        if groups is not None and group_directory.name not in groups:
            continue
        target_id, pose_type = parse_group_name(group_directory.name)
        location_id = representative_location_id(target_id, pose_type, slot_mapping)
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
                    "location_id": location_id,
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
        depth_path, valid_depth_pixels = validate_depth(
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
        response = execution.response
        reviewed_by_region: dict[int, Any] = {}
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
            except INSPECT_API.QwenReviewError as error:
                recognition_error = {
                    "stage": error.stage,
                    "message": str(error),
                }

        combined_mask = np.zeros(execution.review_mask.shape, dtype=np.uint8)
        findings: list[dict[str, Any]] = []
        for region_index, finding in enumerate(response.findings, start=1):
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
                    "product_name": product_name,
                    "confidence": (
                        reviewed.confidence if reviewed is not None else None
                    ),
                }
            )

        output_directory.mkdir(parents=True, exist_ok=True)
        aligned_path = output_directory / "aligned_current.jpg"
        combined_mask_path = output_directory / "combined_mask.png"
        overlay_path = output_directory / "overlay.jpg"
        write_image(aligned_path, execution.review_image)
        write_image(combined_mask_path, combined_mask)
        write_image(
            overlay_path,
            build_overlay(execution.review_image, combined_mask, findings),
        )

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
                "findings": findings,
                "recognized_count": recognized_count,
                "recognition_error": recognition_error,
                "artifacts": {
                    "aligned_current": relative_path(aligned_path, data_root),
                    "combined_mask": relative_path(combined_mask_path, data_root),
                    "overlay": relative_path(overlay_path, data_root),
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

    reviewer = None
    if not args.detection_only:
        reviewer_kwargs: dict[str, Any] = {
            "debug_root": data_root / "qwen_debug",
        }
        if args.qwen_url:
            reviewer_kwargs["qwen_url"] = args.qwen_url
        if args.sku_base_url:
            reviewer_kwargs["sku_base_url"] = args.sku_base_url
        reviewer = INSPECT_API.QwenReviewer(**reviewer_kwargs)

    scans: dict[str, InitialScan] = {}
    summary_path = data_root / DEFAULT_SUMMARY_NAME
    for index, entry in enumerate(records, start=1):
        group = entry["group"]
        if group not in scans:
            scans[group] = load_initial_scan(
                entry["inspection_target_id"],
                entry["pose_type"],
            )
        result = run_record(
            entry,
            data_root=data_root,
            initial_scan=scans[group],
            reviewer=reviewer,
            detection_only=args.detection_only,
            overwrite=args.overwrite,
        )
        print(
            f"[{index}/{len(records)}] {group}/{entry['record']}: "
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
                total_records=len(discover_records(data_root)),
                detection_only=args.detection_only,
            ),
        )

    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
