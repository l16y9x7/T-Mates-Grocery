"""Build per-row shelf masks with SAM3 and black out everything else.

Run from the perception directory::

    python inspect/batch_shelf_row_mask.py --workers 4 --overwrite

For every baseline/current row crop in ``real_shortage_regression`` the script
calls SAM3 with the ``shelf`` prompt, splits every returned mask into connected
components, and keeps the largest component that:

* has pixels on both the left and right sides of the image center, and
* has most of its pixels in the lower half of the row image.

Other non-red components wider than 10% of the image and not touching either
horizontal edge are merged into the primary component before the retained ROI
is constructed.

The selected binary mask and the RGB image with all other pixels set to black
are written under ``test_data/real_shortage_shelf_rows``.  For every image
column covered by the selected shelf component, pixels above that column's
maximum mask y are retained; the retained ROI is then expanded by 10 px and
everything else is set to black.  This is an offline batch utility; it is not
used by the formal inspection endpoint.
"""

from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT_ROOT = PERCEPTION_ROOT / "test_data" / "real_shortage_shelf_rows"
SCHEMA_VERSION = 2
PRINT_LOCK = threading.Lock()


def discover_records(data_root: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    for group_directory in sorted(data_root.iterdir(), key=lambda path: path.name):
        if (
            not group_directory.is_dir()
            or web_server.INITIAL_SCAN_DIRECTORY_PATTERN.fullmatch(
                group_directory.name
            )
            is None
        ):
            continue
        for record_directory in sorted(
            group_directory.iterdir(),
            key=lambda path: path.name,
        ):
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
    success, encoded = cv2.imencode(path.suffix or ".png", np.asarray(image))
    if not success:
        raise RuntimeError(f"无法编码图像: {path}")
    encoded.tofile(path)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise RuntimeError(f"无法读取图像: {path}")
    return image


def build_retained_roi_mask(
    shelf_component_mask: np.ndarray,
    *,
    expansion_px: int,
) -> np.ndarray:
    """Keep pixels above the shelf component's per-column bottom boundary."""

    component = np.asarray(shelf_component_mask) > 0
    height, width = component.shape
    retained = np.zeros((height, width), dtype=np.uint8)
    supported_x = np.flatnonzero(np.any(component, axis=0))
    if supported_x.size == 0:
        return retained

    bottom_y = np.asarray(
        [int(np.flatnonzero(component[:, x]).max()) for x in supported_x],
        dtype=np.float32,
    )
    left = int(supported_x.min())
    right = int(supported_x.max())
    span_x = np.arange(left, right + 1, dtype=np.float32)
    interpolated_bottom = np.rint(
        np.interp(span_x, supported_x.astype(np.float32), bottom_y)
    ).astype(np.int32)
    for x, boundary_y in zip(range(left, right + 1), interpolated_bottom):
        retained[: min(height, int(boundary_y) + 1), x] = 255

    if expansion_px > 0:
        kernel_size = expansion_px * 2 + 1
        retained = cv2.dilate(
            retained,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            ),
            iterations=1,
        )
    return retained


def component_candidates(
    sam_result: dict[str, Any],
    rgb: np.ndarray,
    *,
    min_lower_half_ratio: float,
    min_merge_width_ratio: float,
    max_red_pixel_ratio: float,
) -> list[dict[str, Any]]:
    height, width = rgb.shape[:2]
    center_x = width / 2.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    red_pixels = (
        ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170))
        & (hsv[:, :, 1] >= 80)
        & (hsv[:, :, 2] >= 50)
    )
    candidates: list[dict[str, Any]] = []

    for instance_index, raw_instance in enumerate(
        sam_result.get("instances", []),
        start=1,
    ):
        if not isinstance(raw_instance, dict):
            continue
        mask = web_server.decode_sam_row_mask(
            raw_instance.get("mask_png_base64"),
            (height, width),
        )
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.where(mask > 0, 1, 0).astype(np.uint8),
            connectivity=8,
        )
        for component_index in range(1, component_count):
            x = int(stats[component_index, cv2.CC_STAT_LEFT])
            y = int(stats[component_index, cv2.CC_STAT_TOP])
            component_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            component = labels == component_index
            lower_pixels = int(np.count_nonzero(component[height // 2 :, :]))
            lower_half_ratio = float(lower_pixels) / float(area)
            right = x + component_width
            crosses_image_center = x < center_x < right
            width_ratio = float(component_width) / float(width)
            touches_horizontal_edge = x <= 0 or right >= width
            red_pixel_ratio = float(np.count_nonzero(red_pixels & component)) / float(
                area
            )
            merge_eligible = (
                width_ratio > min_merge_width_ratio
                and not touches_horizontal_edge
                and red_pixel_ratio < max_red_pixel_ratio
            )
            candidates.append(
                {
                    "instance_index": instance_index,
                    "component_index": component_index,
                    "score": (
                        round(float(raw_instance["score"]), 6)
                        if isinstance(raw_instance.get("score"), (int, float))
                        else None
                    ),
                    "area_px": area,
                    "bbox_xywh": [x, y, component_width, component_height],
                    "width_ratio": round(width_ratio, 6),
                    "lower_half_pixels": lower_pixels,
                    "lower_half_ratio": round(lower_half_ratio, 6),
                    "red_pixel_ratio": round(red_pixel_ratio, 6),
                    "touches_horizontal_edge": touches_horizontal_edge,
                    "crosses_image_center": crosses_image_center,
                    "merge_eligible": merge_eligible,
                    "valid": (
                        crosses_image_center
                        and lower_half_ratio >= min_lower_half_ratio
                    ),
                    "mask": component,
                }
            )
    return candidates


def process_row(
    task: dict[str, Any],
    *,
    output_root: Path,
    prompt: str,
    detection_threshold: float,
    retry_threshold: float,
    mask_threshold: float,
    min_lower_half_ratio: float,
    min_merge_width_ratio: float,
    max_red_pixel_ratio: float,
    expansion_px: int,
    overwrite: bool,
) -> dict[str, Any]:
    group = str(task["group"])
    record = str(task["record"])
    source = str(task["source"])
    row = task["row"]
    row_index = int(row["row_index"])
    level = str(row.get("level") or f"ROW_{row_index}")
    output_directory = (
        output_root
        / group
        / record
        / source
        / f"row_{row_index:02d}_{level}"
    )
    result_path = output_directory / "result.json"
    if result_path.is_file() and not overwrite:
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        return {**cached, "cached": True}

    row_root = Path(task["row_root"])
    rgb_path = web_server.resolve_descendant(row_root, str(row["rgb"]))
    rgb = read_image(rgb_path)
    attempts: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    merged_candidates: list[dict[str, Any]] = []

    thresholds = [detection_threshold]
    if retry_threshold < detection_threshold:
        thresholds.append(retry_threshold)
    for threshold in thresholds:
        sam_result = web_server.call_sam3_image_path(
            rgb_path,
            prompt,
            threshold=threshold,
            mask_threshold=mask_threshold,
        )
        candidates = component_candidates(
            sam_result,
            rgb,
            min_lower_half_ratio=min_lower_half_ratio,
            min_merge_width_ratio=min_merge_width_ratio,
            max_red_pixel_ratio=max_red_pixel_ratio,
        )
        valid = [candidate for candidate in candidates if candidate["valid"]]
        attempts.append(
            {
                "threshold": threshold,
                "instance_count": len(sam_result.get("instances", [])),
                "component_count": len(candidates),
                "valid_component_count": len(valid),
            }
        )
        all_candidates = candidates
        if valid:
            selected = max(valid, key=lambda candidate: int(candidate["area_px"]))
            merged_candidates = [
                candidate
                for candidate in candidates
                if candidate is selected or candidate["merge_eligible"]
            ]
            break

    shelf_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for candidate in merged_candidates:
        shelf_mask[candidate["mask"]] = 255
    retained_mask = build_retained_roi_mask(
        shelf_mask,
        expansion_px=expansion_px,
    )
    filtered_rgb = np.zeros_like(rgb)
    filtered_rgb[retained_mask > 0] = rgb[retained_mask > 0]

    write_image(output_directory / "shelf_mask.png", shelf_mask)
    write_image(output_directory / "retained_mask.png", retained_mask)
    write_image(output_directory / "shelf_filtered.jpg", filtered_rgb)
    serializable_candidates = [
        {key: value for key, value in candidate.items() if key != "mask"}
        for candidate in all_candidates
    ]
    selected_serializable = (
        {key: value for key, value in selected.items() if key != "mask"}
        if selected is not None
        else None
    )
    merged_serializable = [
        {
            "instance_index": candidate["instance_index"],
            "component_index": candidate["component_index"],
            "bbox_xywh": candidate["bbox_xywh"],
            "area_px": candidate["area_px"],
            "width_ratio": candidate["width_ratio"],
            "red_pixel_ratio": candidate["red_pixel_ratio"],
            "is_primary": candidate is selected,
        }
        for candidate in merged_candidates
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "group": group,
        "record": record,
        "source": source,
        "row_index": row_index,
        "level": level,
        "prompt": prompt,
        "status": "success" if selected is not None else "no_valid_component",
        "source_rgb": str(rgb_path.resolve()),
        "image_size": [rgb.shape[1], rgb.shape[0]],
        "selection_rule": {
            "must_cross_image_center": True,
            "min_lower_half_ratio": min_lower_half_ratio,
            "ranking": "largest_valid_connected_component",
            "mask_threshold": mask_threshold,
            "merge_rule": {
                "min_width_ratio_exclusive": min_merge_width_ratio,
                "must_not_touch_horizontal_edge": True,
                "max_red_pixel_ratio_exclusive": max_red_pixel_ratio,
            },
            "retained_region": "above_per_column_max_y",
            "expansion_px": expansion_px,
        },
        "attempts": attempts,
        "selected_component": selected_serializable,
        "merged_components": merged_serializable,
        "components": serializable_candidates,
        "artifacts": {
            "shelf_mask": "shelf_mask.png",
            "retained_mask": "retained_mask.png",
            "shelf_filtered": "shelf_filtered.jpg",
        },
    }
    write_json(result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--group", default="", help="只处理指定 view group")
    parser.add_argument("--record", default="", help="只处理指定 record")
    parser.add_argument(
        "--source",
        choices=("both", "baseline", "current"),
        default="both",
    )
    parser.add_argument("--prompt", default="shelf")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--retry-threshold", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.35)
    parser.add_argument("--min-lower-half-ratio", type=float, default=0.5)
    parser.add_argument("--min-merge-width-ratio", type=float, default=0.1)
    parser.add_argument("--max-red-pixel-ratio", type=float, default=0.5)
    parser.add_argument("--expansion-px", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.min_lower_half_ratio <= 1.0:
        raise ValueError("--min-lower-half-ratio 必须在 [0, 1] 内")
    if args.expansion_px < 0:
        raise ValueError("--expansion-px 不能小于 0")
    if not 0.0 <= args.mask_threshold <= 1.0:
        raise ValueError("--mask-threshold 必须在 [0, 1] 内")
    if not 0.0 <= args.min_merge_width_ratio <= 1.0:
        raise ValueError("--min-merge-width-ratio 必须在 [0, 1] 内")
    if not 0.0 <= args.max_red_pixel_ratio <= 1.0:
        raise ValueError("--max-red-pixel-ratio 必须在 [0, 1] 内")
    if not 0.0 <= args.retry_threshold <= args.threshold <= 1.0:
        raise ValueError("SAM3 threshold 必须满足 0 <= retry <= threshold <= 1")

    records = [
        item
        for item in discover_records(DEFAULT_DATA_ROOT)
        if (not args.group or item[0] == args.group)
        and (not args.record or item[1] == args.record)
    ]
    if not records:
        raise RuntimeError("没有找到包含 baseline/current RGB-D 的 record")
    sources = ("baseline", "current") if args.source == "both" else (args.source,)
    tasks: list[dict[str, Any]] = []
    for group, record in records:
        for source in sources:
            row_root, metadata = web_server.ensure_sam_row_export(
                group,
                record,
                source,
            )
            for row in metadata.get("rows", []):
                if isinstance(row, dict):
                    tasks.append(
                        {
                            "group": group,
                            "record": record,
                            "source": source,
                            "row_root": str(row_root),
                            "row": row,
                        }
                    )

    if args.dry_run:
        print(f"rows: {len(tasks)}")
        for task in tasks:
            row = task["row"]
            print(
                f"{task['group']}/{task['record']}/{task['source']}/"
                f"ROW {row['row_index']} {row.get('level', '')}"
            )
        return 0

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                process_row,
                task,
                output_root=output_root,
                prompt=args.prompt,
                detection_threshold=args.threshold,
                retry_threshold=args.retry_threshold,
                mask_threshold=args.mask_threshold,
                min_lower_half_ratio=args.min_lower_half_ratio,
                min_merge_width_ratio=args.min_merge_width_ratio,
                max_red_pixel_ratio=args.max_red_pixel_ratio,
                expansion_px=args.expansion_px,
                overwrite=args.overwrite,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            row = task["row"]
            label = (
                f"{task['group']}/{task['record']}/{task['source']}/"
                f"ROW {row['row_index']}"
            )
            try:
                result = future.result()
                completed.append(
                    {
                        "group": result["group"],
                        "record": result["record"],
                        "source": result["source"],
                        "row_index": result["row_index"],
                        "level": result["level"],
                        "status": result["status"],
                    }
                )
                with PRINT_LOCK:
                    print(f"[OK] {label}: {result['status']}")
            except Exception as error:  # noqa: BLE001 - continue batch
                errors.append({"task": label, "error": f"{type(error).__name__}: {error}"})
                with PRINT_LOCK:
                    print(f"[ERROR] {label}: {error}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "prompt": args.prompt,
        "completed_rows": len(completed),
        "failed_rows": len(errors),
        "rows": sorted(
            completed,
            key=lambda item: (
                item["group"],
                item["record"],
                item["source"],
                item["row_index"],
            ),
        ),
        "errors": errors,
    }
    write_json(output_root / "manifest.json", manifest)
    print(f"completed rows: {len(completed)}")
    print(f"failed rows: {len(errors)}")
    print(f"manifest: {output_root / 'manifest.json'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
