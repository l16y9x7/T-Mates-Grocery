"""Import and batch-run the new SHORTAGE captures for the 8082 review page.

Run from the perception directory::

    python inspect/batch_inspect_shortage_new_version.py --workers 4

The source records are converted into ``real_shortage_regression`` and only
those imported records run through the complete local review pipeline:
row_detection/export -> baseline/current shelf masking -> SAM3 front-row
comparison.  Existing records from other datasets are not re-run.
"""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from batch_front_row_compare import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    process_record,
    web_server,
    write_json,
)
from batch_shelf_row_mask import (
    DEFAULT_OUTPUT_ROOT as SHELF_OUTPUT_ROOT,
    process_row as process_shelf_row,
)
from convert_shortage_output_for_review import convert_all, group_name, read_json
from export_sam_rows import export_record as export_sam_row_record


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    PERCEPTION_ROOT / "test_data" / "inspect_shortage_new_version"
)
PRINT_LOCK = threading.Lock()


def rebuild_rows_and_shelf_masks(
    records: list[tuple[str, str]],
    *,
    workers: int,
    overwrite: bool,
) -> tuple[int, list[dict[str, str]]]:
    """Rebuild row crops, then rebuild shelf masks for both image sources."""

    tasks: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for group, record in records:
        try:
            record_directory, pose_type = web_server.resolve_sam_row_record(
                group,
                record,
            )
            for source, rgb_name, depth_name in (
                ("baseline", "baseline_rgb.jpg", "baseline_depth_mm.npy"),
                ("current", "rgb.jpg", "depth_mm.npy"),
            ):
                output_directory = (
                    web_server.REAL_SHORTAGE_SAM_ROWS_ROOT / group / record
                )
                if source == "baseline":
                    output_directory = output_directory / "baseline"
                if overwrite:
                    metadata = export_sam_row_record(
                        record_directory,
                        output_directory,
                        pose_type=pose_type,
                        overwrite=True,
                        rgb_filename=rgb_name,
                        depth_filename=depth_name,
                        source_variant=source,
                    )
                else:
                    output_directory, metadata = web_server.ensure_sam_row_export(
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
                                "row_root": str(output_directory),
                                "row": row,
                            }
                        )
        except Exception as error:  # noqa: BLE001 - report every local record
            errors.append(
                {
                    "group": group,
                    "record": record,
                    "error": f"row export: {type(error).__name__}: {error}",
                }
            )

    SHELF_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                process_shelf_row,
                task,
                output_root=SHELF_OUTPUT_ROOT,
                prompt="shelf",
                detection_threshold=0.5,
                retry_threshold=0.25,
                mask_threshold=0.35,
                max_edge_component_width_ratio=0.30,
                edge_touch_tolerance_ratio=0.02,
                edge_touch_min_tolerance_px=10,
                min_spanning_component_width_ratio=0.60,
                expansion_px=10,
                overwrite=overwrite,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
                completed += 1
            except Exception as error:  # noqa: BLE001 - keep the batch running
                errors.append(
                    {
                        "group": str(task["group"]),
                        "record": str(task["record"]),
                        "error": (
                            f"shelf mask {task['source']} ROW "
                            f"{task['row'].get('row_index')}: "
                            f"{type(error).__name__}: {error}"
                        ),
                    }
                )
    return completed, errors


def discover_source_records(source_root: Path) -> list[tuple[str, str]]:
    """Return the web group/record keys represented by formal API artifacts."""

    records: list[tuple[str, str]] = []
    for source_record in sorted(source_root.iterdir(), key=lambda path: path.name):
        if not source_record.is_dir() or not (source_record / "request.json").is_file():
            continue
        request = read_json(source_record / "request.json")
        if str(request.get("task_type") or "").strip().upper() != "SHORTAGE":
            continue
        location_id = str(request.get("location_id") or "").strip()
        pose_type = str(request.get("pose_type") or "").strip()
        if not location_id or not pose_type:
            raise ValueError(f"request 缺少 location_id/pose_type: {source_record}")
        required = (
            source_record / "baseline_rgb.jpg",
            source_record / "baseline_depth_mm.npy",
            source_record / "current_rgb.jpg",
            source_record / "current_depth_mm.npy",
        )
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"record 缺少 RGB-D 文件 {missing}: {source_record}"
            )
        records.append((group_name(location_id, pose_type), source_record.name))
    return records


def run_records(
    records: list[tuple[str, str]],
    *,
    workers: int,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    web_server.load_sam_row_exporter()
    completed: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                process_record,
                group,
                record,
                output_root=DEFAULT_OUTPUT_ROOT,
                overwrite=overwrite,
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
                        "missing_slot_count": int(
                            result.get("missing_slot_count") or 0
                        ),
                        "cached": bool(result.get("cached")),
                    }
                )
                with PRINT_LOCK:
                    print(f"[OK] {group}/{record}")
            except Exception as error:  # noqa: BLE001 - keep the batch running
                errors.append(
                    {
                        "group": group,
                        "record": record,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                with PRINT_LOCK:
                    print(f"[ERROR] {group}/{record}: {error}")
    return completed, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="强制重新运行当前流程（默认）",
    )
    cache_group.add_argument(
        "--reuse-existing",
        dest="overwrite",
        action="store_false",
        help="允许复用 schema 和配置版本均一致的已有对比结果",
    )
    parser.set_defaults(overwrite=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅检查并列出新版本 records，不复制文件或调用 SAM3",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_root}")
    records = discover_source_records(source_root)
    if not records:
        raise RuntimeError("inspect_shortage_new_version 中没有有效 SHORTAGE record")
    print(f"records: {len(records)}")
    if args.dry_run:
        for group, record in records:
            print(f"{group}/{record}")
        return 0

    converted_count, review_summary = convert_all(
        source_root,
        DEFAULT_DATA_ROOT,
        # Do not expose the sparse result.json saved by the online endpoint as
        # a front-compare result.  The batch below re-runs SAM3 and writes the
        # complete overlays, masks, per-instance depth data and slot matching.
        None,
    )
    print("phase 1/3: row_detection + row export")
    print("phase 2/3: baseline/current shelf masks")
    shelf_rows, preprocessing_errors = rebuild_rows_and_shelf_masks(
        records,
        workers=max(1, int(args.workers)),
        overwrite=bool(args.overwrite),
    )
    print(f"shelf rows completed: {shelf_rows}")
    print("phase 3/3: front-row comparison")
    completed, errors = run_records(
        records,
        workers=max(1, int(args.workers)),
        overwrite=bool(args.overwrite),
    )
    errors = preprocessing_errors + errors
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_root": str(source_root),
        "data_root": str(DEFAULT_DATA_ROOT.resolve()),
        "output_root": str(DEFAULT_OUTPUT_ROOT.resolve()),
        "workers": max(1, int(args.workers)),
        "converted_records": converted_count,
        "completed_records": len(completed),
        "completed_shelf_rows": shelf_rows,
        "failed_records": len(errors),
        "records": sorted(completed, key=lambda item: (item["group"], item["record"])),
        "errors": sorted(errors, key=lambda item: (item["group"], item["record"])),
    }
    manifest_path = DEFAULT_OUTPUT_ROOT / "inspect_shortage_new_version_manifest.json"
    write_json(manifest_path, manifest)
    print(f"review summary: {review_summary}")
    print(f"batch manifest: {manifest_path}")
    print(f"completed records: {len(completed)}")
    print(f"failed records: {len(errors)}")
    print("open http://127.0.0.1:8082/sam-row-debug")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
