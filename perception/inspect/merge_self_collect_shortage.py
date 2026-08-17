"""Merge grouped self-collect shortage RGB-D into the unified regression set."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PERCEPTION_ROOT / "test_data" / "2026-08-16-self-collect-shortage-grouped"
)
DEFAULT_TARGET = PERCEPTION_ROOT / "test_data" / "real_shortage_regression"
GROUP_PATTERN = re.compile(
    r"^(?P<target>H[12]_[FB]_[LR]_INSPECT)_(?P<pose>UPPER|LOWER)$"
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def link_or_copy(source: Path, target: Path) -> str:
    if target.is_file():
        return "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "linked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def initial_scan_files(target_id: str, pose: str) -> tuple[Path, Path]:
    import sys

    if str(PERCEPTION_ROOT) not in sys.path:
        sys.path.insert(0, str(PERCEPTION_ROOT))
    from initial_scan import load_initial_scan

    scan = load_initial_scan(target_id, f"SHELF_VIEW_{pose}")
    return scan.rgb_path, scan.depth_path


def merge_dataset(source_root: Path, target_root: Path) -> dict[str, Any]:
    if not source_root.is_dir():
        raise RuntimeError(f"自采 shortage 目录不存在: {source_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    file_actions = {"linked": 0, "copied": 0, "existing": 0}

    for group_directory in sorted(source_root.iterdir(), key=lambda path: path.name):
        match = GROUP_PATTERN.fullmatch(group_directory.name)
        if not group_directory.is_dir() or match is None:
            continue
        try:
            baseline_rgb, baseline_depth = initial_scan_files(
                match.group("target"),
                match.group("pose"),
            )
        except (OSError, RuntimeError, ValueError) as error:
            errors.append({"group": group_directory.name, "error": str(error)})
            continue

        for source_record in sorted(group_directory.glob("record_*")):
            if not source_record.is_dir():
                continue
            required = {
                "rgb.jpg": source_record / "rgb.jpg",
                "depth_mm.npy": source_record / "depth_mm.npy",
                "baseline_rgb.jpg": baseline_rgb,
                "baseline_depth_mm.npy": baseline_depth,
            }
            missing = [name for name, path in required.items() if not path.is_file()]
            if missing:
                errors.append(
                    {
                        "group": group_directory.name,
                        "record": source_record.name,
                        "error": "缺少文件: " + ", ".join(missing),
                    }
                )
                continue
            target_record = target_root / group_directory.name / source_record.name
            actions = {
                name: link_or_copy(source, target_record / name)
                for name, source in required.items()
            }
            robot_state = source_record / "robot_state.json"
            if robot_state.is_file():
                actions["robot_state.json"] = link_or_copy(
                    robot_state,
                    target_record / "robot_state.json",
                )
            candidate_source = source_root / "qwen_debug" / source_record.name / "candidates.json"
            if candidate_source.is_file():
                actions["candidates.json"] = link_or_copy(
                    candidate_source,
                    target_root / "qwen_debug" / source_record.name / "candidates.json",
                )
            for action in actions.values():
                file_actions[action] += 1
            write_json(
                target_record / "source_dataset.json",
                {
                    "source_dataset": source_root.name,
                    "source_record": str(source_record.resolve()),
                    "location_id": match.group("target"),
                    "pose_type": f"SHELF_VIEW_{match.group('pose')}",
                },
            )
            records.append(
                {
                    "group": group_directory.name,
                    "record": source_record.name,
                    "actions": actions,
                }
            )

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_root": str(source_root.resolve()),
        "target_root": str(target_root.resolve()),
        "merged_records": len(records),
        "failed_records": len(errors),
        "file_actions": file_actions,
        "records": records,
        "errors": errors,
    }
    write_json(target_root / "merged_self_collect_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    manifest = merge_dataset(args.source.resolve(), args.target.resolve())
    print(f"merged records: {manifest['merged_records']}")
    print(f"failed records: {manifest['failed_records']}")
    print(f"file actions: {manifest['file_actions']}")
    return 0 if not manifest["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
