"""Export row-aligned RGB-D crops from real shortage records for SAM tuning.

Run from the perception directory::

    python inspect/export_sam_rows.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PERCEPTION_ROOT / "test_data" / "real_shortage_regression"
DEFAULT_OUTPUT_ROOT = PERCEPTION_ROOT / "test_data" / "real_shortage_sam_rows"
GROUP_PATTERN = re.compile(
    r"^(?P<target>H[12]_[FB]_[LR]_INSPECT)_(?P<pose>UPPER|LOWER)$"
)
POSE_LEVELS = {
    "SHELF_VIEW_UPPER": ("L1", "L2"),
    "SHELF_VIEW_LOWER": ("L3", "L4", "L5"),
}


def load_row_detection() -> tuple[Any, Any]:
    try:
        from row_detection import RowDetectionConfig, detect_rows
    except ModuleNotFoundError:
        import sys

        if str(PERCEPTION_ROOT) not in sys.path:
            sys.path.insert(0, str(PERCEPTION_ROOT))
        from row_detection import RowDetectionConfig, detect_rows

    return RowDetectionConfig, detect_rows


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
    success, encoded = cv2.imencode(path.suffix or ".png", np.asarray(image))
    if not success:
        raise RuntimeError(f"无法编码图像: {path}")
    encoded.tofile(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def depth_preview(depth_mm: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    preview = np.zeros((*depth.shape, 3), dtype=np.uint8)
    values = depth[valid]
    if values.size == 0:
        return preview
    near, far = np.percentile(values, (2.0, 98.0))
    if far <= near:
        far = near + 1.0
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (far - depth[valid]) * 255.0 / (far - near),
        0,
        255,
    ).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def clamp_bbox(
    bbox: tuple[int, int, int, int] | list[int],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, width, height = (int(value) for value in bbox)
    image_height, image_width = image_shape
    left = max(0, min(image_width, x))
    top = max(0, min(image_height, y))
    right = max(left, min(image_width, x + width))
    bottom = max(top, min(image_height, y + height))
    if right <= left or bottom <= top:
        raise RuntimeError(f"货架层 crop 为空: {bbox}")
    return left, top, right - left, bottom - top


def fit_shelf_boundary_model(
    rails: list[Any],
    *,
    image_width: int,
) -> dict[str, Any] | None:
    """Fit the perspective shelf sides from the changing red-rail spans."""

    samples: list[tuple[float, float, float]] = []
    for rail in rails:
        line = getattr(rail, "line", None)
        if isinstance(line, (tuple, list)) and len(line) == 4:
            x1, _, x2, _ = (float(value) for value in line)
        else:
            bbox = getattr(rail, "bbox", None)
            if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                continue
            x1 = float(bbox[0])
            x2 = float(bbox[0] + bbox[2] - 1)
        left, right = sorted((x1, x2))
        if right - left < image_width * 0.30:
            continue
        samples.append((float(getattr(rail, "y_center")), left, right))
    if not samples:
        return None

    def robust_line(value_index: int) -> tuple[float, float]:
        slopes = [
            (second[value_index] - first[value_index]) / (second[0] - first[0])
            for first_index, first in enumerate(samples)
            for second in samples[first_index + 1 :]
            if abs(second[0] - first[0]) >= 2
        ]
        slope = float(np.median(slopes)) if slopes else 0.0
        intercept = float(
            np.median([sample[value_index] - slope * sample[0] for sample in samples])
        )
        return slope, intercept

    left_slope, left_intercept = robust_line(1)
    right_slope, right_intercept = robust_line(2)
    return {
        "left": [left_slope, left_intercept],
        "right": [right_slope, right_intercept],
        "padding_px": max(4, round(image_width * 0.008)),
        "rail_samples": [
            {"y": round(y, 2), "left": round(left, 2), "right": round(right, 2)}
            for y, left, right in samples
        ],
    }


def shelf_bounds_at_y(
    model: dict[str, Any] | None,
    y: float,
    image_width: int,
) -> tuple[int, int]:
    if model is None:
        return 0, image_width
    left_slope, left_intercept = model["left"]
    right_slope, right_intercept = model["right"]
    padding = int(model["padding_px"])
    left = int(np.floor(left_slope * y + left_intercept)) - padding
    right = int(np.ceil(right_slope * y + right_intercept)) + padding + 1
    left = max(0, min(image_width, left))
    right = max(left, min(image_width, right))
    if right - left < image_width * 0.30:
        return 0, image_width
    return left, right


def perspective_row_crop(
    image: np.ndarray,
    depth: np.ndarray,
    bbox: tuple[int, int, int, int],
    boundary_model: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop one row and black out pixels outside its perspective shelf sides."""

    _, y, _, height = bbox
    image_width = image.shape[1]
    top_left, top_right = shelf_bounds_at_y(boundary_model, y, image_width)
    bottom_left, bottom_right = shelf_bounds_at_y(
        boundary_model,
        y + height - 1,
        image_width,
    )
    crop_left = min(top_left, bottom_left)
    crop_right = max(top_right, bottom_right)
    if crop_right <= crop_left:
        crop_left, crop_right = 0, image_width

    rgb_crop = image[y : y + height, crop_left:crop_right].copy()
    depth_crop = np.asarray(depth)[y : y + height, crop_left:crop_right].copy()
    shelf_mask = np.zeros((height, crop_right - crop_left), dtype=np.uint8)
    polygon = np.array(
        [
            [top_left - crop_left, 0],
            [top_right - crop_left - 1, 0],
            [bottom_right - crop_left - 1, height - 1],
            [bottom_left - crop_left, height - 1],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(shelf_mask, polygon, 255)
    rgb_crop[shelf_mask == 0] = 0
    depth_crop[shelf_mask == 0] = 0
    return (
        rgb_crop,
        depth_crop,
        shelf_mask,
        (crop_left, y, crop_right - crop_left, height),
    )


def export_record(
    record_directory: Path,
    output_directory: Path,
    *,
    pose_type: str,
    overwrite: bool,
) -> dict[str, Any]:
    rgb_path = record_directory / "rgb.jpg"
    depth_path = record_directory / "depth_mm.npy"
    if not rgb_path.is_file() or not depth_path.is_file():
        missing = [
            path.name for path in (rgb_path, depth_path) if not path.is_file()
        ]
        raise RuntimeError("缺少实测 RGB-D 文件: " + ", ".join(missing))

    image = read_image(rgb_path)
    try:
        depth = np.load(depth_path, allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"读取深度失败 {depth_path}: {error}") from error
    if depth.ndim != 2 or depth.shape != image.shape[:2]:
        raise RuntimeError(
            f"RGB/深度尺寸不一致: rgb={image.shape[:2]}, depth={depth.shape}"
        )
    if not np.issubdtype(depth.dtype, np.number):
        raise RuntimeError(f"深度必须是数值数组: {depth_path}")

    RowDetectionConfig, detect_rows = load_row_detection()
    detection = detect_rows(
        image,
        RowDetectionConfig(target_size=None, pose_type=pose_type),
    )
    expected_levels = POSE_LEVELS[pose_type]
    if not detection.rows:
        raise RuntimeError("row_detection 没有检测到任何货架层")
    boundary_model = fit_shelf_boundary_model(
        list(getattr(detection, "rails", [])),
        image_width=image.shape[1],
    )

    rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(detection.rows, start=1):
        x, y, width, height = clamp_bbox(row.bbox, image.shape[:2])
        level = (
            expected_levels[row_number - 1]
            if row_number <= len(expected_levels)
            else None
        )
        row_name = f"row_{row_number:02d}_{level or 'UNKNOWN'}"
        row_directory = output_directory / row_name
        rgb_output = row_directory / "rgb.jpg"
        depth_output = row_directory / "depth_mm.npy"
        shelf_mask_output = row_directory / "shelf_mask.png"
        valid_mask_output = row_directory / "valid_depth_mask.png"
        preview_output = row_directory / "depth_preview.png"
        if not overwrite and any(
            path.exists()
            for path in (
                rgb_output,
                depth_output,
                shelf_mask_output,
                valid_mask_output,
                preview_output,
            )
        ):
            raise RuntimeError(f"输出已存在，请使用 --overwrite: {row_directory}")

        rgb_crop, depth_crop, shelf_mask, crop_bbox = perspective_row_crop(
            image,
            depth,
            (x, y, width, height),
            boundary_model,
        )
        x, y, width, height = crop_bbox
        valid = np.isfinite(depth_crop) & (depth_crop > 0)
        write_image(rgb_output, rgb_crop)
        depth_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_output, depth_crop, allow_pickle=False)
        write_image(shelf_mask_output, shelf_mask)
        write_image(valid_mask_output, valid.astype(np.uint8) * 255)
        write_image(preview_output, depth_preview(depth_crop))
        rows.append(
            {
                "row_index": row_number,
                "detected_row_index": int(row.index),
                "level": level,
                "crop_bbox_xywh": [x, y, width, height],
                "crop_origin_xy": [x, y],
                "rgb": f"{row_name}/rgb.jpg",
                "depth_mm": f"{row_name}/depth_mm.npy",
                "shelf_mask": f"{row_name}/shelf_mask.png",
                "valid_depth_mask": f"{row_name}/valid_depth_mask.png",
                "depth_preview": f"{row_name}/depth_preview.png",
                "depth_dtype": str(depth_crop.dtype),
                "valid_depth_pixels": int(np.count_nonzero(valid)),
                "valid_depth_ratio": round(
                    float(np.count_nonzero(valid)) / max(1, valid.size),
                    6,
                ),
            }
        )

    detection_overlay = detection.draw()
    if boundary_model is not None:
        for side in ("left", "right"):
            top = shelf_bounds_at_y(boundary_model, 0, image.shape[1])
            bottom = shelf_bounds_at_y(
                boundary_model,
                image.shape[0] - 1,
                image.shape[1],
            )
            side_index = 0 if side == "left" else 1
            cv2.line(
                detection_overlay,
                (top[side_index], 0),
                (bottom[side_index], image.shape[0] - 1),
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )
    write_image(output_directory / "row_detection.jpg", detection_overlay)
    metadata = {
        "schema_version": 1,
        "source_record": str(record_directory.resolve()),
        "source_rgb": str(rgb_path.resolve()),
        "source_depth_mm": str(depth_path.resolve()),
        "pose_type": pose_type,
        "source_image_size": [int(image.shape[1]), int(image.shape[0])],
        "expected_row_count": len(expected_levels),
        "detected_row_count": len(rows),
        "shelf_boundary_model": boundary_model,
        "row_detection": detection.as_dict(),
        "rows": rows,
    }
    write_json(output_directory / "rows.json", metadata)
    return metadata


def export_dataset(
    data_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    if not data_root.is_dir():
        raise RuntimeError(f"实测数据目录不存在: {data_root}")
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for group_directory in sorted(data_root.iterdir(), key=lambda path: path.name):
        match = GROUP_PATTERN.fullmatch(group_directory.name)
        if not group_directory.is_dir() or match is None:
            continue
        pose_type = f"SHELF_VIEW_{match.group('pose')}"
        for record_directory in sorted(
            group_directory.iterdir(), key=lambda path: path.name
        ):
            if not record_directory.is_dir():
                continue
            relative_record = (
                Path(group_directory.name) / record_directory.name
            )
            try:
                metadata = export_record(
                    record_directory,
                    output_root / relative_record,
                    pose_type=pose_type,
                    overwrite=overwrite,
                )
                records.append(
                    {
                        "group": group_directory.name,
                        "record": record_directory.name,
                        "pose_type": pose_type,
                        "rows_json": f"{relative_record.as_posix()}/rows.json",
                        "detected_row_count": metadata["detected_row_count"],
                    }
                )
            except (RuntimeError, ValueError, cv2.error) as error:
                errors.append(
                    {
                        "group": group_directory.name,
                        "record": record_directory.name,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_root": str(data_root.resolve()),
        "output_root": str(output_root.resolve()),
        "completed_records": len(records),
        "failed_records": len(errors),
        "exported_rows": sum(item["detected_row_count"] for item in records),
        "records": records,
        "errors": errors,
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = export_dataset(
        args.data_root.resolve(),
        args.output_root.resolve(),
        overwrite=args.overwrite,
    )
    print(f"completed records: {manifest['completed_records']}")
    print(f"exported rows: {manifest['exported_rows']}")
    print(f"failed records: {manifest['failed_records']}")
    print(f"manifest: {args.output_root.resolve() / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
