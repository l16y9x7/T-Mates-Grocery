"""Run shelf-row detection on individual images or sample directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .detector import RowDetectionConfig, detect_rows
else:
    from detector import RowDetectionConfig, detect_rows


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="one or more image files/directories",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="glob pattern used inside directories (default: *)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_data/row_detection_results"),
        help="directory for masks, annotations, and JSON",
    )
    parser.add_argument(
        "--keep-input-size",
        action="store_true",
        help="do not standardize inputs to 1280x720",
    )
    parser.add_argument(
        "--include-trailing-row",
        action="store_true",
        help="also treat the area below the last detected rail as a product row",
    )
    return parser


def _collect_images(inputs: list[Path], pattern: str) -> list[Path]:
    images: list[Path] = []
    for source in inputs:
        if source.is_file():
            if source.suffix.lower() in IMAGE_SUFFIXES:
                images.append(source)
            continue
        if source.is_dir():
            images.extend(
                path
                for path in source.glob(pattern)
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            continue
        raise FileNotFoundError(source)
    return sorted(set(images), key=lambda path: str(path).lower())


def main() -> int:
    args = build_parser().parse_args()
    images = _collect_images(args.inputs, args.pattern)
    if not images:
        raise SystemExit("No matching input images")

    config = RowDetectionConfig(
        target_size=None if args.keep_input_size else (1280, 720),
        include_trailing_row=args.include_trailing_row,
    )
    summaries = []
    for index, image_path in enumerate(images, start=1):
        group_name = image_path.parent.name
        sample_dir = args.output_dir / group_name / image_path.stem
        result = detect_rows(image_path, config)
        artifacts = result.save_debug(sample_dir)
        summaries.append(
            {
                "input": str(image_path),
                "rail_count": len(result.rails),
                "row_count": len(result.rows),
                "rails": [list(rail.bbox) for rail in result.rails],
                "rows": [list(row.bbox) for row in result.rows],
                "annotated": str(artifacts["annotated"]),
            }
        )
        print(
            f"[{index}/{len(images)}] {image_path}: "
            f"rails={len(result.rails)}, rows={len(result.rows)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
