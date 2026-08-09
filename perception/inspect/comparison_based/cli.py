"""Command-line runner for comparison-based shortage detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .detector import ComparisonConfig, detect_shortage, write_image
else:
    from detector import ComparisonConfig, detect_shortage, write_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="full-shelf reference image")
    parser.add_argument("current", type=Path, help="image captured after picking")
    parser.add_argument("--output", type=Path, help="annotated output image")
    parser.add_argument("--debug-dir", type=Path, help="save aligned/diff/mask images")
    parser.add_argument(
        "--item-area",
        type=float,
        help="single product area in baseline pixels; threshold is 0.8 times this",
    )
    parser.add_argument(
        "--no-registration",
        action="store_true",
        help="disable ORB + homography alignment for a truly fixed camera",
    )
    parser.add_argument(
        "--keep-input-size",
        action="store_true",
        help="do not standardize both inputs to 1280x720",
    )
    parser.add_argument(
        "--task-type",
        choices=("shortage", "misplaced"),
        default="shortage",
        help="misplaced suppresses luminance-only residual boxes",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ComparisonConfig(
        target_size=None if args.keep_input_size else (1280, 720),
        reference_item_area=args.item_area,
        enable_registration=not args.no_registration,
        min_chroma_dominance_ratio=(
            0.35 if args.task_type == "misplaced" else 0.0
        ),
        difference_mode=("chroma" if args.task_type == "misplaced" else "hybrid"),
    )
    result = detect_shortage(args.baseline, args.current, config)
    response = result.as_dict()

    if args.output:
        write_image(args.output, result.draw(args.baseline))
    if args.debug_dir:
        artifacts = result.save_debug(args.debug_dir, args.baseline)
        response["debug_artifacts"] = {
            key: str(path) for key, path in artifacts.items()
        }

    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
