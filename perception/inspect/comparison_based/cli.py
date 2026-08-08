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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ComparisonConfig(
        reference_item_area=args.item_area,
        enable_registration=not args.no_registration,
    )
    result = detect_shortage(args.baseline, args.current, config)

    if args.output:
        write_image(args.output, result.draw(args.baseline))
    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)
        write_image(args.debug_dir / "aligned_current.jpg", result.aligned_current)
        write_image(args.debug_dir / "difference.png", result.difference)
        write_image(args.debug_dir / "mask.png", result.mask)

    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
