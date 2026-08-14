"""Run the Place Locate RGB registration stage on inspection paired images.

The inspection misplaced dataset contains RGB only, so this command validates
static keypoint filtering and homography initialization.  It deliberately marks
SE(3) estimation as skipped instead of manufacturing a 6D transform without
depth and camera intrinsics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .pose_transfer import PoseTransferError
from .registration import draw_registration_matches, register_rgb_images


PERCEPTION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PERCEPTION_ROOT / "test_data" / "inspect_misplaced_paired"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASET_ROOT / "place_locate_registration"


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"unsupported output image extension: {suffix}")
    parameters = [cv2.IMWRITE_JPEG_QUALITY, 95] if suffix in {".jpg", ".jpeg"} else []
    success, encoded = cv2.imencode(suffix, image, parameters)
    if not success:
        raise OSError(f"cannot write image: {path}")
    path.write_bytes(encoded.tobytes())


def discover_pairs(dataset_root: Path) -> list[tuple[int, Path, Path]]:
    pairs: list[tuple[int, Path, Path]] = []
    for reference_path in sorted(dataset_root.glob("*_1.jpg")):
        prefix = reference_path.stem.rsplit("_", 1)[0]
        try:
            pair_number = int(prefix)
        except ValueError:
            continue
        current_path = dataset_root / f"{pair_number}_2.jpg"
        if current_path.is_file():
            pairs.append((pair_number, reference_path, current_path))
    return pairs


def run_pair(
    pair_number: int,
    reference_path: Path,
    current_path: Path,
    output_root: Path,
) -> dict[str, object]:
    reference = _read_image(reference_path)
    current = _read_image(current_path)
    pair_root = output_root / f"pair_{pair_number}"
    pair_root.mkdir(parents=True, exist_ok=True)
    try:
        result = register_rgb_images(reference, current)
    except (PoseTransferError, ValueError) as error:
        payload: dict[str, object] = {
            "pair_number": pair_number,
            "reference": str(reference_path),
            "current": str(current_path),
            "rgb_registration_success": False,
            "error": str(error),
            "se3_status": "skipped_missing_depth_and_intrinsics",
        }
    else:
        _write_image(pair_root / "01_reference.jpg", reference)
        _write_image(pair_root / "02_current.jpg", current)
        _write_image(pair_root / "03_aligned_current.jpg", result.aligned_current)
        _write_image(
            pair_root / "04_change_mask_reference.png",
            result.change_mask_reference,
        )
        _write_image(
            pair_root / "05_static_mask_reference.png",
            result.static_mask_reference,
        )
        _write_image(
            pair_root / "06_static_feature_matches.jpg",
            draw_registration_matches(reference, current, result),
        )
        payload = {
            "pair_number": pair_number,
            "reference": str(reference_path),
            "current": str(current_path),
            "rgb_registration_success": True,
            **result.as_dict(),
            "se3_status": "skipped_missing_depth_and_intrinsics",
            "artifacts": {
                "reference": "01_reference.jpg",
                "current": "02_current.jpg",
                "aligned_current": "03_aligned_current.jpg",
                "change_mask_reference": "04_change_mask_reference.png",
                "static_mask_reference": "05_static_mask_reference.png",
                "static_feature_matches": "06_static_feature_matches.jpg",
            },
        }
    (pair_root / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Place Locate RGB registration on inspection pairs"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    pairs = discover_pairs(args.data_root)
    if not pairs:
        parser.error(f"no *_1.jpg / *_2.jpg pairs found in {args.data_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = [
        run_pair(pair_number, reference, current, args.output_root)
        for pair_number, reference, current in pairs
    ]
    summary = {
        "dataset_root": str(args.data_root.resolve()),
        "pair_count": len(results),
        "rgb_registration_success_count": sum(
            bool(result["rgb_registration_success"]) for result in results
        ),
        "se3_status": "skipped_missing_depth_and_intrinsics",
        "pairs": results,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["rgb_registration_success_count"] == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
