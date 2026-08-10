"""Generate review prompts and expanded bbox crops for the bundled pair samples.

This utility deliberately does not call Qwen.  It runs the comparison detector,
loads the same candidate catalog data used by ``/sku/get_candidate_SKU`` and
``/sku/get_image``, and writes the exact readable prompt assembled by the runtime
reviewer.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np


# Running the artifact generator should not dirty tracked cache files.
sys.dont_write_bytecode = True


PERCEPTION_ROOT = Path(__file__).resolve().parents[3]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))
INSPECT_ROOT = PERCEPTION_ROOT / "inspect"
if str(INSPECT_ROOT) not in sys.path:
    sys.path.insert(0, str(INSPECT_ROOT))

from comparison_based import ComparisonConfig, detect_shortage  # noqa: E402
from comparison_based.qwen_review.reviewer import (  # noqa: E402
    CandidateProduct,
    _payload_as_readable_prompt,
    _resize_image,
    build_qwen_payload,
    crop_review_region,
    normalize_reference_image,
)
from config import QWEN3_MODEL  # noqa: E402
from sku.api import DEFAULT_CATALOG_PATH, SkuCatalog  # noqa: E402


TaskType = Literal["SHORTAGE", "MISPLACED"]
PoseType = Literal["SHELF_VIEW_UPPER", "SHELF_VIEW_LOWER"]


@dataclass(frozen=True)
class SampleSpec:
    task_type: TaskType
    dataset: str
    pair_number: int
    location_id: str
    pose_type: PoseType


SAMPLES: tuple[SampleSpec, ...] = (
    SampleSpec("SHORTAGE", "inspect_shortage_paired", 1, "H2_B_L3_C01", "SHELF_VIEW_LOWER"),
    SampleSpec("SHORTAGE", "inspect_shortage_paired", 2, "H2_B_L3_C01", "SHELF_VIEW_LOWER"),
    SampleSpec("SHORTAGE", "inspect_shortage_paired", 3, "H2_B_L1_C01", "SHELF_VIEW_UPPER"),
    SampleSpec("SHORTAGE", "inspect_shortage_paired", 4, "H2_B_L3_C01", "SHELF_VIEW_LOWER"),
    SampleSpec("MISPLACED", "inspect_misplaced_paired", 1, "H1_F_L1_C01", "SHELF_VIEW_UPPER"),
    SampleSpec("MISPLACED", "inspect_misplaced_paired", 2, "H1_F_L1_C01", "SHELF_VIEW_UPPER"),
    SampleSpec("MISPLACED", "inspect_misplaced_paired", 3, "H1_F_L1_C01", "SHELF_VIEW_UPPER"),
    SampleSpec("MISPLACED", "inspect_misplaced_paired", 4, "H2_B_L1_C01", "SHELF_VIEW_UPPER"),
    SampleSpec("MISPLACED", "inspect_misplaced_paired", 5, "H2_B_L1_C01", "SHELF_VIEW_UPPER"),
    SampleSpec("MISPLACED", "inspect_misplaced_paired", 6, "H2_F_L3_C01", "SHELF_VIEW_LOWER"),
)


def load_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    success, encoded = cv2.imencode(
        suffix,
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95] if suffix in {".jpg", ".jpeg"} else [],
    )
    if not success:
        raise ValueError(f"cannot encode image: {path}")
    encoded.tofile(path)


def build_local_candidates(
    catalog: SkuCatalog,
    catalog_root: Path,
    location_id: str,
    pose_type: PoseType,
) -> tuple[list[list[dict[str, str]]], list[CandidateProduct]]:
    products_by_row = catalog.candidate_products(location_id, pose_type)
    if not products_by_row:
        raise ValueError(f"no candidates for {location_id} / {pose_type}")

    rows = [
        [
            {"sku_id": str(product["sku_id"]), "name": str(product["name"])}
            for product in row
        ]
        for row in products_by_row
    ]
    row_numbers_by_name: dict[str, list[int]] = {}
    product_by_name: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(products_by_row, start=1):
        for product in row:
            name = str(product["name"])
            row_numbers_by_name.setdefault(name, []).append(row_number)
            product_by_name.setdefault(name, product)

    candidates: list[CandidateProduct] = []
    for name, product in product_by_name.items():
        image_paths = product.get("images")
        if not isinstance(image_paths, list) or not image_paths:
            raise ValueError(f"candidate has no image: {name}")
        image_path = catalog_root / str(image_paths[0])
        image, media_type = normalize_reference_image(
            image_path.read_bytes(),
            mimetypes.guess_type(image_path.name)[0] or "image/jpeg",
        )
        candidates.append(
            CandidateProduct(
                sku_id=str(product["sku_id"]),
                name=name,
                row_numbers=tuple(row_numbers_by_name[name]),
                image=image,
                media_type=media_type,
            )
        )
    return rows, candidates


def detect_bboxes(spec: SampleSpec, baseline: np.ndarray, current: np.ndarray) -> list[list[int]]:
    misplaced = spec.task_type == "MISPLACED"
    result = detect_shortage(
        baseline,
        current,
        ComparisonConfig(
            target_size=(1280, 720),
            difference_mode="chroma" if misplaced else "hybrid",
            min_chroma_dominance_ratio=0.35 if misplaced else 0.0,
        ),
    )
    return [list(region.bbox) for region in result.shortages]


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "candidate"


def generate_sample(
    spec: SampleSpec,
    *,
    data_root: Path,
    catalog: SkuCatalog,
    catalog_root: Path,
    model: str,
) -> dict[str, Any]:
    dataset_root = data_root / spec.dataset
    baseline_path = dataset_root / f"{spec.pair_number}_1.jpg"
    current_path = dataset_root / f"{spec.pair_number}_2.jpg"
    baseline = load_image(baseline_path)
    current = load_image(current_path)
    bboxes = detect_bboxes(spec, baseline, current)

    pair_root = dataset_root / "qwen_prompt_samples" / f"pair_{spec.pair_number}"
    pair_root.mkdir(parents=True, exist_ok=True)
    rows, candidates = build_local_candidates(
        catalog,
        catalog_root,
        spec.location_id,
        spec.pose_type,
    )

    candidate_manifest: list[dict[str, Any]] = []
    candidate_root = pair_root / "candidate_images"
    for index, candidate in enumerate(candidates, start=1):
        extension = mimetypes.guess_extension(candidate.media_type) or ".jpg"
        image_name = f"{index:02d}_{candidate.sku_id}_{safe_filename(candidate.name)}{extension}"
        image_path = candidate_root / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(candidate.image)
        candidate_manifest.append(
            {
                "prompt_image_number": index + 1,
                "sku_id": candidate.sku_id,
                "name": candidate.name,
                "visible_row_numbers": list(candidate.row_numbers),
                "path": str(image_path.relative_to(pair_root)),
            }
        )

    resized_current = _resize_image(current)
    regions: list[dict[str, Any]] = []
    for region_index, bbox in enumerate(bboxes, start=1):
        region_root = pair_root / f"region_{region_index:02d}"
        region_root.mkdir(parents=True, exist_ok=True)
        crop = crop_review_region(resized_current, bbox, spec.task_type)
        payload = build_qwen_payload(
            task_type=spec.task_type,
            location_id=spec.location_id,
            pose_type=spec.pose_type,
            region_image=crop,
            candidate_rows=rows,
            candidates=candidates,
            model=model,
        )
        crop_path = region_root / "bbox_expanded.jpg"
        prompt_path = region_root / "prompt.txt"
        save_image(crop_path, crop)
        prompt_path.write_text(_payload_as_readable_prompt(payload), encoding="utf-8")
        regions.append(
            {
                "region_index": region_index,
                "bbox": bbox,
                "prompt_image_1": str(crop_path.relative_to(pair_root)),
                "prompt": str(prompt_path.relative_to(pair_root)),
            }
        )

    manifest = {
        "task_type": spec.task_type,
        "pair_number": spec.pair_number,
        "location_id": spec.location_id,
        "pose_type": spec.pose_type,
        "baseline": str(baseline_path),
        "current": str(current_path),
        "bbox_format": ["x", "y", "width", "height"],
        "regions": regions,
        "candidate_rows": rows,
        "candidate_images": candidate_manifest,
        "note": "[IMAGE 1] is the expanded bbox; [IMAGE 2+] are candidate_images in order.",
    }
    (pair_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"output": str(pair_root), "bbox_count": len(bboxes), **manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PERCEPTION_ROOT / "test_data")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--model", default=QWEN3_MODEL)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    catalog_path = args.catalog.resolve()
    catalog = SkuCatalog.load(catalog_path)
    results = [
        generate_sample(
            spec,
            data_root=args.data_root.resolve(),
            catalog=catalog,
            catalog_root=catalog_path.parent,
            model=args.model,
        )
        for spec in SAMPLES
    ]
    summary_path = args.data_root.resolve() / "qwen_prompt_samples_summary.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_path)
    for result in results:
        print(
            f"{result['task_type']} pair_{result['pair_number']}: "
            f"{result['bbox_count']} bbox -> {result['output']}"
        )


if __name__ == "__main__":
    main()
