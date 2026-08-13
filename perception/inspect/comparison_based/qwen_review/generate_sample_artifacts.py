"""Generate review prompts and expanded bbox crops for the bundled pair samples.

This utility deliberately does not call Qwen.  It runs the comparison detector,
loads candidates and reference images from the configured SKU service, and
writes the exact readable prompt assembled by the runtime reviewer.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np
import requests


# Running the artifact generator should not dirty tracked cache files.
sys.dont_write_bytecode = True


PERCEPTION_ROOT = Path(__file__).resolve().parents[3]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))
INSPECT_ROOT = PERCEPTION_ROOT / "inspect"
if str(INSPECT_ROOT) not in sys.path:
    sys.path.insert(0, str(INSPECT_ROOT))

from comparison_based import (  # noqa: E402
    ComparisonConfig,
    ComparisonResult,
    detect_shortage,
)
from comparison_based.qwen_review.reviewer import (  # noqa: E402
    CandidateProduct,
    _payload_as_readable_prompt,
    _resize_image,
    build_candidate_contact_sheets,
    build_expected_product_row_image,
    build_qwen_payload,
    crop_review_region,
    normalize_reference_image,
)
from config import QWEN3_MODEL, SKU_API_URL  # noqa: E402
from row_detection import RowDetectionResult, ShelfRowMatch, detect_rows  # noqa: E402


TaskType = Literal["SHORTAGE", "MISPLACED"]
PoseType = Literal["SHELF_VIEW_UPPER", "SHELF_VIEW_LOWER"]
EXPECTED_ROW_COUNTS: dict[PoseType, int] = {
    "SHELF_VIEW_UPPER": 2,
    "SHELF_VIEW_LOWER": 3,
}
MIN_ROW_OVERLAP_RATIO = 0.6
SKU_TIMEOUT_SECONDS = 10.0


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


def build_api_candidates(
    sku_base_url: str,
    location_id: str,
    pose_type: PoseType,
    *,
    session: Any = requests,
    timeout: float = SKU_TIMEOUT_SECONDS,
) -> tuple[list[list[dict[str, str]]], list[CandidateProduct]]:
    base_url = sku_base_url.rstrip("/")
    response = session.get(
        f"{base_url}/sku/get_candidate_SKU",
        json={"location_id": location_id, "pose_type": pose_type},
        timeout=timeout,
    )
    response.raise_for_status()
    products_by_row = response.json()
    if not isinstance(products_by_row, list) or not products_by_row or not all(
        isinstance(row, list) for row in products_by_row
    ):
        raise ValueError(
            f"SKU API returned invalid candidates for {location_id} / {pose_type}"
        )

    rows: list[list[dict[str, str]]] = []
    row_numbers_by_name: dict[str, list[int]] = {}
    product_by_name: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(products_by_row, start=1):
        normalized_row: list[dict[str, str]] = []
        for product in row:
            if not isinstance(product, dict):
                raise ValueError(f"SKU API row {row_number} contains invalid item")
            sku_id = product.get("sku_id")
            name = product.get("name")
            if not isinstance(sku_id, str) or not isinstance(name, str):
                raise ValueError(
                    f"SKU API row {row_number} item is missing sku_id/name"
                )
            normalized_row.append({"sku_id": sku_id, "name": name})
            row_numbers_by_name.setdefault(name, []).append(row_number)
            product_by_name.setdefault(name, product)
        rows.append(normalized_row)

    candidates: list[CandidateProduct] = []
    for name, product in product_by_name.items():
        paths_response = session.get(
            f"{base_url}/sku/get_image",
            params={"name": name},
            timeout=timeout,
        )
        paths_response.raise_for_status()
        image_paths = paths_response.json()
        if not isinstance(image_paths, list) or not image_paths or not isinstance(
            image_paths[0], str
        ):
            raise ValueError(f"SKU API returned no image for candidate: {name}")
        image_response = session.get(
            f"{base_url}/{image_paths[0].lstrip('/')}",
            timeout=timeout,
        )
        image_response.raise_for_status()
        media_type = image_response.headers.get("Content-Type", "image/jpeg")
        media_type = media_type.split(";", 1)[0]
        image, media_type = normalize_reference_image(
            image_response.content,
            media_type,
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


def compare_images(
    spec: SampleSpec,
    baseline: np.ndarray,
    current: np.ndarray,
) -> ComparisonResult:
    misplaced = spec.task_type == "MISPLACED"
    return detect_shortage(
        baseline,
        current,
        ComparisonConfig(
            target_size=(1280, 720),
            difference_mode="chroma" if misplaced else "hybrid",
            min_chroma_dominance_ratio=0.35 if misplaced else 0.0,
        ),
    )


def row_match_as_dict(match: ShelfRowMatch | None) -> dict[str, Any] | None:
    if match is None:
        return None
    value = {
        "row_index": match.row_index,
        "row_bbox": list(match.row_bbox),
        "overlap_ratio": match.overlap_ratio,
    }
    if match.detected_row_index is not None:
        value["detected_row_index"] = match.detected_row_index
    return value


def draw_row_review_overlay(
    row_detection: RowDetectionResult,
    bboxes: Sequence[Sequence[int]],
    row_matches: Sequence[ShelfRowMatch | None],
) -> np.ndarray:
    canvas = row_detection.draw()
    for region_index, (bbox, match) in enumerate(zip(bboxes, row_matches), start=1):
        x, y, width, height = (int(value) for value in bbox)
        color = (255, 255, 0) if match is not None else (0, 165, 255)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 3)
        assignment = (
            (
                f"DETECTED {match.detected_row_index} -> SKU {match.row_index}"
                if match.detected_row_index is not None
                and match.detected_row_index != match.row_index
                else f"SKU ROW {match.row_index}"
            )
            if match is not None
            else "NO ROW"
        )
        cv2.putText(
            canvas,
            f"REGION {region_index} -> {assignment}",
            (max(4, x), max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "candidate"


def generate_sample(
    spec: SampleSpec,
    *,
    data_root: Path,
    sku_base_url: str,
    model: str,
    session: Any = requests,
    sku_timeout: float = SKU_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    dataset_root = data_root / spec.dataset
    baseline_path = dataset_root / f"{spec.pair_number}_1.jpg"
    current_path = dataset_root / f"{spec.pair_number}_2.jpg"
    baseline = load_image(baseline_path)
    current = load_image(current_path)
    comparison = compare_images(spec, baseline, current)
    bboxes = [list(region.bbox) for region in comparison.shortages]
    row_detection = detect_rows(baseline)
    expected_row_count = EXPECTED_ROW_COUNTS[spec.pose_type]
    row_matches = row_detection.match_bboxes_to_row_window(
        bboxes,
        row_count=expected_row_count,
        anchor="top" if spec.pose_type == "SHELF_VIEW_UPPER" else "bottom",
        min_overlap_ratio=MIN_ROW_OVERLAP_RATIO,
    )

    pair_root = dataset_root / "qwen_prompt_samples" / f"pair_{spec.pair_number}"
    pair_root.mkdir(parents=True, exist_ok=True)
    rows, candidates = build_api_candidates(
        sku_base_url,
        spec.location_id,
        spec.pose_type,
        session=session,
        timeout=sku_timeout,
    )

    candidate_manifest: list[dict[str, Any]] = []
    candidate_manifest_by_name: dict[str, dict[str, Any]] = {}
    candidate_root = pair_root / "candidate_images"
    for index, candidate in enumerate(candidates, start=1):
        extension = mimetypes.guess_extension(candidate.media_type) or ".jpg"
        image_name = f"{index:02d}_{candidate.sku_id}_{safe_filename(candidate.name)}{extension}"
        image_path = candidate_root / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(candidate.image)
        record = {
            "prompt_image_number": index + 1,
            "sku_id": candidate.sku_id,
            "name": candidate.name,
            "visible_row_numbers": list(candidate.row_numbers),
            "path": str(image_path.relative_to(pair_root)),
        }
        candidate_manifest.append(record)
        candidate_manifest_by_name[candidate.name] = record

    aligned_current = _resize_image(comparison.aligned_current)
    resized_baseline = _resize_image(baseline)
    aligned_current_path = pair_root / "aligned_current.jpg"
    row_overlay_path = pair_root / "rows_and_regions.jpg"
    save_image(aligned_current_path, aligned_current)
    save_image(
        row_overlay_path,
        draw_row_review_overlay(row_detection, bboxes, row_matches),
    )
    regions: list[dict[str, Any]] = []
    for region_index, (bbox, row_match) in enumerate(
        zip(bboxes, row_matches),
        start=1,
    ):
        expected_candidates = (
            [
                candidate
                for candidate in candidates
                if row_match.row_index in candidate.row_numbers
            ]
            if row_match is not None
            else list(candidates)
        )
        if not expected_candidates:
            row_match = None
            expected_candidates = list(candidates)
        region_root = pair_root / f"region_{region_index:02d}"
        region_root.mkdir(parents=True, exist_ok=True)
        stage_specs: list[dict[str, Any]]
        if spec.task_type == "SHORTAGE":
            stage_specs = [
                {
                    "stage": "shortage_product",
                    "label": "缺货商品（目标行候选）",
                    "image": crop_review_region(
                        resized_baseline,
                        bbox,
                        "SHORTAGE",
                        row_bbox=(
                            row_match.row_bbox if row_match is not None else None
                        ),
                    ),
                    "candidates": expected_candidates,
                    "misplaced_stage": None,
                    "scope": "expected_row",
                }
            ]
        else:
            stage_specs = [
                {
                    "stage": "misplaced_product",
                    "label": "1. 放错商品（全部可见候选）",
                    "image": crop_review_region(
                        aligned_current,
                        bbox,
                        "MISPLACED",
                    ),
                    "candidates": list(candidates),
                    "misplaced_stage": "misplaced_product",
                    "scope": "all_visible_rows",
                },
                {
                    "stage": "expected_product",
                    "label": "2. 应放商品（当前/基准整行 + 目标行候选）",
                    "image": build_expected_product_row_image(
                        aligned_current,
                        resized_baseline,
                        bbox,
                        row_bbox=(
                            row_match.row_bbox if row_match is not None else None
                        ),
                    ),
                    "candidates": expected_candidates,
                    "misplaced_stage": "expected_product",
                    "scope": "expected_row",
                },
            ]

        prompt_stages: list[dict[str, Any]] = []
        for stage_spec in stage_specs:
            stage_key = str(stage_spec["stage"])
            stage_candidates = list(stage_spec["candidates"])
            stage_root = (
                region_root
                if stage_key == "shortage_product"
                else region_root / stage_key
            )
            stage_root.mkdir(parents=True, exist_ok=True)
            stage_image = stage_spec["image"]
            candidate_sheets = (
                build_candidate_contact_sheets(stage_candidates)
                if spec.task_type == "MISPLACED"
                else []
            )
            payload = build_qwen_payload(
                task_type=spec.task_type,
                location_id=spec.location_id,
                pose_type=spec.pose_type,
                region_image=stage_image,
                candidate_rows=rows,
                candidates=stage_candidates,
                model=model,
                expected_row_index=(
                    row_match.row_index if row_match is not None else None
                ),
                detected_row_index=(
                    row_match.detected_row_index if row_match is not None else None
                ),
                misplaced_stage=stage_spec["misplaced_stage"],
                candidate_sheets=(
                    candidate_sheets if candidate_sheets else None
                ),
            )
            image_path = stage_root / (
                "bbox_expanded.jpg"
                if stage_key == "shortage_product"
                else "input.jpg"
            )
            prompt_path = stage_root / "prompt.txt"
            save_image(image_path, stage_image)
            prompt_path.write_text(
                _payload_as_readable_prompt(payload),
                encoding="utf-8",
            )
            candidate_sheet_manifest = []
            for sheet_index, sheet in enumerate(candidate_sheets, start=1):
                sheet_path = stage_root / f"candidate_sheet_{sheet_index:02d}.jpg"
                sheet_path.write_bytes(sheet.image)
                candidate_sheet_manifest.append(
                    {
                        "sheet_index": sheet_index,
                        "prompt_image_number": sheet_index + 1,
                        "first_candidate_number": sheet.first_candidate_number,
                        "last_candidate_number": sheet.last_candidate_number,
                        "candidate_count": (
                            sheet.last_candidate_number
                            - sheet.first_candidate_number
                            + 1
                        ),
                        "path": str(sheet_path.relative_to(pair_root)),
                    }
                )
            stage_candidate_manifest = []
            for candidate_number, candidate in enumerate(stage_candidates, start=1):
                record = dict(candidate_manifest_by_name[candidate.name])
                if candidate_sheets:
                    record.pop("prompt_image_number", None)
                    record["candidate_number"] = candidate_number
                else:
                    record["prompt_image_number"] = candidate_number + 1
                stage_candidate_manifest.append(record)
            prompt_stages.append(
                {
                    "stage": stage_key,
                    "label": stage_spec["label"],
                    "candidate_scope": stage_spec["scope"],
                    "prompt_image_1": str(image_path.relative_to(pair_root)),
                    "prompt": str(prompt_path.relative_to(pair_root)),
                    "candidate_count_before": len(candidates),
                    "candidate_count_after": len(stage_candidates),
                    "candidate_images": stage_candidate_manifest,
                    "candidate_sheets": candidate_sheet_manifest,
                }
            )

        primary_stage = prompt_stages[0]
        regions.append(
            {
                "region_index": region_index,
                "bbox": bbox,
                "prompt_image_1": primary_stage["prompt_image_1"],
                "prompt": primary_stage["prompt"],
                "row_constraint": row_match_as_dict(row_match),
                "candidate_count_before": len(candidates),
                "candidate_count_after": primary_stage["candidate_count_after"],
                "expected_candidate_names": [
                    candidate.name for candidate in expected_candidates
                ],
                "candidate_images": primary_stage["candidate_images"],
                "candidate_sheets": primary_stage["candidate_sheets"],
                "prompt_stages": prompt_stages,
            }
        )

    manifest = {
        "task_type": spec.task_type,
        "pair_number": spec.pair_number,
        "location_id": spec.location_id,
        "pose_type": spec.pose_type,
        "baseline": str(baseline_path.relative_to(data_root)),
        "current": str(current_path.relative_to(data_root)),
        "aligned_current": str(aligned_current_path.relative_to(pair_root)),
        "row_overlay": str(row_overlay_path.relative_to(pair_root)),
        "bbox_format": ["x", "y", "width", "height"],
        "comparison": {
            "difference_mode": comparison.difference_mode,
            "threshold": comparison.threshold,
            "alignment": asdict(comparison.alignment),
        },
        "row_detection": {
            **row_detection.as_dict(),
            "expected_row_count": EXPECTED_ROW_COUNTS[spec.pose_type],
            "row_window_anchor": (
                "top" if spec.pose_type == "SHELF_VIEW_UPPER" else "bottom"
            ),
            "constraints_enabled": (
                EXPECTED_ROW_COUNTS[spec.pose_type]
                <= len(row_detection.rows)
                <= EXPECTED_ROW_COUNTS[spec.pose_type] + 1
            ),
        },
        "regions": regions,
        "candidate_rows": rows,
        "candidate_images": candidate_manifest,
        "candidate_source": {
            "type": "sku_api",
            "base_url": sku_base_url.rstrip("/"),
        },
        "note": (
            "Each prompt stage owns IMAGE 1 and its ordered candidate inputs. "
            "MISPLACED stage 1 uses the current local crop with all visible "
            "candidates; stage 2 stacks the current and baseline target rows "
            "and uses only the "
            "mapped SKU-row candidates. MISPLACED reference images are packed "
            "into numbered contact sheets; SHORTAGE keeps one image per SKU."
        ),
    }
    (pair_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "output": str(pair_root.relative_to(data_root)),
        "bbox_count": len(bboxes),
        **manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PERCEPTION_ROOT / "test_data")
    parser.add_argument("--sku-base-url", default=SKU_API_URL)
    parser.add_argument("--sku-timeout", type=float, default=SKU_TIMEOUT_SECONDS)
    parser.add_argument("--model", default=QWEN3_MODEL)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    session = requests.Session()
    results = [
        generate_sample(
            spec,
            data_root=args.data_root.resolve(),
            sku_base_url=args.sku_base_url,
            model=args.model,
            session=session,
            sku_timeout=args.sku_timeout,
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
