from __future__ import annotations

import importlib.util
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


BATCH_PATH = Path(__file__).resolve().parents[1] / "batch_shortage.py"
SPEC = importlib.util.spec_from_file_location("shortage_batch_test_api", BATCH_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {BATCH_PATH}")
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class ShortageBatchTest(unittest.TestCase):
    def test_parse_args_defaults_to_four_workers(self) -> None:
        with patch.object(sys, "argv", ["batch_shortage.py"]):
            args = batch.parse_args()

        self.assertEqual(args.workers, 4)

    def test_concurrent_runner_uses_independent_reviewers_and_keeps_order(self) -> None:
        records = [
            {"group": "group", "record": f"record_{index}"}
            for index in range(4)
        ]
        barrier = threading.Barrier(4)
        reviewer_ids: list[int] = []

        def run_record(entry: dict, **kwargs: object) -> dict:
            reviewer_ids.append(id(kwargs["reviewer"]))
            barrier.wait(timeout=3)
            return {"record": entry["record"]}

        with (
            patch.object(batch, "run_record", side_effect=run_record),
            patch.object(
                batch.INSPECT_API,
                "QwenReviewer",
                side_effect=lambda **_: object(),
            ) as reviewer_factory,
        ):
            results = batch.run_records_concurrently(
                records,
                data_root=Path("data"),
                scans={"group": SimpleNamespace()},
                reviewer_kwargs={},
                detection_only=False,
                overwrite=True,
                workers=4,
            )

        self.assertEqual(
            [result["record"] for result in results],
            [f"record_{index}" for index in range(4)],
        )
        self.assertEqual(len(set(reviewer_ids)), 4)
        self.assertEqual(reviewer_factory.call_count, 4)

    def test_discovers_grouped_records_and_uses_inspection_target_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = (
                root
                / "H1_B_R_INSPECT_UPPER"
                / "record_20260816_010203_123456"
            )
            record.mkdir(parents=True)

            records = batch.discover_records(root)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["inspection_target_id"], "H1_B_R_INSPECT")
        self.assertEqual(records[0]["pose_type"], "SHELF_VIEW_UPPER")
        self.assertEqual(records[0]["location_id"], "H1_B_R_INSPECT")

    def test_clipped_region_mask_keeps_only_bbox_pixels(self) -> None:
        mask = np.full((20, 30), 255, dtype=np.uint8)

        clipped = batch.clipped_region_mask(mask, [5, 6, 7, 8])

        self.assertEqual(int(np.count_nonzero(clipped)), 56)
        self.assertTrue(np.all(clipped[6:14, 5:12] == 255))

    def test_detection_only_record_saves_bbox_and_region_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_directory = (
                root
                / "H1_B_L_INSPECT_UPPER"
                / "record_20260816_010203_123456"
            )
            record_directory.mkdir(parents=True)
            image = np.full((72, 128, 3), 80, dtype=np.uint8)
            batch.write_image(record_directory / "rgb.jpg", image)
            np.save(
                record_directory / "depth_mm.npy",
                np.full((72, 128), 1100, dtype=np.uint16),
            )
            review_mask = np.zeros((72, 128), dtype=np.uint8)
            review_mask[20:40, 30:60] = 255
            finding = batch.INSPECT_API.Finding(
                bbox=[30, 20, 30, 20],
                center=[45, 30],
                sources=["comparison_based"],
                votes=1,
            )
            response = batch.INSPECT_API.InspectResponse(
                location_id="H1_B_L1_C01",
                pose_type="SHELF_VIEW_UPPER",
                task_type="SHORTAGE",
                has_anomaly=True,
                findings=[finding],
                image_size=[128, 72],
                bbox_format=["x", "y", "width", "height"],
                algorithms=[
                    batch.INSPECT_API.AlgorithmResult(
                        name="comparison_based",
                        success=True,
                        elapsed_ms=1.0,
                        alignment_success=True,
                    )
                ],
            )
            execution = batch.INSPECT_API.InspectionExecution(
                response=response,
                review_image=image,
                review_mask=review_mask,
                review_homography=np.eye(3, dtype=np.float64),
            )
            initial_scan = SimpleNamespace(
                rgb=image,
                rgb_path=Path("task0/H1_B_L_INSPECT_UPPER/rgb.jpg"),
                depth_mm=np.full((72, 128), 900, dtype=np.float32),
            )
            entry = {
                "group": "H1_B_L_INSPECT_UPPER",
                "record": record_directory.name,
                "record_directory": record_directory,
                "inspection_target_id": "H1_B_L_INSPECT",
                "location_id": "H1_B_L1_C01",
                "pose_type": "SHELF_VIEW_UPPER",
            }

            with patch.object(
                batch.INSPECT_API,
                "inspect_images_with_artifacts",
                return_value=execution,
            ):
                result = batch.run_record(
                    entry,
                    data_root=root,
                    initial_scan=initial_scan,
                    reviewer=None,
                    detection_only=True,
                    overwrite=True,
                )

            mask_path = root / result["findings"][0]["mask"]
            saved_mask = batch.read_image(mask_path)

        self.assertEqual(result["status"], "detection_only")
        self.assertEqual(result["findings"][0]["bbox"], [30, 20, 30, 20])
        self.assertEqual(result["findings"][0]["mask_pixels"], 600)
        self.assertEqual(
            result["findings"][0]["depth_support"]["farther_ratio"],
            1.0,
        )
        self.assertTrue(mask_path.name.endswith("_mask.png"))
        self.assertEqual(saved_mask.shape[:2], (72, 128))

    def test_depth_filter_rejects_illumination_only_region(self) -> None:
        image = np.full((72, 128, 3), 80, dtype=np.uint8)
        mask = np.zeros((72, 128), dtype=np.uint8)
        mask[20:40, 30:60] = 255
        response = batch.INSPECT_API.InspectResponse(
            location_id="H1_B_L1_C01",
            pose_type="SHELF_VIEW_UPPER",
            task_type="SHORTAGE",
            has_anomaly=True,
            image_size=[128, 72],
            bbox_format=["x", "y", "width", "height"],
            findings=[
                batch.INSPECT_API.Finding(
                    bbox=[30, 20, 30, 20],
                    center=[45, 30],
                    sources=["comparison_based"],
                    votes=1,
                )
            ],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=mask,
            review_homography=np.eye(3, dtype=np.float64),
        )
        depth = np.full((72, 128), 900, dtype=np.float32)

        filtered, supports, summary, _ = batch.filter_execution_with_depth(
            execution,
            depth,
            depth.copy(),
        )

        self.assertEqual(filtered.response.findings, [])
        self.assertEqual(supports, [])
        self.assertEqual(summary["rejected_findings"], 1)

    def test_depth_filter_promotes_large_hole_with_rgb_fragments(self) -> None:
        image = np.full((72, 128, 3), 80, dtype=np.uint8)
        photometric_mask = np.zeros((72, 128), dtype=np.uint8)
        photometric_mask[42:50, 36:54] = 255
        response = batch.INSPECT_API.InspectResponse(
            location_id="H1_B_L1_C01",
            pose_type="SHELF_VIEW_UPPER",
            task_type="SHORTAGE",
            has_anomaly=False,
            image_size=[128, 72],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=photometric_mask,
            review_homography=np.eye(3, dtype=np.float64),
        )
        baseline_depth = np.full((72, 128), 900, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[35:55, 30:60] = 1100
        rows = SimpleNamespace(
            rows=[
                SimpleNamespace(
                    bbox=(0, 0, 128, 72),
                    lower_rail_index=None,
                )
            ],
        )

        with patch.object(batch.INSPECT_API, "detect_rows", return_value=rows):
            filtered, supports, summary, _ = batch.filter_execution_with_depth(
                execution,
                baseline_depth,
                current_depth,
            )

        self.assertEqual(len(filtered.response.findings), 1)
        self.assertEqual(filtered.response.findings[0].bbox, [30, 35, 30, 20])
        self.assertEqual(
            filtered.response.findings[0].sources,
            ["depth_rgb_fusion"],
        )
        self.assertEqual(int(np.count_nonzero(filtered.review_mask)), 600)
        self.assertTrue(supports[0]["promoted"])
        self.assertTrue(supports[0]["open_ended_row"])
        self.assertEqual(supports[0]["interior_fill_ratio"], 1.0)
        self.assertEqual(summary["promoted_findings"], 1)

    def test_depth_filter_rejects_open_row_edge_outline(self) -> None:
        image = np.full((72, 128, 3), 80, dtype=np.uint8)
        outline = np.zeros((72, 128), dtype=np.uint8)
        outline[40:44, 30:70] = 255
        outline[40:65, 30:34] = 255
        response = batch.INSPECT_API.InspectResponse(
            location_id="H1_B_L1_C01",
            pose_type="SHELF_VIEW_LOWER",
            task_type="SHORTAGE",
            has_anomaly=False,
            image_size=[128, 72],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=outline,
            review_homography=np.eye(3, dtype=np.float64),
        )
        baseline_depth = np.full((72, 128), 900, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[outline > 0] = 1100
        rows = SimpleNamespace(
            rows=[
                SimpleNamespace(
                    bbox=(0, 35, 128, 37),
                    lower_rail_index=None,
                )
            ],
        )

        with patch.object(batch.INSPECT_API, "detect_rows", return_value=rows):
            filtered, supports, summary, _ = batch.filter_execution_with_depth(
                execution,
                baseline_depth,
                current_depth,
            )

        self.assertEqual(filtered.response.findings, [])
        self.assertEqual(supports, [])
        self.assertEqual(summary["promoted_findings"], 0)
        self.assertEqual(
            summary["minimum_open_row_interior_fill_ratio"],
            0.2,
        )

    def test_depth_filter_does_not_promote_depth_without_rgb_evidence(self) -> None:
        image = np.full((72, 128, 3), 80, dtype=np.uint8)
        response = batch.INSPECT_API.InspectResponse(
            location_id="H1_B_L1_C01",
            pose_type="SHELF_VIEW_UPPER",
            task_type="SHORTAGE",
            has_anomaly=False,
            image_size=[128, 72],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=np.zeros((72, 128), dtype=np.uint8),
            review_homography=np.eye(3, dtype=np.float64),
        )
        baseline_depth = np.full((72, 128), 900, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[35:55, 30:60] = 1100
        rows = SimpleNamespace(
            rows=[SimpleNamespace(bbox=(0, 0, 128, 72))],
        )

        with patch.object(batch.INSPECT_API, "detect_rows", return_value=rows):
            filtered, supports, summary, _ = batch.filter_execution_with_depth(
                execution,
                baseline_depth,
                current_depth,
            )

        self.assertEqual(filtered.response.findings, [])
        self.assertEqual(supports, [])
        self.assertEqual(summary["promoted_findings"], 0)


if __name__ == "__main__":
    unittest.main()
