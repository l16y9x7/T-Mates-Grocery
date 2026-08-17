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

    def test_discovers_converted_live_shortage_record_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_name = (
                "20260816T205750_798810Z_"
                "H1_F_L_INSPECT_SHORTAGE_1aa7e2d9"
            )
            record = root / "H1_F_L_INSPECT_UPPER" / record_name
            record.mkdir(parents=True)

            records = batch.discover_records(root)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["record"], record_name)
        self.assertEqual(records[0]["inspection_target_id"], "H1_F_L_INSPECT")
        self.assertEqual(records[0]["pose_type"], "SHELF_VIEW_UPPER")

    def test_collects_result_from_converted_live_shortage_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_name = (
                "20260816T205750_798810Z_"
                "H1_F_L_INSPECT_SHORTAGE_1aa7e2d9"
            )
            result_path = (
                root
                / "H1_F_L_INSPECT_UPPER"
                / record_name
                / batch.RESULT_DIRECTORY_NAME
                / "result.json"
            )
            batch.write_json_atomic(
                result_path,
                {"record": record_name, "status": "success"},
            )

            results = batch.collect_results(root)

        self.assertEqual(
            results,
            [{"record": record_name, "status": "success"}],
        )

    def test_detection_stage_rejects_candidate_outside_shelf_span(self) -> None:
        row = SimpleNamespace(
            bbox=[0, 273, 1280, 377],
            index=2,
            lower_rail_index=0,
        )
        rail = SimpleNamespace(
            line=[128, 650, 1151, 650],
            y_center=650,
        )
        inside = batch.INSPECT_API.Finding(
            bbox=[670, 350, 78, 240],
            center=[709, 470],
            sources=["comparison_based"],
            votes=1,
        )
        outside = batch.INSPECT_API.Finding(
            bbox=[1149, 571, 95, 46],
            center=[1196, 594],
            sources=["comparison_based"],
            votes=1,
        )

        kept, rejected = batch.filter_findings_to_shelf_range(
            [inside, outside],
            [row],
            [rail],
            image_width=1280,
            pose_type="SHELF_VIEW_UPPER",
        )

        self.assertEqual(
            [list(finding.bbox) for finding in kept],
            [[670, 350, 78, 240]],
        )
        self.assertEqual(len(rejected), 1)
        self.assertLess(rejected[0]["shelf_overlap_ratio"], 0.75)

    def test_regression_evaluation_matches_bbox_and_product(self) -> None:
        findings = [
            {
                "region_index": 1,
                "bbox": [670, 350, 78, 240],
                "product_name": "奥利奥冰淇淋抹茶味",
            }
        ]
        expected = {
            "findings": [
                {
                    "bbox": [672, 352, 76, 236],
                    "product_name": "奥利奥冰淇淋抹茶味",
                }
            ],
            "minimum_bbox_iou": 0.5,
        }

        result = batch.evaluate_expected_findings(
            findings,
            expected,
            detection_only=False,
        )

        self.assertTrue(result["detection_pass"])
        self.assertTrue(result["recognition_pass"])

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

    def test_depth_filter_refines_mixed_rgb_bbox_to_farther_component(self) -> None:
        image = np.full((100, 140, 3), 80, dtype=np.uint8)
        photometric_mask = np.zeros((100, 140), dtype=np.uint8)
        photometric_mask[25:55, 25:55] = 255
        photometric_mask[25:55, 65:95] = 255
        response = batch.INSPECT_API.InspectResponse(
            location_id="H1_B_L1_C01",
            pose_type="SHELF_VIEW_LOWER",
            task_type="SHORTAGE",
            has_anomaly=True,
            image_size=[140, 100],
            bbox_format=["x", "y", "width", "height"],
            findings=[
                batch.INSPECT_API.Finding(
                    bbox=[20, 20, 80, 50],
                    center=[60, 45],
                    sources=["comparison_based"],
                    votes=1,
                )
            ],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=photometric_mask,
            review_homography=np.eye(3, dtype=np.float64),
        )
        baseline_depth = np.full((100, 140), 900, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[25:55, 25:55] = 1100
        current_depth[25:55, 65:95] = 700
        rows = SimpleNamespace(
            rows=[
                SimpleNamespace(
                    bbox=(0, 0, 140, 100),
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
        self.assertEqual(filtered.response.findings[0].bbox, [25, 20, 30, 50])
        self.assertEqual(int(np.count_nonzero(filtered.review_mask)), 900)
        self.assertTrue(supports[0]["refined"])
        self.assertEqual(supports[0]["original_bbox"], [20, 20, 80, 50])
        self.assertEqual(summary["refined_findings"], 1)
        self.assertEqual(summary["promoted_findings"], 0)

    def test_dominant_depth_hole_overrides_unrelated_rgb_candidate(self) -> None:
        image = np.full((200, 300, 3), 80, dtype=np.uint8)
        photometric_mask = np.zeros((200, 300), dtype=np.uint8)
        photometric_mask[5:45, 30:70] = 255
        photometric_mask[130:170, 180:240] = 255
        response = batch.INSPECT_API.InspectResponse(
            location_id="H1_B_L1_C01",
            pose_type="SHELF_VIEW_LOWER",
            task_type="SHORTAGE",
            has_anomaly=True,
            image_size=[300, 200],
            bbox_format=["x", "y", "width", "height"],
            findings=[
                batch.INSPECT_API.Finding(
                    bbox=[180, 130, 60, 40],
                    center=[210, 150],
                    sources=["comparison_based"],
                    votes=1,
                )
            ],
            algorithms=[
                batch.INSPECT_API.AlgorithmResult(
                    name="comparison_based",
                    success=True,
                    elapsed_ms=1.0,
                    findings=[
                        batch.INSPECT_API.AlgorithmFinding(
                            bbox=[180, 130, 60, 40],
                            center=[210, 150],
                            contour_area=2000,
                            changed_pixels=2400,
                            chroma_dominance_ratio=0.05,
                        )
                    ],
                )
            ],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=photometric_mask,
            review_homography=np.eye(3, dtype=np.float64),
        )
        baseline_depth = np.full((200, 300), 900, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[5:45, 30:70] = 1100
        rows = SimpleNamespace(
            rows=[
                SimpleNamespace(
                    bbox=(0, 0, 300, 100),
                    lower_rail_index=0,
                ),
                SimpleNamespace(
                    bbox=(0, 100, 300, 100),
                    lower_rail_index=None,
                ),
            ],
        )

        with patch.object(batch.INSPECT_API, "detect_rows", return_value=rows):
            filtered, supports, summary, _ = batch.filter_execution_with_depth(
                execution,
                baseline_depth,
                current_depth,
            )

        self.assertEqual(len(filtered.response.findings), 1)
        self.assertEqual(filtered.response.findings[0].bbox, [30, 5, 40, 40])
        self.assertTrue(supports[0]["promoted"])
        self.assertEqual(summary["rgb_fallback_findings"], 0)

    def test_rgb_fallback_keeps_low_chroma_same_product_change(self) -> None:
        image = np.full((100, 140, 3), 80, dtype=np.uint8)
        photometric_mask = np.zeros((100, 140), dtype=np.uint8)
        photometric_mask[25:65, 35:85] = 255
        finding = batch.INSPECT_API.Finding(
            bbox=[35, 25, 50, 40],
            center=[60, 45],
            sources=["comparison_based"],
            votes=1,
        )
        response = batch.INSPECT_API.InspectResponse(
            location_id="H1_B_L1_C01",
            pose_type="SHELF_VIEW_UPPER",
            task_type="SHORTAGE",
            has_anomaly=True,
            image_size=[140, 100],
            bbox_format=["x", "y", "width", "height"],
            findings=[finding],
            algorithms=[
                batch.INSPECT_API.AlgorithmResult(
                    name="comparison_based",
                    success=True,
                    elapsed_ms=1.0,
                    findings=[
                        batch.INSPECT_API.AlgorithmFinding(
                            bbox=[35, 25, 50, 40],
                            center=[60, 45],
                            contour_area=1800,
                            changed_pixels=2000,
                            chroma_dominance_ratio=0.05,
                        )
                    ],
                )
            ],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=photometric_mask,
            review_homography=np.eye(3, dtype=np.float64),
        )
        depth = np.full((100, 140), 900, dtype=np.float32)
        rows = SimpleNamespace(
            rows=[
                SimpleNamespace(
                    bbox=(0, 0, 140, 100),
                    lower_rail_index=None,
                )
            ],
        )

        with patch.object(batch.INSPECT_API, "detect_rows", return_value=rows):
            filtered, supports, summary, _ = batch.filter_execution_with_depth(
                execution,
                depth,
                depth.copy(),
            )

        self.assertEqual(filtered.response.findings, [finding])
        self.assertTrue(supports[0]["rgb_fallback"])
        self.assertEqual(summary["rgb_fallback_findings"], 1)

    def test_low_contrast_depth_pair_recovers_identical_rear_item(self) -> None:
        baseline = np.full((200, 300, 3), 80, dtype=np.uint8)
        current = baseline.copy()
        current[60:120, 100:125] = 110
        response = batch.INSPECT_API.InspectResponse(
            location_id="H1_B_L1_C01",
            pose_type="SHELF_VIEW_UPPER",
            task_type="SHORTAGE",
            has_anomaly=False,
            image_size=[300, 200],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=current,
            review_mask=np.zeros((200, 300), dtype=np.uint8),
            review_homography=np.eye(3, dtype=np.float64),
        )
        baseline_depth = np.full((200, 300), 900, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[60:120, 100:125] = 1100
        current_depth[60:120, 128:153] = 700
        rows = SimpleNamespace(
            rows=[
                SimpleNamespace(
                    bbox=(0, 0, 300, 200),
                    lower_rail_index=0,
                )
            ],
        )

        with patch.object(batch.INSPECT_API, "detect_rows", return_value=rows):
            filtered, supports, summary, _ = batch.filter_execution_with_depth(
                execution,
                baseline_depth,
                current_depth,
                baseline,
            )

        self.assertEqual(len(filtered.response.findings), 1)
        self.assertTrue(supports[0]["low_contrast_promotion"])
        self.assertEqual(summary["low_contrast_promoted_findings"], 1)

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

    def test_large_object_like_depth_hole_may_touch_open_row_bottom(self) -> None:
        shape = (120, 200)
        photometric_mask = np.zeros(shape, dtype=np.uint8)
        photometric_mask[80:120, 50:110] = 255
        baseline_depth = np.full(shape, 900, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[80:120, 50:110] = 1100
        depth_mask = np.where(
            current_depth - baseline_depth > 60,
            255,
            0,
        ).astype(np.uint8)

        promoted = batch.promote_depth_components(
            photometric_mask,
            depth_mask,
            baseline_depth,
            current_depth,
            [[0, 60, 200, 60]],
            [[0, 60, 200, 60]],
            [],
        )

        self.assertEqual(len(promoted), 1)
        finding, support, _ = promoted[0]
        self.assertEqual(finding.bbox, [50, 80, 60, 40])
        self.assertTrue(support["bottom_border_exception"]["allowed"])
        self.assertEqual(
            support["bottom_border_exception"]["movement_balance_ratio"],
            0.0,
        )

    def test_thin_bottom_border_depth_strip_remains_rejected(self) -> None:
        shape = (120, 200)
        photometric_mask = np.zeros(shape, dtype=np.uint8)
        photometric_mask[114:120, 35:165] = 255
        baseline_depth = np.full(shape, 900, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[114:120, 35:165] = 1100
        depth_mask = np.where(
            current_depth - baseline_depth > 60,
            255,
            0,
        ).astype(np.uint8)

        promoted = batch.promote_depth_components(
            photometric_mask,
            depth_mask,
            baseline_depth,
            current_depth,
            [[0, 60, 200, 60]],
            [[0, 60, 200, 60]],
            [],
        )

        self.assertEqual(promoted, [])

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

    def test_shelf_filter_rejects_back_left_outside_shelf_candidate(self) -> None:
        image = np.full((100, 200, 3), 90, dtype=np.uint8)
        response = batch.INSPECT_API.InspectResponse(
            location_id="H2_B_L_INSPECT",
            pose_type="SHELF_VIEW_UPPER",
            task_type="SHORTAGE",
            has_anomaly=True,
            image_size=[200, 100],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=np.zeros((100, 200), dtype=np.uint8),
            review_homography=np.eye(3, dtype=np.float64),
        )
        finding = batch.INSPECT_API.Finding(
            bbox=[10, 20, 40, 50],
            center=[30, 45],
            sources=["depth_rgb_fusion"],
            votes=1,
        )
        row = SimpleNamespace(index=1, bbox=(0, 0, 200, 100), lower_rail_index=0)
        rail = SimpleNamespace(line=(40, 90, 190, 90), y_center=90)
        baseline_depth = np.full((100, 200), 900, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[20:70, 10:50] = 1100

        kept, rejected = batch.filter_shelf_interference_candidates(
            [(finding, {"promoted": True}, np.zeros((100, 200), dtype=np.uint8))],
            execution=execution,
            baseline_image=image,
            baseline_depth_mm=baseline_depth,
            current_depth_mm=current_depth,
            rows=[row],
            rails=[rail],
        )

        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 1)
        self.assertLess(rejected[0]["shelf_overlap_ratio"], 0.75)

    def test_shelf_filter_rejects_open_row_shift_pair(self) -> None:
        image = np.full((100, 200, 3), 90, dtype=np.uint8)
        response = batch.INSPECT_API.InspectResponse(
            location_id="H2_F_R_INSPECT",
            pose_type="SHELF_VIEW_LOWER",
            task_type="SHORTAGE",
            has_anomaly=True,
            image_size=[200, 100],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=np.zeros((100, 200), dtype=np.uint8),
            review_homography=np.eye(3, dtype=np.float64),
        )
        finding = batch.INSPECT_API.Finding(
            bbox=[50, 25, 20, 45],
            center=[60, 47],
            sources=["comparison_based"],
            votes=1,
        )
        row = SimpleNamespace(index=3, bbox=(0, 0, 200, 100), lower_rail_index=None)
        rail = SimpleNamespace(line=(0, 0, 180, 0), y_center=0)
        baseline_depth = np.full((100, 200), 900, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[25:70, 50:70] = 1100
        current_depth[25:70, 72:92] = 700

        kept, rejected = batch.filter_shelf_interference_candidates(
            [(finding, {"refined": True}, np.zeros((100, 200), dtype=np.uint8))],
            execution=execution,
            baseline_image=image,
            baseline_depth_mm=baseline_depth,
            current_depth_mm=current_depth,
            rows=[row],
            rails=[rail],
        )

        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 1)
        self.assertGreater(rejected[0]["movement_balance_ratio"], 0.55)

    def test_merge_fragmented_candidates_joins_one_product_column(self) -> None:
        row = SimpleNamespace(index=1, bbox=(0, 0, 200, 100), lower_rail_index=0)
        first = batch.INSPECT_API.Finding(
            bbox=[60, 10, 30, 25],
            center=[75, 22],
            sources=["comparison_based"],
            votes=1,
        )
        second = batch.INSPECT_API.Finding(
            bbox=[58, 40, 32, 35],
            center=[74, 57],
            sources=["depth_rgb_fusion"],
            votes=1,
        )
        first_mask = np.zeros((100, 200), dtype=np.uint8)
        second_mask = first_mask.copy()
        first_mask[10:35, 60:90] = 255
        second_mask[40:75, 58:90] = 255

        merged = batch.merge_fragmented_candidates(
            [
                (first, {"refined": True}, first_mask),
                (second, {"promoted": True}, second_mask),
            ],
            [row],
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0][0].bbox, [58, 10, 32, 65])
        self.assertIn("merged_fragments", merged[0][1])
        self.assertEqual(int(np.count_nonzero(merged[0][2])), 1870)

    def test_closed_depth_recovery_joins_fragmented_silhouette(self) -> None:
        baseline = np.full((200, 300, 3), 80, dtype=np.uint8)
        current = baseline.copy()
        current[30:100, 50:115] = 130
        response = batch.INSPECT_API.InspectResponse(
            location_id="H2_F_L_INSPECT",
            pose_type="SHELF_VIEW_UPPER",
            task_type="SHORTAGE",
            has_anomaly=False,
            image_size=[300, 200],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=current,
            review_mask=np.zeros((200, 300), dtype=np.uint8),
            review_homography=np.eye(3, dtype=np.float64),
        )
        baseline_depth = np.full((200, 300), 900, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[30:100, 50:80] = 1100
        current_depth[30:100, 85:115] = 1100
        depth_mask = np.where(
            current_depth - baseline_depth > 60,
            255,
            0,
        ).astype(np.uint8)
        row = SimpleNamespace(index=1, bbox=(0, 0, 300, 200), lower_rail_index=0)
        rail = SimpleNamespace(line=(0, 190, 280, 190), y_center=190)

        recovered = batch.recover_closed_depth_candidate(
            execution=execution,
            baseline_image=baseline,
            depth_change_mask=depth_mask,
            baseline_depth_mm=baseline_depth,
            current_depth_mm=current_depth,
            rows=[row],
            rails=[rail],
        )

        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertLessEqual(recovered[0].bbox[0], 50)
        self.assertGreaterEqual(recovered[0].bbox[2], 65)
        self.assertTrue(recovered[1]["closed_depth_recovery"])

    def test_lower_shelf_roi_crops_sides_and_follows_rail_span(self) -> None:
        rows = [
            SimpleNamespace(index=1, bbox=(0, 0, 200, 50), lower_rail_index=0),
            SimpleNamespace(index=2, bbox=(0, 50, 200, 50), lower_rail_index=1),
        ]
        rails = [
            SimpleNamespace(line=(0, 49, 180, 49), y_center=49),
            SimpleNamespace(line=(10, 99, 160, 99), y_center=99),
        ]

        mask = batch._build_shelf_roi_mask(
            (100, 200),
            rows,
            rails,
            "SHELF_VIEW_LOWER",
        )

        self.assertEqual(int(mask[25, 19]), 0)
        self.assertEqual(int(mask[25, 20]), 255)
        self.assertEqual(int(mask[75, 155]), 255)
        self.assertEqual(int(mask[75, 156]), 0)

    def test_front_right_depth_balance_is_diagnostic_not_hard_rejection(self) -> None:
        image = np.full((120, 200, 3), 90, dtype=np.uint8)
        response = batch.INSPECT_API.InspectResponse(
            location_id="H1_F_R_INSPECT",
            pose_type="SHELF_VIEW_LOWER",
            task_type="SHORTAGE",
            has_anomaly=True,
            image_size=[200, 120],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=np.zeros((120, 200), dtype=np.uint8),
            review_homography=np.eye(3, dtype=np.float64),
        )
        finding = batch.INSPECT_API.Finding(
            bbox=[70, 40, 45, 55],
            center=[92, 67],
            sources=["comparison_based"],
            votes=1,
        )
        region_mask = np.zeros((120, 200), dtype=np.uint8)
        region_mask[40:95, 70:115] = 255
        baseline_depth = np.full((120, 200), 900, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[40:95, 60:92] = 1080
        current_depth[40:95, 93:125] = 720
        row = SimpleNamespace(index=2, bbox=(0, 30, 200, 80), lower_rail_index=0)
        rail = SimpleNamespace(line=(0, 109, 199, 109), y_center=109)

        kept, rejected = batch.filter_shelf_interference_candidates(
            [(finding, {"refined": True}, region_mask)],
            execution=execution,
            baseline_image=None,
            baseline_depth_mm=baseline_depth,
            current_depth_mm=current_depth,
            rows=[row],
            rails=[rail],
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(rejected, [])

    def test_front_left_upper_rejects_narrow_rgb_fallback_without_depth_hole(
        self,
    ) -> None:
        image = np.full((120, 200, 3), 90, dtype=np.uint8)
        response = batch.INSPECT_API.InspectResponse(
            location_id="H1_F_L_INSPECT",
            pose_type="SHELF_VIEW_UPPER",
            task_type="SHORTAGE",
            has_anomaly=True,
            image_size=[200, 120],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=np.zeros((120, 200), dtype=np.uint8),
            review_homography=np.eye(3, dtype=np.float64),
        )
        finding = batch.INSPECT_API.Finding(
            bbox=[35, 35, 7, 27],
            center=[38, 48],
            sources=["comparison_based"],
            votes=1,
        )
        region_mask = np.zeros((120, 200), dtype=np.uint8)
        region_mask[35:62, 35:42] = 255
        baseline_depth = np.full((120, 200), 900, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[36:61, 35:39] = 1100
        row = SimpleNamespace(index=2, bbox=(0, 20, 200, 100), lower_rail_index=0)
        rail = SimpleNamespace(line=(0, 119, 199, 119), y_center=119)

        kept, rejected = batch.filter_shelf_interference_candidates(
            [(finding, {"rgb_fallback": True}, region_mask)],
            execution=execution,
            baseline_image=None,
            baseline_depth_mm=baseline_depth,
            current_depth_mm=current_depth,
            rows=[row],
            rails=[rail],
        )

        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 1)
        self.assertLess(
            rejected[0]["object_window_farther_ratio"],
            batch.MIN_NARROW_RGB_FALLBACK_FARTHER_WINDOW_RATIO,
        )
        self.assertIn("weak object-window", rejected[0]["reasons"][0])

    def test_promoted_background_depth_is_rejected_by_dynamic_row_limit(self) -> None:
        image = np.full((120, 200, 3), 90, dtype=np.uint8)
        response = batch.INSPECT_API.InspectResponse(
            location_id="H1_F_R_INSPECT",
            pose_type="SHELF_VIEW_UPPER",
            task_type="SHORTAGE",
            has_anomaly=True,
            image_size=[200, 120],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=np.zeros((120, 200), dtype=np.uint8),
            review_homography=np.eye(3, dtype=np.float64),
        )
        finding = batch.INSPECT_API.Finding(
            bbox=[70, 40, 45, 55],
            center=[92, 67],
            sources=["depth_rgb_fusion"],
            votes=1,
        )
        region_mask = np.zeros((120, 200), dtype=np.uint8)
        region_mask[40:95, 70:115] = 255
        baseline_depth = np.full((120, 200), 900, dtype=np.float32)
        baseline_depth[40:95, 70:115] = 3500
        current_depth = baseline_depth.copy()
        current_depth[40:95, 70:115] = 4200
        row = SimpleNamespace(index=2, bbox=(0, 30, 200, 80), lower_rail_index=0)
        rail = SimpleNamespace(line=(0, 109, 199, 109), y_center=109)

        kept, rejected = batch.filter_shelf_interference_candidates(
            [(finding, {"promoted": True}, region_mask)],
            execution=execution,
            baseline_image=None,
            baseline_depth_mm=baseline_depth,
            current_depth_mm=current_depth,
            rows=[row],
            rails=[rail],
        )

        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 1)
        depth_filter = rejected[0]["baseline_foreground_depth_filter"]
        self.assertTrue(depth_filter["applicable"])
        self.assertFalse(depth_filter["accepted"])
        self.assertEqual(depth_filter["candidate_baseline_median_mm"], 3500.0)
        self.assertLess(depth_filter["baseline_foreground_ratio"], 0.5)

    def test_promoted_product_depth_passes_dynamic_row_limit(self) -> None:
        image = np.full((120, 200, 3), 90, dtype=np.uint8)
        response = batch.INSPECT_API.InspectResponse(
            location_id="H1_F_R_INSPECT",
            pose_type="SHELF_VIEW_UPPER",
            task_type="SHORTAGE",
            has_anomaly=True,
            image_size=[200, 120],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=np.zeros((120, 200), dtype=np.uint8),
            review_homography=np.eye(3, dtype=np.float64),
        )
        finding = batch.INSPECT_API.Finding(
            bbox=[70, 40, 45, 55],
            center=[92, 67],
            sources=["depth_rgb_fusion"],
            votes=1,
        )
        region_mask = np.zeros((120, 200), dtype=np.uint8)
        region_mask[40:95, 70:115] = 255
        baseline_depth = np.full((120, 200), 900, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[40:95, 70:115] = 1100
        row = SimpleNamespace(index=2, bbox=(0, 30, 200, 80), lower_rail_index=0)
        rail = SimpleNamespace(line=(0, 109, 199, 109), y_center=109)

        kept, rejected = batch.filter_shelf_interference_candidates(
            [(finding, {"promoted": True}, region_mask)],
            execution=execution,
            baseline_image=None,
            baseline_depth_mm=baseline_depth,
            current_depth_mm=current_depth,
            rows=[row],
            rails=[rail],
        )

        self.assertEqual(len(kept), 1)
        self.assertEqual(rejected, [])
        depth_filter = kept[0][1]["shelf_interference_filter"][
            "baseline_foreground_depth_filter"
        ]
        self.assertTrue(depth_filter["accepted"])
        self.assertEqual(depth_filter["baseline_foreground_ratio"], 1.0)

    def test_candidate_mostly_on_rail_is_rejected(self) -> None:
        image = np.full((100, 200, 3), 90, dtype=np.uint8)
        response = batch.INSPECT_API.InspectResponse(
            location_id="H1_F_L_INSPECT",
            pose_type="SHELF_VIEW_UPPER",
            task_type="SHORTAGE",
            has_anomaly=True,
            image_size=[200, 100],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=np.zeros((100, 200), dtype=np.uint8),
            review_homography=np.eye(3, dtype=np.float64),
        )
        finding = batch.INSPECT_API.Finding(
            bbox=[70, 92, 45, 20],
            center=[92, 102],
            sources=["comparison_based"],
            votes=1,
        )
        region_mask = np.zeros((100, 200), dtype=np.uint8)
        region_mask[92:100, 70:115] = 255
        depth = np.full((100, 200), 900, dtype=np.float32)
        row = SimpleNamespace(index=1, bbox=(0, 0, 200, 95), lower_rail_index=0)
        rail = SimpleNamespace(line=(0, 94, 199, 94), y_center=94)

        kept, rejected = batch.filter_shelf_interference_candidates(
            [(finding, {"rgb_fallback": True}, region_mask)],
            execution=execution,
            baseline_image=None,
            baseline_depth_mm=depth,
            current_depth_mm=depth.copy(),
            rows=[row],
            rails=[rail],
        )

        self.assertEqual(kept, [])
        self.assertIn("shelf rail", rejected[0]["reasons"][0])

    def test_back_left_fixed_slot_depth_recovers_hidden_stack_shortage(self) -> None:
        image = np.full((720, 1280, 3), 90, dtype=np.uint8)
        response = batch.INSPECT_API.InspectResponse(
            location_id="H2_B_L_INSPECT",
            pose_type="SHELF_VIEW_UPPER",
            task_type="SHORTAGE",
            has_anomaly=False,
            image_size=[1280, 720],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=np.zeros((720, 1280), dtype=np.uint8),
            review_homography=np.eye(3, dtype=np.float64),
        )
        baseline_depth = np.full((720, 1280), 1000, dtype=np.float32)
        current_depth = baseline_depth.copy()
        current_depth[145:250, 350:525] = 1040

        promoted = batch.promote_fixed_layout_depth_slot(
            execution=execution,
            photometric_mask=execution.review_mask,
            baseline_depth_mm=baseline_depth,
            current_depth_mm=current_depth,
        )

        self.assertIsNotNone(promoted)
        assert promoted is not None
        self.assertEqual(promoted[0].bbox, [335, 125, 210, 145])
        self.assertEqual(promoted[1]["slot_index"], 2)
        self.assertTrue(promoted[1]["fixed_layout_depth_promotion"])

    def test_front_left_fixed_slot_uses_local_depth_offset_and_rgb_evidence(self) -> None:
        image = np.full((720, 1280, 3), 90, dtype=np.uint8)
        review_mask = np.zeros((720, 1280), dtype=np.uint8)
        review_mask[20:220, 450:600] = 255
        response = batch.INSPECT_API.InspectResponse(
            location_id="H2_F_L_INSPECT",
            pose_type="SHELF_VIEW_UPPER",
            task_type="SHORTAGE",
            has_anomaly=False,
            image_size=[1280, 720],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=review_mask,
            review_homography=np.eye(3, dtype=np.float64),
        )
        baseline_depth = np.full((720, 1280), 1000, dtype=np.float32)
        current_depth = np.full((720, 1280), 960, dtype=np.float32)
        current_depth[0:240, 430:625] = 1040

        promoted = batch.promote_fixed_layout_depth_slot(
            execution=execution,
            photometric_mask=review_mask,
            baseline_depth_mm=baseline_depth,
            current_depth_mm=current_depth,
        )

        self.assertIsNotNone(promoted)
        assert promoted is not None
        self.assertEqual(promoted[0].bbox, [430, 0, 195, 240])
        self.assertEqual(promoted[1]["slot_index"], 3)
        self.assertGreater(promoted[1]["slot_score_mm"], 10)

    def test_front_left_fixed_slot_requires_rgb_change_evidence(self) -> None:
        image = np.full((720, 1280, 3), 90, dtype=np.uint8)
        response = batch.INSPECT_API.InspectResponse(
            location_id="H2_F_L_INSPECT",
            pose_type="SHELF_VIEW_UPPER",
            task_type="SHORTAGE",
            has_anomaly=False,
            image_size=[1280, 720],
            bbox_format=["x", "y", "width", "height"],
            findings=[],
            algorithms=[],
        )
        execution = batch.INSPECT_API.InspectionExecution(
            response=response,
            review_image=image,
            review_mask=np.zeros((720, 1280), dtype=np.uint8),
            review_homography=np.eye(3, dtype=np.float64),
        )
        depth = np.full((720, 1280), 1000, dtype=np.float32)

        promoted = batch.promote_fixed_layout_depth_slot(
            execution=execution,
            photometric_mask=execution.review_mask,
            baseline_depth_mm=depth,
            current_depth_mm=depth.copy(),
        )

        self.assertIsNone(promoted)


if __name__ == "__main__":
    unittest.main()
