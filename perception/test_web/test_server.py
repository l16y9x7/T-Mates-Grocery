from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException

import server


class PromptMappingTest(unittest.TestCase):
    def test_sam_row_debug_page_exposes_row_prompt_controls(self) -> None:
        html = (server.STATIC_DIR / "sam_row_debug.html").read_text(encoding="utf-8")
        js = (server.STATIC_DIR / "sam_row_debug.js").read_text(encoding="utf-8")
        review_html = (server.STATIC_DIR / "qwen_review.html").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="rowSelect"', html)
        self.assertIn('id="promptInput"', html)
        self.assertIn('id="runAll"', html)
        self.assertIn('id="baselineShelfFiltered"', html)
        self.assertIn('id="currentShelfFiltered"', html)
        self.assertIn("/api/sam-row-debug/records", js)
        self.assertIn("/api/sam-row-debug/run", js)
        self.assertIn('href="/sam-row-debug"', review_html)

    def test_sam_row_run_returns_mask_depth_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row_directory = root / "row_01_L1"
            row_directory.mkdir(parents=True)
            image = server.np.full((20, 30, 3), 80, dtype=server.np.uint8)
            depth = server.np.full((20, 30), 900, dtype=server.np.uint16)
            depth[2:10, 3:12] = 1200
            success, encoded_rgb = server.cv2.imencode(".jpg", image)
            self.assertTrue(success)
            encoded_rgb.tofile(row_directory / "rgb.jpg")
            server.np.save(row_directory / "depth_mm.npy", depth, allow_pickle=False)
            mask = server.np.zeros((20, 30), dtype=server.np.uint8)
            mask[2:10, 3:12] = 255
            success, encoded_mask = server.cv2.imencode(".png", mask)
            self.assertTrue(success)
            mask_base64 = base64.b64encode(encoded_mask.tobytes()).decode("ascii")
            metadata = {
                "rows": [
                    {
                        "row_index": 1,
                        "level": "L1",
                        "crop_bbox_xywh": [0, 40, 30, 20],
                        "rgb": "row_01_L1/rgb.jpg",
                        "depth_mm": "row_01_L1/depth_mm.npy",
                    }
                ]
            }

            with (
                patch.object(
                    server,
                    "ensure_sam_row_export",
                    return_value=(root, metadata),
                ),
                patch.object(
                    server,
                    "call_sam3_image_path",
                    return_value={
                        "instances": [
                            {
                                "instance_id": 7,
                                "score": 0.94,
                                "bbox_xyxy": [3, 2, 12, 10],
                                "mask_png_base64": mask_base64,
                            }
                        ]
                    },
                ),
                patch.object(
                    server,
                    "resolve_shelf_row_result",
                    return_value=(
                        root,
                        {"artifacts": {"shelf_filtered": "row_01_L1/rgb.jpg"}},
                    ),
                ),
            ):
                result = server.run_sam_row_debug(
                    server.SamRowRunRequest(
                        group="H1_F_L_INSPECT_UPPER",
                        record="record",
                        row_index=1,
                        sku_name="景田饮用纯净水",
                        prompt="frontmost bottle",
                    )
                )

        self.assertEqual(len(result["instances"]), 1)
        instance = result["instances"][0]
        self.assertEqual(instance["front_depth_mm"], 1200.0)
        self.assertEqual(instance["depth_median_mm"], 1200.0)
        self.assertEqual(instance["valid_depth_ratio"], 1.0)
        self.assertEqual(instance["bbox_original_xyxy"], [3.0, 42.0, 12.0, 50.0])
        self.assertTrue(instance["mask_data_url"].startswith("data:image/png;base64,"))
        self.assertTrue(result["overlay_data_url"].startswith("data:image/png;base64,"))

    def test_sam_row_records_include_row_candidate_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "rows"
            group = "H1_F_L_INSPECT_UPPER"
            record = "20260816T205750_798810Z_H1_F_L_INSPECT_SHORTAGE_1aa7e2d9"
            source = root / group / record
            output = output_root / group / record
            source.mkdir(parents=True)
            output.mkdir(parents=True)
            for name in ("rgb.jpg", "depth_mm.npy", "baseline_rgb.jpg"):
                (source / name).write_bytes(b"data")
            for name in (
                "rows.json",
                "row_detection.jpg",
                "row_01_L1/rgb.jpg",
                "row_01_L1/depth_preview.png",
                "row_01_L1/valid_depth_mask.png",
            ):
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"data")
            debug = root / "qwen_debug" / record
            debug.mkdir(parents=True)
            (debug / "candidates.json").write_text(
                json.dumps({"rows": [[{"name": "NFC桔汁"}]]}, ensure_ascii=False),
                encoding="utf-8",
            )
            metadata = {
                "pose_type": "SHELF_VIEW_UPPER",
                "rows": [
                    {
                        "row_index": 1,
                        "level": "L1",
                        "rgb": "row_01_L1/rgb.jpg",
                        "depth_mm": "row_01_L1/depth_mm.npy",
                        "depth_preview": "row_01_L1/depth_preview.png",
                        "valid_depth_mask": "row_01_L1/valid_depth_mask.png",
                    }
                ],
            }
            with (
                patch.object(server, "REAL_SHORTAGE_BATCH_ROOT", root),
                patch.object(server, "REAL_SHORTAGE_SAM_ROWS_ROOT", output_root),
                patch.object(
                    server,
                    "ensure_sam_row_export",
                    return_value=(output, metadata),
                ),
                patch.object(
                    server,
                    "load_prompt_pair_mapping",
                    return_value={
                        "NFC桔汁": {
                            "qwen3_prompt": "unused",
                            "sam3_prompt": "frontmost carton",
                        }
                    },
                ),
                patch.object(
                    server,
                    "shelf_row_artifact_urls",
                    return_value={
                        "shelf_mask_url": "/shelf-mask.png",
                        "retained_mask_url": "/retained-mask.png",
                        "shelf_filtered_url": "/filtered.jpg",
                        "selected_component": {"width_ratio": 0.72},
                    },
                ),
            ):
                payload = server.list_sam_row_debug_records()

        self.assertEqual(len(payload["records"]), 1)
        row = payload["records"][0]["rows"][0]
        self.assertEqual(row["candidate_skus"], ["NFC桔汁"])
        self.assertEqual(
            row["shelf_inputs"]["current"]["shelf_filtered_url"],
            "/filtered.jpg",
        )
        self.assertEqual(
            payload["prompt_mapping"],
            [{"sku_name": "NFC桔汁", "prompt": "frontmost carton"}],
        )

    def test_locate_pages_expose_offline_full_pipeline_controls(self) -> None:
        index_html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
        app_js = (server.STATIC_DIR / "app.js").read_text(encoding="utf-8")
        qwen_html = (server.STATIC_DIR / "qwen_debug.html").read_text(encoding="utf-8")
        qwen_js = (server.STATIC_DIR / "qwen_debug.js").read_text(encoding="utf-8")
        self.assertIn('id="locateImageInput"', index_html)
        self.assertIn('id="batchRgbImage"', index_html)
        self.assertIn('id="batchDepthImage"', index_html)
        self.assertIn('id="batchResultFileSelect"', index_html)
        self.assertIn("loadBatchResultFiles", app_js)
        self.assertIn("/api/sorting-batch-results", app_js)
        self.assertIn("rerunBatchResultsWithOverwrite", app_js)
        self.assertIn("重跑当前项（--overwrite）", index_html)
        self.assertIn("requestPayload.image_base64", app_js)
        self.assertIn('id="runFullLocate"', qwen_html)
        self.assertIn('id="locateInventory"', qwen_html)
        self.assertIn('id="locateSlot"', qwen_html)
        self.assertIn('image_name: originalImageName', qwen_js)
        self.assertIn('image_base64: originalImageDataUrl', qwen_js)
        self.assertIn('mock_inventory: mockInventory', qwen_js)

    def test_qwen_review_generates_reference_mask_from_existing_finding(self) -> None:
        review_html = (server.STATIC_DIR / "qwen_review.html").read_text(
            encoding="utf-8"
        )
        review_js = (server.STATIC_DIR / "qwen_review.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="shortageReferenceMaskList"', review_html)
        self.assertIn(
            "/api/qwen-review/shortage-batch/reference-mask",
            review_js,
        )
        self.assertNotIn("/place-mask", review_html)
        self.assertFalse((server.STATIC_DIR / "place_mask.html").exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask_path = root / "group" / "record" / "region_01_mask.png"
            mask_path.parent.mkdir(parents=True)
            component_mask = server.np.zeros((100, 120), dtype=server.np.uint8)
            component_mask[22:58, 31:69] = 255
            success, encoded = server.cv2.imencode(".png", component_mask)
            self.assertTrue(success)
            encoded.tofile(mask_path)

            result = {
                "location_id": "H1_B_L_INSPECT",
                "pose_type": "SHELF_VIEW_LOWER",
                "findings": [
                    {
                        "region_index": 1,
                        "bbox": [30, 20, 40, 40],
                        "mask": "group/record/region_01_mask.png",
                        "product_name": "妙洁海绵百洁布",
                    }
                ],
                "row_detection": {
                    "rows": [{"bbox": [0, 10, 120, 70]}],
                },
            }
            initial_scan = Mock(
                rgb=server.np.zeros((100, 120, 3), dtype=server.np.uint8),
                inspection_target_id="H1_B_L_INSPECT",
                pose_type="SHELF_VIEW_LOWER",
            )
            generated_mask = server.np.zeros((100, 120), dtype=server.np.uint8)
            generated_mask[24:60, 34:72] = 255
            generated = Mock(
                mask=generated_mask,
                sam_prompt="product package",
                crop_box=(10, 5, 90, 90),
                selected_bbox=(34.0, 24.0, 72.0, 60.0),
                selected_score=0.93,
                candidate_count=2,
            )
            with (
                patch.object(server, "SHORTAGE_BATCH_ROOT", root),
                patch.object(server, "shortage_batch_result", return_value=result),
                patch.object(server, "load_initial_scan", return_value=initial_scan),
                patch.object(
                    server,
                    "generate_reference_mask",
                    return_value=generated,
                ) as generate,
            ):
                payload = server.generate_shortage_reference_mask(
                    server.ShortageReferenceMaskRequest(
                        group="group",
                        record="record",
                        region_index=1,
                    )
                )

        self.assertEqual(payload["product_name"], "妙洁海绵百洁布")
        self.assertEqual(payload["row_bbox"], [0.0, 10.0, 120.0, 70.0])
        self.assertTrue(payload["reference_mask_data_url"].startswith("data:image/png;base64,"))
        self.assertTrue(payload["reference_overlay_data_url"].startswith("data:image/png;base64,"))
        self.assertEqual(generate.call_args.args[1], [30.0, 20.0, 40.0, 40.0])
        self.assertEqual(generate.call_args.args[2], "妙洁海绵百洁布")
        self.assertEqual(
            generate.call_args.kwargs["component_mask"].shape,
            (100, 120),
        )
        self.assertEqual(
            generate.call_args.kwargs["row_bbox"],
            [0.0, 10.0, 120.0, 70.0],
        )

    def test_moved_web_paths_are_anchored_to_perception_root(self) -> None:
        expected_root = Path(__file__).resolve().parent

        self.assertEqual(server.ROOT, expected_root)
        self.assertEqual(server.PERCEPTION_ROOT, expected_root.parent)
        self.assertEqual(
            server.LOCATE_ROOT,
            expected_root.parent / "pick" / "locate",
        )
        self.assertEqual(
            server.INSPECT_ROOT,
            expected_root.parent / "inspect",
        )
        self.assertEqual(
            server.SKU_CATALOG_PATH,
            expected_root.parent / "sku" / "products.json",
        )

    def test_qwen_review_lists_and_serves_initial_scan_row_detection(self) -> None:
        review_html = (server.STATIC_DIR / "qwen_review.html").read_text(
            encoding="utf-8"
        )
        review_js = (server.STATIC_DIR / "qwen_review.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="initialScanSelect"', review_html)
        self.assertIn("/api/qwen-review/initial-scans", review_js)

        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            scan_root = temporary_root / "task0"
            scan_directory = scan_root / "H1_B_L_INSPECT_UPPER"
            scan_directory.mkdir(parents=True)
            image = server.np.zeros((720, 1280, 3), dtype=server.np.uint8)
            server.cv2.line(image, (0, 220), (1279, 220), (0, 0, 255), 10)
            server.cv2.line(image, (0, 500), (1279, 500), (0, 0, 255), 10)
            success, encoded = server.cv2.imencode(".jpg", image)
            self.assertTrue(success)
            encoded.tofile(scan_directory / "rgb.jpg")
            result_root = temporary_root / "row_results"

            with (
                patch.object(server, "initial_scan_root", return_value=scan_root),
                patch.object(server, "INITIAL_SCAN_ROW_RESULTS_ROOT", result_root),
                patch.object(
                    server,
                    "detect_rows",
                    wraps=server.detect_rows,
                ) as detect_rows,
            ):
                first = server.list_initial_scan_rows()
                second = server.list_initial_scan_rows()
                source_response = server.get_initial_scan_source(
                    "H1_B_L_INSPECT_UPPER"
                )
                overlay_response = server.get_initial_scan_row_overlay(
                    "H1_B_L_INSPECT_UPPER"
                )

            self.assertEqual(detect_rows.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(len(first["samples"]), 1)
            sample = first["samples"][0]
            self.assertEqual(sample["pose_type"], "SHELF_VIEW_UPPER")
            self.assertEqual(sample["rail_count"], 2)
            self.assertEqual(sample["row_count"], 2)
            self.assertEqual(Path(source_response.path), scan_directory / "rgb.jpg")
            self.assertTrue(Path(overlay_response.path).is_file())

        with self.assertRaises(HTTPException) as raised:
            server.resolve_initial_scan_for_web("../H1_B_L_INSPECT_UPPER")
        self.assertEqual(raised.exception.status_code, 400)

    def test_qwen_review_lists_shortage_batch_bbox_mask_and_product(self) -> None:
        review_html = (server.STATIC_DIR / "qwen_review.html").read_text(
            encoding="utf-8"
        )
        review_js = (server.STATIC_DIR / "qwen_review.js").read_text(
            encoding="utf-8"
        )
        review_css = (server.STATIC_DIR / "qwen_review.css").read_text(
            encoding="utf-8"
        )
        self.assertIn('id="shortageBatchGroupSelect"', review_html)
        self.assertIn('id="shortageBatchDatasetSelect"', review_html)
        self.assertIn('value="real_shortage">真实数据测试', review_html)
        self.assertIn('id="shortageBatchExpectedProducts"', review_html)
        self.assertIn('id="shortageBatchBaselineImage"', review_html)
        self.assertIn('id="shortageBatchRowImage"', review_html)
        self.assertIn('id="shortageBatchPromptList"', review_html)
        self.assertNotIn("缺货检测 → 商品身份完整链路", review_html)
        self.assertNotIn('id="taskSelect"', review_html)
        self.assertNotIn('id="constraintTitle"', review_html)
        self.assertNotIn('id="modelInputs"', review_html)
        self.assertIn("shortage-qwen-image-grid", review_js)
        self.assertIn('input.kind === "candidate_sheet"', review_js)
        self.assertNotIn("input.description || input.kind", review_js)
        self.assertIn(
            "grid-template-columns: repeat(2, minmax(0, 1fr))",
            review_css,
        )
        self.assertIn("使用修改后的 Prompt 重试 Qwen", review_js)
        self.assertIn("Qwen 模型原始返回", review_js)
        self.assertNotIn("/api/qwen-review/samples", review_js)
        self.assertIn("/api/qwen-review/shortage-batch", review_js)
        self.assertIn("dataset=${encodeURIComponent(dataset)}", review_js)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group = "H1_B_L_INSPECT_UPPER"
            record = "record_20260816_010203_123456"
            result_root = root / group / record / "shortage_inspection"
            result_root.mkdir(parents=True)
            source_path = root / group / record / "rgb.jpg"
            overlay_path = result_root / "overlay.jpg"
            mask_path = result_root / "region_01_mask.png"
            row_detection_path = result_root / "row_detection.jpg"
            source_path.write_bytes(b"rgb")
            overlay_path.write_bytes(b"overlay")
            mask_path.write_bytes(b"mask")
            row_detection_path.write_bytes(b"rows")
            debug_directory = root / "qwen_debug" / "sample_SHORTAGE_debug"
            (debug_directory / "region_01").mkdir(parents=True)
            (debug_directory / "request.json").write_text(
                json.dumps(
                    {
                        "task_type": "SHORTAGE",
                        "location_id": "H1_B_L1_C01",
                        "pose_type": "SHELF_VIEW_UPPER",
                        "bboxes": [[10, 20, 30, 40]],
                    }
                ),
                encoding="utf-8",
            )
            (debug_directory / "region_01" / "prompt.txt").write_text(
                "=== SYSTEM ===\ntest prompt",
                encoding="utf-8",
            )
            (debug_directory / "region_01" / "qwen_image_01.jpg").write_bytes(
                b"reference"
            )
            (debug_directory / "region_01" / "qwen_raw.txt").write_text(
                '{"shortage_product_name":"NFC桔汁","confidence":0.9}',
                encoding="utf-8",
            )
            (debug_directory / "region_01" / "parsed_result.json").write_text(
                json.dumps(
                    {
                        "accepted": True,
                        "shortage_product_name": "NFC桔汁",
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (debug_directory / "candidates.json").write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "sku_id": "SKU_001",
                                "name": "NFC桔汁",
                                "row_numbers": [1],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            scan_root = root / "task0"
            baseline_path = scan_root / group / "rgb.jpg"
            baseline_path.parent.mkdir(parents=True)
            baseline_path.write_bytes(b"baseline")
            summary_path = root / "shortage_inspection_batch_results.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "total_records": 1,
                        "completed_records": 1,
                        "status_counts": {"success": 1},
                        "results": [
                            {
                                "group": group,
                                "record": record,
                                "status": "success",
                                "location_id": "H1_B_L1_C01",
                                "pose_type": "SHELF_VIEW_UPPER",
                                "source_rgb": f"{group}/{record}/rgb.jpg",
                                "findings": [
                                    {
                                        "region_index": 1,
                                        "bbox": [10, 20, 30, 40],
                                        "mask": (
                                            f"{group}/{record}/shortage_inspection/"
                                            "region_01_mask.png"
                                        ),
                                        "product_name": "可口可乐罐装",
                                    }
                                ],
                                "artifacts": {
                                    "overlay": (
                                        f"{group}/{record}/shortage_inspection/"
                                        "overlay.jpg"
                                    ),
                                    "row_detection": (
                                        f"{group}/{record}/shortage_inspection/"
                                        "row_detection.jpg"
                                    ),
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with (
                patch.object(server, "SHORTAGE_BATCH_ROOT", root),
                patch.object(server, "SHORTAGE_BATCH_SUMMARY_PATH", summary_path),
                patch.object(server, "initial_scan_root", return_value=scan_root),
            ):
                payload = server.list_shortage_batch_results()
                file_response = server.get_shortage_batch_file(
                    f"{group}/{record}/shortage_inspection/region_01_mask.png"
                )

        sample = payload["samples"][0]
        self.assertEqual(sample["findings"][0]["bbox"], [10, 20, 30, 40])
        self.assertEqual(
            sample["findings"][0]["product_name"],
            "可口可乐罐装",
        )
        self.assertIn("region_01_mask.png", sample["findings"][0]["mask_url"])
        self.assertEqual(
            sample["findings"][0]["qwen_prompt"],
            "=== SYSTEM ===\ntest prompt",
        )
        qwen_images = sample["findings"][0]["qwen_images"]
        self.assertEqual(len(qwen_images), 2)
        self.assertIn("qwen_image_01.jpg", qwen_images[0]["url"])
        self.assertEqual(qwen_images[1]["label"], "NFC桔汁")
        self.assertIn("sku-image/SKU_001", qwen_images[1]["url"])
        self.assertIn(
            '"shortage_product_name":"NFC桔汁"',
            sample["findings"][0]["qwen_original_raw_output"],
        )
        self.assertEqual(
            sample["findings"][0]["qwen_original_parsed_result"][
                "shortage_product_name"
            ],
            "NFC桔汁",
        )
        self.assertTrue(sample["overlay_url"])
        self.assertIn("row_detection.jpg", sample["row_detection_url"])
        self.assertIn(
            f"/api/qwen-review/initial-scan/{group}/source",
            sample["baseline_rgb_url"],
        )
        self.assertEqual(Path(file_response.path), mask_path)

    def test_qwen_review_lists_real_shortage_regression_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group = "H1_F_L_INSPECT_UPPER"
            record = "record_20260816_160659_204194"
            record_root = root / group / record
            result_root = record_root / "shortage_inspection"
            result_root.mkdir(parents=True)
            (record_root / "rgb.jpg").write_bytes(b"current")
            (record_root / "baseline_rgb.jpg").write_bytes(b"baseline")
            (result_root / "overlay.jpg").write_bytes(b"overlay")
            (root / "shortage_inspection_batch_results.json").write_text(
                json.dumps(
                    {
                        "total_records": 1,
                        "completed_records": 1,
                        "status_counts": {"no_anomaly": 1},
                        "results": [
                            {
                                "group": group,
                                "record": record,
                                "status": "no_anomaly",
                                "source_rgb": f"{group}/{record}/rgb.jpg",
                                "findings": [],
                                "expected": {"findings": []},
                                "artifacts": {
                                    "overlay": (
                                        f"{group}/{record}/shortage_inspection/overlay.jpg"
                                    )
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch.object(server, "REAL_SHORTAGE_BATCH_ROOT", root):
                payload = server.list_shortage_batch_results("real_shortage")
                file_response = server.get_shortage_batch_file(
                    f"{group}/{record}/rgb.jpg",
                    "real_shortage",
                )

        self.assertEqual(payload["dataset"], "real_shortage")
        self.assertEqual(payload["dataset_label"], "真实数据测试")
        self.assertEqual(len(payload["samples"]), 1)
        sample = payload["samples"][0]
        self.assertEqual(sample["dataset"], "real_shortage")
        self.assertIn("dataset=real_shortage", sample["source_rgb_url"])
        self.assertIn("baseline_rgb.jpg", sample["baseline_rgb_url"])
        self.assertEqual(Path(file_response.path), record_root / "rgb.jpg")

    def test_shortage_batch_qwen_retry_uses_edited_prompt_and_returns_raw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_1 = root / "image_01.jpg"
            image_2 = root / "image_02.jpg"
            image_1.write_bytes(b"reference")
            image_2.write_bytes(b"candidate")
            retry_path = root / "retry.json"
            prompt = (
                "=== SYSTEM ===\nmodified system\n\n"
                "=== USER ===\nlook here\n[IMAGE 1]\nthen candidate\n[IMAGE 2]\n"
            )
            raw_output = (
                '{"product_name":"NFC桔汁","confidence":0.93}'
            )
            captured: dict = {}

            def fake_qwen(system_prompt, user_content, *, temperature):
                captured["system_prompt"] = system_prompt
                captured["user_content"] = user_content
                captured["temperature"] = temperature
                return raw_output

            with (
                patch.object(server, "shortage_batch_result", return_value={}),
                patch.object(server, "shortage_debug_region", return_value={}),
                patch.object(
                    server,
                    "shortage_qwen_image_paths",
                    return_value=[image_1, image_2],
                ),
                patch.object(server, "call_qwen_messages", side_effect=fake_qwen),
                patch.object(
                    server,
                    "shortage_retry_result_path",
                    return_value=retry_path,
                ),
            ):
                result = server.retry_shortage_batch_qwen(
                    server.ShortageBatchQwenRetryRequest(
                        group="H1_B_L_INSPECT_LOWER",
                        record="record_20260816_010203_123456",
                        region_index=1,
                        prompt=prompt,
                        temperature=0.2,
                    )
                )

            saved = json.loads(retry_path.read_text(encoding="utf-8"))

        self.assertEqual(captured["system_prompt"], "modified system")
        self.assertEqual(captured["temperature"], 0.2)
        self.assertEqual(
            [item["type"] for item in captured["user_content"]],
            ["text", "image_url", "text", "image_url"],
        )
        self.assertEqual(result["raw_output"], raw_output)
        self.assertEqual(result["parsed_result"]["product_name"], "NFC桔汁")
        self.assertEqual(saved["prompt_used"], prompt.strip())

    def test_write_json_atomic_creates_missing_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shortage_inspection" / "retry" / "result.json"

            server.write_json_atomic(path, {"ok": True}, "测试结果")

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"ok": True},
            )

    def test_sorting_batch_gallery_lists_and_serves_rgb_depth_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_name = "record_20260813_081055_905807"
            record_root = root / record_name
            record_root.mkdir()
            (record_root / "rgb.jpg").write_bytes(b"rgb")
            (record_root / "测试商品.jpg").write_bytes(b"result")
            (record_root / "测试商品.json").write_text(
                json.dumps(
                    {
                        "response": {
                            "qwen3_prompt_used": "测试 Qwen prompt",
                            "sam3_prompt_used": "测试 SAM prompt",
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            server.np.save(
                record_root / "depth_mm.npy",
                server.np.array([[0, 450], [500, 700]], dtype=server.np.uint16),
            )
            summary_path = root / "sorting_pick_locate_batch_results.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "total_records": 1,
                        "total_detections": 1,
                        "completed": 1,
                        "successes": 1,
                        "failures": 0,
                        "results": [
                            {
                                "record": record_name,
                                "product_name": "测试商品",
                                "status": "success",
                                "selected_depth_mm": 450,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with (
                patch.object(server, "SORTING_BATCH_ROOT", root),
                patch.object(server, "SORTING_BATCH_RESULTS_PATH", summary_path),
                patch.object(
                    server,
                    "is_hard_case_request",
                    return_value=False,
                ),
                patch.object(
                    server,
                    "load_prompt_pair_mapping",
                    return_value={
                        "测试商品": {
                            "qwen3_prompt": "测试 Qwen prompt",
                            "sam3_prompt": "测试 SAM prompt",
                        }
                    },
                ),
            ):
                result = server.list_sorting_batch_results()
                record = result["records"][0]
                self.assertEqual(record["record"], record_name)
                self.assertIn("/rgb?", record["rgb_url"])
                self.assertIn("/depth?", record["depth_url"])
                self.assertIn("product_name=", record["items"][0]["result_url"])
                self.assertEqual(
                    record["items"][0]["qwen3_prompt_used"],
                    "测试 Qwen prompt",
                )
                self.assertTrue(
                    record["items"][0]["prompt_matches_current_mapping"]
                )

                rgb_response = server.get_sorting_batch_image(record_name, "rgb")
                self.assertEqual(Path(rgb_response.path), record_root / "rgb.jpg")
                depth_response = server.get_sorting_batch_image(record_name, "depth")
                self.assertTrue(depth_response.body.startswith(b"\x89PNG\r\n\x1a\n"))
                result_response = server.get_sorting_batch_image(
                    record_name,
                    "result",
                    product_name="测试商品",
                )
                self.assertEqual(Path(result_response.path), record_root / "测试商品.jpg")

    def test_sorting_batch_file_selector_supports_self_collect_depth_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            batch_root = data_root / "2026-08-15-self-collect"
            batch_root.mkdir()
            record_name = "H1_B_L_INSPECT-L1-LEFT"
            record_root = batch_root / record_name
            record_root.mkdir()
            (record_root / "rgb.jpg").write_bytes(b"rgb")
            ok, depth_png = server.cv2.imencode(
                ".png",
                server.np.array([[0, 450], [500, 700]], dtype=server.np.uint16),
            )
            self.assertTrue(ok)
            (record_root / "depth.png").write_bytes(depth_png.tobytes())
            (record_root / "小苏打.jpg").write_bytes(b"result")
            (record_root / "小苏打.json").write_text(
                json.dumps(
                    {
                        "response": {
                            "qwen3_prompt_used": "测试 Qwen prompt",
                            "sam3_prompt_used": "测试 SAM prompt",
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (batch_root / "sorting_pick_locate_batch.json").write_text(
                json.dumps(
                    {
                        "rgb_file": "rgb.jpg",
                        "depth_file": "depth.png",
                    }
                ),
                encoding="utf-8",
            )
            results_path = batch_root / "sorting_pick_locate_batch_results.json"
            results_path.write_text(
                json.dumps(
                    {
                        "total_records": 1,
                        "total_detections": 1,
                        "completed": 1,
                        "successes": 1,
                        "failures": 0,
                        "results": [
                            {
                                "record": record_name,
                                "product_name": "小苏打",
                                "status": "success",
                                "level": "L1",
                                "hand": "left",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result_id = "2026-08-15-self-collect/sorting_pick_locate_batch_results.json"
            with (
                patch.object(server, "DATA_ROOT", data_root),
                patch.object(server, "is_hard_case_request", return_value=False),
                patch.object(
                    server,
                    "load_prompt_pair_mapping",
                    return_value={
                        "小苏打": {
                            "qwen3_prompt": "测试 Qwen prompt",
                            "sam3_prompt": "测试 SAM prompt",
                        }
                    },
                ),
            ):
                files = server.list_sorting_batch_result_files()
                result = server.list_sorting_batch_results(result_id)
                depth_response = server.get_sorting_batch_image(
                    record_name,
                    "depth",
                    result_file=result_id,
                )

        self.assertEqual(files["default"], result_id)
        self.assertEqual(files["files"][0]["label"], "2026-08-15-self-collect")
        self.assertEqual(result["dataset"], "2026-08-15-self-collect")
        self.assertIn("result_file=", result["records"][0]["rgb_url"])
        self.assertTrue(depth_response.body.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_sorting_batch_rerun_starts_selected_overwrite_once(self) -> None:
        process = Mock()
        process.pid = 4321
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner_path = root / "batch_record_inference.py"
            runner_path.write_text("# test runner\n", encoding="utf-8")
            log_path = root / "rerun.log"
            record_name = "record_20260813_081055_905807"
            record_root = root / record_name
            record_root.mkdir()
            (record_root / "测试商品.json").write_text("{}", encoding="utf-8")
            with (
                patch.object(server, "SORTING_BATCH_ROOT", root),
                patch.object(server, "SORTING_BATCH_RESULTS_PATH", root / "results.json"),
                patch.object(server, "SORTING_BATCH_RUNNER_PATH", runner_path),
                patch.object(server, "SORTING_BATCH_RERUN_LOG_PATH", log_path),
                patch.object(server, "SORTING_BATCH_PROCESS", None),
                patch.object(server, "SORTING_BATCH_PROCESS_STARTED_AT", None),
                patch.object(server, "SORTING_BATCH_PROCESS_TARGET", None),
                patch.object(server.subprocess, "Popen", return_value=process) as popen,
            ):
                request = server.SortingBatchRerunRequest(
                    record=record_name,
                    product_name="测试商品",
                )
                first = server.start_sorting_batch_rerun(request)
                second = server.start_sorting_batch_rerun(request)

        self.assertTrue(first["running"])
        self.assertEqual(first["pid"], 4321)
        self.assertEqual(second["pid"], 4321)
        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertIn("--overwrite", command)
        self.assertEqual(command[-4:], ["--record", record_name, "--product-name", "测试商品"])

    def test_qwen_infer_samples_include_all_current_pairs(self) -> None:
        result = server.list_qwen_infer_samples()

        self.assertEqual(len(result["samples"]), 10)
        counts = {
            (sample["task_type"], sample["pair_number"]): len(sample["regions"])
            for sample in result["samples"]
        }
        self.assertGreaterEqual(counts[("SHORTAGE", 1)], 1)
        self.assertEqual(counts[("MISPLACED", 3)], 2)
        self.assertTrue(
            result["samples"][0]["regions"][0]["expanded_image_url"].startswith(
                "/api/qwen-review/file/"
            )
        )
        pair_three = next(
            sample
            for sample in result["samples"]
            if sample["task_type"] == "SHORTAGE" and sample["pair_number"] == 3
        )
        self.assertTrue(pair_three["aligned_current_url"])
        self.assertTrue(pair_three["row_overlay_url"])
        self.assertEqual(pair_three["row_detection"]["expected_row_count"], 2)
        region = pair_three["regions"][0]
        self.assertEqual(region["row_constraint"]["row_index"], 1)
        self.assertLess(
            region["candidate_count_after"],
            region["candidate_count_before"],
        )
        self.assertEqual(
            len(region["candidate_images"]),
            region["candidate_count_after"],
        )

        pair_one = next(
            sample
            for sample in result["samples"]
            if sample["task_type"] == "SHORTAGE" and sample["pair_number"] == 1
        )
        region = pair_one["regions"][0]
        self.assertEqual(pair_one["row_detection"]["row_window_anchor"], "bottom")
        self.assertEqual(region["row_constraint"]["detected_row_index"], 3)
        self.assertEqual(region["row_constraint"]["row_index"], 3)
        self.assertEqual(region["candidate_count_before"], 9)
        self.assertEqual(region["candidate_count_after"], 2)
        self.assertEqual(
            [candidate["name"] for candidate in region["candidate_images"]],
            ["拖鞋", "心相印厨房纸巾"],
        )
        self.assertEqual(region["prompt_source"], "generated")
        self.assertIsNone(region["prompt_warning"])
        self.assertIn("你会看到一张货架摆放特写图", region["prompt"])
        self.assertIn(
            "请只审核下面这一张货架局部图："
            "缺货商品只能从以下候选商品中选择。",
            region["prompt"],
        )
        self.assertNotIn("缺货前 reference", region["prompt"])
        self.assertIn("候选商品：拖鞋、心相印厨房纸巾", region["prompt"])
        self.assertNotIn("Dove沐浴泡泡樱花甜香", region["prompt"])

        misplaced_pair = next(
            sample
            for sample in result["samples"]
            if sample["task_type"] == "MISPLACED" and sample["pair_number"] == 1
        )
        misplaced_region = misplaced_pair["regions"][0]
        stages = misplaced_region["prompt_stages"]
        self.assertEqual(
            [stage["stage"] for stage in stages],
            ["misplaced_product", "expected_product"],
        )
        self.assertEqual(stages[0]["candidate_scope"], "all_visible_rows")
        self.assertEqual(stages[0]["candidate_count_after"], 20)
        self.assertEqual(stages[1]["candidate_scope"], "expected_row")
        self.assertEqual(stages[1]["candidate_count_after"], 13)
        self.assertEqual(len(stages[0]["candidate_sheets"]), 1)
        self.assertEqual(len(stages[1]["candidate_sheets"]), 1)
        self.assertEqual(stages[0]["candidate_sheets"][0]["prompt_image_number"], 2)
        self.assertTrue(stages[0]["candidate_sheets"][0]["url"])
        self.assertIn("misplaced_product_name", stages[0]["prompt"])
        self.assertIn(
            "任务:识别局部图红色 bbox 中当前实际放置的商品。",
            stages[0]["prompt"],
        )
        self.assertIn("SKU 1: NFC桔汁", stages[0]["prompt"])
        self.assertNotIn("location_id=", stages[0]["prompt"])
        self.assertNotIn("gt_product_name", stages[0]["prompt"].split("=== USER ===")[0])
        self.assertIn("gt_product_name", stages[1]["prompt"])
        self.assertNotIn("misplaced_product_name", stages[1]["prompt"])
        self.assertIn("货架标准放置图", stages[1]["prompt"])
        self.assertIn("标准放置组合图", stages[1]["prompt"])
        self.assertIn("下方是 bbox 内物体的原图抠图", stages[1]["prompt"])
        self.assertIn("红色 bbox 中的物体是什么", stages[1]["prompt"])
        self.assertIn("这一层从左到右SKU标准放置编号如下", stages[1]["prompt"])
        self.assertIn("输出商品名必须从以上名称中逐字选择", stages[1]["prompt"])
        self.assertNotIn("原本应该放置", stages[1]["prompt"])
        self.assertNotIn("对比图", stages[1]["prompt"])
        self.assertIn("任务：根据候选SKU，识别红色 bbox 中的物体是什么", stages[1]["prompt"])
        self.assertNotIn("候选 SKU 1..N 按标准货位列号从左到右排列", stages[1]["prompt"])
        self.assertNotIn("任务=MISPLACED 第二阶段", stages[1]["prompt"])

        for sample in result["samples"]:
            if sample["task_type"] != "MISPLACED":
                continue
            for sample_region in sample["regions"]:
                expected_stage = next(
                    stage
                    for stage in sample_region["prompt_stages"]
                    if stage["stage"] == "expected_product"
                )
                self.assertIn("货架标准放置图", expected_stage["prompt"])
                self.assertIn("标准放置组合图", expected_stage["prompt"])
                self.assertIn(
                    "下方是 bbox 内物体的原图抠图",
                    expected_stage["prompt"],
                )
                self.assertIn(
                    "这一层从左到右SKU标准放置编号如下",
                    expected_stage["prompt"],
                )
                self.assertNotIn("对比图", expected_stage["prompt"])
                self.assertNotIn(
                    "任务=MISPLACED 第二阶段",
                    expected_stage["prompt"],
                )

    def test_saved_prompt_builds_interleaved_multimodal_content(self) -> None:
        pair_root, manifest = server.load_qwen_sample("shortage", 1)
        region = manifest["regions"][0]
        prompt = (
            pair_root / "region_01" / "prompt.txt"
        ).read_text(encoding="utf-8")

        system_prompt, content = server.build_saved_qwen_messages(
            prompt,
            pair_root,
            region,
            server.candidate_images_for_region(manifest, region),
        )

        self.assertIn("shortage_product_name", system_prompt)
        image_items = [item for item in content if item["type"] == "image_url"]
        self.assertEqual(
            len(image_items),
            len(server.candidate_images_for_region(manifest, region)) + 1,
        )
        self.assertTrue(
            image_items[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )

    def test_misplaced_generated_prompts_match_each_stage_inputs(self) -> None:
        pair_root, manifest = server.load_qwen_sample("misplaced", 1)
        region = manifest["regions"][0]
        region_root = pair_root / "region_01"
        stages = server.prompt_stages_for_region(manifest, region)

        self.assertEqual(len(stages), 2)
        for stage in stages:
            generated_path, _, _, _ = server.resolve_prompt_stage_paths(
                pair_root,
                region_root,
                stage,
            )
            prompt = generated_path.read_text(encoding="utf-8")
            candidates = server.candidate_images_for_stage(stage)
            references = server.prompt_reference_images_for_stage(stage)
            server.validate_saved_prompt_images(prompt, len(references) + 1)
            server.validate_saved_prompt_candidates(prompt, candidates)
            _, content = server.build_saved_qwen_messages(
                prompt,
                pair_root,
                stage,
                candidates,
            )
            image_items = [item for item in content if item["type"] == "image_url"]
            self.assertEqual(len(image_items), len(references) + 1)
            self.assertLess(len(image_items), len(candidates) + 1)

    def test_numbered_candidate_mapping_rejects_wrong_order(self) -> None:
        valid = """=== SYSTEM ===
system
=== USER ===
[IMAGE 1]
候选 SKU 编号（与下方标准图拼图上方数字一致）：
SKU 1: 商品A
SKU 2: 商品B
[IMAGE 2]
"""
        self.assertTrue(
            server.prompt_has_expected_candidates(valid, ["商品A", "商品B"])
        )
        self.assertFalse(
            server.prompt_has_expected_candidates(valid, ["商品B", "商品A"])
        )

        shortage = """=== SYSTEM ===
system
=== USER ===
[IMAGE 1]
候选（与下方标准图拼图上方数字一致）：
1: 商品A
2: 商品B
[IMAGE 2]
"""
        self.assertTrue(
            server.prompt_has_expected_candidates(shortage, ["商品A", "商品B"])
        )
        self.assertFalse(
            server.prompt_has_expected_candidates(shortage, ["商品B", "商品A"])
        )

    def test_misplaced_stage_prompt_saves_against_its_own_candidates(self) -> None:
        prompt = """=== SYSTEM ===
system
=== USER ===
[IMAGE 1]
候选商品：商品A、商品B
CANDIDATE 1: 商品A;
[IMAGE 2]
CANDIDATE 2: 商品B;
[IMAGE 3]
"""
        region = {
            "region_index": 1,
            "prompt_stages": [
                {
                    "stage": "expected_product",
                    "prompt": "region_01/expected_product/prompt.txt",
                    "prompt_image_1": "region_01/expected_product/input.jpg",
                    "candidate_images": [
                        {"name": "商品A", "path": "candidate/a.jpg"},
                        {"name": "商品B", "path": "candidate/b.jpg"},
                    ],
                }
            ],
        }
        manifest = {"task_type": "MISPLACED", "regions": [region]}
        with tempfile.TemporaryDirectory() as temporary_directory:
            pair_root = Path(temporary_directory)
            region_root = pair_root / "region_01"
            stage_root = region_root / "expected_product"
            stage_root.mkdir(parents=True)
            request = server.SaveSavedQwenPromptRequest(
                dataset="misplaced",
                pair_number=1,
                region_index=1,
                stage="expected_product",
                prompt=prompt,
            )
            with patch.object(
                server,
                "load_qwen_region",
                return_value=(manifest, region_root, region, pair_root),
            ):
                result = server.save_qwen_infer_prompt(request)

            self.assertEqual(result["stage"], "expected_product")
            self.assertEqual(
                (stage_root / "prompt_override.txt").read_text(encoding="utf-8"),
                prompt,
            )

    def test_qwen_infer_file_rejects_parent_traversal(self) -> None:
        root, _ = server.load_qwen_sample("shortage", 1)
        with self.assertRaises(HTTPException) as raised:
            server.resolve_descendant(root, "../manifest.json")
        self.assertEqual(raised.exception.status_code, 400)

    def test_legacy_absolute_sample_path_is_reanchored_to_current_data(self) -> None:
        legacy = (
            r"C:\old\checkout\perception\test_data"
            r"\inspect_shortage_paired\1_1.jpg"
        )

        self.assertEqual(
            server.resolve_sample_data_path(legacy),
            (
                server.DATA_ROOT
                / "inspect_shortage_paired"
                / "1_1.jpg"
            ).resolve(),
        )

    def test_absolute_path_outside_test_data_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            server.resolve_sample_data_path(r"C:\private\receipt.jpg")
        self.assertEqual(raised.exception.status_code, 400)

    def test_saved_prompt_rejects_stale_candidate_image_count(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            server.validate_saved_prompt_images(
                "=== SYSTEM ===\nsystem\n=== USER ===\n[IMAGE 1]\n[IMAGE 2]",
                3,
            )

        self.assertEqual(raised.exception.status_code, 400)

    def test_saved_prompt_rejects_same_count_with_stale_candidate_names(self) -> None:
        prompt = """=== SYSTEM ===
system
=== USER ===
[IMAGE 1]
候选商品：旧商品A、旧商品B
CANDIDATE 1: 旧商品A;
[IMAGE 2]
CANDIDATE 2: 旧商品B;
[IMAGE 3]
"""

        with self.assertRaises(HTTPException) as raised:
            server.validate_saved_prompt_candidates(
                prompt,
                [{"name": "新商品A"}, {"name": "新商品B"}],
            )

        self.assertEqual(raised.exception.status_code, 400)

    def test_qwen_review_run_reports_backend_and_stage_timings(self) -> None:
        request = server.SavedQwenInferRequest(
            dataset="shortage",
            pair_number=1,
            region_index=1,
            stage="shortage_product",
            prompt="=== SYSTEM ===\nsystem\n=== USER ===\nuser",
            temperature=0.0,
        )
        manifest = {"task_type": "SHORTAGE", "candidate_images": []}
        region = {
            "region_index": 1,
            "prompt_stages": [
                {
                    "stage": "shortage_product",
                    "prompt": "region_01/shortage_product/prompt.txt",
                    "prompt_image_1": "region_01/shortage_product/input.jpg",
                    "candidate_images": [],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                patch.object(
                    server,
                    "load_qwen_region",
                    return_value=(manifest, root / "region_01", region, root),
                ),
                patch.object(
                    server,
                    "build_saved_qwen_messages",
                    return_value=("system", [{"type": "text", "text": "user"}]),
                ),
                patch.object(
                    server,
                    "call_qwen_messages",
                    return_value='{"shortage_product_name":"拖鞋","confidence":0.9}',
                ),
                patch.object(server, "write_json_atomic") as write_mock,
                patch.object(
                    server.time,
                    "perf_counter",
                    side_effect=[10.0, 10.1, 10.6, 10.65],
                ),
            ):
                result = server.run_saved_qwen_infer(request)

        self.assertEqual(result["elapsed_ms"], 500.0)
        self.assertEqual(result["stage"], "shortage_product")
        self.assertEqual(result["qwen_elapsed_ms"], 500.0)
        self.assertEqual(result["backend_elapsed_ms"], 650.0)
        self.assertEqual(
            result["timings"],
            {
                "prepare_inputs_ms": 100.0,
                "qwen_request_ms": 500.0,
                "parse_result_ms": 50.0,
                "backend_processing_ms": 650.0,
            },
        )
        self.assertEqual(result["parsed_result"]["shortage_product_name"], "拖鞋")
        self.assertEqual(write_mock.call_args.args[1], result)

    def test_full_inspect_run_times_real_pipeline_entry(self) -> None:
        class FakeFinding:
            def model_dump(self, *, mode: str) -> dict:
                self.mode = mode
                return {"shortage_product_name": "心相印厨房纸巾"}

        class FakeInspectResponse:
            def __init__(self, findings: list[FakeFinding]) -> None:
                self.findings = findings

        fake_inspect_api = Mock()
        fake_inspect_api.decode_image.side_effect = ["baseline-array", "current-array"]
        fake_inspect_api.inspect_supplied_images.return_value = FakeInspectResponse(
            [FakeFinding()]
        )
        manifest = {
            "task_type": "SHORTAGE",
            "location_id": "H2_B_L3_C01",
            "pose_type": "SHELF_VIEW_LOWER",
            "baseline": "sample/baseline.jpg",
            "current": "sample/current.jpg",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "sample").mkdir()
            (root / "sample" / "baseline.jpg").write_bytes(b"baseline")
            (root / "sample" / "current.jpg").write_bytes(b"current")
            with (
                patch.object(server, "DATA_ROOT", root),
                patch.object(server, "INSPECT_API", fake_inspect_api),
                patch.object(server, "load_qwen_sample", return_value=(root, manifest)),
                patch.object(
                    server.time,
                    "perf_counter",
                    side_effect=[20.0, 20.05, 20.8, 20.81],
                ),
            ):
                result = server.run_full_inspect(
                    server.FullInspectRunRequest(dataset="shortage", pair_number=1)
                )

        inspect_request = fake_inspect_api.inspect_supplied_images.call_args.kwargs
        self.assertEqual(inspect_request["task_type"], "SHORTAGE")
        self.assertEqual(inspect_request["location_id"], "H2_B_L3_C01")
        self.assertEqual(inspect_request["pose_type"], "SHELF_VIEW_LOWER")
        self.assertEqual(inspect_request["baseline"], "baseline-array")
        self.assertEqual(inspect_request["current"], "current-array")
        self.assertEqual(
            base64.b64decode(fake_inspect_api.decode_image.call_args_list[0].args[0]),
            b"baseline",
        )
        self.assertEqual(result["inspect_elapsed_ms"], 750.0)
        self.assertEqual(result["backend_elapsed_ms"], 810.0)
        self.assertEqual(result["finding_count"], 1)
        self.assertEqual(
            result["result"],
            {"findings": [{"shortage_product_name": "心相印厨房纸巾"}]},
        )

    def test_locate_debug_proxy_sends_no_local_image(self) -> None:
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {
            "image_base64": "aW1hZ2U=",
            "qwen_bboxes": [],
            "instances": [],
        }
        with patch.object(
            server.requests,
            "post",
            return_value=response,
        ) as post_mock:
            result = server.run_locate_debug(
                server.LocateDebugProxyRequest(
                    task_type="SORTING",
                    product_name="可口可乐",
                    level="L1",
                    hand="left",
                )
            )

        self.assertEqual(result["image_base64"], "aW1hZ2U=")
        post_mock.assert_called_once_with(
            server.LOCATE_DEBUG_URL,
            json={
                "task_type": "SORTING",
                "product_name": "可口可乐",
                "level": "L1",
                "hand": "left",
            },
            timeout=600,
        )

    def test_locate_debug_proxy_forwards_offline_image(self) -> None:
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {
            "image_base64": "aW1hZ2U=",
            "qwen_bboxes": [],
            "instances": [],
        }
        with patch.object(server.requests, "post", return_value=response) as post_mock:
            server.run_locate_debug(
                server.LocateDebugProxyRequest(
                    task_type="SORTING",
                    product_name="脉动菠萝口味",
                    level="L4",
                    hand="left",
                    image_name="hard_case.jpg",
                    image_base64="data:image/jpeg;base64,aW1hZ2U=",
                    depth_image_name="hard_case_depth.raw",
                    depth_image_base64="data:application/octet-stream;base64,ZGVwdGg=",
                )
            )
        self.assertEqual(
            post_mock.call_args.kwargs["json"],
            {
                "task_type": "SORTING",
                "product_name": "脉动菠萝口味",
                "level": "L4",
                "hand": "left",
                "image_name": "hard_case.jpg",
                "image_base64": "data:image/jpeg;base64,aW1hZ2U=",
                "depth_image_name": "hard_case_depth.raw",
                "depth_image_base64": "data:application/octet-stream;base64,ZGVwdGg=",
                "depth_is_bigendian": False,
            },
        )

    def test_locate_debug_proxy_forwards_request_scoped_mock_inventory(self) -> None:
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {
            "image_base64": "aW1hZ2U=",
            "qwen_bboxes": [],
            "instances": [],
        }
        with patch.object(server.requests, "post", return_value=response) as post_mock:
            server.run_locate_debug(
                server.LocateDebugProxyRequest(
                    task_type="SORTING",
                    product_name="测试商品",
                    level="L1",
                    hand="left",
                    slot_id="H1_L01_C03",
                    mock_inventory=["H1_L01_C02", "H1_L01_C03"],
                    image_name="rgb.jpg",
                    image_base64="aW1hZ2U=",
                )
            )

        payload = post_mock.call_args.kwargs["json"]
        self.assertEqual(payload["slot_id"], "H1_L01_C03")
        self.assertEqual(
            payload["mock_inventory"],
            ["H1_L01_C02", "H1_L01_C03"],
        )

    def test_locate_debug_proxy_allows_offline_hard_case_without_depth(self) -> None:
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {
            "image_base64": "aW1hZ2U=",
            "qwen_bboxes": [],
            "instances": [],
        }
        with patch.object(server.requests, "post", return_value=response) as post_mock:
            server.run_locate_debug(
                server.LocateDebugProxyRequest(
                    task_type="SORTING",
                    product_name="脉动菠萝口味",
                    level="L4",
                    hand="left",
                    image_name="hard_case.jpg",
                    image_base64="data:image/jpeg;base64,aW1hZ2U=",
                )
            )

        self.assertNotIn("depth_image_name", post_mock.call_args.kwargs["json"])

    def test_locate_debug_proxy_allows_normal_offline_case_without_depth(self) -> None:
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {
            "image_base64": "aW1hZ2U=",
            "qwen_bboxes": [],
            "instances": [],
        }
        with patch.object(server.requests, "post", return_value=response) as post_mock:
            server.run_locate_debug(
                server.LocateDebugProxyRequest(
                    task_type="SORTING",
                    product_name="可口可乐",
                    level="L1",
                    hand="left",
                    image_name="normal.jpg",
                    image_base64="data:image/jpeg;base64,aW1hZ2U=",
                )
            )

        self.assertNotIn("depth_image_name", post_mock.call_args.kwargs["json"])

    def test_locate_debug_proxy_rejects_half_offline_image(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            server.run_locate_debug(
                server.LocateDebugProxyRequest(
                    task_type="SORTING",
                    product_name="脉动菠萝口味",
                    level="L4",
                    hand="left",
                    image_name="hard_case.jpg",
                )
            )
        self.assertEqual(raised.exception.status_code, 400)

    def test_locate_debug_proxy_rejects_depth_without_rgb(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            server.run_locate_debug(
                server.LocateDebugProxyRequest(
                    task_type="SORTING",
                    product_name="可口可乐",
                    level="L1",
                    hand="left",
                    depth_image_name="depth.raw",
                    depth_image_base64="ZGVwdGg=",
                )
            )
        self.assertEqual(raised.exception.status_code, 400)

    def test_qwen_save_updates_canonical_pair_and_preserves_sam_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            mapping_path = directory / "qwen_sam_prompt_mapping.json"
            mapping_path.write_text(
                json.dumps(
                    {
                        "蒙牛纯牛奶": {
                            "qwen3_prompt": "旧 Qwen",
                            "sam3_prompt": "frontmost carton",
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with (
                patch.object(server, "PROMPT_PAIR_MAPPING_PATH", mapping_path),
                patch.object(
                    server,
                    "load_skus",
                    return_value=[{"name": "蒙牛纯牛奶"}],
                ),
            ):
                result = server.save_qwen_prompt(
                    server.SaveQwenPromptRequest(
                        task_type="SORTING",
                        sku_name="蒙牛纯牛奶",
                        prompt="新 Qwen",
                    )
                )

            saved = json.loads(mapping_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["蒙牛纯牛奶"]["qwen3_prompt"], "新 Qwen")
            self.assertEqual(
                saved["蒙牛纯牛奶"]["sam3_prompt"],
                "frontmost carton",
            )
            self.assertEqual(result["sam3_prompt"], "frontmost carton")
            self.assertFalse((directory / "qwen_prompt_mapping.json").exists())

    def test_qwen_only_save_rejects_sku_without_sam_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mapping_path = Path(temporary_directory) / "qwen_sam_prompt_mapping.json"
            mapping_path.write_text("{}\n", encoding="utf-8")
            with (
                patch.object(server, "PROMPT_PAIR_MAPPING_PATH", mapping_path),
                patch.object(
                    server,
                    "load_skus",
                    return_value=[{"name": "新商品"}],
                ),
                self.assertRaises(HTTPException) as raised,
            ):
                server.save_qwen_prompt(
                    server.SaveQwenPromptRequest(
                        task_type="SORTING",
                        sku_name="新商品",
                        prompt="Qwen Prompt",
                    )
                )

            self.assertEqual(raised.exception.status_code, 409)

    def test_sku_list_loads_both_prompts_from_canonical_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mapping_path = Path(temporary_directory) / "qwen_sam_prompt_mapping.json"
            mapping_path.write_text(
                json.dumps(
                    {
                        "商品": {
                            "qwen3_prompt": "Qwen Prompt",
                            "sam3_prompt": "SAM Prompt",
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with (
                patch.object(server, "PROMPT_PAIR_MAPPING_PATH", mapping_path),
                patch.object(
                    server,
                    "load_skus",
                    return_value=[{"name": "商品"}],
                ),
            ):
                result = server.list_skus()

            self.assertEqual(
                result["skus"],
                [
                    {
                        "name": "商品",
                        "locations": [],
                        "inventory": [],
                        "qwen3_prompt": "Qwen Prompt",
                        "sam3_prompt": "SAM Prompt",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
