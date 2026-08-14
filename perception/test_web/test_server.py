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
        class FakeInspectRequest:
            def __init__(self, **values: object) -> None:
                self.values = values

        class FakeFinding:
            def model_dump(self, *, mode: str) -> dict:
                self.mode = mode
                return {"shortage_product_name": "心相印厨房纸巾"}

        class FakeInspectResponse:
            def __init__(self, findings: list[FakeFinding]) -> None:
                self.findings = findings

        fake_inspect_api = Mock()
        fake_inspect_api.InspectRequest = FakeInspectRequest
        fake_inspect_api.inspect_shelf.return_value = FakeInspectResponse(
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

        inspect_request = fake_inspect_api.inspect_shelf.call_args.args[0]
        self.assertEqual(inspect_request.values["task_type"], "SHORTAGE")
        self.assertEqual(inspect_request.values["location_id"], "H2_B_L3_C01")
        self.assertEqual(inspect_request.values["pose_type"], "SHELF_VIEW_LOWER")
        self.assertEqual(
            base64.b64decode(inspect_request.values["baseline_image_base64"]),
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
                    hand="left",
                )
            )

        self.assertEqual(result["image_base64"], "aW1hZ2U=")
        post_mock.assert_called_once_with(
            server.LOCATE_DEBUG_URL,
            json={
                "task_type": "SORTING",
                "product_name": "可口可乐",
                "hand": "left",
            },
            timeout=600,
        )

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
                        "qwen3_prompt": "Qwen Prompt",
                        "sam3_prompt": "SAM Prompt",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
