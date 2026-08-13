from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException

import server


class PromptMappingTest(unittest.TestCase):
    def test_qwen_infer_samples_include_all_current_pairs(self) -> None:
        result = server.list_qwen_infer_samples()

        self.assertEqual(len(result["samples"]), 10)
        counts = {
            (sample["task_type"], sample["pair_number"]): len(sample["regions"])
            for sample in result["samples"]
        }
        self.assertEqual(counts[("SHORTAGE", 1)], 1)
        self.assertEqual(counts[("MISPLACED", 3)], 2)
        self.assertTrue(
            result["samples"][0]["regions"][0]["expanded_image_url"].startswith(
                "/api/qwen-infer/file/"
            )
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
            manifest["candidate_images"],
        )

        self.assertIn("shortage_product_name", system_prompt)
        image_items = [item for item in content if item["type"] == "image_url"]
        self.assertEqual(len(image_items), len(manifest["candidate_images"]) + 1)
        self.assertTrue(
            image_items[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )

    def test_qwen_infer_file_rejects_parent_traversal(self) -> None:
        root, _ = server.load_qwen_sample("shortage", 1)
        with self.assertRaises(HTTPException) as raised:
            server.resolve_descendant(root, "../manifest.json")
        self.assertEqual(raised.exception.status_code, 400)

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
