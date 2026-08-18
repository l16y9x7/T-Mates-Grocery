from __future__ import annotations

import base64
import unittest
from unittest.mock import Mock, call, patch

from pick.locate import main as locate_main


class ShortageReferenceImageTest(unittest.TestCase):
    def test_non_shortage_qwen_request_keeps_original_single_image_shape(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": '{"name":"商品","bbox":[1,2,3,4]}'}}]
        }
        with patch.object(locate_main.requests, "post", return_value=response) as post:
            locate_main.call_qwen3("定位商品", b"shelf-image")

        content = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertEqual([item["type"] for item in content], ["text", "image_url"])

    def test_qwen_request_labels_reference_and_scene_images(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": '{"name":"商品","bbox":[1,2,3,4]}'}}]
        }
        reference = locate_main.QwenReferenceImage(
            logical_name="images/SKU_TEST.jpg",
            media_type="image/png",
            content=b"reference-image",
        )

        with patch.object(locate_main.requests, "post", return_value=response) as post:
            locate_main.call_qwen3(
                "定位商品",
                b"shelf-image",
                reference_image=reference,
            )

        content = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertEqual(
            [item["type"] for item in content],
            ["text", "text", "image_url", "text", "image_url"],
        )
        self.assertIn("样例图", content[1]["text"])
        self.assertIn("第二张图", content[3]["text"])
        self.assertEqual(
            base64.b64decode(content[2]["image_url"]["url"].split(",", 1)[1]),
            b"reference-image",
        )
        self.assertEqual(
            base64.b64decode(content[4]["image_url"]["url"].split(",", 1)[1]),
            b"shelf-image",
        )

    def test_three_sample_consensus_reuses_same_reference_image(self) -> None:
        reference = locate_main.QwenReferenceImage(
            logical_name="images/SKU_TEST.jpg",
            media_type="image/jpeg",
            content=b"reference-image",
        )
        output = '{"name":"商品","bbox":[100,100,300,300]}'
        with patch.object(
            locate_main,
            "call_qwen3",
            side_effect=[output, output, output],
        ) as qwen:
            boxes = locate_main.get_stable_qwen_bboxes(
                "定位商品",
                b"shelf-image",
                reference_image=reference,
            )

        self.assertEqual(boxes, [[100.0, 100.0, 300.0, 300.0]])
        self.assertEqual(
            qwen.call_args_list,
            [
                call("定位商品", b"shelf-image", reference_image=reference),
                call("定位商品", b"shelf-image", reference_image=reference),
                call("定位商品", b"shelf-image", reference_image=reference),
            ],
        )

    def test_fetch_reference_image_uses_first_catalog_image(self) -> None:
        response = Mock()
        response.content = b"sku-image"
        response.headers = {"Content-Type": "image/jpeg; charset=binary"}
        response.raise_for_status.return_value = None
        with patch.object(locate_main.requests, "get", return_value=response) as get:
            reference = locate_main.fetch_sku_reference_image(
                {
                    "sku_id": "SKU_TEST",
                    "name": "商品",
                    "images": ["images/SKU_TEST.jpg"],
                }
            )

        self.assertEqual(reference.logical_name, "images/SKU_TEST.jpg")
        self.assertEqual(reference.media_type, "image/jpeg")
        self.assertEqual(reference.content, b"sku-image")
        self.assertEqual(
            get.call_args.args[0],
            f"{locate_main.SKU_API_URL}/images/SKU_TEST.jpg",
        )


if __name__ == "__main__":
    unittest.main()
