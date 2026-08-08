"""Tests for the pick check API."""

from __future__ import annotations

import base64
import json
import unittest
from unittest.mock import Mock, call, patch

import server


class PickCheckTest(unittest.TestCase):
    def test_uses_requested_hand_camera_and_returns_qwen_result(self) -> None:
        camera_response = Mock(
            content=b"hand image",
            headers={"Content-Type": "image/jpeg"},
        )
        camera_response.raise_for_status.return_value = None
        product_response = Mock()
        product_response.raise_for_status.return_value = None
        product_response.json.return_value = {"images": ["images/SKU_001.jpg"]}
        reference_response = Mock(
            content=b"reference image",
            headers={"Content-Type": "image/png"},
        )
        reference_response.raise_for_status.return_value = None
        qwen_response = Mock()
        qwen_response.raise_for_status.return_value = None
        qwen_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"pick_status": "Success"})
                    }
                }
            ]
        }

        with (
            patch.object(
                server,
                "hand_camera_snapshot_url",
                return_value="http://camera/right_wrist",
            ) as camera_url,
            patch.object(
                server.requests,
                "get",
                side_effect=[camera_response, product_response, reference_response],
            ) as get,
            patch.object(
                server.requests,
                "post",
                return_value=qwen_response,
            ) as post,
        ):
            result = server.check_product(
                server.PickCheckRequest(
                    task_type="SORTING",
                    product_name="可口可乐罐装",
                    hand="right",
                )
            )

        self.assertEqual(result.pick_status, "Success")
        camera_url.assert_called_once_with("right")
        self.assertEqual(
            get.call_args_list,
            [
                call(
                    "http://camera/right_wrist",
                    timeout=server.CAMERA_TIMEOUT_SECONDS,
                ),
                call(
                    f"{server.SKU_API_URL}/sku/search_by_name",
                    params={"name": "可口可乐罐装"},
                    timeout=server.SKU_TIMEOUT_SECONDS,
                ),
                call(
                    f"{server.SKU_API_URL}/images/SKU_001.jpg",
                    timeout=server.SKU_TIMEOUT_SECONDS,
                ),
            ],
        )

        content = post.call_args.kwargs["json"]["messages"][0]["content"]
        image_blocks = [block for block in content if block["type"] == "image_url"]
        self.assertEqual(len(image_blocks), 2)
        self.assertEqual(
            base64.b64decode(image_blocks[0]["image_url"]["url"].split(",", 1)[1]),
            b"reference image",
        )
        self.assertEqual(
            base64.b64decode(image_blocks[1]["image_url"]["url"].split(",", 1)[1]),
            b"hand image",
        )


if __name__ == "__main__":
    unittest.main()
