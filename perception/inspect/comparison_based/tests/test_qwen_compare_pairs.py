from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import qwen_compare_pairs as qwen  # noqa: E402


class FakeResponse:
    ok = True
    status_code = 200
    text = ""

    def __init__(self, content: str) -> None:
        self.content = content

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self.content}}]}


class QwenComparePairsTest(unittest.TestCase):
    def test_discovers_only_complete_pairs_in_natural_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("10_1.jpg", "2_2.jpg", "2_1.jpg", "10_2.jpg", "3_1.jpg"):
                (root / name).write_bytes(b"test")

            pairs = qwen.discover_pairs(root)

        self.assertEqual([pair[0] for pair in pairs], ["2", "10"])

    def test_parses_fenced_json_and_clamps_bboxes(self) -> None:
        content = """```json
        {
          "has_difference": true,
          "summary": "商品移动",
          "changes": [{
            "type": "moved",
            "object": "蓝色商品",
            "before_bbox": [-5, 20, 100, 200],
            "after_bbox": [1200, 680, 200, 100],
            "confidence": 1.2,
            "description": "从左侧移动到右侧"
          }]
        }
        ```"""

        result = qwen.parse_qwen_result(content)

        self.assertTrue(result["has_difference"])
        self.assertEqual(result["changes"][0]["before_bbox"], [0, 20, 100, 200])
        self.assertEqual(result["changes"][0]["after_bbox"], [1200, 680, 80, 40])
        self.assertEqual(result["changes"][0]["confidence"], 1.0)

    def test_request_sends_two_720p_images(self) -> None:
        content = '{"has_difference":false,"summary":"无变化","changes":[]}'
        image = np.zeros((720, 1280, 3), dtype=np.uint8)

        with patch("qwen_compare_pairs.requests.post", return_value=FakeResponse(content)) as post:
            returned = qwen.request_qwen(
                image,
                image,
                url="http://qwen.test/v1",
                model="Qwen3-VL-test",
                timeout=10,
            )

        self.assertEqual(returned, content)
        request = post.call_args
        self.assertEqual(request.args[0], "http://qwen.test/v1/chat/completions")
        payload = request.kwargs["json"]
        blocks = payload["messages"][1]["content"]
        self.assertEqual(sum(block["type"] == "image_url" for block in blocks), 2)
        encoded_url = next(
            block["image_url"]["url"]
            for block in blocks
            if block["type"] == "image_url"
        )
        self.assertTrue(encoded_url.startswith("data:image/jpeg;base64,"))

    def test_saves_json_and_annotated_comparison(self) -> None:
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = {
            "has_difference": True,
            "summary": "商品移动",
            "changes": [
                {
                    "type": "moved",
                    "object": "商品",
                    "before_bbox": [10, 20, 100, 120],
                    "after_bbox": [300, 200, 100, 120],
                    "confidence": 0.9,
                    "description": "位置变化",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = qwen.save_pair_results(
                directory,
                "1",
                image,
                image,
                json.dumps(result, ensure_ascii=False),
                result,
                model="test",
                before_path=Path("1_1.jpg"),
                after_path=Path("1_2.jpg"),
            )
            metadata = json.loads(paths["result"].read_text(encoding="utf-8"))
            comparison_bytes = np.fromfile(paths["comparison"], dtype=np.uint8)
            comparison = cv2.imdecode(comparison_bytes, cv2.IMREAD_COLOR)

        self.assertEqual(metadata["image_size"], [1280, 720])
        self.assertEqual(comparison.shape[:2], (720, 2560))


if __name__ == "__main__":
    unittest.main()

