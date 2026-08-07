"""Tests for the compact receipt parsing service."""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from PIL import Image

import server


def jpeg_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(output, format="JPEG")
    return output.getvalue()


def settings() -> server.Settings:
    return server.Settings(
        camera_url="http://camera.test/snapshot",
        qwen_base_url="http://qwen.test/v1",
        sku_base_url="http://sku.test",
    )


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, *_: object) -> bytes:
        return self.body


class ReceiptServerTests(unittest.TestCase):
    def setUp(self) -> None:
        server._SKU_NAMES_CACHE.clear()

    def test_capture_one_frame_gets_camera_without_writing_file(self) -> None:
        image = jpeg_bytes()
        with patch("server.urlopen", return_value=FakeResponse(image)) as mocked:
            self.assertEqual(server.capture_one_frame(settings()), image)
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "http://camera.test/snapshot")
        self.assertEqual(request.method, "GET")

    def test_receipt_endpoint_has_no_upload_request_body(self) -> None:
        operation = server.app.openapi()["paths"]["/receipt/parse"]["post"]
        self.assertNotIn("requestBody", operation)

    def test_recognize_one_frame_sends_one_image_url(self) -> None:
        qwen = {
            "choices": [
                {
                    "message": {
                        "content": '[{"name":"NFC桔汁","specification":"500ml"}]'
                    }
                }
            ]
        }
        with patch("server._request_qwen", return_value=qwen) as mocked:
            items = server.recognize_frames([jpeg_bytes()], settings())
        content = mocked.call_args.args[0]["messages"][1]["content"]
        self.assertEqual(sum(block["type"] == "image_url" for block in content), 1)
        self.assertEqual(mocked.call_args.args[0]["temperature"], 0)
        self.assertEqual(
            items, [{"name": "NFC桔汁", "specification": "500ml"}]
        )

    def test_recognize_three_frames_keeps_multi_image_path(self) -> None:
        qwen = {"choices": [{"message": {"content": "[]"}}]}
        with patch("server._request_qwen", return_value=qwen) as mocked:
            server.recognize_frames([jpeg_bytes()] * 3, settings())
        content = mocked.call_args.args[0]["messages"][1]["content"]
        self.assertEqual(sum(block["type"] == "image_url" for block in content), 3)

    def test_qwen_schema_allows_only_name_and_specification(self) -> None:
        self.assertEqual(
            server.parse_qwen_items(
                '[{"name":"呀！土豆番茄酱味","specification":null}]'
            ),
            [{"name": "呀！土豆番茄酱味", "specification": None}],
        )
        with self.assertRaises(server.ServiceError):
            server.parse_qwen_items(
                '[{"name":"呀！土豆","specification":null,"count":1}]'
            )

    def test_empty_qwen_array_is_valid(self) -> None:
        self.assertEqual(server.parse_qwen_items("[]"), [])

    def test_invalid_qwen_json_fails(self) -> None:
        with self.assertRaisesRegex(server.ServiceError, "严格 JSON"):
            server.parse_qwen_items("```json\n[]\n```")

    def test_parse_receipt_returns_only_sku_name_and_locations(self) -> None:
        recognized = [{"name": "NFC桔汁", "specification": "500ml"}]
        sku_result = [{"name": "NFC桔汁", "locations": ["H1_F_L1_C01"]}]
        with (
            patch("server.Settings.from_env", return_value=settings()),
            patch("server.capture_one_frame", return_value=jpeg_bytes()),
            patch("server.recognize_frames", return_value=recognized),
            patch("server.lookup_sku_items", return_value=sku_result),
        ):
            self.assertEqual(server.parse_receipt(), sku_result)

    def test_parse_receipt_returns_empty_array_without_sku_call(self) -> None:
        with (
            patch("server.Settings.from_env", return_value=settings()),
            patch("server.capture_one_frame", return_value=jpeg_bytes()),
            patch("server.recognize_frames", return_value=[]),
            patch("server.lookup_sku_items") as lookup,
        ):
            self.assertEqual(server.parse_receipt(), [])
            lookup.assert_not_called()

    def test_sku_exact_match_discards_specification(self) -> None:
        product = {"name": "NFC桔汁", "locations": ["H1_F_L1_C01"]}
        with patch("server._sku_product_for_name", return_value=product):
            result = server.lookup_sku_items(
                [{"name": "NFC桔汁", "specification": "500ml"}], settings()
            )
        self.assertEqual(result, [product])
        self.assertNotIn("specification", result[0])

    def test_sku_edit_distance_fallback_returns_nearest_product(self) -> None:
        expected = {
            "name": "草原红太阳烧烤酱原味",
            "locations": ["H1_B_L1_C07"],
        }
        with (
            patch(
                "server._sku_product_for_name",
                side_effect=[server.SKUNotFoundError("基原红太阳烧烤酱原味"), expected],
            ),
            patch(
                "server._all_sku_names",
                return_value=["草原红太阳烧烤料原味", "草原红太阳烧烤酱原味"],
            ),
        ):
            result = server.lookup_sku_items(
                [{"name": "基原红太阳烧烤酱原味", "specification": None}],
                settings(),
            )
        self.assertEqual(result, [expected])

    def test_camera_connection_failure_is_diagnostic(self) -> None:
        with patch("server.urlopen", side_effect=URLError("offline")):
            with self.assertRaisesRegex(server.ServiceError, "无法连接相机接口"):
                server.capture_one_frame(settings())

    def test_non_image_camera_response_is_rejected_in_memory(self) -> None:
        with self.assertRaisesRegex(server.ServiceError, "不是有效 JPEG/PNG"):
            server.image_bytes_to_data_url(b"not an image")


if __name__ == "__main__":
    unittest.main()
