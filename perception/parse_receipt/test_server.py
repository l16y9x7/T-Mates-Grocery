"""Tests for the compact receipt parsing service."""

from __future__ import annotations

import asyncio
import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import URLError

from PIL import Image
from starlette.requests import Request as StarletteRequest

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
    def test_capture_one_frame_gets_camera_without_writing_file(self) -> None:
        image = jpeg_bytes()
        with patch("server.urlopen", return_value=FakeResponse(image)) as mocked:
            self.assertEqual(server.capture_one_frame(settings()), image)
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "http://camera.test/snapshot")
        self.assertEqual(request.method, "GET")

    def test_receipt_endpoint_has_no_upload_request_body(self) -> None:
        operation = server.app.openapi()["paths"]["/perception/parse"]["post"]
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
            items = server.recognize_frame(jpeg_bytes(), settings())
        content = mocked.call_args.args[0]["messages"][1]["content"]
        self.assertEqual(sum(block["type"] == "image_url" for block in content), 1)
        self.assertEqual(mocked.call_args.args[0]["temperature"], 0)
        self.assertEqual(
            items, [{"name": "NFC桔汁", "specification": "500ml"}]
        )

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
        self.assertEqual(
            server.parse_qwen_items(
                '[{"name":"NFC桔汁","specification":null}, []]'
            ),
            [{"name": "NFC桔汁", "specification": None}],
        )

    def test_invalid_qwen_json_fails(self) -> None:
        with self.assertRaisesRegex(server.ServiceError, "严格 JSON"):
            server.parse_qwen_items("```json\n[]\n```")

    def test_parse_receipt_returns_two_names_in_object(self) -> None:
        recognized = [
            {"name": "NFC桔汁", "specification": "500ml"},
            {"name": "蒙牛纯牛奶", "specification": "250ml"},
        ]
        sku_result = ["NFC桔汁", "蒙牛纯牛奶"]
        with (
            patch("server.Settings.from_env", return_value=settings()),
            patch("server.capture_one_frame", return_value=jpeg_bytes()),
            patch("server.recognize_frame", return_value=recognized),
            patch("server.lookup_sku_items", return_value=sku_result),
        ):
            self.assertEqual(server.parse_receipt().product_names, sku_result)

    def test_parse_receipt_rejects_result_without_two_items(self) -> None:
        with (
            patch("server.Settings.from_env", return_value=settings()),
            patch("server.capture_one_frame", return_value=jpeg_bytes()),
            patch("server.recognize_frame", return_value=[]),
            patch("server.lookup_sku_items") as lookup,
        ):
            with self.assertRaisesRegex(server.ServiceError, "必须识别出两个商品"):
                server.parse_receipt()
            lookup.assert_not_called()

    def test_sku_exact_match_discards_specification(self) -> None:
        with patch("server._all_sku_names", return_value=["NFC桔汁"]):
            result = server.lookup_sku_items(
                [{"name": "NFC桔汁", "specification": "500ml"}], settings()
            )
        self.assertEqual(result, ["NFC桔汁"])

    def test_non_numeric_specification_is_appended_for_exact_match(self) -> None:
        expected = "外星人电解质水白桃口味"
        with (
            patch(
                "server._all_sku_names",
                return_value=["外星人电解质水青柠口味", "外星人电解质水白桃口味"],
            ),
        ):
            result = server.lookup_sku_items(
                [{"name": "外星人电解质水", "specification": "白桃口味"}],
                settings(),
            )
        self.assertEqual(result, [expected])

    def test_exact_name_has_priority_over_non_numeric_specification(self) -> None:
        self.assertEqual(
            server.match_sku_name(
                "NFC桔汁",
                "白桃口味",
                ["NFC桔汁", "NFC桔汁白桃口味"],
            ),
            "NFC桔汁",
        )

    def test_numeric_unit_specification_is_excluded_from_matching(self) -> None:
        self.assertEqual(server.specification_for_matching("500ml"), "")
        self.assertEqual(server.specification_for_matching("55 g"), "")
        self.assertEqual(server.specification_for_matching("2盒"), "")
        self.assertEqual(
            server.specification_for_matching("白桃口味"),
            "白桃口味",
        )

    def test_sku_edit_distance_fallback_returns_nearest_product(self) -> None:
        expected = "草原红太阳烧烤酱原味"
        with (
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

    def test_default_camera_is_head_snapshot(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            configured = server.Settings.from_env()
        self.assertEqual(
            configured.camera_url,
            "http://192.168.130.50:8085/camera/snapshot?camera=head&type=color",
        )

    def test_camera_connection_failure_is_diagnostic(self) -> None:
        with patch("server.urlopen", side_effect=URLError("offline")):
            with self.assertRaisesRegex(
                server.ServiceError,
                "无法连接相机接口",
            ) as raised:
                server.capture_one_frame(settings())
        error = raised.exception
        self.assertEqual(error.stage, "camera_capture")
        self.assertEqual(error.upstream, "http://camera.test/snapshot")
        self.assertTrue(error.retryable)
        self.assertIsNotNone(error.elapsed_ms)
        self.assertEqual(error.timeout_seconds, 5.0)

    def test_service_error_response_and_log_contain_diagnostics(self) -> None:
        request = StarletteRequest(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/perception/parse",
                "raw_path": b"/perception/parse",
                "query_string": b"",
                "headers": [(b"x-request-id", b"receipt-test-123")],
                "client": ("192.168.130.59", 45800),
                "server": ("127.0.0.1", 8083),
            }
        )
        error = server.ServiceError(
            502,
            "camera_connection_error",
            "无法连接相机接口：connection refused",
            upstream="http://192.168.130.50:8085/camera/snapshot",
            elapsed_ms=3012.4,
            timeout_seconds=5.0,
        )

        with self.assertLogs("uvicorn.error", level="ERROR") as logs:
            response = asyncio.run(server.handle_service_error(request, error))

        body = json.loads(response.body)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.headers["x-request-id"], "receipt-test-123")
        self.assertEqual(body["error"]["stage"], "camera_capture")
        self.assertEqual(body["error"]["request_id"], "receipt-test-123")
        self.assertEqual(body["error"]["elapsed_ms"], 3012.4)
        self.assertEqual(body["error"]["timeout_seconds"], 5.0)
        self.assertTrue(body["error"]["retryable"])
        self.assertIn("request_id=receipt-test-123", logs.output[0])
        self.assertIn("stage=camera_capture", logs.output[0])

    def test_upstream_url_redacts_credentials_and_query(self) -> None:
        self.assertEqual(
            server._safe_upstream_url(
                "http://user:secret@camera.test:8085/snapshot?token=secret"
            ),
            "http://camera.test:8085/snapshot",
        )

    def test_non_image_camera_response_is_rejected_in_memory(self) -> None:
        with self.assertRaisesRegex(server.ServiceError, "不是有效 JPEG/PNG"):
            server.image_bytes_to_data_url(b"not an image")


if __name__ == "__main__":
    unittest.main()
