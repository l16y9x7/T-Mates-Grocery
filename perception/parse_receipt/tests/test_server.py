import asyncio
import json
import unittest
from io import BytesIO
from unittest.mock import patch

from receipt_recognizer import server
from receipt_recognizer.errors import (
    APIResponseError,
    InputFileError,
    SKUNotFoundError,
)
from receipt_recognizer.service import Recognition


class FakeRecognizer:
    def __init__(self, settings):
        self.settings = settings

    def recognize_data_urls(self, data_urls):
        self.data_urls = data_urls
        return Recognition(
            business_items=[
                {
                    "name": "好丽友土豆薯条番茄味",
                    "specification": "70g",
                }
            ],
            diagnostics={
                "receipt_status": "ok",
                "line_items": [],
                "review_items": [],
            },
            finish_reason="stop",
            usage={"total_tokens": 123},
            corrected_once=False,
            page_count=1,
        )


class FakeSkuClient:
    def __init__(self, settings):
        self.settings = settings

    def lookup_items(self, items):
        return [
            {
                "name": items[0]["name"],
                "locations": ["H1_F_L1_C01"],
            }
        ]


class MissingSkuClient:
    def __init__(self, settings):
        self.settings = settings

    def lookup_items(self, items):
        raise SKUNotFoundError(items[0]["name"])


class ServerTests(unittest.TestCase):
    def test_receipt_parse_returns_sku_locations(self):
        with patch.object(
            server,
            "image_bytes_to_data_url",
            return_value="data:image/jpeg;base64,abc",
        ) as convert, patch.object(
            server,
            "ReceiptRecognizer",
            FakeRecognizer,
        ), patch.object(
            server,
            "SkuLookupClient",
            FakeSkuClient,
        ):
            response = asyncio.run(
                server.parse_receipt(
                    file=_upload_file(b"fake image"),
                    diagnostics=False,
                    max_edge=2200,
                )
            )

        self.assertEqual(
            response,
            [
                {
                    "name": "好丽友土豆薯条番茄味",
                    "locations": ["H1_F_L1_C01"],
                }
            ],
        )
        convert.assert_called_once()

    def test_receipt_parse_can_return_diagnostics(self):
        with patch.object(
            server,
            "image_bytes_to_data_url",
            return_value="data:image/jpeg;base64,abc",
        ), patch.object(
            server,
            "ReceiptRecognizer",
            FakeRecognizer,
        ), patch.object(
            server,
            "SkuLookupClient",
            FakeSkuClient,
        ):
            response = asyncio.run(
                server.parse_receipt(
                    file=_upload_file(b"fake image"),
                    diagnostics=True,
                    max_edge=2200,
                )
            )

        self.assertEqual(
            response["items"],
            [
                {
                    "name": "好丽友土豆薯条番茄味",
                    "locations": ["H1_F_L1_C01"],
                }
            ],
        )
        self.assertEqual(
            response["diagnostics"]["receipt_status"],
            "ok",
        )

    def test_receipt_parse_propagates_sku_not_found_as_404(self):
        with patch.object(
            server,
            "image_bytes_to_data_url",
            return_value="data:image/jpeg;base64,abc",
        ), patch.object(
            server,
            "ReceiptRecognizer",
            FakeRecognizer,
        ), patch.object(
            server,
            "SkuLookupClient",
            MissingSkuClient,
        ):
            response = asyncio.run(
                server.parse_receipt(
                    file=_upload_file(b"fake image"),
                    diagnostics=False,
                    max_edge=2200,
                )
            )

        body = json.loads(response.body)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(body, {"error_code": "SKU_NOT_FOUND"})

    def test_receipt_parse_returns_structured_input_error(self):
        with patch.object(
            server,
            "image_bytes_to_data_url",
            side_effect=InputFileError("文件内容不是有效的 JPEG/PNG 图片。"),
        ):
            response = asyncio.run(
                server.parse_receipt(
                    file=_upload_file(b"fake image"),
                    diagnostics=False,
                    max_edge=2200,
                )
            )

        body = json.loads(response.body)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(body["error"]["type"], "invalid_input")
        self.assertIn("JPEG/PNG", body["error"]["message"])

    def test_receipt_parse_includes_upstream_status_code(self):
        with patch.object(
            server,
            "image_bytes_to_data_url",
            return_value="data:image/jpeg;base64,abc",
        ), patch.object(
            server,
            "ReceiptRecognizer",
            side_effect=APIResponseError(
                "模型 API 返回 HTTP 502: 响应体为空",
                status_code=502,
            ),
        ):
            response = asyncio.run(
                server.parse_receipt(
                    file=_upload_file(b"fake image"),
                    diagnostics=False,
                    max_edge=2200,
                )
            )

        body = json.loads(response.body)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(body["error"]["type"], "upstream_response_error")
        self.assertEqual(body["error"]["upstream_status_code"], 502)

    def test_openapi_exposes_parse_not_legacy_recognize(self):
        paths = server.app.openapi()["paths"]
        self.assertIn("/receipt/parse", paths)
        self.assertNotIn("/receipt/recognize", paths)

def _upload_file(content):
    return server.UploadFile(
        filename="receipt.jpg",
        file=BytesIO(content),
    )


if __name__ == "__main__":
    unittest.main()
