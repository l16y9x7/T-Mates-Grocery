import json
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from receipt_recognizer.config import Settings
from receipt_recognizer.errors import SKUConnectionError
from receipt_recognizer.sku_client import SkuLookupClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class SkuLookupClientTests(unittest.TestCase):
    def test_locations_for_name_matches_sku_service(self) -> None:
        with patch(
            "receipt_recognizer.sku_client.urlopen",
            return_value=FakeResponse(
                {"name": "NFC桔汁", "locations": ["H1_F_L1_C01"]}
            ),
        ) as urlopen:
            result = SkuLookupClient(Settings()).locations_for_name("NFC桔汁")

        self.assertEqual(
            result.to_dict(),
            {
                "name": "NFC桔汁",
                "matched": True,
                "locations": ["H1_F_L1_C01"],
            },
        )
        request = urlopen.call_args.args[0]
        self.assertIsInstance(request, Request)
        self.assertIn("/sku/locations?name=NFC", request.full_url)

    def test_locations_for_name_keeps_404_as_unmatched_item(self) -> None:
        error = HTTPError(
            url="http://127.0.0.1:8080/sku/locations",
            code=404,
            msg="not found",
            hdrs={},
            fp=BytesIO(b'{"error_code": "SKU_NOT_FOUND"}'),
        )
        with patch(
            "receipt_recognizer.sku_client.urlopen",
            side_effect=error,
        ):
            result = SkuLookupClient(Settings()).locations_for_name("不存在")

        self.assertEqual(
            result.to_dict(),
            {
                "name": "不存在",
                "matched": False,
                "locations": [],
                "error_code": "SKU_NOT_FOUND",
            },
        )

    def test_connection_error_is_raised_for_unreachable_sku_service(self) -> None:
        with patch(
            "receipt_recognizer.sku_client.urlopen",
            side_effect=URLError("refused"),
        ):
            with self.assertRaises(SKUConnectionError):
                SkuLookupClient(Settings()).locations_for_name("NFC桔汁")


if __name__ == "__main__":
    unittest.main()
