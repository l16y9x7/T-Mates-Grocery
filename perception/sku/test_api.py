from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from api import create_app


class SkuApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(create_app())

    def test_health(self) -> None:
        response = self.client.get("/sku/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "READY"})

    def test_query_locations_by_name(self) -> None:
        response = self.client.get("/sku/locations", params={"name": "NFC桔汁"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"name": "NFC桔汁", "locations": ["H1_F_L1_C01"]},
        )

    def test_query_images_by_name(self) -> None:
        response = self.client.get("/sku/images", params={"name": "NFC桔汁"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"name": "NFC桔汁", "images": []})

    def test_query_name_by_location(self) -> None:
        response = self.client.get(
            "/sku/name", params={"location": "h1_f_l1_c01"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"location": "H1_F_L1_C01", "name": "NFC桔汁"},
        )

    def test_unknown_sku(self) -> None:
        response = self.client.get(
            "/sku/locations", params={"name": "不存在的商品"}
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"error_code": "SKU_NOT_FOUND"})

    def test_missing_query_parameter(self) -> None:
        response = self.client.get("/sku/name")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error_code": "INVALID_REQUEST"})

    def test_openapi_document(self) -> None:
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/sku/locations", response.json()["paths"])


if __name__ == "__main__":
    unittest.main()
