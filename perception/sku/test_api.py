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

    def test_search_by_sku(self) -> None:
        response = self.client.get("/sku/search_by_SKU", params={"sku": "sku_001"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "sku_id": "SKU_001",
                "name": "NFC桔汁",
                "images": ["images/SKU_001.jpg"],
                "locations": ["H1_F_L1_C01"],
            },
        )

    def test_search_by_name(self) -> None:
        response = self.client.get("/sku/search_by_name", params={"name": "NFC桔汁"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "sku_id": "SKU_001",
                "name": "NFC桔汁",
                "images": ["images/SKU_001.jpg"],
                "locations": ["H1_F_L1_C01"],
            },
        )

    def test_get_image(self) -> None:
        response = self.client.get("/images/SKU_001.jpg")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertGreater(len(response.content), 0)

    def test_search_by_location(self) -> None:
        response = self.client.get(
            "/sku/search_by_location", params={"location": "h1_f_l1_c01"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "sku_id": "SKU_001",
                "name": "NFC桔汁",
                "images": ["images/SKU_001.jpg"],
                "locations": ["H1_F_L1_C01"],
            },
        )

    def test_get_image_paths_by_name(self) -> None:
        response = self.client.get("/sku/get_image", params={"name": "NFC桔汁"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), ["images/SKU_001.jpg"])

    def test_unknown_sku(self) -> None:
        response = self.client.get(
            "/sku/search_by_name", params={"name": "不存在的商品"}
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"error_code": "SKU_NOT_FOUND"})

    def test_missing_query_parameter(self) -> None:
        response = self.client.get("/sku/search_by_name")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error_code": "INVALID_REQUEST"})

    def test_openapi_document(self) -> None:
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        paths = response.json()["paths"]
        self.assertIn("/sku/search_by_SKU", paths)
        self.assertIn("/sku/search_by_name", paths)
        self.assertIn("/sku/search_by_location", paths)
        self.assertIn("/sku/get_image", paths)


if __name__ == "__main__":
    unittest.main()
