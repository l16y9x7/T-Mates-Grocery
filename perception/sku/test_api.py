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

    def test_old_png_catalog_path_uses_new_jpg_image(self) -> None:
        product = self.client.get("/sku/search_by_SKU", params={"sku": "SKU_057"})
        self.assertEqual(product.status_code, 200)
        self.assertEqual(product.json()["images"], ["images/SKU_057.jpg"])

        response = self.client.get("/images/SKU_057.png")
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

    def test_get_all_names(self) -> None:
        response = self.client.get("/sku/get_all_names")
        self.assertEqual(response.status_code, 200)
        names = response.json()
        self.assertEqual(len(names), 107)
        self.assertEqual(names[0], "NFC桔汁")
        self.assertEqual(names[-1], "心相印厨房纸巾")

    def test_get_candidate_sku_for_same_row(self) -> None:
        response = self.client.request(
            "GET",
            "/sku/get_candidate_SKU",
            json={"location_id": "h2_f_l4_c05", "pose_type": ""},
        )

        self.assertEqual(response.status_code, 200)
        rows = response.json()
        self.assertEqual(len(rows), 1)
        self.assertGreater(len(rows[0]), 0)
        names = [product["name"] for product in rows[0]]
        self.assertEqual(len(names), len(set(names)))
        columns = [
            min(
                int(location.rsplit("C", 1)[1])
                for location in product["locations"]
                if location.startswith("H2_F_L4_")
            )
            for product in rows[0]
        ]
        self.assertEqual(columns, sorted(columns))

    def test_get_candidate_sku_for_upper_two_rows(self) -> None:
        response = self.client.request(
            "GET",
            "/sku/get_candidate_SKU",
            json={
                "location_id": "H2_F_L4_C05",
                "pose_type": "SHELF_VIEW_UPPER",
            },
        )

        self.assertEqual(response.status_code, 200)
        rows = response.json()
        self.assertEqual(len(rows), 2)
        for level, row in enumerate(rows, start=1):
            self.assertGreater(len(row), 0)
            self.assertTrue(
                all(
                    any(
                        location.startswith(f"H2_F_L{level}_")
                        for location in product["locations"]
                    )
                    for product in row
                )
            )

    def test_get_candidate_sku_for_lower_three_rows(self) -> None:
        response = self.client.request(
            "GET",
            "/sku/get_candidate_SKU",
            json={
                "location_id": "H2_F_L4_C05",
                "pose_type": "SHELF_VIEW_LOWER",
            },
        )

        self.assertEqual(response.status_code, 200)
        rows = response.json()
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(rows))

    def test_get_candidate_sku_rejects_invalid_location(self) -> None:
        response = self.client.request(
            "GET",
            "/sku/get_candidate_SKU",
            json={"location_id": "H3_F_L4_C05", "pose_type": ""},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error_code": "INVALID_LOCATION_ID"})

    def test_get_candidate_sku_rejects_invalid_pose_type(self) -> None:
        response = self.client.request(
            "GET",
            "/sku/get_candidate_SKU",
            json={"location_id": "H2_F_L4_C05", "pose_type": "UNKNOWN"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error_code": "INVALID_REQUEST"})

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
        self.assertIn("/sku/get_all_names", paths)
        self.assertIn("/sku/get_candidate_SKU", paths)


if __name__ == "__main__":
    unittest.main()
