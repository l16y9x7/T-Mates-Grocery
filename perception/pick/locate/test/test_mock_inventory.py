from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pick.locate import main as locate_main


class MockInventoryTest(unittest.TestCase):
    def test_local_mock_inventory_overrides_catalog_inventory(self) -> None:
        catalog = {
            "products": [
                {
                    "sku_id": "SKU_TEST",
                    "name": "测试商品",
                    "locations": [
                        "H1_L01_C01",
                        "H1_L01_C02",
                        "H1_L01_C03",
                    ],
                    "inventory": ["H1_L01_C01"],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "products.json"
            catalog_path.write_text(
                json.dumps(catalog, ensure_ascii=False),
                encoding="utf-8",
            )
            with patch.object(locate_main, "MOCK_SKU_CATALOG_PATH", catalog_path):
                product = locate_main.lookup_mock_sku_by_name(
                    "测试商品",
                    ["H1_L01_C02", "H1_L01_C03"],
                )

        self.assertEqual(
            product["inventory"],
            ["H1_L01_C02", "H1_L01_C03"],
        )
        self.assertEqual(catalog["products"][0]["inventory"], ["H1_L01_C01"])

    def test_c02_c03_mock_inventory_can_select_c03(self) -> None:
        product = {
            "sku_id": "SKU_TEST",
            "name": "测试商品",
            "locations": ["H1_L01_C01", "H1_L01_C02", "H1_L01_C03"],
            "inventory": ["H1_L01_C02", "H1_L01_C03"],
        }
        inventory = locate_main.inventory_slots_for_pick(
            product,
            "SORTING",
            "H1_L01_C03",
            "L1",
        )
        instances = [
            locate_main.LocatedInstance(bbox=[10, 10, 20, 20], mask="mask-1"),
            locate_main.LocatedInstance(bbox=[30, 10, 40, 20], mask="mask-2"),
        ]

        mapped, selected = locate_main.map_inventory_slots_to_instances(
            instances,
            inventory,
            "H1_L01_C03",
            100,
        )

        self.assertEqual([item.mapped_slot_id for item in mapped], inventory)
        self.assertEqual(selected.mapped_slot_id, "H1_L01_C03")
        self.assertEqual(selected.bbox, [30, 10, 40, 20])

    def test_partial_left_view_maps_to_inventory_suffix(self) -> None:
        inventory = ["H1_L01_C01", "H1_L01_C02", "H1_L01_C03"]
        instances = [
            locate_main.LocatedInstance(bbox=[10, 10, 20, 20], mask="mask-1"),
            locate_main.LocatedInstance(bbox=[30, 10, 40, 20], mask="mask-2"),
        ]

        mapped, selected = locate_main.map_inventory_slots_to_instances(
            instances,
            inventory,
            "H1_L01_C03",
            100,
        )

        self.assertEqual(
            [item.mapped_slot_id for item in mapped],
            ["H1_L01_C02", "H1_L01_C03"],
        )
        self.assertEqual(selected.mapped_slot_id, "H1_L01_C03")
        self.assertEqual(selected.bbox, [30, 10, 40, 20])

    def test_partial_right_view_maps_to_inventory_prefix(self) -> None:
        inventory = ["H1_L01_C01", "H1_L01_C02", "H1_L01_C03"]
        instances = [
            locate_main.LocatedInstance(bbox=[60, 10, 70, 20], mask="mask-1"),
            locate_main.LocatedInstance(bbox=[80, 10, 90, 20], mask="mask-2"),
        ]

        mapped, selected = locate_main.map_inventory_slots_to_instances(
            instances,
            inventory,
            "H1_L01_C02",
            100,
        )

        self.assertEqual(
            [item.mapped_slot_id for item in mapped],
            ["H1_L01_C01", "H1_L01_C02"],
        )
        self.assertEqual(selected.mapped_slot_id, "H1_L01_C02")
        self.assertEqual(selected.bbox, [80, 10, 90, 20])

    def test_partial_view_rejects_target_outside_visible_suffix(self) -> None:
        inventory = ["H1_L01_C01", "H1_L01_C02", "H1_L01_C03"]
        instances = [
            locate_main.LocatedInstance(bbox=[10, 10, 20, 20], mask="mask-1"),
            locate_main.LocatedInstance(bbox=[30, 10, 40, 20], mask="mask-2"),
        ]

        with self.assertRaises(locate_main.HTTPException) as raised:
            locate_main.map_inventory_slots_to_instances(
                instances,
                inventory,
                "H1_L01_C01",
                100,
            )

        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn("当前可见库存范围", str(raised.exception.detail))

    def test_debug_mock_inventory_does_not_query_live_sku_service(self) -> None:
        request = locate_main.LocateDebugRequest(
            task_type="SORTING",
            product_name="测试商品",
            level="L1",
            hand="left",
            slot_id="H1_L01_C03",
            mock_inventory=["H1_L01_C02", "H1_L01_C03"],
            image_name="rgb.jpg",
            image_base64=base64.b64encode(b"image").decode("ascii"),
        )
        mocked_product = {
            "sku_id": "SKU_TEST",
            "name": "测试商品",
            "locations": ["H1_L01_C01", "H1_L01_C02", "H1_L01_C03"],
            "inventory": ["H1_L01_C02", "H1_L01_C03"],
        }
        expected_response = Mock()
        with (
            patch.object(
                locate_main,
                "lookup_mock_sku_by_name",
                return_value=mocked_product,
            ) as mock_lookup,
            patch.object(locate_main, "lookup_sku_by_name") as live_lookup,
            patch.object(
                locate_main,
                "locate_product_in_image",
                return_value=expected_response,
            ) as locate,
        ):
            response = locate_main.locate_product_debug(
                request,
                mock_inventory=request.mock_inventory,
            )

        self.assertIs(response, expected_response)
        mock_lookup.assert_called_once_with(
            "测试商品",
            ["H1_L01_C02", "H1_L01_C03"],
        )
        live_lookup.assert_not_called()
        self.assertIs(locate.call_args.args[0], mocked_product)
        self.assertIs(
            locate.call_args.kwargs["sku_row_lookup"],
            locate_main.lookup_mock_sku_row,
        )

    def test_mock_inventory_only_appears_on_debug_request_schema(self) -> None:
        self.assertNotIn(
            "mock_inventory",
            locate_main.LocateRequest.model_json_schema()["properties"],
        )
        self.assertIn(
            "mock_inventory",
            locate_main.LocateDebugRequest.model_json_schema()["properties"],
        )


if __name__ == "__main__":
    unittest.main()
