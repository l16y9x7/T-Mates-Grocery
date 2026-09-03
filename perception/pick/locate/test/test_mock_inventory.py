from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from pick.locate import main as locate_main


def encoded_mask(size: tuple[int, int]) -> str:
    buffer = io.BytesIO()
    Image.new("L", size, 255).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encoded_mask_width(width: int, size: tuple[int, int] = (20, 20)) -> str:
    mask = Image.new("L", size, 0)
    mask.paste(255, (0, 0, width, size[1]))
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


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

    def test_inventory_row_prefers_shelf_front_distance_after_tiny_mask_filter(self) -> None:
        instances = [
            locate_main.LocatedInstance(
                bbox=[10, 10, 30, 30],
                mask=encoded_mask_width(20),
                score=0.99,
                display_row_index=1,
                shelf_front_distance_ratio=0.15,
            ),
            locate_main.LocatedInstance(
                bbox=[40, 10, 60, 30],
                mask=encoded_mask_width(19),
                score=0.90,
                display_row_index=1,
                shelf_front_distance_ratio=0.08,
            ),
            locate_main.LocatedInstance(
                bbox=[70, 10, 90, 30],
                mask=encoded_mask_width(18),
                score=0.80,
                display_row_index=1,
                shelf_front_distance_ratio=0.06,
            ),
            locate_main.LocatedInstance(
                bbox=[100, 10, 120, 30],
                mask=encoded_mask_width(2),
                score=0.95,
                display_row_index=1,
                shelf_front_distance_ratio=0.01,
            ),
        ]

        selected = locate_main.keep_display_rows_for_inventory(instances, 2)

        self.assertEqual([instance.bbox[0] for instance in selected], [40, 70])

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

    def test_single_location_skips_inventory_row_branch_but_keeps_slot(self) -> None:
        product = {
            "sku_id": "SKU_SINGLE",
            "name": "单位置测试商品",
            "locations": ["H1_L01_C07"],
            "inventory": ["H1_L01_C07"],
        }
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "rgb.jpg"
            Image.new("RGB", (640, 480), "white").save(image_path)

            def sam_instances(_prompt: str, crop_image: Image.Image):
                return [
                    {
                        "bbox_xyxy": [0.0, 0.0, *map(float, crop_image.size)],
                        "mask_png_base64": encoded_mask(crop_image.size),
                        "score": 0.9,
                    }
                ]

            with (
                patch.object(
                    locate_main,
                    "load_prompt_pair",
                    return_value=("qwen prompt", "sam prompt"),
                ),
                patch.object(
                    locate_main,
                    "get_stable_qwen_bboxes",
                    return_value=locate_main.QwenConsensusBBoxes(
                        [[300, 300, 700, 800]],
                        [],
                    ),
                ),
                patch.object(locate_main, "call_sam3", side_effect=sam_instances),
                patch.object(
                    locate_main,
                    "store_monitor_image",
                    return_value=str(image_path),
                ),
                patch.object(
                    locate_main,
                    "detect_red_shelf_front_line",
                ) as detect_shelf,
            ):
                response = locate_main.locate_product_in_image(
                    product,
                    image_path,
                    task_type="SORTING",
                    level="L1",
                    hand="left",
                    slot_id="H1_L01_C07",
                )

        detect_shelf.assert_not_called()
        self.assertEqual(response.selected_instance.mapped_slot_id, "H1_L01_C07")
        self.assertIsNone(response.selected_instance.display_row_index)


if __name__ == "__main__":
    unittest.main()
