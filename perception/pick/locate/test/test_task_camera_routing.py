from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from pick.locate import main as locate_main


class TaskCameraRoutingTest(unittest.TestCase):
    def test_shortage_and_misplaced_use_head_camera(self) -> None:
        self.assertEqual(locate_main.camera_for_task("SHORTAGE", "left"), "head")
        self.assertEqual(locate_main.camera_for_task("SHORTAGE", "right"), "head")
        self.assertEqual(locate_main.camera_for_task("MISPLACED", "left"), "head")
        self.assertEqual(locate_main.camera_for_task("MISPLACED", "right"), "head")

    def test_sorting_keeps_selected_wrist_camera(self) -> None:
        self.assertEqual(locate_main.camera_for_task("SORTING", "left"), "left")
        self.assertEqual(locate_main.camera_for_task("SORTING", "right"), "right")

    def test_live_shortage_uses_head_rgb_without_depth(self) -> None:
        request = locate_main.LocateRequest(
            task_type="SHORTAGE",
            product_name="test product",
            level="L1",
            hand="right",
        )
        expected_response = object()
        with (
            patch.object(
                locate_main,
                "lookup_sku_by_name",
                return_value={"sku_id": "SKU_TEST", "name": "test product"},
            ),
            patch.object(
                locate_main,
                "get_latest_rgb",
                return_value=Path("head.jpg"),
            ) as get_latest_rgb,
            patch.object(
                locate_main,
                "locate_product_in_image",
                return_value=expected_response,
            ) as locate_product_in_image,
        ):
            response = locate_main.locate_product_debug(request)

        self.assertIs(response, expected_response)
        get_latest_rgb.assert_called_once_with("head")
        self.assertIsNone(
            locate_product_in_image.call_args.kwargs["depth_image_provider"]
        )

    def test_live_misplaced_uses_matching_head_rgb_and_depth(self) -> None:
        request = locate_main.LocateRequest(
            task_type="MISPLACED",
            product_name="test product",
            level="L1",
            hand="left",
        )
        expected_response = object()
        with (
            patch.object(
                locate_main,
                "lookup_sku_by_name",
                return_value={"sku_id": "SKU_TEST", "name": "test product"},
            ),
            patch.object(
                locate_main,
                "get_latest_rgb",
                return_value=Path("head.jpg"),
            ) as get_latest_rgb,
            patch.object(
                locate_main,
                "fetch_camera_depth",
                return_value=None,
            ) as fetch_camera_depth,
            patch.object(
                locate_main,
                "locate_product_in_image",
                return_value=expected_response,
            ) as locate_product_in_image,
        ):
            response = locate_main.locate_product_debug(request)
            depth_provider = locate_product_in_image.call_args.kwargs[
                "depth_image_provider"
            ]
            self.assertIsNotNone(depth_provider)
            depth_provider((640, 480))

        self.assertIs(response, expected_response)
        get_latest_rgb.assert_called_once_with("head")
        fetch_camera_depth.assert_called_once_with("head", (640, 480))


if __name__ == "__main__":
    unittest.main()
