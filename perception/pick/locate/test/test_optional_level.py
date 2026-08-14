from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from pick.locate import main as locate_main


class OptionalLevelTest(unittest.TestCase):
    def test_request_model_allows_omitting_level(self) -> None:
        request = locate_main.LocateRequest(
            task_type="SHORTAGE",
            product_name="test product",
            hand="left",
        )

        self.assertIsNone(request.level)

    def test_normal_sorting_passes_none_level_to_pipeline(self) -> None:
        request = locate_main.LocateRequest(
            task_type="SORTING",
            product_name="test product",
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
                return_value=Path("right.jpg"),
            ),
            patch.object(
                locate_main,
                "locate_product_in_image",
                return_value=expected_response,
            ) as locate_product_in_image,
        ):
            response = locate_main.locate_product_debug(request)

        self.assertIs(response, expected_response)
        self.assertIsNone(locate_product_in_image.call_args.kwargs["level"])

    def test_configured_sorting_hard_case_requires_level(self) -> None:
        request = locate_main.LocateRequest(
            task_type="SORTING",
            product_name="脉动观梅止渴饮",
            hand="left",
        )
        with patch.object(
            locate_main,
            "lookup_sku_by_name",
            return_value={"sku_id": "SKU_TEST", "name": "脉动观梅止渴饮"},
        ):
            with self.assertRaises(HTTPException) as context:
                locate_main.locate_product_debug(request)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("必须提供 level", str(context.exception.detail))

    def test_shortage_never_requires_hard_case_level(self) -> None:
        self.assertFalse(
            locate_main.hard_case_level_required(
                "脉动观梅止渴饮",
                "SHORTAGE",
                "left",
            )
        )
        self.assertIsNone(
            locate_main.hard_case_group_for_product(
                "脉动观梅止渴饮",
                "SHORTAGE",
                None,
                "left",
            )
        )


if __name__ == "__main__":
    unittest.main()
