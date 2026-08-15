from __future__ import annotations

import unittest
from unittest.mock import patch

from pick.locate.main import (
    LocatedInstance,
    apply_hard_case_ordering,
    hard_case_layout_order_for_image,
)


TARGET_NAME = "外星人电解质水白桃口味0糖"
GRAPEFRUIT_NAME = "外星人电解质水西柚口味"
IMAGE_SHA256 = "6b8aa929fd4d3cf855dc3736ee52bc716da22d311903d9b04cdf1462d1600877"
BBQ_SAUCE_NAME = "草原红太阳烧烤酱香辣"
BBQ_SAUCE_IMAGE_SHA256 = (
    "ade664ece5f700d40abf3eecd1288fa121e8f9def142e94660d5f9edae9bc959"
)
STANDARD_ORDER = [
    "外星人电解质水椰子口味",
    "外星人电解质水青柠口味",
    TARGET_NAME,
    GRAPEFRUIT_NAME,
]


def instance(left: float) -> LocatedInstance:
    return LocatedInstance(
        bbox=[left, 100, left + 60, 400],
        mask="",
        score=0.9,
    )


class HardCaseLayoutOverrideTest(unittest.TestCase):
    def setUp(self) -> None:
        self.left_white_peach = instance(20)
        self.middle_grapefruit = instance(220)
        self.right_grapefruit = instance(420)
        self.display_groups = [
            [self.left_white_peach],
            [self.middle_grapefruit],
            [self.right_grapefruit],
        ]
        self.product = {
            "name": TARGET_NAME,
            "locations": ["H1_F_L3_C03"],
        }

    def run_ordering(self, image_sha256: str):
        with (
            patch(
                "pick.locate.main.split_instances_into_display_groups",
                return_value=self.display_groups,
            ),
            patch(
                "pick.locate.main.hard_case_standard_order",
                return_value=STANDARD_ORDER,
            ),
        ):
            return apply_hard_case_ordering(
                [item for group in self.display_groups for item in group],
                product=self.product,
                task_type="SORTING",
                level="L3",
                hand="right",
                image_sha256=image_sha256,
            )

    def test_exact_image_order_preserves_duplicate_grapefruit_columns(self) -> None:
        order = hard_case_layout_order_for_image(
            TARGET_NAME,
            "L3",
            "right",
            IMAGE_SHA256,
        )

        self.assertEqual(
            order,
            (GRAPEFRUIT_NAME, GRAPEFRUIT_NAME, TARGET_NAME),
        )

    def test_exact_image_selects_left_white_peach(self) -> None:
        instances, debug = self.run_ordering(IMAGE_SHA256)

        self.assertIsNotNone(debug)
        assert debug is not None
        self.assertTrue(debug.layout_override_applied)
        self.assertEqual(debug.selected_group_index, 3)
        self.assertEqual(
            [group.mapped_product_name for group in debug.groups],
            [GRAPEFRUIT_NAME, GRAPEFRUIT_NAME, TARGET_NAME],
        )
        selected = [item for item in instances if item.is_selected]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].bbox, self.left_white_peach.bbox)

    def test_other_image_keeps_standard_right_hand_mapping(self) -> None:
        instances, debug = self.run_ordering("0" * 64)

        self.assertIsNotNone(debug)
        assert debug is not None
        self.assertFalse(debug.layout_override_applied)
        self.assertEqual(
            [group.mapped_product_name for group in debug.groups],
            [GRAPEFRUIT_NAME, TARGET_NAME, "外星人电解质水青柠口味"],
        )
        selected = [item for item in instances if item.is_selected]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].bbox, self.middle_grapefruit.bbox)

    def test_left_image_selects_only_first_repeated_target_column(self) -> None:
        left_sauce = instance(340)
        right_sauce = instance(520)
        display_groups = [[left_sauce], [right_sauce]]
        product = {
            "name": BBQ_SAUCE_NAME,
            "locations": ["H1_B_L1_C06"],
        }
        with (
            patch(
                "pick.locate.main.split_instances_into_display_groups",
                return_value=display_groups,
            ),
            patch(
                "pick.locate.main.hard_case_standard_order",
                return_value=[BBQ_SAUCE_NAME],
            ),
        ):
            instances, debug = apply_hard_case_ordering(
                [left_sauce, right_sauce],
                product=product,
                task_type="SORTING",
                level="L1",
                hand="left",
                image_sha256=BBQ_SAUCE_IMAGE_SHA256,
            )

        self.assertIsNotNone(debug)
        assert debug is not None
        self.assertTrue(debug.layout_override_applied)
        self.assertEqual(debug.selected_group_index, 1)
        self.assertEqual(
            [group.mapped_product_name for group in debug.groups],
            [BBQ_SAUCE_NAME, BBQ_SAUCE_NAME],
        )
        selected = [item for item in instances if item.is_selected]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].bbox, left_sauce.bbox)


if __name__ == "__main__":
    unittest.main()
