from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from pick.locate.main import LocatedInstance, apply_hard_case_ordering


COCONUT = "外星人电解质水椰子口味"
LIME = "外星人电解质水青柠口味"
WHITE_PEACH = "外星人电解质水白桃口味0糖"
ROW = [
    {"name": COCONUT, "location_id": "H2_L03_C01"},
    {"name": LIME, "location_id": "H2_L03_C02"},
    {"name": WHITE_PEACH, "location_id": "H2_L03_C03"},
    {"name": "百岁山矿泉水", "location_id": "H2_L03_C04"},
]

HARD_CASE_ROWS = {
    "L3": ROW[:3],
    "L4": [
        {"name": "脉动观梅止渴饮", "location_id": "H2_L04_C01"},
        {"name": "脉动芒果口味", "location_id": "H2_L04_C02"},
        {"name": "脉动菠萝口味", "location_id": "H2_L04_C03"},
    ],
    "L5": [
        {"name": "脉动观梅止渴饮", "location_id": "H2_L05_C01"},
        {"name": "脉动芒果口味", "location_id": "H2_L05_C02"},
    ],
}


def instance(left: float) -> LocatedInstance:
    return LocatedInstance(bbox=[left, 100, left + 60, 400], mask="", score=0.9)


class SlotHardCaseViewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = [[instance(20)], [instance(220)], [instance(420)]]
        self.product = {
            "name": WHITE_PEACH,
            "locations": ["H2_L03_C03"],
        }

    def run_ordering(self, target_id: str, hand: str, *, image_width: int | None = None):
        with (
            patch(
                "pick.locate.main.split_instances_into_display_groups",
                return_value=self.columns,
            ),
            patch("pick.locate.main.lookup_sku_row", return_value=ROW),
        ):
            return apply_hard_case_ordering(
                [item for column in self.columns for item in column],
                product=self.product,
                task_type="SORTING",
                level="L3",
                hand=hand,
                slot_id="H2_L03_C03",
                target_id=target_id,
                image_width=image_width,
            )

    def assert_exact_slot_mapping(self, target_id: str, hand: str) -> None:
        instances, debug = self.run_ordering(target_id, hand)
        self.assertIsNotNone(debug)
        assert debug is not None
        self.assertTrue(debug.slot_view_applied)
        self.assertEqual(debug.target_slot_id, "H2_L03_C03")
        self.assertEqual(debug.target_id, target_id)
        self.assertEqual(
            debug.visible_slot_order,
            ["H2_L03_C01", "H2_L03_C02", "H2_L03_C03"],
        )
        self.assertEqual(
            [group.mapped_slot_id for group in debug.groups],
            debug.visible_slot_order,
        )
        selected = [item for item in instances if item.is_selected]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].mapped_slot_id, "H2_L03_C03")
        self.assertEqual(selected[0].bbox, self.columns[2][0].bbox)

    def test_front_left_maps_by_exact_slot(self) -> None:
        self.assert_exact_slot_mapping("H2_INSPECT", "left")

    def test_connector_right_does_not_reverse_to_row_suffix(self) -> None:
        self.assert_exact_slot_mapping("H12_INSPECT", "right")

    def test_detection_count_mismatch_fails_closed(self) -> None:
        self.columns.pop()
        with self.assertRaises(HTTPException) as context:
            self.run_ordering("H12_INSPECT", "right")
        self.assertEqual(context.exception.status_code, 422)
        self.assertIn("检测列数", str(context.exception.detail))

    def test_excess_truncated_left_or_right_column_is_removed_before_mapping(self) -> None:
        complete_columns = self.columns
        for side, bbox in (
            ("left", [0, 100, 20, 400]),
            ("right", [620, 100, 640, 400]),
        ):
            with self.subTest(side=side):
                edge_column = [LocatedInstance(bbox=bbox, mask="", score=0.9)]
                self.columns = (
                    [edge_column] + complete_columns
                    if side == "left"
                    else complete_columns + [edge_column]
                )
                instances, debug = self.run_ordering(
                    "H12_INSPECT", "right", image_width=640
                )

                self.assertEqual([item.bbox for item in instances], [
                    column[0].bbox for column in complete_columns
                ])
                self.assertEqual([item.mapped_slot_id for item in instances], [
                    "H2_L03_C01", "H2_L03_C02", "H2_L03_C03"
                ])
                self.assertEqual(debug.selected_group_index, 3)
                self.assertTrue(instances[2].is_selected)

    def test_excess_full_edge_or_narrow_interior_column_still_fails_closed(self) -> None:
        complete_columns = self.columns
        for bbox, image_width in (
            ([580, 100, 640, 400], 640),  # Full-shape bottle at the edge.
            ([540, 100, 560, 400], 640),  # Narrow but not image-truncated.
            ([620, 100, 640, 400], None),  # Cannot infer the image boundary.
        ):
            with self.subTest(bbox=bbox, image_width=image_width):
                self.columns = complete_columns + [
                    [LocatedInstance(bbox=bbox, mask="", score=0.9)]
                ]
                with self.assertRaises(HTTPException) as context:
                    self.run_ordering("H12_INSPECT", "right", image_width=image_width)
                self.assertEqual(context.exception.status_code, 422)
                self.assertIn("visible=4, configured=3", str(context.exception.detail))

    def test_exact_count_keeps_truncated_edge_column(self) -> None:
        self.columns[-1] = [
            LocatedInstance(bbox=[620, 100, 640, 400], mask="", score=0.9)
        ]
        instances, _ = self.run_ordering("H12_INSPECT", "right", image_width=640)
        self.assertEqual(len(instances), 3)
        self.assertEqual(instances[-1].bbox, self.columns[-1][0].bbox)
        self.assertTrue(instances[-1].is_selected)

    def test_both_truncated_edges_remove_narrower_first_only_up_to_overflow(self) -> None:
        complete_columns = self.columns
        left = [LocatedInstance(bbox=[0, 100, 10, 400], mask="", score=0.9)]
        right = [LocatedInstance(bbox=[620, 100, 640, 400], mask="", score=0.9)]
        for middle_count in (2, 3):
            with self.subTest(middle_count=middle_count):
                middle = complete_columns[:middle_count]
                self.columns = [left] + middle + [right]
                instances, _ = self.run_ordering(
                    "H12_INSPECT", "right", image_width=640
                )
                expected = middle + [right] if middle_count == 2 else middle
                self.assertEqual([item.bbox for item in instances], [
                    column[0].bbox for column in expected
                ])
                self.assertEqual(len(instances), 3)

    def test_invalid_hand_target_view_fails_closed(self) -> None:
        with self.assertRaises(HTTPException) as context:
            self.run_ordering("H2_INSPECT", "right")
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("没有 hard case 腕部视角配置", str(context.exception.detail))

    def test_every_configured_route_selects_the_exact_slot(self) -> None:
        for level, row in HARD_CASE_ROWS.items():
            columns = [[instance(20 + index * 200)] for index in range(len(row))]
            for target_id, hand in (
                ("H2_INSPECT", "left"),
                ("H12_INSPECT", "right"),
            ):
                for expected_index, item in enumerate(row):
                    with self.subTest(
                        level=level,
                        target_id=target_id,
                        hand=hand,
                        slot_id=item["location_id"],
                    ), patch(
                        "pick.locate.main.split_instances_into_display_groups",
                        return_value=columns,
                    ), patch(
                        "pick.locate.main.lookup_sku_row",
                        return_value=row,
                    ):
                        instances, debug = apply_hard_case_ordering(
                            [candidate for column in columns for candidate in column],
                            product={
                                "name": item["name"],
                                "locations": [item["location_id"]],
                            },
                            task_type="SORTING",
                            level=level,
                            hand=hand,
                            slot_id=item["location_id"],
                            target_id=target_id,
                        )

                    self.assertIsNotNone(debug)
                    selected = [candidate for candidate in instances if candidate.is_selected]
                    self.assertEqual(len(selected), 1)
                    self.assertEqual(selected[0].mapped_slot_id, item["location_id"])
                    self.assertEqual(selected[0].bbox, columns[expected_index][0].bbox)

    def test_shuke_passion_fruit_keeps_legacy_ordering_with_request_context(self) -> None:
        names = [
            "舒克牙膏竹炭薄荷",
            "舒克牙膏柠檬百香果",
            "舒克牙膏海盐薄荷",
        ]
        columns = [[instance(20 + index * 200)] for index in range(len(names))]
        row = [
            {"name": name, "location_id": f"H1_L02_C{index:02d}"}
            for index, name in enumerate(names, start=1)
        ]
        with patch(
            "pick.locate.main.split_instances_into_display_groups",
            return_value=columns,
        ), patch(
            "pick.locate.main.lookup_sku_row",
            return_value=row,
        ):
            instances, debug = apply_hard_case_ordering(
                [candidate for column in columns for candidate in column],
                product={
                    "name": "舒克牙膏柠檬百香果",
                    "locations": ["H1_L02_C02"],
                },
                task_type="SORTING",
                level="L2",
                hand="left",
                slot_id="H1_L02_C02",
                target_id="H1_INSPECT",
            )

        self.assertIsNotNone(debug)
        assert debug is not None
        self.assertEqual(debug.group_id, "shuke_toothpaste")
        self.assertFalse(debug.slot_view_applied)
        selected = [candidate for candidate in instances if candidate.is_selected]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].bbox, columns[1][0].bbox)


if __name__ == "__main__":
    unittest.main()
