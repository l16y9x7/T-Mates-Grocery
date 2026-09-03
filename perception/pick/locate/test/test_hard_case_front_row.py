from __future__ import annotations

import base64
import io
import unittest

from PIL import Image, ImageDraw

from pick.locate.main import (
    LocatedInstance,
    detect_red_shelf_front_line,
    keep_front_depth_row,
    split_instances_into_display_groups,
)


def instance(bbox: list[float]) -> LocatedInstance:
    return LocatedInstance(bbox=bbox, mask="", score=0.9)


def masked_instance(
    bbox: list[float],
    depth_mm: float | None = None,
) -> LocatedInstance:
    mask = Image.new("L", (640, 480), 0)
    ImageDraw.Draw(mask).rectangle(tuple(bbox), fill=255)
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    return LocatedInstance(
        bbox=bbox,
        mask=base64.b64encode(buffer.getvalue()).decode("ascii"),
        score=0.9,
        depth_mm=depth_mm,
    )


def shaped_mask_instance(
    bbox: list[float],
    rectangles: list[tuple[int, int, int, int]],
) -> LocatedInstance:
    mask = Image.new("L", (640, 480), 0)
    draw = ImageDraw.Draw(mask)
    for rectangle in rectangles:
        draw.rectangle(rectangle, fill=255)
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    return LocatedInstance(
        bbox=bbox,
        mask=base64.b64encode(buffer.getvalue()).decode("ascii"),
        score=0.9,
    )


class HardCaseShelfFrontTest(unittest.TestCase):
    def test_detects_upper_edge_of_thick_red_shelf_strip(self) -> None:
        image = Image.new("RGB", (640, 480), "#20262b")
        draw = ImageDraw.Draw(image)
        top_left = 373
        top_right = 309
        draw.polygon(
            [
                (0, top_left),
                (639, top_right),
                (639, top_right + 44),
                (0, top_left + 44),
            ],
            fill="#c83b2f",
        )

        line = detect_red_shelf_front_line(image)

        self.assertIsNotNone(line)
        assert line is not None
        slope, intercept = line
        detected_center_y = slope * 320 + intercept
        expected_center_y = (top_left + top_right) / 2
        self.assertAlmostEqual(detected_center_y, expected_center_y, delta=8)

    def test_detects_bottom_shelf_edge_interrupted_by_gripper(self) -> None:
        image = Image.new("RGB", (640, 480), "#20262b")
        draw = ImageDraw.Draw(image)
        draw.polygon(
            [(0, 469), (639, 475), (639, 479), (0, 474)],
            fill="#c83b2f",
        )
        # Simulate a dark gripper hiding a wide middle section of the red edge.
        draw.rectangle((65, 455, 275, 479), fill="#111111")

        line = detect_red_shelf_front_line(image)

        self.assertIsNotNone(line)
        assert line is not None
        slope, intercept = line
        self.assertAlmostEqual(slope * 320 + intercept, 472, delta=5)

    def test_single_base_column_relaxes_from_twenty_five_to_thirty_five(self) -> None:
        right_full = instance([259.1, 215.3, 340.1, 342.6])
        left_full = instance([94.9, 216.2, 190.6, 355.5])
        rear_fragment = instance([67.3, 225.8, 106.0, 337.0])

        selected = keep_front_depth_row(
            [right_full, left_full, rear_fragment],
            (-0.175, 419.85),
        )

        self.assertIn(right_full, selected)
        self.assertIn(left_full, selected)
        self.assertNotIn(rear_fragment, selected)

    def test_maiydong_rear_bottle_above_twenty_five_percent_is_removed(self) -> None:
        candidates = [
            masked_instance([486, 122, 585, 390]),
            masked_instance([621, 160, 640, 367]),
            masked_instance([174, 101, 279, 422]),
            masked_instance([156, 119, 202, 374]),
            masked_instance([344, 112, 443, 405]),
            masked_instance([290, 131, 352, 356]),
            masked_instance([430, 136, 482, 349]),
        ]

        groups = split_instances_into_display_groups(
            candidates,
            (-0.1, 446.1),
        )

        self.assertEqual(len(groups), 4)
        self.assertEqual(
            [round((group[0].bbox[0] + group[0].bbox[2]) / 2) for group in groups],
            [226, 394, 536, 630],
        )

    def test_progressive_relaxation_stops_at_distance_gap(self) -> None:
        baseline = instance([0, -25, 50, 75])
        borderline = instance([60, -32, 110, 68])
        rear = instance([120, -60, 170, 40])

        selected = keep_front_depth_row(
            [baseline, borderline, rear],
            (0.0, 100.0),
        )

        self.assertEqual(selected, [baseline, borderline])

    def test_twenty_five_percent_candidates_are_split_at_distance_gap(self) -> None:
        first_front = instance([0, 0, 50, 95])
        second_front = instance([60, 0, 110, 92])
        rear_inside_baseline = instance([120, -22, 170, 78])
        second_rear_inside_baseline = instance([180, -24, 230, 76])

        selected = keep_front_depth_row(
            [
                first_front,
                second_front,
                rear_inside_baseline,
                second_rear_inside_baseline,
            ],
            (0.0, 100.0),
        )

        self.assertEqual(selected, [first_front, second_front])

    def test_mask_tail_does_not_make_rear_instance_touch_shelf(self) -> None:
        rear_with_thin_tail = shaped_mask_instance(
            [0, 0, 50, 100],
            [(0, 0, 49, 70), (24, 70, 25, 99)],
        )
        front = shaped_mask_instance(
            [60, 0, 110, 96],
            [(60, 0, 109, 95)],
        )

        selected = keep_front_depth_row(
            [rear_with_thin_tail, front],
            (0.0, 100.0),
        )

        self.assertEqual(selected, [front])

    def test_does_not_relax_when_multiple_base_columns_exist(self) -> None:
        first_front = instance([0, -25, 50, 75])
        second_front = instance([60, -20, 110, 80])
        borderline_rear = instance([120, -32, 170, 68])

        selected = keep_front_depth_row(
            [first_front, second_front, borderline_rear],
            (0.0, 100.0),
        )

        self.assertEqual(selected, [first_front, second_front])

    def test_bbq_fragment_is_deduplicated_after_two_columns_survive(self) -> None:
        right_full = masked_instance([259.1, 215.3, 340.1, 342.6])
        left_full = masked_instance([94.9, 216.2, 190.6, 355.5])
        left_fragment = masked_instance([67.3, 225.8, 106.0, 337.0])

        groups = split_instances_into_display_groups(
            [right_full, left_full, left_fragment],
            (-0.1, 373.0),
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0], [left_full])
        self.assertEqual(groups[1], [right_full])

    def test_cross_column_depth_gradient_does_not_remove_condiment_columns(self) -> None:
        vinegar = masked_instance([227, 134, 314, 353], 618)
        steamed_fish_soy = masked_instance([323, 123, 408, 362], 566)
        light_soy = masked_instance([469, 100, 567, 381], 483)

        groups = split_instances_into_display_groups(
            [light_soy, steamed_fish_soy, vinegar],
            (0.125, 357.75),
        )

        self.assertEqual(groups, [[vinegar], [steamed_fish_soy], [light_soy]])

    def test_maiydong_rear_bbox_is_not_restored_after_four_columns_exist(self) -> None:
        bboxes = [
            [243.4, 118.1, 329.5, 382.2],
            [483.9, 106.1, 598.7, 425.0],
            [622.3, 170.3, 640.0, 390.0],
            [89.4, 71.2, 227.6, 477.8],
            [219.6, 125.3, 264.6, 358.9],
            [76.1, 98.6, 125.1, 408.0],
            [42.5, 111.8, 97.4, 378.9],
            [424.3, 137.4, 464.6, 344.0],
            [331.2, 93.1, 448.6, 451.4],
            [424.3, 125.8, 464.6, 365.8],
        ]
        candidates = [masked_instance(bbox) for bbox in bboxes]

        groups = split_instances_into_display_groups(
            candidates,
            (-0.15, 510.3),
        )

        self.assertEqual(len(groups), 4)
        self.assertEqual(
            [round((group[0].bbox[0] + group[0].bbox[2]) / 2, 1) for group in groups],
            [158.5, 389.9, 541.3, 631.1],
        )


if __name__ == "__main__":
    unittest.main()
