from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from row_detection import RowDetectionConfig, detect_rows  # noqa: E402


def _synthetic_shelf() -> np.ndarray:
    image = np.full((720, 1280, 3), 45, dtype=np.uint8)
    for top in (205, 438, 665):
        cv2.rectangle(image, (20, top), (1259, top + 15), (25, 25, 210), -1)
    # Red vertical packages should not become shelf rails.
    cv2.rectangle(image, (120, 250), (205, 420), (10, 10, 220), -1)
    cv2.rectangle(image, (750, 475), (820, 640), (15, 20, 210), -1)
    return image


def _synthetic_sloped_shelf() -> np.ndarray:
    image = np.full((720, 1280, 3), 45, dtype=np.uint8)
    for left_y in (170, 445):
        right_y = left_y + 92
        polygon = np.array(
            [
                [0, left_y],
                [1279, right_y],
                [1279, right_y + 18],
                [0, left_y + 18],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(image, polygon, (25, 25, 210))
    return image


class RowDetectionTest(unittest.TestCase):
    def test_detects_three_rails_and_product_rows(self) -> None:
        result = detect_rows(_synthetic_shelf())

        self.assertEqual(len(result.rails), 3)
        self.assertEqual(len(result.rows), 3)
        self.assertEqual([rail.y_center for rail in result.rails], [213, 446, 673])


    def test_assigns_bbox_to_row_with_largest_vertical_overlap(self) -> None:
        result = detect_rows(_synthetic_shelf())

        row = result.row_for_bbox([500, 275, 100, 120])

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.index, 2)


    def test_blank_image_has_no_rails_or_rows(self) -> None:
        image = np.full((720, 1280, 3), 80, dtype=np.uint8)

        result = detect_rows(image)

        self.assertEqual(result.rails, [])
        self.assertEqual(result.rows, [])


    def test_can_keep_input_size(self) -> None:
        image = cv2.resize(_synthetic_shelf(), (640, 360))
        config = RowDetectionConfig(target_size=None)

        result = detect_rows(image, config)

        self.assertEqual(result.image_size, (640, 360))
        self.assertEqual(len(result.rails), 3)

    def test_hough_fallback_detects_perspective_sloped_rails(self) -> None:
        result = detect_rows(_synthetic_sloped_shelf())

        self.assertEqual(len(result.rails), 2)
        self.assertEqual(len(result.rows), 2)
        self.assertTrue(all(rail.line is not None for rail in result.rails))

    def test_lower_pose_returns_bottom_three_rows_and_uses_image_bottom(self) -> None:
        image = np.full((720, 1280, 3), 45, dtype=np.uint8)
        for top in (105, 315, 530):
            cv2.rectangle(image, (20, top), (1259, top + 15), (25, 25, 210), -1)
        config = RowDetectionConfig(pose_type="SHELF_VIEW_LOWER")

        result = detect_rows(image, config)

        self.assertEqual(len(result.rows), 3)
        self.assertEqual(result.rows[0].bbox[1], 113)
        self.assertEqual(result.rows[-1].bbox[1] + result.rows[-1].bbox[3], 720)
        self.assertIsNone(result.rows[-1].lower_rail_index)

    def test_upper_pose_returns_top_two_rows(self) -> None:
        config = RowDetectionConfig(pose_type="SHELF_VIEW_UPPER")

        result = detect_rows(_synthetic_shelf(), config)

        self.assertEqual(len(result.rows), 2)
        self.assertEqual(result.rows[0].bbox[1], 0)
        self.assertEqual(result.rows[-1].bbox[1] + result.rows[-1].bbox[3], 446)

    def test_lower_pose_does_not_treat_floor_after_fourth_rail_as_row(self) -> None:
        image = np.full((720, 1280, 3), 45, dtype=np.uint8)
        for top in (75, 285, 455, 590):
            cv2.rectangle(image, (20, top), (1259, top + 15), (25, 25, 210), -1)
        config = RowDetectionConfig(pose_type="SHELF_VIEW_LOWER")

        result = detect_rows(image, config)

        self.assertEqual(len(result.rows), 3)
        self.assertEqual(result.rows[0].bbox[1], 83)
        self.assertEqual(result.rows[-1].bbox[1] + result.rows[-1].bbox[3], 598)
        self.assertIsNotNone(result.rows[-1].lower_rail_index)


if __name__ == "__main__":
    unittest.main()
