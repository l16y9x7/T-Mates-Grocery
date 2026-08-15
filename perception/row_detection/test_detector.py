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

    def test_matches_bbox_only_when_layout_and_overlap_are_reliable(self) -> None:
        result = detect_rows(_synthetic_shelf())

        matches = result.match_bboxes(
            [[500, 275, 100, 120], [500, 180, 100, 80]],
            expected_row_count=3,
            min_overlap_ratio=0.6,
        )

        self.assertIsNotNone(matches[0])
        assert matches[0] is not None
        self.assertEqual(matches[0].row_index, 2)
        self.assertEqual(matches[0].row_bbox, result.rows[1].bbox)
        self.assertGreaterEqual(matches[0].overlap_ratio, 0.6)
        self.assertIsNone(matches[1])

    def test_match_bboxes_falls_back_when_row_count_is_unexpected(self) -> None:
        result = detect_rows(_synthetic_shelf())

        matches = result.match_bboxes(
            [[500, 275, 100, 120]],
            expected_row_count=2,
        )

        self.assertEqual(matches, [None])

    def test_bottom_row_window_maps_extra_detected_row_to_sku_row(self) -> None:
        result = detect_rows(_synthetic_shelf())

        matches = result.match_bboxes_to_row_window(
            [[500, 500, 100, 100], [500, 40, 100, 100]],
            row_count=2,
            anchor="bottom",
            min_overlap_ratio=0.6,
        )

        self.assertIsNotNone(matches[0])
        assert matches[0] is not None
        self.assertEqual(matches[0].detected_row_index, 3)
        self.assertEqual(matches[0].row_index, 2)
        self.assertIsNone(matches[1])

    def test_row_window_falls_back_with_more_than_one_extra_row(self) -> None:
        image = np.full((720, 1280, 3), 45, dtype=np.uint8)
        for top in (120, 255, 390, 525, 660):
            cv2.rectangle(image, (20, top), (1259, top + 15), (25, 25, 210), -1)
        result = detect_rows(image)

        matches = result.match_bboxes_to_row_window(
            [[500, 550, 100, 70]],
            row_count=3,
            anchor="bottom",
        )

        self.assertEqual(len(result.rows), 5)
        self.assertEqual(matches, [None])


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

    def test_merges_nearby_product_band_into_stronger_shelf_rail(self) -> None:
        image = np.full((720, 1280, 3), 45, dtype=np.uint8)
        cv2.rectangle(image, (140, 180), (710, 188), (25, 25, 210), -1)
        cv2.rectangle(image, (0, 242), (1279, 295), (25, 25, 210), -1)
        cv2.rectangle(image, (0, 599), (1279, 654), (25, 25, 210), -1)

        result = detect_rows(
            image,
            RowDetectionConfig(pose_type="SHELF_VIEW_UPPER"),
        )

        self.assertEqual(len(result.rails), 2)
        self.assertEqual(len(result.rows), 2)
        self.assertGreater(result.rails[0].y_center, 240)
        self.assertEqual(result.rows[0].bbox[3], result.rails[0].y_center)

    def test_does_not_merge_close_rails_on_opposite_image_sides(self) -> None:
        image = np.full((720, 1280, 3), 45, dtype=np.uint8)
        cv2.rectangle(image, (0, 200), (510, 215), (25, 25, 210), -1)
        cv2.rectangle(image, (770, 270), (1279, 285), (25, 25, 210), -1)

        result = detect_rows(image)

        self.assertEqual(len(result.rails), 2)

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
