from __future__ import annotations

import unittest

from shortage_depth_outlier import select_positive_depth_outliers


class ShortageDepthOutlierTest(unittest.TestCase):
    def test_keeps_only_the_strong_outlier_from_multiple_threshold_hits(self) -> None:
        result = select_positive_depth_outliers([20, 29, 39, 48, 156, 58])
        self.assertEqual(result.indices, (4,))

    def test_rejects_small_group_wide_depth_drift(self) -> None:
        self.assertEqual(
            select_positive_depth_outliers([22, 28, 30, 36, 42]).indices,
            (),
        )
        self.assertEqual(
            select_positive_depth_outliers([34, 21, 32, 41, 39]).indices,
            (),
        )

    def test_can_keep_a_two_item_outlier_tail(self) -> None:
        result = select_positive_depth_outliers([20, 60, 65])
        self.assertEqual(result.indices, (1, 2))

    def test_outlier_survives_inside_the_old_systematic_shift_band(self) -> None:
        result = select_positive_depth_outliers([30, 35, 40, 75])
        self.assertEqual(result.indices, (3,))

    def test_single_slot_requires_the_100mm_hard_threshold(self) -> None:
        self.assertEqual(select_positive_depth_outliers([60]).indices, ())
        self.assertEqual(select_positive_depth_outliers([100]).indices, (0,))


if __name__ == "__main__":
    unittest.main()
