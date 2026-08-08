from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from comparison_based import detect_shortage  # noqa: E402


class SuppliedSamplePairsTest(unittest.TestCase):
    data_dir = Path(__file__).resolve().parents[3] / "test_data" / "inspect_shortage_paired"

    # Expected centers are deliberately tolerant: this is a regression guard,
    # not pixel-perfect ground truth segmentation.
    expected_centers = {
        1: (694, 858),
        2: (720, 647),
        3: (758, 475),
        4: (541, 334),
    }

    def test_all_supplied_pairs_find_the_removed_item(self) -> None:
        if not self.data_dir.exists():
            self.skipTest("supplied paired sample images are not available")

        for pair, expected_center in self.expected_centers.items():
            with self.subTest(pair=pair):
                result = detect_shortage(
                    self.data_dir / f"{pair}_1.jpg",
                    self.data_dir / f"{pair}_2.jpg",
                )
                self.assertTrue(result.alignment.success)
                self.assertEqual(len(result.shortages), 1)
                center = result.shortages[0].center
                self.assertLessEqual(abs(center[0] - expected_center[0]), 30)
                self.assertLessEqual(abs(center[1] - expected_center[1]), 30)

    def test_same_image_and_small_translation_do_not_false_alarm(self) -> None:
        if not self.data_dir.exists():
            self.skipTest("supplied paired sample images are not available")

        for pair in self.expected_centers:
            encoded = np.fromfile(self.data_dir / f"{pair}_1.jpg", dtype=np.uint8)
            baseline = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            self.assertIsNotNone(baseline)
            height, width = baseline.shape[:2]

            for shift in (0, 2, 5, 10, 20):
                with self.subTest(pair=pair, shift=shift):
                    transform = np.float32(
                        [[1.0, 0.0, shift], [0.0, 1.0, shift]]
                    )
                    current = cv2.warpAffine(
                        baseline,
                        transform,
                        (width, height),
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                    result = detect_shortage(baseline, current)

                    self.assertTrue(result.alignment.success)
                    self.assertFalse(result.has_shortage)
                    self.assertEqual(result.shortages, [])


if __name__ == "__main__":
    unittest.main()
