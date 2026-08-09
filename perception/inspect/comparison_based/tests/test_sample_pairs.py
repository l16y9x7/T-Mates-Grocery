from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from comparison_based import ComparisonConfig, detect_shortage  # noqa: E402


class SuppliedSamplePairsTest(unittest.TestCase):
    data_dir = Path(__file__).resolve().parents[3] / "test_data" / "inspect_shortage_paired"
    misplaced_data_dir = (
        Path(__file__).resolve().parents[3]
        / "test_data"
        / "inspect_misplaced_paired"
    )

    # Expected centers are deliberately tolerant: this is a regression guard,
    # not pixel-perfect ground truth segmentation.
    expected_centers = {
        1: (617, 572),
        2: (640, 431),
        3: (674, 317),
        4: (481, 223),
    }

    def load_test_image(self, path: Path) -> np.ndarray:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        return cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)

    def test_all_supplied_pairs_find_the_removed_item(self) -> None:
        if not self.data_dir.exists():
            self.skipTest("supplied paired sample images are not available")

        for pair, expected_center in self.expected_centers.items():
            with self.subTest(pair=pair):
                baseline = self.load_test_image(
                    self.data_dir / f"{pair}_1.jpg"
                )
                current = self.load_test_image(
                    self.data_dir / f"{pair}_2.jpg"
                )
                result = detect_shortage(
                    baseline,
                    current,
                )
                self.assertEqual(result.image_size, (1280, 720))
                self.assertTrue(result.alignment.success)
                self.assertEqual(len(result.shortages), 1)
                center = result.shortages[0].center
                self.assertLessEqual(abs(center[0] - expected_center[0]), 30)
                self.assertLessEqual(abs(center[1] - expected_center[1]), 30)

    def test_same_image_and_small_translation_do_not_false_alarm(self) -> None:
        if not self.data_dir.exists():
            self.skipTest("supplied paired sample images are not available")

        for pair in self.expected_centers:
            baseline = self.load_test_image(self.data_dir / f"{pair}_1.jpg")
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

    def test_color_sensitive_mode_finds_only_swapped_middle_oreo_boxes(self) -> None:
        if not self.misplaced_data_dir.exists():
            self.skipTest("supplied misplaced sample images are not available")

        result = detect_shortage(
            self.misplaced_data_dir / "3_1.jpg",
            self.misplaced_data_dir / "3_2.jpg",
            ComparisonConfig(
                difference_mode="chroma",
                min_chroma_dominance_ratio=0.35,
            ),
        )

        self.assertEqual(len(result.shortages), 2)
        regions = sorted(result.shortages, key=lambda region: region.center[0])
        self.assertLessEqual(abs(regions[0].center[0] - 870), 30)
        self.assertLessEqual(abs(regions[1].center[0] - 988), 30)
        self.assertTrue(
            all(region.chroma_dominance_ratio >= 0.35 for region in regions)
        )
        self.assertTrue(all(region.center[0] < 1050 for region in regions))


if __name__ == "__main__":
    unittest.main()
