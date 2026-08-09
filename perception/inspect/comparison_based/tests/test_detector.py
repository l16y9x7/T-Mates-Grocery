from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from comparison_based import ComparisonConfig, ShortageDetector  # noqa: E402


class ShortageDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = np.full((240, 320, 3), 35, dtype=np.uint8)
        self.current = self.baseline.copy()
        cv2.rectangle(self.baseline, (110, 75), (209, 154), (220, 220, 220), -1)

    def test_detects_removed_product_with_reference_area(self) -> None:
        detector = ShortageDetector(
            ComparisonConfig(
                target_size=None,
                enable_registration=False,
                reference_item_area=100 * 80,
                open_kernel_size=3,
                close_kernel_size=5,
            )
        )

        result = detector.detect(self.baseline, self.current)

        self.assertTrue(result.has_shortage)
        self.assertEqual(len(result.shortages), 1)
        x, y, width, height = result.shortages[0].bbox
        self.assertLessEqual(abs(x - 110), 3)
        self.assertLessEqual(abs(y - 75), 3)
        self.assertGreaterEqual(width, 95)
        self.assertGreaterEqual(height, 75)
        self.assertGreaterEqual(result.shortages[0].area_ratio_to_reference, 0.8)

    def test_identical_images_have_no_shortage(self) -> None:
        detector = ShortageDetector(
            ComparisonConfig(target_size=None, enable_registration=False)
        )
        result = detector.detect(self.baseline, self.baseline.copy())
        self.assertFalse(result.has_shortage)
        self.assertEqual(result.shortages, [])

    def test_save_debug_writes_intermediate_images_and_bbox_metadata(self) -> None:
        detector = ShortageDetector(
            ComparisonConfig(
                target_size=None,
                enable_registration=False,
                reference_item_area=100 * 80,
                open_kernel_size=3,
                close_kernel_size=5,
            )
        )
        result = detector.detect(self.baseline, self.current)

        with tempfile.TemporaryDirectory() as directory:
            artifacts = result.save_debug(directory, self.baseline)
            expected = {
                "baseline",
                "aligned_current",
                "luminance_difference",
                "chroma_difference",
                "difference",
                "difference_heatmap",
                "binary_mask",
                "baseline_bboxes",
                "current_bboxes",
                "difference_bboxes",
                "comparison_bboxes",
                "metadata",
            }
            self.assertEqual(set(artifacts), expected)
            self.assertTrue(all(path.is_file() for path in artifacts.values()))

            metadata = json.loads(artifacts["metadata"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["bbox_format"], ["x", "y", "width", "height"])
            self.assertEqual(
                metadata["shortages"][0]["bbox"],
                list(result.shortages[0].bbox),
            )
            comparison = cv2.imdecode(
                np.fromfile(artifacts["comparison_bboxes"], dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            self.assertEqual(comparison.shape[:2], (240, 640))

    def test_rejects_invalid_kernel_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive odd integer"):
            ComparisonConfig(open_kernel_size=4)

    def test_rejects_non_uint8_array(self) -> None:
        detector = ShortageDetector()
        with self.assertRaisesRegex(ValueError, "uint8"):
            detector.detect(self.baseline.astype(np.float32), self.current)


if __name__ == "__main__":
    unittest.main()
