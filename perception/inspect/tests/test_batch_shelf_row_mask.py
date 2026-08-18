from __future__ import annotations

import base64
import importlib.util
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "batch_shelf_row_mask.py"
SPEC = importlib.util.spec_from_file_location("batch_shelf_row_mask_test_api", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT_PATH}")
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


def encoded_mask(mask: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", mask)
    if not success:
        raise RuntimeError("cannot encode test mask")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class ShelfMaskComponentTest(unittest.TestCase):
    def test_only_narrow_components_touching_horizontal_edges_are_removed(self) -> None:
        mask = np.zeros((100, 1000), dtype=np.uint8)
        mask[5:25, 0:100] = 255
        mask[45:95, 150:900] = 255
        mask[5:25, 920:1000] = 255
        candidates = script.component_candidates(
            {"instances": [{"score": 0.9, "mask_png_base64": encoded_mask(mask)}]},
            np.zeros((100, 1000, 3), dtype=np.uint8),
            max_edge_component_width_ratio=0.15,
        )

        self.assertEqual(len(candidates), 3)
        removed = [item for item in candidates if item["removed_edge_sliver"]]
        kept = [item for item in candidates if item["kept"]]
        self.assertEqual(len(removed), 2)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0]["width_ratio"], 0.75)

        cleaned = np.zeros(mask.shape, dtype=np.uint8)
        cleaned[kept[0]["mask"]] = 255
        spanning = script.spanning_components(cleaned, min_width_ratio=0.70)
        self.assertEqual(len(spanning), 1)
        self.assertAlmostEqual(spanning[0]["width_ratio"], 0.75)

    def test_exactly_seventy_percent_does_not_satisfy_strict_rule(self) -> None:
        mask = np.zeros((60, 1000), dtype=np.uint8)
        mask[10:50, 150:850] = 255

        self.assertEqual(
            script.spanning_components(mask, min_width_ratio=0.70),
            [],
        )

    def test_default_edge_threshold_is_thirty_percent(self) -> None:
        from unittest.mock import patch

        with patch.object(script.sys, "argv", ["batch_shelf_row_mask.py"]):
            args = script.parse_args()

        self.assertEqual(args.max_edge_component_width_ratio, 0.30)
        self.assertEqual(args.edge_touch_tolerance_ratio, 0.02)
        self.assertEqual(args.edge_touch_min_tolerance_px, 10)
        self.assertEqual(args.min_spanning_component_width_ratio, 0.60)

    def test_near_right_edge_fragment_uses_tolerance(self) -> None:
        mask = np.zeros((100, 1000), dtype=np.uint8)
        mask[20:90, 850:985] = 255
        candidates = script.component_candidates(
            {"instances": [{"score": 0.9, "mask_png_base64": encoded_mask(mask)}]},
            np.zeros((100, 1000, 3), dtype=np.uint8),
            max_edge_component_width_ratio=0.30,
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate["right_edge_gap_px"], 15)
        self.assertEqual(candidate["edge_touch_tolerance_px"], 20)
        self.assertTrue(candidate["touches_right_edge"])
        self.assertTrue(candidate["removed_edge_sliver"])

    def test_edge_removal_uses_mask_bottom_edge_not_full_bbox_width(self) -> None:
        mask = np.zeros((100, 1000), dtype=np.uint8)
        polygon = np.asarray([[0, 10], [350, 10], [80, 99], [0, 99]], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        candidates = script.component_candidates(
            {"instances": [{"score": 0.9, "mask_png_base64": encoded_mask(mask)}]},
            np.zeros((100, 1000, 3), dtype=np.uint8),
            max_edge_component_width_ratio=0.30,
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertGreater(candidate["width_ratio"], 0.30)
        self.assertLess(candidate["bottom_edge_width_ratio"], 0.30)
        self.assertTrue(candidate["removed_edge_sliver"])

    def test_edge_masks_are_removed_before_remaining_masks_are_merged(self) -> None:
        left_edge = np.zeros((80, 1000), dtype=np.uint8)
        middle_left = np.zeros((80, 1000), dtype=np.uint8)
        middle_right = np.zeros((80, 1000), dtype=np.uint8)
        right_edge = np.zeros((80, 1000), dtype=np.uint8)
        left_edge[0:10, 0:100] = 255
        middle_left[20:70, 100:600] = 255
        middle_right[20:70, 550:950] = 255
        right_edge[0:10, 920:1000] = 255
        rgb = np.zeros((80, 1000, 3), dtype=np.uint8)
        candidates = script.component_candidates(
            {
                "instances": [
                    {"score": 0.9, "mask_png_base64": encoded_mask(left_edge)},
                    {"score": 0.9, "mask_png_base64": encoded_mask(middle_left)},
                    {"score": 0.9, "mask_png_base64": encoded_mask(middle_right)},
                    {"score": 0.9, "mask_png_base64": encoded_mask(right_edge)},
                ]
            },
            rgb,
            max_edge_component_width_ratio=0.15,
        )
        kept = [candidate for candidate in candidates if candidate["kept"]]
        cleaned = np.zeros(rgb.shape[:2], dtype=np.uint8)
        for candidate in kept:
            cleaned[candidate["mask"]] = 255

        self.assertEqual(len(candidates) - len(kept), 2)
        spanning = script.spanning_components(cleaned, min_width_ratio=0.70)
        self.assertEqual(len(spanning), 1)
        self.assertAlmostEqual(spanning[0]["width_ratio"], 0.85)

if __name__ == "__main__":
    unittest.main()
