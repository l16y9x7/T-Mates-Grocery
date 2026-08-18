from __future__ import annotations

import unittest

try:
    from .shortage_slot_matching import match_normalized_slots
except ImportError:
    from shortage_slot_matching import match_normalized_slots


class ShortageSlotMatchingTest(unittest.TestCase):
    def test_regular_camera_translation_keeps_ordinal_matching(self) -> None:
        result = match_normalized_slots(
            [0.20, 0.30, 0.40],
            [0.23, 0.33, 0.43],
            normalized_pitch=0.10,
        )

        self.assertEqual(result.strategy, "ordinal_left_to_right")
        self.assertEqual(result.matches, {0: 0, 1: 1, 2: 2})
        self.assertAlmostEqual(result.normalized_shift, 0.03, places=6)

    def test_inconsistent_span_anchors_to_original_image_edge(self) -> None:
        # Regression for H1_F_R_INSPECT_UPPER/...42d88cd4/L2/group_01:
        # left endpoint moves left while the right endpoint moves right. Those
        # candidates cannot result from one camera translation.
        baseline = [173.59 / 1280.0, 314.12 / 1280.0, 443.20 / 1280.0]
        current = [40.93 / 1280.0, 277.45 / 1280.0, 543.55 / 1280.0]

        result = match_normalized_slots(
            baseline,
            current,
            normalized_pitch=0.105317,
        )

        self.assertEqual(result.strategy, "source_image_endpoint_bounds")
        self.assertEqual(result.matches, {1: 1})
        self.assertEqual(result.normalized_shift, 0.0)


if __name__ == "__main__":
    unittest.main()
