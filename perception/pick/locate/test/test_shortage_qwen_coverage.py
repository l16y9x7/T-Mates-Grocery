from __future__ import annotations

import unittest

from pick.locate.main import (
    LocatedInstance,
    bbox_coverage_ratio,
    keep_visibly_complete_pick_candidates,
    keep_sam_instances_with_qwen_coverage,
)


def instance(bbox: list[float], source: int | None = 0) -> LocatedInstance:
    return LocatedInstance(
        bbox=bbox,
        mask="unused-by-bbox-coverage-filter",
        source_qwen_index=source,
    )


class ShortageQwenCoverageTest(unittest.TestCase):
    def test_bbox_coverage_uses_original_qwen_area(self) -> None:
        self.assertAlmostEqual(
            bbox_coverage_ratio(
                [120.0, 120.0, 180.0, 180.0],
                [100.0, 100.0, 200.0, 200.0],
            ),
            0.36,
        )

    def test_tiny_sam_fragment_is_filtered_before_selection(self) -> None:
        complete = instance([105.0, 110.0, 195.0, 290.0])
        tiny = instance([140.0, 240.0, 170.0, 280.0])

        filtered = keep_sam_instances_with_qwen_coverage(
            [complete, tiny],
            {0: [100.0, 100.0, 200.0, 300.0]},
            minimum_coverage=0.25,
        )

        self.assertEqual(filtered, [complete])

    def test_sam_extending_outside_qwen_is_measured_by_intersection(self) -> None:
        covering = instance([50.0, 50.0, 250.0, 350.0])

        filtered = keep_sam_instances_with_qwen_coverage(
            [covering],
            {0: [100.0, 100.0, 200.0, 300.0]},
            minimum_coverage=0.9,
        )

        self.assertEqual(filtered, [covering])

    def test_largest_bbox_is_preserved_even_below_threshold(self) -> None:
        largest = instance([110.0, 110.0, 150.0, 150.0])
        smaller = instance([160.0, 160.0, 170.0, 170.0])

        filtered = keep_sam_instances_with_qwen_coverage(
            [largest, smaller],
            {0: [0.0, 0.0, 1000.0, 1000.0]},
            minimum_coverage=0.25,
        )

        self.assertEqual(filtered, [largest])

    def test_instance_without_matching_qwen_source_is_preserved(self) -> None:
        unknown_source = instance([0.0, 0.0, 1.0, 1.0], source=None)

        filtered = keep_sam_instances_with_qwen_coverage(
            [unknown_source],
            {0: [100.0, 100.0, 200.0, 300.0]},
        )

        self.assertEqual(filtered, [unknown_source])

    def test_record_187404_area_filter_removes_bottle_cap_bbox(self) -> None:
        complete_bboxes = [
            [371.35, 156.79, 437.93, 316.03],
            [175.51, 173.04, 209.24, 317.53],
            [210.10, 166.68, 253.88, 327.91],
            [219.92, 159.28, 267.02, 328.69],
            [326.97, 165.84, 375.08, 319.51],
            [255.50, 144.81, 328.10, 326.83],
        ]
        bottle_cap = instance([306.78, 172.07, 339.07, 208.00])
        candidates = [instance(bbox) for bbox in complete_bboxes] + [bottle_cap]

        filtered = keep_sam_instances_with_qwen_coverage(
            candidates,
            {0: [174.93, 144.80, 439.89, 325.28]},
            minimum_coverage=0.25,
        )

        self.assertNotIn(bottle_cap, filtered)
        self.assertEqual(filtered, [candidates[5]])

    def test_short_square_cannot_set_complete_candidate_reference(self) -> None:
        tall_complete = instance([0.0, 0.0, 72.0, 182.0])
        short_square = instance([0.0, 0.0, 32.0, 36.0])

        filtered = keep_visibly_complete_pick_candidates(
            [tall_complete, short_square],
            min_ratio_to_best=0.75,
            min_height_ratio_to_tallest=0.60,
        )

        self.assertEqual(filtered, [tall_complete])


if __name__ == "__main__":
    unittest.main()
