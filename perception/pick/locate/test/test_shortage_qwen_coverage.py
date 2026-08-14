from __future__ import annotations

import unittest

from pick.locate.main import (
    LocatedInstance,
    bbox_coverage_ratio,
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


if __name__ == "__main__":
    unittest.main()
