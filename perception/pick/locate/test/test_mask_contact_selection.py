from __future__ import annotations

import base64
import io
import unittest

from PIL import Image, ImageDraw

from pick.locate import main as locate_main


def instance_from_polygon(
    points: list[tuple[int, int]],
    distance: float | None,
    *,
    row: int = 1,
    score: float = 0.9,
) -> locate_main.LocatedInstance:
    mask = Image.new("L", (640, 480), 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    return locate_main.LocatedInstance(
        bbox=list(mask.getbbox()),
        mask=base64.b64encode(buffer.getvalue()).decode("ascii"),
        score=score,
        display_row_index=row,
        shelf_front_distance_ratio=distance,
    )


class MaskContactSelectionTest(unittest.TestCase):
    def test_tilted_separate_front_bottles_keep_both_inventory_slots(self) -> None:
        polygon = [(250, 100), (309, 100), (259, 399), (200, 399)]
        left = instance_from_polygon(polygon, 0.0954)
        right = instance_from_polygon([(x + 80, y) for x, y in polygon], 0.0934)
        self.assertGreater(
            locate_main.bbox_overlap_by_smaller_area(left.bbox, right.bbox),
            locate_main.SAM_BBOX_OVERLAP_MIN_RATIO,
        )

        candidates = locate_main.keep_display_rows_for_inventory([left, right], 2)
        mapped, target = locate_main.map_inventory_slots_to_instances(
            candidates, ["H2_L05_C03", "H2_L05_C04"], "H2_L05_C03", 640,
        )

        self.assertEqual([item.bbox for item in candidates], [left.bbox, right.bbox])
        self.assertEqual([item.mapped_slot_id for item in mapped], ["H2_L05_C03", "H2_L05_C04"])
        self.assertEqual(target.bbox, left.bbox)

    def test_abutting_masks_need_distance_evidence_to_remove_rear(self) -> None:
        front = instance_from_polygon([(250, 100), (349, 100), (349, 399), (250, 399)], 0.04)
        rear_points = [(350, 100), (449, 100), (449, 399), (350, 399)]
        # The rectangles share a long boundary but have no overlapping pixels.
        for rear_distance, expected_count in [(0.12, 1), (0.045, 2), (None, 2)]:
            with self.subTest(rear_distance=rear_distance):
                rear = instance_from_polygon(rear_points, rear_distance, row=2)
                result = locate_main.keep_display_rows_for_inventory([front, rear], 2)
                self.assertEqual(len(result), expected_count)
                self.assertIs(result[0], front)

    def test_near_identical_masks_are_deduplicated_without_distance(self) -> None:
        polygon = [(200, 100), (299, 100), (299, 399), (200, 399)]
        higher_score = instance_from_polygon(polygon, None, score=0.95)
        duplicate = instance_from_polygon([(x + 1, y) for x, y in polygon], None, score=0.85)

        result = locate_main.keep_frontmost_by_mask_contact([duplicate, higher_score])

        self.assertEqual(len(result), 1)
        self.assertIs(result[0], higher_score)

    def test_directed_contact_chain_removes_both_rear_masks(self) -> None:
        points = [(100, 100), (199, 100), (199, 399), (100, 399)]
        front = instance_from_polygon(points, 0.03, row=1)
        middle = instance_from_polygon([(x + 100, y) for x, y in points], 0.13, row=2)
        rear = instance_from_polygon([(x + 200, y) for x, y in points], 0.23, row=3)

        result = locate_main.keep_display_rows_for_inventory([rear, front, middle], 3)

        self.assertEqual(len(result), 1)
        self.assertIs(result[0], front)


if __name__ == "__main__":
    unittest.main()
