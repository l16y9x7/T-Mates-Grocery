from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from pick.locate import main as locate_main
from pick.locate.main import (
    LocatedInstance,
    select_largest_mask_area_instance,
    uses_max_mask_area_pick,
)


def rectangle_mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> str:
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).rectangle(box, fill=255)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class MaxMaskAreaSelectionTest(unittest.TestCase):
    def test_only_refined_salt_uses_max_mask_area(self) -> None:
        self.assertTrue(uses_max_mask_area_pick("中盐精制盐", "SORTING"))
        self.assertFalse(uses_max_mask_area_pick("中盐精制盐", "MISPLACED"))
        self.assertFalse(uses_max_mask_area_pick("小苏打", "SORTING"))

    def test_selects_actual_largest_mask_not_highest_or_uppermost(self) -> None:
        small_high_score = LocatedInstance(
            bbox=[0, 0, 40, 40],
            mask=rectangle_mask((100, 100), (0, 0, 19, 19)),
            score=0.99,
        )
        largest_lower_score = LocatedInstance(
            bbox=[10, 20, 90, 90],
            mask=rectangle_mask((100, 100), (10, 20, 89, 89)),
            score=0.80,
        )

        selected = select_largest_mask_area_instance(
            [small_high_score, largest_lower_score]
        )

        self.assertIs(selected, largest_lower_score)

    def test_salt_pipeline_preserves_candidates_and_skips_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "rgb.jpg"
            Image.new("RGB", (640, 480), "white").save(image_path)
            small_mask = rectangle_mask((1280, 720), (0, 0, 99, 99))
            largest_mask = rectangle_mask((1280, 720), (0, 0, 599, 499))
            sam_instances = [
                {
                    "bbox_xyxy": [0, 100, 300, 300],
                    "mask_png_base64": small_mask,
                    "score": 0.95,
                },
                {
                    "bbox_xyxy": [200, 200, 900, 650],
                    "mask_png_base64": largest_mask,
                    "score": 0.80,
                },
            ]

            def unexpected_depth_provider(_size: tuple[int, int]) -> Image.Image:
                self.fail("max-mask-area SORTING must not request depth")

            with (
                patch.object(
                    locate_main,
                    "load_prompt_pair",
                    return_value=("qwen prompt", "sam prompt"),
                ),
                patch.object(
                    locate_main,
                    "get_stable_qwen_bboxes",
                    return_value=locate_main.QwenConsensusBBoxes(
                        [[0, 0, 1000, 1000]],
                        [],
                    ),
                ),
                patch.object(
                    locate_main,
                    "call_sam3",
                    return_value=sam_instances,
                ),
                patch.object(
                    locate_main,
                    "store_monitor_image",
                    return_value=str(image_path),
                ),
            ):
                response = locate_main.locate_product_in_image(
                    {"sku_id": "SKU_TEST", "name": "中盐精制盐"},
                    image_path,
                    task_type="SORTING",
                    level="L5",
                    hand="left",
                    depth_image_provider=unexpected_depth_provider,
                )

        self.assertEqual(len(response.instances), 2)
        self.assertEqual(response.selected_instance_index, 2)
        self.assertAlmostEqual(response.selected_instance.score, 0.80)
        self.assertIsNone(response.selected_instance.depth_mm)


if __name__ == "__main__":
    unittest.main()
