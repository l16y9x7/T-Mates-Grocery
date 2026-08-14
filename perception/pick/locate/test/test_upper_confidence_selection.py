from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from pick.locate import main as locate_main
from pick.locate.main import (
    LocatedInstance,
    UPPER_CONFIDENCE_PICK_PRODUCTS,
    select_upper_high_confidence_instance,
    uses_upper_confidence_pick,
)


def instance(bbox: list[float], score: float | None) -> LocatedInstance:
    return LocatedInstance(bbox=bbox, mask="unused-by-selection", score=score)


class UpperConfidenceSelectionTest(unittest.TestCase):
    def test_configured_products_are_sorting_only(self) -> None:
        self.assertEqual(len(UPPER_CONFIDENCE_PICK_PRODUCTS), 14)
        self.assertTrue(uses_upper_confidence_pick("拖鞋", "SORTING"))
        self.assertFalse(uses_upper_confidence_pick("拖鞋", "SHORTAGE"))
        self.assertFalse(uses_upper_confidence_pick("拖鞋", "MISPLACED"))
        self.assertFalse(uses_upper_confidence_pick("杯子", "SORTING"))

    def test_slipper_batch_candidates_select_upper_high_score_bbox(self) -> None:
        spanning = instance([282.80, 236.37, 640.0, 463.55], 0.7382)
        lower = instance([285.86, 332.94, 639.97, 463.06], 0.5928)
        upper = instance([282.26, 238.34, 640.0, 340.49], 0.7865)

        selected = select_upper_high_confidence_instance(
            [spanning, lower, upper]
        )

        self.assertIs(selected, upper)

    def test_upper_candidate_wins_inside_best_score_margin(self) -> None:
        lower_best = instance([0, 200, 100, 300], 0.90)
        upper_near_best = instance([0, 20, 100, 100], 0.85)

        selected = select_upper_high_confidence_instance(
            [lower_best, upper_near_best]
        )

        self.assertIs(selected, upper_near_best)

    def test_low_confidence_upper_false_positive_is_ignored(self) -> None:
        lower_best = instance([0, 200, 100, 300], 0.90)
        upper_low_score = instance([0, 20, 100, 100], 0.60)

        selected = select_upper_high_confidence_instance(
            [lower_best, upper_low_score]
        )

        self.assertIs(selected, lower_best)

    def test_sorting_pipeline_preserves_candidates_and_never_requests_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "rgb.jpg"
            Image.new("RGB", (640, 480), "white").save(image_path)
            mask_buffer = io.BytesIO()
            Image.new("L", (640, 480), 255).save(mask_buffer, format="PNG")
            mask = base64.b64encode(mask_buffer.getvalue()).decode("ascii")
            sam_instances = [
                {
                    "bbox_xyxy": [282.80, 236.37, 640.0, 463.55],
                    "mask_png_base64": mask,
                    "score": 0.7382,
                },
                {
                    "bbox_xyxy": [285.86, 332.94, 639.97, 463.06],
                    "mask_png_base64": mask,
                    "score": 0.5928,
                },
                {
                    "bbox_xyxy": [282.26, 238.34, 640.0, 340.49],
                    "mask_png_base64": mask,
                    "score": 0.7865,
                },
            ]

            def unexpected_depth_provider(_size: tuple[int, int]) -> Image.Image:
                self.fail("upper-confidence SORTING must not request depth")

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
                patch.object(locate_main, "call_sam3", return_value=sam_instances),
                patch.object(
                    locate_main,
                    "store_monitor_image",
                    return_value=str(image_path),
                ),
            ):
                response = locate_main.locate_product_in_image(
                    {"sku_id": "SKU_104", "name": "拖鞋"},
                    image_path,
                    task_type="SORTING",
                    level="L2",
                    hand="left",
                    depth_image_provider=unexpected_depth_provider,
                )

        self.assertEqual(len(response.instances), 3)
        self.assertEqual(response.selected_instance_index, 3)
        self.assertEqual(response.selected_instance.score, 0.7865)
        self.assertTrue(all(item.depth_mm is None for item in response.instances))

    def test_uploaded_depth_is_not_decoded_for_configured_sorting_product(self) -> None:
        rgb_buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "white").save(rgb_buffer, format="JPEG")
        expected_response = Mock()
        request = locate_main.LocateRequest(
            task_type="SORTING",
            product_name="拖鞋",
            level="L2",
            hand="left",
            image_name="rgb.jpg",
            image_base64=base64.b64encode(rgb_buffer.getvalue()).decode("ascii"),
            depth_image_name="depth_mm.npy",
            depth_image_base64="this-is-intentionally-not-valid-depth",
        )

        with (
            patch.object(
                locate_main,
                "lookup_sku_by_name",
                return_value={"sku_id": "SKU_104", "name": "拖鞋"},
            ),
            patch.object(
                locate_main,
                "decode_uploaded_depth_image",
            ) as decode_depth,
            patch.object(
                locate_main,
                "locate_product_in_image",
                return_value=expected_response,
            ) as locate,
        ):
            response = locate_main.locate_product_debug(request)

        self.assertIs(response, expected_response)
        decode_depth.assert_not_called()
        self.assertIsNone(locate.call_args.kwargs["depth_image"])


if __name__ == "__main__":
    unittest.main()
