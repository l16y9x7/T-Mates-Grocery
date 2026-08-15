from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from pick.locate import main as locate_main


def encoded_mask(size: tuple[int, int]) -> str:
    buffer = io.BytesIO()
    Image.new("L", size, 255).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class RgbInferenceMappingTest(unittest.TestCase):
    def test_floss_box_qwen_crop_uses_fifty_percent_padding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "rgb.jpg"
            Image.new("RGB", (640, 480), "white").save(image_path)
            observed: dict[str, object] = {}

            def sam_instances(_prompt: str, crop_image: Image.Image):
                observed["sam_crop_size"] = crop_image.size
                return [
                    {
                        "bbox_xyxy": [0.0, 0.0, *map(float, crop_image.size)],
                        "mask_png_base64": encoded_mask(crop_image.size),
                        "score": 0.9,
                    }
                ]

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
                        [[400, 400, 600, 600]],
                        [],
                    ),
                ),
                patch.object(
                    locate_main,
                    "call_sam3",
                    side_effect=sam_instances,
                ),
                patch.object(
                    locate_main,
                    "store_monitor_image",
                    return_value=str(image_path),
                ),
            ):
                response = locate_main.locate_product_in_image(
                    {"sku_id": "SKU_037", "name": "小鹿妈妈牙线"},
                    image_path,
                    task_type="SORTING",
                    hand="left",
                )

        self.assertEqual(observed["sam_crop_size"], (512, 288))
        self.assertEqual(
            response.qwen_bboxes[0].bbox_original,
            [256.0, 192.0, 384.0, 288.0],
        )
        self.assertEqual(
            response.qwen_bboxes[0].crop_box_original,
            [192, 144, 448, 336],
        )

    def test_locate_uses_1280x720_rgb_and_maps_sam_back_to_original(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "rgb.jpg"
            Image.new("RGB", (640, 480), "white").save(image_path)
            observed: dict[str, object] = {}

            def stable_bboxes(_prompt: str, image_bytes: bytes):
                with Image.open(io.BytesIO(image_bytes)) as inference_image:
                    observed["qwen_size"] = inference_image.size
                return locate_main.QwenConsensusBBoxes(
                    [[0, 0, 1000, 1000]],
                    [(1, [{"name": "target", "bbox": [0, 0, 1000, 1000]}])],
                )

            def sam_instances(_prompt: str, crop_image: Image.Image):
                observed["sam_crop_size"] = crop_image.size
                return [
                    {
                        "bbox_xyxy": [200.0, 150.0, 1000.0, 600.0],
                        "mask_png_base64": encoded_mask(crop_image.size),
                        "score": 0.9,
                    }
                ]

            with (
                patch.object(
                    locate_main,
                    "load_prompt_pair",
                    return_value=("qwen prompt", "sam prompt"),
                ),
                patch.object(
                    locate_main,
                    "get_stable_qwen_bboxes",
                    side_effect=stable_bboxes,
                ),
                patch.object(
                    locate_main,
                    "call_sam3",
                    side_effect=sam_instances,
                ),
                patch.object(
                    locate_main,
                    "store_monitor_image",
                    return_value=str(image_path),
                ),
            ):
                response = locate_main.locate_product_in_image(
                    {"sku_id": "SKU_TEST", "name": "杯子"},
                    image_path,
                    task_type="MISPLACED",
                    level="L1",
                    hand="left",
                )

        self.assertEqual(observed["qwen_size"], (1280, 720))
        self.assertEqual(observed["sam_crop_size"], (1280, 720))
        self.assertEqual(response.image_size, [640, 480])
        self.assertEqual(response.inference_image_size, [1280, 720])
        self.assertEqual(
            response.qwen_bboxes[0].crop_box_original,
            [0, 0, 640, 480],
        )
        self.assertEqual(
            response.selected_instance.bbox,
            [100.0, 100.0, 500.0, 400.0],
        )
        mask_bytes = base64.b64decode(response.selected_instance.mask)
        with Image.open(io.BytesIO(mask_bytes)) as mapped_mask:
            self.assertEqual(mapped_mask.size, (640, 480))

    def test_size_mapping_uses_independent_x_and_y_scales(self) -> None:
        self.assertEqual(
            locate_main.map_bbox_between_sizes(
                [200, 150, 1000, 600],
                (1280, 720),
                (640, 480),
            ),
            [100.0, 100.0, 500.0, 400.0],
        )

    def test_sliver_masks_are_removed_before_requesting_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "rgb.jpg"
            Image.new("RGB", (640, 480), "white").save(image_path)
            inference_mask = encoded_mask((1280, 720))
            sam_instances = [
                {
                    "bbox_xyxy": [1178, 299, 1218, 483],
                    "mask_png_base64": inference_mask,
                    "score": 0.643,
                },
                {
                    "bbox_xyxy": [942, 291, 960, 511],
                    "mask_png_base64": inference_mask,
                    "score": 0.821,
                },
                {
                    "bbox_xyxy": [956, 249, 1190, 505],
                    "mask_png_base64": inference_mask,
                    "score": 0.978,
                },
            ]

            def unexpected_depth_provider(_size: tuple[int, int]) -> Image.Image:
                self.fail("one complete candidate must not request depth")

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
                    {"sku_id": "SKU_TEST", "name": "高纤七色糙米"},
                    image_path,
                    task_type="SORTING",
                    level="L4",
                    hand="left",
                    depth_image_provider=unexpected_depth_provider,
                )

        self.assertEqual(len(response.raw_sam_instances), 3)
        self.assertEqual(len(response.instances), 1)
        self.assertAlmostEqual(response.selected_instance.score, 0.978)
        self.assertEqual(
            response.selected_instance.bbox,
            [478.0, 166.0, 595.0, 336.66666666666663],
        )
        self.assertIsNone(response.selected_instance.depth_mm)


if __name__ == "__main__":
    unittest.main()
