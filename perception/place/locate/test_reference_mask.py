from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from .reference_mask import (
    ReferenceMaskError,
    generate_reference_mask,
    load_shortage_sam_prompt,
    reference_crop_box,
)


def encoded_mask(mask: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", mask)
    if not success:
        raise RuntimeError("failed to encode test mask")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class ReferenceMaskTest(unittest.TestCase):
    def test_loads_existing_shortage_sam_prompt(self) -> None:
        self.assertEqual(load_shortage_sam_prompt("可口可乐罐装"), "can")

    def test_generates_full_resolution_mask_from_best_overlapping_instance(self) -> None:
        image = np.zeros((100, 140, 3), dtype=np.uint8)
        bbox = [55, 25, 24, 55]
        component = np.zeros(image.shape[:2], dtype=np.uint8)
        component[30:78, 58:77] = 255
        crop_box = reference_crop_box(image.shape, bbox)
        left, top, right, bottom = crop_box
        crop_height = bottom - top
        crop_width = right - left

        target_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
        target_mask[30 - top : 78 - top, 58 - left : 77 - left] = 255
        distractor_mask = np.zeros_like(target_mask)
        distractor_mask[5:18, 2:12] = 255

        with tempfile.TemporaryDirectory() as directory:
            mapping_path = Path(directory) / "prompts.json"
            mapping_path.write_text(
                json.dumps(
                    {"测试商品": {"qwen3_prompt": "unused", "sam3_prompt": "carton"}},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            def fake_sam3(prompt: str, crop: np.ndarray) -> list[dict[str, object]]:
                self.assertEqual(prompt, "carton")
                self.assertEqual(crop.shape[:2], (crop_height, crop_width))
                return [
                    {
                        "score": 0.99,
                        "mask_png_base64": encoded_mask(distractor_mask),
                    },
                    {
                        "score": 0.75,
                        "mask_png_base64": encoded_mask(target_mask),
                    },
                ]

            result = generate_reference_mask(
                image,
                bbox,
                "测试商品",
                component_mask=component,
                mapping_path=mapping_path,
                sam3_client=fake_sam3,
            )

        self.assertEqual(result.mask.shape, image.shape[:2])
        np.testing.assert_array_equal(result.mask[30:78, 58:77], 255)
        self.assertEqual(np.count_nonzero(result.mask), np.count_nonzero(target_mask))
        self.assertEqual(result.sam_prompt, "carton")
        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(result.selected_score, 0.75)

    def test_rejects_instances_outside_the_shortage_region(self) -> None:
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        bbox = [50, 25, 20, 35]
        crop_box = reference_crop_box(image.shape, bbox)
        crop_height = crop_box[3] - crop_box[1]
        crop_width = crop_box[2] - crop_box[0]
        outside = np.zeros((crop_height, crop_width), dtype=np.uint8)
        outside[:5, :5] = 255

        with tempfile.TemporaryDirectory() as directory:
            mapping_path = Path(directory) / "prompts.json"
            mapping_path.write_text(
                '{"测试商品":{"sam3_prompt":"box"}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ReferenceMaskError, "均不与完整图缺货区域重叠"):
                generate_reference_mask(
                    image,
                    bbox,
                    "测试商品",
                    mapping_path=mapping_path,
                    sam3_client=lambda _prompt, _crop: [
                        {"score": 0.9, "mask_png_base64": encoded_mask(outside)}
                    ],
                )


if __name__ == "__main__":
    unittest.main()
