from __future__ import annotations

import base64
import unittest

import cv2
import numpy as np

import shelf_mask


def encoded_mask(mask: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", mask)
    if not success:
        raise RuntimeError("cannot encode mask")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


class ShelfMaskTest(unittest.TestCase):
    def test_near_edge_sliver_is_removed_and_main_shelf_is_kept(self) -> None:
        image = np.full((100, 1000, 3), 120, dtype=np.uint8)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[20:90, 100:751] = 255
        mask[20:90, 850:985] = 255

        def caller(
            _image: np.ndarray,
            _prompt: str,
            _threshold: float,
            _mask_threshold: float,
        ) -> dict:
            return {
                "instances": [
                    {"score": 0.9, "mask_png_base64": encoded_mask(mask)}
                ]
            }

        result = shelf_mask.apply_shelf_mask(image, sam3_caller=caller)

        self.assertFalse(result.fallback_to_full_image)
        self.assertEqual(len(result.components), 2)
        removed = [item for item in result.components if item["removed_edge_sliver"]]
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["right_edge_gap_px"], 15)
        self.assertEqual(removed[0]["edge_touch_tolerance_px"], 20)
        self.assertEqual(result.filtered_rgb[50, 900].tolist(), [0, 0, 0])
        self.assertEqual(result.filtered_rgb[50, 500].tolist(), [120, 120, 120])

    def test_exactly_sixty_percent_falls_back_to_full_image(self) -> None:
        image = np.full((60, 1000, 3), 80, dtype=np.uint8)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[10:50, 200:800] = 255

        def caller(
            _image: np.ndarray,
            _prompt: str,
            _threshold: float,
            _mask_threshold: float,
        ) -> dict:
            return {"instances": [{"mask_png_base64": encoded_mask(mask)}]}

        result = shelf_mask.apply_shelf_mask(
            image,
            detection_thresholds=(0.5,),
            sam3_caller=caller,
        )

        self.assertTrue(result.fallback_to_full_image)
        np.testing.assert_array_equal(result.filtered_rgb, image)


if __name__ == "__main__":
    unittest.main()
