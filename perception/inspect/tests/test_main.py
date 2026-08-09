from __future__ import annotations

import base64
import importlib.util
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException


INSPECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "perception_inspect_test_api",
    INSPECT_ROOT / "main.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load inspect/main.py")
inspect_api = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inspect_api
SPEC.loader.exec_module(inspect_api)


def encode_image(image: np.ndarray, *, data_url: bool = False) -> str:
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("failed to encode test image")
    value = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{value}" if data_url else value


class InspectMainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = np.full((240, 320, 3), 35, dtype=np.uint8)
        cv2.rectangle(self.baseline, (110, 75), (209, 154), (220, 220, 220), -1)
        self.current = np.full((240, 320, 3), 35, dtype=np.uint8)

    def test_shortage_runs_comparison_algorithm_and_fuses_bbox(self) -> None:
        response = inspect_api.inspect_images(
            "SHORTAGE",
            self.baseline,
            self.current,
            location_id="H1_F",
            reference_item_area=8000,
        )

        self.assertTrue(response.has_anomaly)
        self.assertEqual(response.location_id, "H1_F")
        self.assertEqual(response.image_size, [1280, 720])
        self.assertEqual(len(response.findings), 1)
        self.assertEqual(response.findings[0].sources, ["comparison_based"])
        self.assertEqual(response.findings[0].votes, 1)
        self.assertEqual(response.algorithms[0].difference_mode, "hybrid")

    def test_identical_images_return_no_findings(self) -> None:
        response = inspect_api.inspect_images(
            "MISPLACED",
            self.baseline,
            self.baseline.copy(),
            location_id="H1_F",
        )

        self.assertFalse(response.has_anomaly)
        self.assertEqual(response.findings, [])
        self.assertEqual(response.algorithms[0].difference_mode, "chroma")

    def test_route_accepts_plain_base64_and_data_url(self) -> None:
        request = inspect_api.InspectRequest(
            task_type="SHORTAGE",
            location_id=" H1_F ",
            baseline_image_base64=encode_image(self.baseline, data_url=True),
            current_image_base64=encode_image(self.current),
            reference_item_area=8000,
        )

        response = inspect_api.inspect_shelf(request)

        self.assertEqual(request.location_id, "H1_F")
        self.assertEqual(len(response), 1)
        self.assertEqual(response[0].shortage_product_name, "")

    def test_no_change_route_returns_empty_array(self) -> None:
        encoded = encode_image(self.baseline)
        request = inspect_api.InspectRequest(
            task_type="SHORTAGE",
            location_id="H1_F",
            baseline_image_base64=encoded,
            current_image_base64=encoded,
        )

        self.assertEqual(inspect_api.inspect_shelf(request), [])

    def test_misplaced_public_shape_is_stable(self) -> None:
        localized = [
            inspect_api.Finding(
                bbox=[100, 200, 80, 160],
                center=[140, 280],
                sources=["comparison_based"],
                votes=1,
            )
        ]

        response = inspect_api.build_product_findings("MISPLACED", localized)

        self.assertEqual(len(response), 1)
        self.assertEqual(
            response[0].model_dump(),
            {"misplaced_product_name": "", "gt_product_name": ""},
        )

    def test_invalid_image_has_clear_client_error(self) -> None:
        with self.assertRaises(HTTPException) as context:
            inspect_api.decode_image("not-base64!", "baseline_image_base64")

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("baseline_image_base64", context.exception.detail)

    def test_python_entry_rejects_unsupported_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHORTAGE or MISPLACED"):
            inspect_api.inspect_images(
                "SORTING",
                self.baseline,
                self.current,
                location_id="H1_F",
            )

    def test_python_entry_rejects_blank_location_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "location_id must not be blank"):
            inspect_api.inspect_images(
                "SHORTAGE",
                self.baseline,
                self.current,
                location_id="  ",
            )

    def test_openapi_registers_main_route(self) -> None:
        self.assertIn("/perception/inspect", inspect_api.app.openapi()["paths"])


if __name__ == "__main__":
    unittest.main()
