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
            reference_item_area=8000,
        )

        self.assertTrue(response.has_anomaly)
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
        )

        self.assertFalse(response.has_anomaly)
        self.assertEqual(response.findings, [])
        self.assertEqual(response.algorithms[0].difference_mode, "chroma")

    def test_route_accepts_plain_base64_and_data_url(self) -> None:
        request = inspect_api.InspectRequest(
            task_type="SHORTAGE",
            baseline_image_base64=encode_image(self.baseline, data_url=True),
            current_image_base64=encode_image(self.current),
            reference_item_area=8000,
        )

        response = inspect_api.inspect_shelf(request)

        self.assertTrue(response.has_anomaly)
        self.assertEqual(response.algorithms[0].name, "comparison_based")

    def test_invalid_image_has_clear_client_error(self) -> None:
        with self.assertRaises(HTTPException) as context:
            inspect_api.decode_image("not-base64!", "baseline_image_base64")

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("baseline_image_base64", context.exception.detail)

    def test_python_entry_rejects_unsupported_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHORTAGE or MISPLACED"):
            inspect_api.inspect_images("SORTING", self.baseline, self.current)

    def test_openapi_registers_main_route(self) -> None:
        self.assertIn("/perception/inspect", inspect_api.app.openapi()["paths"])


if __name__ == "__main__":
    unittest.main()
