from __future__ import annotations

import base64
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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


class _FakeReviewer:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def review(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.result


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
            pose_type="",
            reference_item_area=8000,
        )

        self.assertTrue(response.has_anomaly)
        self.assertEqual(response.location_id, "H1_F")
        self.assertEqual(response.pose_type, "")
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
            pose_type="SHELF_VIEW_UPPER",
        )

        self.assertFalse(response.has_anomaly)
        self.assertEqual(response.findings, [])
        self.assertEqual(response.algorithms[0].difference_mode, "chroma")

    def test_route_accepts_plain_base64_and_data_url(self) -> None:
        request = inspect_api.InspectRequest(
            task_type="SHORTAGE",
            location_id=" H1_F ",
            pose_type="",
            baseline_image_base64=encode_image(self.baseline, data_url=True),
            current_image_base64=encode_image(self.current),
            reference_item_area=8000,
        )

        reviewed = inspect_api.QwenReviewResult(
            findings=(
                inspect_api.ReviewedFinding(
                    region_index=1,
                    confidence=0.95,
                    shortage_product_name="测试商品",
                ),
            ),
            raw_response="{}",
            candidate_names=("测试商品",),
        )
        reviewer = _FakeReviewer(reviewed)
        with patch.object(
            inspect_api,
            "QwenReviewer",
            return_value=reviewer,
        ):
            response = inspect_api.inspect_shelf(request)

        self.assertEqual(request.location_id, "H1_F")
        self.assertEqual(len(response.findings), 1)
        self.assertEqual(
            response.findings[0].shortage_product_name,
            "测试商品",
        )
        self.assertEqual(
            response.model_dump(),
            {"findings": [{"shortage_product_name": "测试商品"}]},
        )
        review_image = reviewer.calls[0]["current"]
        self.assertIsInstance(review_image, np.ndarray)
        assert isinstance(review_image, np.ndarray)
        self.assertEqual(review_image.shape[:2], (720, 1280))
        baseline_image = reviewer.calls[0]["baseline"]
        self.assertIsInstance(baseline_image, np.ndarray)
        assert isinstance(baseline_image, np.ndarray)
        self.assertEqual(baseline_image.shape[:2], self.baseline.shape[:2])
        self.assertEqual(reviewer.calls[0]["row_constraints"], [None])

    def test_row_detection_assigns_finding_to_matching_visible_row(self) -> None:
        shelf = np.full((720, 1280, 3), 45, dtype=np.uint8)
        for rail_y in (330, 670):
            cv2.rectangle(shelf, (10, rail_y), (1270, rail_y + 15), (20, 20, 220), -1)
        row_detection = inspect_api.detect_rows(shelf)
        findings = [
            inspect_api.Finding(
                bbox=[420, 390, 120, 180],
                center=[480, 480],
                sources=["comparison_based"],
                votes=1,
            )
        ]

        constraints = inspect_api.build_row_constraints(
            findings,
            row_detection,
            "SHELF_VIEW_UPPER",
        )

        self.assertIsNotNone(constraints[0])
        assert constraints[0] is not None
        self.assertEqual(constraints[0].row_index, 2)
        self.assertGreaterEqual(constraints[0].overlap_ratio, 0.6)

    def test_row_detection_falls_back_when_visible_row_count_mismatches(self) -> None:
        shelf = np.full((720, 1280, 3), 45, dtype=np.uint8)
        cv2.rectangle(shelf, (10, 500), (1270, 515), (20, 20, 220), -1)
        row_detection = inspect_api.detect_rows(shelf)
        findings = [
            inspect_api.Finding(
                bbox=[420, 200, 120, 180],
                center=[480, 290],
                sources=["comparison_based"],
                votes=1,
            )
        ]

        constraints = inspect_api.build_row_constraints(
            findings,
            row_detection,
            "SHELF_VIEW_UPPER",
        )

        self.assertEqual(constraints, [None])

    def test_lower_pose_maps_bottom_three_of_four_detected_rows(self) -> None:
        shelf = np.full((720, 1280, 3), 45, dtype=np.uint8)
        for rail_y in (150, 330, 510, 690):
            cv2.rectangle(
                shelf,
                (10, rail_y),
                (1270, rail_y + 15),
                (20, 20, 220),
                -1,
            )
        row_detection = inspect_api.detect_rows(shelf)
        findings = [
            inspect_api.Finding(
                bbox=[530, 550, 170, 60],
                center=[615, 580],
                sources=["comparison_based"],
                votes=1,
            )
        ]

        constraints = inspect_api.build_row_constraints(
            findings,
            row_detection,
            "SHELF_VIEW_LOWER",
        )

        self.assertIsNotNone(constraints[0])
        assert constraints[0] is not None
        self.assertEqual(constraints[0].detected_row_index, 4)
        self.assertEqual(constraints[0].row_index, 3)

    def test_no_change_route_returns_empty_findings(self) -> None:
        encoded = encode_image(self.baseline)
        request = inspect_api.InspectRequest(
            task_type="SHORTAGE",
            location_id="H1_F",
            pose_type="SHELF_VIEW_LOWER",
            baseline_image_base64=encoded,
            current_image_base64=encoded,
        )

        self.assertEqual(
            inspect_api.inspect_shelf(request).model_dump(),
            {"findings": []},
        )

    def test_misplaced_public_shape_is_stable(self) -> None:
        reviewed = [
            inspect_api.ReviewedFinding(
                region_index=1,
                confidence=0.91,
                misplaced_product_name="实际商品",
                gt_product_name="标准商品",
            )
        ]

        response = inspect_api.build_product_findings("MISPLACED", reviewed)

        self.assertEqual(len(response.findings), 1)
        self.assertEqual(
            response.model_dump(),
            {"findings": [
                {
                    "misplaced_product_name": "实际商品",
                    "gt_product_name": "标准商品",
                }
            ]},
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
                pose_type="",
            )

    def test_python_entry_rejects_blank_location_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "location_id must not be blank"):
            inspect_api.inspect_images(
                "SHORTAGE",
                self.baseline,
                self.current,
                location_id="  ",
                pose_type="",
            )

    def test_openapi_requires_pose_type(self) -> None:
        schema = inspect_api.app.openapi()
        required = schema["components"]["schemas"]["InspectRequest"]["required"]
        self.assertIn("pose_type", required)

    def test_openapi_registers_main_route(self) -> None:
        self.assertIn("/perception/inspect", inspect_api.app.openapi()["paths"])

    def test_openapi_exposes_findings_response_wrapper(self) -> None:
        schema = inspect_api.app.openapi()
        response_schema = schema["paths"]["/perception/inspect"]["post"][
            "responses"
        ]["200"]["content"]["application/json"]["schema"]

        self.assertEqual(
            response_schema,
            {"$ref": "#/components/schemas/InspectApiResponse"},
        )
        self.assertIn(
            "findings",
            schema["components"]["schemas"]["InspectApiResponse"]["properties"],
        )


if __name__ == "__main__":
    unittest.main()
