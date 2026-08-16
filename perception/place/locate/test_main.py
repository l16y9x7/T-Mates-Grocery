from __future__ import annotations

import base64
import io
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from pydantic import ValidationError

from . import main as api
from initial_scan import InitialScan
from row_detection import ShelfRow


def encode_rgb(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("failed to encode test RGB")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def encode_depth(depth: np.ndarray) -> str:
    output = io.BytesIO()
    np.save(output, depth, allow_pickle=False)
    return base64.b64encode(output.getvalue()).decode("ascii")


class PlaceLocateApiTest(unittest.TestCase):
    def make_request(self) -> api.PlaceLocateDebugRequest:
        height, width = 240, 320
        generator = np.random.default_rng(7)
        background = generator.integers(
            15,
            100,
            size=(height, width, 3),
            dtype=np.uint8,
        )
        reference = background.copy()
        cv2.rectangle(reference, (120, 70), (199, 179), (245, 245, 245), -1)
        for y in range(80, 175, 15):
            cv2.line(reference, (128, y), (191, y), (20, 20, 20), 2)
        current = background.copy()

        reference_depth = np.full((height, width), 1200, dtype=np.uint16)
        reference_depth[70:180, 120:200] = 900
        current_depth = np.full((height, width), 1200, dtype=np.uint16)
        self.reference_image = reference
        self.reference_depth = reference_depth.astype(np.float32)
        return api.PlaceLocateDebugRequest(
            task_type="SHORTAGE",
            product_name="测试商品",
            location_id="H1_F_L2_C01",
            pose_type="SHELF_VIEW_UPPER",
            current_image_base64=encode_rgb(current),
            current_depth_image_base64=encode_depth(current_depth),
            reference_bbox=[115, 65, 90, 120],
        )

    def test_rgbd_route_returns_reference_inputs_and_camera_transform(self) -> None:
        request = self.make_request()
        scan = InitialScan(
            inspection_target_id="H1_F_L_INSPECT",
            pose_type="SHELF_VIEW_UPPER",
            directory=Path("task0/H1_F_L_INSPECT_UPPER"),
            rgb_path=Path("task0/H1_F_L_INSPECT_UPPER/rgb.jpg"),
            depth_path=Path("task0/H1_F_L_INSPECT_UPPER/depth_mm.npy"),
            rgb=self.reference_image,
            depth_mm=self.reference_depth,
            metadata={},
        )
        reference_mask = np.zeros(self.reference_image.shape[:2], dtype=np.uint8)
        reference_mask[70:180, 120:200] = 255
        sam_result = api.ReferenceMaskResult(
            mask=reference_mask,
            sam_prompt="box",
            crop_box=(100, 50, 220, 200),
            selected_bbox=(120.0, 70.0, 200.0, 180.0),
            selected_score=0.95,
            candidate_count=1,
        )
        with (
            patch.object(api, "load_initial_scan", return_value=scan),
            patch.object(api, "generate_reference_mask", return_value=sam_result) as generate,
        ):
            result = api.locate_place_debug(request)

        self.assertEqual(result.product_name, "测试商品")
        self.assertEqual(result.level, "L2")
        self.assertEqual(result.location_id, "H1_F_L2_C01")
        self.assertEqual(result.inspection_target_id, "H1_F_L_INSPECT")
        self.assertTrue(result.baseline_path.endswith("rgb.jpg"))
        self.assertEqual(result.image_path, result.baseline_path)
        self.assertEqual(result.image_size, [320, 240])
        self.assertEqual(result.current_image_size, [320, 240])
        self.assertGreater(result.registration.inlier_count, 12)
        self.assertEqual(result.reference_mask_source, "sam3")
        self.assertEqual(result.reference_sam3_prompt, "box")
        self.assertEqual(result.reference_sam3_crop_box, [100, 50, 220, 200])
        self.assertEqual(result.reference_sam3_bbox, [120.0, 70.0, 200.0, 180.0])
        self.assertEqual(result.reference_sam3_candidate_count, 1)
        generate.assert_called_once()
        self.assertEqual(generate.call_args.args[2], "测试商品")
        self.assertTrue(
            np.allclose(
                np.asarray(result.rotate_matrix),
                np.asarray(result.registration.current_from_reference),
                atol=1e-6,
            )
        )
        self.assertEqual(result.bbox, [120, 70, 200, 180])
        mask_bytes = base64.b64decode(result.mask)
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mask.shape, (240, 320))
        self.assertEqual(
            np.count_nonzero(mask),
            np.count_nonzero(reference_mask),
        )

    def test_reference_mask_debug_stops_before_pose_transfer(self) -> None:
        full_request = self.make_request()
        request = api.PlaceReferenceMaskRequest.model_validate(
            full_request.model_dump()
        )
        scan = InitialScan(
            inspection_target_id="H1_F_L_INSPECT",
            pose_type="SHELF_VIEW_UPPER",
            directory=Path("task0/H1_F_L_INSPECT_UPPER"),
            rgb_path=Path("task0/H1_F_L_INSPECT_UPPER/rgb.jpg"),
            depth_path=Path("task0/H1_F_L_INSPECT_UPPER/depth_mm.npy"),
            rgb=self.reference_image,
            depth_mm=self.reference_depth,
            metadata={},
        )
        reference_mask = np.zeros(self.reference_image.shape[:2], dtype=np.uint8)
        reference_mask[70:180, 120:200] = 255
        sam_result = api.ReferenceMaskResult(
            mask=reference_mask,
            sam_prompt="box",
            crop_box=(100, 50, 220, 200),
            selected_bbox=(120.0, 70.0, 200.0, 180.0),
            selected_score=0.95,
            candidate_count=2,
        )
        with (
            patch.object(api, "load_initial_scan", return_value=scan),
            patch.object(api, "generate_reference_mask", return_value=sam_result),
        ):
            result = api.locate_reference_mask_debug(request)

        self.assertEqual(result.reference_mask_source, "sam3")
        self.assertEqual(result.reference_image_size, [320, 240])
        self.assertEqual(result.current_image_size, [320, 240])
        self.assertEqual(result.reference_sam3_prompt, "box")
        self.assertEqual(result.reference_sam3_candidate_count, 2)
        self.assertGreater(len(result.reference_image_base64), 100)
        decoded_mask = cv2.imdecode(
            np.frombuffer(base64.b64decode(result.reference_mask), np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertEqual(decoded_mask.shape, (240, 320))
        self.assertEqual(np.count_nonzero(decoded_mask), np.count_nonzero(reference_mask))

    def test_public_response_returns_pose_estimator_inputs(self) -> None:
        debug = api.PlaceLocateDebugResponse(
            product_name="测试商品",
            bbox=[100, 200, 300, 400],
            mask="mask",
            image_path="task0/rgb.jpg",
            rotate_matrix=np.eye(4).tolist(),
            level="L2",
            task_type="SHORTAGE",
            location_id="H1_F",
            inspection_target_id="H1_F_L_INSPECT",
            baseline_path="task0/H1_F_L_INSPECT_UPPER/rgb.jpg",
            region_index=1,
            image_size=[320, 240],
            current_image_size=[320, 240],
            reference_bbox=[1, 2, 3, 4],
            reference_mask="reference",
            registration=api.RegistrationMetrics(
                current_from_reference=np.eye(4).tolist(),
                rmse_mm=1,
                depth_correspondence_count=20,
                inlier_count=18,
                inlier_ratio=0.9,
                reprojection_rmse_px=0.5,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            capture_root = Path(directory)
            rgb_path = capture_root / "rgb.jpg"
            depth_path = capture_root / "depth_mm.npy"
            rgb_path.write_bytes(b"rgb")
            depth_path.write_bytes(b"depth")
            captured = SimpleNamespace(rgb_path=rgb_path, depth_path=depth_path)
            public_request = api.PlaceLocateRequest(
                task_type="SHORTAGE",
                product_name="测试商品",
                location_id="H1_F_L_INSPECT",
                pose_type="SHELF_VIEW_UPPER",
            )
            with (
                patch.object(
                    api,
                    "inspection_temporary_directory",
                    return_value=nullcontext(directory),
                ),
                patch.object(api, "capture_head_rgbd", return_value=captured),
                patch.object(api, "locate_place_debug", return_value=debug) as locate,
            ):
                response = api.locate_place(public_request)

        self.assertEqual(
            response.model_dump(),
            {
                "product_name": "测试商品",
                "bbox": [100, 200, 300, 400],
                "mask": "mask",
                "image_path": "task0/rgb.jpg",
                "rotate_matrix": np.eye(4).tolist(),
                "level": "L2",
            },
        )
        captured_request = locate.call_args.args[0]
        self.assertIsInstance(captured_request, api.PlaceLocateDebugRequest)
        self.assertEqual(captured_request.location_id, "H1_F_L_INSPECT")
        self.assertEqual(captured_request.product_name, "测试商品")

    def test_public_request_matches_inspect_and_captures_current_rgbd(self) -> None:
        required = set(api.PlaceLocateRequest.model_json_schema()["required"])
        self.assertEqual(
            required,
            {"task_type", "location_id", "pose_type", "product_name"},
        )
        properties = api.PlaceLocateRequest.model_json_schema()["properties"]
        self.assertIn("reference_item_area", properties)
        self.assertNotIn("current_image_base64", properties)
        self.assertNotIn("current_depth_image_base64", properties)
        self.assertNotIn("baseline_image_base64", properties)
        self.assertNotIn("baseline_depth_image_base64", properties)

        reference_required = set(
            api.PlaceLocateDebugRequest.model_json_schema()["required"]
        )
        self.assertIn("current_image_base64", reference_required)
        self.assertIn("current_depth_image_base64", reference_required)
        self.assertNotIn("reference_pose", reference_required)

    def test_uses_fixed_head_camera_intrinsics(self) -> None:
        native = api.head_camera_intrinsics(1280, 720)
        self.assertEqual(native.width, 1280)
        self.assertEqual(native.height, 720)
        self.assertAlmostEqual(native.fx, 910.744324)
        self.assertAlmostEqual(native.fy, 910.395020)
        self.assertAlmostEqual(native.cx, 650.132690)
        self.assertAlmostEqual(native.cy, 381.874634)

        half = api.head_camera_intrinsics(640, 360)
        self.assertAlmostEqual(half.fx, native.fx / 2)
        self.assertAlmostEqual(half.fy, native.fy / 2)
        self.assertAlmostEqual(half.cx, native.cx / 2)
        self.assertAlmostEqual(half.cy, native.cy / 2)

    def test_detected_row_maps_to_physical_shelf_level(self) -> None:
        upper_row = ShelfRow(index=2, bbox=(0, 100, 1280, 200), lower_rail_index=1)
        lower_row = ShelfRow(index=3, bbox=(0, 400, 1280, 200), lower_rail_index=None)

        self.assertEqual(
            api.resolve_shelf_level("SHELF_VIEW_UPPER", upper_row, "H1_F_L_INSPECT"),
            "L2",
        )
        self.assertEqual(
            api.resolve_shelf_level("SHELF_VIEW_LOWER", lower_row, "H1_F_L_INSPECT"),
            "L5",
        )

    def test_request_rejects_unknown_fields(self) -> None:
        payload = api.PlaceLocateRequest(
            task_type="SHORTAGE",
            product_name="测试商品",
            location_id="H1_F_L_INSPECT",
            pose_type="SHELF_VIEW_UPPER",
        ).model_dump()
        payload["reference_pose"] = {"matrix": np.eye(4).tolist()}
        with self.assertRaises(ValidationError):
            api.PlaceLocateRequest.model_validate(payload)

    def test_depth_must_be_aligned_to_rgb(self) -> None:
        with self.assertRaisesRegex(Exception, "must be aligned"):
            api.decode_depth_image(
                encode_depth(np.ones((10, 20), dtype=np.uint16)),
                "depth",
                (20, 10),
            )

    def test_reference_change_mask_is_limited_to_detected_shelf_row(self) -> None:
        component = np.zeros((100, 120), dtype=np.uint8)
        component[10:90, 40:80] = 255
        row = ShelfRow(index=2, bbox=(0, 35, 120, 30), lower_rail_index=1)

        constrained = api.constrain_mask_to_shelf_row(component, row)

        self.assertEqual(np.count_nonzero(constrained[:35]), 0)
        self.assertGreater(np.count_nonzero(constrained[35:65]), 0)
        self.assertEqual(np.count_nonzero(constrained[65:]), 0)


if __name__ == "__main__":
    unittest.main()
