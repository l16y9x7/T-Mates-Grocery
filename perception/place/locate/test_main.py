from __future__ import annotations

import base64
import io
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from pydantic import ValidationError

from . import main as api
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
    def make_request(self) -> api.PlaceLocateRequest:
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
        return api.PlaceLocateRequest(
            task_type="SHORTAGE",
            product_name="测试商品",
            location_id="H1_F_L2_C01",
            baseline_image_base64=encode_rgb(reference),
            baseline_depth_image_base64=encode_depth(reference_depth),
            current_image_base64=encode_rgb(current),
            current_depth_image_base64=encode_depth(current_depth),
            reference_pose=api.PoseInput(
                matrix=[
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 900],
                    [0, 0, 0, 1],
                ]
            ),
            camera_intrinsics=api.CameraIntrinsicsInput(
                fx=260,
                fy=260,
                cx=160,
                cy=120,
                width=width,
                height=height,
            ),
            reference_bbox=[115, 65, 90, 120],
        )

    def test_rgbd_route_returns_theoretical_current_mask(self) -> None:
        result = api.locate_place_debug(self.make_request())

        self.assertEqual(result.product_name, "测试商品")
        self.assertEqual(result.location_id, "H1_F_L2_C01")
        self.assertEqual(result.image_size, [320, 240])
        self.assertGreater(result.projected_point_count, 1000)
        self.assertGreater(result.registration.inlier_count, 12)
        self.assertEqual(result.target_pose.frame_id, "current_head_camera")
        expected_target_pose = (
            np.asarray(result.registration.current_from_reference)
            @ np.asarray(self.make_request().reference_pose.matrix)
        )
        self.assertTrue(
            np.allclose(
                np.asarray(result.target_pose.matrix),
                expected_target_pose,
                atol=1e-6,
            )
        )
        mask_bytes = base64.b64decode(result.mask)
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mask.shape, (240, 320))
        self.assertGreater(np.count_nonzero(mask), 1000)

        expected_depth = np.load(
            io.BytesIO(base64.b64decode(result.expected_depth_npy_base64)),
            allow_pickle=False,
        )
        self.assertEqual(expected_depth.shape, (240, 320))
        self.assertAlmostEqual(float(np.median(expected_depth[mask > 0])), 900, delta=5)

    def test_public_response_adds_transferred_pose_to_mask_fields(self) -> None:
        debug = api.PlaceLocateDebugResponse(
            product_name="测试商品",
            bbox=[100, 200, 300, 400],
            mask="mask",
            image_path="current.jpg",
            target_pose=api.PoseOutput(
                frame_id="current_head_camera",
                matrix=np.eye(4).tolist(),
            ),
            task_type="SHORTAGE",
            location_id="H1_F",
            region_index=1,
            image_size=[320, 240],
            reference_bbox=[1, 2, 3, 4],
            target_bbox_pixels=[5, 6, 7, 8],
            reference_mask="reference",
            visible_mask="visible",
            expected_depth_npy_base64="depth",
            projected_point_count=10,
            visible_point_count=9,
            registration=api.RegistrationMetrics(
                current_from_reference=np.eye(4).tolist(),
                rmse_mm=1,
                depth_correspondence_count=20,
                inlier_count=18,
                inlier_ratio=0.9,
                reprojection_rmse_px=0.5,
            ),
        )
        with patch.object(api, "locate_place_debug", return_value=debug):
            response = api.locate_place(self.make_request())

        self.assertEqual(
            response.model_dump(),
            {
                "product_name": "测试商品",
                "bbox": [100, 200, 300, 400],
                "mask": "mask",
                "image_path": "current.jpg",
                "target_pose": {
                    "frame_id": "current_head_camera",
                    "unit": "millimeter",
                    "matrix": np.eye(4).tolist(),
                },
            },
        )

    def test_request_requires_both_rgb_and_depth_pairs(self) -> None:
        required = set(api.PlaceLocateRequest.model_json_schema()["required"])
        self.assertTrue(
            {
                "baseline_image_base64",
                "baseline_depth_image_base64",
                "current_image_base64",
                "current_depth_image_base64",
                "reference_pose",
            }.issubset(required)
        )

    def test_request_rejects_unknown_fields(self) -> None:
        payload = self.make_request().model_dump()
        payload["baseline_depth_base64"] = "misspelled"
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
