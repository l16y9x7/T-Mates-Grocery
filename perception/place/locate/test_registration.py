from __future__ import annotations

import unittest

import cv2
import numpy as np

from .pose_transfer import transform_points
from .registration import (
    CameraIntrinsics,
    RGBRegistrationConfig,
    backproject_pixels,
    estimate_rigid_transform_ransac,
    project_points,
    register_rgb_images,
    reproject_reference_mask,
)


def transform_z(rotation_degrees: float, translation: tuple[float, float, float]) -> np.ndarray:
    radians = np.deg2rad(rotation_degrees)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    result[:3, 3] = translation
    return result


class RegistrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(
            fx=600.0,
            fy=600.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
        )

    def test_pixel_backprojection_round_trip(self) -> None:
        pixels = np.array([[320.0, 240.0], [380.0, 210.0]])
        depths = np.array([1000.0, 1200.0])

        points = backproject_pixels(pixels, depths, self.intrinsics)
        projected, valid = project_points(points, self.intrinsics)

        self.assertTrue(valid.all())
        np.testing.assert_allclose(projected, pixels)

    def test_ransac_rejects_bad_3d_correspondences(self) -> None:
        generator = np.random.default_rng(12)
        reference = generator.uniform(-200.0, 200.0, size=(80, 3))
        reference[:, 2] += 1000.0
        expected = transform_z(7.0, (35.0, -12.0, 18.0))
        current = transform_points(expected, reference)
        current += generator.normal(0.0, 0.5, size=current.shape)
        current[:15] = generator.uniform(-500.0, 500.0, size=(15, 3))

        result, inliers = estimate_rigid_transform_ransac(
            reference,
            current,
            residual_threshold_mm=3.0,
            minimum_inliers=40,
        )

        self.assertGreaterEqual(int(inliers.sum()), 60)
        np.testing.assert_allclose(
            result.current_from_reference,
            expected,
            atol=0.5,
        )
        self.assertLess(result.rmse_mm, 1.0)

    def test_identity_mask_reprojection(self) -> None:
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[180:260, 280:360] = 255
        depth = np.full(mask.shape, 1000, dtype=np.uint16)

        result = reproject_reference_mask(
            mask,
            depth,
            np.eye(4),
            self.intrinsics,
            splat_radius_px=0,
        )

        np.testing.assert_array_equal(result.full_mask, mask)
        np.testing.assert_array_equal(result.visible_mask, mask)
        np.testing.assert_array_equal(
            result.expected_depth_mm[mask > 0],
            np.full(np.count_nonzero(mask), 1000, dtype=np.float32),
        )
        self.assertTrue(np.all(result.expected_depth_mm[mask == 0] == 0))

    def test_current_depth_only_filters_visibility(self) -> None:
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[180:260, 280:360] = 255
        reference_depth = np.full(mask.shape, 1000, dtype=np.uint16)
        current_depth = np.full(mask.shape, 1200, dtype=np.uint16)
        current_depth[180:260, 280:320] = 800

        result = reproject_reference_mask(
            mask,
            reference_depth,
            np.eye(4),
            self.intrinsics,
            current_depth_mm=current_depth,
            occlusion_tolerance_mm=20.0,
            splat_radius_px=0,
        )

        self.assertTrue(np.all(result.full_mask[180:260, 280:360] == 255))
        self.assertTrue(np.all(result.visible_mask[180:260, 280:320] == 0))
        self.assertTrue(np.all(result.visible_mask[180:260, 320:360] == 255))
        self.assertTrue(
            np.all(result.expected_depth_mm[180:260, 280:360] == 1000)
        )

    def test_rgb_registration_filters_a_changed_foreground_patch(self) -> None:
        generator = np.random.default_rng(4)
        reference = generator.integers(0, 256, size=(360, 480, 3), dtype=np.uint8)
        reference = cv2.GaussianBlur(reference, (5, 5), 0)
        current_to_reference = np.float32([[1.0, 0.0, 8.0], [0.0, 1.0, -6.0]])
        current = cv2.warpAffine(
            reference,
            cv2.invertAffineTransform(current_to_reference),
            (480, 360),
        )
        current[130:230, 190:290] = (0, 0, 255)

        result = register_rgb_images(
            reference,
            current,
            config=RGBRegistrationConfig(
                minimum_matches=12,
                minimum_inliers=8,
                minimum_static_matches=8,
                change_mask_dilation_px=5,
            ),
        )

        self.assertGreater(result.final_inlier_count, 20)
        self.assertLess(result.reprojection_rmse_px, 1.5)
        self.assertGreater(int(np.count_nonzero(result.change_mask_reference)), 0)

    def test_rgb_registration_accepts_different_image_resolutions(self) -> None:
        generator = np.random.default_rng(9)
        reference = generator.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
        reference = cv2.GaussianBlur(reference, (5, 5), 0)
        current = cv2.resize(reference, (800, 600), interpolation=cv2.INTER_CUBIC)

        result = register_rgb_images(
            reference,
            current,
            config=RGBRegistrationConfig(
                minimum_matches=12,
                minimum_inliers=8,
                minimum_static_matches=8,
                change_mask_dilation_px=3,
            ),
        )

        self.assertGreater(result.final_inlier_count, 20)
        homography = result.current_to_reference_homography
        np.testing.assert_allclose(
            [homography[0, 0], homography[1, 1]],
            [0.5, 0.5],
            atol=0.01,
        )
        self.assertLess(abs(float(homography[0, 1])), 0.01)
        self.assertLess(abs(float(homography[1, 0])), 0.01)
        self.assertLess(abs(float(homography[0, 2])), 0.5)
        self.assertLess(abs(float(homography[1, 2])), 0.5)


if __name__ == "__main__":
    unittest.main()
