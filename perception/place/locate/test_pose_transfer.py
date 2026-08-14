from __future__ import annotations

import unittest

import numpy as np

from .pose_transfer import (
    PoseTransferError,
    estimate_rigid_transform,
    target_pose_in_robot_frame,
    transfer_reference_pose,
    transform_points,
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


class PoseTransferTest(unittest.TestCase):
    def test_transfers_reference_pose_into_current_camera(self) -> None:
        current_from_reference = transform_z(90.0, (100.0, 20.0, -5.0))
        reference_from_object = transform_z(0.0, (10.0, 0.0, 300.0))

        result = transfer_reference_pose(
            current_from_reference,
            reference_from_object,
        )

        np.testing.assert_allclose(result[:3, 3], [100.0, 30.0, 295.0])

    def test_composes_robot_camera_extrinsics(self) -> None:
        robot_from_current = transform_z(0.0, (1000.0, 0.0, 0.0))
        current_from_reference = transform_z(0.0, (0.0, 20.0, 0.0))
        reference_from_object = transform_z(0.0, (10.0, 0.0, 300.0))

        result = target_pose_in_robot_frame(
            robot_from_current,
            current_from_reference,
            reference_from_object,
        )

        np.testing.assert_allclose(result[:3, 3], [1010.0, 20.0, 300.0])

    def test_recovers_rigid_transform_from_point_correspondences(self) -> None:
        reference_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [0.0, 80.0, 0.0],
                [20.0, 10.0, 50.0],
            ]
        )
        expected = transform_z(12.0, (45.0, -20.0, 8.0))
        current_points = transform_points(expected, reference_points)

        result = estimate_rigid_transform(reference_points, current_points)

        np.testing.assert_allclose(result.current_from_reference, expected, atol=1e-8)
        self.assertLess(result.rmse_mm, 1e-8)
        self.assertEqual(result.correspondence_count, 4)

    def test_rejects_a_homography_as_a_pose_transform(self) -> None:
        with self.assertRaises(PoseTransferError):
            transfer_reference_pose(np.eye(3), np.eye(4))


if __name__ == "__main__":
    unittest.main()
