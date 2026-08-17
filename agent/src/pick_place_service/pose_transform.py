"""Convert reference-camera 6D poses into the current camera frame."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pick_place_service.models import PoseResponse, ServiceError


def _rigid_transform(value: Sequence[Sequence[float]]) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ServiceError(
            "INVALID_ROTATE_MATRIX", "rotate_matrix must contain numeric values"
        ) from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ServiceError(
            "INVALID_ROTATE_MATRIX", "rotate_matrix must be a finite 4x4 matrix"
        )
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        raise ServiceError(
            "INVALID_ROTATE_MATRIX", "rotate_matrix has an invalid homogeneous row"
        )
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ServiceError(
            "INVALID_ROTATE_MATRIX", "rotate_matrix rotation is not orthonormal"
        )
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ServiceError(
            "INVALID_ROTATE_MATRIX", "rotate_matrix rotation determinant must be +1"
        )
    return matrix


def _zyx_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    return np.asarray(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


def _matrix_to_zyx(rotation: np.ndarray) -> tuple[float, float, float]:
    horizontal = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if horizontal > 1e-9:
        rx = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        ry = math.atan2(-float(rotation[2, 0]), horizontal)
        rz = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        rx = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        ry = math.atan2(-float(rotation[2, 0]), horizontal)
        rz = 0.0
    return rx, ry, rz


def transfer_reference_pose(
    pose: PoseResponse, rotate_matrix: Sequence[Sequence[float]]
) -> PoseResponse:
    """Apply ``current_from_reference`` to an mm/rad ZYX camera pose."""

    if len(pose.pose) != 6 or not all(math.isfinite(value) for value in pose.pose):
        raise ServiceError("INVALID_POSE", "place pose must contain six finite values")
    if (pose.frame or "camera") != "camera":
        raise ServiceError("INVALID_POSE", "place pose frame must be camera")
    if (pose.pose_unit or "mm_rad") != "mm_rad":
        raise ServiceError("INVALID_POSE", "place pose unit must be mm_rad")
    if (pose.rotation_order or "zyx").lower() != "zyx":
        raise ServiceError("INVALID_POSE", "place pose rotation order must be zyx")

    reference_from_object = np.eye(4, dtype=np.float64)
    reference_from_object[:3, 3] = pose.pose[:3]
    reference_from_object[:3, :3] = _zyx_matrix(*pose.pose[3:])
    current_from_object = _rigid_transform(rotate_matrix) @ reference_from_object
    rx, ry, rz = _matrix_to_zyx(current_from_object[:3, :3])
    transformed = [
        float(current_from_object[0, 3]),
        float(current_from_object[1, 3]),
        float(current_from_object[2, 3]),
        rx,
        ry,
        rz,
    ]
    return PoseResponse(
        pose=transformed,
        frame="camera",
        pose_unit="mm_rad",
        rotation_order="zyx",
    )
