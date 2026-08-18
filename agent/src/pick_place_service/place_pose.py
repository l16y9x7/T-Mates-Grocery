"""Synthesize a target place pose from current-image reference objects."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

from pick_place_service.models import PoseResponse, ServiceError

ReferenceDirection = Literal["left", "right", "both", "up"]


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


def _validated_pose(pose: PoseResponse) -> tuple[np.ndarray, np.ndarray]:
    if len(pose.pose) != 6 or not all(math.isfinite(value) for value in pose.pose):
        raise ServiceError("INVALID_POSE", "reference place pose must contain six finite values")
    if pose.frame is not None and pose.frame != "camera":
        raise ServiceError("INVALID_POSE", "reference place pose frame must be camera")
    if pose.pose_unit is not None and pose.pose_unit != "mm_rad":
        raise ServiceError("INVALID_POSE", "reference place pose unit must be mm_rad")
    if pose.rotation_order is not None and pose.rotation_order.lower() != "zyx":
        raise ServiceError("INVALID_POSE", "reference place pose rotation order must be zyx")
    return (
        np.asarray(pose.pose[:3], dtype=np.float64),
        _zyx_matrix(*pose.pose[3:]),
    )


def _mean_rotation(rotations: Sequence[np.ndarray]) -> np.ndarray:
    combined = np.sum(np.stack(rotations), axis=0)
    left, _, right = np.linalg.svd(combined)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation


def synthesize_place_pose(
    reference_poses: Sequence[PoseResponse],
    direction: ReferenceDirection,
) -> PoseResponse:
    """Interpolate or extrapolate a target pose from left-to-right references."""

    if direction not in {"left", "right", "both", "up"}:
        raise ServiceError("INVALID_DIRECTION", f"unsupported place direction: {direction}")
    expected_count = 1 if direction == "up" else 2
    if len(reference_poses) != expected_count:
        raise ServiceError(
            "INVALID_REFERENCE_POSES",
            f"direction {direction} requires {expected_count} reference pose(s)",
        )
    validated = [_validated_pose(pose) for pose in reference_poses]
    translations = [item[0] for item in validated]
    rotations = [item[1] for item in validated]

    if direction == "up":
        target_translation = translations[0]
        target_rotation = rotations[0]
    else:
        left_translation, right_translation = translations
        if direction in {"left", "right"} and np.linalg.norm(
            right_translation - left_translation
        ) <= 1e-6:
            raise ServiceError(
                "INVALID_REFERENCE_POSES",
                "horizontal reference poses must have distinct translations",
            )
        if direction == "both":
            target_translation = (left_translation + right_translation) / 2.0
        elif direction == "left":
            target_translation = 2.0 * right_translation - left_translation
        else:
            target_translation = 2.0 * left_translation - right_translation
        target_rotation = _mean_rotation(rotations)

    rx, ry, rz = _matrix_to_zyx(target_rotation)
    target = [*target_translation.tolist(), rx, ry, rz]
    if not all(math.isfinite(value) for value in target):
        raise ServiceError("INVALID_POSE", "synthesized place pose contains non-finite values")
    return PoseResponse(
        pose=target,
        frame="camera",
        pose_unit="mm_rad",
        rotation_order="zyx",
    )
