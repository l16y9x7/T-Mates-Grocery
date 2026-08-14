"""Geometry primitives for the Place Locate RGB-D pose-transfer draft.

Matrix convention
-----------------
``target_from_source`` transforms a homogeneous column vector from ``source``
coordinates into ``target`` coordinates::

    point_target = target_from_source @ point_source

Translations are expected to use millimetres throughout this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class PoseTransferError(ValueError):
    """Raised when a pose or a registration input is not a valid rigid transform."""


@dataclass(frozen=True)
class RegistrationResult:
    """Rigid transform estimated from corresponding 3D points."""

    current_from_reference: np.ndarray
    rmse_mm: float
    correspondence_count: int


def as_rigid_transform(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str = "transform",
    atol: float = 1e-4,
) -> np.ndarray:
    """Return ``value`` as a validated 4x4 SE(3) matrix."""

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise PoseTransferError(f"{name} must be a 4x4 matrix")
    if not np.isfinite(matrix).all():
        raise PoseTransferError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise PoseTransferError(f"{name} has an invalid homogeneous bottom row")

    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise PoseTransferError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise PoseTransferError(f"{name} rotation determinant must be +1")
    return matrix


def invert_transform(
    target_from_source: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Invert a rigid transform without a general-purpose matrix inverse."""

    transform = as_rigid_transform(target_from_source, name="target_from_source")
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def transfer_reference_pose(
    current_from_reference: Sequence[Sequence[float]] | np.ndarray,
    reference_from_object: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Transfer a standard-scene object pose into the current camera frame."""

    current_from_reference_matrix = as_rigid_transform(
        current_from_reference,
        name="current_from_reference",
    )
    reference_from_object_matrix = as_rigid_transform(
        reference_from_object,
        name="reference_from_object",
    )
    return current_from_reference_matrix @ reference_from_object_matrix


def target_pose_in_robot_frame(
    robot_from_current: Sequence[Sequence[float]] | np.ndarray,
    current_from_reference: Sequence[Sequence[float]] | np.ndarray,
    reference_from_object: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Compose camera registration and camera extrinsics into a robot-frame pose."""

    robot_from_current_matrix = as_rigid_transform(
        robot_from_current,
        name="robot_from_current",
    )
    current_from_object = transfer_reference_pose(
        current_from_reference,
        reference_from_object,
    )
    return robot_from_current_matrix @ current_from_object


def transform_points(
    target_from_source: Sequence[Sequence[float]] | np.ndarray,
    points_source: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Transform an ``N x 3`` point array into the target coordinate frame."""

    transform = as_rigid_transform(target_from_source, name="target_from_source")
    points = np.asarray(points_source, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise PoseTransferError("points_source must have shape (N, 3)")
    if not np.isfinite(points).all():
        raise PoseTransferError("points_source contains non-finite values")
    return points @ transform[:3, :3].T + transform[:3, 3]


def estimate_rigid_transform(
    reference_points: Sequence[Sequence[float]] | np.ndarray,
    current_points: Sequence[Sequence[float]] | np.ndarray,
) -> RegistrationResult:
    """Estimate ``current_from_reference`` from matched 3D points using SVD.

    This is the closed-form registration core. Production code must find robust
    static-scene correspondences first and should wrap this estimator in RANSAC
    and/or refine the result with ICP.
    """

    reference = np.asarray(reference_points, dtype=np.float64)
    current = np.asarray(current_points, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise PoseTransferError("reference_points must have shape (N, 3)")
    if current.shape != reference.shape:
        raise PoseTransferError("current_points must match reference_points shape")
    if len(reference) < 3:
        raise PoseTransferError("at least three 3D correspondences are required")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise PoseTransferError("registration points contain non-finite values")

    reference_center = reference.mean(axis=0)
    current_center = current.mean(axis=0)
    reference_centered = reference - reference_center
    current_centered = current - current_center
    if np.linalg.matrix_rank(reference_centered) < 2:
        raise PoseTransferError("reference points are collinear or degenerate")

    covariance = reference_centered.T @ current_centered
    left_vectors, _, right_vectors_transposed = np.linalg.svd(covariance)
    rotation = right_vectors_transposed.T @ left_vectors.T
    if np.linalg.det(rotation) < 0:
        right_vectors_transposed[-1, :] *= -1
        rotation = right_vectors_transposed.T @ left_vectors.T

    translation = current_center - rotation @ reference_center
    current_from_reference = np.eye(4, dtype=np.float64)
    current_from_reference[:3, :3] = rotation
    current_from_reference[:3, 3] = translation
    current_from_reference = as_rigid_transform(
        current_from_reference,
        name="estimated current_from_reference",
    )

    predicted_current = transform_points(current_from_reference, reference)
    residuals = np.linalg.norm(predicted_current - current, axis=1)
    rmse_mm = float(np.sqrt(np.mean(np.square(residuals))))
    return RegistrationResult(
        current_from_reference=current_from_reference,
        rmse_mm=rmse_mm,
        correspondence_count=len(reference),
    )
