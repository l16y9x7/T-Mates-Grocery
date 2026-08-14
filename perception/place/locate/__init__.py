"""RGB-D reference-scene pose transfer for Place Locate."""

from .pose_transfer import (
    PoseTransferError,
    RegistrationResult,
    as_rigid_transform,
    estimate_rigid_transform,
    invert_transform,
    target_pose_in_robot_frame,
    transfer_reference_pose,
    transform_points,
)
from .registration import (
    CameraIntrinsics,
    RGBDRegistrationResult,
    RGBRegistrationConfig,
    RGBRegistrationResult,
    ReprojectedMask,
    backproject_pixels,
    draw_registration_matches,
    estimate_rigid_transform_ransac,
    project_points,
    register_rgb_images,
    register_rgbd_images,
    reproject_reference_mask,
)

__all__ = [
    "PoseTransferError",
    "RegistrationResult",
    "as_rigid_transform",
    "estimate_rigid_transform",
    "invert_transform",
    "target_pose_in_robot_frame",
    "transfer_reference_pose",
    "transform_points",
    "CameraIntrinsics",
    "RGBDRegistrationResult",
    "RGBRegistrationConfig",
    "RGBRegistrationResult",
    "ReprojectedMask",
    "backproject_pixels",
    "draw_registration_matches",
    "estimate_rigid_transform_ransac",
    "project_points",
    "register_rgb_images",
    "register_rgbd_images",
    "reproject_reference_mask",
]
