"""RGB and RGB-D registration for reference-scene Place Locate.

The RGB stage follows the same ORB + homography idea used by inspection, but it
keeps the matched pixel pairs.  When aligned depth is available those pairs are
back-projected and a rigid ``current_from_reference`` SE(3) transform is
estimated.  The homography is only a correspondence filter and visualization
aid; it is never multiplied with a 6D pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from .pose_transfer import (
    PoseTransferError,
    RegistrationResult,
    estimate_rigid_transform,
    transform_points,
)


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera image dimensions must be positive")


@dataclass(frozen=True)
class RGBRegistrationConfig:
    orb_features: int = 8000
    match_ratio: float = 0.75
    ransac_reprojection_threshold_px: float = 4.0
    minimum_matches: int = 20
    minimum_inliers: int = 12
    minimum_static_matches: int = 12
    change_threshold_min: int = 24
    change_threshold_percentile: float = 95.0
    change_mask_dilation_px: int = 9
    border_exclusion_ratio: float = 0.015

    def __post_init__(self) -> None:
        if self.orb_features <= 0:
            raise ValueError("orb_features must be positive")
        if not 0 < self.match_ratio < 1:
            raise ValueError("match_ratio must be between 0 and 1")
        if self.minimum_matches < 4 or self.minimum_inliers < 4:
            raise ValueError("homography matching requires at least four points")
        if not 0 < self.change_threshold_percentile < 100:
            raise ValueError("change_threshold_percentile must be between 0 and 100")
        if self.change_mask_dilation_px < 0:
            raise ValueError("change_mask_dilation_px cannot be negative")


@dataclass
class RGBRegistrationResult:
    current_to_reference_homography: np.ndarray
    reference_to_current_homography: np.ndarray
    matched_current_pixels: np.ndarray
    matched_reference_pixels: np.ndarray
    final_inlier_mask: np.ndarray
    initial_match_count: int
    initial_inlier_count: int
    static_match_count: int
    final_inlier_count: int
    reprojection_rmse_px: float
    convex_hull_coverage_ratio: float
    grid_coverage_ratio: float
    difference_threshold: float
    aligned_current: np.ndarray = field(repr=False)
    change_mask_reference: np.ndarray = field(repr=False)
    static_mask_reference: np.ndarray = field(repr=False)

    @property
    def inlier_current_pixels(self) -> np.ndarray:
        return self.matched_current_pixels[self.final_inlier_mask]

    @property
    def inlier_reference_pixels(self) -> np.ndarray:
        return self.matched_reference_pixels[self.final_inlier_mask]

    def as_dict(self) -> dict[str, object]:
        return {
            "initial_match_count": self.initial_match_count,
            "initial_inlier_count": self.initial_inlier_count,
            "static_match_count": self.static_match_count,
            "final_inlier_count": self.final_inlier_count,
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "convex_hull_coverage_ratio": self.convex_hull_coverage_ratio,
            "grid_coverage_ratio": self.grid_coverage_ratio,
            "difference_threshold": self.difference_threshold,
            "current_to_reference_homography": (
                self.current_to_reference_homography.tolist()
            ),
            "reference_to_current_homography": (
                self.reference_to_current_homography.tolist()
            ),
        }


@dataclass
class RGBDRegistrationResult:
    rgb: RGBRegistrationResult
    current_from_reference: np.ndarray
    rmse_mm: float
    depth_correspondence_count: int
    inlier_count: int
    inlier_ratio: float
    reference_points_mm: np.ndarray = field(repr=False)
    current_points_mm: np.ndarray = field(repr=False)
    inlier_mask: np.ndarray = field(repr=False)

    def as_dict(self) -> dict[str, object]:
        return {
            "rgb": self.rgb.as_dict(),
            "current_from_reference": self.current_from_reference.tolist(),
            "rmse_mm": self.rmse_mm,
            "depth_correspondence_count": self.depth_correspondence_count,
            "inlier_count": self.inlier_count,
            "inlier_ratio": self.inlier_ratio,
        }


@dataclass
class ReprojectedMask:
    full_mask: np.ndarray = field(repr=False)
    visible_mask: np.ndarray = field(repr=False)
    expected_depth_mm: np.ndarray = field(repr=False)
    projected_point_count: int
    visible_point_count: int


def _validate_color_image(image: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError(f"{name} must be a uint8 BGR image")
    return value


def _gray_features(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _homography_inliers(
    current_pixels: np.ndarray,
    reference_pixels: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    homography, mask = cv2.findHomography(
        current_pixels.reshape(-1, 1, 2),
        reference_pixels.reshape(-1, 1, 2),
        cv2.RANSAC,
        threshold,
    )
    if homography is None or mask is None:
        raise PoseTransferError("homography estimation failed")
    return homography.astype(np.float64), mask.reshape(-1).astype(bool)


def _reprojection_errors(
    homography: np.ndarray,
    current_pixels: np.ndarray,
    reference_pixels: np.ndarray,
) -> np.ndarray:
    predicted = cv2.perspectiveTransform(
        current_pixels.reshape(-1, 1, 2).astype(np.float32),
        homography,
    ).reshape(-1, 2)
    return np.linalg.norm(predicted - reference_pixels, axis=1)


def _match_spatial_coverage(
    reference_pixels: np.ndarray,
    image_size: tuple[int, int],
    *,
    grid_columns: int = 4,
    grid_rows: int = 3,
) -> tuple[float, float]:
    width, height = image_size
    if len(reference_pixels) < 3:
        return 0.0, 0.0
    hull = cv2.convexHull(reference_pixels.astype(np.float32))
    hull_ratio = float(cv2.contourArea(hull) / (width * height))
    columns = np.clip(
        (reference_pixels[:, 0] / width * grid_columns).astype(np.int32),
        0,
        grid_columns - 1,
    )
    rows = np.clip(
        (reference_pixels[:, 1] / height * grid_rows).astype(np.int32),
        0,
        grid_rows - 1,
    )
    occupied = len(set(zip(columns.tolist(), rows.tolist())))
    grid_ratio = occupied / (grid_columns * grid_rows)
    return hull_ratio, grid_ratio


def _build_change_mask(
    reference: np.ndarray,
    aligned_current: np.ndarray,
    valid_overlap: np.ndarray,
    config: RGBRegistrationConfig,
    reference_exclusion_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    reference_gray = cv2.GaussianBlur(_gray_features(reference), (5, 5), 0)
    current_gray = cv2.GaussianBlur(_gray_features(aligned_current), (5, 5), 0)
    difference = cv2.absdiff(reference_gray, current_gray)
    valid_values = difference[valid_overlap > 0]
    percentile_threshold = (
        float(np.percentile(valid_values, config.change_threshold_percentile))
        if valid_values.size
        else 0.0
    )
    threshold = max(float(config.change_threshold_min), percentile_threshold)
    change_mask = np.where(difference >= threshold, 255, 0).astype(np.uint8)
    change_mask[valid_overlap == 0] = 255

    if reference_exclusion_mask is not None:
        exclusion = np.asarray(reference_exclusion_mask)
        if exclusion.shape != change_mask.shape:
            raise ValueError("reference_exclusion_mask must match reference image size")
        change_mask[exclusion > 0] = 255

    dilation = config.change_mask_dilation_px
    if dilation > 0:
        kernel_size = dilation * 2 + 1
        change_mask = cv2.dilate(
            change_mask,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1,
        )
    static_mask = np.where(change_mask == 0, 255, 0).astype(np.uint8)
    return change_mask, static_mask, threshold


def register_rgb_images(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    *,
    config: RGBRegistrationConfig | None = None,
    reference_exclusion_mask: np.ndarray | None = None,
) -> RGBRegistrationResult:
    """Register current RGB to the reference and retain static keypoint pairs."""

    settings = config or RGBRegistrationConfig()
    reference = _validate_color_image(reference_image, "reference_image")
    current = _validate_color_image(current_image, "current_image")
    height, width = reference.shape[:2]
    current_height, current_width = current.shape[:2]

    orb = cv2.ORB_create(nfeatures=settings.orb_features)
    reference_keypoints, reference_descriptors = orb.detectAndCompute(
        _gray_features(reference),
        None,
    )
    current_keypoints, current_descriptors = orb.detectAndCompute(
        _gray_features(current),
        None,
    )
    if reference_descriptors is None or current_descriptors is None:
        raise PoseTransferError("not enough RGB features for registration")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(current_descriptors, reference_descriptors, k=2)
    good_matches = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < settings.match_ratio * second.distance
    ]
    if len(good_matches) < settings.minimum_matches:
        raise PoseTransferError(
            f"not enough reliable RGB matches: {len(good_matches)}"
        )

    current_pixels = np.float32(
        [current_keypoints[match.queryIdx].pt for match in good_matches]
    )
    reference_pixels = np.float32(
        [reference_keypoints[match.trainIdx].pt for match in good_matches]
    )
    initial_homography, initial_inliers = _homography_inliers(
        current_pixels,
        reference_pixels,
        settings.ransac_reprojection_threshold_px,
    )
    initial_inlier_count = int(initial_inliers.sum())
    if initial_inlier_count < settings.minimum_inliers:
        raise PoseTransferError(
            f"not enough homography inliers: {initial_inlier_count}"
        )

    aligned_current = cv2.warpPerspective(
        current,
        initial_homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid_overlap = cv2.warpPerspective(
        np.full((current_height, current_width), 255, dtype=np.uint8),
        initial_homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    border = round(min(height, width) * settings.border_exclusion_ratio)
    if border > 0:
        valid_overlap[:border, :] = 0
        valid_overlap[-border:, :] = 0
        valid_overlap[:, :border] = 0
        valid_overlap[:, -border:] = 0

    change_mask, static_mask, difference_threshold = _build_change_mask(
        reference,
        aligned_current,
        valid_overlap,
        settings,
        reference_exclusion_mask,
    )
    rounded_reference = np.rint(reference_pixels).astype(np.int32)
    rounded_reference[:, 0] = np.clip(rounded_reference[:, 0], 0, width - 1)
    rounded_reference[:, 1] = np.clip(rounded_reference[:, 1], 0, height - 1)
    static_matches = initial_inliers & (
        static_mask[rounded_reference[:, 1], rounded_reference[:, 0]] > 0
    )
    static_match_count = int(static_matches.sum())
    if static_match_count < settings.minimum_static_matches:
        raise PoseTransferError(
            f"not enough static RGB matches after change filtering: {static_match_count}"
        )

    static_indices = np.flatnonzero(static_matches)
    refined_homography, refined_local_inliers = _homography_inliers(
        current_pixels[static_indices],
        reference_pixels[static_indices],
        settings.ransac_reprojection_threshold_px,
    )
    final_inliers = np.zeros(len(good_matches), dtype=bool)
    final_inliers[static_indices[refined_local_inliers]] = True
    final_inlier_count = int(final_inliers.sum())
    if final_inlier_count < settings.minimum_inliers:
        raise PoseTransferError(
            f"not enough refined homography inliers: {final_inlier_count}"
        )

    aligned_current = cv2.warpPerspective(
        current,
        refined_homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    errors = _reprojection_errors(
        refined_homography,
        current_pixels[final_inliers],
        reference_pixels[final_inliers],
    )
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    hull_coverage, grid_coverage = _match_spatial_coverage(
        reference_pixels[final_inliers],
        (width, height),
    )
    try:
        inverse_homography = np.linalg.inv(refined_homography)
    except np.linalg.LinAlgError as error:
        raise PoseTransferError("refined homography is singular") from error

    return RGBRegistrationResult(
        current_to_reference_homography=refined_homography,
        reference_to_current_homography=inverse_homography,
        matched_current_pixels=current_pixels,
        matched_reference_pixels=reference_pixels,
        final_inlier_mask=final_inliers,
        initial_match_count=len(good_matches),
        initial_inlier_count=initial_inlier_count,
        static_match_count=static_match_count,
        final_inlier_count=final_inlier_count,
        reprojection_rmse_px=rmse,
        convex_hull_coverage_ratio=hull_coverage,
        grid_coverage_ratio=grid_coverage,
        difference_threshold=difference_threshold,
        aligned_current=aligned_current,
        change_mask_reference=change_mask,
        static_mask_reference=static_mask,
    )


def _sample_depth_mm(
    depth_mm: np.ndarray,
    pixels: np.ndarray,
    *,
    radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError("depth image must be a two-dimensional array")
    height, width = depth.shape
    samples = np.zeros(len(pixels), dtype=np.float64)
    valid = np.zeros(len(pixels), dtype=bool)
    for index, (u_value, v_value) in enumerate(pixels):
        u = int(round(float(u_value)))
        v = int(round(float(v_value)))
        if not 0 <= u < width or not 0 <= v < height:
            continue
        x0 = max(0, u - radius)
        x1 = min(width, u + radius + 1)
        y0 = max(0, v - radius)
        y1 = min(height, v + radius + 1)
        neighborhood = depth[y0:y1, x0:x1].astype(np.float64)
        neighborhood = neighborhood[np.isfinite(neighborhood) & (neighborhood > 0)]
        if neighborhood.size:
            samples[index] = float(np.median(neighborhood))
            valid[index] = True
    return samples, valid


def backproject_pixels(
    pixels: Sequence[Sequence[float]] | np.ndarray,
    depths_mm: Sequence[float] | np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Back-project RGB pixels and aligned millimetre depths to camera XYZ."""

    uv = np.asarray(pixels, dtype=np.float64)
    depths = np.asarray(depths_mm, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("pixels must have shape (N, 2)")
    if depths.shape != (len(uv),):
        raise ValueError("depths_mm must have one value per pixel")
    if not np.isfinite(uv).all() or not np.isfinite(depths).all():
        raise ValueError("pixels and depths_mm must be finite")
    if np.any(depths <= 0):
        raise ValueError("depths_mm must be positive")
    x = (uv[:, 0] - intrinsics.cx) * depths / intrinsics.fx
    y = (uv[:, 1] - intrinsics.cy) * depths / intrinsics.fy
    return np.column_stack((x, y, depths))


def project_points(
    points_camera_mm: Sequence[Sequence[float]] | np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera XYZ points to pixels and return a positive-depth mask."""

    points = np.asarray(points_camera_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera_mm must have shape (N, 3)")
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    z = points[valid, 2]
    pixels[valid, 0] = intrinsics.fx * points[valid, 0] / z + intrinsics.cx
    pixels[valid, 1] = intrinsics.fy * points[valid, 1] / z + intrinsics.cy
    return pixels, valid


def estimate_rigid_transform_ransac(
    reference_points_mm: Sequence[Sequence[float]] | np.ndarray,
    current_points_mm: Sequence[Sequence[float]] | np.ndarray,
    *,
    residual_threshold_mm: float = 20.0,
    iterations: int = 1000,
    minimum_inliers: int = 12,
    random_seed: int = 0,
) -> tuple[RegistrationResult, np.ndarray]:
    """Robustly estimate ``current_from_reference`` from 3D correspondences."""

    reference = np.asarray(reference_points_mm, dtype=np.float64)
    current = np.asarray(current_points_mm, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise PoseTransferError("reference_points_mm must have shape (N, 3)")
    if current.shape != reference.shape:
        raise PoseTransferError("current_points_mm must match reference points")
    if len(reference) < max(3, minimum_inliers):
        raise PoseTransferError("not enough 3D correspondences for RANSAC")
    if residual_threshold_mm <= 0 or iterations <= 0:
        raise ValueError("RANSAC threshold and iterations must be positive")

    generator = np.random.default_rng(random_seed)
    best_inliers: np.ndarray | None = None
    best_rmse = float("inf")
    for _ in range(iterations):
        sample_indices = generator.choice(len(reference), size=3, replace=False)
        try:
            candidate = estimate_rigid_transform(
                reference[sample_indices],
                current[sample_indices],
            )
        except PoseTransferError:
            continue
        predicted = transform_points(candidate.current_from_reference, reference)
        residuals = np.linalg.norm(predicted - current, axis=1)
        inliers = residuals <= residual_threshold_mm
        inlier_count = int(inliers.sum())
        if inlier_count < 3:
            continue
        inlier_rmse = float(
            np.sqrt(np.mean(np.square(residuals[inliers])))
        )
        if (
            best_inliers is None
            or inlier_count > int(best_inliers.sum())
            or (inlier_count == int(best_inliers.sum()) and inlier_rmse < best_rmse)
        ):
            best_inliers = inliers
            best_rmse = inlier_rmse

    if best_inliers is None or int(best_inliers.sum()) < minimum_inliers:
        count = 0 if best_inliers is None else int(best_inliers.sum())
        raise PoseTransferError(
            f"3D RANSAC has only {count} inliers; need {minimum_inliers}"
        )
    refined = estimate_rigid_transform(
        reference[best_inliers],
        current[best_inliers],
    )
    predicted = transform_points(refined.current_from_reference, reference)
    residuals = np.linalg.norm(predicted - current, axis=1)
    final_inliers = residuals <= residual_threshold_mm
    if int(final_inliers.sum()) < minimum_inliers:
        raise PoseTransferError("refined 3D transform lost too many inliers")
    final = estimate_rigid_transform(
        reference[final_inliers],
        current[final_inliers],
    )
    return final, final_inliers


def register_rgbd_images(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    reference_intrinsics: CameraIntrinsics,
    current_intrinsics: CameraIntrinsics | None = None,
    *,
    rgb_config: RGBRegistrationConfig | None = None,
    reference_exclusion_mask: np.ndarray | None = None,
    depth_sample_radius: int = 2,
    ransac_residual_threshold_mm: float = 20.0,
    ransac_iterations: int = 1000,
    minimum_depth_correspondences: int = 12,
) -> RGBDRegistrationResult:
    """Estimate the reference-to-current camera SE(3) transform from RGB-D."""

    current_camera = current_intrinsics or reference_intrinsics
    rgb_result = register_rgb_images(
        reference_image,
        current_image,
        config=rgb_config,
        reference_exclusion_mask=reference_exclusion_mask,
    )
    if reference_depth_mm.shape != reference_image.shape[:2]:
        raise ValueError("reference depth must be aligned to reference RGB")
    if current_depth_mm.shape != current_image.shape[:2]:
        raise ValueError("current depth must be aligned to current RGB")

    reference_pixels = rgb_result.inlier_reference_pixels
    current_pixels = rgb_result.inlier_current_pixels
    reference_depths, reference_valid = _sample_depth_mm(
        reference_depth_mm,
        reference_pixels,
        radius=depth_sample_radius,
    )
    current_depths, current_valid = _sample_depth_mm(
        current_depth_mm,
        current_pixels,
        radius=depth_sample_radius,
    )
    valid = reference_valid & current_valid
    if int(valid.sum()) < minimum_depth_correspondences:
        raise PoseTransferError(
            "not enough RGB matches have valid aligned depth: "
            f"{int(valid.sum())}"
        )

    reference_points = backproject_pixels(
        reference_pixels[valid],
        reference_depths[valid],
        reference_intrinsics,
    )
    current_points = backproject_pixels(
        current_pixels[valid],
        current_depths[valid],
        current_camera,
    )
    registration, inliers = estimate_rigid_transform_ransac(
        reference_points,
        current_points,
        residual_threshold_mm=ransac_residual_threshold_mm,
        iterations=ransac_iterations,
        minimum_inliers=minimum_depth_correspondences,
    )
    inlier_count = int(inliers.sum())
    return RGBDRegistrationResult(
        rgb=rgb_result,
        current_from_reference=registration.current_from_reference,
        rmse_mm=registration.rmse_mm,
        depth_correspondence_count=len(reference_points),
        inlier_count=inlier_count,
        inlier_ratio=inlier_count / len(reference_points),
        reference_points_mm=reference_points,
        current_points_mm=current_points,
        inlier_mask=inliers,
    )


def reproject_reference_mask(
    reference_mask: np.ndarray,
    reference_depth_mm: np.ndarray,
    current_from_reference: Sequence[Sequence[float]] | np.ndarray,
    reference_intrinsics: CameraIntrinsics,
    current_intrinsics: CameraIntrinsics | None = None,
    *,
    current_depth_mm: np.ndarray | None = None,
    occlusion_tolerance_mm: float = 20.0,
    splat_radius_px: int = 1,
) -> ReprojectedMask:
    """Project a real reference object mask into the current camera view.

    The returned full mask is the expected silhouette.  The visible mask removes
    pixels for which current depth contains a closer obstacle.
    """

    current_camera = current_intrinsics or reference_intrinsics
    mask = np.asarray(reference_mask)
    depth = np.asarray(reference_depth_mm)
    if mask.shape != depth.shape:
        raise ValueError("reference mask and depth must have the same shape")
    if current_depth_mm is not None and current_depth_mm.shape != (
        current_camera.height,
        current_camera.width,
    ):
        raise ValueError("current depth shape does not match current intrinsics")
    ys, xs = np.where((mask > 0) & np.isfinite(depth) & (depth > 0))
    if not len(xs):
        raise PoseTransferError("reference target mask has no valid depth pixels")
    reference_pixels = np.column_stack((xs, ys)).astype(np.float64)
    reference_points = backproject_pixels(
        reference_pixels,
        depth[ys, xs].astype(np.float64),
        reference_intrinsics,
    )
    current_points = transform_points(current_from_reference, reference_points)
    current_pixels, positive_depth = project_points(current_points, current_camera)
    rounded = np.rint(current_pixels[positive_depth]).astype(np.int32)
    projected_depths = current_points[positive_depth, 2]
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < current_camera.width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < current_camera.height)
    )
    rounded = rounded[inside]
    projected_depths = projected_depths[inside]
    if not len(rounded):
        raise PoseTransferError("transformed target mask is outside the current image")

    expected_depth_flat = np.full(
        current_camera.height * current_camera.width,
        np.inf,
        dtype=np.float32,
    )
    offsets = [(0, 0)]
    if splat_radius_px > 0:
        offsets = [
            (offset_x, offset_y)
            for offset_y in range(-splat_radius_px, splat_radius_px + 1)
            for offset_x in range(-splat_radius_px, splat_radius_px + 1)
        ]
    for offset_x, offset_y in offsets:
        splat_x = rounded[:, 0] + offset_x
        splat_y = rounded[:, 1] + offset_y
        splat_inside = (
            (splat_x >= 0)
            & (splat_x < current_camera.width)
            & (splat_y >= 0)
            & (splat_y < current_camera.height)
        )
        splat_x = splat_x[splat_inside]
        splat_y = splat_y[splat_inside]
        splat_depth = projected_depths[splat_inside]
        # Multiple reference pixels may project to the same current pixel.  Keep
        # the closest transformed surface, which is the normal z-buffer rule.
        flat_indices = splat_y * current_camera.width + splat_x
        np.minimum.at(expected_depth_flat, flat_indices, splat_depth)

    expected_depth_flat[~np.isfinite(expected_depth_flat)] = 0
    expected_depth = expected_depth_flat.reshape(
        current_camera.height,
        current_camera.width,
    )

    full = np.where(expected_depth > 0, 255, 0).astype(np.uint8)
    visible = np.zeros_like(full)
    projected_y, projected_x = np.where(expected_depth > 0)
    predicted = expected_depth[projected_y, projected_x].astype(np.float64)
    visible_points = np.ones(len(projected_x), dtype=bool)
    if current_depth_mm is not None:
        observed = current_depth_mm[projected_y, projected_x].astype(np.float64)
        observed_valid = np.isfinite(observed) & (observed > 0)
        visible_points[observed_valid] = (
            predicted[observed_valid]
            <= observed[observed_valid] + occlusion_tolerance_mm
        )
    visible[projected_y[visible_points], projected_x[visible_points]] = 255
    return ReprojectedMask(
        full_mask=full,
        visible_mask=visible,
        expected_depth_mm=expected_depth,
        projected_point_count=len(projected_x),
        visible_point_count=int(visible_points.sum()),
    )


def draw_registration_matches(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    result: RGBRegistrationResult,
    *,
    maximum_matches: int = 150,
) -> np.ndarray:
    """Draw final static inlier correspondences for offline diagnostics."""

    reference = _validate_color_image(reference_image, "reference_image")
    current = _validate_color_image(current_image, "current_image")
    height = max(reference.shape[0], current.shape[0])
    width = reference.shape[1] + current.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: reference.shape[0], : reference.shape[1]] = reference
    canvas[: current.shape[0], reference.shape[1] :] = current
    reference_points = result.inlier_reference_pixels
    current_points = result.inlier_current_pixels
    if len(reference_points) > maximum_matches:
        indices = np.linspace(
            0,
            len(reference_points) - 1,
            maximum_matches,
            dtype=np.int32,
        )
        reference_points = reference_points[indices]
        current_points = current_points[indices]
    for reference_point, current_point in zip(reference_points, current_points):
        reference_xy = tuple(np.rint(reference_point).astype(int))
        current_xy = tuple(
            np.rint(current_point).astype(int) + np.array([reference.shape[1], 0])
        )
        cv2.circle(canvas, reference_xy, 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, current_xy, 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.line(canvas, reference_xy, current_xy, (0, 180, 255), 1, cv2.LINE_AA)
    return canvas
