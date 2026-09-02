"""CPU-only 2D and 3D visualization for compact pose service responses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw

from ui.pointcloud import (
    camera_points_m_to_glb,
    camera_pose_m_to_glb,
    create_pose_axes_mesh,
    depth_rgb_to_pointcloud,
    export_scene_files,
    inject_axis_points,
)


CUBOID_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


@dataclass(frozen=True)
class SceneArtifacts:
    """Files and point count produced by one point-cloud render."""

    glb_path: Path
    ply_path: Path
    num_points: int


def euler_zyx_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert response Euler angles to Rz @ Ry @ Rx rotation."""

    sin_x, cos_x = math.sin(rx), math.cos(rx)
    sin_y, cos_y = math.sin(ry), math.cos(ry)
    sin_z, cos_z = math.sin(rz), math.cos(rz)
    rotation_x = np.array(
        [[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]],
        dtype=np.float64,
    )
    rotation_y = np.array(
        [[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return rotation_z @ rotation_y @ rotation_x


def project_camera_points(
    points_m: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """Project camera-frame meter points, leaving invalid depth as NaN."""

    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-9)
    projected = (np.asarray(intrinsics, dtype=np.float64) @ points[valid].T).T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels


def render_box(image: Image.Image, box: Sequence[int]) -> Image.Image:
    """Render the user box on an RGB copy."""

    output = image.convert("RGB").copy()
    ImageDraw.Draw(output).rectangle(tuple(box), outline=(255, 190, 0), width=3)
    return output


def render_mask(mask: np.ndarray) -> Image.Image:
    """Render a boolean mask as a single-channel image."""

    array = np.asarray(mask, dtype=bool)
    return Image.fromarray((array * 255).astype(np.uint8), mode="L")


def render_mask_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Blend the mask in green over RGB."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    foreground = np.asarray(mask, dtype=bool)
    if foreground.shape != rgb.shape[:2]:
        foreground = np.asarray(
            Image.fromarray(foreground.astype(np.uint8)).resize(
                image.size, Image.Resampling.NEAREST
            ),
            dtype=bool,
        )
    green = np.array([0, 255, 80], dtype=np.float32)
    rgb[foreground] = (
        0.5 * rgb[foreground].astype(np.float32) + 0.5 * green
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def render_depth_colormap(depth: np.ndarray) -> Image.Image:
    """Create a robust Turbo colormap while keeping zero depth black."""

    values = np.asarray(depth)
    valid = np.isfinite(values) & (values > 0)
    normalized = np.zeros(values.shape, dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(values[valid].astype(np.float64), [2, 98])
        if high <= low:
            high = low + 1.0
        normalized[valid] = np.clip(
            (values[valid] - low) / (high - low) * 255.0,
            0,
            255,
        ).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    colored_rgb[~valid] = 0
    return Image.fromarray(colored_rgb, mode="RGB")


def render_pose_overlay(
    image: Image.Image,
    response: Dict[str, Any],
    intrinsics: np.ndarray,
    *,
    axis_length_m: float = 0.08,
) -> Image.Image:
    """Draw all cuboid edges and RGB pose axes on the source image."""

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    corners_m = np.asarray(response["corners_mm"], dtype=np.float64) / 1000.0
    corner_pixels = project_camera_points(corners_m, intrinsics)
    for first, second in CUBOID_EDGES:
        first_pixel = corner_pixels[first]
        second_pixel = corner_pixels[second]
        if np.isfinite(first_pixel).all() and np.isfinite(second_pixel).all():
            draw.line(
                [tuple(first_pixel), tuple(second_pixel)],
                fill=(255, 220, 0),
                width=3,
            )

    pose = np.asarray(response["pose"], dtype=np.float64)
    origin = pose[:3] / 1000.0
    rotation = euler_zyx_matrix(*pose[3:])
    endpoints = np.vstack(
        [origin]
        + [origin + rotation[:, index] * axis_length_m for index in range(3)]
    )
    pixels = project_camera_points(endpoints, intrinsics)
    for index, color in enumerate(
        ((255, 0, 0), (0, 255, 0), (0, 96, 255))
    ):
        pair = pixels[[0, index + 1]]
        if np.isfinite(pair).all():
            draw.line([tuple(pair[0]), tuple(pair[1])], fill=color, width=4)
    if np.isfinite(pixels[0]).all():
        x, y = pixels[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255))
    return output


def create_cuboid_mesh(
    corners_glb: np.ndarray,
    edges: Sequence[Tuple[int, int]] = CUBOID_EDGES,
    *,
    radius_m: float = 0.0015,
) -> trimesh.Trimesh:
    """Create yellow cylinders for the eight-corner 3D bounding cuboid."""

    corners = np.asarray(corners_glb, dtype=np.float64).reshape(8, 3)
    parts = []
    for first, second in edges:
        point_a, point_b = corners[first], corners[second]
        direction = point_b - point_a
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            continue
        cylinder = trimesh.creation.cylinder(
            radius=radius_m,
            height=length,
            sections=10,
        )
        transform = trimesh.geometry.align_vectors(
            [0.0, 0.0, 1.0], direction / length
        )
        if transform is not None:
            cylinder.apply_transform(transform)
        cylinder.apply_translation((point_a + point_b) * 0.5)
        cylinder.visual.face_colors = np.tile(
            np.array([255, 220, 0, 255], dtype=np.uint8),
            (len(cylinder.faces), 1),
        )
        parts.append(cylinder)
    if not parts:
        raise ValueError("cuboid contains no valid edges")
    return trimesh.util.concatenate(parts)


def export_pose_scene(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: np.ndarray,
    pose_response: Dict[str, Any],
    output_dir: Path,
    *,
    factor_depth: float = 1000.0,
    max_points: int = 80_000,
) -> SceneArtifacts:
    """Export RGB-D points, pose axes, and measured cuboid as GLB/PLY."""

    points_glb, colors, stats = depth_rgb_to_pointcloud(
        depth_mm,
        rgb,
        intrinsics,
        factor_depth=factor_depth,
        max_points=max_points,
    )
    pose = np.asarray(pose_response["pose"], dtype=np.float64)
    origin_glb, rotation_glb = camera_pose_m_to_glb(
        pose[:3] / 1000.0,
        euler_zyx_matrix(*pose[3:]),
    )
    axes_mesh = create_pose_axes_mesh(origin_glb, rotation_glb)
    axis_points, axis_colors = inject_axis_points(origin_glb, rotation_glb)
    corners_glb = camera_points_m_to_glb(
        np.asarray(pose_response["corners_mm"], dtype=np.float64) / 1000.0
    )
    cuboid_mesh = create_cuboid_mesh(corners_glb)
    files = export_scene_files(
        points_glb,
        colors,
        Path(output_dir),
        "pose_scene",
        extra_axis_pts=[axis_points],
        extra_axis_cols=[axis_colors],
        extra_geometries=[axes_mesh, cuboid_mesh],
    )
    return SceneArtifacts(
        glb_path=Path(files["glb_path"]),
        ply_path=Path(files["ply_path"]),
        num_points=int(stats["total"]),
    )
