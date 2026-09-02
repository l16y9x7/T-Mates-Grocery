"""Point cloud + GLB/PLY export (camera → Three.js / Gradio Model3D)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from PIL import Image

DEFAULT_MAX_POINTS = 80_000
GRASP_CLOUD_MAX_POINTS = 80_000
# Match PEM_service / Gen6D: OpenCV (Y down) → GLB/Three.js (Y up). Only flip Y.
CAMERA_TO_GLB = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


def scale_intrinsics(
    intrinsics: np.ndarray,
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
) -> np.ndarray:
    """Scale K when image size changes. ``*_size`` is ``(width, height)``."""
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    k = np.asarray(intrinsics, dtype=np.float64).copy()
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k


_POSE_AXIS_COLORS = (
    [255, 64, 64, 255],
    [64, 220, 64, 255],
    [64, 96, 255, 255],
)


def camera_pose_m_to_glb(
    position_m: Sequence[float],
    rotation_3x3: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Camera frame meters → GLB preview frame (match RGB image upright)."""
    t = CAMERA_TO_GLB @ np.asarray(position_m, dtype=np.float64)
    r = CAMERA_TO_GLB @ np.asarray(rotation_3x3, dtype=np.float64) @ CAMERA_TO_GLB
    return t, r


def camera_points_m_to_glb(points_cam: np.ndarray) -> np.ndarray:
    """(N,3) camera meters → GLB preview meters."""
    pts = np.asarray(points_cam, dtype=np.float64)
    if pts.size == 0:
        return pts.astype(np.float32)
    return (pts @ CAMERA_TO_GLB.T).astype(np.float32)


def _align_z_to_direction(direction: np.ndarray) -> np.ndarray:
    """Same as Gen6D: rotate local +Z onto ``direction``."""
    direction = direction.astype(np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.eye(4)
    direction = direction / norm
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if np.allclose(direction, z_axis):
        return np.eye(4)
    if np.allclose(direction, -z_axis):
        return trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    axis = np.cross(z_axis, direction)
    axis = axis / np.linalg.norm(axis)
    angle = float(np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))
    return trimesh.transformations.rotation_matrix(angle, axis)


def create_pose_axes_mesh(
    origin: np.ndarray,
    rotation_glb: np.ndarray,
    *,
    axis_length_m: float | Sequence[float] = 0.08,
    radius_m: float = 0.002,
    head_length_ratio: float = 0.28,
    head_radius_ratio: float = 2.4,
) -> trimesh.Trimesh:
    """
    Gen6D-style solid RGB axes for Model3D GLB (trimesh mesh, not point tubes).

    Each axis = shaft cylinder + cone arrowhead (meters).
    """
    parts: list = []
    origin = np.asarray(origin, dtype=np.float64)
    rotation_glb = np.asarray(rotation_glb, dtype=np.float64)
    if isinstance(axis_length_m, (int, float)):
        lengths = [float(axis_length_m)] * 3
    else:
        lengths = [float(v) for v in axis_length_m]
        if len(lengths) != 3:
            raise ValueError("axis_length_m 需为标量或长度 3 的序列 [X,Y,Z]")

    for axis_idx, color in enumerate(_POSE_AXIS_COLORS):
        length = lengths[axis_idx]
        direction = rotation_glb[:, axis_idx]
        direction = direction / max(np.linalg.norm(direction), 1e-12)
        head_len = min(length * float(head_length_ratio), length * 0.45)
        shaft_len = max(length - head_len, length * 0.55)
        head_radius = float(radius_m) * float(head_radius_ratio)
        align = _align_z_to_direction(direction)

        shaft = trimesh.creation.cylinder(radius=radius_m, height=shaft_len, sections=12)
        shaft.apply_transform(align)
        shaft.apply_translation(origin + direction * (shaft_len / 2.0))
        shaft.visual.face_colors = np.tile(np.array(color, dtype=np.uint8), (len(shaft.faces), 1))
        parts.append(shaft)

        # trimesh cone: base at local z=0, tip at z=height → place base at shaft end
        head = trimesh.creation.cone(radius=head_radius, height=head_len, sections=12)
        head.apply_transform(align)
        head.apply_translation(origin + direction * shaft_len)
        head.visual.face_colors = np.tile(np.array(color, dtype=np.uint8), (len(head.faces), 1))
        parts.append(head)

    marker = trimesh.creation.icosphere(radius=radius_m * 2.5, subdivisions=2)
    marker.apply_translation(origin)
    marker.visual.face_colors = np.tile(
        np.array([255, 220, 0, 255], dtype=np.uint8),
        (len(marker.faces), 1),
    )
    parts.append(marker)
    return trimesh.util.concatenate(parts)


def create_pose_marker_sphere(
    origin: np.ndarray,
    *,
    radius_m: float = 0.008,
    color_rgba: Sequence[int] = (255, 0, 255, 255),
) -> trimesh.Trimesh:
    """Solid sphere marker at pose origin (Gen6D-style)."""
    origin = np.asarray(origin, dtype=np.float64)
    sphere = trimesh.creation.icosphere(radius=float(radius_m), subdivisions=2)
    sphere.apply_translation(origin)
    rgba = np.array(color_rgba, dtype=np.uint8)
    if rgba.shape[0] == 3:
        rgba = np.concatenate([rgba, np.array([255], dtype=np.uint8)])
    sphere.visual.face_colors = np.tile(rgba, (len(sphere.faces), 1))
    return sphere


def _cylinder_between(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    radius_m: float,
    color_rgba: Sequence[int],
    sections: int = 10,
) -> trimesh.Trimesh | None:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        return None
    cyl = trimesh.creation.cylinder(radius=radius_m, height=length, sections=sections)
    cyl.apply_transform(_align_z_to_direction(vec / length))
    cyl.apply_translation((p0 + p1) * 0.5)
    rgba = np.asarray(color_rgba, dtype=np.uint8)
    if rgba.shape[0] == 3:
        rgba = np.concatenate([rgba, np.array([255], dtype=np.uint8)])
    cyl.visual.face_colors = np.tile(rgba, (len(cyl.faces), 1))
    return cyl


def create_grasp_gripper_mesh(
    origin: np.ndarray,
    rotation_glb: np.ndarray,
    *,
    width_m: float,
    depth_m: float,
    color_rgba: Sequence[int] = (255, 220, 0, 255),
    radius_m: float = 0.0018,
    depth_base_m: float = 0.02,
    tail_length_m: float = 0.04,
) -> trimesh.Trimesh:
    """
    Parallel-jaw gripper silhouette (GraspNet ``plot_gripper_pro_max`` frame).

    Local gripper frame (before ``rotation_glb``):
      - X: approach / finger direction (tips toward +X)
      - Y: closing / width
      - Z: binormal

    Geometry matches the sketch: two fingers + base bar + approach tail,
    with a sphere at the grasp center between the fingers.
    """
    origin = np.asarray(origin, dtype=np.float64)
    R = np.asarray(rotation_glb, dtype=np.float64)
    width = max(float(width_m), 0.01)
    depth = max(float(depth_m), 0.005)
    depth_base = float(depth_base_m)
    tail_len = float(tail_length_m)
    half_w = width * 0.5

    # Local keypoints → world via R @ p + origin
    def _w(p_local: Sequence[float]) -> np.ndarray:
        return origin + R @ np.asarray(p_local, dtype=np.float64)

    # Finger tips at +depth; palm/base at -depth_base; grasp center at 0
    left_base = _w([-depth_base, -half_w, 0.0])
    left_tip = _w([depth, -half_w, 0.0])
    right_base = _w([-depth_base, half_w, 0.0])
    right_tip = _w([depth, half_w, 0.0])
    mid_base = _w([-depth_base, 0.0, 0.0])
    tail_end = _w([-depth_base - tail_len, 0.0, 0.0])

    rgba = tuple(int(v) for v in color_rgba)
    if len(rgba) == 3:
        rgba = (*rgba, 255)

    parts: list = []
    for seg in (
        _cylinder_between(left_base, left_tip, radius_m=radius_m, color_rgba=rgba),
        _cylinder_between(right_base, right_tip, radius_m=radius_m, color_rgba=rgba),
        _cylinder_between(left_base, right_base, radius_m=radius_m, color_rgba=rgba),
        _cylinder_between(mid_base, tail_end, radius_m=radius_m * 1.15, color_rgba=rgba),
    ):
        if seg is not None:
            parts.append(seg)

    # Grasp point between fingers
    parts.append(
        create_pose_marker_sphere(origin, radius_m=max(radius_m * 2.8, 0.004), color_rgba=rgba)
    )
    return trimesh.util.concatenate(parts)


def inject_axis_points(
    origin_glb: np.ndarray,
    rotation_glb: np.ndarray,
    *,
    axis_length_m: float | Sequence[float] = 0.08,
    samples_per_axis: int = 24,
    core_color: Sequence[int] = (255, 0, 255),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin point-line axes for PLY / PointCloud fallback (Gen6D ``inject_axis_points``).
    Solid arrows for browser preview come from ``create_pose_axes_mesh``.
    """
    origin_glb = np.asarray(origin_glb, dtype=np.float64)
    rotation_glb = np.asarray(rotation_glb, dtype=np.float64)
    if isinstance(axis_length_m, (int, float)):
        lengths = [float(axis_length_m)] * 3
    else:
        lengths = [float(v) for v in axis_length_m]
        if len(lengths) != 3:
            raise ValueError("axis_length_m 需为标量或长度 3 的序列 [X,Y,Z]")
    axis_colors = (
        np.array([255, 64, 64], dtype=np.uint8),
        np.array([64, 220, 64], dtype=np.uint8),
        np.array([64, 96, 255], dtype=np.uint8),
    )
    pts: list = []
    cols: list = []
    rng = np.random.default_rng(0)
    core = origin_glb + rng.normal(0.0, 0.0025, size=(80, 3))
    pts.append(core.astype(np.float32))
    cols.append(np.tile(np.asarray(core_color, dtype=np.uint8), (core.shape[0], 1)))
    for axis_idx, color in enumerate(axis_colors):
        direction = rotation_glb[:, axis_idx]
        length = lengths[axis_idx]
        n_samples = max(samples_per_axis, int(round(samples_per_axis * length / max(lengths))))
        for t in np.linspace(0.0, length, n_samples):
            pts.append((origin_glb + direction * float(t)).astype(np.float32))
            cols.append(color)
    return np.vstack(pts).astype(np.float32), np.vstack(cols).astype(np.uint8)


def inject_grasp_gripper_points(
    origin_glb: np.ndarray,
    rotation_glb: np.ndarray,
    *,
    width_m: float,
    depth_m: float,
    color: Sequence[int] = (255, 220, 0),
    depth_base_m: float = 0.02,
    tail_length_m: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel-jaw gripper as point segments for PLY (same frame as mesh).

    GraspNet local frame: X=approach, Y=width/closing, Z=binormal.
    """
    origin_glb = np.asarray(origin_glb, dtype=np.float64)
    R = np.asarray(rotation_glb, dtype=np.float64)
    width = max(float(width_m), 0.01)
    depth = max(float(depth_m), 0.005)
    half_w = width * 0.5
    depth_base = float(depth_base_m)
    tail_len = float(tail_length_m)

    def _w(p_local: Sequence[float]) -> np.ndarray:
        return origin_glb + R @ np.asarray(p_local, dtype=np.float64)

    left_base = _w([-depth_base, -half_w, 0.0])
    left_tip = _w([depth, -half_w, 0.0])
    right_base = _w([-depth_base, half_w, 0.0])
    right_tip = _w([depth, half_w, 0.0])
    mid_base = _w([-depth_base, 0.0, 0.0])
    tail_end = _w([-depth_base - tail_len, 0.0, 0.0])

    pts: list = []
    cols: list = []
    color_arr = np.asarray(color, dtype=np.uint8)
    rng = np.random.default_rng(1)

    def _add_segment(p0, p1, n=36, radius=0.0018):
        for t in np.linspace(0.0, 1.0, n):
            c = (1.0 - t) * p0 + t * p1
            cloud = c + rng.normal(0.0, radius, size=(10, 3))
            pts.append(cloud.astype(np.float32))
            cols.append(np.tile(color_arr, (cloud.shape[0], 1)))

    _add_segment(left_base, left_tip)
    _add_segment(right_base, right_tip)
    _add_segment(left_base, right_base)
    _add_segment(mid_base, tail_end, n=28, radius=0.0022)
    # grasp point
    core = origin_glb + rng.normal(0.0, 0.003, size=(60, 3))
    pts.append(core.astype(np.float32))
    cols.append(np.tile(color_arr, (core.shape[0], 1)))

    return np.vstack(pts).astype(np.float32), np.vstack(cols).astype(np.uint8)


def dim_point_colors(colors: np.ndarray, scale: float = 0.45) -> np.ndarray:
    """Dim scene colors so grasp markers stand out."""
    out = np.clip(colors.astype(np.float32) * float(scale), 0, 255).astype(np.uint8)
    return out


def export_pointcloud_glb(points: np.ndarray, colors: np.ndarray, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if points.size == 0:
        cloud = trimesh.points.PointCloud(vertices=np.zeros((1, 3)), colors=[[128, 128, 128]])
    else:
        cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
    cloud.export(output_path)
    return output_path


def export_pointcloud_ply(points: np.ndarray, colors: np.ndarray, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if points.size == 0:
        cloud = trimesh.points.PointCloud(vertices=np.zeros((1, 3)), colors=[[128, 128, 128]])
    else:
        cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
    cloud.export(output_path)
    return output_path


def export_scene_glb(
    geometries: Sequence[trimesh.Trimesh | trimesh.points.PointCloud],
    output_path: Path,
) -> Path:
    """Export PointCloud + solid meshes as one GLB scene (Gen6D-style)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for geom in geometries:
        scene.add_geometry(geom)
    scene.export(output_path)
    return output_path


def depth_rgb_to_pointcloud(
    depth_mm: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    *,
    factor_depth: float = 1000.0,
    max_points: int = DEFAULT_MAX_POINTS,
    max_depth_m: Optional[float] = 3.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    RGB-D back-project (PEM/Gen6D style): same ``(u,v)`` for depth and color.

    If RGB size differs, resize RGB → depth grid (do **not** warp depth).
    """
    h, w = depth_mm.shape
    k = np.asarray(intrinsics, dtype=np.float64)
    if rgb.shape[:2] != (h, w):
        src_h, src_w = int(rgb.shape[0]), int(rgb.shape[1])
        rgb = np.array(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
        k = scale_intrinsics(k, (src_w, src_h), (w, h))

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    u_coords, v_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    if max_depth_m is not None:
        if not np.isfinite(max_depth_m) or max_depth_m <= 0:
            raise ValueError("max_depth_m must be positive or None")
        valid &= depth_mm <= float(max_depth_m) * float(factor_depth)
    if not valid.any():
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            {"total": 0},
        )

    z = depth_mm[valid].astype(np.float32) / float(factor_depth)
    x = (u_coords[valid] - cx) * z / fx
    y = (v_coords[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    # PEM: points[:, 1] *= -1  (== CAMERA_TO_GLB with only Y flip)
    points = camera_points_m_to_glb(points)

    colors = rgb[valid]
    if colors.dtype != np.uint8:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    if colors.ndim == 1:
        colors = np.repeat(colors.reshape(-1, 1), 3, axis=1)
    elif colors.shape[-1] == 1:
        colors = np.repeat(colors.reshape(-1, 1), 3, axis=1)
    else:
        colors = np.ascontiguousarray(colors[..., :3])

    n = points.shape[0]
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    return points, colors, {"total": int(points.shape[0])}


def depth_rgb_instance_to_pointcloud(
    depth_mm: np.ndarray,
    rgb: np.ndarray,
    instance_id_map: np.ndarray,
    intrinsics: np.ndarray,
    *,
    factor_depth: float = 1000.0,
    max_points: int = DEFAULT_MAX_POINTS,
    background_dim: float = 1.0,
    prefer_instances: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    RGB-D point cloud with camera colors on **every** valid depth pixel.

    Same-pixel coloring as PEM/Gen6D (resize RGB→depth if needed; no depth warp).
    ``background_dim``: multiply background RGB (1.0 = full color). Prefer
    uniform downsample so the whole scene stays colored (not instance-only).
    """
    h, w = depth_mm.shape
    k = np.asarray(intrinsics, dtype=np.float64)
    if rgb.shape[:2] != (h, w):
        src_h, src_w = int(rgb.shape[0]), int(rgb.shape[1])
        rgb = np.array(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
        k = scale_intrinsics(k, (src_w, src_h), (w, h))
    if instance_id_map.shape != (h, w):
        id_img = Image.fromarray(instance_id_map.astype(np.uint8)).resize((w, h), Image.NEAREST)
        instance_id_map = np.array(id_img, dtype=np.uint8)

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    u_coords, v_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = depth_mm > 0
    if not valid.any():
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            {"background": 0, "instances": 0, "total": 0},
        )

    z = depth_mm[valid].astype(np.float32) / float(factor_depth)
    x = (u_coords[valid] - cx) * z / fx
    y = (v_coords[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    points = camera_points_m_to_glb(points)

    colors = rgb[valid]
    if colors.dtype != np.uint8:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    if colors.ndim == 1:
        colors = np.repeat(colors.reshape(-1, 1), 3, axis=1)
    elif colors.shape[-1] == 1:
        colors = np.repeat(colors.reshape(-1, 1), 3, axis=1)
    else:
        colors = np.ascontiguousarray(colors[..., :3])

    ids = instance_id_map[valid].astype(np.int32)
    bg = ids == 0
    if np.any(bg) and float(background_dim) < 0.999:
        colors = colors.copy()
        colors[bg] = dim_point_colors(colors[bg], scale=float(background_dim))

    n = points.shape[0]
    if n > max_points:
        rng = np.random.default_rng(42)
        if prefer_instances:
            keep = np.zeros(n, dtype=bool)
            remaining = max_points
            for mask in (ids > 0, ids == 0):
                idx = np.flatnonzero(mask)
                if idx.size == 0:
                    continue
                if idx.size <= remaining:
                    keep[idx] = True
                    remaining -= int(idx.size)
                else:
                    keep[rng.choice(idx, remaining, replace=False)] = True
                    remaining = 0
                    break
            points = points[keep]
            colors = colors[keep]
            ids = ids[keep]
        else:
            # Uniform sample: keep full-scene RGB coverage
            idx = rng.choice(n, int(max_points), replace=False)
            points = points[idx]
            colors = colors[idx]
            ids = ids[idx]

    return points, colors, {
        "background": int(np.count_nonzero(ids == 0)),
        "instances": int(np.count_nonzero(ids > 0)),
        "total": int(points.shape[0]),
        "max_points": int(max_points),
    }


def depth_instance_to_pointcloud(
    depth_mm: np.ndarray,
    instance_id_map: np.ndarray,
    intrinsics: np.ndarray,
    instance_colors_rgb: Sequence[Sequence[int]],
    *,
    factor_depth: float = 1000.0,
    max_points: int = DEFAULT_MAX_POINTS,
    background_color: Sequence[int] = (88, 88, 96),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Back-project depth to GLB preview meters (upright like RGB image).

    Returns points/colors already in ``preview_frame=glb`` (meters).
    """
    h, w = depth_mm.shape
    if instance_id_map.shape != (h, w):
        id_img = Image.fromarray(instance_id_map.astype(np.uint8)).resize((w, h), Image.NEAREST)
        instance_id_map = np.array(id_img, dtype=np.uint8)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u_coords, v_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = depth_mm > 0
    if not valid.any():
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            {"background": 0, "instances": 0},
        )

    z = depth_mm[valid].astype(np.float32) / float(factor_depth)
    x = (u_coords[valid] - cx) * z / fx
    y = (v_coords[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    # camera → GLB preview (same upright orientation as RGB image)
    points = camera_points_m_to_glb(points)

    ids = instance_id_map[valid].astype(np.int32)
    colors = np.tile(np.asarray(background_color, dtype=np.uint8), (points.shape[0], 1))
    palette = [np.asarray(c, dtype=np.uint8)[:3] for c in instance_colors_rgb]
    for inst_id in np.unique(ids):
        if int(inst_id) <= 0:
            continue
        color = (
            palette[(int(inst_id) - 1) % max(len(palette), 1)]
            if palette
            else np.array([255, 140, 30], dtype=np.uint8)
        )
        colors[ids == inst_id] = color

    n = points.shape[0]
    if n > max_points:
        rng = np.random.default_rng(42)
        keep = np.zeros(n, dtype=bool)
        remaining = max_points
        for mask in (ids > 0, ids == 0):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            if idx.size <= remaining:
                keep[idx] = True
                remaining -= int(idx.size)
            else:
                keep[rng.choice(idx, remaining, replace=False)] = True
                remaining = 0
                break
        points = points[keep]
        colors = colors[keep]
        ids = ids[keep]

    return points, colors, {
        "background": int(np.count_nonzero(ids == 0)),
        "instances": int(np.count_nonzero(ids > 0)),
        "total": int(points.shape[0]),
    }


def build_instance_id_map(
    masks: List[np.ndarray],
    height: int,
    width: int,
) -> np.ndarray:
    id_map = np.zeros((height, width), dtype=np.uint8)
    next_id = 1
    for mask in masks:
        fill = mask.astype(bool) & (id_map == 0)
        if not np.any(fill):
            continue
        id_map[fill] = np.uint8(next_id)
        next_id += 1
        if next_id > 254:
            break
    return id_map


def points_to_colored_mesh(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    scale_m: float = 0.0020,
) -> trimesh.Trimesh:
    """
    Tiny tetrahedra with vertex colors (visible from all angles, compact GLB).

    Octahedra were ~2× heavier; single triangles vanish under back-face culling.
    """
    pts = np.asarray(points, dtype=np.float64)
    cols = np.asarray(colors, dtype=np.uint8)
    if pts.size == 0:
        mesh = trimesh.creation.icosphere(radius=0.001, subdivisions=1)
        mesh.visual.vertex_colors = np.tile(
            np.array([128, 128, 128, 255], dtype=np.uint8), (len(mesh.vertices), 1)
        )
        return mesh
    if cols.ndim == 1 or cols.shape[-1] == 1:
        cols = np.repeat(cols.reshape(-1, 1), 3, axis=1)
    cols = cols[:, :3]
    n = int(pts.shape[0])
    s = float(scale_m)
    # Regular-ish tetrahedron offsets
    off = np.array(
        [
            [s, s, s],
            [s, -s, -s],
            [-s, s, -s],
            [-s, -s, s],
        ],
        dtype=np.float64,
    )
    local_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2],
        ],
        dtype=np.int64,
    )
    vertices = (pts[:, None, :] + off[None, :, :]).reshape(-1, 3)
    faces = (local_faces[None, :, :] + (np.arange(n, dtype=np.int64) * 4)[:, None, None]).reshape(
        -1, 3
    )
    rgba = np.concatenate(
        [np.repeat(cols, 4, axis=0), np.full((n * 4, 1), 255, dtype=np.uint8)],
        axis=1,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = rgba
    try:
        mat = getattr(mesh.visual, "material", None)
        if mat is not None and hasattr(mat, "doubleSided"):
            mat.doubleSided = True
    except Exception:  # noqa: BLE001
        pass
    return mesh


def export_scene_files(
    points_glb: np.ndarray,
    colors: np.ndarray,
    output_dir: Path,
    stem: str,
    extra_axis_pts: List[np.ndarray] | None = None,
    extra_axis_cols: List[np.ndarray] | None = None,
    extra_geometries: Sequence[trimesh.Trimesh] | None = None,
) -> Dict[str, Path]:
    pts = np.asarray(points_glb, dtype=np.float32)
    cols = np.asarray(colors, dtype=np.uint8)

    # PLY: include gripper point strokes; GLB: colored cloud mesh + solid grippers
    ply_pts, ply_cols = pts, cols
    if extra_axis_pts:
        ply_pts = np.concatenate(
            [pts] + [np.asarray(p, dtype=np.float32) for p in extra_axis_pts], axis=0
        )
        ply_cols = np.concatenate(
            [cols] + [np.asarray(c, dtype=np.uint8) for c in extra_axis_cols or []], axis=0
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    glb_path = output_dir / f"{stem}.glb"
    ply_path = output_dir / f"{stem}.ply"
    export_pointcloud_ply(ply_pts, ply_cols, ply_path)

    geometries: list = [points_to_colored_mesh(pts, cols)]
    if extra_geometries:
        geometries.extend(list(extra_geometries))
    export_scene_glb(geometries, glb_path)
    return {"glb_path": glb_path, "ply_path": ply_path}
