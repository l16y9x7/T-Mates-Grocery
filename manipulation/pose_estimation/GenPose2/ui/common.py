"""Shared helpers for Gradio UI (depth/camera IO, SAM3 previews)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# OpenCV BGR colors aligned with scripts/sam3_seg.py
VIS_COLORS_BGR: Tuple[Tuple[int, int, int], ...] = (
    (0, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 128),
    (64, 64, 255),
    (255, 64, 64),
)


def filepath_from_upload(file_obj: Any) -> Optional[Path]:
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        return Path(file_obj)
    if isinstance(file_obj, dict):
        path = file_obj.get("path") or file_obj.get("name")
        if path:
            return Path(path)
    name = getattr(file_obj, "name", None)
    if name:
        return Path(name)
    return None


def load_depth_mm(path: Path) -> np.ndarray:
    """Load depth as uint16 millimeters. Supports .npy / .png / .tif."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"depth not found: {path}")
    if path.suffix.lower() == ".npy":
        depth = np.load(str(path))
    else:
        depth = np.array(Image.open(path))
    if np.issubdtype(depth.dtype, np.floating):
        # assume meters
        depth = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    elif depth.dtype != np.uint16:
        depth = depth.astype(np.uint16)
    return depth


def load_camera_json(path: Path) -> Tuple[np.ndarray, float]:
    """Return (K 3x3, factor_depth) where depth_m = depth_mm / factor_depth typically."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        camera = json.load(f)
    return _intrinsics_and_factor_from_camera(camera)


def _intrinsics_and_factor_from_camera(camera: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    """Support warmup ``cam_K`` and GenPose2 ``camera.intrinsics`` formats."""
    if "cam_K" in camera:
        intrinsic = np.array(camera["cam_K"], dtype=np.float64).reshape(3, 3)
        depth_scale = float(camera.get("depth_scale", 0.001))
    elif "camera" in camera and "intrinsics" in camera.get("camera", {}):
        intr = camera["camera"]["intrinsics"]
        intrinsic = np.array(
            [
                [float(intr["fx"]), 0.0, float(intr["cx"])],
                [0.0, float(intr["fy"]), float(intr["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        depth_scale = float(camera.get("depth_scale", 0.001))
    else:
        raise ValueError("camera.json must contain cam_K or camera.intrinsics")
    # depth stored as mm → factor_depth ≈ 1000；depth_scale=0.001 表示 mm→m
    if depth_scale >= 0.1:
        factor_depth = 1000.0
    else:
        factor_depth = 1.0 / depth_scale
    return intrinsic, factor_depth


def load_camera_json_full(path: Path) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Like ``load_camera_json`` but also returns the raw camera dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        camera = json.load(f)
    intrinsic, factor_depth = _intrinsics_and_factor_from_camera(camera)
    return intrinsic, factor_depth, camera


def estimate_depth_to_rgb_shift(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    *,
    search_x: int = 32,
    search_y: int = 24,
    step: int = 2,
    min_improve_ratio: float = 1.25,
    max_shift_px: int = 40,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Estimate 2D pixel shift that warps depth into RGB (edge correlation).

    Conservative by default: PEM/Gen6D assume already-aligned RGB-D and color
    with the same ``(u,v)``. Large spurious shifts (e.g. dx≈90) make colors
    slide off geometry, so we reject weak / oversized matches.

    Returns ``(dx, dy, stats)`` where aligned_depth(u,v) ≈ depth(u-dx, v-dy)
    i.e. ``cv2.warpAffine(depth, [[1,0,dx],[0,1,dy]])``.
    """
    rgb_u8 = np.asarray(rgb)
    if rgb_u8.ndim == 3:
        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb_u8
    depth = np.asarray(depth_mm)
    if depth.shape[:2] != gray.shape[:2]:
        depth = np.array(
            Image.fromarray(depth).resize((gray.shape[1], gray.shape[0]), Image.NEAREST)
        )

    valid = depth > 0
    d_norm = np.zeros(depth.shape[:2], dtype=np.uint8)
    if np.any(valid):
        vals = depth[valid].astype(np.float64)
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
        if hi <= lo:
            hi = lo + 1.0
        d_norm[valid] = np.clip((vals - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    depth_edges = cv2.Canny(d_norm, 40, 120)
    rgb_edges = cv2.Canny(gray, 50, 150)
    if int(depth_edges.sum()) == 0 or int(rgb_edges.sum()) == 0:
        return 0, 0, {
            "enabled": True,
            "dx": 0,
            "dy": 0,
            "score": 0,
            "baseline": 0,
            "method": "edge",
            "applied": False,
            "reject_reason": "empty_edges",
        }

    h, w = depth_edges.shape
    baseline = int(((depth_edges > 0) & (rgb_edges > 0)).sum())
    best_score = baseline
    best_dx, best_dy = 0, 0
    for dy in range(-int(search_y), int(search_y) + 1, int(step)):
        for dx in range(-int(search_x), int(search_x) + 1, int(step)):
            if dx == 0 and dy == 0:
                score = baseline
            else:
                M = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
                shifted = cv2.warpAffine(
                    depth_edges, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
                )
                score = int(((shifted > 0) & (rgb_edges > 0)).sum())
            if score > best_score:
                best_score = score
                best_dx, best_dy = int(dx), int(dy)

    # Optional 1px refine around best
    for dy in range(best_dy - 1, best_dy + 2):
        for dx in range(best_dx - 1, best_dx + 2):
            M = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
            shifted = cv2.warpAffine(depth_edges, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            score = int(((shifted > 0) & (rgb_edges > 0)).sum())
            if score > best_score:
                best_score = score
                best_dx, best_dy = int(dx), int(dy)

    improved = bool(best_score > baseline)
    reject_reason = None
    apply_dx, apply_dy = best_dx, best_dy
    if not improved:
        apply_dx, apply_dy = 0, 0
        reject_reason = "no_improvement"
    elif best_score < float(min_improve_ratio) * max(float(baseline), 1.0):
        apply_dx, apply_dy = 0, 0
        reject_reason = "weak_improvement"
    elif max(abs(best_dx), abs(best_dy)) > int(max_shift_px):
        apply_dx, apply_dy = 0, 0
        reject_reason = "shift_too_large"

    return apply_dx, apply_dy, {
        "enabled": True,
        "method": "edge_correlation",
        "dx": int(apply_dx),
        "dy": int(apply_dy),
        "raw_dx": int(best_dx),
        "raw_dy": int(best_dy),
        "score": best_score,
        "baseline": baseline,
        "improved": improved,
        "applied": bool(apply_dx != 0 or apply_dy != 0),
        "reject_reason": reject_reason,
        "min_improve_ratio": float(min_improve_ratio),
        "max_shift_px": int(max_shift_px),
    }


def shift_rgb_xy(
    rgb: np.ndarray,
    dx: int,
    dy: int = 0,
    *,
    nearest: bool = False,
) -> np.ndarray:
    """
    Translate image / label map content. ``dx>0`` moves content to the right.

    Use ``nearest=True`` for instance id maps / masks.
    """
    img = np.asarray(rgb)
    dx_i, dy_i = int(dx), int(dy)
    if dx_i == 0 and dy_i == 0:
        return img
    h, w = img.shape[:2]
    M = np.float32([[1.0, 0.0, float(dx_i)], [0.0, 1.0, float(dy_i)]])
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if img.ndim == 2:
        border: Any = 0
    else:
        border = (0, 0, 0)
    out = cv2.warpAffine(img, M, (w, h), flags=flags, borderValue=border)
    if out.dtype != img.dtype:
        out = out.astype(img.dtype)
    return out


def align_depth_to_rgb(
    depth_mm: np.ndarray,
    rgb: np.ndarray,
    *,
    shift_xy: Optional[Tuple[int, int]] = None,
    auto: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Warp depth into RGB pixel grid.

    ``shift_xy=(dx,dy)`` from camera.json overrides auto estimation when provided.
    """
    depth = np.asarray(depth_mm)
    rgb_arr = np.asarray(rgb)
    h, w = rgb_arr.shape[:2]
    if depth.shape[:2] != (h, w):
        depth = np.array(Image.fromarray(depth).resize((w, h), Image.NEAREST))

    if shift_xy is not None:
        dx, dy = int(shift_xy[0]), int(shift_xy[1])
        stats = {
            "enabled": True,
            "method": "manual_shift",
            "dx": dx,
            "dy": dy,
            "score": None,
            "baseline": None,
            "improved": True,
        }
    elif auto:
        dx, dy, stats = estimate_depth_to_rgb_shift(rgb_arr, depth)
    else:
        return depth, {"enabled": False, "dx": 0, "dy": 0}

    if dx == 0 and dy == 0:
        return depth, stats

    M = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    aligned = cv2.warpAffine(depth, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if aligned.dtype != depth.dtype:
        aligned = aligned.astype(depth.dtype)
    return aligned, stats


def resolve_rgb_depth_alignment(
    depth_mm: np.ndarray,
    rgb: np.ndarray,
    camera_meta: Dict[str, Any],
    *,
    ui_rgb_shift_x: float = 0.0,
    ui_rgb_shift_y: float = 0.0,
    enable_depth_align: bool = True,
    auto_if_zero: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Prepare depth + cloud-color for RGB-D pipelines.

    When ``enable_depth_align`` is True (recommended):
      - Warp **depth → RGB** so SAM3 mask / GenPose2 / 2D overlay share ``(u,v)``.
      - ``camera.json`` ``depth_to_rgb_shift`` / ``depth_shift`` used as-is for depth warp.
      - Else ``rgb_shift`` / ``color_shift`` / UI dx,dy are treated as *RGB coloring*
        convention (historical ``dx=-45``): depth warp uses ``(-dx, -dy)``.
      - After geometric align, cloud color uses **unshifted** RGB.

    When False: keep legacy behavior (only shift RGB/id for GLB coloring).

    Returns:
      ``(depth_mm_out, color_for_cloud, meta)``
    """
    color_np = np.asarray(rgb)
    if color_np.ndim == 2:
        color_for_cloud = color_np
    else:
        color_for_cloud = color_np.copy()

    cam_depth_shift = camera_meta.get("depth_to_rgb_shift") or camera_meta.get(
        "depth_shift"
    )
    cam_rgb_shift = camera_meta.get("rgb_shift") or camera_meta.get("color_shift")
    if isinstance(cam_rgb_shift, (list, tuple)) and len(cam_rgb_shift) == 2:
        sx, sy = int(cam_rgb_shift[0]), int(cam_rgb_shift[1])
        rgb_src = "camera.json:rgb_shift"
    else:
        sx = int(round(float(ui_rgb_shift_x or 0)))
        sy = int(round(float(ui_rgb_shift_y or 0)))
        rgb_src = "ui"

    if not bool(enable_depth_align):
        meta: Dict[str, Any] = {
            "depth_align": {"enabled": False},
            "rgb_shift": {"dx": sx, "dy": sy, "source": rgb_src, "applied_to": "color"},
        }
        if sx != 0 or sy != 0:
            color_for_cloud = shift_rgb_xy(color_for_cloud, sx, sy)
        return np.asarray(depth_mm), color_for_cloud, meta

    if isinstance(cam_depth_shift, (list, tuple)) and len(cam_depth_shift) == 2:
        d_shift: Optional[Tuple[int, int]] = (
            int(cam_depth_shift[0]),
            int(cam_depth_shift[1]),
        )
        method = "camera.json:depth_to_rgb_shift"
        auto = False
    elif sx != 0 or sy != 0:
        # Historical UI: shift RGB by (sx,sy) for coloring ⇔ warp depth by (-sx,-sy)
        d_shift = (-sx, -sy)
        method = f"{rgb_src}->depth_inverse"
        auto = False
    elif auto_if_zero:
        d_shift = None
        method = "auto_edge"
        auto = True
    else:
        d_shift = (0, 0)
        method = "none"
        auto = False

    aligned, align_stats = align_depth_to_rgb(
        depth_mm, color_np, shift_xy=d_shift, auto=auto
    )
    align_stats = dict(align_stats)
    align_stats["method_resolved"] = method
    meta = {
        "depth_align": align_stats,
        "rgb_shift": {
            "dx": 0,
            "dy": 0,
            "source": "aligned",
            "applied_to": "none",
            "legacy_rgb_shift_input": {"dx": sx, "dy": sy, "source": rgb_src},
        },
    }
    # Geometry aligned: color and depth share RGB grid — no extra color warp
    return aligned, color_for_cloud, meta


def depth_to_colormap(depth_mm: np.ndarray) -> Image.Image:
    depth = np.asarray(depth_mm)
    valid = depth > 0
    vis = np.zeros(depth.shape[:2], dtype=np.uint8)
    if np.any(valid):
        vals = depth[valid].astype(np.float64)
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
        vis[valid] = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def vis_colors_rgb() -> List[Tuple[int, int, int]]:
    return [(int(r), int(g), int(b)) for (b, g, r) in VIS_COLORS_BGR]


def erode_bool_mask(mask: np.ndarray, pixels: int = 2) -> np.ndarray:
    """二值/bool mask 边界腐蚀 ``pixels`` 像素（与 SAM3 实例后处理一致）。"""
    if pixels <= 0:
        return mask
    region = (np.asarray(mask) > 0).astype(np.uint8)
    eroded = cv2.erode(region, np.ones((3, 3), np.uint8), iterations=int(pixels))
    if mask.dtype == bool:
        return eroded.astype(bool)
    return eroded


def render_mask_bbox_previews(
    image: Image.Image,
    instance_dets: List[Dict[str, Any]],
    decode_mask_fn,
    *,
    prompt: str = "obj",
    mask_alpha: float = 0.5,
    edge_erode_px: int = 2,
) -> Tuple[Image.Image, Image.Image]:
    """Return (mask-only overlay, bbox-only overlay) as RGB PIL images."""
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    image_size = (w, h)
    label = (prompt or "obj").split()[0] or "obj"

    mask_overlay = bgr.astype(np.float32).copy()
    bbox_overlay = bgr.astype(np.float32).copy()

    for idx, det in enumerate(instance_dets):
        mask = decode_mask_fn(det, image_size)
        mask = erode_bool_mask(mask, edge_erode_px)
        color = VIS_COLORS_BGR[idx % len(VIS_COLORS_BGR)]
        color_arr = np.array(color, dtype=np.float32)
        mask_overlay[mask] = mask_alpha * color_arr + (1.0 - mask_alpha) * mask_overlay[mask]

        bbox = det.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, bw, bh = [int(round(v)) for v in bbox]
        else:
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            x, y = int(xs.min()), int(ys.min())
            bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        score = float(det.get("score", 0.0))
        cv2.rectangle(bbox_overlay, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(
            bbox_overlay,
            f"{label} id={idx + 1} {score:.3f}",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    mask_rgb = cv2.cvtColor(mask_overlay.astype(np.uint8), cv2.COLOR_BGR2RGB)
    bbox_rgb = cv2.cvtColor(bbox_overlay.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return Image.fromarray(mask_rgb), Image.fromarray(bbox_rgb)


def _project_point_m(
    point_m: np.ndarray,
    intrinsic: np.ndarray,
    *,
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    z = float(point_m[2])
    if z <= 1e-6:
        return None
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    u = int(round(fx * float(point_m[0]) / z + cx))
    v = int(round(fy * float(point_m[1]) / z + cy))
    if u < -width or u > 2 * width or v < -height or v > 2 * height:
        return None
    return u, v


def render_grasp_pose_overlay(
    image: Image.Image,
    grasps: List[Dict[str, Any]],
    intrinsic: np.ndarray,
    *,
    axis_length_m: float = 0.05,
    max_grasps: int = 20,
) -> Image.Image:
    """
    Project grasp poses onto RGB (Gen6D-style RGB axes with arrow tips).

    GraspNet camera frame: X right, Y down, Z forward. Axis colors (BGR):
    X=red, Y=green, Z=blue.
    """
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    axis_colors_bgr = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    axis_labels = ("X", "Y", "Z")

    for idx, g in enumerate(grasps[: max(0, int(max_grasps))]):
        center = np.asarray(g.get("center"), dtype=np.float64).reshape(3)
        rotation = np.asarray(g.get("rotation"), dtype=np.float64).reshape(3, 3)
        depth = float(g.get("depth", 0.02) or 0.02)
        width_m = float(g.get("width", 0.0) or 0.0)
        scale = max(float(axis_length_m), depth * 3.0)

        p0 = _project_point_m(center, intrinsic, width=w, height=h)
        if p0 is None:
            continue
        cv2.circle(bgr, p0, 5, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(bgr, p0, 7, (40, 40, 40), 1, cv2.LINE_AA)
        score = float(g.get("score", 0.0))
        inst = g.get("instance_id")
        label = f"G{idx + 1}"
        if inst is not None:
            label += f" i{int(inst)}"
        label += f" {score:.2f}"
        if width_m > 0:
            label += f" w{width_m * 1000:.0f}mm"
        cv2.putText(
            bgr,
            label,
            (p0[0] + 8, p0[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        # Gripper opening width along grasp X (closing direction)
        if width_m > 1e-4:
            half = 0.5 * width_m
            x_axis = rotation[:, 0]
            left = center - half * x_axis
            right = center + half * x_axis
            pl = _project_point_m(left, intrinsic, width=w, height=h)
            pr = _project_point_m(right, intrinsic, width=w, height=h)
            if pl is not None and pr is not None:
                cv2.line(bgr, pl, pr, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(bgr, pl, 3, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(bgr, pr, 3, (0, 255, 255), -1, cv2.LINE_AA)

        for axis_idx, color in enumerate(axis_colors_bgr):
            end = center + scale * rotation[:, axis_idx]
            p1 = _project_point_m(end, intrinsic, width=w, height=h)
            if p1 is None:
                continue
            cv2.arrowedLine(bgr, p0, p1, color, 2, tipLength=0.28, line_type=cv2.LINE_AA)
            cv2.putText(
                bgr,
                axis_labels[axis_idx],
                (p1[0] + 4, p1[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )

    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out)


def detection_summary(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for idx, det in enumerate(detections, start=1):
        rows.append({"id": idx, "score": float(det.get("score", 0.0)), "bbox": det.get("bbox")})
    return rows


def clean_workspace_mask(
    depth_mm: np.ndarray,
    intrinsic: np.ndarray,
    *,
    factor_depth: float = 1000.0,
    base_mask: Optional[np.ndarray] = None,
    max_depth_mm: float = 2500.0,
    min_depth_mm: float = 50.0,
    depth_percentile_hi: float = 99.0,
    remove_statistical_outliers: bool = True,
    sor_nb_neighbors: int = 20,
    sor_std_ratio: float = 1.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a cleaned workspace mask to suppress flying pixels / sparse outliers.

    Steps:
      1) depth range + optional high percentile clip
      2) statistical outlier removal in 3D (cKDTree)
    """
    from scipy.spatial import cKDTree

    depth = np.asarray(depth_mm)
    h, w = depth.shape[:2]
    mask = (depth > 0).astype(bool)
    if base_mask is not None:
        mask &= base_mask.astype(bool)

    n0 = int(mask.sum())
    if min_depth_mm and min_depth_mm > 0:
        mask &= depth >= float(min_depth_mm)
    if max_depth_mm and max_depth_mm > 0:
        mask &= depth <= float(max_depth_mm)

    n_range = int(mask.sum())
    if depth_percentile_hi and 0 < float(depth_percentile_hi) < 100 and n_range > 0:
        hi = float(np.percentile(depth[mask].astype(np.float64), float(depth_percentile_hi)))
        mask &= depth <= hi

    n_perc = int(mask.sum())
    n_sor_removed = 0
    if remove_statistical_outliers and n_perc >= max(int(sor_nb_neighbors) + 2, 50):
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        ys, xs = np.where(mask)
        z = depth[ys, xs].astype(np.float64) / float(factor_depth)
        x = (xs.astype(np.float64) - cx) * z / fx
        y = (ys.astype(np.float64) - cy) * z / fy
        pts = np.stack([x, y, z], axis=1)

        k = int(max(3, sor_nb_neighbors))
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=k + 1, workers=-1)
        mean_d = dists[:, 1:].mean(axis=1)
        mu = float(mean_d.mean())
        sigma = float(mean_d.std())
        keep = mean_d <= (mu + float(sor_std_ratio) * sigma)
        n_sor_removed = int((~keep).sum())

        cleaned = np.zeros_like(mask, dtype=bool)
        cleaned[ys[keep], xs[keep]] = True
        mask = cleaned

    n_final = int(mask.sum())
    stats = {
        "pixels_raw": n0,
        "pixels_after_depth_range": n_range,
        "pixels_after_percentile": n_perc,
        "pixels_final": n_final,
        "sor_removed": n_sor_removed,
        "max_depth_mm": float(max_depth_mm) if max_depth_mm else None,
        "min_depth_mm": float(min_depth_mm) if min_depth_mm else None,
        "depth_percentile_hi": float(depth_percentile_hi) if depth_percentile_hi else None,
        "remove_statistical_outliers": bool(remove_statistical_outliers),
        "sor_nb_neighbors": int(sor_nb_neighbors),
        "sor_std_ratio": float(sor_std_ratio),
    }
    if n_final == 0:
        raise RuntimeError("离群点剔除后无有效点，请放宽 max_depth / SOR 参数")
    return mask, stats


def _instance_inlier_mask_3d(
    points_m: np.ndarray,
    *,
    std_ratio: float = 1.5,
    z_mad_ratio: float = 2.5,
    sor_nb_neighbors: int = 20,
) -> np.ndarray:
    """
    Per-instance inlier mask (Gen6D median/MAD + optional neighbor SOR).

    ``points_m``: (N,3) camera meters.
    """
    from scipy.spatial import cKDTree

    n = int(points_m.shape[0])
    if n < 8:
        return np.ones(n, dtype=bool)

    keep = np.ones(n, dtype=bool)
    center = np.median(points_m, axis=0)
    dist = np.linalg.norm(points_m - center, axis=1)
    d_med = float(np.median(dist))
    d_mad = float(np.median(np.abs(dist - d_med))) + 1e-6
    keep &= dist <= (d_med + float(std_ratio) * 1.4826 * d_mad)

    z = points_m[:, 2]
    if keep.any():
        z_med = float(np.median(z[keep]))
        z_mad = float(np.median(np.abs(z[keep] - z_med))) + 1e-6
    else:
        z_med = float(np.median(z))
        z_mad = 1e-6
    keep &= np.abs(z - z_med) <= (float(z_mad_ratio) * 1.4826 * z_mad)

    k = int(max(3, sor_nb_neighbors))
    if int(keep.sum()) >= k + 2:
        pts = points_m[keep]
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=k + 1, workers=-1)
        mean_d = dists[:, 1:].mean(axis=1)
        mu = float(mean_d.mean())
        sigma = float(mean_d.std()) + 1e-9
        keep_local = mean_d <= (mu + float(std_ratio) * sigma)
        idx = np.flatnonzero(keep)
        keep[:] = False
        keep[idx[keep_local]] = True

    if int(keep.sum()) < 3:
        return np.ones(n, dtype=bool)
    return keep


def clean_instance_id_map(
    depth_mm: np.ndarray,
    instance_id_map: np.ndarray,
    intrinsic: np.ndarray,
    *,
    factor_depth: float = 1000.0,
    std_ratio: float = 1.5,
    z_mad_ratio: float = 2.5,
    sor_nb_neighbors: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Remove statistical outliers **per SAM3 instance** (not global workspace).

    Returns:
      cleaned_id_map: outlier instance pixels set to 0
      removed_mask: True where an instance pixel was rejected
      stats: per-instance raw/inlier counts
    """
    depth = np.asarray(depth_mm)
    id_map = np.asarray(instance_id_map).copy()
    h, w = depth.shape[:2]
    if id_map.shape != (h, w):
        raise ValueError(f"id_map shape {id_map.shape} != depth {depth.shape}")

    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    removed = np.zeros((h, w), dtype=bool)
    per_inst: List[Dict[str, Any]] = []
    total_raw = 0
    total_kept = 0

    for inst_id in sorted(int(v) for v in np.unique(id_map) if int(v) > 0):
        ys, xs = np.where(id_map == inst_id)
        raw_n = int(ys.size)
        total_raw += raw_n
        if raw_n == 0:
            continue
        z = depth[ys, xs].astype(np.float64) / float(factor_depth)
        valid = z > 1e-6
        if not np.any(valid):
            id_map[ys, xs] = 0
            removed[ys, xs] = True
            per_inst.append({"instance_id": inst_id, "num_raw": raw_n, "num_inlier": 0})
            continue
        ys_v, xs_v, z_v = ys[valid], xs[valid], z[valid]
        x = (xs_v.astype(np.float64) - cx) * z_v / fx
        y = (ys_v.astype(np.float64) - cy) * z_v / fy
        pts = np.stack([x, y, z_v], axis=1)
        keep = _instance_inlier_mask_3d(
            pts,
            std_ratio=float(std_ratio),
            z_mad_ratio=float(z_mad_ratio),
            sor_nb_neighbors=int(sor_nb_neighbors),
        )
        drop_y = ys_v[~keep]
        drop_x = xs_v[~keep]
        # also drop invalid-depth pixels that belonged to this instance
        bad = ~valid
        if np.any(bad):
            removed[ys[bad], xs[bad]] = True
            id_map[ys[bad], xs[bad]] = 0
        if drop_y.size:
            removed[drop_y, drop_x] = True
            id_map[drop_y, drop_x] = 0
        kept_n = int(keep.sum())
        total_kept += kept_n
        per_inst.append(
            {
                "instance_id": inst_id,
                "num_raw": raw_n,
                "num_inlier": kept_n,
                "num_removed": int(raw_n - kept_n),
            }
        )

    stats = {
        "enabled": True,
        "mode": "per_instance",
        "pixels_instance_raw": total_raw,
        "pixels_instance_inlier": total_kept,
        "pixels_instance_removed": int(removed.sum()),
        "std_ratio": float(std_ratio),
        "z_mad_ratio": float(z_mad_ratio),
        "sor_nb_neighbors": int(sor_nb_neighbors),
        "instances": per_inst,
    }
    return id_map, removed, stats
