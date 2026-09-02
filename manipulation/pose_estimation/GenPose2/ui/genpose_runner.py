"""GenPose2 inference helpers for Gradio UI (SAM3 mask → 6D pose)."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from config import get_genpose2_conf, resolve_repo_path  # noqa: E402
from ui.pointcloud import (  # noqa: E402
    GRASP_CLOUD_MAX_POINTS,
    camera_pose_m_to_glb,
    create_pose_axes_mesh,
    create_pose_marker_sphere,
    depth_rgb_to_pointcloud,
    export_scene_files,
)

logger = logging.getLogger("genpose_runner")

_genpose_holder: Dict[str, Any] = {"model": None, "ckpts": None}


def _rotation_matrix_to_euler_zyx(rotation: List[List[float]]) -> List[float]:
    r00, r10 = float(rotation[0][0]), float(rotation[1][0])
    r20, r21, r22 = float(rotation[2][0]), float(rotation[2][1]), float(rotation[2][2])
    r01, r11 = float(rotation[0][1]), float(rotation[1][1])
    sy = math.sqrt(r00 * r00 + r10 * r10)
    if sy > 1e-6:
        rx = math.atan2(r21, r22)
        ry = math.atan2(-r20, sy)
        rz = math.atan2(r10, r00)
    else:
        rx = math.atan2(-r01, r11)
        ry = math.atan2(-r20, sy)
        rz = 0.0
    return [rx, ry, rz]


def _pose_4x4_to_t_and_R(pose_4x4: np.ndarray) -> Tuple[List[float], List[List[float]]]:
    t_mm = (pose_4x4[:3, 3].astype(float) * 1000.0).tolist()
    rot = pose_4x4[:3, :3].astype(float).tolist()
    return t_mm, rot


def _cube_corners_mm(
    position_mm: Any,
    rotation_3x3: Any,
    size_3d_m: Any,
) -> List[List[float]]:
    """目标空间正方体（包围盒）8 角点，相机系，单位 mm。"""
    pos = np.asarray(position_mm, dtype=np.float64).reshape(3)
    rot = np.asarray(rotation_3x3, dtype=np.float64).reshape(3, 3)
    size = np.asarray(size_3d_m, dtype=np.float64).reshape(3)
    hx, hy, hz = 0.5 * size * 1000.0
    local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    corners = (rot @ local.T).T + pos.reshape(1, 3)
    return [[round(float(v), 2) for v in row] for row in corners]


def format_grasp_xyzrxryrz(
    position_mm: Any,
    rotation_3x3: Any,
    *,
    size_3d_m: Optional[Any] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    抓取/物体 6D 位姿：``[x, y, z, rx, ry, rz]``（对齐 Gen6D 摘取点格式）。
    xyz 单位 mm；rx/ry/rz 为 °（旋转矩阵按 ZYX 欧拉角换算）。
    若提供 ``size_3d_m``，一并给出目标空间正方体尺寸与 8 角点。
    """
    pos = np.asarray(position_mm, dtype=np.float64).reshape(3)
    rot = np.asarray(rotation_3x3, dtype=np.float64).reshape(3, 3)
    euler_rad = _rotation_matrix_to_euler_zyx(rot.tolist())
    rx = round(float(np.degrees(euler_rad[0])), 3)
    ry = round(float(np.degrees(euler_rad[1])), 3)
    rz = round(float(np.degrees(euler_rad[2])), 3)
    xyz = [round(float(pos[0]), 2), round(float(pos[1]), 2), round(float(pos[2]), 2)]
    xyzrxryrz = [xyz[0], xyz[1], xyz[2], rx, ry, rz]
    out: Dict[str, Any] = {
        "success": True,
        "xyzrxryrz": xyzrxryrz,
        "frame": "camera",
        "preview_frame": "glb_y_up",
        "axis_hint": "X right, Y down, Z forward (mm)",
        "unit": {
            "xyz": "mm",
            "rx_ry_rz": "deg",
            "size_3d": "m",
            "size_3d_mm": "mm",
            "cube_corners_mm": "mm",
        },
        "euler_convention": "ZYX (rz,ry,rx) → displayed as [x,y,z,rx,ry,rz]",
        "position_mm": xyz,
        "rpy_deg": {"rx": rx, "ry": ry, "rz": rz},
        "rotation_matrix": [[round(float(v), 6) for v in row] for row in rot.tolist()],
    }
    if size_3d_m is not None:
        size = np.asarray(size_3d_m, dtype=np.float64).reshape(3)
        size_m = [round(float(v), 6) for v in size.tolist()]
        size_mm = [round(float(v) * 1000.0, 2) for v in size.tolist()]
        out["size_3d"] = size_m
        out["size_3d_mm"] = size_mm
        out["cube"] = {
            "center_mm": xyz,
            "size_m": size_m,
            "size_mm": size_mm,
            "corners_mm": _cube_corners_mm(xyz, rot, size),
            "frame": "camera",
            "note": "目标空间正方体：物体局部系 [±l/2, ±w/2, ±h/2] 变换到相机系",
        }
    if meta:
        out["meta"] = meta
    return out


def build_grasp_display_payload(
    poses_np: np.ndarray,
    lengths_np: np.ndarray,
    *,
    instance_scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """UI 展示用：首实例为主字段，全部实例列入 ``instances``（mm / ° + 正方体）。"""
    if poses_np is None or poses_np.shape[0] == 0:
        return {
            "success": False,
            "message": "尚未计算出抓取位姿",
            "xyzrxryrz": None,
            "cube": None,
            "unit": {
                "xyz": "mm",
                "rx_ry_rz": "deg",
                "size_3d": "m",
                "size_3d_mm": "mm",
                "cube_corners_mm": "mm",
            },
            "instances": [],
            "num_poses": 0,
        }

    instances: List[Dict[str, Any]] = []
    for i in range(poses_np.shape[0]):
        t_mm, rot = _pose_4x4_to_t_and_R(poses_np[i])
        score = (
            float(instance_scores[i])
            if instance_scores is not None and i < len(instance_scores)
            else 1.0
        )
        entry = format_grasp_xyzrxryrz(
            t_mm,
            rot,
            size_3d_m=lengths_np[i],
            meta={
                "instance_id": i + 1,
                "score": score,
                "role": "object_6d_pose",
            },
        )
        instances.append(entry)

    primary = dict(instances[0])
    primary["num_poses"] = len(instances)
    primary["instances"] = instances
    return primary


def camera_json_to_meta(camera: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    if "camera" in camera and "intrinsics" in camera.get("camera", {}):
        meta = dict(camera)
        meta.setdefault("depth_scale", float(camera.get("depth_scale", 0.001)))
        return meta
    if "cam_K" not in camera:
        raise ValueError("camera.json must contain cam_K or camera.intrinsics")
    k = camera["cam_K"]
    if len(k) != 9:
        raise ValueError("cam_K must have 9 elements (3x3 row-major)")
    return {
        "camera": {
            "intrinsics": {
                "fx": float(k[0]),
                "fy": float(k[4]),
                "cx": float(k[2]),
                "cy": float(k[5]),
                "width": int(width),
                "height": int(height),
            }
        },
        "depth_scale": float(camera.get("depth_scale", 0.001)),
    }


def load_depth_meters(depth_path: Path, depth_scale: float) -> np.ndarray:
    from cutoop.data_loader import Dataset

    suf = depth_path.suffix.lower()
    if suf == ".exr":
        return Dataset.load_depth(str(depth_path))
    if suf == ".npy":
        d = np.load(str(depth_path))
        d = np.asarray(d)
        if np.issubdtype(d.dtype, np.floating):
            return d.astype(np.float32)
        return d.astype(np.float32) * float(depth_scale)
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d is None:
        # fallback for 16-bit via PIL
        d = np.array(Image.open(depth_path))
    if d.ndim == 3:
        d = d[:, :, 0]
    d = np.asarray(d)
    if d.dtype == np.uint16 or (np.issubdtype(d.dtype, np.integer) and d.max() > 255):
        return d.astype(np.float32) * float(depth_scale)
    if np.issubdtype(d.dtype, np.floating):
        return d.astype(np.float32)
    return d.astype(np.float32) * float(depth_scale)


def resolve_ckpt_paths(
    score_ckpt: Optional[str] = None,
    energy_ckpt: Optional[str] = None,
    scale_ckpt: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    cfg = get_genpose2_conf()
    score = resolve_repo_path(score_ckpt or cfg.get("score_ckpt") or "results/ckpts/ScoreNet/scorenet.pth")
    energy = resolve_repo_path(
        energy_ckpt or cfg.get("energy_ckpt") or "results/ckpts/EnergyNet/energynet.pth"
    )
    scale = resolve_repo_path(scale_ckpt or cfg.get("scale_ckpt") or "results/ckpts/ScaleNet/scalenet.pth")
    for name, path in (("score", score), ("energy", energy), ("scale", scale)):
        if not path.is_file():
            raise FileNotFoundError(f"GenPose2 {name} checkpoint not found: {path}")
    return score, energy, scale


def get_or_load_genpose(
    score_ckpt: Optional[str] = None,
    energy_ckpt: Optional[str] = None,
    scale_ckpt: Optional[str] = None,
):
    """Lazy-load and cache GenPose2 model (reload if ckpt paths change).

    请尽量在主线程预加载（见 ``run_ui.preload_genpose2``），避免 Gradio
    worker 线程里首次初始化 CUDA 导致假死。
    """
    score, energy, scale = resolve_ckpt_paths(score_ckpt, energy_ckpt, scale_ckpt)
    key = (str(score), str(energy), str(scale))
    if _genpose_holder.get("model") is not None and _genpose_holder.get("ckpts") == key:
        return _genpose_holder["model"]

    argv_saved = sys.argv[:]
    sys.argv = [sys.argv[0] if sys.argv else "genpose_ui"]
    try:
        from runners.infer import create_genpose2

        logger.info("loading GenPose2: score=%s energy=%s scale=%s", score, energy, scale)
        print(
            f"[genpose_runner] loading GenPose2...\n"
            f"  score={score}\n  energy={energy}\n  scale={scale}",
            flush=True,
        )
        t0 = time.perf_counter()
        model = create_genpose2(
            score_model_path=str(score),
            energy_model_path=str(energy),
            scale_model_path=str(scale),
        )
        print(f"[genpose_runner] GenPose2 loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    finally:
        sys.argv = argv_saved

    _genpose_holder["model"] = model
    _genpose_holder["ckpts"] = key
    return model


def preload_genpose2(
    score_ckpt: Optional[str] = None,
    energy_ckpt: Optional[str] = None,
    scale_ckpt: Optional[str] = None,
):
    """在主线程预加载三网权重（启动 UI 时调用）。"""
    return get_or_load_genpose(score_ckpt, energy_ckpt, scale_ckpt)


def load_mask_u8(mask_path: Path) -> np.ndarray:
    """Load instance id mask (0=bg, 1..N=instances) from .exr/.png."""
    from cutoop.data_loader import Dataset

    path = Path(mask_path)
    if path.suffix.lower() == ".exr" and path.is_file():
        return Dataset.load_mask(str(path))
    # sam3_seg may write .png when OpenEXR unavailable
    png = path if path.suffix.lower() == ".png" else path.with_suffix(".png")
    if not png.is_file() and path.is_file():
        png = path
    m = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if m is None:
        m = np.array(Image.open(png))
    if m.ndim == 3:
        m = m[:, :, 0]
    return np.asarray(m).astype(np.uint8)


def build_poses_payload(
    poses_np: np.ndarray,
    lengths_np: np.ndarray,
    *,
    instance_scores: Optional[List[float]] = None,
    instance_bboxes: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    poses: List[Dict[str, Any]] = []
    for i in range(poses_np.shape[0]):
        t_mm, rot = _pose_4x4_to_t_and_R(poses_np[i])
        euler = _rotation_matrix_to_euler_zyx(rot)
        score = (
            float(instance_scores[i])
            if instance_scores is not None and i < len(instance_scores)
            else 1.0
        )
        entry: Dict[str, Any] = {
            "instance_id": i + 1,
            "score": score,
            "xyz_mm": t_mm,
            "rotation_matrix": rot,
            "rotation_euler_zyx_rad": euler,
            "xyzrxryrz": list(t_mm) + list(euler),
            "size_3d": lengths_np[i].astype(float).tolist(),
            "position_m": (poses_np[i][:3, 3].astype(float)).tolist(),
        }
        if instance_bboxes is not None and i < len(instance_bboxes):
            entry["bbox"] = instance_bboxes[i]
        poses.append(entry)

    first = poses[0] if poses else None
    return {
        "success": bool(poses),
        "frame": "camera",
        "preview_frame": "glb_y_up",
        "axis_hint": "X right, Y down, Z forward",
        "unit": {"xyz": "mm", "rx_ry_rz": "rad", "size_3d": "m"},
        "euler_convention": "ZYX (rz,ry,rx) → displayed as [x,y,z,rx,ry,rz]",
        "num_poses": len(poses),
        "poses": poses,
        "xyzrxryrz": first["xyzrxryrz"] if first else None,
        "xyz_mm": first["xyz_mm"] if first else None,
        "size_3d": first["size_3d"] if first else None,
    }


def export_pose_scene(
    *,
    depth_mm: np.ndarray,
    color_rgb: np.ndarray,
    intrinsic: np.ndarray,
    factor_depth: float,
    poses_np: np.ndarray,
    lengths_np: np.ndarray,
    run_dir: Path,
    stem: str = "scene_sam3_genpose2",
    max_points: int = GRASP_CLOUD_MAX_POINTS,
    axis_length_m: float = 0.08,
) -> Dict[str, str]:
    points, colors, _ = depth_rgb_to_pointcloud(
        depth_mm,
        color_rgb,
        intrinsic,
        factor_depth=factor_depth,
        max_points=int(max_points) if max_points else GRASP_CLOUD_MAX_POINTS,
    )
    geometries = []
    core_colors = (
        (255, 0, 255, 255),
        (0, 255, 255, 255),
        (255, 255, 0, 255),
        (255, 128, 0, 255),
        (0, 255, 128, 255),
    )
    for i in range(poses_np.shape[0]):
        t_m = poses_np[i][:3, 3]
        r = poses_np[i][:3, :3]
        t_glb, r_glb = camera_pose_m_to_glb(t_m, r)
        size = lengths_np[i]
        axis_len = float(axis_length_m)
        if size is not None and np.all(np.asarray(size) > 0):
            axis_len = max(float(np.max(size)) * 0.6, 0.04)
        geometries.append(
            create_pose_axes_mesh(t_glb, r_glb, axis_length_m=axis_len, radius_m=0.002)
        )
        geometries.append(
            create_pose_marker_sphere(
                t_glb,
                radius_m=0.006,
                color_rgba=core_colors[i % len(core_colors)],
            )
        )
    files = export_scene_files(
        points,
        colors,
        run_dir,
        stem=stem,
        extra_geometries=geometries or None,
    )
    return {"glb": str(files["glb_path"]), "ply": str(files["ply_path"])}


def run_genpose2_from_mask(
    *,
    color_rgb: np.ndarray,
    depth_m: np.ndarray,
    mask_u8: np.ndarray,
    meta: Dict[str, Any],
    score_ckpt: Optional[str] = None,
    energy_ckpt: Optional[str] = None,
    scale_ckpt: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
    """
    Run GenPose2 on prepared arrays.

    Returns:
      poses_np (N,4,4), lengths_np (N,3), pose_overlay_rgb (PIL)
    """
    from cutoop.data_loader import Dataset  # noqa: F401
    from datasets.datasets_infer import InferDataset
    from runners.infer import visualize_pose

    if color_rgb.shape[:2] != depth_m.shape[:2] or color_rgb.shape[:2] != mask_u8.shape[:2]:
        raise ValueError(
            f"color/depth/mask size mismatch: "
            f"color={color_rgb.shape[:2]} depth={depth_m.shape[:2]} mask={mask_u8.shape[:2]}"
        )
    if not np.any(mask_u8 > 0):
        raise RuntimeError("mask 无前景像素，无法估计位姿")

    genpose = get_or_load_genpose(score_ckpt, energy_ckpt, scale_ckpt)
    argv_saved = sys.argv[:]
    sys.argv = [sys.argv[0] if sys.argv else "genpose_ui"]
    try:
        data = InferDataset(
            {"depth": depth_m, "color": color_rgb, "mask": mask_u8, "meta": meta},
            img_size=genpose.cfg.img_size,
            device=genpose.cfg.device,
            n_pts=genpose.cfg.num_points,
        )
        pose, length = genpose.inference(data, prev_pose=None, tracking=False, tracking_T0=0.15)
        vis_bgr = visualize_pose(data, pose, length, visualize_pts=False, visualize_image=False)
    finally:
        sys.argv = argv_saved

    poses_np = pose[0].cpu().numpy()
    lengths_np = length[0].cpu().numpy()
    if poses_np.shape[0] == 0:
        raise RuntimeError("GenPose2 未返回任何位姿")
    vis_rgb = Image.fromarray(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB))
    return poses_np, lengths_np, vis_rgb
