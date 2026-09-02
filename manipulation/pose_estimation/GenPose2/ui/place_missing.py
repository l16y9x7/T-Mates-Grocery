"""缺货商品：由当前实例位姿 + VLM 位移估计目的 6D，并可视化。"""

from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ui.common import _project_point_m
from ui.genpose_runner import (
    _pose_4x4_to_t_and_R,
    _rotation_matrix_to_euler_zyx,
)
from ui.pointcloud import (
    GRASP_CLOUD_MAX_POINTS,
    camera_points_m_to_glb,
    camera_pose_m_to_glb,
    create_pose_axes_mesh,
    create_pose_marker_sphere,
    depth_rgb_to_pointcloud,
    export_scene_files,
)
from scripts.vlm_prompt import estimate_place_offset_from_image

logger = logging.getLogger("place_missing")

# 目的位姿显眼色（RGB）
DEST_CLOUD_RGB = (255, 32, 255)  # 品红
DEST_OVERLAY_BGR = (255, 0, 255)  # magenta in BGR for OpenCV draw

_CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)


@lru_cache(maxsize=8)
def _load_cjk_font(size: int) -> ImageFont.ImageFont:
    for path in _CJK_FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size, index=0)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def _needs_cjk(text: str) -> bool:
    return any(ord(ch) > 127 for ch in (text or ""))


def put_text_bgr(
    bgr: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color_bgr: Tuple[int, int, int],
    *,
    font_scale: float = 0.65,
    thickness: int = 2,
) -> None:
    """绘制标签；含中文时用 Noto CJK，避免 OpenCV 显示为 ???。"""
    text = str(text or "")
    if not text:
        return
    x, y = int(org[0]), int(org[1])
    if not _needs_cjk(text):
        cv2.putText(
            bgr,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(font_scale),
            color_bgr,
            int(thickness),
            cv2.LINE_AA,
        )
        return

    # PIL: org 为文字基线附近左下；转为左上绘制
    font_px = max(14, int(round(32 * float(font_scale))))
    font = _load_cjk_font(font_px)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    # 估算高度，把基线 y 转为顶边
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        th = int(bbox[3] - bbox[1])
    except Exception:  # noqa: BLE001
        th = font_px
    top_left = (x, max(0, y - th))
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    # 描边，提高可读性
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)):
        draw.text((top_left[0] + dx, top_left[1] + dy), text, font=font, fill=(0, 0, 0))
    draw.text(top_left, text, font=font, fill=color_rgb)
    bgr[:, :, :] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# 多实例 mask 调色板（BGR）
_INSTANCE_COLORS_BGR = (
    (0, 255, 255),  # yellow
    (255, 128, 0),  # blue-ish
    (0, 255, 0),  # green
    (0, 128, 255),  # orange
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (128, 0, 255),  # pink
    (0, 200, 200),
)


def pick_best_pose_index(poses_payload: Dict[str, Any]) -> int:
    poses = poses_payload.get("poses") or []
    if not poses:
        raise RuntimeError("无可用实例位姿")
    best_i = 0
    best_s = float(poses[0].get("score") or 0.0)
    for i, p in enumerate(poses):
        s = float(p.get("score") or 0.0)
        if s > best_s:
            best_s = s
            best_i = i
    return best_i


def find_pose_index_by_instance_id(
    poses_payload: Dict[str, Any], instance_id: Optional[int]
) -> int:
    poses = poses_payload.get("poses") or []
    if not poses:
        raise RuntimeError("无可用实例位姿")
    if instance_id is None:
        return pick_best_pose_index(poses_payload)
    for i, p in enumerate(poses):
        if int(p.get("instance_id") or (i + 1)) == int(instance_id):
            return i
    logger.warning(
        "VLM instance_id=%s 不在候选中，回退最高置信度", instance_id
    )
    return pick_best_pose_index(poses_payload)


def pose_entry_to_4x4(entry: Dict[str, Any]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(entry["rotation_matrix"], dtype=np.float64)
    pose[:3, 3] = np.asarray(entry["position_m"], dtype=np.float64)
    return pose


def build_instances_catalog(poses_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(poses_payload.get("poses") or []):
        out.append(
            {
                "instance_id": int(p.get("instance_id") or (i + 1)),
                "score": float(p.get("score") or 0.0),
                "xyz_mm": list(p.get("xyz_mm") or []),
                "size_3d": list(p.get("size_3d") or [0.05, 0.05, 0.05]),
                "bbox": p.get("bbox"),
                "position_m": list(p.get("position_m") or []),
            }
        )
    return out


def compute_spatial_place_prior(
    instances: List[Dict[str, Any]],
    *,
    column_bin_mm: float = 70.0,
    empty_gap_mm: float = 50.0,
) -> Dict[str, Any]:
    """根据实例 xyz 推断「哪一列缺前排」及建议源实例/目的 xyz。

    规则：按 X 分列；全局前排深度 front_z_ref≈各实例较浅的 Z；
    若某列最前实例仍明显更深，则该列前排空，建议把该列最前（仍偏深）实例挪到 front_z_ref。
    """
    items: List[Dict[str, Any]] = []
    for c in instances:
        xyz = c.get("xyz_mm") or []
        if len(xyz) < 3:
            continue
        items.append(
            {
                "instance_id": int(c.get("instance_id") or 0),
                "score": float(c.get("score") or 0.0),
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "size_3d": list(c.get("size_3d") or [0.05, 0.05, 0.05]),
            }
        )
    if not items:
        return {"success": False, "message": "无有效 xyz"}

    zs = sorted(float(i["z"]) for i in items)
    k = max(1, (len(zs) + 2) // 3)
    front_z_ref = float(sum(zs[:k]) / k)
    min_z = float(zs[0])

    # 按 X 聚类成列
    ordered = sorted(items, key=lambda t: t["x"])
    columns: List[List[Dict[str, Any]]] = []
    for it in ordered:
        if not columns:
            columns.append([it])
            continue
        col_x = float(sum(c["x"] for c in columns[-1]) / len(columns[-1]))
        if abs(it["x"] - col_x) > float(column_bin_mm):
            columns.append([it])
        else:
            columns[-1].append(it)

    empty_columns: List[Dict[str, Any]] = []
    for col in columns:
        col_sorted = sorted(col, key=lambda t: t["z"])
        front = col_sorted[0]
        gap = float(front["z"] - front_z_ref)
        col_x = float(sum(c["x"] for c in col) / len(col))
        col_y = float(sum(c["y"] for c in col) / len(col))
        info = {
            "column_x_mm": round(col_x, 1),
            "column_y_mm": round(col_y, 1),
            "front_instance_id": int(front["instance_id"]),
            "front_z_mm": round(float(front["z"]), 1),
            "gap_to_global_front_mm": round(gap, 1),
            "member_ids": [int(c["instance_id"]) for c in col_sorted],
            "is_empty_front": bool(gap >= float(empty_gap_mm)),
        }
        if info["is_empty_front"]:
            dest = [round(float(front["x"]), 1), round(float(front["y"]), 1), round(front_z_ref, 1)]
            info["suggested_source_id"] = int(front["instance_id"])
            info["suggested_dest_xyz_mm"] = dest
            info["suggested_offset_mm"] = {
                "right_mm": round(dest[0] - float(front["x"]), 1),
                "down_mm": round(dest[1] - float(front["y"]), 1),
                "forward_mm": round(dest[2] - float(front["z"]), 1),
            }
            empty_columns.append(info)

    empty_columns.sort(key=lambda e: -float(e["gap_to_global_front_mm"]))
    suggested = empty_columns[0] if empty_columns else None
    return {
        "success": True,
        "front_z_ref_mm": round(front_z_ref, 1),
        "min_z_mm": round(min_z, 1),
        "max_z_mm": round(float(zs[-1]), 1),
        "num_columns": len(columns),
        "empty_columns": empty_columns,
        "suggested": suggested,
        "rule": (
            "Empty front slot = column whose nearest instance is still deeper than "
            "global front_z_ref; move that instance toward camera onto front_z_ref, "
            "keeping roughly the same X/Y (same column)."
        ),
    }


def offset_from_src_to_dest_mm(
    src_xyz_mm: Sequence[float], dest_xyz_mm: Sequence[float]
) -> Dict[str, float]:
    return {
        "right_mm": float(dest_xyz_mm[0]) - float(src_xyz_mm[0]),
        "down_mm": float(dest_xyz_mm[1]) - float(src_xyz_mm[1]),
        "forward_mm": float(dest_xyz_mm[2]) - float(src_xyz_mm[2]),
    }


def resolve_place_with_spatial_prior(
    *,
    catalog: List[Dict[str, Any]],
    vlm_instance_id: Optional[int],
    vlm_offset_mm: Dict[str, float],
    prior: Dict[str, Any],
) -> Dict[str, Any]:
    """用空间先验校正明显不合理的 VLM 位移（例如把前排货挪出货架）。"""
    by_id = {int(c["instance_id"]): c for c in catalog if c.get("xyz_mm")}
    front_z = float((prior or {}).get("front_z_ref_mm") or 0.0)
    suggested = (prior or {}).get("suggested") or None

    chosen_id = int(vlm_instance_id) if vlm_instance_id is not None else None
    if chosen_id is None or chosen_id not in by_id:
        if suggested:
            chosen_id = int(suggested["suggested_source_id"])
        else:
            # 回退最深实例
            chosen_id = max(by_id.values(), key=lambda c: float(c["xyz_mm"][2]))[
                "instance_id"
            ]

    src = by_id[int(chosen_id)]
    src_xyz = [float(v) for v in src["xyz_mm"][:3]]
    off = {
        "right_mm": float(vlm_offset_mm.get("right_mm") or 0.0),
        "down_mm": float(vlm_offset_mm.get("down_mm") or 0.0),
        "forward_mm": float(vlm_offset_mm.get("forward_mm") or 0.0),
    }
    dest_xyz = [src_xyz[0] + off["right_mm"], src_xyz[1] + off["down_mm"], src_xyz[2] + off["forward_mm"]]

    reason_bits: List[str] = []
    used_prior = False

    # 1) 目的 Z 不得明显浅于前排（避免飞出货架）
    if front_z > 1 and dest_xyz[2] < front_z - 40.0:
        reason_bits.append(
            f"clamp dest Z {dest_xyz[2]:.0f}→{front_z:.0f} (not in front of shelf)"
        )
        dest_xyz[2] = front_z
        used_prior = True

    # 2) 若选了已在全局前排的实例还继续大幅前移 → 改用空列先验
    already_front = front_z > 1 and (src_xyz[2] - front_z) < 40.0
    if already_front and off["forward_mm"] < -40.0 and suggested:
        chosen_id = int(suggested["suggested_source_id"])
        src = by_id[chosen_id]
        src_xyz = [float(v) for v in src["xyz_mm"][:3]]
        dest_xyz = [float(v) for v in suggested["suggested_dest_xyz_mm"]]
        used_prior = True
        reason_bits.append(
            f"reject moving already-front id; use empty-column prior id={chosen_id}"
        )
    elif suggested and int(chosen_id) != int(suggested["suggested_source_id"]):
        # 3) VLM 选的列没有空位、先验有空位：若 VLM 位移很小或把货挪歪，采用先验
        sug_src = by_id.get(int(suggested["suggested_source_id"]))
        if sug_src is not None:
            vlm_col_empty = any(
                int(chosen_id) in (e.get("member_ids") or []) and e.get("is_empty_front")
                for e in (prior.get("empty_columns") or [])
            )
            if not vlm_col_empty:
                chosen_id = int(suggested["suggested_source_id"])
                src = sug_src
                src_xyz = [float(v) for v in src["xyz_mm"][:3]]
                dest_xyz = [float(v) for v in suggested["suggested_dest_xyz_mm"]]
                used_prior = True
                reason_bits.append(
                    f"prefer empty-column source id={chosen_id} over VLM id"
                )

    # 4) 若有空列先验且当前选择就是该源，强制目的对齐 front_z / 同列 X（轻微纠偏）
    if suggested and int(chosen_id) == int(suggested["suggested_source_id"]):
        sug_dest = [float(v) for v in suggested["suggested_dest_xyz_mm"]]
        # X 偏差过大则拉回同列
        if abs(dest_xyz[0] - sug_dest[0]) > 40.0:
            dest_xyz[0] = sug_dest[0]
            used_prior = True
            reason_bits.append("align dest X to empty column")
        if abs(dest_xyz[2] - sug_dest[2]) > 25.0:
            dest_xyz[2] = sug_dest[2]
            used_prior = True
            reason_bits.append("align dest Z to front_z_ref")

    off = offset_from_src_to_dest_mm(src_xyz, dest_xyz)
    return {
        "instance_id": int(chosen_id),
        "offset_mm": off,
        "dest_xyz_mm": [round(v, 1) for v in dest_xyz],
        "src_xyz_mm": [round(v, 1) for v in src_xyz],
        "used_spatial_prior": used_prior,
        "spatial_note": "; ".join(reason_bits),
        "front_z_ref_mm": front_z,
    }


def offset_mm_to_delta_m(offset_mm: Dict[str, float]) -> np.ndarray:
    """camera: +X right, +Y down, +Z forward."""
    return np.array(
        [
            float(offset_mm.get("right_mm", 0.0)) / 1000.0,
            float(offset_mm.get("down_mm", 0.0)) / 1000.0,
            float(offset_mm.get("forward_mm", 0.0)) / 1000.0,
        ],
        dtype=np.float64,
    )


def apply_translation_to_pose(pose_4x4: np.ndarray, delta_m: np.ndarray) -> np.ndarray:
    out = np.asarray(pose_4x4, dtype=np.float64).copy()
    out[:3, 3] = out[:3, 3] + np.asarray(delta_m, dtype=np.float64).reshape(3)
    return out


def pose_entry_from_4x4(
    pose_4x4: np.ndarray,
    *,
    size_3d: Sequence[float],
    score: float,
    instance_id: int,
    bbox: Any = None,
) -> Dict[str, Any]:
    t_mm, rot = _pose_4x4_to_t_and_R(pose_4x4)
    euler = _rotation_matrix_to_euler_zyx(rot)
    entry: Dict[str, Any] = {
        "instance_id": int(instance_id),
        "score": float(score),
        "xyz_mm": t_mm,
        "rotation_matrix": rot,
        "rotation_euler_zyx_rad": euler,
        "xyzrxryrz": list(t_mm) + list(euler),
        "size_3d": [float(v) for v in size_3d],
        "position_m": pose_4x4[:3, 3].astype(float).tolist(),
    }
    if bbox is not None:
        entry["bbox"] = bbox
    return entry


def extract_instance_points_cam_m(
    depth_mm: np.ndarray,
    mask_u8: np.ndarray,
    intrinsic: np.ndarray,
    instance_id: int,
    *,
    factor_depth: float = 1000.0,
    max_points: int = 20000,
) -> np.ndarray:
    """Back-project one instance to camera-frame meters (N,3)."""
    depth = np.asarray(depth_mm)
    mask = np.asarray(mask_u8)
    h, w = depth.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST)
        )
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    sel = (mask == int(instance_id)) & (depth > 0)
    if not np.any(sel):
        return np.empty((0, 3), dtype=np.float32)
    vs, us = np.where(sel)
    z = depth[vs, us].astype(np.float64) / float(factor_depth)
    x = (us.astype(np.float64) - cx) * z / fx
    y = (vs.astype(np.float64) - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(pts.shape[0], max_points, replace=False)]
    return pts.astype(np.float32)


def annotate_source_for_vlm(
    image: Image.Image,
    *,
    pose: np.ndarray,
    size_3d: Sequence[float],
    intrinsic: np.ndarray,
    product_name: str,
    bbox: Any = None,
) -> Image.Image:
    """兼容：单实例标注（黄框）。"""
    return annotate_instances_for_vlm(
        image,
        poses_entries=[
            {
                "instance_id": 1,
                "pose_4x4": pose,
                "size_3d": size_3d,
                "bbox": bbox,
                "xyz_mm": (np.asarray(pose)[:3, 3] * 1000.0).tolist(),
                "score": 1.0,
            }
        ],
        mask_u8=None,
        intrinsic=intrinsic,
        product_name=product_name,
    )


def annotate_instances_for_vlm(
    image: Image.Image,
    *,
    poses_entries: List[Dict[str, Any]],
    mask_u8: Optional[np.ndarray],
    intrinsic: np.ndarray,
    product_name: str,
) -> Image.Image:
    """在 RGB 上叠加所有同款实例 mask + ID + 投影位置，供 VLM 选型。"""
    rgb = np.array(image.convert("RGB"))
    base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = base.shape[:2]
    color_layer = base.copy()

    mask = None
    if mask_u8 is not None:
        mask = np.asarray(mask_u8)
        if mask.shape[:2] != (h, w):
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8)).resize((w, h), Image.NEAREST)
            )

    for idx, entry in enumerate(poses_entries):
        iid = int(entry.get("instance_id") or (idx + 1))
        color = _INSTANCE_COLORS_BGR[(iid - 1) % len(_INSTANCE_COLORS_BGR)]
        if mask is not None:
            sel = mask == iid
            if np.any(sel):
                color_layer[sel] = color

    if mask is not None:
        bgr = cv2.addWeighted(color_layer, 0.45, base, 0.55, 0)
    else:
        bgr = base.copy()

    for idx, entry in enumerate(poses_entries):
        iid = int(entry.get("instance_id") or (idx + 1))
        color = _INSTANCE_COLORS_BGR[(iid - 1) % len(_INSTANCE_COLORS_BGR)]
        if mask is not None:
            sel = mask == iid
            if np.any(sel):
                sel_u8 = sel.astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    sel_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(bgr, contours, -1, color, 2, cv2.LINE_AA)

        bbox = entry.get("bbox")
        if bbox is not None and len(bbox) >= 4:
            a, b, c, d = [float(v) for v in bbox[:4]]
            if c > a and d > b and c <= w * 1.5 and d <= h * 1.5:
                x1, y1, x2, y2 = int(a), int(b), int(c), int(d)
            else:
                x1, y1 = int(a), int(b)
                x2, y2 = int(a + c), int(b + d)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        pose = entry.get("pose_4x4")
        if pose is None and "rotation_matrix" in entry and "position_m" in entry:
            pose = pose_entry_to_4x4(entry)
        if pose is not None:
            pose = np.asarray(pose, dtype=np.float64)
            center = pose[:3, 3]
            p0 = _project_point_m(center, intrinsic, width=w, height=h)
            if p0 is not None:
                cv2.circle(bgr, p0, 7, color, -1, cv2.LINE_AA)
                xyz = entry.get("xyz_mm") or (center * 1000.0).tolist()
                label = f"id={iid}"
                if len(xyz) >= 3:
                    label += f" z={float(xyz[2]):.0f}"
                put_text_bgr(
                    bgr,
                    label,
                    (p0[0] + 8, max(18, p0[1] - 8)),
                    color,
                    font_scale=0.65,
                    thickness=2,
                )

    put_text_bgr(
        bgr,
        f"{product_name}: 选择 instance_id + 移动",
        (12, 28),
        (255, 255, 255),
        font_scale=0.75,
        thickness=2,
    )
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def render_destination_overlay(
    image: Image.Image,
    *,
    src_pose: np.ndarray,
    dest_pose: np.ndarray,
    size_3d: Sequence[float],
    intrinsic: np.ndarray,
    label: str = "place",
    axis_length_m: float = 0.08,
) -> Image.Image:
    """Draw source (thin) + destination (thick magenta) axes on RGB."""
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    size = np.asarray(size_3d, dtype=np.float64).reshape(3)
    axis_len = max(float(np.max(size)) * 0.6, float(axis_length_m))

    def _draw(pose: np.ndarray, color_bgr: Tuple[int, int, int], thickness: int, tag: str) -> None:
        center = pose[:3, 3]
        rot = pose[:3, :3]
        p0 = _project_point_m(center, intrinsic, width=w, height=h)
        if p0 is None:
            return
        cv2.circle(bgr, p0, 7 if thickness > 2 else 4, color_bgr, -1, cv2.LINE_AA)
        put_text_bgr(
            bgr,
            tag,
            (p0[0] + 8, max(16, p0[1] - 8)),
            color_bgr,
            font_scale=0.65,
            thickness=2,
        )
        axis_colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
        for i, ac in enumerate(axis_colors):
            end = center + axis_len * rot[:, i]
            p1 = _project_point_m(end, intrinsic, width=w, height=h)
            if p1 is None:
                continue
            cv2.arrowedLine(bgr, p0, p1, ac, thickness, tipLength=0.25, line_type=cv2.LINE_AA)
        hx, hy, hz = 0.5 * size
        corners = np.array(
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
        corners = (rot @ corners.T).T + center.reshape(1, 3)
        pts2d: List[Optional[Tuple[int, int]]] = []
        for c in corners:
            pts2d.append(_project_point_m(c, intrinsic, width=w, height=h))
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in edges:
            if pts2d[a] is None or pts2d[b] is None:
                continue
            cv2.line(bgr, pts2d[a], pts2d[b], color_bgr, thickness, cv2.LINE_AA)

    _draw(src_pose, (180, 180, 180), 1, "src")
    _draw(dest_pose, DEST_OVERLAY_BGR, 3, label)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def export_place_destination_glb(
    *,
    depth_mm: np.ndarray,
    color_rgb: np.ndarray,
    intrinsic: np.ndarray,
    factor_depth: float,
    src_pose: np.ndarray,
    dest_pose: np.ndarray,
    size_3d: Sequence[float],
    instance_points_cam: np.ndarray,
    delta_m: np.ndarray,
    run_dir: Path,
    stem: str = "scene_place_dest",
    max_points: int = GRASP_CLOUD_MAX_POINTS,
) -> Dict[str, str]:
    """Full shelf cloud + magenta moved instance + destination axes."""
    points, colors, _ = depth_rgb_to_pointcloud(
        depth_mm,
        color_rgb,
        intrinsic,
        factor_depth=factor_depth,
        max_points=int(max_points) if max_points else GRASP_CLOUD_MAX_POINTS,
    )
    geometries = []
    size = np.asarray(size_3d, dtype=np.float64).reshape(3)
    axis_len = max(float(np.max(size)) * 0.6, 0.05)

    t_s, r_s = camera_pose_m_to_glb(src_pose[:3, 3], src_pose[:3, :3])
    geometries.append(
        create_pose_axes_mesh(t_s, r_s, axis_length_m=axis_len * 0.8, radius_m=0.0015)
    )
    t_d, r_d = camera_pose_m_to_glb(dest_pose[:3, 3], dest_pose[:3, :3])
    geometries.append(
        create_pose_axes_mesh(t_d, r_d, axis_length_m=axis_len * 1.2, radius_m=0.0035)
    )
    geometries.append(
        create_pose_marker_sphere(t_d, radius_m=0.01, color_rgba=(255, 0, 255, 255))
    )

    pts = np.asarray(instance_points_cam, dtype=np.float64)
    if pts.size:
        pts_moved = pts + np.asarray(delta_m, dtype=np.float64).reshape(1, 3)
        pts_glb = camera_points_m_to_glb(pts_moved)
        n = pts_glb.shape[0]
        cols = np.tile(np.asarray(DEST_CLOUD_RGB, dtype=np.uint8), (n, 1))
        points = np.concatenate([points, pts_glb.astype(np.float32)], axis=0)
        colors = np.concatenate([colors, cols], axis=0)

    files = export_scene_files(
        points,
        colors,
        run_dir,
        stem=stem,
        extra_geometries=geometries or None,
    )
    return {"glb": str(files["glb_path"]), "ply": str(files["ply_path"])}


def run_place_missing_stage(
    *,
    run_dir: Path,
    product_name: str,
    rgb: Image.Image,
    poses_payload: Dict[str, Any],
    vlm_api_url: str,
    vlm_model: str,
    vlm_timeout_s: float,
    max_points: int = GRASP_CLOUD_MAX_POINTS,
    identify_dialogue: str = "",
) -> Dict[str, Any]:
    """
    After SAM3+GenPose2 artifacts exist in ``run_dir``, estimate place offset via VLM
    and export destination pose / overlays.
    """
    run_dir = Path(run_dir)
    poses_path = run_dir / "poses.json"
    depth_path = run_dir / "depth_mm.npy"
    mask_path = run_dir / "mask_u8.npy"
    K_path = run_dir / "intrinsic.json"
    for p in (poses_path, depth_path, mask_path, K_path):
        if not p.is_file():
            raise FileNotFoundError(f"缺少放置阶段输入: {p}")

    if not poses_payload:
        poses_payload = json.loads(poses_path.read_text(encoding="utf-8"))
    if not (poses_payload.get("poses") or []):
        raise RuntimeError("无可用实例位姿")

    meta_k = json.loads(K_path.read_text(encoding="utf-8"))
    intrinsic = np.asarray(meta_k["intrinsic"], dtype=np.float64).reshape(3, 3)
    factor_depth = float(meta_k.get("factor_depth") or 1000.0)
    depth_mm = np.load(str(depth_path))
    mask_u8 = np.load(str(mask_path))
    color_np = np.asarray(rgb.convert("RGB"), dtype=np.uint8)

    # 所有同款实例：mask + 空间位姿 → 交给 VLM 选型
    catalog = build_instances_catalog(poses_payload)
    pose_entries_for_viz: List[Dict[str, Any]] = []
    for p in poses_payload["poses"]:
        e = dict(p)
        e["pose_4x4"] = pose_entry_to_4x4(p)
        pose_entries_for_viz.append(e)

    vlm_img = annotate_instances_for_vlm(
        rgb,
        poses_entries=pose_entries_for_viz,
        mask_u8=mask_u8,
        intrinsic=intrinsic,
        product_name=product_name,
    )
    vlm_img_path = run_dir / "vlm_place_input.png"
    vlm_img.save(vlm_img_path)

    # 空间先验：哪一列缺前排、建议源实例与目的 xyz
    spatial_prior = compute_spatial_place_prior(catalog)
    logger.info("spatial place prior: %s", json.dumps(spatial_prior, ensure_ascii=False))

    fallback_i = pick_best_pose_index(poses_payload)
    fallback = poses_payload["poses"][fallback_i]
    fallback_size = list(fallback.get("size_3d") or [0.05, 0.05, 0.05])
    fallback_xyz = list(
        fallback.get("xyz_mm")
        or (np.asarray(fallback.get("position_m") or [0, 0, 0]) * 1000.0).tolist()
    )

    t_vlm = time.perf_counter()
    vlm = estimate_place_offset_from_image(
        vlm_img,
        product_name,
        xyz_mm=fallback_xyz,
        size_3d_m=fallback_size,
        score=float(fallback.get("score") or 0.0),
        identify_dialogue=identify_dialogue,
        instances=catalog,
        spatial_prior=spatial_prior,
        api_url=vlm_api_url,
        model=vlm_model,
        timeout_s=vlm_timeout_s,
    )
    vlm_s = time.perf_counter() - t_vlm

    resolved = resolve_place_with_spatial_prior(
        catalog=catalog,
        vlm_instance_id=vlm.get("instance_id"),
        vlm_offset_mm=vlm.get("offset_mm") or {},
        prior=spatial_prior,
    )
    chosen_i = find_pose_index_by_instance_id(
        poses_payload, resolved.get("instance_id")
    )
    src_entry = poses_payload["poses"][chosen_i]
    instance_id = int(src_entry.get("instance_id") or (chosen_i + 1))
    src_pose = pose_entry_to_4x4(src_entry)
    size_3d = list(src_entry.get("size_3d") or [0.05, 0.05, 0.05])
    score = float(src_entry.get("score") or 0.0)
    xyz_mm = list(src_entry.get("xyz_mm") or (src_pose[:3, 3] * 1000.0).tolist())

    offset_mm = resolved["offset_mm"]
    delta_m = offset_mm_to_delta_m(offset_mm)
    dest_pose = apply_translation_to_pose(src_pose, delta_m)
    dest_entry = pose_entry_from_4x4(
        dest_pose,
        size_3d=size_3d,
        score=score,
        instance_id=instance_id,
        bbox=src_entry.get("bbox"),
    )
    dest_entry["source_instance_id"] = instance_id
    dest_entry["place_offset_mm"] = offset_mm
    dest_entry["place_delta_m"] = delta_m.astype(float).tolist()
    dest_entry["vlm_reason"] = vlm.get("reason") or ""
    dest_entry["spatial_note"] = resolved.get("spatial_note") or ""
    if resolved.get("used_spatial_prior"):
        dest_entry["selected_by"] = "spatial_prior+vlm"
    elif vlm.get("instance_id") is not None:
        dest_entry["selected_by"] = "vlm"
    else:
        dest_entry["selected_by"] = "fallback_score"

    inst_pts = extract_instance_points_cam_m(
        depth_mm,
        mask_u8,
        intrinsic,
        instance_id,
        factor_depth=factor_depth,
    )
    overlay = render_destination_overlay(
        rgb,
        src_pose=src_pose,
        dest_pose=dest_pose,
        size_3d=size_3d,
        intrinsic=intrinsic,
        label=f"dest:{product_name or 'place'}",
    )
    overlay_path = run_dir / "vis_place_dest.png"
    overlay.save(overlay_path)

    files = export_place_destination_glb(
        depth_mm=depth_mm,
        color_rgb=color_np,
        intrinsic=intrinsic,
        factor_depth=factor_depth,
        src_pose=src_pose,
        dest_pose=dest_pose,
        size_3d=size_3d,
        instance_points_cam=inst_pts,
        delta_m=delta_m,
        run_dir=run_dir,
        stem="scene_place_dest",
        max_points=int(max_points) if max_points else GRASP_CLOUD_MAX_POINTS,
    )

    place_payload = {
        "success": True,
        "frame": "camera",
        "axis_hint": "X right, Y down, Z forward",
        "unit": {"xyz": "mm", "rx_ry_rz": "rad", "size_3d": "m", "offset": "mm"},
        "product_name": product_name,
        "candidate_instances": catalog,
        "spatial_prior": spatial_prior,
        "spatial_resolve": resolved,
        "selected_instance_id": instance_id,
        "selected_by": dest_entry["selected_by"],
        "vlm_reason": vlm.get("reason") or "",
        "spatial_note": dest_entry.get("spatial_note") or "",
        "source_pose": src_entry,
        "destination_pose": dest_entry,
        "place_offset_mm": offset_mm,
        "place_delta_m": delta_m.astype(float).tolist(),
        "xyzrxryrz": dest_entry["xyzrxryrz"],
        "xyz_mm": dest_entry["xyz_mm"],
        "identify_dialogue": (identify_dialogue or "").strip(),
        "vlm": {
            "raw_reply": vlm["raw_reply"],
            "prompt": vlm.get("prompt"),
            "decision": vlm.get("decision"),
            "requested_instance_id": vlm.get("instance_id"),
            "elapsed_s": round(vlm_s, 3),
            "input_image": str(vlm_img_path),
            "used_identify_dialogue": bool((identify_dialogue or "").strip()),
            "used_spatial_prior": bool(resolved.get("used_spatial_prior")),
        },
        "overlay": str(overlay_path),
        "scene_glb": files["glb"],
        "scene_ply": files["ply"],
        "num_instance_points": int(inst_pts.shape[0]),
    }
    (run_dir / "place_destination.json").write_text(
        json.dumps(place_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "place select id=%s by=%s xyz_mm=%s offset_mm=%s",
        instance_id,
        dest_entry["selected_by"],
        dest_entry["xyz_mm"],
        offset_mm,
    )
    return place_payload
