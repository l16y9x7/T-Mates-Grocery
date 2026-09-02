"""
读取 GenPose / cutoop 约定的 mask.exr（实例 id 编码），按物体 id 着色并用 OpenCV 显示或保存。

示例::

    python utils/exr_visualize.py learning/inputs/1_mask.exr \\
        --rgb learning/inputs/1_.png \\
        -o learning/outputs/1_mask_vis.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def load_instance_mask_exr(exr_path: str | Path) -> np.ndarray:
    """
    读取 mask.exr，返回 (H, W) uint8，像素值为实例 id（0=背景，1..N=物体）。
    与 cutoop.data_loader.Dataset.load_mask 一致。
    """
    exr_path = str(exr_path)
    try:
        from cutoop.data_loader import Dataset

        return Dataset.load_mask(exr_path)
    except ImportError:
        img = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f"无法读取 EXR: {exr_path}")
        if img.ndim == 3:
            img = img[:, :, 2]
        return np.asarray(img * 255, dtype=np.uint8)


def _color_bgr_for_id(obj_id: int) -> Tuple[int, int, int]:
    """为实例 id 生成稳定、可区分的 BGR 颜色（id=0 不使用）。"""
    if obj_id <= 0:
        return (0, 0, 0)
    hue = int((obj_id * 47) % 180)
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def colorize_instance_mask(mask: np.ndarray) -> np.ndarray:
    """将实例 id 图转为 BGR 彩色图，每个 id 一种颜色，背景为黑。"""
    if mask.ndim != 2:
        raise ValueError("mask 必须是二维 (H, W)")
    h, w = mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for obj_id in np.unique(mask):
        oid = int(obj_id)
        if oid == 0:
            continue
        vis[mask == oid] = _color_bgr_for_id(oid)
    return vis


def draw_instance_legend(
    image: np.ndarray,
    mask: np.ndarray,
    origin: Tuple[int, int] = (10, 24),
    line_height: int = 22,
) -> np.ndarray:
    """在图像左上角绘制实例 id 图例。"""
    out = image.copy()
    ids = sorted(int(i) for i in np.unique(mask) if int(i) != 0)
    x, y = origin
    for obj_id in ids:
        color = _color_bgr_for_id(obj_id)
        cv2.rectangle(out, (x, y - 14), (x + 18, y + 2), color, -1)
        cv2.putText(
            out,
            f"id={obj_id}",
            (x + 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"id={obj_id}",
            (x + 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        y += line_height
    return out


def overlay_mask_on_bgr(
    bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """将彩色实例 mask 半透明叠到 BGR 底图上（仅前景区域）。"""
    if bgr.shape[:2] != mask.shape[:2]:
        raise ValueError(f"RGB 与 mask 尺寸不一致: {bgr.shape[:2]} vs {mask.shape[:2]}")
    colored = colorize_instance_mask(mask)
    fg = mask > 0
    out = bgr.astype(np.float32)
    blend = (
        alpha * colored.astype(np.float32) + (1.0 - alpha) * out
    ).astype(np.uint8)
    out[fg] = blend[fg]
    return out


def visualize_mask_exr(
    exr_path: str | Path,
    *,
    rgb_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    show: bool = False,
    alpha: float = 0.55,
    legend: bool = True,
) -> np.ndarray:
    """
    读取 mask.exr 并渲染；若提供 rgb_path 则叠加到底图，否则只输出彩色 mask。

    :return: BGR 可视化图像
    """
    mask = load_instance_mask_exr(exr_path)
    ids = [int(i) for i in np.unique(mask) if int(i) != 0]
    print(f"[exr_visualize] {exr_path} 实例 id: {ids if ids else '(无前景)'}")

    if rgb_path is not None:
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            raise FileNotFoundError(f"无法读取 RGB: {rgb_path}")
        if bgr.shape[:2] != mask.shape[:2]:
            bgr = cv2.resize(bgr, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        vis = overlay_mask_on_bgr(bgr, mask, alpha=alpha)
    else:
        vis = colorize_instance_mask(mask)

    if legend and ids:
        vis = draw_instance_legend(vis, mask)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), vis)
        print(f"[exr_visualize] 已保存: {out.resolve()}")

    if show:
        cv2.imshow("mask_exr", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return vis


def _cli() -> int:
    parser = argparse.ArgumentParser(description="可视化 mask.exr 中的实例 id")
    parser.add_argument("exr", type=Path, help="mask.exr 路径")
    parser.add_argument("--rgb", type=Path, default=None, help="可选：叠加的 RGB/BGR 底图")
    parser.add_argument("-o", "--output", type=Path, default=None, help="保存可视化 PNG")
    parser.add_argument("--show", action="store_true", help="弹窗显示")
    parser.add_argument("--alpha", type=float, default=0.55, help="叠加透明度")
    parser.add_argument("--no-legend", action="store_true", help="不绘制 id 图例")
    args = parser.parse_args()

    if not args.exr.is_file():
        print(f"找不到 EXR: {args.exr}", file=sys.stderr)
        return 1
    if args.rgb is not None and not args.rgb.is_file():
        print(f"找不到 RGB: {args.rgb}", file=sys.stderr)
        return 1
    if not args.show and args.output is None:
        print("请指定 -o/--output 或 --show", file=sys.stderr)
        return 1

    visualize_mask_exr(
        args.exr,
        rgb_path=args.rgb,
        output_path=args.output,
        show=args.show,
        alpha=args.alpha,
        legend=not args.no_legend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
