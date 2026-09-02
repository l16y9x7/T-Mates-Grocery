"""
单张 RGB-D 推理：先 YOLO 分割生成 mask.exr，再调用 GenPose2（与 runners/infer.py 相同管线）。

注意：项目内部分模块在 import 时会调用 configs.get_config() 并解析 sys.argv，
因此本脚本在作为主程序运行时会在导入前暂存并清空 argv，解析完本脚本参数后再恢复。

示例（仓库根目录、已激活 genpose2）::

    python runners/smt_infer.py \\
        --rgb learning/inputs/1_.png \\
        --depth learning/inputs/1_depth.png \\
        --meta learning/inputs/1_meta.json \\
        --yolo-weights segment/yolo_seg.pt \\
        --score-ckpt results/ckpts/ScoreNet/scorenet.pth \\
        --energy-ckpt results/ckpts/EnergyNet/energynet.pth \\
        --scale-ckpt results/ckpts/ScaleNet/scalenet.pth \\
        --save-vis learning/outputs/1_pose_vis.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# 仓库根加入 path（与 runners/infer.py 一致）
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

# 避免 networks/pts_encoder/pointnet2.py 等模块在 import 时执行 get_config() 吃掉 CLI
_SAVED_SYS_ARGV_FOR_IMPORT: Optional[List[str]] = None
if __name__ == "__main__":
    _SAVED_SYS_ARGV_FOR_IMPORT = sys.argv[:]
    sys.argv = [sys.argv[0]]

from cutoop.data_loader import Dataset  # noqa: E402

from runners.infer import create_genpose2, visualize_pose  # noqa: E402
from datasets.datasets_infer import InferDataset  # noqa: E402
from segment.yolo_seg_backend import run_yolo_segmentation  # noqa: E402

if _SAVED_SYS_ARGV_FOR_IMPORT is not None:
    sys.argv = _SAVED_SYS_ARGV_FOR_IMPORT[:]


def _load_color_any(rgb_path: Path) -> np.ndarray:
    """RGB 顺序 H×W×3；若文件名为 *color.png 则走 Dataset.load_color。"""
    rgb_path = Path(rgb_path)
    if rgb_path.name.endswith("color.png"):
        return Dataset.load_color(str(rgb_path))
    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(f"无法读取 RGB: {rgb_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_depth_any(depth_path: Path, depth_scale: float | None) -> np.ndarray:
    """支持 .exr（米）或 .png（默认 uint16 按毫米 ×0.001→米）。"""
    depth_path = Path(depth_path)
    suf = depth_path.suffix.lower()
    if suf == ".exr":
        return Dataset.load_depth(str(depth_path))

    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"无法读取深度: {depth_path}")
    if d.ndim == 3:
        d = d[:, :, 0]
    d = np.asarray(d)
    if d.dtype == np.uint16:
        scale = 0.001 if depth_scale is None else float(depth_scale)
        return d.astype(np.float32) * scale
    return d.astype(np.float32)


def _load_meta_dict(meta_path: Path) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "camera" not in meta or "intrinsics" not in meta["camera"]:
        raise ValueError("meta.json 需包含 camera.intrinsics（见 learning/infer.md）")
    return meta


def _ensure_hw_match(color: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> None:
    h0, w0 = color.shape[:2]
    if depth.shape[:2] != (h0, w0) or mask.shape[:2] != (h0, w0):
        raise ValueError(
            f"color/depth/mask 空间尺寸需一致: color={color.shape[:2]}, depth={depth.shape[:2]}, mask={mask.shape[:2]}"
        )


def _print_timing_report(timings: Dict[str, float], total_s: float) -> None:
    labels = {
        "yolo_mask": "1. YOLO 分割 (mask.exr)",
        "genpose_load": "2. GenPose 模型加载",
        "genpose_infer": "3. GenPose 模型推理",
        "visualize": "4. 可视化",
    }
    print("[smt_infer] ---------- 耗时 (wall time) ----------")
    for key in ("yolo_mask", "genpose_load", "genpose_infer", "visualize"):
        if key not in timings:
            continue
        sec = timings[key]
        pct = (sec / total_s * 100.0) if total_s > 0 else 0.0
        print(f"[smt_infer]   {labels[key]:28s} {sec:8.3f} s  ({pct:5.1f}%)")
    print(f"[smt_infer]   {'合计':28s} {total_s:8.3f} s")
    print("[smt_infer] ------------------------------------")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="单张图：YOLO 分割 + GenPose2 推理")
    p.add_argument("--rgb", type=Path, required=True, help="RGB 图路径（任意 png/jpg，或 *color.png）")
    p.add_argument("--depth", type=Path, required=True, help="深度 .exr 或 .png")
    p.add_argument("--meta", type=Path, required=True, help="meta.json（含 camera.intrinsics）")
    p.add_argument("--yolo-weights", type=Path, required=True, help="YOLO 分割权重 .pt")
    p.add_argument(
        "--mask-exr",
        type=Path,
        default=None,
        help="YOLO 输出的 mask.exr；默认写到 --work-dir/mask/{rgb_stem}_mask.exr",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("output/smt_infer"),
        help="中间结果目录（YOLO 临时输出等）",
    )
    p.add_argument("--skip-yolo", action="store_true", help="若 mask 已存在且路径由 --mask-exr 指定，可跳过 YOLO")
    p.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="深度为 uint16 PNG 时的乘子；默认 0.001（毫米→米）。.exr 忽略此项。",
    )
    p.add_argument(
        "--score-ckpt",
        type=Path,
        default=Path("results/ckpts/ScoreNet/scorenet.pth"),
        help="ScoreNet 权重（避免与全局 get_config 的 --scale* 缩写冲突，不用 --score）",
    )
    p.add_argument("--energy-ckpt", type=Path, default=Path("results/ckpts/EnergyNet/energynet.pth"))
    p.add_argument("--scale-ckpt", type=Path, default=Path("results/ckpts/ScaleNet/scalenet.pth"))
    p.add_argument("--save-vis", type=Path, default=None, help="保存可视化 BGR 图；不指定则不写盘")
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--yolo-class-id", type=int, default=0, help="设为 -1 表示不按类别过滤")
    p.add_argument("--yolo-max-instances", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t_pipeline = time.perf_counter()
    timings: Dict[str, float] = {}

    rgb_path = Path(args.rgb).resolve()
    depth_path = Path(args.depth).resolve()
    meta_path = Path(args.meta).resolve()
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = (_REPO_ROOT / work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = work_dir / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)

    if args.mask_exr is not None:
        mask_exr_path = Path(args.mask_exr).resolve()
    else:
        mask_exr_path = mask_dir / f"{rgb_path.stem}_mask.exr"

    t0 = time.perf_counter()
    if not args.skip_yolo:
        class_id = None if args.yolo_class_id < 0 else args.yolo_class_id
        run_yolo_segmentation(
            Path(args.yolo_weights).resolve(),
            rgb_path,
            work_dir / "yolo_tmp",
            conf=args.yolo_conf,
            imgsz=args.yolo_imgsz,
            class_id=class_id,
            max_instances=args.yolo_max_instances,
            mask_exr_out=mask_exr_path,
        )
    elif not mask_exr_path.is_file():
        print("已指定 --skip-yolo 但找不到 mask.exr:", mask_exr_path, file=sys.stderr)
        return 1
    timings["yolo_mask"] = time.perf_counter() - t0

    meta = _load_meta_dict(meta_path)
    color = _load_color_any(rgb_path)
    depth = _load_depth_any(depth_path, args.depth_scale)
    mask = Dataset.load_mask(str(mask_exr_path))
    _ensure_hw_match(color, depth, mask)

    _argv_user = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        t0 = time.perf_counter()
        genpose = create_genpose2(
            score_model_path=str(Path(args.score_ckpt).resolve()),
            energy_model_path=str(Path(args.energy_ckpt).resolve()),
            scale_model_path=str(Path(args.scale_ckpt).resolve()),
        )
        timings["genpose_load"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        data = InferDataset(
            {"depth": depth, "color": color, "mask": mask, "meta": meta},
            img_size=genpose.cfg.img_size,
            device=genpose.cfg.device,
            n_pts=genpose.cfg.num_points,
        )
        pose, length = genpose.inference(data, prev_pose=None, tracking=False, tracking_T0=0.15)
        timings["genpose_infer"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        vis_bgr = visualize_pose(data, pose, length, visualize_pts=False, visualize_image=False)
        if args.save_vis:
            out = Path(args.save_vis)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), vis_bgr)
        timings["visualize"] = time.perf_counter() - t0
    finally:
        sys.argv = _argv_user

    total_s = time.perf_counter() - t_pipeline
    _print_timing_report(timings, total_s)

    print("推理完成。物体数:", int(pose[0].shape[0]) if pose and len(pose) > 0 else 0)
    if args.save_vis:
        print("可视化已保存:", Path(args.save_vis).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
