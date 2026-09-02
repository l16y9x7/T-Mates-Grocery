import os

# OpenCV EXR I/O（与 cutoop Dataset.load_mask 一致）需在 import cv2 之前设置
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


_MODEL_CACHE: Dict[str, YOLO] = {}


@dataclass
class YoloSegmentationResult:
    """YOLO 分割输出：GenPose2 / cutoop 可读的单通道 mask.exr。"""

    mask_exr: Path
    score: float = 1.0


def _load_model(weights_path: Path) -> YOLO:
    key = str(weights_path.resolve())
    print(f"[yolo_seg_backend] load request: {weights_path} (resolved: {key})")
    if key not in _MODEL_CACHE:
        if not weights_path.is_file():
            print(f"[yolo_seg_backend] weights not found: {weights_path}")
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
        print(f"[yolo_seg_backend] loading YOLO model: {key}")
        # Force segmentation task to avoid TRT engine auto-guess as detect.
        _MODEL_CACHE[key] = YOLO(key, task="segment")
        print(f"[yolo_seg_backend] model loaded and cached: {key}")
    else:
        print(f"[yolo_seg_backend] model cache hit: {key}")
    return _MODEL_CACHE[key]


def _resize_mask(mask: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """将 YOLO mask 缩放到与 RGB 一致。image_size 为 (width, height)。"""
    width, height = image_size
    resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return resized > 0.5


def _read_rgb_size(rgb_path: Path) -> Tuple[int, int]:
    """返回 (width, height)，与 PIL Image.size 一致。"""
    with Image.open(rgb_path) as image:
        w, h = image.size
    return w, h


def instance_ids_to_genpose_exr_float(instance_u8: np.ndarray) -> np.ndarray:
    """
    cutoop Dataset.load_mask：cv2.imread 后对单通道执行 (img * 255).astype(uint8)。
    因此 EXR 中应保存 float32：像素值为 instance_id / 255（背景为 0），
    这样读回后 *255 得到 0,1,2,... 的 uint8 实例编号（与 Omni6D / GenPose2 InferDataset 一致）。
    """
    out = np.zeros(instance_u8.shape, dtype=np.float32)
    fg = instance_u8 > 0
    out[fg] = instance_u8[fg].astype(np.float32) / 255.0
    return out


def save_genpose2_mask_exr(instance_ids_hw: np.ndarray, exr_path: Path) -> None:
    """
    将 uint8 / int 实例图（0=背景，1..N=实例 id）写成 GenPose2 / cutoop 可读的 mask.exr。
    """
    if instance_ids_hw.ndim != 2:
        raise ValueError("instance_ids_hw 必须是二维 (H, W)")
    if int(instance_ids_hw.max()) > 254:
        raise ValueError("实例编号不能超过 254（uint8 与 /255 编码约定）")
    exr_path = Path(exr_path)
    exr_path.parent.mkdir(parents=True, exist_ok=True)
    to_write = instance_ids_to_genpose_exr_float(instance_ids_hw.astype(np.uint8))
    if not cv2.imwrite(str(exr_path), to_write):
        raise RuntimeError(f"cv2.imwrite 失败: {exr_path}（确认 OPENCV_IO_ENABLE_OPENEXR 且 OpenCV 带 EXR）")


def build_instance_mask_from_yolo(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    image_size: Tuple[int, int],
    *,
    class_id: Optional[int] = 0,
    max_instances: int = 1,
) -> np.ndarray:
    """
    根据 YOLO 输出构造 (H, W) uint8 实例图：0 背景，1..N 为实例（高分优先占像素）。
    image_size: (width, height)
    """
    width, height = image_size
    h, w = height, width
    candidate_indexes = list(range(len(scores)))
    if class_id is not None:
        candidate_indexes = [i for i in candidate_indexes if int(classes[i]) == class_id]
    if not candidate_indexes:
        raise RuntimeError(f"YOLO 在 class_id={class_id} 下无可用实例")

    order = sorted(candidate_indexes, key=lambda i: float(scores[i]), reverse=True)
    order = order[: max(1, max_instances)]

    composite = np.zeros((h, w), dtype=np.uint8)
    for rank, det_idx in enumerate(order):
        m = _resize_mask(masks[det_idx], image_size)
        fill = m & (composite == 0)
        composite[fill] = np.uint8(rank + 1)
    return composite


def run_yolo_segmentation(
    weights_path: Path,
    rgb_path: Path,
    output_dir: Path,
    conf: float = 0.25,
    imgsz: int = 640,
    class_id: Optional[int] = 0,
    max_instances: int = 1,
    mask_exr_out: Optional[Path] = None,
) -> YoloSegmentationResult:
    """
    YOLO 分割：仅写 GenPose2 / cutoop 所需的 mask.exr（独立管线，不产出其它检测 JSON）。

    :param mask_exr_out: mask.exr 输出路径；默认 ``output_dir/{rgb词干}_mask.exr``
    :param max_instances: >1 时按置信度从高到低填充实例 id，重叠处保留高分实例。
    """
    t0 = time.perf_counter()
    t_load0 = time.perf_counter()
    model = _load_model(weights_path)
    t_load1 = time.perf_counter()
    print(f"[yolo_seg_backend] _load_model elapsed_ms={(t_load1 - t_load0) * 1000:.3f}")

    rgb_path = Path(rgb_path)
    image_size = _read_rgb_size(rgb_path)

    t_pred0 = time.perf_counter()
    results = model.predict(str(rgb_path), conf=conf, imgsz=imgsz, verbose=False, task="segment")
    t_pred1 = time.perf_counter()
    print(f"[yolo_seg_backend] model.predict elapsed_ms={(t_pred1 - t_pred0) * 1000:.3f}")
    print(f"[yolo_seg_backend] load+predict elapsed_ms={(t_pred1 - t0) * 1000:.3f}")
    if not results:
        raise RuntimeError("YOLO returned no results")

    result = results[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("YOLO returned no segmentation masks")

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    masks = result.masks.data.detach().cpu().numpy()

    candidate_indexes = list(range(len(scores)))
    if class_id is not None:
        candidate_indexes = [idx for idx in candidate_indexes if classes[idx] == class_id]
    if not candidate_indexes:
        raise RuntimeError(f"YOLO returned no masks for class_id={class_id}")
    best_idx = max(candidate_indexes, key=lambda idx: float(scores[idx]))
    best_score = float(scores[best_idx])

    instance_u8 = build_instance_mask_from_yolo(
        masks, boxes, scores, classes, image_size, class_id=class_id, max_instances=max_instances
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if mask_exr_out is None:
        mask_exr_path = output_dir / f"{rgb_path.stem}_mask.exr"
    else:
        mask_exr_path = Path(mask_exr_out)

    save_genpose2_mask_exr(instance_u8, mask_exr_path)
    print(f"[yolo_seg_backend] mask.exr -> {mask_exr_path} (instances max={int(instance_u8.max())})")

    return YoloSegmentationResult(mask_exr=mask_exr_path, score=best_score)


def preload_yolo_model(
    weights_path: Path,
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    class_id: Optional[int] = 0,
) -> None:
    """Preload YOLO model into cache and run one dummy warmup predict."""
    t0 = time.perf_counter()
    model = _load_model(weights_path)
    t1 = time.perf_counter()
    print(f"[yolo_seg_backend] preload _load_model elapsed_ms={(t1 - t0) * 1000:.3f}")

    # Dummy image (H, W, C) 与常见 RealSense 帧一致 480x640
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    t2 = time.perf_counter()
    results = model.predict(dummy, conf=conf, imgsz=imgsz, verbose=False, task="segment")
    t3 = time.perf_counter()
    print(f"[yolo_seg_backend] preload warmup predict elapsed_ms={(t3 - t2) * 1000:.3f}")

    if results:
        result = results[0]
        n = 0 if result.boxes is None else len(result.boxes)
        print(f"[yolo_seg_backend] preload warmup detections={n} class_filter={class_id}")
    print(f"[yolo_seg_backend] preload total elapsed_ms={(t3 - t0) * 1000:.3f}")


def _resolve_existing_file(path: Path, search_roots: Tuple[Path, ...]) -> Path:
    """依次尝试 path（绝对/相对 cwd）与各 search_roots 下的相对路径。"""
    path = Path(path)
    if path.is_file():
        return path.resolve()
    for root in search_roots:
        cand = (root / path).resolve()
        if cand.is_file():
            return cand
    return path.resolve()


def _cli_main() -> int:
    import argparse
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    seg_dir = Path(__file__).resolve().parent
    roots = (Path.cwd(), repo_root, seg_dir)

    parser = argparse.ArgumentParser(
        description="YOLO 分割：生成 GenPose2 用 mask.exr（cutoop Dataset.load_mask 可读）",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/yolo_seg.pt"),
        help="YOLO 分割权重 .pt（默认相对仓库/segment 或当前目录）",
    )
    parser.add_argument(
        "--rgb",
        type=Path,
        default=Path("data/rgb.png"),
        help="输入 RGB 图像（png/jpg 等）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/yolo_seg_test"),
        help="输出目录；默认在此目录写入 {rgb 文件名}_mask.exr",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="只保留该类；设为 -1 表示不过滤类别",
    )
    parser.add_argument("--max-instances", type=int, default=1)
    parser.add_argument(
        "--mask-exr",
        type=Path,
        default=None,
        help="可选：直接指定 mask.exr 输出路径（默认 output-dir/{rgb_stem}_mask.exr）",
    )
    parser.add_argument(
        "--verify-cutoop",
        action="store_true",
        help="跑完后用 cutoop.Dataset.load_mask 读回 mask 并打印 unique 像素值",
    )
    args = parser.parse_args()

    weights_path = _resolve_existing_file(args.weights, roots)
    rgb_path = _resolve_existing_file(args.rgb, roots)
    if not weights_path.is_file():
        print(f"[cli] 找不到权重: {args.weights}（已尝试 cwd / 仓库根 / segment 目录）", file=sys.stderr)
        return 1
    if not rgb_path.is_file():
        print(f"[cli] 找不到 RGB: {args.rgb}", file=sys.stderr)
        print("[cli] 示例: python segment/yolo_seg_backend.py --rgb path/to/frame_color.png", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    class_id = None if args.class_id < 0 else args.class_id
    try:
        result = run_yolo_segmentation(
            weights_path,
            rgb_path,
            output_dir,
            conf=args.conf,
            imgsz=args.imgsz,
            class_id=class_id,
            max_instances=args.max_instances,
            mask_exr_out=args.mask_exr,
        )
    except Exception as e:
        print(f"[cli] 推理失败: {e}", file=sys.stderr)
        return 1

    print("[cli] 完成")
    print(f"  mask_exr: {result.mask_exr}")

    if args.verify_cutoop:
        try:
            from cutoop.data_loader import Dataset

            loaded = Dataset.load_mask(str(result.mask_exr))
            uniq = np.unique(loaded)
            print(f"[cli] cutoop load_mask unique ids (uint8): {uniq.tolist()}")
        except Exception as e:
            print(f"[cli] verify-cutoop 跳过或失败: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())