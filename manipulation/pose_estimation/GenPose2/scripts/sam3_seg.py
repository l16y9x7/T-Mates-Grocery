"""
SAM3 text-prompt segmentation backend (REST HTTP).

当前通过 SAM3 在线服务调用，推荐用 ``image_base64`` 传图（避免跨机/容器路径不可见）::

    curl -X POST http://127.0.0.1:18003/infer \
      -H "Content-Type: application/json" \
      -d '{
        "image_base64": "<base64>",
        "prompt": "Plastic Reel",
        "threshold": 0.41,
        "mask_threshold": 0.50
      }'

响应中的 ``detections`` 会落盘为 ``{output_dir}/sam6d_results/detection_ism.json``，
本模块继续生成实例 mask 与可视化。

默认配置见 ``config/conf.json``（sam3 / graspnet 外部依赖），模块内常量会在导入时从该文件加载：

    DEFAULT_SAM3_API_URL
    DEFAULT_SAM3_PROMPT
    DEFAULT_SAM3_THRESHOLD
    DEFAULT_SAM3_MASK_THRESHOLD
    DEFAULT_SAM3_TIMEOUT_S
"""

from __future__ import annotations

import base64
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image

_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from config import get_sam3_conf  # noqa: E402

_COCOMASK = None

# SAM3 实例 mask 后处理：各实例边界向内收缩的像素数
DEFAULT_INSTANCE_EDGE_ERODE_PX = 2


def erode_bool_mask(mask: np.ndarray, pixels: int = DEFAULT_INSTANCE_EDGE_ERODE_PX) -> np.ndarray:
    """二值/bool mask 边界腐蚀 ``pixels`` 像素（3x3 核，iterations=pixels）。"""
    if pixels <= 0:
        return mask
    region = (np.asarray(mask) > 0).astype(np.uint8)
    eroded = cv2.erode(region, np.ones((3, 3), np.uint8), iterations=int(pixels))
    if mask.dtype == bool:
        return eroded.astype(bool)
    return eroded


def erode_instance_id_map(
    id_map: np.ndarray,
    pixels: int = DEFAULT_INSTANCE_EDGE_ERODE_PX,
) -> np.ndarray:
    """对实例 id 图中每个实例独立腐蚀边界，背景保持 0。"""
    if pixels <= 0:
        return id_map
    src = np.asarray(id_map)
    out = np.zeros_like(src)
    kernel = np.ones((3, 3), np.uint8)
    for inst_id in sorted(int(v) for v in np.unique(src) if int(v) > 0):
        region = (src == inst_id).astype(np.uint8)
        eroded = cv2.erode(region, kernel, iterations=int(pixels))
        out[eroded > 0] = inst_id
    return out


def _save_instance_id_mask(composite: np.ndarray, mask_path: Path) -> Path:
    """保存实例 id 图（像素值 1..N 表示各实例）。"""
    mask_path = Path(mask_path).expanduser().resolve()
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_path.suffix.lower() == ".exr":
        mask_path = mask_path.with_suffix(".png")
    cv2.imwrite(str(mask_path), composite)
    return mask_path


def _cocomask():
    """延迟加载 pycocotools（genpose2 环境需 ``pip install pycocotools``）。"""
    global _COCOMASK
    if _COCOMASK is None:
        try:
            from pycocotools import mask as cocomask_mod
        except ImportError as exc:
            raise ImportError(
                "SAM3 分割依赖 pycocotools，请在当前 Python 环境安装：pip install pycocotools"
            ) from exc
        _COCOMASK = cocomask_mod
    return _COCOMASK


def _load_sam3_defaults() -> Dict[str, Any]:
    cfg = get_sam3_conf()
    return {
        "api_url": str(cfg.get("api_url") or "http://127.0.0.1:18003/infer"),
        "default_prompt": str(
            cfg.get("default_prompt") or "Plastic Reel Connected With Tape"
        ),
        "threshold": float(cfg.get("threshold", 0.41)),
        "mask_threshold": float(cfg.get("mask_threshold", 0.50)),
        "timeout_s": float(cfg.get("timeout_s", 300.0)),
    }


_SAM3_DEFAULTS = _load_sam3_defaults()
DEFAULT_SAM3_API_URL = _SAM3_DEFAULTS["api_url"]
DEFAULT_SAM3_PROMPT = _SAM3_DEFAULTS["default_prompt"]
DEFAULT_SAM3_THRESHOLD = _SAM3_DEFAULTS["threshold"]
DEFAULT_SAM3_MASK_THRESHOLD = _SAM3_DEFAULTS["mask_threshold"]
DEFAULT_SAM3_TIMEOUT_S = _SAM3_DEFAULTS["timeout_s"]


@dataclass
class Sam3SegmentationResult:
    """SAM3 分割输出：detection_ism.json + GenPose2 用 mask.exr。"""

    detection_ism_path: Path
    mask_exr: Path
    score: float
    num_instances: int = 1
    instance_scores: Optional[List[float]] = None
    instance_dets: Optional[List[Dict[str, Any]]] = None
    vis_ism_path: Optional[Path] = None


# 与 SAM3 可视化颜色约定一致（OpenCV BGR）
_VIS_COLORS_BGR: Tuple[Tuple[int, int, int], ...] = (
    (0, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 128),
    (64, 64, 255),
    (255, 64, 64),
)


def _mask_to_rle(binary_mask: np.ndarray) -> Dict[str, object]:
    mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = _cocomask().encode(mask)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


def _bbox_xywh_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("SAM3 mask is empty")
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1, y1, int(x2 - x1 + 1), int(y2 - y1 + 1)]


def _load_mask_image(path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    gray = np.array(Image.open(path).convert("L"))
    if gray.shape[0] != height or gray.shape[1] != width:
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
    return gray > 127


def _decode_rle_dict(rle: Dict[str, Any]) -> np.ndarray:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    size = rle["size"]
    h, w = int(size[0]), int(size[1])
    decoded = _cocomask().decode({"counts": counts, "size": [h, w]})
    return decoded.astype(bool)


def _pick_from_pred_json(
    data: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, float, List[int]]:
    scores = data.get("pred_scores") or data.get("scores")
    masks = data.get("pred_masks") or data.get("masks")
    boxes = data.get("pred_boxes") or data.get("bbox") or data.get("bboxes")

    if masks is None and "segmentation" in data:
        seg = data["segmentation"]
        if isinstance(seg, dict) and "counts" in seg:
            mask = _decode_rle_dict(seg)
            score = float(data.get("score", data.get("confidence", 1.0)))
            if mask.shape[0] != image_size[1] or mask.shape[1] != image_size[0]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    image_size,
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            return mask, score, _bbox_xywh_from_mask(mask)

    if isinstance(masks, list) and masks:
        idx = 0
        if scores is not None and len(scores):
            idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
            score = float(scores[idx])
        else:
            score = float(data.get("confidence", 1.0))
        item = masks[idx]
        if isinstance(item, dict) and "counts" in item:
            mask = _decode_rle_dict(item)
        elif isinstance(item, (list, np.ndarray)):
            mask = np.asarray(item, dtype=bool)
        else:
            raise RuntimeError(f"unsupported mask entry type in SAM3 json: {type(item)}")
        if mask.shape[0] != image_size[1] or mask.shape[1] != image_size[0]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                image_size,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if boxes is not None and len(boxes) > idx:
            box = boxes[idx]
            if isinstance(box, (list, tuple)) and len(box) == 4:
                if all(0.0 <= float(v) <= 1.0 for v in box):
                    w, h = image_size
                    x, y, bw, bh = [float(v) for v in box]
                    bbox = [int(x * w), int(y * h), int(bw * w), int(bh * h)]
                else:
                    bbox = [int(round(v)) for v in box]
            else:
                bbox = _bbox_xywh_from_mask(mask)
        else:
            bbox = _bbox_xywh_from_mask(mask)
        return mask, score, bbox

    raise RuntimeError("SAM3 json has no usable mask fields")


def _parse_sam3_output(work_dir: Path, rgb_path: Path) -> Tuple[np.ndarray, float, List[int]]:
    with Image.open(rgb_path) as im:
        image_size = im.size  # (W, H)

    for jf in sorted(work_dir.rglob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            try:
                return _pick_from_pred_json(data, image_size)
            except RuntimeError:
                continue
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if "segmentation" in data[0] or "bbox" in data[0]:
                det = max(data, key=lambda d: float(d.get("score", 0.0)))
                seg = det["segmentation"]
                mask = _decode_rle_dict(seg) if isinstance(seg, dict) else np.asarray(seg, dtype=bool)
                if mask.shape[0] != image_size[1] or mask.shape[1] != image_size[0]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        image_size,
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                return mask, float(det.get("score", 1.0)), list(det.get("bbox", _bbox_xywh_from_mask(mask)))

    mask_candidates = [
        p
        for p in work_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and "mask" in p.name.lower()
    ]
    if not mask_candidates:
        mask_candidates = list(work_dir.rglob("*.png"))
    if mask_candidates:
        path = sorted(mask_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        mask = _load_mask_image(path, image_size)
        return mask, 1.0, _bbox_xywh_from_mask(mask)

    raise RuntimeError(f"no mask or json found under SAM3 output dir: {work_dir}")


def _sam3_api_url() -> str:
    return DEFAULT_SAM3_API_URL


def path_for_sam3_service(path: Path | str) -> str:
    """把本机路径改写成 SAM3 服务可访问的形式（仅兼容旧的 image_path 调用）。

    ``/home/ubuntu/stephen`` 是指向 ``/data/ubuntu/stephen`` 的符号链接；
    ``Path.resolve()`` 会落到 ``/data/ubuntu/...``，而 SAM3 服务侧通常只挂载/认
    ``/home/ubuntu/...``，因此发给 HTTP API 的路径需要改回 home 前缀。
    """
    p = str(Path(path).expanduser().absolute())
    data_prefix = "/data/ubuntu"
    home_prefix = "/home/ubuntu"
    try:
        resolved = str(Path(p).resolve())
    except OSError:
        resolved = p
    if resolved.startswith(data_prefix + "/") or resolved == data_prefix:
        return home_prefix + resolved[len(data_prefix) :]
    if p.startswith(data_prefix + "/") or p == data_prefix:
        return home_prefix + p[len(data_prefix) :]
    if p.startswith(home_prefix + "/") or p.startswith(data_prefix + "/"):
        return p.replace(data_prefix, home_prefix, 1) if p.startswith(data_prefix) else p
    if resolved.startswith(home_prefix + "/"):
        return resolved
    return resolved


def _image_file_to_base64(rgb_path: Path) -> str:
    """读取本地图片并编码为 base64（供 SAM3 ``image_base64`` 字段）。"""
    data = Path(rgb_path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def _sam3_health_url() -> str:
    parsed = urlparse(_sam3_api_url())
    path = parsed.path or "/infer"
    if path.endswith("/infer"):
        path = path[: -len("/infer")] + "/health"
    else:
        path = "/health"
    return parsed._replace(path=path, params="", query="", fragment="").geturl()


def _sam3_timeout_s() -> float:
    return DEFAULT_SAM3_TIMEOUT_S


def _validate_sam3_service_config(api_url: str) -> None:
    parsed = urlparse(api_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"SAM3 API URL 无效: {api_url!r}\n"
            f"请检查 DEFAULT_SAM3_API_URL（当前默认 {DEFAULT_SAM3_API_URL}）"
        )


def check_sam3_service_health(*, timeout_s: float = 2.0) -> Dict[str, Any]:
    api_url = _sam3_api_url()
    health_url = _sam3_health_url()
    _validate_sam3_service_config(api_url)
    _validate_sam3_service_config(health_url)
    try:
        resp = requests.get(health_url, timeout=timeout_s)
    except requests.RequestException as exc:
        return {
            "ok": False,
            "api_url": api_url,
            "health_url": health_url,
            "error": str(exc),
        }

    payload: Dict[str, Any] = {
        "ok": resp.ok,
        "api_url": api_url,
        "health_url": health_url,
        "status_code": resp.status_code,
    }
    try:
        payload["response"] = resp.json()
    except ValueError:
        text = resp.text.strip()
        payload["response_text"] = text[:500] if text else ""
    return payload


def _decode_detection_mask(det: Dict[str, Any], image_size: Tuple[int, int]) -> np.ndarray:
    """从 detection_ism 单条记录解码 bool mask，image_size 为 (W, H)。"""
    width, height = image_size
    if "segmentation" in det:
        seg = det["segmentation"]
        if isinstance(seg, dict) and "counts" in seg:
            mask = _decode_rle_dict(seg)
        else:
            mask = np.asarray(seg, dtype=bool)
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return mask
    exr = det.get("segmentation_mask_exr")
    if exr and Path(exr).is_file():
        gray = np.array(Image.open(exr).convert("L"))
        if gray.shape[0] != height or gray.shape[1] != width:
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
        return gray > 127
    raise RuntimeError("detection entry has no segmentation or segmentation_mask_exr")


def write_genpose_mask_exr_from_ism(
    ism_json_path: Path,
    rgb_path: Path,
    mask_exr_path: Path,
    *,
    max_instances: int = 0,
) -> tuple[float, int, List[float], List[Dict[str, Any]]]:
    """
    读取 ``detection_ism.json``（COCO RLE），写出 GenPose2 ``mask.exr``。

    ``max_instances <= 0`` 表示保留 ISM 中全部实例（最多 254 个 id）；``max_instances=N`` 则按
    score 取 Top-N。mask 像素 id 与 score 排序一致：id=1 为最高分实例。

    :return: (best_score, num_instances, scores_for_mask_id_1..N, dets_for_each_mask_id)
    """
    ism_json_path = Path(ism_json_path).expanduser().resolve()
    rgb_path = Path(rgb_path).expanduser().resolve()
    mask_exr_path = Path(mask_exr_path).expanduser().resolve()

    dets = json.loads(ism_json_path.read_text(encoding="utf-8"))
    if not isinstance(dets, list) or not dets:
        raise RuntimeError(f"empty or invalid detection_ism.json: {ism_json_path}")

    with Image.open(rgb_path) as im:
        image_size = im.size  # (W, H)
    width, height = image_size

    ordered = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    if max_instances <= 0:
        ordered = ordered[:254]
    else:
        ordered = ordered[: max(1, min(max_instances, 254))]

    composite = np.zeros((height, width), dtype=np.uint8)
    best_score = float(ordered[0].get("score", 1.0))
    instance_scores: List[float] = []
    instance_dets: List[Dict[str, Any]] = []
    next_id = 1
    for det in ordered:
        mask = _decode_detection_mask(det, image_size)
        fill = mask & (composite == 0)
        if not np.any(fill):
            continue
        composite[fill] = np.uint8(next_id)
        instance_scores.append(float(det.get("score", 0.0)))
        instance_dets.append(det)
        next_id += 1

    if not np.any(composite > 0):
        raise RuntimeError(f"no foreground pixels in mask from {ism_json_path}")

    # 后处理：每个实例去掉边缘 2 像素；腐蚀后为空的实例丢弃并重排 id
    eroded = erode_instance_id_map(composite, DEFAULT_INSTANCE_EDGE_ERODE_PX)
    alive = {int(v) for v in np.unique(eroded) if int(v) > 0}
    if not alive:
        raise RuntimeError(
            f"instance edge erode ({DEFAULT_INSTANCE_EDGE_ERODE_PX}px) removed all foreground "
            f"from {ism_json_path}"
        )
    new_composite = np.zeros_like(eroded)
    new_scores: List[float] = []
    new_dets: List[Dict[str, Any]] = []
    new_id = 1
    for old_id, score, det in zip(
        range(1, len(instance_scores) + 1), instance_scores, instance_dets
    ):
        if old_id not in alive:
            continue
        new_composite[eroded == old_id] = np.uint8(new_id)
        new_scores.append(score)
        new_dets.append(det)
        new_id += 1
    composite = new_composite
    instance_scores = new_scores
    instance_dets = new_dets

    num_instances = len(instance_scores)
    saved_mask_path = _save_instance_id_mask(composite, mask_exr_path)
    print(
        f"[sam3_seg] instance mask -> {saved_mask_path} "
        f"(instances={num_instances}/{len(dets)} ism dets, "
        f"edge_erode_px={DEFAULT_INSTANCE_EDGE_ERODE_PX}, best_score={best_score:.4f})"
    )
    return best_score, num_instances, instance_scores, instance_dets


def visualize_sam3_ism(
    rgb_path: Path,
    instance_dets: List[Dict[str, Any]],
    output_path: Path,
    *,
    prompt: Optional[str] = None,
    mask_alpha: float = 0.5,
    instance_ids: Optional[List[int]] = None,
) -> Path:
    """
    在 RGB 上绘制 SAM3 多实例 mask、bbox 与 score，写入 ``vis_ism.png`` 等路径。

    ``instance_ids`` 与 ``mask.exr`` 中 id 一致时（默认 1..N），图例显示 ``id=#``。
    """
    rgb_path = Path(rgb_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read rgb for SAM3 vis: {rgb_path}")
    height, width = bgr.shape[:2]
    image_size = (width, height)

    if not instance_dets:
        cv2.imwrite(str(output_path), bgr)
        print(f"[sam3_seg_backend] vis (empty dets) -> {output_path}")
        return output_path

    label = (prompt or DEFAULT_SAM3_PROMPT).split()[0] or "obj"
    overlay = bgr.astype(np.float32)

    for idx, det in enumerate(instance_dets):
        mask = _decode_detection_mask(det, image_size)
        color = _VIS_COLORS_BGR[idx % len(_VIS_COLORS_BGR)]
        color_arr = np.array(color, dtype=np.float32)
        overlay[mask] = mask_alpha * color_arr + (1.0 - mask_alpha) * overlay[mask]

        bbox = det.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, bw, bh = [int(round(v)) for v in bbox]
        else:
            x, y, bw, bh = _bbox_xywh_from_mask(mask)
        x2, y2 = x + bw, y + bh

        inst_id = (
            int(instance_ids[idx])
            if instance_ids is not None and idx < len(instance_ids)
            else idx + 1
        )
        score = float(det.get("score", 0.0))
        cv2.rectangle(overlay, (x, y), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"{label} id={inst_id} {score:.3f}",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), overlay.astype(np.uint8))
    print(f"[sam3_seg_backend] vis_ism -> {output_path} ({len(instance_dets)} instances)")
    return output_path


def visualize_sam3_mask_exr(
    rgb_path: Path,
    mask_exr_path: Path,
    output_path: Path,
    *,
    alpha: float = 0.55,
) -> Path:
    """基于实例 id mask 图着色叠加。"""
    rgb_path = Path(rgb_path).expanduser().resolve()
    mask_path = Path(mask_exr_path).expanduser().resolve()
    if mask_path.suffix.lower() == ".exr" and not mask_path.is_file():
        mask_path = mask_path.with_suffix(".png")
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read rgb for SAM3 vis: {rgb_path}")
    id_map = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if id_map is None:
        raise FileNotFoundError(f"cannot read instance mask: {mask_path}")
    if id_map.ndim == 3:
        id_map = id_map[:, :, 0]
    overlay = bgr.astype(np.float32)
    for inst_id in sorted(int(v) for v in np.unique(id_map) if int(v) > 0):
        color = np.array(_VIS_COLORS_BGR[(inst_id - 1) % len(_VIS_COLORS_BGR)], dtype=np.float32)
        region = id_map == inst_id
        overlay[region] = alpha * color + (1.0 - alpha) * overlay[region]
        ys, xs = np.where(region)
        if len(xs):
            cv2.putText(
                overlay,
                f"id={inst_id}",
                (int(xs.min()), max(0, int(ys.min()) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color.tolist(),
                2,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(out), overlay.astype(np.uint8))
    print(f"[sam3_seg] vis_sam3_seg -> {out}")
    return out


def get_instance_bool_masks(
    instance_dets: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    *,
    edge_erode_px: int = DEFAULT_INSTANCE_EDGE_ERODE_PX,
) -> List[np.ndarray]:
    """从 SAM3 detection 列表解码各实例 bool mask，``image_size`` 为 (W, H)。

    默认对各实例做边界腐蚀，与 ``write_genpose_mask_exr_from_ism`` 后处理一致。
    """
    masks = [_decode_detection_mask(det, image_size) for det in instance_dets]
    if edge_erode_px > 0:
        masks = [erode_bool_mask(m, edge_erode_px) for m in masks]
    return masks


def run_sam3_segmentation(
    rgb_path: Path,
    output_dir: Path,
    *,
    prompt: Optional[str] = None,
    threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    mask_exr_out: Optional[Path] = None,
    max_instances: int = 0,
) -> Sam3SegmentationResult:
    rgb_path = rgb_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    api_url = _sam3_api_url()
    _validate_sam3_service_config(api_url)

    prompt_text = (
        prompt
        if prompt is not None
        else DEFAULT_SAM3_PROMPT
    )
    thresh = (
        float(threshold)
        if threshold is not None
        else float(DEFAULT_SAM3_THRESHOLD)
    )
    mask_thresh = (
        float(mask_threshold)
        if mask_threshold is not None
        else float(DEFAULT_SAM3_MASK_THRESHOLD)
    )

    sam6d_results = output_dir / "sam6d_results"
    sam6d_results.mkdir(parents=True, exist_ok=True)
    json_path = sam6d_results / "detection_ism.json"

    image_b64 = _image_file_to_base64(rgb_path)
    payload = {
        "image_base64": image_b64,
        "prompt": prompt_text,
        "threshold": float(thresh),
        "mask_threshold": float(mask_thresh),
        "save_vis": False,
        "return_vis_base64": False,
    }
    print(f"[sam3_seg_backend] POST {api_url}")
    print(
        f"[sam3_seg_backend] payload keys={list(payload.keys())} "
        f"prompt={prompt_text!r} image_base64_len={len(image_b64)}"
    )

    t0 = time.perf_counter()
    try:
        resp = requests.post(api_url, json=payload, timeout=_sam3_timeout_s())
    except requests.RequestException as exc:
        raise RuntimeError(f"SAM3 HTTP request failed: {exc}") from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[sam3_seg_backend] elapsed_ms={elapsed_ms:.3f} status_code={resp.status_code}")

    try:
        body = resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"SAM3 invalid JSON response: status={resp.status_code}\n{resp.text[:2000]}"
        ) from exc

    if not resp.ok or (isinstance(body, dict) and body.get("ok") is False):
        err = body.get("error") if isinstance(body, dict) else resp.text[:2000]
        raise RuntimeError(f"SAM3 HTTP infer failed: status={resp.status_code}\n{err}")

    detections = None
    if isinstance(body, dict):
        detections = body.get("detections")
    if not isinstance(detections, list):
        # 兼容旧服务：可能仍写盘到 output_dir
        print(f"[sam3_seg_backend] response has no detections list, fallback parse under {output_dir}")
        if resp.text:
            print(f"[sam3_seg_backend] response head: {resp.text[:1000]}")
        if not json_path.is_file():
            mask, score, bbox_xywh = _parse_sam3_output(output_dir, rgb_path)
            detections = [
                {
                    "scene_id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "bbox": bbox_xywh,
                    "score": float(score),
                    "time": 0.0,
                    "segmentation": _mask_to_rle(mask),
                }
            ]
        else:
            detections = json.loads(json_path.read_text(encoding="utf-8"))

    if not isinstance(detections, list):
        raise RuntimeError(
            f"SAM3 detections must be a list, got {type(detections).__name__}"
        )
    json_path.write_text(json.dumps(detections, ensure_ascii=False), encoding="utf-8")
    print(f"[sam3_seg_backend] wrote {json_path} n_detections={len(detections)}")

    if mask_exr_out is None:
        mask_exr_out = sam6d_results / "mask.exr"
    else:
        mask_exr_out = Path(mask_exr_out)

    best_score, num_instances, instance_scores, instance_dets = write_genpose_mask_exr_from_ism(
        json_path,
        rgb_path,
        mask_exr_out,
        max_instances=max_instances,
    )

    vis_ism_path = sam6d_results / "vis_ism.png"
    vis_sam3_seg_path = sam6d_results / "vis_sam3_seg.png"
    instance_ids = list(range(1, num_instances + 1))
    if instance_dets:
        visualize_sam3_ism(
            rgb_path,
            instance_dets,
            vis_ism_path,
            prompt=prompt_text,
            instance_ids=instance_ids,
        )
        visualize_sam3_mask_exr(rgb_path, mask_exr_out, vis_sam3_seg_path)
    elif vis_ism_path.is_file():
        print(f"[sam3_seg_backend] keep existing vis_ism: {vis_ism_path}")
    else:
        print(f"[sam3_seg_backend] no instance_dets for vis, skip")

    return Sam3SegmentationResult(
        detection_ism_path=json_path,
        mask_exr=mask_exr_out,
        score=best_score,
        num_instances=num_instances,
        instance_scores=instance_scores,
        instance_dets=instance_dets,
        vis_ism_path=vis_ism_path if vis_ism_path.is_file() else None,
    )
