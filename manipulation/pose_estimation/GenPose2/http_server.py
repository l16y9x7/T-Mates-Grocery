"""
GenPose2 HTTP 服务：/infer 接收 rgb、depth、camera 三个 multipart 文件（与 warmup_http_service 一致）。
可选上传 ``mask`` 时跳过分割，直接进行 GenPose2 推理（mask 支持 ``.exr`` / ``.png`` 等）。
SAM3 分割时可通过表单字段 ``sam3_prompt`` 传入文本提示（优先于环境变量）。

分割后端（``GENPOSE2_SEG_BACKEND`` 或 ``--seg-backend``）::

- ``yolo``（默认）：``segment/yolo_seg_backend.py``
- ``sam3``：``segment/sam3_seg_backend.py``（文本提示，子进程调用 SAM3）

启动示例（YOLO）::

    export GENPOSE2_SCORE_CKPT=results/ckpts/ScoreNet/scorenet.pth
    export GENPOSE2_ENERGY_CKPT=results/ckpts/EnergyNet/energynet.pth
    export GENPOSE2_SCALE_CKPT=results/ckpts/ScaleNet/scalenet.pth
    python http_server.py --host 0.0.0.0 --port 8084

启动示例（SAM3 文本分割）::

    export GENPOSE2_SEG_BACKEND=sam3
    export SAM6D_SAM3_PROMPT="white plate"
    export SAM6D_SAM3_ROOT=/path/to/sam3
    python http_server.py --host 0.0.0.0 --port 8084 --seg-backend sam3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, TypedDict

import cv2
import numpy as np

# 避免 import 链中 get_config() 解析 uvicorn 命令行
_ARGV_SAVED = sys.argv[:]
sys.argv = [_ARGV_SAVED[0] if _ARGV_SAVED else "http_server"]

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from cutoop.data_loader import Dataset  # noqa: E402
from datasets.datasets_infer import InferDataset  # noqa: E402
from runners.infer import create_genpose2, visualize_pose  # noqa: E402
from segment.yolo_seg_backend import preload_yolo_model, run_yolo_segmentation  # noqa: E402

sys.argv = _ARGV_SAVED

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.exception_handlers import (  # noqa: E402
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402


DEFAULT_OUTPUT_ROOT = ROOT_DIR / "service_outputs"
DEFAULT_YOLO_WEIGHTS = "segment/yolo_seg.pt"
DEFAULT_SEG_BACKEND = "yolo"
app = FastAPI(title="GenPose2 HTTP Service")
LOGGER = logging.getLogger(__name__)


@app.exception_handler(RequestValidationError)
async def _request_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    if request.url.path.startswith("/manipulation/"):
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "INVALID_REQUEST",
                "message": "request validation failed",
            },
        )
    return await request_validation_exception_handler(request, exc)


def _apply_seg_backend_from_argv(argv: List[str]) -> None:
    """
    从命令行解析 ``--seg-backend`` 并写入环境变量。

    ``uvicorn http_server:app --seg-backend sam3`` 不会走 ``main()``，须在 import 阶段解析。
    """
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--seg-backend" and i + 1 < len(argv):
            os.environ["GENPOSE2_SEG_BACKEND"] = argv[i + 1].strip().lower()
            i += 2
            continue
        if arg.startswith("--seg-backend="):
            os.environ["GENPOSE2_SEG_BACKEND"] = arg.split("=", 1)[1].strip().lower()
        i += 1


_apply_seg_backend_from_argv(_ARGV_SAVED)
infer_lock = asyncio.Lock()

_genpose_holder: Dict[str, Any] = {"model": None}


class InferTiming(TypedDict, total=False):
    """与 warmup_http_service 的 timing 字段命名一致。"""
    ism_s: Optional[float]
    yolo_s: Optional[float]
    templates_s: float
    pose_s: float
    pipeline_s: float
    upload_s: float
    total_s: float


class _PipelineResult(NamedTuple):
    payload: Dict[str, Any]
    primary_pose_4x4: np.ndarray
    primary_size_3d_m: np.ndarray


def _output_root() -> Path:
    return Path(os.environ.get("GENPOSE2_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))).expanduser().resolve()


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    else:
        path = path.resolve()
    return path


def _seg_backend() -> str:
    """分割后端：``yolo`` 或 ``sam3``。"""
    name = os.environ.get("GENPOSE2_SEG_BACKEND", DEFAULT_SEG_BACKEND).strip().lower()
    if name not in ("yolo", "sam3"):
        raise RuntimeError(
            f"unsupported GENPOSE2_SEG_BACKEND={name!r} (use 'yolo' or 'sam3')"
        )
    return name


def _yolo_weights_path() -> Path:
    """默认 ``segment/yolo_seg.pt``（相对仓库根目录）。"""
    value = os.environ.get("GENPOSE2_YOLO_WEIGHTS", DEFAULT_YOLO_WEIGHTS)
    path = _resolve_repo_path(value)
    if not path.is_file():
        raise RuntimeError(f"YOLO weights not found: {path} (GENPOSE2_YOLO_WEIGHTS={value!r})")
    return path


def _resolve_sam3_prompt(request_prompt: Optional[str]) -> str:
    """请求 ``sam3_prompt`` > 环境变量 > ``DEFAULT_SAM3_PROMPT``。"""
    if request_prompt is not None and request_prompt.strip():
        return request_prompt.strip()
    env_prompt = os.environ.get("GENPOSE2_SAM3_PROMPT") or os.environ.get("SAM6D_SAM3_PROMPT")
    if env_prompt and env_prompt.strip():
        return env_prompt.strip()
    from segment.sam3_seg_backend import DEFAULT_SAM3_PROMPT

    return DEFAULT_SAM3_PROMPT


def _run_segmentation(
    *,
    backend: str,
    rgb_path: Path,
    output_dir: Path,
    mask_exr_path: Path,
    detection_ism_path: Path,
    sam3_prompt: Optional[str] = None,
) -> tuple[float, float, Optional[Any], Optional[str]]:
    """
    运行分割，写出 ``mask.exr`` 与 ``detection_ism.json``。

    :return: (det_score, seg_elapsed_s, sam3_result 或 None, 实际使用的 sam3_prompt 或 None)
    """
    t0 = time.perf_counter()
    if backend == "yolo":
        yolo_weights = _yolo_weights_path()
        yolo_conf = float(os.environ.get("GENPOSE2_YOLO_CONF", "0.25"))
        yolo_imgsz = int(os.environ.get("GENPOSE2_YOLO_IMGSZ", "640"))
        yolo_class_id = int(os.environ.get("GENPOSE2_YOLO_CLASS_ID", "0"))
        yolo_max_inst = int(os.environ.get("GENPOSE2_YOLO_MAX_INSTANCES", "1"))
        class_id = None if yolo_class_id < 0 else yolo_class_id
        yolo_result = run_yolo_segmentation(
            yolo_weights,
            rgb_path,
            output_dir / "yolo_tmp",
            conf=yolo_conf,
            imgsz=yolo_imgsz,
            class_id=class_id,
            max_instances=yolo_max_inst,
            mask_exr_out=mask_exr_path,
        )
        det_score = float(yolo_result.score)
        _write_yolo_detection_ism(detection_ism_path, det_score, mask_exr_path)
        return det_score, time.perf_counter() - t0, None, None
    elif backend == "sam3":
        from segment.sam3_seg_backend import run_sam3_segmentation

        # 0 = 保留 SAM3 ISM 全部实例（与 detection_ism.json 条数一致，供多实例 PEM）
        sam3_max_inst = int(os.environ.get("GENPOSE2_SAM3_MAX_INSTANCES", "0"))
        prompt = _resolve_sam3_prompt(sam3_prompt)
        threshold = os.environ.get("GENPOSE2_SAM3_THRESHOLD") or os.environ.get("SAM6D_SAM3_THRESHOLD")
        mask_threshold = os.environ.get("GENPOSE2_SAM3_MASK_THRESHOLD") or os.environ.get(
            "SAM6D_SAM3_MASK_THRESHOLD"
        )
        sam3_result = run_sam3_segmentation(
            rgb_path,
            output_dir,
            prompt=prompt,
            threshold=float(threshold) if threshold is not None else None,
            mask_threshold=float(mask_threshold) if mask_threshold is not None else None,
            mask_exr_out=mask_exr_path,
            max_instances=sam3_max_inst,
        )
        det_score = float(sam3_result.score)
        print(
            f"[http_server] SAM3 prompt={prompt!r} instances for PEM: {sam3_result.num_instances} "
            f"(GENPOSE2_SAM3_MAX_INSTANCES={sam3_max_inst})"
        )
        _patch_detection_ism_mask_path(detection_ism_path, mask_exr_path)
        return det_score, time.perf_counter() - t0, sam3_result, prompt
    else:
        raise RuntimeError(f"unsupported segmentation backend: {backend}")


def _write_yolo_detection_ism(
    detection_ism_path: Path,
    det_score: float,
    mask_exr_path: Path,
) -> None:
    detection_ism_path.write_text(
        json.dumps(
            [
                {
                    "scene_id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "score": det_score,
                    "segmentation_mask_exr": str(mask_exr_path),
                }
            ]
        ),
        encoding="utf-8",
    )


def _ism_detections_sorted_by_score(detection_ism_path: Path) -> List[Dict[str, Any]]:
    """与 ``write_genpose_mask_exr_from_ism`` 相同：按 score 降序，对应 mask id 1..N。"""
    dets = json.loads(detection_ism_path.read_text(encoding="utf-8"))
    if not isinstance(dets, list):
        return []
    return sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)


def _patch_detection_ism_mask_path(detection_ism_path: Path, mask_exr_path: Path) -> None:
    """SAM3 已写入 RLE；补充 mask.exr 路径供下游使用。"""
    if not detection_ism_path.is_file():
        return
    dets = json.loads(detection_ism_path.read_text(encoding="utf-8"))
    if not isinstance(dets, list):
        return
    for det in dets:
        if isinstance(det, dict):
            det["segmentation_mask_exr"] = str(mask_exr_path)
    detection_ism_path.write_text(json.dumps(dets), encoding="utf-8")


async def _save_upload(upload: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _upload_basename(upload: UploadFile, default: str) -> str:
    name = Path(upload.filename or default).name
    if not name or name.startswith("."):
        return default
    return name


def _load_mask_array(mask_path: Path) -> np.ndarray:
    """读取 mask 为 (H,W) uint8 实例图：0=背景，1..N=实例 id。"""
    suf = mask_path.suffix.lower()
    if suf == ".exr":
        return Dataset.load_mask(str(mask_path))
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"cannot read mask: {mask_path}")
    if m.ndim == 3:
        m = m[:, :, 0]
    m = np.asarray(m)
    if m.dtype == np.bool_:
        return m.astype(np.uint8)
    if np.issubdtype(m.dtype, np.floating):
        fg = m > 0.5
        if not np.any(fg):
            raise ValueError("mask has no foreground pixels")
        uniq = np.unique(m[fg])
        if uniq.size == 1:
            out = np.zeros(m.shape, dtype=np.uint8)
            out[fg] = 1
            return out
        return m.astype(np.uint8)
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    fg_vals = np.unique(m[m > 0])
    if fg_vals.size == 1 and fg_vals[0] == 255:
        out = np.zeros(m.shape, dtype=np.uint8)
        out[m > 127] = 1
        return out
    return m


def _prepare_mask_for_inference(mask_path: Path, mask_exr_path: Path) -> np.ndarray:
    """将上传的 mask 规范为 GenPose2/cutoop 可读的 mask.exr 并读回。"""
    from segment.yolo_seg_backend import save_genpose2_mask_exr

    mask_u8 = _load_mask_array(mask_path)
    if not np.any(mask_u8 > 0):
        raise ValueError("mask has no foreground pixels")
    if int(mask_u8.max()) > 254:
        raise ValueError("mask instance id must be in 1..254")
    mask_exr_path.parent.mkdir(parents=True, exist_ok=True)
    save_genpose2_mask_exr(mask_u8, mask_exr_path)
    return Dataset.load_mask(str(mask_exr_path))


def _write_provided_mask_detection_ism(
    detection_ism_path: Path,
    mask_exr_path: Path,
    mask: np.ndarray,
) -> None:
    instance_ids = sorted(int(x) for x in np.unique(mask) if int(x) not in (0, 255))
    detection_ism_path.write_text(
        json.dumps(
            [
                {
                    "scene_id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "instance_id": iid,
                    "score": 1.0,
                    "segmentation_mask_exr": str(mask_exr_path),
                }
                for iid in instance_ids
            ]
        ),
        encoding="utf-8",
    )


def _camera_json_to_meta(camera: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    """将 warmup 风格 camera.json（cam_K + depth_scale）转为 InferDataset 用的 meta。"""
    if "camera" in camera and "intrinsics" in camera.get("camera", {}):
        return camera
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
        "depth_scale": float(camera.get("depth_scale", 1.0)),
    }


def _load_rgb_array(rgb_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read rgb: {rgb_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_depth_array(depth_path: Path, depth_scale: float) -> np.ndarray:
    suf = depth_path.suffix.lower()
    if suf == ".exr":
        return Dataset.load_depth(str(depth_path))
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"cannot read depth: {depth_path}")
    if d.ndim == 3:
        d = d[:, :, 0]
    d = np.asarray(d)
    if d.dtype == np.uint16:
        return d.astype(np.float32) * float(depth_scale)
    return d.astype(np.float32)


def _save_depth_colormap(
    depth_m: np.ndarray,
    output_path: Path,
    *,
    mask: Optional[np.ndarray] = None,
    vmin: float = 0.3,
    vmax: float = 1.0,
    max_depth: float = 4.0,
) -> None:
    """深度伪彩色图；可选叠加 mask 目标框（bbox + 轮廓），不含 ROI。"""
    valid = (depth_m > 0) & (depth_m <= max_depth)
    d_show = depth_m.copy()
    d_show[~valid] = 0
    span = max(vmax - vmin, 1e-6)
    d_norm = np.clip((d_show - vmin) / span, 0, 1)
    d_u8 = (d_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
    depth_color[~valid] = (30, 30, 30)

    if mask is not None:
        fg = np.asarray(mask) > 0
        if np.any(fg):
            fg_u8 = fg.astype(np.uint8)
            contours, _ = cv2.findContours(fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(depth_color, contours, -1, (0, 255, 0), 2)
            ys, xs = np.where(fg)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            cv2.rectangle(depth_color, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                depth_color,
                "target",
                (x1 + 4, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), depth_color):
        raise RuntimeError(f"failed to write depth colormap: {output_path}")


def _rotation_matrix_to_euler_zyx(rotation: List[List[float]]) -> List[float]:
    """与 warmup_http_service 相同：ZYX 欧拉角，单位弧度。"""
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


def _pose_4x4_to_t_and_R(pose_4x4: np.ndarray) -> tuple[List[float], List[List[float]]]:
    """GenPose 平移为米，转为与 warmup 一致的 mm。"""
    t_mm = (pose_4x4[:3, 3].astype(float) * 1000.0).tolist()
    rot = pose_4x4[:3, :3].astype(float).tolist()
    return t_mm, rot


def _cuboid_corners_mm(
    pose_4x4: np.ndarray,
    size_3d_m: np.ndarray,
) -> List[List[float]]:
    """Return the estimated oriented 3D bounding-box corners in camera-frame mm."""
    pose = np.asarray(pose_4x4, dtype=np.float64).reshape(4, 4)
    size = np.asarray(size_3d_m, dtype=np.float64).reshape(3)
    half_size_mm = 0.5 * size * 1000.0
    hx, hy, hz = half_size_mm.tolist()
    local_corners = np.array(
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
    center_mm = pose[:3, 3] * 1000.0
    corners = (pose[:3, :3] @ local_corners.T).T + center_mm
    return [[round(float(value), 2) for value in row] for row in corners]


def _build_manipulation_pose_response(
    pose_4x4: np.ndarray,
    size_3d_m: np.ndarray,
) -> Dict[str, Any]:
    """Build the compact pose contract shared by pick_pose and place_pose."""
    t_mm, rotation_matrix = _pose_4x4_to_t_and_R(pose_4x4)
    euler_zyx_rad = _rotation_matrix_to_euler_zyx(rotation_matrix)
    return {
        "pose": list(t_mm) + list(euler_zyx_rad),
        "corners_mm": _cuboid_corners_mm(pose_4x4, size_3d_m),
        "frame": "camera",
        "pose_unit": "mm_rad",
        "rotation_order": "zyx",
    }


def _validate_single_instance_mask(mask_path: Path) -> int:
    """Return the only foreground instance id or reject ambiguous masks."""
    mask = _load_mask_array(mask_path)
    instance_ids = sorted(
        int(value)
        for value in np.unique(mask)
        if int(value) not in (0, 255)
    )
    if len(instance_ids) != 1:
        raise ValueError(
            "mask must contain exactly one instance id; "
            f"found {len(instance_ids)}"
        )
    return instance_ids[0]


def _build_warmup_style_response(
    *,
    output_dir: Path,
    detection_ism_path: Path,
    detection_pem_path: Path,
    vis_ism_path: Path,
    vis_pem_path: Path,
    pose_4x4: np.ndarray,
    det_score: float,
    timing: InferTiming,
) -> Dict[str, Any]:
    t_mm, rotation_matrix = _pose_4x4_to_t_and_R(pose_4x4)
    euler_zyx_rad = _rotation_matrix_to_euler_zyx(rotation_matrix)
    return {
        "score": float(det_score),
        "xyz_mm": t_mm,
        "rotation_euler_zyx_rad": euler_zyx_rad,
        "rotation_order": "zyx",
        "pose_convention": "xyz is camera-frame translation in mm; rx, ry, rz are ZYX Euler angles in radians.",
        "xyzrxryrz": list(t_mm) + list(euler_zyx_rad),
        "xyzrxryrz_unit": "mm_rad",
        "result_dir": str(output_dir),
        "detection_ism_path": str(detection_ism_path),
        "detection_pem_path": str(detection_pem_path),
        "vis_ism_path": str(vis_ism_path),
        "vis_pem_path": str(vis_pem_path),
        "all_detections": str(detection_pem_path),
        "timing": timing,
    }


def _get_genpose():
    model = _genpose_holder.get("model")
    if model is None:
        raise RuntimeError("GenPose2 model not loaded; check server startup logs")
    return model


def _run_genpose2_pipeline(
    rgb_path: Path,
    depth_path: Path,
    camera_path: Path,
    output_dir: Path,
    *,
    sam3_prompt: Optional[str] = None,
    mask_path: Optional[Path] = None,
) -> _PipelineResult:
    timing: InferTiming = {}

    with camera_path.open("r", encoding="utf-8") as f:
        camera_json = json.load(f)
    if "cam_K" not in camera_json and "camera" not in camera_json:
        raise ValueError("camera.json must contain cam_K and/or camera.intrinsics")

    color = _load_rgb_array(rgb_path)
    h, w = color.shape[:2]
    meta = _camera_json_to_meta(camera_json, width=w, height=h)
    depth_scale = float(meta.get("depth_scale", camera_json.get("depth_scale", 0.001)))
    if depth_path.suffix.lower() == ".png" and depth_scale == 1.0:
        depth_scale = 0.001
    depth = _load_depth_array(depth_path, depth_scale)

    sam6d_dir = output_dir / "sam6d_results"
    sam6d_dir.mkdir(parents=True, exist_ok=True)
    mask_exr_path = sam6d_dir / "mask.exr"
    detection_ism_path = sam6d_dir / "detection_ism.json"
    detection_pem_path = sam6d_dir / "detection_pem.json"
    vis_pem_path = sam6d_dir / "vis_pem.png"
    vis_ism_path = sam6d_dir / "vis_ism.png"
    depth_colormap_path = sam6d_dir / "depth_colormap.png"

    sam3_seg_result: Optional[Any] = None
    resolved_sam3_prompt: Optional[str] = None

    if mask_path is not None:
        backend = "provided"
        print("[http_server] inference: using uploaded mask, skip segmentation")
        t0 = time.perf_counter()
        mask = _prepare_mask_for_inference(mask_path, mask_exr_path)
        _write_provided_mask_detection_ism(detection_ism_path, mask_exr_path, mask)
        det_score = 1.0
        seg_s = time.perf_counter() - t0
        yolo_s = None
        ism_s = None
    else:
        backend = _seg_backend()
        print(f"[http_server] inference segmentation backend: {backend}")
        det_score, seg_s, sam3_seg_result, resolved_sam3_prompt = _run_segmentation(
            backend=backend,
            rgb_path=rgb_path,
            output_dir=output_dir,
            mask_exr_path=mask_exr_path,
            detection_ism_path=detection_ism_path,
            sam3_prompt=sam3_prompt,
        )
        yolo_s = seg_s if backend == "yolo" else None
        ism_s = seg_s if backend == "sam3" else None
        mask = Dataset.load_mask(str(mask_exr_path))
    if depth.shape[:2] != (h, w) or mask.shape[:2] != (h, w):
        raise ValueError(
            f"rgb/depth/mask size mismatch: rgb={(h, w)}, depth={depth.shape[:2]}, mask={mask.shape[:2]}"
        )

    _save_depth_colormap(depth, depth_colormap_path, mask=mask)

    _argv_user = sys.argv[:]
    sys.argv = [sys.argv[0] if sys.argv else "http_server"]
    try:
        genpose = _get_genpose()
        t0 = time.perf_counter()
        data = InferDataset(
            {"depth": depth, "color": color, "mask": mask, "meta": meta},
            img_size=genpose.cfg.img_size,
            device=genpose.cfg.device,
            n_pts=genpose.cfg.num_points,
        )
        pose, length = genpose.inference(data, prev_pose=None, tracking=False, tracking_T0=0.15)
        infer_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        vis_bgr = visualize_pose(data, pose, length, visualize_pts=False, visualize_image=False)
        cv2.imwrite(str(vis_pem_path), vis_bgr)
        if backend == "sam3":
            # vis_ism.png / vis_sam3_seg.png 由 sam3_seg_backend 在分割阶段生成，勿用 PEM 覆盖
            if not vis_ism_path.is_file():
                from segment.sam3_seg_backend import visualize_sam3_mask_exr

                visualize_sam3_mask_exr(rgb_path, mask_exr_path, vis_ism_path)
        elif backend in ("yolo", "provided"):
            from utils.exr_visualize import visualize_mask_exr

            visualize_mask_exr(
                mask_exr_path,
                rgb_path=rgb_path,
                output_path=vis_ism_path,
            )
        else:
            cv2.imwrite(str(vis_ism_path), vis_bgr)
        vis_s = time.perf_counter() - t0
        pose_s = infer_s + vis_s
    finally:
        sys.argv = _argv_user

    templates_s = 0.0
    seg_elapsed = float(seg_s)
    timing = {
        "ism_s": ism_s,
        "yolo_s": yolo_s,
        "templates_s": templates_s,
        "pose_s": pose_s,
        "pipeline_s": templates_s + seg_elapsed + pose_s,
    }

    poses_np = pose[0].cpu().numpy()
    lengths_np = length[0].cpu().numpy()
    if poses_np.shape[0] == 0:
        raise RuntimeError("GenPose2 returned no object poses")

    ism_ordered = _ism_detections_sorted_by_score(detection_ism_path)

    num_mask_ids = int(np.unique(mask[mask > 0]).size) if np.any(mask > 0) else 0
    print(
        f"[http_server] GenPose2 PEM: {poses_np.shape[0]} pose(s), "
        f"mask instance ids={num_mask_ids}"
    )

    detections_pem: List[Dict[str, Any]] = []
    for i in range(poses_np.shape[0]):
        t_mm, rot = _pose_4x4_to_t_and_R(poses_np[i])
        src_det: Optional[Dict[str, Any]] = None
        if sam3_seg_result is not None and sam3_seg_result.instance_dets and i < len(
            sam3_seg_result.instance_dets
        ):
            src_det = sam3_seg_result.instance_dets[i]
            inst_score = float(sam3_seg_result.instance_scores[i])
        elif i < len(ism_ordered):
            src_det = ism_ordered[i]
            inst_score = float(src_det.get("score", det_score))
        else:
            inst_score = float(det_score)
        entry: Dict[str, Any] = {
            "scene_id": 0,
            "image_id": 0,
            "category_id": int(src_det.get("category_id", 1)) if src_det else 1,
            "instance_id": i + 1,
            "score": inst_score,
            "t": t_mm,
            "R": rot,
            "size_3d": lengths_np[i].astype(float).tolist(),
        }
        if src_det and "bbox" in src_det:
            entry["bbox"] = src_det["bbox"]
        detections_pem.append(entry)
    detection_pem_path.write_text(json.dumps(detections_pem), encoding="utf-8")

    resp = _build_warmup_style_response(
        output_dir=output_dir,
        detection_ism_path=detection_ism_path,
        detection_pem_path=detection_pem_path,
        vis_ism_path=vis_ism_path,
        vis_pem_path=vis_pem_path,
        pose_4x4=poses_np[0],
        det_score=det_score,
        timing=timing,
    )
    resp["seg_backend"] = backend
    resp["num_instances"] = len(detections_pem)
    resp["detections_pem"] = detections_pem
    resp["depth_colormap_path"] = str(depth_colormap_path)
    if backend == "provided":
        resp["mask_source"] = "upload"
    if backend == "sam3":
        resp["sam3_prompt"] = resolved_sam3_prompt
        vis_sam3_seg = sam6d_dir / "vis_sam3_seg.png"
        if vis_sam3_seg.is_file():
            resp["vis_sam3_seg_path"] = str(vis_sam3_seg)
    return _PipelineResult(
        payload=resp,
        primary_pose_4x4=poses_np[0],
        primary_size_3d_m=lengths_np[0],
    )


@app.on_event("startup")
async def _startup_load_models() -> None:
    backend = _seg_backend()
    print(f"[http_server] segmentation backend: {backend}")
    if backend == "yolo":
        try:
            wp = _yolo_weights_path()
            preload_yolo_model(
                wp,
                imgsz=int(os.environ.get("GENPOSE2_YOLO_IMGSZ", "640")),
                conf=float(os.environ.get("GENPOSE2_YOLO_CONF", "0.25")),
                class_id=int(os.environ.get("GENPOSE2_YOLO_CLASS_ID", "0")),
            )
            print(f"[http_server] YOLO preload done: {wp}")
        except Exception as exc:
            print(f"[http_server] YOLO preload failed: {type(exc).__name__}: {exc}")
    else:
        from segment.sam3_seg_backend import (
            DEFAULT_SAM3_INFER_SCRIPT,
            DEFAULT_SAM3_PYTHON,
            _cocomask,
            _sam3_infer_script,
            _sam3_python,
            _sam3_root,
            _validate_sam3_toolchain,
        )

        py = _sam3_python()
        script = _sam3_infer_script()
        print(
            f"[http_server] SAM3 (no preload): cwd={_sam3_root()} "
            f"python={py} "
            f"infer_script={script} "
            f"(default infer: {DEFAULT_SAM3_INFER_SCRIPT})"
        )
        try:
            _validate_sam3_toolchain(py, script)
            print("[http_server] SAM3 toolchain paths: ok")
        except FileNotFoundError as exc:
            print(f"[http_server] SAM3 toolchain missing: {exc}")
        try:
            _cocomask()
            print("[http_server] SAM3 dependency pycocotools: ok")
        except ImportError as exc:
            print(f"[http_server] SAM3 dependency missing: {exc}")

    score = os.environ.get("GENPOSE2_SCORE_CKPT", "results/ckpts/ScoreNet/scorenet.pth")
    energy = os.environ.get("GENPOSE2_ENERGY_CKPT", "results/ckpts/EnergyNet/energynet.pth")
    scale = os.environ.get("GENPOSE2_SCALE_CKPT", "results/ckpts/ScaleNet/scalenet.pth")

    _argv_user = sys.argv[:]
    sys.argv = [sys.argv[0] if sys.argv else "http_server"]
    try:
        print("[http_server] loading GenPose2 models...")
        _genpose_holder["model"] = create_genpose2(
            score_model_path=str(Path(score).expanduser().resolve()),
            energy_model_path=str(Path(energy).expanduser().resolve()),
            scale_model_path=str(Path(scale).expanduser().resolve()),
        )
        print("[http_server] GenPose2 models loaded")
    except Exception as exc:
        print(f"[http_server] GenPose2 load failed: {type(exc).__name__}: {exc}")
    finally:
        sys.argv = _argv_user


@app.get("/health")
def health() -> Dict[str, Any]:
    backend = _seg_backend()
    yolo_cfg = os.environ.get("GENPOSE2_YOLO_WEIGHTS", DEFAULT_YOLO_WEIGHTS)
    try:
        yolo_resolved = str(_yolo_weights_path())
        yolo_exists = True
    except Exception:
        yolo_resolved = str(_resolve_repo_path(yolo_cfg))
        yolo_exists = Path(yolo_resolved).is_file()
    payload: Dict[str, Any] = {
        "status": "ok",
        "root_dir": str(ROOT_DIR),
        "output_root": str(_output_root()),
        "genpose_loaded": _genpose_holder.get("model") is not None,
        "seg_backend": backend,
        "yolo_weights": yolo_cfg,
        "yolo_weights_resolved": yolo_resolved,
        "yolo_weights_exists": yolo_exists,
        "yolo_weights_default": DEFAULT_YOLO_WEIGHTS,
    }
    if backend == "sam3":
        from segment.sam3_seg_backend import (
            DEFAULT_SAM3_INFER_SCRIPT,
            DEFAULT_SAM3_PYTHON,
            DEFAULT_SAM3_PROMPT,
            _sam3_infer_script,
            _sam3_python,
            _sam3_root,
        )

        payload["sam3_root"] = str(_sam3_root())
        payload["sam3_python"] = _sam3_python()
        payload["sam3_python_default"] = DEFAULT_SAM3_PYTHON
        payload["sam3_infer_script"] = str(_sam3_infer_script())
        payload["sam3_infer_script_default"] = DEFAULT_SAM3_INFER_SCRIPT
        payload["sam3_prompt"] = os.environ.get("GENPOSE2_SAM3_PROMPT") or os.environ.get(
            "SAM6D_SAM3_PROMPT", DEFAULT_SAM3_PROMPT
        )
    return payload


@app.get("/manipulation/health")
def manipulation_health() -> Any:
    """Return the minimal readiness contract used by the Agent."""
    if _genpose_holder.get("model") is None:
        return JSONResponse(
            status_code=503,
            content={"error_code": "MODEL_NOT_READY"},
        )
    return {"status": "READY"}


def _validate_camera_file(camera_path: Path) -> None:
    with camera_path.open("r", encoding="utf-8") as file_handle:
        camera_json = json.load(file_handle)
    if not isinstance(camera_json, dict):
        raise ValueError("camera.json root must be an object")
    if "cam_K" not in camera_json and "camera" not in camera_json:
        raise ValueError("camera.json must contain cam_K or camera.intrinsics")
    if "cam_K" in camera_json and "depth_scale" not in camera_json:
        raise ValueError("camera.json with cam_K must contain depth_scale")


async def _execute_pose_request(
    *,
    operation: str,
    rgb: UploadFile,
    depth: UploadFile,
    camera: UploadFile,
    mask: Optional[UploadFile],
    sam3_prompt: Optional[str],
    require_single_instance: bool,
) -> _PipelineResult:
    if _genpose_holder.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail="GenPose2 model not loaded on server startup",
        )

    request_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output_dir = _output_root() / request_id
    input_dir = output_dir / "inputs"
    output_dir.mkdir(parents=True, exist_ok=False)

    rgb_path = input_dir / _upload_basename(rgb, "rgb.png")
    depth_path = input_dir / _upload_basename(depth, "depth.png")
    camera_path = input_dir / _upload_basename(camera, "camera.json")
    mask_path: Optional[Path] = None

    upload_started = time.perf_counter()
    await _save_upload(rgb, rgb_path)
    await _save_upload(depth, depth_path)
    await _save_upload(camera, camera_path)
    if mask is not None:
        mask_path = input_dir / _upload_basename(mask, "mask.exr")
        await _save_upload(mask, mask_path)
        if not mask_path.is_file() or mask_path.stat().st_size == 0:
            mask_path.unlink(missing_ok=True)
            mask_path = None
    upload_s = time.perf_counter() - upload_started

    if require_single_instance and mask_path is None:
        raise HTTPException(status_code=400, detail="invalid mask: file is empty")
    if require_single_instance and mask_path is not None:
        try:
            _validate_single_instance_mask(mask_path)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid mask: {exc}") from exc

    try:
        _validate_camera_file(camera_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid camera.json: {exc}") from exc

    LOGGER.info(
        "%s %s mask=%s segmentation=%s",
        operation,
        request_id,
        "provided" if mask_path else "none",
        "skip" if mask_path else _seg_backend(),
    )

    async with infer_lock:
        try:
            result = await asyncio.to_thread(
                _run_genpose2_pipeline,
                rgb_path,
                depth_path,
                camera_path,
                output_dir,
                sam3_prompt=sam3_prompt,
                mask_path=mask_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            LOGGER.warning("%s %s invalid input: %s", operation, request_id, exc)
            if require_single_instance:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("%s %s inference failed", operation, request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    timing = result.payload["timing"]
    timing["upload_s"] = upload_s
    timing["total_s"] = upload_s + float(timing["pipeline_s"])
    result.payload["timing"] = timing
    LOGGER.info(
        "%s %s succeeded score=%s total=%.3fs",
        operation,
        request_id,
        result.payload.get("score"),
        timing["total_s"],
    )
    return result


@app.post("/infer")
async def infer(
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    camera: UploadFile = File(...),
    mask: Optional[UploadFile] = File(None),
    sam3_prompt: Optional[str] = Form(None),
) -> JSONResponse:
    result = await _execute_pose_request(
        operation="infer",
        rgb=rgb,
        depth=depth,
        camera=camera,
        mask=mask,
        sam3_prompt=sam3_prompt,
        require_single_instance=False,
    )
    return JSONResponse(content=result.payload)


async def _manipulation_pose(
    *,
    operation: str,
    rgb: UploadFile,
    depth: UploadFile,
    camera: UploadFile,
    mask: UploadFile,
) -> JSONResponse:
    try:
        result = await _execute_pose_request(
            operation=operation,
            rgb=rgb,
            depth=depth,
            camera=camera,
            mask=mask,
            sam3_prompt=None,
            require_single_instance=True,
        )
        response = _build_manipulation_pose_response(
            result.primary_pose_4x4,
            result.primary_size_3d_m,
        )
        return JSONResponse(content=response)
    except HTTPException as exc:
        if exc.status_code == 400:
            error_code = "INVALID_REQUEST"
        elif exc.status_code == 503:
            error_code = "MODEL_NOT_READY"
        else:
            error_code = "EXECUTION_FAILED"
        return JSONResponse(
            status_code=exc.status_code,
            content={"error_code": error_code, "message": str(exc.detail)},
        )


@app.post("/manipulation/pick_pose")
async def manipulation_pick_pose(
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    camera: UploadFile = File(...),
    mask: UploadFile = File(...),
) -> JSONResponse:
    return await _manipulation_pose(
        operation="pick_pose",
        rgb=rgb,
        depth=depth,
        camera=camera,
        mask=mask,
    )


@app.post("/manipulation/place_pose")
async def manipulation_place_pose(
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    camera: UploadFile = File(...),
    mask: UploadFile = File(...),
) -> JSONResponse:
    return await _manipulation_pose(
        operation="place_pose",
        rgb=rgb,
        depth=depth,
        camera=camera,
        mask=mask,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GenPose2 HTTP service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8084, type=int)
    parser.add_argument(
        "--seg-backend",
        choices=("yolo", "sam3"),
        default=os.environ.get("GENPOSE2_SEG_BACKEND", DEFAULT_SEG_BACKEND),
        help="分割后端（默认 yolo，或已 export 的 GENPOSE2_SEG_BACKEND）",
    )
    args = parser.parse_args()
    os.environ["GENPOSE2_SEG_BACKEND"] = args.seg_backend

    import uvicorn

    print(f"[http_server] starting uvicorn seg_backend={args.seg_backend}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
