"""Model-free HTTP and input helpers for the service inference frontend."""

from __future__ import annotations

import base64
import json
import math
import time
from contextlib import ExitStack
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests
from PIL import Image
from pycocotools import mask as cocomask


@dataclass(frozen=True)
class CameraSpec:
    """Resolved camera payload and numeric intrinsics."""

    camera_json: Dict[str, Any]
    intrinsics: np.ndarray
    depth_scale: float


@dataclass(frozen=True)
class ServiceResult:
    """One parsed service response with client-observed wall time."""

    payload: Dict[str, Any]
    elapsed_ms: float


class ServiceCallError(RuntimeError):
    """HTTP, transport, or response-contract failure from a backend service."""

    def __init__(
        self,
        status_code: int,
        payload: Any,
        *,
        elapsed_ms: float = 0.0,
    ) -> None:
        super().__init__(f"service HTTP {status_code}: {payload}")
        self.status_code = status_code
        self.payload = payload
        self.elapsed_ms = elapsed_ms


def normalize_box(
    box: Sequence[float], image_size: Tuple[int, int]
) -> List[int]:
    """Order and clamp an xyxy box to an image's pixel bounds."""

    if len(box) != 4:
        raise ValueError("box must be [x1, y1, x2, y2]")
    width, height = image_size
    if width < 2 or height < 2:
        raise ValueError("image must be at least 2x2 pixels")
    values = [float(value) for value in box]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("box coordinates must be finite")
    x1, x2 = sorted((values[0], values[2]))
    y1, y2 = sorted((values[1], values[3]))
    result = [
        int(round(min(max(x1, 0.0), width - 1))),
        int(round(min(max(y1, 0.0), height - 1))),
        int(round(min(max(x2, 0.0), width - 1))),
        int(round(min(max(y2, 0.0), height - 1))),
    ]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("box must have positive area inside the image")
    return result


def _intrinsics_from_payload(payload: Dict[str, Any]) -> np.ndarray:
    if "cam_K" in payload:
        values = [float(value) for value in payload["cam_K"]]
        if len(values) != 9:
            raise ValueError("camera cam_K must contain nine values")
        intrinsics = np.asarray(values, dtype=np.float64).reshape(3, 3)
    elif "camera" in payload and "intrinsics" in payload.get("camera", {}):
        raw = payload["camera"]["intrinsics"]
        intrinsics = np.asarray(
            [
                [float(raw["fx"]), 0.0, float(raw["cx"])],
                [0.0, float(raw["fy"]), float(raw["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError("camera.json must contain cam_K or camera.intrinsics")
    if not np.isfinite(intrinsics).all():
        raise ValueError("camera intrinsics must be finite")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("camera fx and fy must be positive")
    return intrinsics


def resolve_camera(
    camera_path: Optional[Path],
    *,
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
    depth_scale: Optional[float],
) -> CameraSpec:
    """Resolve camera JSON, with a complete manual tuple taking precedence."""

    manual = [fx, fy, cx, cy, depth_scale]
    if any(value is not None for value in manual):
        if not all(
            value is not None and math.isfinite(float(value)) for value in manual
        ):
            raise ValueError(
                "manual camera parameters must provide all five finite values"
            )
        if float(fx) <= 0 or float(fy) <= 0 or float(depth_scale) <= 0:
            raise ValueError("fx, fy, and depth_scale must be positive")
        values = [
            float(fx),
            0.0,
            float(cx),
            0.0,
            float(fy),
            float(cy),
            0.0,
            0.0,
            1.0,
        ]
        payload: Dict[str, Any] = {
            "cam_K": values,
            "depth_scale": float(depth_scale),
        }
    else:
        if camera_path is None:
            raise ValueError(
                "upload camera.json or provide all manual camera parameters"
            )
        path = Path(camera_path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("camera.json must contain a JSON object")

    intrinsics = _intrinsics_from_payload(payload)
    resolved_scale = float(payload.get("depth_scale", 0.001))
    if not math.isfinite(resolved_scale) or resolved_scale <= 0:
        raise ValueError("camera depth_scale must be a positive finite value")
    return CameraSpec(
        camera_json=payload,
        intrinsics=intrinsics,
        depth_scale=resolved_scale,
    )


def write_camera_json(camera: CameraSpec, output_path: Path) -> Path:
    """Write the effective camera payload used for the pose request."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(camera.camera_json, handle, ensure_ascii=False, indent=2)
    return path


def decode_coco_rle(rle: Dict[str, Any]) -> np.ndarray:
    """Decode compressed or uncompressed COCO RLE into a boolean mask."""

    encoded = dict(rle)
    if isinstance(encoded.get("counts"), str):
        encoded["counts"] = encoded["counts"].encode("ascii")
    decoded = np.asarray(cocomask.decode(encoded), dtype=bool)
    if decoded.ndim == 3 and decoded.shape[2] == 1:
        decoded = decoded[:, :, 0]
    if decoded.ndim != 2:
        raise ValueError("SAM3 RLE must decode to one two-dimensional mask")
    if not decoded.any():
        raise ValueError("SAM3 mask is empty")
    return decoded


def write_mask_png(mask: np.ndarray, output_path: Path) -> Path:
    """Write a pose-service-compatible single-channel mask PNG."""

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array * 255).astype(np.uint8), mode="L").save(path)
    return path


def _mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    encoded = cocomask.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {
        "size": [int(encoded["size"][0]), int(encoded["size"][1])],
        "counts": counts,
    }


def _bbox_from_mask(mask: np.ndarray) -> List[int]:
    rows, columns = np.where(mask)
    if columns.size == 0:
        return [0, 0, 0, 0]
    x1, x2 = int(columns.min()), int(columns.max()) + 1
    y1, y2 = int(rows.min()), int(rows.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def _health_url(service_url: str) -> str:
    parsed = urlsplit(service_url)
    return urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))


def _is_text_only_trt_service(
    client: requests.Session,
    service_url: str,
    timeout_s: float,
) -> bool:
    get_method = getattr(client, "get", None)
    if get_method is None:
        return False
    try:
        response = get_method(
            _health_url(service_url), timeout=min(float(timeout_s), 3.0)
        )
        body = response.json()
    except (requests.RequestException, ValueError, TypeError):
        return False
    return bool(
        isinstance(body, dict)
        and (
            body.get("service") == "sam3-trt-infer"
            or body.get("backend") == "tensorrt"
        )
    )


def _remap_roi_detections(
    payload: Dict[str, Any],
    box: Sequence[int],
    image_size: Tuple[int, int],
) -> Dict[str, Any]:
    width, height = image_size
    x1, y1, x2, y2 = [int(value) for value in box]
    crop_shape = (y2 - y1 + 1, x2 - x1 + 1)
    detections = payload.get("detections")
    if not isinstance(detections, list):
        raise ValueError("SAM3 TensorRT response detections must be a list")
    remapped = []
    for detection in detections:
        if not isinstance(detection, dict) or "segmentation" not in detection:
            raise ValueError("SAM3 TensorRT detection is missing segmentation RLE")
        crop_mask = decode_coco_rle(detection["segmentation"])
        if crop_mask.shape != crop_shape:
            raise ValueError(
                f"SAM3 ROI mask shape {crop_mask.shape} does not match "
                f"expected {crop_shape}"
            )
        full_mask = np.zeros((height, width), dtype=bool)
        full_mask[y1 : y2 + 1, x1 : x2 + 1] = crop_mask
        mapped = dict(detection)
        mapped["segmentation"] = _mask_to_rle(full_mask)
        mapped["bbox"] = _bbox_from_mask(full_mask)
        remapped.append(mapped)
    output = dict(payload)
    output["detections"] = remapped
    output["num_detections"] = len(remapped)
    output["box"] = [x1, y1, x2, y2]
    output["box_mode"] = "roi_crop_compat"
    output["source_image_size"] = [width, height]
    return output


def select_best_detection(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Select the highest-scoring detection and validate its RLE presence."""

    detections = payload.get("detections")
    if not isinstance(detections, list) or not detections:
        raise ValueError("SAM3 response contains no detections")
    selected = max(detections, key=lambda item: float(item.get("score", 0.0)))
    if "segmentation" not in selected:
        raise ValueError("SAM3 detection is missing segmentation RLE")
    return selected


def validate_pose_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the compact pose response consumed by all visualizations."""

    if not isinstance(payload, dict):
        raise ValueError("pose response must be a JSON object")
    pose = np.asarray(payload.get("pose"), dtype=float)
    corners = np.asarray(payload.get("corners_mm"), dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("pose must contain six finite values")
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError("corners_mm must contain eight finite xyz points")
    if payload.get("frame") != "camera":
        raise ValueError("pose frame must be camera")
    if payload.get("pose_unit") != "mm_rad":
        raise ValueError("pose_unit must be mm_rad")
    if payload.get("rotation_order") != "zyx":
        raise ValueError("rotation_order must be zyx")
    return payload


def _json_body(response: requests.Response, elapsed_ms: float) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError as exc:
        raise ServiceCallError(
            response.status_code,
            {"error": "service returned non-JSON content"},
            elapsed_ms=elapsed_ms,
        ) from exc
    if not isinstance(body, dict):
        raise ServiceCallError(
            response.status_code,
            {"error": "service JSON response must be an object", "body": body},
            elapsed_ms=elapsed_ms,
        )
    return body


def call_sam3_box(
    url: str,
    rgb_path: Path,
    box: Sequence[int],
    *,
    timeout_s: float,
    session: Optional[requests.Session] = None,
) -> ServiceResult:
    """Call the existing SAM3 process using one positive geometric box."""

    client = session or requests.Session()
    started = time.perf_counter()
    try:
        with Image.open(rgb_path) as source_image:
            rgb = source_image.convert("RGB")
        effective_box = normalize_box(box, rgb.size)
        trt_roi_mode = _is_text_only_trt_service(
            client, url, float(timeout_s)
        )
        if trt_roi_mode:
            x1, y1, x2, y2 = effective_box
            crop = rgb.crop((x1, y1, x2 + 1, y2 + 1))
            buffer = BytesIO()
            crop.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            request_payload = {
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                "prompt": "object",
                "save_vis": False,
                "return_vis_base64": False,
            }
        else:
            request_payload = {
                "image_base64": base64.b64encode(Path(rgb_path).read_bytes()).decode(
                    "ascii"
                ),
                "box": effective_box,
                "save_vis": False,
                "return_vis_base64": False,
            }
        try:
            response = client.post(
                url,
                json=request_payload,
                timeout=float(timeout_s),
            )
        except requests.RequestException as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            raise ServiceCallError(
                0, {"error": str(exc)}, elapsed_ms=elapsed_ms
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        body = _json_body(response, elapsed_ms)
        if not response.ok or body.get("ok") is False:
            raise ServiceCallError(
                response.status_code, body, elapsed_ms=elapsed_ms
            )
        if trt_roi_mode:
            body = _remap_roi_detections(body, effective_box, rgb.size)
        else:
            body.setdefault("box", effective_box)
            body.setdefault("box_mode", "native_box")
        return ServiceResult(payload=body, elapsed_ms=elapsed_ms)
    finally:
        if session is None:
            client.close()


def call_pose_service(
    url: str,
    rgb_path: Path,
    depth_path: Path,
    camera_path: Path,
    mask_path: Path,
    *,
    timeout_s: float,
    session: Optional[requests.Session] = None,
) -> ServiceResult:
    """Call a documented manipulation pose endpoint with exactly four files."""

    client = session or requests.Session()
    started = time.perf_counter()
    try:
        try:
            with ExitStack() as stack:
                paths = {
                    "rgb": Path(rgb_path),
                    "depth": Path(depth_path),
                    "camera": Path(camera_path),
                    "mask": Path(mask_path),
                }
                files = {
                    key: (
                        path.name,
                        stack.enter_context(path.open("rb")),
                    )
                    for key, path in paths.items()
                }
                response = client.post(
                    url,
                    files=files,
                    timeout=float(timeout_s),
                )
        except requests.RequestException as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            raise ServiceCallError(
                0, {"error": str(exc)}, elapsed_ms=elapsed_ms
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        body = _json_body(response, elapsed_ms)
        if not response.ok or body.get("ok") is False:
            raise ServiceCallError(
                response.status_code, body, elapsed_ms=elapsed_ms
            )
        try:
            validated = validate_pose_response(body)
        except ValueError as exc:
            raise ServiceCallError(
                response.status_code,
                {"error": str(exc), "body": body},
                elapsed_ms=elapsed_ms,
            ) from exc
        return ServiceResult(payload=validated, elapsed_ms=elapsed_ms)
    finally:
        if session is None:
            client.close()
