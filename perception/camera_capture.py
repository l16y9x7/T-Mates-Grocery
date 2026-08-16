"""Capture and validate one RGB-D snapshot from a configured robot camera."""

from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from config import camera_depth_snapshot_url, camera_snapshot_url


MAX_RGB_BYTES = 20 * 1024 * 1024
MAX_DEPTH_BYTES = 32 * 1024 * 1024
CAMERA_TIMEOUT_SECONDS = float(
    os.getenv(
        "INSPECT_CAMERA_TIMEOUT_SECONDS",
        os.getenv("CAMERA_SNAPSHOT_TIMEOUT_SECONDS", "5"),
    )
)
INSPECT_TEMP_DIR_ENVIRONMENT = "INSPECT_TEMP_DIR"


class CameraCaptureError(RuntimeError):
    """Raised when a live camera frame cannot be fetched or validated."""


@dataclass(frozen=True)
class CapturedRgbd:
    directory: Path
    rgb_path: Path
    depth_path: Path
    rgb: np.ndarray
    depth_mm: np.ndarray


def inspection_temp_root() -> Path | None:
    """Return the optional configured root for per-request temporary captures."""

    configured = os.getenv(INSPECT_TEMP_DIR_ENVIRONMENT, "").strip()
    if not configured:
        return None
    root = Path(configured).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def inspection_temporary_directory() -> tempfile.TemporaryDirectory[str]:
    """Create an automatically cleaned directory for one inspection request."""

    root = inspection_temp_root()
    return tempfile.TemporaryDirectory(
        prefix="inspect-",
        dir=str(root) if root is not None else None,
    )


def capture_head_rgbd(
    directory: str | Path,
    *,
    session: Any = requests,
) -> CapturedRgbd:
    """Fetch head-camera RGB and aligned 16UC1 depth into ``directory``."""

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    rgb_url = camera_snapshot_url("head")
    depth_url = camera_depth_snapshot_url("head")
    rgb_response = _get(session, rgb_url, "RGB")
    depth_response = _get(session, depth_url, "depth")

    rgb_bytes = bytes(rgb_response.content)
    depth_bytes = bytes(depth_response.content)
    if not rgb_bytes or len(rgb_bytes) > MAX_RGB_BYTES:
        raise CameraCaptureError("head camera RGB is empty or exceeds 20 MB")
    if not depth_bytes or len(depth_bytes) > MAX_DEPTH_BYTES:
        raise CameraCaptureError("head camera depth is empty or exceeds 32 MB")

    rgb = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise CameraCaptureError("head camera RGB is not a valid color image")
    depth_mm = _decode_depth(depth_bytes, depth_response.headers, rgb.shape[:2])

    rgb_path = target / "rgb.jpg"
    depth_path = target / "depth_mm.npy"
    success, encoded_rgb = cv2.imencode(".jpg", rgb)
    if not success:
        raise CameraCaptureError("failed to encode the temporary RGB image")
    try:
        rgb_path.write_bytes(encoded_rgb.tobytes())
        np.save(depth_path, depth_mm, allow_pickle=False)
        (target / "meta.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "camera": "head",
                    "captured_at": datetime.now(UTC).isoformat(),
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                    "rgb": {"file": rgb_path.name, "encoding": "bgr8"},
                    "depth": {
                        "file": depth_path.name,
                        "encoding": "16UC1",
                        "dtype": "uint16",
                        "unit": "millimeter",
                        "aligned_to": "rgb",
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise CameraCaptureError(f"failed to save temporary RGB-D capture: {error}") from error

    return CapturedRgbd(
        directory=target,
        rgb_path=rgb_path,
        depth_path=depth_path,
        rgb=rgb,
        depth_mm=depth_mm,
    )


def _get(session: Any, url: str, label: str) -> Any:
    try:
        response = session.get(url, timeout=CAMERA_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as error:
        raise CameraCaptureError(f"head camera {label} request failed: {error}") from error
    return response


def _decode_depth(
    depth_bytes: bytes,
    headers: Any,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    raw_header_names = (
        "X-Image-Width",
        "X-Image-Height",
        "X-Image-Encoding",
        "X-Image-Step",
        "X-Image-Is-Bigendian",
    )
    if all(_header(headers, name) is not None for name in raw_header_names):
        depth = _decode_raw_16uc1(depth_bytes, headers)
    elif depth_bytes.startswith(b"\x93NUMPY"):
        try:
            depth = np.load(io.BytesIO(depth_bytes), allow_pickle=False)
        except (OSError, ValueError, TypeError) as error:
            raise CameraCaptureError(f"head camera NPY depth is invalid: {error}") from error
    else:
        depth = cv2.imdecode(
            np.frombuffer(depth_bytes, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if depth is None:
            raise CameraCaptureError(
                "head camera depth must be raw 16UC1, NPY, or a 16-bit image"
            )

    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        raise CameraCaptureError("head camera depth must be a two-dimensional array")
    if depth.shape != expected_shape:
        raise CameraCaptureError(
            "head camera RGB/depth size mismatch: "
            f"rgb={expected_shape}, depth={depth.shape}"
        )
    if not np.issubdtype(depth.dtype, np.integer):
        raise CameraCaptureError("head camera depth must use an integer millimeter format")
    if depth.dtype.itemsize != 2:
        raise CameraCaptureError("head camera depth must be 16-bit")
    return np.asarray(depth, dtype=np.uint16)


def _decode_raw_16uc1(depth_bytes: bytes, headers: Any) -> np.ndarray:
    try:
        width = int(_header(headers, "X-Image-Width"))
        height = int(_header(headers, "X-Image-Height"))
        encoding = str(_header(headers, "X-Image-Encoding")).strip().upper()
        step = int(_header(headers, "X-Image-Step"))
        is_bigendian = int(_header(headers, "X-Image-Is-Bigendian"))
    except (TypeError, ValueError) as error:
        raise CameraCaptureError("head camera depth headers are invalid") from error
    if (
        width <= 0
        or height <= 0
        or encoding != "16UC1"
        or step != width * 2
        or is_bigendian not in {0, 1}
        or len(depth_bytes) != step * height
    ):
        raise CameraCaptureError("head camera raw depth metadata is inconsistent")
    dtype = np.dtype(">u2" if is_bigendian else "<u2")
    return np.frombuffer(depth_bytes, dtype=dtype).reshape(height, width)


def _header(headers: Any, name: str) -> Any:
    if hasattr(headers, "get"):
        value = headers.get(name)
        if value is None:
            value = headers.get(name.lower())
        return value
    return None
