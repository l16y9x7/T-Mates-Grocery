"""Shared service and camera endpoint configuration for perception APIs."""

from __future__ import annotations

import os


CAMERA_SERVICE_URL = os.getenv(
    "CAMERA_SERVICE_URL",
    "http://192.168.130.50:8085",
).rstrip("/")

SKU_API_URL = os.getenv(
    "SKU_API_URL",
    os.getenv("SKU_BASE_URL", "http://127.0.0.1:25540"),
).rstrip("/")
SAM3_URL = os.getenv(
    "SAM3_URL",
    "http://211.137.21.33:25541/api/v1/segment",
)
QWEN3_URL = os.getenv(
    "QWEN3_URL",
    "http://211.137.21.33:25542/v1/chat/completions",
)
QWEN3_MODEL = os.getenv(
    "QWEN3_MODEL",
    os.getenv("QWEN_MODEL", "Qwen3-VL-4B-Instruct"),
)


def camera_snapshot_url(camera: str) -> str:
    """Return the configured color snapshot URL for a camera name."""

    defaults = {
        "left": f"{CAMERA_SERVICE_URL}/camera/snapshot?camera=left_wrist&type=color",
        "right": f"{CAMERA_SERVICE_URL}/camera/snapshot?camera=right_wrist&type=color",
        "head": f"{CAMERA_SERVICE_URL}/camera/snapshot?camera=head&type=color",
    }
    environment_names = {
        "left": "LEFT_CAMERA_SNAPSHOT_URL",
        "right": "RIGHT_CAMERA_SNAPSHOT_URL",
        "head": "RECEIPT_CAMERA_URL",
    }
    normalized_camera = camera.strip().lower()
    if normalized_camera not in defaults:
        raise ValueError(f"unsupported camera: {camera}")

    legacy_url = os.getenv("CAMERA_SNAPSHOT_URL", "").strip()
    if normalized_camera == "left" and legacy_url:
        return os.getenv(environment_names[normalized_camera], "").strip() or legacy_url
    return (
        os.getenv(environment_names[normalized_camera], "").strip()
        or defaults[normalized_camera]
    )


def hand_camera_snapshot_url(hand: str) -> str:
    """Resolve left/right robot hand names to their wrist camera URLs."""

    normalized_hand = hand.strip().lower()
    if normalized_hand not in {"left", "right"}:
        raise ValueError("hand must be left or right")
    return camera_snapshot_url(normalized_hand)
