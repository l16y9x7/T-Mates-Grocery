from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as cocomask

from ui.service_client import (
    ServiceCallError,
    call_pose_service,
    call_sam3_box,
    decode_coco_rle,
    normalize_box,
    resolve_camera,
    select_best_detection,
    validate_pose_response,
    write_camera_json,
    write_mask_png,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def json(self) -> dict[str, Any]:
        return self._payload


class RecordingSession:
    def __init__(
        self,
        response: FakeResponse,
        health_response: FakeResponse | None = None,
    ) -> None:
        self.response = response
        self.health_response = health_response or FakeResponse(
            {"ok": True, "service": "sam3-infer"}
        )
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        recorded = {"url": url, **kwargs}
        if "files" in recorded:
            recorded["file_keys"] = list(recorded["files"].keys())
        self.calls.append(recorded)
        return self.response

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, "method": "GET", **kwargs})
        return self.health_response

    def close(self) -> None:
        self.closed = True


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    camera = tmp_path / "camera.json"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(rgb)
    Image.fromarray(np.full((6, 8), 900, dtype=np.uint16)).save(depth)
    camera.write_text(
        json.dumps({"cam_K": [100, 0, 4, 0, 100, 3, 0, 0, 1], "depth_scale": 0.001}),
        encoding="utf-8",
    )
    Image.fromarray(np.full((6, 8), 255, dtype=np.uint8)).save(mask)
    return rgb, depth, camera, mask


def test_normalize_box_orders_and_clamps() -> None:
    assert normalize_box([40, 30, -5, 10], (32, 24)) == [0, 10, 31, 23]


def test_manual_camera_overrides_uploaded_camera(tmp_path: Path) -> None:
    uploaded = tmp_path / "camera.json"
    uploaded.write_text(
        json.dumps({"cam_K": [1, 0, 2, 0, 1, 2, 0, 0, 1], "depth_scale": 1.0}),
        encoding="utf-8",
    )
    camera = resolve_camera(
        uploaded, fx=600, fy=601, cx=320, cy=240, depth_scale=0.001
    )
    assert camera.camera_json["cam_K"] == [
        600.0, 0.0, 320.0, 0.0, 601.0, 240.0, 0.0, 0.0, 1.0
    ]
    assert camera.depth_scale == 0.001


def test_partial_manual_camera_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="all five"):
        resolve_camera(
            None, fx=600, fy=None, cx=320, cy=240, depth_scale=0.001
        )


def test_nested_camera_intrinsics_are_supported(tmp_path: Path) -> None:
    uploaded = tmp_path / "camera.json"
    uploaded.write_text(
        json.dumps(
            {
                "camera": {
                    "intrinsics": {"fx": 600, "fy": 601, "cx": 320, "cy": 240}
                },
                "depth_scale": 0.001,
            }
        ),
        encoding="utf-8",
    )
    camera = resolve_camera(
        uploaded, fx=None, fy=None, cx=None, cy=None, depth_scale=None
    )
    assert camera.intrinsics.tolist() == [
        [600.0, 0.0, 320.0], [0.0, 601.0, 240.0], [0.0, 0.0, 1.0]
    ]


def test_validate_pose_response_requires_eight_corners() -> None:
    with pytest.raises(ValueError, match="corners_mm"):
        validate_pose_response(
            {"pose": [0] * 6, "corners_mm": [], "frame": "camera"}
        )


def test_coco_rle_roundtrip_and_artifact_writers(tmp_path: Path) -> None:
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:4, 2:6] = 1
    encoded = cocomask.encode(np.asfortranarray(mask))
    rle = {
        "size": encoded["size"],
        "counts": encoded["counts"].decode("ascii"),
    }
    decoded = decode_coco_rle(rle)
    assert np.array_equal(decoded, mask.astype(bool))

    mask_path = write_mask_png(decoded, tmp_path / "mask.png")
    camera = resolve_camera(
        None, fx=600, fy=601, cx=320, cy=240, depth_scale=0.001
    )
    camera_path = write_camera_json(camera, tmp_path / "camera.json")
    assert Image.open(mask_path).mode == "L"
    assert json.loads(camera_path.read_text(encoding="utf-8"))["depth_scale"] == 0.001


def test_sam3_request_contains_true_box(tmp_path: Path) -> None:
    rgb, _, _, _ = _write_inputs(tmp_path)
    session = RecordingSession(
        FakeResponse({"ok": True, "detections": [{"score": 0.9}]})
    )

    result = call_sam3_box(
        "http://127.0.0.1:18003/infer",
        rgb,
        [1, 2, 6, 5],
        timeout_s=10,
        session=session,
    )

    request_json = next(call["json"] for call in session.calls if "json" in call)
    assert request_json["box"] == [1, 2, 6, 5]
    assert request_json["save_vis"] is False
    assert result.payload["ok"] is True


def test_pose_request_uses_exact_documented_file_keys(tmp_path: Path) -> None:
    rgb, depth, camera, mask = _write_inputs(tmp_path)
    response = {
        "pose": [0, 0, 900, 0, 0, 0],
        "corners_mm": [[0, 0, 900]] * 8,
        "frame": "camera",
        "pose_unit": "mm_rad",
        "rotation_order": "zyx",
    }
    session = RecordingSession(FakeResponse(response))

    result = call_pose_service(
        "http://127.0.0.1:8084/manipulation/pick_pose",
        rgb,
        depth,
        camera,
        mask,
        timeout_s=120,
        session=session,
    )

    assert session.calls[0]["file_keys"] == ["rgb", "depth", "camera", "mask"]
    assert result.payload["pose_unit"] == "mm_rad"


def test_service_error_preserves_backend_body(tmp_path: Path) -> None:
    rgb, _, _, _ = _write_inputs(tmp_path)
    session = RecordingSession(
        FakeResponse({"ok": False, "error": "service unavailable"}, status_code=503)
    )
    with pytest.raises(ServiceCallError) as error:
        call_sam3_box(
            "http://127.0.0.1:18003/infer",
            rgb,
            [1, 2, 6, 5],
            timeout_s=10,
            session=session,
        )
    assert error.value.status_code == 503
    assert error.value.payload["error"] == "service unavailable"


def test_select_best_detection_prefers_highest_score() -> None:
    selected = select_best_detection(
        {
            "detections": [
                {"score": 0.2, "id": 1, "segmentation": {}},
                {"score": 0.9, "id": 2, "segmentation": {}},
            ]
        }
    )
    assert selected["id"] == 2


def test_trt_service_uses_roi_crop_and_remaps_mask(tmp_path: Path) -> None:
    rgb, _, _, _ = _write_inputs(tmp_path)
    crop_mask = np.zeros((5, 6), dtype=np.uint8)
    crop_mask[1:4, 2:5] = 1
    encoded = cocomask.encode(np.asfortranarray(crop_mask))
    response = FakeResponse(
        {
            "ok": True,
            "detections": [
                {
                    "score": 0.9,
                    "bbox": [2, 1, 3, 3],
                    "segmentation": {
                        "size": encoded["size"],
                        "counts": encoded["counts"].decode("ascii"),
                    },
                }
            ],
        }
    )
    session = RecordingSession(
        response,
        health_response=FakeResponse(
            {"ok": True, "service": "sam3-trt-infer", "backend": "tensorrt"}
        ),
    )

    result = call_sam3_box(
        "http://127.0.0.1:18003/infer",
        rgb,
        [1, 1, 6, 5],
        timeout_s=10,
        session=session,
    )

    request_json = next(call["json"] for call in session.calls if "json" in call)
    sent_image = Image.open(
        BytesIO(base64.b64decode(request_json["image_base64"]))
    )
    assert sent_image.size == (6, 5)
    assert "box" not in request_json
    assert request_json["prompt"] == "object"
    assert result.payload["box"] == [1, 1, 6, 5]
    assert result.payload["box_mode"] == "roi_crop_compat"
    full_mask = decode_coco_rle(
        result.payload["detections"][0]["segmentation"]
    )
    assert full_mask.shape == (6, 8)
    assert result.payload["detections"][0]["bbox"] == [3, 2, 3, 3]
