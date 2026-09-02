from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as cocomask

from ui.service_client import ServiceCallError, ServiceResult
from ui.service_frontend import (
    BoxState,
    PipelineState,
    _optional_float,
    advance_box_click,
    clear_box,
    run_full_pipeline,
    run_pose_stage,
    run_sam3_stage,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    camera = tmp_path / "camera.json"
    Image.new("RGB", (8, 6), color=(120, 80, 40)).save(rgb)
    Image.fromarray(np.full((6, 8), 900, dtype=np.uint16)).save(depth)
    camera.write_text(
        json.dumps(
            {
                "cam_K": [100, 0, 4, 0, 100, 3, 0, 0, 1],
                "depth_scale": 0.001,
            }
        ),
        encoding="utf-8",
    )
    return rgb, depth, camera


def _sam3_payload() -> dict[str, object]:
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1:5, 2:7] = 1
    encoded = cocomask.encode(np.asfortranarray(mask))
    return {
        "ok": True,
        "box": [2, 1, 7, 5],
        "detections": [
            {
                "score": 0.93,
                "segmentation": {
                    "size": encoded["size"],
                    "counts": encoded["counts"].decode("ascii"),
                },
            }
        ],
    }


def _pose_payload() -> dict[str, object]:
    return {
        "pose": [0, 0, 900, 0, 0, 0],
        "corners_mm": [
            [-20, -20, 850],
            [20, -20, 850],
            [20, 20, 850],
            [-20, 20, 850],
            [-20, -20, 950],
            [20, -20, 950],
            [20, 20, 950],
            [-20, 20, 950],
        ],
        "frame": "camera",
        "pose_unit": "mm_rad",
        "rotation_order": "zyx",
    }


def test_two_clicks_complete_box() -> None:
    first = advance_box_click(None, (100, 80), (640, 480))
    assert first.status == "awaiting_second"
    second = advance_box_click(first, (400, 300), (640, 480))
    assert second.box == [100, 80, 400, 300]
    assert second.status == "complete"


def test_third_click_starts_new_box() -> None:
    complete = BoxState(
        first=(1, 2), box=[1, 2, 10, 20], status="complete"
    )
    state = advance_box_click(complete, (5, 6), (32, 24))
    assert state.first == (5, 6)
    assert state.box is None
    assert state.status == "awaiting_second"


def test_clear_box_returns_empty_state() -> None:
    assert clear_box() == BoxState()


def test_optional_float_accepts_blank_manual_camera_fields() -> None:
    assert _optional_float(None) is None
    assert _optional_float("") is None
    assert _optional_float("  ") is None
    assert _optional_float("600.5") == 600.5


def test_sam3_stage_writes_mask_and_timings(tmp_path: Path) -> None:
    rgb, _, _ = _write_inputs(tmp_path)

    def fake_sam3(*args, **kwargs) -> ServiceResult:
        return ServiceResult(payload=_sam3_payload(), elapsed_ms=12.5)

    state = run_sam3_stage(
        rgb,
        [2, 1, 7, 5],
        "http://sam3/infer",
        10,
        tmp_path / "runs",
        sam3_caller=fake_sam3,
    )

    assert state.error is None
    assert Path(state.mask_path).is_file()
    assert state.sam3_payload["ok"] is True
    assert state.timings_ms["sam3_http_ms"] == 12.5
    assert Path(state.artifacts["mask_overlay"]).is_file()


def test_pose_failure_preserves_successful_sam3_stage(tmp_path: Path) -> None:
    rgb, depth, camera = _write_inputs(tmp_path)

    def fake_sam3(*args, **kwargs) -> ServiceResult:
        return ServiceResult(payload=_sam3_payload(), elapsed_ms=10.0)

    def failing_pose(*args, **kwargs) -> ServiceResult:
        raise ServiceCallError(
            503,
            {"error": "pose service offline"},
            elapsed_ms=4.5,
        )

    sam_state = run_sam3_stage(
        rgb,
        [2, 1, 7, 5],
        "http://sam3/infer",
        10,
        tmp_path / "runs",
        sam3_caller=fake_sam3,
    )
    pose_state = run_pose_stage(
        sam_state,
        depth,
        camera,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        depth_scale=None,
        pose_url="http://pose/manipulation/pick_pose",
        timeout_s=10,
        pose_caller=failing_pose,
    )

    assert pose_state.mask_path == sam_state.mask_path
    assert pose_state.sam3_payload == sam_state.sam3_payload
    assert pose_state.timings_ms["sam3_http_ms"] == 10.0
    assert pose_state.timings_ms["pose_http_ms"] == 4.5
    assert "pose service offline" in pose_state.error


def test_full_pipeline_exports_pose_artifacts(tmp_path: Path) -> None:
    rgb, depth, camera = _write_inputs(tmp_path)

    def fake_sam3(*args, **kwargs) -> ServiceResult:
        return ServiceResult(payload=_sam3_payload(), elapsed_ms=8.0)

    def fake_pose(*args, **kwargs) -> ServiceResult:
        return ServiceResult(payload=_pose_payload(), elapsed_ms=20.0)

    state = run_full_pipeline(
        rgb,
        depth,
        camera,
        [2, 1, 7, 5],
        sam3_url="http://sam3/infer",
        pose_url="http://pose/manipulation/pick_pose",
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        depth_scale=None,
        sam3_timeout_s=10,
        pose_timeout_s=10,
        output_root=tmp_path / "runs",
        sam3_caller=fake_sam3,
        pose_caller=fake_pose,
    )

    assert state.error is None
    assert state.pose_payload["pose_unit"] == "mm_rad"
    assert Path(state.artifacts["pose_overlay"]).is_file()
    assert Path(state.artifacts["scene_glb"]).is_file()
    assert Path(state.artifacts["scene_ply"]).is_file()
    assert state.timings_ms["total_ms"] >= 28.0


def test_frontend_build_process_exits_without_analytics_hang() -> None:
    command = [
        sys.executable,
        "-c",
        "from ui.service_frontend import build_service_frontend; "
        "app = build_service_frontend(); assert len(app.blocks) > 0",
    ]
    subprocess.run(command, check=True, timeout=8)


def test_entrypoint_import_does_not_import_torch() -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; import run_service_frontend; "
        "assert 'torch' not in sys.modules",
    ]
    subprocess.run(command, check=True, timeout=5)
