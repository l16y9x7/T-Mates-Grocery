from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import httpx
import pytest

import web.app as web_app


@pytest.mark.asyncio
async def test_place_start_uses_place_service_and_exposes_stream(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[dict[str, object], str]] = []

    async def fake_run(
        state: web_app.PickTask,
        request: web_app.PickRequest,
        target_url: str,
    ) -> None:
        calls.append((request.model_dump(mode="json"), target_url))
        state.result = {"status_code": 200, "ok": True, "body": {"status": "SUCCEEDED"}}
        state.finished = True

    monkeypatch.setattr(web_app, "_run_pick_place", fake_run)
    monkeypatch.setattr(web_app, "LOG_ROOT", tmp_path)

    payload = {"task_type": "SORTING", "product_name": "可口可乐罐装", "hand": "left"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=web_app.app), base_url="http://web"
    ) as client:
        started = await client.post("/api/place/start", json=payload)
        task_id = started.json()["task_id"]
        await asyncio.sleep(0)
        events = await client.get(f"/api/place/{task_id}/events")

    assert started.status_code == 200
    assert started.json()["operation_key"].startswith("web-place-")
    assert started.json()["events_url"] == f"/api/place/{task_id}/events"
    assert calls == [(payload, web_app.SERVICES.place_url)]
    assert events.status_code == 200
    assert "event: result" in events.text
    assert '"status": "SUCCEEDED"' in events.text
    web_app.TASKS.pop(task_id, None)


def test_place_visual_uses_current_image_and_transformed_pose(tmp_path: Path) -> None:
    locate_dir = tmp_path / "interfaces" / "perception_place_locate"
    pose_dir = tmp_path / "interfaces" / "manipulation_place_pose"
    current_dir = tmp_path / "current"
    locate_dir.mkdir(parents=True)
    pose_dir.mkdir(parents=True)
    current_dir.mkdir()
    (current_dir / "rgb.jpg").write_bytes(b"current-rgb")
    (current_dir / "head.json").write_text(
        json.dumps({"cam_K": [100, 0, 10, 0, 200, 20, 0, 0, 1]}),
        encoding="utf-8",
    )
    (locate_dir / "response.json").write_text(
        json.dumps(
            {
                "status_code": 200,
                "body": {
                    "product_name": "可口可乐罐装",
                    "bbox": [100, 200, 300, 400],
                    "mask": "cGxhY2UtbWFzaw==",
                },
            }
        ),
        encoding="utf-8",
    )
    (pose_dir / "response.json").write_text(
        json.dumps(
            {
                "status_code": 200,
                "body": {
                    "pose": [1, 2, 3, 4, 5, 6],
                    "frame": "camera",
                    "pose_unit": "mm_rad",
                    "rotation_order": "zyx",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "位姿转换",
                "status": "succeeded",
                "reference_pose": [1, 2, 3, 4, 5, 6],
                "current_pose": [0, 0, 1000, 0, 0, 0],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    visual = web_app._operation_visual(tmp_path)

    assert visual["available"] is True
    assert visual["bbox"] is None
    assert visual["mask_data"] is None
    assert visual["pose"] == [0, 0, 1000, 0, 0, 0]
    assert visual["frame"] == "camera"
    assert visual["image_data"] == "data:image/jpeg;base64,Y3VycmVudC1yZ2I="
    assert visual["pose_axes"] == {
        "origin": [10.0, 20.0],
        "axis_length_mm": 100.0,
        "x": [20.0, 20.0],
        "y": [10.0, 40.0],
        "z": [10.0, 20.0],
    }
    assert tmp_path.name in str(visual["visual_revision"])


def test_project_pose_axes_supports_zyx_rotation_and_rejects_behind_camera() -> None:
    cam_k = [100, 0, 10, 0, 100, 20, 0, 0, 1]

    rotated = web_app._project_pose_axes(
        [0, 0, 1000, 0, 0, math.pi / 2], cam_k
    )

    assert rotated is not None
    assert rotated["origin"] == pytest.approx([10, 20])
    assert rotated["x"] == pytest.approx([10, 30])
    assert rotated["y"] == pytest.approx([0, 20])
    assert web_app._project_pose_axes([0, 0, -1, 0, 0, 0], cam_k) is None
    assert web_app._project_pose_axes([0, 0, float("nan"), 0, 0, 0], cam_k) is None


def test_task_visual_log_selection_includes_place_and_recovery_retry(
    monkeypatch, tmp_path: Path
) -> None:
    operation_key = "web-task2-run"

    def add_log(name: str, interface: str | None) -> Path:
        log_dir = tmp_path / name
        log_dir.mkdir()
        if interface:
            response = log_dir / "interfaces" / interface / "response.json"
            response.parent.mkdir(parents=True)
            response.write_text(
                json.dumps({"status_code": 200, "body": {"bbox": [0, 0, 1, 1]}}),
                encoding="utf-8",
            )
        return log_dir

    add_log("20260101-000000-000001-web-task2-run_task2.pick.0", "perception_pick_locate")
    expected = add_log(
        "20260101-000000-000002-web-task2-run_task2.place.0_recovery.retry",
        "perception_place_locate",
    )
    add_log("20260101-000000-000003-web-task2-run_task2.place.1", None)
    monkeypatch.setattr(web_app, "LOG_ROOT", tmp_path)

    assert web_app._find_task_visual_log_dir(operation_key, "2") == expected


def test_visual_revision_changes_between_identical_locate_calls(tmp_path: Path) -> None:
    visuals = []
    for suffix in ("first", "second"):
        log_dir = tmp_path / suffix
        response = log_dir / "interfaces" / "perception_pick_locate" / "response.json"
        response.parent.mkdir(parents=True)
        response.write_text(
            json.dumps({"status_code": 200, "body": {"bbox": [1, 2, 3, 4]}}),
            encoding="utf-8",
        )
        visuals.append(web_app._operation_visual(log_dir))

    assert visuals[0]["visual_revision"] != visuals[1]["visual_revision"]


@pytest.mark.asyncio
async def test_place_routes_reject_unknown_task() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=web_app.app), base_url="http://web"
    ) as client:
        events = await client.get("/api/place/missing/events")
        visual = await client.get("/api/place/missing/visual")

    assert events.status_code == 404
    assert visual.status_code == 404
