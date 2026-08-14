from __future__ import annotations

import asyncio
import json
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


def test_place_visual_reads_place_locate_and_pose_logs(tmp_path: Path) -> None:
    locate_dir = tmp_path / "interfaces" / "perception_place_locate"
    pose_dir = tmp_path / "interfaces" / "manipulation_place_pose"
    locate_dir.mkdir(parents=True)
    pose_dir.mkdir(parents=True)
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

    visual = web_app._operation_visual(tmp_path)

    assert visual["available"] is True
    assert visual["bbox"] == [100, 200, 300, 400]
    assert visual["mask_data"] == "data:image/png;base64,cGxhY2UtbWFzaw=="
    assert visual["pose"] == [1, 2, 3, 4, 5, 6]
    assert visual["frame"] == "camera"


@pytest.mark.asyncio
async def test_place_routes_reject_unknown_task() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=web_app.app), base_url="http://web"
    ) as client:
        events = await client.get("/api/place/missing/events")
        visual = await client.get("/api/place/missing/visual")

    assert events.status_code == 404
    assert visual.status_code == 404
