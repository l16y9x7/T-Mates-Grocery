from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

import web.app as web_app


@pytest.mark.asyncio
async def test_task1_start_and_sse_events(monkeypatch, tmp_path: Path) -> None:
    async def fake_run(state: web_app.PickTask, request: web_app.Task1Request) -> None:
        state.result = {
            "status_code": 200,
            "ok": True,
            "body": {"status": "SUCCEEDED", "requested_pick_count": request.pick_count},
        }
        state.finished = True

    monkeypatch.setattr(web_app, "_run_task1", fake_run)
    monkeypatch.setattr(web_app, "LOG_ROOT", tmp_path)
    operation_key = "web-task1-test"
    log_dir = tmp_path / f"20260101-000000-000000-{operation_key}"
    log_dir.mkdir()
    (log_dir / "events.jsonl").write_text(
        json.dumps({"event": "小票识别", "status": "succeeded", "product_names": ["A", "B"]}) + "\n",
        encoding="utf-8",
    )

    app = web_app.app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://web") as client:
        started = await client.post("/api/task1/start", json={"pick_count": 1})
        task_id = started.json()["task_id"]
        await asyncio.sleep(0)
        events = await client.get(f"/api/task1/{task_id}/events")

    assert started.status_code == 200
    assert started.json()["events_url"] == f"/api/task1/{task_id}/events"
    assert events.status_code == 200
    assert "event: result" in events.text
    web_app.TASKS.pop(task_id, None)


@pytest.mark.asyncio
async def test_task1_event_stream_reads_pickplace_style_log(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(web_app, "LOG_ROOT", tmp_path)
    state = web_app.PickTask(task_id="task", operation_key="web-task1-log", finished=True)
    state.result = {"status_code": 200, "ok": True, "body": {"status": "SUCCEEDED"}}
    web_app.TASKS[state.task_id] = state
    log_dir = tmp_path / "20260101-000000-000000-web-task1-log"
    log_dir.mkdir()
    (log_dir / "events.jsonl").write_text(
        json.dumps({"event": "抓取", "status": "succeeded", "hand": "LEFT"}) + "\n",
        encoding="utf-8",
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web_app.app), base_url="http://web") as client:
        response = await client.get("/api/task1/task/events")

    assert response.status_code == 200
    assert "event: flow" in response.text
    assert '"event": "抓取"' in response.text
    assert "event: result" in response.text
    web_app.TASKS.pop(state.task_id, None)
