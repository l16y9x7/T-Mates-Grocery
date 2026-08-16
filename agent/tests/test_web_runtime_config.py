from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.responses import JSONResponse

import web.app as web_app


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class IdleCoordinator:
    active_task_id = None


@pytest.mark.asyncio
async def test_gripper_open_proxies_uppercase_hand_and_idempotency_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str, dict[str, object], str, str | None]] = []

    async def fake_robot_request(
        method: str,
        base_url: str,
        path: str,
        payload: dict[str, object] | None,
        operation: str,
        idempotency_key: str | None,
        timeout: float | None = None,
    ) -> JSONResponse:
        calls.append((method, base_url, path, payload or {}, operation, idempotency_key))
        return JSONResponse(status_code=200, content={"ok": True})

    monkeypatch.setattr(web_app, "_robot_request", fake_robot_request)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=web_app.app), base_url="http://web"
    ) as client:
        response = await client.post(
            "/api/robot/gripper/open",
            json={"hand": "LEFT"},
            headers={"Idempotency-Key": "open-left-test"},
        )

    assert response.status_code == 200
    assert calls == [
        (
            "POST",
            web_app.SERVICES.pose_url,
            "/manipulation/gripper/open",
            {"hand": "LEFT"},
            "gripper_open",
            "open-left-test",
        )
    ]


@pytest.mark.asyncio
async def test_gripper_open_rejects_invalid_hand() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=web_app.app), base_url="http://web"
    ) as client:
        response = await client.post("/api/robot/gripper/open", json={"hand": "left"})

    assert response.status_code == 422


def _runtime_copy(tmp_path: Path) -> Path:
    config_path = tmp_path / "runtime.production.yaml"
    shutil.copy2(CONFIG_DIR / "runtime.production.yaml", config_path)
    shutil.copy2(CONFIG_DIR / "product-hand-options.yaml", tmp_path)
    return config_path


def test_robot_ip_update_is_validated_and_written_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _runtime_copy(tmp_path)
    monkeypatch.setattr(web_app, "RUNTIME_CONFIG_PATH", config_path)

    backup_path = web_app._write_robot_ip("10.0.0.28")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    backup = yaml.safe_load(backup_path.read_text(encoding="utf-8"))
    assert raw["robot"]["ip"] == "10.0.0.28"
    assert backup["robot"]["ip"] != "10.0.0.28"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.asyncio
async def test_robot_ip_update_requires_force_for_active_operations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scheduled: list[bool] = []
    backup_path = tmp_path / "runtime.production.yaml.bak"
    backup_path.write_text("backup", encoding="utf-8")

    monkeypatch.setattr(web_app, "COORDINATOR_GETTER", lambda: IdleCoordinator())
    monkeypatch.setattr(web_app, "_restart_preflight", lambda: (True, None))

    async def active_operations() -> dict[str, object]:
        return {
            "task_id": "1",
            "web_operations": [{"workflow": "task1", "operation_key": "run-1"}],
            "pick_place_active_operations": 1,
            "pick_place_status_unknown": False,
        }

    monkeypatch.setattr(web_app, "_active_operations", active_operations)
    monkeypatch.setattr(web_app, "_write_robot_ip", lambda _: backup_path)
    monkeypatch.setattr(
        web_app, "_schedule_runtime_restart", lambda: scheduled.append(True)
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=web_app.app), base_url="http://web"
    ) as client:
        blocked = await client.put(
            "/api/system/robot-ip",
            json={"robot_ip": "10.0.0.29", "force_restart": False},
        )
        forced = await client.put(
            "/api/system/robot-ip",
            json={"robot_ip": "10.0.0.29", "force_restart": True},
        )

    assert blocked.status_code == 409
    assert blocked.json()["error_code"] == "OPERATIONS_IN_PROGRESS"
    assert blocked.json()["requires_force"] is True
    assert forced.status_code == 202
    assert forced.json()["status"] == "RESTARTING"
    assert forced.json()["forced"] is True
    assert scheduled == [True]


@pytest.mark.asyncio
async def test_robot_ip_update_rejects_invalid_ip_before_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_app, "_restart_preflight", lambda: (True, None))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=web_app.app), base_url="http://web"
    ) as client:
        response = await client.put(
            "/api/system/robot-ip",
            json={"robot_ip": "not-an-ip", "force_restart": True},
        )

    assert response.status_code == 422
