from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from fastapi.responses import JSONResponse

import web.app as web_app


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class IdleCoordinator:
    active_task_id = None


@pytest.mark.asyncio
async def test_config_exposes_navigation_targets_from_active_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace(
        tasks=SimpleNamespace(
            task0=SimpleNamespace(
                start_target_id="configured_start",
                inspection_points=["POINT_A", "POINT_B"],
            ),
            task1=SimpleNamespace(
                task_boundary="configured_boundary",
                start_target_id="configured_start",
                receipt_viewpoint="configured_receipt",
                delivery_place="configured_delivery",
            ),
            task2=SimpleNamespace(
                task_boundary="configured_boundary",
                start_target_id="configured_start",
                replenishment_pickup="configured_replenishment",
                inspection_points=["POINT_B", "POINT_C"],
            ),
            task3=SimpleNamespace(
                task_boundary="configured_boundary",
                start_target_id="configured_start",
                inspection_points=[
                    SimpleNamespace(target_id="POINT_C"),
                    SimpleNamespace(target_id="POINT_D"),
                ],
            ),
        )
    )
    monkeypatch.setattr(web_app, "RUNTIME_SETTINGS", runtime)
    monkeypatch.setattr(web_app, "_restart_preflight", lambda: (True, None))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=web_app.app), base_url="http://web"
    ) as client:
        response = await client.get("/api/config")

    assert response.status_code == 200
    navigation_targets = response.json()["navigation_targets"]
    assert navigation_targets["task_points"] == [
        {"target_id": "configured_boundary", "label": "任务判定点"},
        {"target_id": "configured_start", "label": "起点"},
        {"target_id": "configured_receipt", "label": "小票识别点"},
        {"target_id": "configured_delivery", "label": "交付台放货点"},
        {"target_id": "configured_replenishment", "label": "补货台取货点"},
    ]
    assert navigation_targets["inspection_points"] == [
        {"target_id": "POINT_A", "label": "1号巡检点"},
        {"target_id": "POINT_B", "label": "2号巡检点"},
        {"target_id": "POINT_C", "label": "3号巡检点"},
        {"target_id": "POINT_D", "label": "4号巡检点"},
    ]


def test_navigation_target_options_are_not_hardcoded_in_html() -> None:
    html = (
        Path(__file__).resolve().parents[1] / "web/static/index.html"
    ).read_text(encoding="utf-8")
    javascript = (
        Path(__file__).resolve().parents[1] / "web/static/app.js"
    ).read_text(encoding="utf-8")

    assert 'id="taskNavigationTargets"' in html
    assert 'id="inspectionNavigationTargets"' in html
    assert '<option value="task_boundary">' not in html
    assert '<option value="H1_INSPECT">' not in html
    assert '<option value="H1_F_L_INSPECT">' not in html
    assert "applyNavigationTargets(config.navigation_targets)" in javascript


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


def test_restart_preflight_does_not_require_running_pid_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    for name in ("restart-runtime.sh", "tasks.sh", "pick-place.sh"):
        script = scripts_dir / name
        script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        script.chmod(0o755)

    monkeypatch.setattr(web_app, "PROJECT_ROOT", tmp_path)

    assert web_app._restart_preflight() == (True, None)


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
