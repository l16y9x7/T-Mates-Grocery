from __future__ import annotations

import asyncio

import httpx
import pytest

from mock_external_service.app import create_app
from mock_external_service.service import MockExternalService
from mock_external_service.settings import MockExternalSettings


async def _wait_for(callbacks: list[dict], count: int) -> None:
    for _ in range(100):
        if len(callbacks) >= count:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"expected {count} callbacks, got {len(callbacks)}")


def _settings() -> MockExternalSettings:
    return MockExternalSettings(
        access_token="external-secret",
        callback_url="https://callback.example/default",
        callback_access_token="callback-secret",
        stage_delay_seconds=0,
        retry_backoff_seconds=0,
        inspection_points=["H1_INSPECT"],
    )


@pytest.mark.asyncio
async def test_mock_task0_returns_document_response_and_full_callback_flow() -> None:
    callbacks: list[dict] = []

    async def sender(_: str, payload: dict, headers: dict[str, str]) -> None:
        assert headers["Authorization"] == "Bearer callback-secret"
        assert headers["X-Event-Id"] == payload["event_id"]
        assert headers["X-Task-Run-Id"] == payload["task_run_id"]
        callbacks.append(payload)

    settings = _settings()
    service = MockExternalService(settings, callback_sender=sender)
    app = create_app(settings, service=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mock.local") as client:
        health = await client.get(
            "/api/external/v1/health",
            headers={"Authorization": "Bearer external-secret", "X-Request-Id": "health-1"},
        )
        response = await client.post(
            "/api/external/v1/tasks/0/runs",
            headers={
                "Authorization": "Bearer external-secret",
                "Idempotency-Key": "task0-key",
                "X-Request-Id": "request-1",
            },
            json={
                "external_task_id": "PREP-1",
                "status_callback_url": "https://callback.example/status",
            },
        )
        await _wait_for(callbacks, 10)
        status = await client.get(
            f"/api/external/v1/tasks/{response.json()['task_run_id']}/status",
            headers={"Authorization": "Bearer external-secret"},
        )
    await service.close()

    assert health.status_code == 200
    assert health.json()["schema_version"] == "1.0"
    assert health.json()["request_id"] == "health-1"
    assert response.status_code == 202
    assert response.json()["task_type"] == "TASK0_INVENTORY"
    assert response.json()["task_name"] == "理货"
    assert status.status_code == 200
    assert status.json() == callbacks[-1]
    assert [payload["current_step"]["code"] for payload in callbacks] == [
        "ACCEPTED",
        "HEALTH_CHECKING",
        "NAVIGATING_TO_START",
        "INSPECTING",
        "CAPTURING",
        "CAPTURING",
        "CAPTURING",
        "CAPTURING",
        "RETURNING_TO_START",
        "SUCCEEDED",
    ]
    assert callbacks[-1]["event_type"] == "TASK_COMPLETED"
    assert callbacks[-1]["summary"]["captures_completed"] == 2


@pytest.mark.asyncio
async def test_mock_task1_has_complete_item_progress_and_idempotency() -> None:
    callbacks: list[dict] = []

    async def sender(_: str, payload: dict, __: dict[str, str]) -> None:
        callbacks.append(payload)

    settings = _settings()
    service = MockExternalService(settings, callback_sender=sender)
    app = create_app(settings, service=service)
    payload = {
        "external_task_id": "ORD-1",
        "external_order_id": "ORD-1",
        "items": [{"sku_id": "SKU_001"}, {"sku_id": "SKU_002"}],
        "status_callback_url": "https://callback.example/status",
    }
    headers = {
        "Authorization": "Bearer external-secret",
        "Idempotency-Key": "order-key",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mock.local") as client:
        first = await client.post("/api/external/v1/task1/orders", headers=headers, json=payload)
        duplicate = await client.post("/api/external/v1/task1/orders", headers=headers, json=payload)
        await _wait_for(callbacks, 16)
    await service.close()

    assert first.status_code == 202
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["task_run_id"] == first.json()["task_run_id"]
    codes = [payload["current_step"]["code"] for payload in callbacks]
    assert codes[0:5] == [
        "ACCEPTED",
        "HEALTH_CHECKING",
        "ORDER_CONFIRMED",
        "RESOLVING_PRODUCTS",
        "PLANNING",
    ]
    assert "NAVIGATING_TO_SHELF" in codes
    assert "PICKING" in codes
    assert "NAVIGATING_TO_DELIVERY" in codes
    assert "PLACING" in codes
    assert codes[-2:] == ["FINISHING", "SUCCEEDED"]
    assert callbacks[-1]["items"][-1]["placed"] is True
    assert callbacks[-1]["summary"]["items_completed"] == 2
    progress = [payload["current_step"]["progress_percent"] for payload in callbacks]
    assert progress == sorted(progress)


@pytest.mark.asyncio
async def test_mock_task2_requires_task0_and_reports_replenishment_items() -> None:
    callbacks: list[dict] = []

    async def sender(_: str, payload: dict, __: dict[str, str]) -> None:
        callbacks.append(payload)

    settings = _settings()
    service = MockExternalService(settings, callback_sender=sender)
    app = create_app(settings, service=service)
    headers = {
        "Authorization": "Bearer external-secret",
        "Idempotency-Key": "task2-key",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mock.local") as client:
        before_task0 = await client.post(
            "/api/external/v1/tasks/2/runs",
            headers=headers,
            json={"external_task_id": "REPLENISH-BEFORE"},
        )
        task0 = await client.post(
            "/api/external/v1/tasks/0/runs",
            headers={**headers, "Idempotency-Key": "task0-key"},
            json={"external_task_id": "PREP-FOR-TASK2"},
        )
        await _wait_for(callbacks, 10)
        task2 = await client.post(
            "/api/external/v1/tasks/2/runs",
            headers=headers,
            json={"external_task_id": "REPLENISH-AFTER"},
        )
        await _wait_for(callbacks, 22)
        status = await client.get(
            f"/api/external/v1/tasks/{task2.json()['task_run_id']}/status",
            headers={"Authorization": "Bearer external-secret"},
        )
    await service.close()

    assert before_task0.status_code == 503
    assert task0.status_code == 202
    assert task2.status_code == 202
    assert status.json() == callbacks[-1]
    task2_callbacks = [payload for payload in callbacks if payload["task_run_id"] == task2.json()["task_run_id"]]
    assert task2_callbacks[0]["items"] == []
    assert any(payload["current_step"]["code"] == "IDENTIFYING_SHORTAGE" for payload in task2_callbacks)
    assert task2_callbacks[-1]["current_step"]["code"] == "SUCCEEDED"
    assert task2_callbacks[-1]["summary"]["replenishment_items_placed"] == 2
    progress = [payload["current_step"]["progress_percent"] for payload in task2_callbacks]
    assert progress == sorted(progress)


@pytest.mark.asyncio
async def test_mock_rejects_invalid_auth_and_callback_url() -> None:
    settings = _settings()
    service = MockExternalService(settings)
    app = create_app(settings, service=service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mock.local") as client:
        unauthorized = await client.get("/api/external/v1/health")
        invalid_callback = await client.post(
            "/api/external/v1/tasks/0/runs",
            headers={"Authorization": "Bearer external-secret", "Idempotency-Key": "invalid-callback"},
            json={"external_task_id": "INVALID", "status_callback_url": "ftp://callback.example/status"},
        )
    await service.close()

    assert unauthorized.status_code == 401
    assert invalid_callback.status_code == 422
    assert invalid_callback.json()["error_code"] == "INVALID_CALLBACK_URL"
