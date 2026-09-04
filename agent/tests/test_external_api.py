from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from task_service.app import create_app
from task_service.settings import TaskServiceSettings

from tests.test_task_service import FakeOrchestrator, app_for, orchestrators


CONFIG = Path(__file__).resolve().parents[1] / "config" / "runtime.production.yaml"


@pytest.mark.asyncio
async def test_external_task1_is_accepted_and_converted_to_internal_order() -> None:
    bindings = orchestrators()
    app = app_for(bindings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://tasks.local"
    ) as client:
        response = await client.post(
            "/api/external/v1/task1/orders",
            headers={
                "Idempotency-Key": "order-key",
                "X-Request-Id": "request-1",
            },
            json={
                "external_task_id": "order-1",
                "external_order_id": "order-1",
                "items": [
                    {"sku_id": "SKU_001"},
                    {"sku_id": "SKU_002"},
                ],
            },
        )
        await asyncio.sleep(0)

    assert response.status_code == 202
    assert response.json()["task_type"] == "TASK1_PICKUP"
    assert response.json()["status"] == "ACCEPTED"
    assert bindings["1"].last_request.order_id == "order-1"
    assert bindings["1"].last_request.product_names == ["NFC桔汁", "蒙牛纯牛奶"]


@pytest.mark.asyncio
async def test_external_requires_idempotency_key_and_accepts_callback_url() -> None:
    app = app_for()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://tasks.local"
    ) as client:
        missing_key = await client.post(
            "/api/external/v1/tasks/0/runs", json={"external_task_id": "prep-1"}
        )
        callback_request = await client.post(
            "/api/external/v1/tasks/0/runs",
            headers={"Idempotency-Key": "prep-key"},
            json={
                "external_task_id": "prep-2",
                "status_callback_url": "https://callback.example.test/status",
            },
        )

    assert missing_key.status_code == 400
    assert callback_request.status_code == 202


@pytest.mark.asyncio
async def test_external_order_uses_sku_id_and_rejects_item_id() -> None:
    app = app_for()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://tasks.local"
    ) as client:
        response = await client.post(
            "/api/external/v1/task1/orders",
            headers={"Idempotency-Key": "order-key"},
            json={
                "external_task_id": "order-1",
                "external_order_id": "order-1",
                "items": [
                    {"item_id": "item-1", "sku_id": "SKU_001"},
                    {"item_id": "item-2", "sku_id": "SKU_002"},
                ],
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_external_idempotency_does_not_start_task_twice() -> None:
    bindings = orchestrators()
    app = app_for(bindings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://tasks.local"
    ) as client:
        payload = {"external_task_id": "prep-1"}
        first = await client.post(
            "/api/external/v1/tasks/0/runs",
            headers={"Idempotency-Key": "prep-key"},
            json=payload,
        )
        second = await client.post(
            "/api/external/v1/tasks/0/runs",
            headers={"Idempotency-Key": "prep-key"},
            json=payload,
        )

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert first.json()["task_run_id"] == second.json()["task_run_id"]


@pytest.mark.asyncio
async def test_external_rejects_busy_and_exposes_final_status() -> None:
    blocking = FakeOrchestrator("0", blocking=True)
    app = app_for(orchestrators(**{"0": blocking}))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://tasks.local"
    ) as client:
        first = await client.post(
            "/api/external/v1/tasks/0/runs",
            headers={"Idempotency-Key": "busy-key"},
            json={"external_task_id": "busy-1"},
        )
        await blocking.started.wait()
        busy = await client.post(
            "/api/external/v1/tasks/2/runs",
            headers={"Idempotency-Key": "other-key"},
            json={"external_task_id": "other-1"},
        )
        blocking.release.set()
        status = None
        for _ in range(20):
            await asyncio.sleep(0.01)
            status = await client.get(
                f"/api/external/v1/tasks/{first.json()['task_run_id']}/status"
            )
            if status.json().get("status") == "SUCCEEDED":
                break

    assert first.status_code == 202
    assert busy.status_code == 409
    assert busy.json()["error_code"] == "TASK_BUSY"
    assert status is not None
    assert status.status_code == 200
    assert status.json()["status"] == "SUCCEEDED"
    assert status.json()["task_type"] == "TASK0_INVENTORY"
    assert set(status.json()) == {
        "schema_version",
        "event_id",
        "sequence",
        "event_type",
        "occurred_at",
        "external_task_id",
        "external_order_id",
        "task_run_id",
        "task_type",
        "task_name",
        "status",
        "display_title",
        "display_message",
        "current_step",
        "location",
        "next_step",
        "estimated_remaining_seconds",
        "summary",
        "user_notice",
        "last_updated_at",
        "error",
        "captures",
    }
    assert set(status.json()["summary"]) == {
        "inspection_points_total",
        "inspection_points_completed",
        "captures_total",
        "captures_completed",
        "captures_failed",
    }
