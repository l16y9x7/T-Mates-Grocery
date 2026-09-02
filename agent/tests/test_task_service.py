from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from task1_service.models import InterfaceMetric, Task1ServiceError
from task_service.app import create_app
from task_service.settings import TaskServiceSettings


CONFIG = Path(__file__).resolve().parents[1] / "config" / "runtime.production.yaml"


class ReadyClient:
    def __init__(self, ready: bool = True) -> None:
        self.is_ready = ready

    async def health_ready(self) -> bool:
        return self.is_ready


class FakeOrchestrator:
    def __init__(self, task_id: str, *, ready: bool = True, blocking: bool = False) -> None:
        self.task_id = task_id
        self.client = ReadyClient(ready)
        self.blocking = blocking
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.last_request = None

    async def ready(self) -> bool:
        return self.client.is_ready

    async def run(self, request, operation_key: str | None = None):
        assert type(request).__name__ == f"Task{self.task_id}Request"
        self.last_request = request
        if self.blocking:
            self.started.set()
            await self.release.wait()
        key = operation_key or f"generated-{self.task_id}"
        if self.task_id == "0":
            return {
                "task_run_id": key,
                "task_type": "PREPARATION",
                "status": "SUCCEEDED",
                "inspection_points": [],
                "captures": [],
            }
        if self.task_id == "1":
            return {
                "task_run_id": key,
                "task_type": "SORTING",
                "status": "SUCCEEDED",
                "product_names": [],
                "target_items": [],
                "held_items": {},
            }
        if self.task_id == "2":
            return {
                "task_run_id": key,
                "task_type": "SHORTAGE",
                "status": "SUCCEEDED",
                "inspection_pass": 1,
                "product_names": [],
                "target_items": [],
                "held_items": {},
            }
        return {
            "task_run_id": key,
            "task_type": "MISPLACED",
            "status": "SUCCEEDED",
            "inspection_pass": 1,
            "finding": {
                "misplaced_product_name": "错误商品",
                "gt_product_name": "正确商品",
                "inspection_target_id": "H1_F_L_INSPECT",
                "inspection_location_id": "H1_F_L1_C01",
                "inspection_pose_type": "SHELF_VIEW_UPPER",
            },
            "product_names": [],
            "target_items": [],
            "held_items": {},
        }

    async def create_mock_order(self):
        assert self.task_id == "1"
        return SimpleNamespace(
            order_id="preview-order",
            source="mock_random",
            catalog_size=43,
            product_names=["可口可乐罐装", "百事可乐瓶装"],
        )


class FailingTask1(FakeOrchestrator):
    async def run(self, request, operation_key: str | None = None):
        del request, operation_key
        raise Task1ServiceError(
            "EXECUTION_FAILED",
            "pick_place returned HTTP 502: /manipulation/release: MoveL 规划失败，错误码 -1022",
            step="商品放置",
            failed_interface="manipulation_release",
            url="http://robot:8084/manipulation/release",
            interface_metrics=[
                InterfaceMetric(
                    interface="pick_place/place",
                    service="pick_place",
                    method="POST",
                    url="http://pick-place.local/place",
                    call_count=2,
                    success_count=1,
                    failure_count=1,
                    total_duration_ms=125.0,
                    average_duration_ms=62.5,
                )
            ],
        )


def orchestrators(**overrides) -> dict[str, FakeOrchestrator]:
    values = {task_id: FakeOrchestrator(task_id) for task_id in ("0", "1", "2", "3")}
    values.update(overrides)
    return values


def app_for(bindings=None):
    return create_app(
        TaskServiceSettings.load(CONFIG),
        orchestrators=bindings or orchestrators(),
    )


@pytest.mark.asyncio
async def test_unified_api_dispatches_all_tasks_and_preserves_idempotency_key() -> None:
    expected_types = ["PREPARATION", "SORTING", "SHORTAGE", "MISPLACED"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for()), base_url="http://tasks.local"
    ) as client:
        health = await client.get("/health")
        responses = [
            await client.post(
                f"/tasks/{task_id}/run",
                json={},
                headers={"Idempotency-Key": f"operation-{task_id}"},
            )
            for task_id in range(4)
        ]

    assert health.status_code == 200
    assert set(health.json()["tasks"].values()) == {"READY"}
    assert [response.status_code for response in responses] == [200, 200, 200, 200]
    assert [response.json()["task_type"] for response in responses] == expected_types
    assert [response.json()["task_run_id"] for response in responses] == [
        f"operation-{task_id}" for task_id in range(4)
    ]


@pytest.mark.asyncio
async def test_unified_api_rejects_invalid_task_body_and_removed_routes() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for()), base_url="http://tasks.local"
    ) as client:
        invalid = await client.post("/tasks/2/run", json={"count": 2})
        unknown = await client.post("/tasks/9/run", json={})
        removed = await client.post("/task2/run", json={})

    assert invalid.status_code == 422
    assert unknown.status_code == 404
    assert unknown.json()["error_code"] == "TASK_NOT_FOUND"
    assert removed.status_code == 404


@pytest.mark.asyncio
async def test_unified_health_reports_each_task_and_aggregate_failure() -> None:
    bindings = orchestrators()
    bindings["2"].client.is_ready = False
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for(bindings)),
        base_url="http://tasks.local",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "ERROR",
        "tasks": {"0": "READY", "1": "READY", "2": "ERROR", "3": "READY"},
    }


@pytest.mark.asyncio
async def test_global_lock_rejects_a_different_task() -> None:
    blocking = FakeOrchestrator("0", blocking=True)
    bindings = orchestrators(**{"0": blocking})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for(bindings)),
        base_url="http://tasks.local",
    ) as client:
        first = asyncio.create_task(client.post("/tasks/0/run", json={}))
        await blocking.started.wait()
        second = await client.post("/tasks/3/run", json={})
        blocking.release.set()
        first_response = await first

    assert first_response.status_code == 200
    assert second.status_code == 409
    assert second.json()["error_code"] == "TASK_IN_PROGRESS"


@pytest.mark.asyncio
async def test_web_and_direct_api_share_the_global_lock() -> None:
    blocking = FakeOrchestrator("0", blocking=True)
    bindings = orchestrators(**{"0": blocking})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for(bindings)),
        base_url="http://tasks.local",
    ) as client:
        started = await client.post("/api/tasks/0/start", json={})
        await blocking.started.wait()
        direct = await client.post("/tasks/1/run", json={})
        blocking.release.set()
        events = await client.get(started.json()["events_url"])

    assert started.status_code == 200
    assert direct.status_code == 409
    assert direct.json()["error_code"] == "TASK_IN_PROGRESS"
    assert "event: result" in events.text


@pytest.mark.asyncio
async def test_web_terminates_task_and_releases_global_lock() -> None:
    blocking = FakeOrchestrator("0", blocking=True)
    bindings = orchestrators(**{"0": blocking})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for(bindings)),
        base_url="http://tasks.local",
    ) as client:
        started = await client.post("/api/tasks/0/start", json={})
        run_id = started.json()["run_id"]

        terminated = await client.post(f"/api/task-runs/{run_id}/terminate")
        events = await client.get(started.json()["events_url"])
        repeated = await client.post(f"/api/task-runs/{run_id}/terminate")
        next_task = await client.post("/tasks/1/run", json={})

    assert terminated.status_code == 200
    assert terminated.json()["status"] == "TERMINATED"
    assert terminated.json()["result"]["body"]["error_code"] == "TASK_TERMINATED"
    assert '"error_code": "TASK_TERMINATED"' in events.text
    assert repeated.status_code == 200
    assert repeated.json()["status"] == "ALREADY_FINISHED"
    assert next_task.status_code == 200


@pytest.mark.asyncio
async def test_web_terminate_rejects_unknown_run() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for()),
        base_url="http://tasks.local",
    ) as client:
        response = await client.post("/api/task-runs/missing/terminate")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_domain_error_keeps_failed_step() -> None:
    bindings = orchestrators(**{"1": FailingTask1("1")})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for(bindings)),
        base_url="http://tasks.local",
    ) as client:
        response = await client.post("/tasks/1/run", json={})

    assert response.status_code == 502
    assert response.json() == {
        "error_code": "EXECUTION_FAILED",
        "message": (
            "pick_place returned HTTP 502: /manipulation/release: "
            "MoveL 规划失败，错误码 -1022"
        ),
        "failed_step": "商品放置",
        "failed_interface": "manipulation_release",
        "url": "http://robot:8084/manipulation/release",
        "interface_metrics": [
            {
                "interface": "pick_place/place",
                "service": "pick_place",
                "method": "POST",
                "url": "http://pick-place.local/place",
                "call_count": 2,
                "success_count": 1,
                "failure_count": 1,
                "total_duration_ms": 125.0,
                "average_duration_ms": 62.5,
            }
        ],
    }


@pytest.mark.asyncio
async def test_web_task1_failure_result_keeps_interface_metrics() -> None:
    bindings = orchestrators(**{"1": FailingTask1("1")})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for(bindings)),
        base_url="http://tasks.local",
    ) as client:
        preview = await client.post("/api/task1/mock-order", json={})
        started = await client.post(
            "/api/tasks/1/start",
            json={
                "order_source": "mock_random",
                "order_id": preview.json()["order_id"],
                "product_names": preview.json()["product_names"],
            },
        )
        event_stream = await client.get(started.json()["events_url"])

    assert started.status_code == 200
    assert '"interface_metrics"' in event_stream.text
    assert '"call_count": 2' in event_stream.text


@pytest.mark.asyncio
async def test_web_uses_one_task_panel_and_common_sse_routes() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for()), base_url="http://tasks.local"
    ) as client:
        page = await client.get("/")
        started = await client.post("/api/tasks/3/start", json={})
        event_stream = await client.get(started.json()["events_url"])
    script = (CONFIG.parents[1] / "web" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert page.status_code == 200
    assert 'id="taskForm"' in page.text
    assert 'name="task_id" value="0"' in page.text
    assert 'name="task_id" value="3"' in page.text
    assert 'id="task0Form"' not in page.text
    assert 'id="taskElapsedTime"' in page.text
    assert 'id="pickElapsedTime"' in page.text
    assert 'id="placeElapsedTime"' in page.text
    assert page.text.count('class="pick-place-section"') == 1
    assert 'id="pickWorkspace"' in page.text
    assert 'id="placeWorkspace"' in page.text
    assert 'data-operation-mode="pick"' in page.text
    assert 'data-operation-mode="place"' in page.text
    assert 'id="robotIpForm"' in page.text
    assert 'id="taskTerminateButton"' in page.text
    assert 'id="task1MockOrder"' in page.text
    assert 'id="taskInterfaceMetrics"' in page.text
    assert 'id="parseReceiptButton"' not in page.text
    assert "elapsedTimers.task.start()" in script
    assert "elapsedTimers.pick.start()" in script
    assert "elapsedTimers.place.start()" in script
    assert "setOperationMode(button.dataset.operationMode)" in script
    assert "/api/task-runs/${runId}/terminate" in script
    assert 'fetch("/api/task1/mock-order"' in script
    assert "applyInterfaceMetric(flowEvent)" in script
    assert page.headers["cache-control"] == "no-store"
    assert started.status_code == 200
    assert started.json()["task_id"] == "3"
    assert "event: result" in event_stream.text
    assert '"task_type": "MISPLACED"' in event_stream.text


@pytest.mark.asyncio
async def test_web_starts_task1_through_the_unified_route() -> None:
    bindings = orchestrators()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for(bindings)),
        base_url="http://tasks.local",
    ) as client:
        preview = await client.post("/api/task1/mock-order", json={})
        started = await client.post(
            "/api/tasks/1/start",
            json={
                "order_source": "mock_random",
                "order_id": preview.json()["order_id"],
                "product_names": preview.json()["product_names"],
            },
        )
        event_stream = await client.get(started.json()["events_url"])

    assert preview.status_code == 200
    assert preview.json()["catalog_size"] == 43
    assert len(preview.json()["product_names"]) == 2
    assert started.status_code == 200
    assert started.json()["task_id"] == "1"
    assert started.json()["events_url"].startswith("/api/task-runs/")
    assert "event: result" in event_stream.text
    assert '"task_type": "SORTING"' in event_stream.text
    request = bindings["1"].last_request
    assert request.order_id == "preview-order"
    assert request.product_names == ["可口可乐罐装", "百事可乐瓶装"]
