from __future__ import annotations

import json

import httpx
import pytest

from task1_service.app import create_app
from task1_service.client import Task1Client
from task1_service.models import Task1Request, Task1ServiceError, Task1Settings, Task1Timeouts
from task1_service.service import Task1Orchestrator


class Task1Mock:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.health = {"navigation": "READY", "perception": "READY", "pose": "READY", "pick_place": "READY", "sku": "READY"}
        self.names = {
            "可口可乐罐装": "H2_F_L1_C01",
            "百事可乐瓶装": "H2_F_L1_C04",
        }
        self.pick_timeout_once = False
        self.pick_attempts = 0

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        service = {
            "navigation.local": "navigation",
            "perception.local": "perception",
            "pose.local": "pose",
            "pick-place.local": "pick_place",
            "sku.local": "sku",
        }[host]
        path = request.url.path
        if path == "/" + service + "/health" or (service == "pick_place" and path == "/health"):
            return httpx.Response(200, json={"status": self.health[service]})
        if path == "/perception/parse":
            return httpx.Response(200, json={"product_names": list(self.names)})
        if path == "/sku/search_by_name":
            payload = json.loads(request.content)
            name = payload["name"]
            return httpx.Response(
                200,
                json={"sku_id": "SKU", "name": name, "images": [], "locations": [self.names[name]]},
            )
        if path == "/sku/search_by_location":
            payload = json.loads(request.content)
            location = payload["location"]
            name = next(name for name, slot in self.names.items() if slot == location)
            return httpx.Response(
                200,
                json={"sku_id": "SKU", "name": name, "images": [], "locations": [location]},
            )
        if path == "/pick":
            self.pick_attempts += 1
            if self.pick_timeout_once:
                self.pick_timeout_once = False
                raise httpx.ReadTimeout("temporary timeout", request=request)
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        if path in {"/navigation/navigate", "/pose/prepare"}:
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        return httpx.Response(404, json={"error_code": "UNKNOWN_ENDPOINT"})


def settings() -> Task1Settings:
    return Task1Settings(
        services={
            "navigation": "http://navigation.local",
            "perception": "http://perception.local",
            "pose": "http://pose.local",
            "pick_place": "http://pick-place.local",
            "sku": "http://sku.local",
        },
        timeouts=Task1Timeouts(
            connect_seconds=0.1,
            health_seconds=0.2,
            receipt_seconds=0.2,
            sku_seconds=0.2,
            navigation_seconds=0.2,
            pose_seconds=0.2,
            pick_seconds=0.2,
        ),
    )


def paths(mock: Task1Mock) -> list[str]:
    return [request.url.path for request in mock.requests]


def payload(request: httpx.Request) -> dict:
    return json.loads(request.content)


@pytest.mark.asyncio
async def test_task1_pick_count_one_is_serial_and_uses_sku_locations() -> None:
    mock = Task1Mock()
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        result = await Task1Orchestrator(settings(), client).run(Task1Request(pick_count=1))

    assert result.status == "SUCCEEDED"
    assert result.target_items[0].product_slot_id == "H2_F_L1_C01"
    assert result.target_items[0].picked is True
    assert result.target_items[1].picked is False
    assert result.held_items == {"LEFT": "可口可乐罐装"}
    assert paths(mock) == [
        "/navigation/health", "/perception/health", "/pose/health", "/health", "/sku/health",
        "/navigation/navigate", "/pose/prepare", "/perception/parse",
        "/sku/search_by_name", "/sku/search_by_name",
        "/navigation/navigate", "/pose/prepare", "/pick",
    ]
    assert payload(mock.requests[-3]) == {"target_id": "H2_F_L1_C01"}
    assert payload(mock.requests[-2]) == {"pose_type": "SHELF_PICK_READY", "shelf_level": "L1"}


@pytest.mark.asyncio
async def test_task1_pick_count_two_picks_both_products_in_receipt_order() -> None:
    mock = Task1Mock()
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        result = await Task1Orchestrator(settings(), client).run(Task1Request(pick_count=2))

    assert [item.picked for item in result.target_items] == [True, True]
    assert result.held_items == {"LEFT": "可口可乐罐装", "RIGHT": "百事可乐瓶装"}
    pick_requests = [request for request in mock.requests if request.url.path == "/pick"]
    assert [payload(request)["hand"] for request in pick_requests] == ["LEFT", "RIGHT"]
    assert [payload(request)["product_name"] for request in pick_requests] == list(mock.names)


@pytest.mark.asyncio
async def test_task1_does_not_prepare_pose_after_navigation_failure() -> None:
    mock = Task1Mock()
    mock.health["navigation"] = "ERROR"
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        with pytest.raises(Task1ServiceError) as error:
            await Task1Orchestrator(settings(), client).run(Task1Request())

    assert error.value.code == "CAPABILITY_NOT_READY"
    assert "/navigation/navigate" not in paths(mock)


@pytest.mark.asyncio
async def test_task1_retries_pick_with_same_idempotency_key() -> None:
    mock = Task1Mock()
    mock.pick_timeout_once = True
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        await Task1Orchestrator(settings(), client).run(Task1Request(pick_count=1))

    pick_requests = [request for request in mock.requests if request.url.path == "/pick"]
    assert mock.pick_attempts == 2
    assert len(pick_requests) == 2
    assert pick_requests[0].headers["Idempotency-Key"] == pick_requests[1].headers["Idempotency-Key"]


@pytest.mark.asyncio
async def test_task1_writes_pickplace_style_operation_log(tmp_path) -> None:
    mock = Task1Mock()
    task_settings = settings().model_copy(update={"log_dir": str(tmp_path)})
    client = Task1Client(task_settings, transport=mock.transport)
    async with client:
        await Task1Orchestrator(task_settings, client).run(Task1Request(pick_count=1), "web-task1-log")

    directories = list(tmp_path.iterdir())
    assert len(directories) == 1
    assert (directories[0] / "operation.json").exists()
    events = (directories[0] / "events.jsonl").read_text(encoding="utf-8")
    assert '"event": "小票识别"' in events
    assert '"event": "SKU货位转换"' in events
    assert '"event": "抓取"' in events
    assert '"event": "operation"' in events


@pytest.mark.asyncio
async def test_task1_app_exposes_run_endpoint() -> None:
    mock = Task1Mock()
    client = Task1Client(settings(), transport=mock.transport)
    orchestrator = Task1Orchestrator(settings(), client)
    app = create_app(settings(), orchestrator=orchestrator)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://task1") as http_client:
        response = await http_client.post("/task1/run", json={"pick_count": 1})

    assert response.status_code == 200
    assert response.json()["requested_pick_count"] == 1
    await client.aclose()
