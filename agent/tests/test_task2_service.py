from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path

import httpx
import pytest

from task2_service.app import create_app
from task2_service.client import Task2Client
from task2_service.models import (
    Hand,
    InspectionPose,
    TargetItem,
    Task2Request,
    Task2Result,
    Task2ServiceError,
    Task2Settings,
    Task2Timeouts,
)
from task2_service.service import Task2Orchestrator


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class Task2Mock:
    def __init__(self, inspection_results: list[list[str]]) -> None:
        self.requests: list[httpx.Request] = []
        self.inspection_results = deque(inspection_results)
        self.health = {
            "navigation": "READY",
            "perception": "READY",
            "pose": "READY",
            "pick_place": "READY",
        }
        self.pick_timeout_once = False

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        service = {
            "navigation.local": "navigation",
            "perception.local": "perception",
            "pose.local": "pose",
            "pick-place.local": "pick_place",
        }[request.url.host]
        path = request.url.path
        health_path = "/health" if service == "pick_place" else f"/{service}/health"
        if path == health_path:
            return httpx.Response(200, json={"status": self.health[service]})
        if path == "/perception/inspect":
            if not self.inspection_results:
                raise AssertionError("test did not provide enough inspection responses")
            return httpx.Response(200, json={"findings": self.inspection_results.popleft()})
        if path == "/pick":
            if self.pick_timeout_once:
                self.pick_timeout_once = False
                raise httpx.ReadTimeout("temporary timeout", request=request)
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        if path in {"/place", "/navigation/navigate", "/pose/prepare"}:
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        return httpx.Response(404, json={"error_code": "UNKNOWN_ENDPOINT"})


def settings(tmp_path: Path, *, left_only: bool = False) -> Task2Settings:
    first_hands = ["LEFT"] if left_only else ["LEFT", "RIGHT"]
    second_hands = ["LEFT"] if left_only else ["LEFT", "RIGHT"]
    return Task2Settings(
        services={
            "navigation": "http://navigation.local",
            "perception": "http://perception.local",
            "pose": "http://pose.local",
            "pick_place": "http://pick-place.local",
        },
        timeouts=Task2Timeouts(
            connect_seconds=0.1,
            health_seconds=0.2,
            inspection_seconds=0.2,
            navigation_seconds=0.2,
            pose_seconds=0.2,
            pick_seconds=0.2,
            place_seconds=0.2,
        ),
        inspection_points=["H1_F_L_INSPECT", "H1_F_R_INSPECT"],
        product_hand_options_file="unused.yaml",
        product_hand_options={
            "H1_F_L1_C01": {
                "product_name": "上层商品",
                "hands": first_hands,
                "target_id": "H1_F_L_INSPECT",
            },
            "H1_F_L3_C01": {
                "product_name": "下层商品",
                "hands": second_hands,
                "target_id": "H1_F_L_INSPECT",
            },
            "H1_F_L2_C02": {
                "product_name": "第二上层商品",
                "hands": ["LEFT", "RIGHT"],
                "target_id": "H1_F_L_INSPECT",
            },
            "H1_F_L1_C04": {
                "product_name": "右侧商品",
                "hands": ["LEFT", "RIGHT"],
                "target_id": "H1_F_R_INSPECT",
            },
        },
        log_dir=str(tmp_path),
    )


def payload(request: httpx.Request) -> dict:
    return json.loads(request.content) if request.content else {}


def requests_for(mock: Task2Mock, path: str) -> list[httpx.Request]:
    return [request for request in mock.requests if request.url.path == path]


@pytest.mark.asyncio
async def test_task2_records_inspection_context_and_restores_it_for_place(
    tmp_path: Path,
) -> None:
    mock = Task2Mock([["上层商品"], ["下层商品"]])
    task_settings = settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    async with client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.status == "SUCCEEDED"
    assert result.inspection_pass == 1
    assert [item.inspection_target_id for item in result.target_items] == [
        "H1_F_L_INSPECT",
        "H1_F_L_INSPECT",
    ]
    assert [item.inspection_pose_type for item in result.target_items] == [
        "SHELF_VIEW_UPPER",
        "SHELF_VIEW_LOWER",
    ]
    assert [item.hand for item in result.target_items] == ["LEFT", "RIGHT"]
    assert all(item.picked and item.placed for item in result.target_items)
    assert result.held_items == {}

    inspections = requests_for(mock, "/perception/inspect")
    assert [payload(request) for request in inspections] == [
        {"task_type": "SHORTAGE"},
        {"task_type": "SHORTAGE"},
    ]
    place_indexes = [
        index for index, request in enumerate(mock.requests) if request.url.path == "/place"
    ]
    assert payload(mock.requests[place_indexes[0] - 1]) == {
        "pose_type": "SHELF_VIEW_UPPER"
    }
    assert payload(mock.requests[place_indexes[1] - 1]) == {
        "pose_type": "SHELF_VIEW_LOWER"
    }
    assert all(
        payload(request).get("pose_type") != "SHELF_PLACE_READY"
        for request in requests_for(mock, "/pose/prepare")
    )


@pytest.mark.asyncio
async def test_task2_continues_in_reverse_and_ignores_repeated_finding(
    tmp_path: Path,
) -> None:
    mock = Task2Mock(
        [
            ["上层商品"],
            [],
            [],
            [],
            [],
            [],
            ["上层商品", "第二上层商品"],
        ]
    )
    task_settings = settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    async with client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.inspection_pass == 2
    assert result.product_names == ["上层商品", "第二上层商品"]
    inspection_navigation = [
        payload(request)["target_id"]
        for request in requests_for(mock, "/navigation/navigate")
        if payload(request)["target_id"] in task_settings.inspection_points
    ]
    assert inspection_navigation[:3] == [
        "H1_F_L_INSPECT",
        "H1_F_R_INSPECT",
        "H1_F_L_INSPECT",
    ]
    assert len(requests_for(mock, "/perception/inspect")) == 7


@pytest.mark.asyncio
async def test_task2_same_hand_picks_and_places_serially(tmp_path: Path) -> None:
    mock = Task2Mock([["上层商品"], ["下层商品"]])
    task_settings = settings(tmp_path, left_only=True)
    client = Task2Client(task_settings, transport=mock.transport)
    async with client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert [item.hand for item in result.target_items] == ["LEFT", "LEFT"]
    actions = [
        (request.url.path, payload(request).get("product_name"))
        for request in mock.requests
        if request.url.path in {"/pick", "/place"}
    ]
    assert actions == [
        ("/pick", "上层商品"),
        ("/place", "上层商品"),
        ("/pick", "下层商品"),
        ("/place", "下层商品"),
    ]


@pytest.mark.asyncio
async def test_task2_retries_physical_action_with_same_key(tmp_path: Path) -> None:
    mock = Task2Mock([["上层商品"], ["下层商品"]])
    mock.pick_timeout_once = True
    task_settings = settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    async with client:
        await Task2Orchestrator(task_settings, client).run(Task2Request())

    picks = requests_for(mock, "/pick")
    assert len(picks) == 3
    assert picks[0].headers["Idempotency-Key"] == picks[1].headers["Idempotency-Key"]


@pytest.mark.asyncio
async def test_task2_rejects_finding_without_matching_hand_config(
    tmp_path: Path,
) -> None:
    mock = Task2Mock([["未知商品"]])
    task_settings = settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    async with client:
        with pytest.raises(Task2ServiceError) as error:
            await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert error.value.code == "UNKNOWN_PRODUCT_HAND_OPTIONS"
    assert error.value.step == "货架巡检"
    assert not requests_for(mock, "/pick")


@pytest.mark.asyncio
async def test_task2_rejects_more_than_two_inspection_results(tmp_path: Path) -> None:
    mock = Task2Mock([["上层商品", "下层商品", "右侧商品"]])
    task_settings = settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    async with client:
        with pytest.raises(Task2ServiceError) as error:
            await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert error.value.code == "INVALID_FINDINGS"
    assert not requests_for(mock, "/pick")


def test_task2_production_config_uses_complete_hand_options() -> None:
    task_settings = Task2Settings.load(CONFIG_DIR / "task2.production.yaml")

    assert len(task_settings.product_hand_options) == 122
    assert task_settings.product_hand_options["H2_F_L2_C05"].hands == ["RIGHT"]
    assert task_settings.inspection_points[0] == "H1_F_L_INSPECT"
    assert task_settings.services.pose == "http://192.168.3.226:8084"


@pytest.mark.asyncio
async def test_task2_app_exposes_run_endpoint(tmp_path: Path) -> None:
    mock = Task2Mock([["上层商品"], ["下层商品"]])
    task_settings = settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    app = create_app(
        task_settings, orchestrator=Task2Orchestrator(task_settings, client)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://task2"
    ) as http_client:
        response = await http_client.post(
            "/task2/run", headers={"Idempotency-Key": "task2-test"}, json={}
        )

    assert response.status_code == 200
    assert response.json()["task_run_id"] == "task2-test"
    assert response.json()["status"] == "SUCCEEDED"
    await client.aclose()


@pytest.mark.asyncio
async def test_task2_app_rejects_extra_request_fields(tmp_path: Path) -> None:
    mock = Task2Mock([])
    task_settings = settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    app = create_app(
        task_settings, orchestrator=Task2Orchestrator(task_settings, client)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://task2"
    ) as http_client:
        response = await http_client.post("/task2/run", json={"count": 2})

    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_REQUEST"
    await client.aclose()


@pytest.mark.asyncio
async def test_task2_health_checks_all_direct_dependencies(tmp_path: Path) -> None:
    mock = Task2Mock([])
    task_settings = settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    app = create_app(
        task_settings, orchestrator=Task2Orchestrator(task_settings, client)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://task2"
    ) as http_client:
        response = await http_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "READY"}
    assert {
        request.url.host
        for request in mock.requests
        if request.url.path.endswith("/health") or request.url.path == "/health"
    } == {"navigation.local", "perception.local", "pose.local", "pick-place.local"}
    await client.aclose()


@pytest.mark.asyncio
async def test_task2_app_rejects_concurrent_run(tmp_path: Path) -> None:
    class BlockingOrchestrator:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def run(
            self, _: Task2Request, operation_key: str | None = None
        ) -> Task2Result:
            self.started.set()
            await self.release.wait()
            items = [
                TargetItem(
                    product_name="上层商品",
                    inspection_target_id="H1_F_L_INSPECT",
                    inspection_pose_type=InspectionPose.UPPER,
                    hand=Hand.LEFT,
                    picked=True,
                    placed=True,
                ),
                TargetItem(
                    product_name="下层商品",
                    inspection_target_id="H1_F_L_INSPECT",
                    inspection_pose_type=InspectionPose.LOWER,
                    hand=Hand.RIGHT,
                    picked=True,
                    placed=True,
                ),
            ]
            return Task2Result(
                task_run_id=operation_key or "first",
                task_type="SHORTAGE",
                status="SUCCEEDED",
                inspection_pass=1,
                product_names=[item.product_name for item in items],
                target_items=items,
                held_items={},
            )

    task_settings = settings(tmp_path)
    orchestrator = BlockingOrchestrator()
    app = create_app(task_settings, orchestrator=orchestrator)  # type: ignore[arg-type]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://task2"
    ) as http_client:
        first = asyncio.create_task(http_client.post("/task2/run", json={}))
        await orchestrator.started.wait()
        second = await http_client.post("/task2/run", json={})
        orchestrator.release.set()
        first_response = await first

    assert second.status_code == 409
    assert second.json()["error_code"] == "TASK_IN_PROGRESS"
    assert first_response.status_code == 200
