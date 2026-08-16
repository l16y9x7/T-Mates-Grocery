from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import httpx
import pytest

from task2_service.client import Task2Client
from task2_service.models import (
    InspectionPose,
    Task2Request,
    Task2ServiceError,
    Task2Settings,
    Task2Timeouts,
)
from task2_service.service import Task2Orchestrator
from task_service.settings import TaskServiceSettings


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class Task2Mock:
    def __init__(self, inspection_results: list[list[object]]) -> None:
        self.requests: list[httpx.Request] = []
        self.inspection_results = deque(inspection_results)
        self.health = {
            "navigation": "READY",
            "perception": "READY",
            "pose": "READY",
            "pick_place": "READY",
            "camera": "READY",
        }
        self.pick_timeout_once = False
        self.pick_attempts = 0
        self.pick_failure_limit = 0
        self.head_color_online = True

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
            "camera.local": "camera",
        }[request.url.host]
        path = request.url.path
        health_path = "/health" if service == "pick_place" else f"/{service}/health"
        if path == health_path:
            payload: dict[str, object] = {"status": self.health[service]}
            if service == "pose":
                payload["current_pose"] = {"pose_type": "START_POSITION"}
            return httpx.Response(200, json=payload)
        if path == "/camera/list":
            return httpx.Response(
                200,
                json={
                    "cameras": [
                        {
                            "id": "head",
                            "online": True,
                            "streams": [
                                {"type": "color", "online": self.head_color_online}
                            ],
                        }
                    ]
                },
            )
        if path == "/perception/inspect":
            if not self.inspection_results:
                raise AssertionError("test did not provide enough inspection responses")
            return httpx.Response(200, json={"findings": self.inspection_results.popleft()})
        if path == "/pick":
            self.pick_attempts += 1
            if self.pick_timeout_once:
                self.pick_timeout_once = False
                raise httpx.ReadTimeout("temporary timeout", request=request)
            if self.pick_attempts <= self.pick_failure_limit:
                return httpx.Response(
                    502,
                    json={"error_code": "EXECUTION_FAILED", "message": "pick failed"},
                )
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        if path == "/navigation/nudge":
            request_payload = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "status": "SUCCEEDED",
                    "station_id": "replenishment_pickup",
                    "nudge_count": 0 if request_payload.get("action") == "return" else 1,
                },
            )
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
            "camera": "http://camera.local",
        },
        timeouts=Task2Timeouts(
            connect_seconds=0.1,
            health_seconds=0.2,
            inspection_seconds=0.2,
            camera_seconds=0.2,
            navigation_seconds=0.2,
            pose_seconds=0.2,
            pick_seconds=0.2,
            place_seconds=0.2,
        ),
        inspection_points=["H1_F_L_INSPECT", "H1_F_R_INSPECT"],
        camera="head",
        baseline_dir=str(tmp_path / "task0"),
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
        log_dir=str(tmp_path / "log"),
    )


def shortage(*names: str) -> list[dict[str, str]]:
    return [{"shortage_product_name": name} for name in names]


def create_baselines(task_settings: Task2Settings) -> None:
    for target_id in task_settings.inspection_points:
        for pose in (InspectionPose.UPPER, InspectionPose.LOWER):
            path = (
                Path(task_settings.baseline_dir)
                / f"{target_id}_{pose.directory_suffix}"
                / "rgb.jpg"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"baseline-{target_id}-{pose.value}".encode())


def ready_settings(tmp_path: Path, *, left_only: bool = False) -> Task2Settings:
    task_settings = settings(tmp_path, left_only=left_only)
    create_baselines(task_settings)
    return task_settings


def payload(request: httpx.Request) -> dict:
    return json.loads(request.content) if request.content else {}


def requests_for(mock: Task2Mock, path: str) -> list[httpx.Request]:
    return [request for request in mock.requests if request.url.path == path]


@pytest.mark.asyncio
async def test_task2_records_inspection_context_and_restores_it_for_place(
    tmp_path: Path,
) -> None:
    mock = Task2Mock([shortage("上层商品"), shortage("下层商品")])
    task_settings = ready_settings(tmp_path)
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
    upper_payload = payload(inspections[0])
    lower_payload = payload(inspections[1])
    assert upper_payload == {
        "task_type": "SHORTAGE",
        "location_id": "H1_F_L1_C01",
        "pose_type": "SHELF_VIEW_UPPER",
    }
    assert lower_payload == {
        "task_type": "SHORTAGE",
        "location_id": "H1_F_L1_C01",
        "pose_type": "SHELF_VIEW_LOWER",
    }
    assert not requests_for(mock, "/camera/snapshot")
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
            shortage("上层商品"),
            [],
            [],
            [],
            [],
            [],
            shortage("上层商品", "第二上层商品"),
        ]
    )
    task_settings = ready_settings(tmp_path)
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
    mock = Task2Mock([shortage("上层商品"), shortage("下层商品")])
    task_settings = ready_settings(tmp_path, left_only=True)
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
    mock = Task2Mock([shortage("上层商品"), shortage("下层商品")])
    mock.pick_timeout_once = True
    task_settings = ready_settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    async with client:
        await Task2Orchestrator(task_settings, client).run(Task2Request())

    picks = requests_for(mock, "/pick")
    assert len(picks) == 3
    assert picks[0].headers["Idempotency-Key"] == picks[1].headers["Idempotency-Key"]


@pytest.mark.asyncio
async def test_task2_continues_other_item_after_pick_retry_failure_and_returns_start(
    tmp_path: Path,
) -> None:
    mock = Task2Mock([shortage("上层商品"), shortage("下层商品")])
    mock.pick_failure_limit = 2
    task_settings = ready_settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    async with client:
        with pytest.raises(Task2ServiceError) as error:
            await Task2Orchestrator(task_settings, client).run(
                Task2Request(), "task2-recovery"
            )

    assert error.value.code == "TASK_ACTIONS_FAILED"
    assert error.value.step == "抓放失败汇总"
    picks = requests_for(mock, "/pick")
    assert [payload(request)["product_name"] for request in picks] == [
        "上层商品",
        "上层商品",
        "下层商品",
    ]
    assert [request.headers["Idempotency-Key"] for request in picks[:2]] == [
        "task2-recovery:task2.pick.0",
        "task2-recovery:task2.pick.0:recovery.retry",
    ]
    assert [payload(request)["product_name"] for request in requests_for(mock, "/place")] == [
        "下层商品"
    ]
    assert [payload(request) for request in requests_for(mock, "/navigation/nudge")] == [
        {"action": "approach", "direction": "back"},
        {"action": "return"},
    ]
    navigation_targets = [
        payload(request)["target_id"]
        for request in requests_for(mock, "/navigation/navigate")
    ]
    assert navigation_targets[-1] == "start"
    assert "task_boundary" not in navigation_targets


@pytest.mark.asyncio
async def test_task2_rejects_finding_without_matching_hand_config(
    tmp_path: Path,
) -> None:
    mock = Task2Mock([shortage("未知商品")])
    task_settings = ready_settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    async with client:
        with pytest.raises(Task2ServiceError) as error:
            await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert error.value.code == "UNKNOWN_PRODUCT_HAND_OPTIONS"
    assert error.value.step == "货架巡检"
    assert not requests_for(mock, "/pick")


@pytest.mark.asyncio
async def test_task2_rejects_more_than_two_inspection_results(tmp_path: Path) -> None:
    mock = Task2Mock([shortage("上层商品", "下层商品", "右侧商品")])
    task_settings = ready_settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    async with client:
        with pytest.raises(Task2ServiceError) as error:
            await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert error.value.code == "INVALID_FINDINGS"
    assert not requests_for(mock, "/pick")


@pytest.mark.asyncio
async def test_task2_rejects_legacy_string_inspection_response(tmp_path: Path) -> None:
    mock = Task2Mock([["上层商品"]])
    task_settings = ready_settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    async with client:
        with pytest.raises(Task2ServiceError) as error:
            await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert error.value.code == "INVALID_RESPONSE"
    assert not requests_for(mock, "/pick")


@pytest.mark.asyncio
async def test_task2_rejects_blank_shortage_product_name(tmp_path: Path) -> None:
    mock = Task2Mock([shortage("  ")])
    task_settings = ready_settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    async with client:
        with pytest.raises(Task2ServiceError) as error:
            await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert error.value.code == "INVALID_FINDINGS"
    assert not requests_for(mock, "/pick")


@pytest.mark.asyncio
async def test_task2_requires_task0_baselines(tmp_path: Path) -> None:
    mock = Task2Mock([])
    task_settings = settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    orchestrator = Task2Orchestrator(task_settings, client)
    async with client:
        assert not await orchestrator.ready()
        with pytest.raises(Task2ServiceError) as error:
            await orchestrator.run(Task2Request())

    assert error.value.code == "BASELINE_NOT_READY"
    assert error.value.step == "健康检查"
    assert not requests_for(mock, "/camera/snapshot")
    assert not requests_for(mock, "/perception/inspect")


@pytest.mark.asyncio
async def test_task2_health_requires_head_color_stream(tmp_path: Path) -> None:
    mock = Task2Mock([])
    mock.head_color_online = False
    task_settings = ready_settings(tmp_path)
    client = Task2Client(task_settings, transport=mock.transport)
    async with client:
        assert not await Task2Orchestrator(task_settings, client).ready()

    health_hosts = {
        request.url.host
        for request in mock.requests
        if request.url.path.endswith("/health") or request.url.path == "/health"
    }
    assert health_hosts == {
        "navigation.local",
        "perception.local",
        "pose.local",
        "pick-place.local",
        "camera.local",
    }
    assert requests_for(mock, "/camera/list")


def test_task2_requires_product_mapping_for_every_inspection_point(
    tmp_path: Path,
) -> None:
    raw = settings(tmp_path).model_dump(mode="python")
    raw["inspection_points"].append("H2_F_L_INSPECT")

    with pytest.raises(ValueError, match="no product location mapping"):
        Task2Settings.model_validate(raw)


def test_task2_production_config_uses_complete_hand_options() -> None:
    task_settings = TaskServiceSettings.load(
        CONFIG_DIR / "runtime.production.yaml"
    ).tasks.task2

    assert len(task_settings.product_hand_options) == 122
    assert task_settings.product_hand_options["H2_F_L2_C05"].hands == ["RIGHT"]
    assert task_settings.inspection_points[0] == "H1_F_L_INSPECT"
    assert task_settings.services.pose.endswith(":8084")
    assert task_settings.services.camera.endswith(":8085")
    assert task_settings.camera == "head"
    assert Path(task_settings.baseline_dir) == CONFIG_DIR.parent / "output" / "task0"
    expected_locations = {
        "H1_F_L_INSPECT": "H1_F_L1_C01",
        "H1_F_R_INSPECT": "H1_F_L1_C04",
        "H1_B_L_INSPECT": "H1_B_L1_C01",
        "H1_B_R_INSPECT": "H1_B_L1_C04",
        "H2_F_L_INSPECT": "H2_F_L1_C01",
        "H2_F_R_INSPECT": "H2_F_L1_C04",
        "H2_B_L_INSPECT": "H2_B_L1_C01",
        "H2_B_R_INSPECT": "H2_B_L1_C04",
    }
    assert {
        target_id: task_settings.location_id_for_target(target_id)
        for target_id in task_settings.inspection_points
    } == expected_locations
