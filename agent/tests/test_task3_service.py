from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path

import httpx
import pytest

from task3_service.client import Task3Client
from task3_service.models import (
    FindingContext,
    Hand,
    InspectionPose,
    SwapItem,
    Task3Request,
    Task3Result,
    Task3ServiceError,
    Task3Settings,
    Task3Timeouts,
)
from task3_service.service import Task3Orchestrator
from manipulation_policy import SPECIAL_SHELF_NUDGE_PRODUCT
from task_service.settings import TaskServiceSettings


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def payload(request: httpx.Request) -> dict[str, object]:
    return json.loads(request.content) if request.content else {}


class Task3Mock:
    def __init__(self, inspection_results: list[list[dict[str, str]]]) -> None:
        self.requests: list[httpx.Request] = []
        self.inspection_results = deque(inspection_results)
        self.pick_timeout_once = False
        self.place_attempts = 0
        self.place_failure_limit = 0
        self.place_failure_payload: dict[str, object] = {
            "error_code": "EXECUTION_FAILED",
            "message": "place failed",
            "failed_interface": "manipulation_release",
            "url": "http://robot:8084/manipulation/release",
            "pose": [1, 2, 3, 4, 5, 6],
        }
        self.skus = {
            "错误商品": {
                "sku_id": "SKU-A",
                "name": "错误商品",
                "images": [],
                "locations": ["H1_F_L1_C04"],
            },
            "应放商品": {
                "sku_id": "SKU-B",
                "name": "应放商品",
                "images": [],
                "locations": ["H1_F_L2_C02"],
            },
        }

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in {
            "/navigation/health",
            "/perception/health",
            "/pose/health",
            "/health",
            "/sku/health",
            "/camera/health",
        }:
            response: dict[str, object] = {"status": "READY"}
            if path == "/pose/health":
                response["current_pose"] = {"pose_type": "START_POSITION"}
            return httpx.Response(200, json=response)
        if path == "/camera/list":
            return httpx.Response(
                200,
                json={
                    "cameras": [
                        {
                            "id": "head",
                            "online": True,
                            "streams": [{"type": "color", "online": True}],
                        }
                    ]
                },
            )
        if path == "/perception/inspect":
            if not self.inspection_results:
                raise AssertionError("test did not provide enough inspection responses")
            return httpx.Response(200, json={"findings": self.inspection_results.popleft()})
        if path == "/sku/search_by_name":
            return httpx.Response(200, json=self.skus[request.url.params["name"]])
        if path == "/pick" and self.pick_timeout_once:
            self.pick_timeout_once = False
            raise httpx.ReadTimeout("temporary timeout", request=request)
        if path == "/place":
            self.place_attempts += 1
            if self.place_attempts <= self.place_failure_limit:
                return httpx.Response(
                    502,
                    json=self.place_failure_payload,
                )
            return httpx.Response(200, json={"status": "SUCCEEDED", "executed": True})
        if path == "/navigation/nudge":
            request_payload = payload(request)
            return httpx.Response(
                200,
                json={
                    "status": "SUCCEEDED",
                    "station_id": "H1_F_R_INSPECT",
                    "nudge_count": 0 if request_payload.get("action") == "return" else 1,
                },
            )
        if path in {"/navigation/navigate", "/pose/prepare", "/pick", "/place"}:
            return httpx.Response(200, json={"status": "SUCCEEDED", "executed": True})
        return httpx.Response(404, json={"error_code": "UNKNOWN_ENDPOINT"})


def finding() -> list[dict[str, str]]:
    return [
        {
            "misplaced_product_name": "错误商品",
            "gt_product_name": "应放商品",
        }
    ]


def settings(tmp_path: Path, *, unsafe_second_slot: bool = False) -> Task3Settings:
    hands = ["RIGHT"] if unsafe_second_slot else ["LEFT", "RIGHT"]
    result = Task3Settings(
        services={
            "navigation": "http://navigation.local",
            "perception": "http://perception.local",
            "pose": "http://pose.local",
            "pick_place": "http://pick-place.local",
            "sku": "http://sku.local",
            "camera": "http://camera.local",
        },
        timeouts=Task3Timeouts(
            connect_seconds=0.1,
            health_seconds=0.2,
            inspection_seconds=0.2,
            camera_seconds=0.2,
            sku_seconds=0.2,
            navigation_seconds=0.2,
            pose_seconds=0.2,
            pick_seconds=0.2,
            place_seconds=0.2,
        ),
        inspection_points=[
            {"target_id": "H1_F_L_INSPECT", "location_id": "H1_F_L1_C01"},
            {"target_id": "H1_F_R_INSPECT", "location_id": "H1_F_L1_C04"},
        ],
        camera="head",
        baseline_dir=str(tmp_path / "task0"),
        task_boundary="task_boundary",
        product_hand_options_file="unused.yaml",
        product_hand_options={
            "H1_F_L1_C01": {
                "product_name": "代表商品",
                "hands": ["LEFT", "RIGHT"],
                "target_id": "H1_F_L_INSPECT",
            },
            "H1_F_L2_C02": {
                "product_name": "应放商品",
                "hands": ["LEFT", "RIGHT"],
                "target_id": "H1_F_L_INSPECT",
            },
            "H1_F_L1_C04": {
                "product_name": "错误商品",
                "hands": hands,
                "target_id": "H1_F_R_INSPECT",
            },
        },
        log_dir=str(tmp_path / "log"),
    )
    create_baselines(result)
    return result


def create_baselines(task_settings: Task3Settings) -> None:
    for point in task_settings.inspection_points:
        for pose in (InspectionPose.UPPER, InspectionPose.LOWER):
            path = (
                Path(task_settings.baseline_dir)
                / f"{point.target_id}_{pose.directory_suffix}"
                / "rgb.jpg"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"baseline-{point.target_id}-{pose.value}".encode())


def requests_for(mock: Task3Mock, path: str) -> list[httpx.Request]:
    return [request for request in mock.requests if request.url.path == path]


@pytest.mark.asyncio
async def test_task3_uses_real_inspection_contract_and_swaps_products(tmp_path: Path) -> None:
    mock = Task3Mock([finding()])
    task_settings = settings(tmp_path)
    client = Task3Client(task_settings, transport=mock.transport)
    async with client:
        result = await Task3Orchestrator(task_settings, client).run(
            Task3Request(), "task3-operation"
        )

    assert result.status == "SUCCEEDED"
    assert result.inspection_pass == 1
    assert result.held_items == {}
    assert [(item.source_slot_id, item.destination_slot_id) for item in result.target_items] == [
        ("H1_F_L2_C02", "H1_F_L1_C04"),
        ("H1_F_L1_C04", "H1_F_L2_C02"),
    ]
    assert [item.hand for item in result.target_items] == [Hand.LEFT, Hand.RIGHT]
    assert all(item.picked and item.placed for item in result.target_items)

    inspect_request = requests_for(mock, "/perception/inspect")[0]
    inspect_payload = payload(inspect_request)
    assert inspect_payload == {
        "task_type": "MISPLACED",
        "location_id": "H1_F_L_INSPECT",
        "pose_type": "SHELF_VIEW_UPPER",
    }
    assert not requests_for(mock, "/camera/snapshot")

    manipulation = [
        request for request in mock.requests if request.url.path in {"/pick", "/place"}
    ]
    assert [(request.url.path, payload(request)["product_name"], payload(request)["hand"]) for request in manipulation] == [
        ("/pick", "错误商品", "LEFT"),
        ("/pick", "应放商品", "RIGHT"),
        ("/place", "错误商品", "LEFT"),
        ("/place", "应放商品", "RIGHT"),
    ]
    assert all(payload(request)["task_type"] == "MISPLACED" for request in manipulation)
    assert [payload(request)["level"] for request in requests_for(mock, "/pick")] == [
        "L2",
        "L1",
    ]
    assert all("level" not in payload(request) for request in requests_for(mock, "/place"))
    assert [
        {
            "location_id": payload(request)["location_id"],
            "pose_type": payload(request)["pose_type"],
        }
        for request in requests_for(mock, "/place")
    ] == [
        {
            "location_id": "H1_F_R_INSPECT",
            "pose_type": "SHELF_VIEW_UPPER",
        },
        {
            "location_id": "H1_F_L_INSPECT",
            "pose_type": "SHELF_VIEW_UPPER",
        },
    ]
    navigation_targets = [payload(request)["target_id"] for request in requests_for(mock, "/navigation/navigate")]
    assert navigation_targets == [
        "H1_F_L_INSPECT",
        "H1_F_R_INSPECT",
        "H1_F_L_INSPECT",
        "task_boundary",
    ]


@pytest.mark.asyncio
async def test_task3_special_product_nudges_for_pick_but_not_shelf_place(
    tmp_path: Path,
) -> None:
    mock = Task3Mock(
        [
            [
                {
                    "misplaced_product_name": "错误商品",
                    "gt_product_name": SPECIAL_SHELF_NUDGE_PRODUCT,
                }
            ]
        ]
    )
    mock.skus[SPECIAL_SHELF_NUDGE_PRODUCT] = mock.skus.pop("应放商品")
    mock.skus[SPECIAL_SHELF_NUDGE_PRODUCT][
        "name"
    ] = SPECIAL_SHELF_NUDGE_PRODUCT
    task_settings = settings(tmp_path)
    task_settings.product_hand_options[
        "H1_F_L2_C02"
    ].product_name = SPECIAL_SHELF_NUDGE_PRODUCT
    client = Task3Client(task_settings, transport=mock.transport)

    async with client:
        result = await Task3Orchestrator(task_settings, client).run(Task3Request())

    assert result.status == "SUCCEEDED"
    nudge_requests = requests_for(mock, "/navigation/nudge")
    assert [payload(request) for request in nudge_requests] == [
        {"action": "approach", "direction": "left"},
        {"action": "return"},
    ]
    special_actions = [
        request
        for request in mock.requests
        if request.url.path in {"/pick", "/place"}
        and payload(request)["product_name"] == SPECIAL_SHELF_NUDGE_PRODUCT
    ]
    assert [payload(request)["hand"] for request in special_actions] == [
        "RIGHT",
        "RIGHT",
    ]
    special_pick, special_place = special_actions
    assert mock.requests.index(nudge_requests[0]) < mock.requests.index(special_pick)
    assert mock.requests.index(special_pick) < mock.requests.index(nudge_requests[1])
    assert mock.requests.index(nudge_requests[1]) < mock.requests.index(special_place)


@pytest.mark.asyncio
async def test_task3_reverses_inspection_route_without_duplicate_boundary_navigation(
    tmp_path: Path,
) -> None:
    mock = Task3Mock([[], [], [], [], [], [], finding()])
    task_settings = settings(tmp_path)
    client = Task3Client(task_settings, transport=mock.transport)
    async with client:
        result = await Task3Orchestrator(task_settings, client).run(Task3Request())

    assert result.inspection_pass == 2
    assert len(requests_for(mock, "/perception/inspect")) == 7
    inspection_targets = [
        payload(request)["target_id"]
        for request in requests_for(mock, "/navigation/navigate")
        if payload(request)["target_id"] in {"H1_F_L_INSPECT", "H1_F_R_INSPECT"}
    ]
    assert inspection_targets[:3] == [
        "H1_F_L_INSPECT",
        "H1_F_R_INSPECT",
        "H1_F_L_INSPECT",
    ]


@pytest.mark.asyncio
async def test_task3_rejects_swap_without_two_safe_hands(tmp_path: Path) -> None:
    mock = Task3Mock([finding()])
    task_settings = settings(tmp_path, unsafe_second_slot=True)
    client = Task3Client(task_settings, transport=mock.transport)
    async with client:
        with pytest.raises(Task3ServiceError) as error:
            await Task3Orchestrator(task_settings, client).run(Task3Request())

    assert error.value.code == "NO_FEASIBLE_HAND_ASSIGNMENT"
    assert not requests_for(mock, "/pick")
    assert not requests_for(mock, "/place")


@pytest.mark.asyncio
async def test_task3_rejects_multiple_findings(tmp_path: Path) -> None:
    mock = Task3Mock([finding() + finding()])
    task_settings = settings(tmp_path)
    client = Task3Client(task_settings, transport=mock.transport)
    async with client:
        with pytest.raises(Task3ServiceError) as error:
            await Task3Orchestrator(task_settings, client).run(Task3Request())

    assert error.value.code == "INVALID_FINDINGS"
    assert not requests_for(mock, "/sku/search_by_name")


@pytest.mark.asyncio
async def test_task3_retries_pick_with_the_same_idempotency_key(tmp_path: Path) -> None:
    mock = Task3Mock([finding()])
    mock.pick_timeout_once = True
    task_settings = settings(tmp_path)
    client = Task3Client(task_settings, transport=mock.transport)
    async with client:
        await Task3Orchestrator(task_settings, client).run(
            Task3Request(), "stable-operation"
        )

    first_pick_attempts = [
        request
        for request in requests_for(mock, "/pick")
        if payload(request)["product_name"] == "错误商品"
    ]
    assert len(first_pick_attempts) == 2
    assert {request.headers["Idempotency-Key"] for request in first_pick_attempts} == {
        "stable-operation:task3.pick.0.pick"
    }


@pytest.mark.asyncio
async def test_task3_continues_other_place_without_nudge_or_retry_and_returns_start(
    tmp_path: Path,
) -> None:
    mock = Task3Mock([finding()])
    mock.place_failure_limit = 1
    task_settings = settings(tmp_path)
    client = Task3Client(task_settings, transport=mock.transport)
    async with client:
        with pytest.raises(Task3ServiceError) as error:
            await Task3Orchestrator(task_settings, client).run(
                Task3Request(), "task3-recovery"
            )

    assert error.value.code == "TASK_ACTIONS_FAILED"
    assert error.value.step == "抓放失败汇总"
    places = requests_for(mock, "/place")
    assert [payload(request)["product_name"] for request in places] == [
        "错误商品",
        "应放商品",
    ]
    assert [request.headers["Idempotency-Key"] for request in places] == [
        "task3-recovery:task3.place.0.place",
        "task3-recovery:task3.place.1.place",
    ]
    assert not requests_for(mock, "/navigation/nudge")
    navigation_targets = [
        payload(request)["target_id"]
        for request in requests_for(mock, "/navigation/navigate")
    ]
    assert navigation_targets[-1] == "start"
    assert "task_boundary" not in navigation_targets


@pytest.mark.asyncio
async def test_task3_does_not_retry_unknown_release_result(tmp_path: Path) -> None:
    mock = Task3Mock([finding()])
    mock.place_failure_limit = 1
    mock.place_failure_payload = {
        "error_code": "ACTION_RESULT_UNKNOWN",
        "message": "release result is unknown",
        "failed_interface": "manipulation_release",
        "pose": [1, 2, 3, 4, 5, 6],
    }
    task_settings = settings(tmp_path)
    client = Task3Client(task_settings, transport=mock.transport)

    async with client:
        with pytest.raises(Task3ServiceError) as error:
            await Task3Orchestrator(task_settings, client).run(Task3Request())

    assert error.value.code == "TASK_ACTIONS_FAILED"
    assert len(requests_for(mock, "/place")) == 2
    assert not requests_for(mock, "/navigation/nudge")


def test_task3_production_config_uses_task0_baselines_and_port_inputs() -> None:
    task_settings = TaskServiceSettings.load(
        CONFIG_DIR / "runtime.production.yaml"
    ).tasks.task3

    assert len(task_settings.inspection_points) == 8
    assert Path(task_settings.baseline_dir) == CONFIG_DIR.parent / "output" / "task0"
    assert task_settings.camera == "head"
    assert task_settings.services.perception == "http://127.0.0.1:8083"
    assert task_settings.services.camera.endswith(":8085")
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
        point.target_id: point.location_id for point in task_settings.inspection_points
    } == expected_locations
