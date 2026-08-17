from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

import task1_service.service as task1_service_module
from task1_service.client import Task1Client
from task1_service.models import (
    ActionResponse,
    ProductHandOptionsFile,
    Task1Request,
    Task1ServiceError,
    Task1Settings,
    Task1Timeouts,
)
from task1_service.service import Task1Orchestrator
from manipulation_policy import SPECIAL_SHELF_NUDGE_PRODUCT
from task_service.settings import TaskServiceSettings


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class Task1Mock:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.health = {"navigation": "READY", "perception": "READY", "pose": "READY", "pick_place": "READY", "sku": "READY"}
        self.names = {
            "可口可乐罐装": "H2_F_L1_C01",
            "百事可乐瓶装": "H2_F_L1_C04",
        }
        self.sku_locations: dict[str, list[str]] = {}
        self.pick_timeout_once = False
        self.pick_attempts = 0
        self.pick_failure: dict[str, object] | None = None
        self.pick_failure_limit: int | None = None
        self.place_timeout_once = False
        self.place_attempts = 0
        self.place_failure: dict[str, str] | None = None
        self.place_failure_limit: int | None = None
        self.nudge_return_failures = 0
        self.navigation_failure = False
        self.perception_failure = False
        self.resolution_failures: set[int] = set()

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
            "camera.local": "camera",
        }[host]
        path = request.url.path
        if path == "/camera/head/resolution":
            resolution = json.loads(request.content)["resolution"]
            if resolution in self.resolution_failures:
                return httpx.Response(
                    500, json={"error_code": "RESOLUTION_SWITCH_FAILED"}
                )
            return httpx.Response(200, json={"resolution": resolution})
        if path == "/" + service + "/health" or (service == "pick_place" and path == "/health"):
            payload: dict[str, object] = {"status": self.health[service]}
            if service == "pose":
                payload["current_pose"] = {"pose_type": "START_POSITION"}
            return httpx.Response(200, json=payload)
        if path == "/perception/parse":
            if self.perception_failure:
                return httpx.Response(
                    500, json={"error_code": "RECEIPT_PARSE_FAILED"}
                )
            return httpx.Response(200, json={"product_names": list(self.names)})
        if path == "/sku/search_by_name":
            name = request.url.params["name"]
            return httpx.Response(
                200,
                json={
                    "sku_id": "SKU",
                    "name": name,
                    "images": [],
                    "locations": self.sku_locations.get(name, [self.names[name]]),
                },
            )
        if path == "/sku/search_by_location":
            location = request.url.params["location"]
            name = next(name for name, slot in self.names.items() if slot == location)
            return httpx.Response(
                200,
                json={"sku_id": "SKU", "name": name, "images": [], "locations": [location]},
            )
        if path == "/pick":
            self.pick_attempts += 1
            if self.pick_failure is not None and (
                self.pick_failure_limit is None
                or self.pick_attempts <= self.pick_failure_limit
            ):
                return httpx.Response(502, json=self.pick_failure)
            if self.pick_timeout_once:
                self.pick_timeout_once = False
                raise httpx.ReadTimeout("temporary timeout", request=request)
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        if path == "/place":
            self.place_attempts += 1
            if self.place_failure is not None and (
                self.place_failure_limit is None
                or self.place_attempts <= self.place_failure_limit
            ):
                return httpx.Response(502, json=self.place_failure)
            if self.place_timeout_once:
                self.place_timeout_once = False
                raise httpx.ReadTimeout("temporary timeout", request=request)
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        if path == "/manipulation/release/both":
            self.place_attempts += 1
            if self.place_failure is not None and (
                self.place_failure_limit is None
                or self.place_attempts <= self.place_failure_limit
            ):
                return httpx.Response(502, json=self.place_failure)
            if self.place_timeout_once:
                self.place_timeout_once = False
                raise httpx.ReadTimeout("temporary timeout", request=request)
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        if path == "/navigation/nudge":
            request_payload = json.loads(request.content)
            if (
                request_payload.get("action") == "return"
                and self.nudge_return_failures > 0
            ):
                self.nudge_return_failures -= 1
                return httpx.Response(500, json={"error_code": "RETURN_FAILED"})
            return httpx.Response(
                200,
                json={
                    "status": "SUCCEEDED",
                    "station_id": "task-boundary",
                    "nudge_count": 0 if request_payload.get("action") == "return" else 1,
                },
            )
        if path == "/navigation/navigate" and self.navigation_failure:
            return httpx.Response(500, json={"error_code": "NAVIGATION_FAILED"})
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
            "camera": "http://camera.local",
        },
        timeouts=Task1Timeouts(
            connect_seconds=0.1,
            health_seconds=0.2,
            receipt_seconds=0.2,
            resolution_seconds=0.2,
            sku_seconds=0.2,
            navigation_seconds=0.2,
            pose_seconds=0.2,
            pick_seconds=0.2,
            place_seconds=0.2,
        ),
        product_hand_options={
            "H2_F_L1_C01": ["LEFT", "RIGHT"],
            "H2_F_L1_C04": ["LEFT", "RIGHT"],
        },
    )


@pytest.fixture(autouse=True)
def disable_receipt_exposure_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        task1_service_module, "RECEIPT_EXPOSURE_SETTLE_SECONDS", 0.0
    )


def paths(mock: Task1Mock) -> list[str]:
    return [request.url.path for request in mock.requests]


def payload(request: httpx.Request) -> dict:
    return json.loads(request.content)


@pytest.mark.asyncio
async def test_sku_requests_use_query_parameters_without_get_bodies() -> None:
    mock = Task1Mock()
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        await client.search_by_name("可口可乐罐装")
        await client.search_by_location("H2_F_L1_C04")

    sku_requests = [request for request in mock.requests if request.url.path.startswith("/sku/search_by_")]
    assert dict(sku_requests[0].url.params) == {"name": "可口可乐罐装"}
    assert dict(sku_requests[1].url.params) == {"location": "H2_F_L1_C04"}
    assert all(request.content == b"" for request in sku_requests)


def test_production_config_loads_complete_product_hand_options() -> None:
    options_path = CONFIG_DIR / "product-hand-options.yaml"
    with options_path.open("r", encoding="utf-8") as options_file:
        options = ProductHandOptionsFile.model_validate(yaml.safe_load(options_file))

    assert len(options.product_hand_options) == 122
    assert options.product_hand_options["H1_F_L1_C01"].product_name == "NFC桔汁"
    assert options.product_hand_options["H2_B_L5_C02"].product_name == "心相印厨房纸巾"

    task_settings = TaskServiceSettings.load(
        CONFIG_DIR / "runtime.production.yaml"
    ).tasks.task1
    assert len(task_settings.product_hand_options) == 122
    assert task_settings.product_hand_options["H1_F_L1_C01"] == ["LEFT"]
    assert task_settings.product_target_ids["H1_F_L1_C01"] == "H1_F_L_INSPECT"
    assert task_settings.product_target_ids["H1_F_L1_C04"] == "H1_F_R_INSPECT"
    assert task_settings.services.pose.endswith(":8084")


def test_action_response_accepts_pose_execution_metadata() -> None:
    response = ActionResponse.model_validate(
        {
            "status": "SUCCEEDED",
            "executed": True,
            "current_pose": {"pose_type": "RECEIPT_VIEW"},
        }
    )

    assert response.status == "SUCCEEDED"


@pytest.mark.asyncio
async def test_health_accepts_pose_metadata() -> None:
    mock = Task1Mock()
    client = Task1Client(settings(), transport=mock.transport)

    async with client:
        await client.check_all_health()


@pytest.mark.asyncio
async def test_head_resolution_client_uses_camera_endpoint() -> None:
    mock = Task1Mock()
    client = Task1Client(settings(), transport=mock.transport)

    async with client:
        await client.set_head_resolution(1080)
        await client.set_head_resolution(720)

    resolution_requests = [
        request
        for request in mock.requests
        if request.url.path == "/camera/head/resolution"
    ]
    assert [request.method for request in resolution_requests] == ["POST", "POST"]
    assert [payload(request) for request in resolution_requests] == [
        {"resolution": 1080},
        {"resolution": 720},
    ]


@pytest.mark.asyncio
async def test_task1_runs_full_pick_place_flow() -> None:
    mock = Task1Mock()
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        result = await Task1Orchestrator(settings(), client).run(Task1Request())

    assert result.status == "SUCCEEDED"
    assert result.target_items[0].product_slot_id == "H2_F_L1_C01"
    assert [item.picked for item in result.target_items] == [True, True]
    assert [item.placed for item in result.target_items] == [True, True]
    assert result.held_items == {}
    assert [payload(request)["hand"] for request in mock.requests if request.url.path == "/pick"] == ["LEFT", "RIGHT"]
    assert [payload(request)["level"] for request in mock.requests if request.url.path == "/pick"] == ["L1", "L1"]
    [release_both] = [
        request
        for request in mock.requests
        if request.url.path == "/manipulation/release/both"
    ]
    assert payload(release_both) == {
        "task_type": "SORTING",
        "left": {"product_name": "可口可乐罐装"},
        "right": {"product_name": "百事可乐瓶装"},
    }
    assert not [request for request in mock.requests if request.url.path == "/place"]
    navigation = [payload(request).get("target_id") for request in mock.requests if request.url.path == "/navigation/navigate"]
    assert navigation[-2:] == ["delivery_place", "task_boundary"]

    resolution_requests = [
        request
        for request in mock.requests
        if request.url.path == "/camera/head/resolution"
    ]
    assert [payload(request)["resolution"] for request in resolution_requests] == [
        1080,
        720,
    ]
    switch_index = mock.requests.index(resolution_requests[0])
    receipt_navigation_index = next(
        index
        for index, request in enumerate(mock.requests)
        if request.url.path == "/navigation/navigate"
        and payload(request).get("target_id") == "receipt_viewpoint"
    )
    receipt_pose_index = next(
        index
        for index, request in enumerate(mock.requests)
        if request.url.path == "/pose/prepare"
        and payload(request).get("pose_type") == "RECEIPT_VIEW"
    )
    parse_index = paths(mock).index("/perception/parse")
    restore_index = mock.requests.index(resolution_requests[1])
    sku_index = paths(mock).index("/sku/search_by_name")
    assert (
        switch_index
        < receipt_navigation_index
        < receipt_pose_index
        < parse_index
        < restore_index
        < sku_index
    )


@pytest.mark.asyncio
async def test_task1_only_waits_for_unelapsed_exposure_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock = Task1Mock()
    waited: list[float] = []
    times = iter((10.0, 10.2))

    async def fake_sleep(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(
        task1_service_module, "RECEIPT_EXPOSURE_SETTLE_SECONDS", 0.5
    )
    monkeypatch.setattr(task1_service_module, "monotonic", lambda: next(times))
    monkeypatch.setattr(task1_service_module, "sleep", fake_sleep)
    client = Task1Client(settings(), transport=mock.transport)

    async with client:
        await Task1Orchestrator(settings(), client).run(Task1Request())

    assert waited == [pytest.approx(0.3)]


@pytest.mark.asyncio
async def test_task1_restores_720_when_receipt_parse_fails() -> None:
    mock = Task1Mock()
    mock.perception_failure = True
    client = Task1Client(settings(), transport=mock.transport)

    async with client:
        with pytest.raises(Task1ServiceError) as error:
            await Task1Orchestrator(settings(), client).run(Task1Request())

    assert error.value.code == "RECEIPT_PARSE_FAILED"
    assert [
        payload(request)["resolution"]
        for request in mock.requests
        if request.url.path == "/camera/head/resolution"
    ] == [1080, 720]
    assert "/sku/search_by_name" not in paths(mock)
    assert "/pick" not in paths(mock)


@pytest.mark.asyncio
async def test_task1_uses_current_resolution_when_1080_switch_fails() -> None:
    mock = Task1Mock()
    mock.resolution_failures.add(1080)
    client = Task1Client(settings(), transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(settings(), client).run(Task1Request())

    assert result.status == "SUCCEEDED"
    assert [
        payload(request)["resolution"]
        for request in mock.requests
        if request.url.path == "/camera/head/resolution"
    ] == [1080, 720]
    assert [
        request
        for request in mock.requests
        if request.url.path == "/navigation/navigate"
        and payload(request).get("target_id") == "receipt_viewpoint"
    ]
    assert "/perception/parse" in paths(mock)


@pytest.mark.asyncio
async def test_task1_continues_when_720_restore_fails() -> None:
    mock = Task1Mock()
    mock.resolution_failures.add(720)
    client = Task1Client(settings(), transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(settings(), client).run(Task1Request())

    assert result.status == "SUCCEEDED"
    assert "/sku/search_by_name" in paths(mock)
    assert "/pick" in paths(mock)


@pytest.mark.asyncio
async def test_task1_uses_first_location_when_sku_has_multiple_locations() -> None:
    mock = Task1Mock()
    mock.sku_locations["可口可乐罐装"] = ["H2_F_L1_C01", "H2_F_L1_C03"]
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        result = await Task1Orchestrator(settings(), client).run(Task1Request())

    assert result.target_items[0].product_slot_id == "H2_F_L1_C01"
    assert result.target_items[0].picked is True


@pytest.mark.asyncio
async def test_task1_navigates_by_inspection_target_and_reuses_same_target() -> None:
    mock = Task1Mock()
    mock.names = {
        "可口可乐罐装": "H2_F_L1_C01",
        "雪碧罐装": "H2_F_L1_C02",
    }
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        result = await Task1Orchestrator(settings(), client).run(Task1Request())

    assert [item.target_id for item in result.target_items] == [
        "H2_F_L_INSPECT",
        "H2_F_L_INSPECT",
    ]
    navigation = [
        payload(request)["target_id"]
        for request in mock.requests
        if request.url.path == "/navigation/navigate"
    ]
    assert navigation == [
        "receipt_viewpoint",
        "H2_F_L_INSPECT",
        "delivery_place",
        "task_boundary",
    ]
    resets = [
        request
        for request in mock.requests
        if request.url.path == "/pose/prepare"
        and payload(request) == {"pose_type": "START_POSITION"}
    ]
    assert len(resets) == len(navigation)


@pytest.mark.asyncio
async def test_task1_resets_pose_immediately_before_every_navigation() -> None:
    mock = Task1Mock()
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        await Task1Orchestrator(settings(), client).run(Task1Request())

    navigation_indexes = [
        index
        for index, request in enumerate(mock.requests)
        if request.url.path == "/navigation/navigate"
    ]
    reset_requests = [
        request
        for request in mock.requests
        if request.url.path == "/pose/prepare"
        and payload(request) == {"pose_type": "START_POSITION"}
    ]

    assert len(reset_requests) == len(navigation_indexes)
    assert all(
        mock.requests[index - 1].url.path == "/pose/prepare"
        and payload(mock.requests[index - 1]) == {"pose_type": "START_POSITION"}
        for index in navigation_indexes
    )


@pytest.mark.asyncio
async def test_task1_picks_both_products_and_places_them() -> None:
    mock = Task1Mock()
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        result = await Task1Orchestrator(settings(), client).run(Task1Request())

    assert [item.picked for item in result.target_items] == [True, True]
    assert result.held_items == {}
    pick_requests = [request for request in mock.requests if request.url.path == "/pick"]
    assert [payload(request)["hand"] for request in pick_requests] == ["LEFT", "RIGHT"]
    assert [payload(request)["product_name"] for request in pick_requests] == list(mock.names)
    release_both = [
        request
        for request in mock.requests
        if request.url.path == "/manipulation/release/both"
    ]
    assert len(release_both) == 1
    assert release_both[0].headers["Idempotency-Key"].endswith(
        ":task1.place.both"
    )


@pytest.mark.asyncio
async def test_task1_same_hand_is_strictly_serial(tmp_path) -> None:
    mock = Task1Mock()
    task_settings = settings().model_copy(update={
        "log_dir": str(tmp_path),
        "product_hand_options": {
            "H2_F_L1_C01": ["LEFT"],
            "H2_F_L1_C04": ["LEFT"],
        },
    })
    client = Task1Client(task_settings, transport=mock.transport)
    async with client:
        result = await Task1Orchestrator(task_settings, client).run(Task1Request())

    assert result.held_items == {}
    actions = [(request.url.path, payload(request).get("target_id"), payload(request).get("product_name")) for request in mock.requests if request.url.path in {"/navigation/navigate", "/pick", "/place"}]
    assert actions.index(("/place", None, "可口可乐罐装")) < actions.index(("/pick", None, "百事可乐瓶装"))
    assert [payload(request)["hand"] for request in mock.requests if request.url.path in {"/pick", "/place"}] == ["LEFT", "LEFT", "LEFT", "LEFT"]


@pytest.mark.asyncio
async def test_task1_special_product_left_pick_nudges_right_before_first_attempt() -> None:
    mock = Task1Mock()
    mock.names = {
        SPECIAL_SHELF_NUDGE_PRODUCT: "H2_F_L1_C01",
        "百事可乐瓶装": "H2_F_L1_C04",
    }
    mock.pick_failure = {
        "error_code": "EXECUTION_FAILED",
        "message": "grasp failed",
        "failed_interface": "manipulation_grasp",
        "pose": [-1, 2, 3, 4, 5, 6],
    }
    mock.pick_failure_limit = 1
    client = Task1Client(settings(), transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(settings(), client).run(Task1Request())

    assert result.status == "SUCCEEDED"
    nudge_requests = [
        request for request in mock.requests if request.url.path == "/navigation/nudge"
    ]
    assert [payload(request) for request in nudge_requests] == [
        {"action": "approach", "direction": "right"},
        {"action": "approach", "direction": "left"},
        {"action": "return"},
    ]
    special_picks = [
        request
        for request in mock.requests
        if request.url.path == "/pick"
        and payload(request)["product_name"] == SPECIAL_SHELF_NUDGE_PRODUCT
    ]
    assert len(special_picks) == 2
    assert mock.requests.index(nudge_requests[0]) < mock.requests.index(special_picks[0])
    assert mock.requests.index(special_picks[0]) < mock.requests.index(nudge_requests[1])
    assert mock.requests.index(nudge_requests[1]) < mock.requests.index(special_picks[1])
    assert mock.requests.index(special_picks[1]) < mock.requests.index(nudge_requests[2])


@pytest.mark.asyncio
async def test_task1_recovers_grasp_after_left_nudge_and_second_return() -> None:
    mock = Task1Mock()
    mock.pick_failure = {
        "error_code": "EXECUTION_FAILED",
        "message": "grasp failed",
        "failed_interface": "manipulation_grasp",
        "pose": [-1, 2, 3, 4, 5, 6],
    }
    mock.pick_failure_limit = 1
    mock.nudge_return_failures = 1
    client = Task1Client(settings(), transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(settings(), client).run(
            Task1Request(), "task1-recovery"
        )

    assert result.status == "SUCCEEDED"
    nudge_requests = [
        request for request in mock.requests if request.url.path == "/navigation/nudge"
    ]
    assert [payload(request) for request in nudge_requests] == [
        {"action": "approach", "direction": "left"},
        {"action": "return"},
        {"action": "return"},
    ]
    assert [request.headers["Idempotency-Key"] for request in nudge_requests] == [
        "task1-recovery:task1.pick.0.pick:recovery.approach",
        "task1-recovery:task1.pick.0.pick:recovery.return.1",
        "task1-recovery:task1.pick.0.pick:recovery.return.2",
    ]
    first_product_picks = [
        request
        for request in mock.requests
        if request.url.path == "/pick"
        and payload(request)["product_name"] == "可口可乐罐装"
    ]
    assert [request.headers["Idempotency-Key"] for request in first_product_picks] == [
        "task1-recovery:task1.pick.0.pick",
        "task1-recovery:task1.pick.0.pick:recovery.retry",
    ]


@pytest.mark.asyncio
async def test_task1_stops_after_two_nudge_return_failures_and_navigates_start() -> None:
    mock = Task1Mock()
    mock.pick_failure = {
        "error_code": "EXECUTION_FAILED",
        "message": "grasp failed",
        "failed_interface": "manipulation_grasp",
        "pose": [1, 2, 3, 4, 5, 6],
    }
    mock.pick_failure_limit = 1
    mock.nudge_return_failures = 2
    client = Task1Client(settings(), transport=mock.transport)

    async with client:
        with pytest.raises(Task1ServiceError) as error:
            await Task1Orchestrator(settings(), client).run(
                Task1Request(), "task1-return-failure"
            )

    assert error.value.code == "NUDGE_RETURN_FAILED"
    nudge_requests = [
        request for request in mock.requests if request.url.path == "/navigation/nudge"
    ]
    assert [payload(request) for request in nudge_requests] == [
        {"action": "approach", "direction": "right"},
        {"action": "return"},
        {"action": "return"},
    ]
    pick_requests = [request for request in mock.requests if request.url.path == "/pick"]
    assert [payload(request)["product_name"] for request in pick_requests] == [
        "可口可乐罐装",
        "可口可乐罐装",
    ]
    navigation_targets = [
        payload(request)["target_id"]
        for request in mock.requests
        if request.url.path == "/navigation/navigate"
    ]
    assert navigation_targets[-1] == "start"


@pytest.mark.asyncio
async def test_task1_reports_failed_release_after_returning_start(tmp_path: Path) -> None:
    mock = Task1Mock()
    mock.place_failure = {
        "error_code": "EXECUTION_FAILED",
        "message": "/manipulation/release: MoveL 规划失败，错误码 -1022",
        "failed_interface": "manipulation_release",
        "url": "http://robot:8084/manipulation/release",
    }
    mock.place_failure_limit = 1
    task_settings = settings().model_copy(update={"log_dir": str(tmp_path)})
    client = Task1Client(task_settings, transport=mock.transport)

    async with client:
        with pytest.raises(Task1ServiceError) as error:
            await Task1Orchestrator(task_settings, client).run(Task1Request())

    assert error.value.code == "TASK_ACTIONS_FAILED"
    assert error.value.step == "抓放失败汇总"
    navigation_targets = [
        payload(request)["target_id"]
        for request in mock.requests
        if request.url.path == "/navigation/navigate"
    ]
    assert navigation_targets[-1] == "start"
    assert "task_boundary" not in navigation_targets
    [log_dir] = list(tmp_path.iterdir())
    events = [
        json.loads(line)
        for line in (log_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failed_event = next(
        event
        for event in events
        if event["event"] == "双手放置"
        and event["status"] == "failed"
        and event["attempt"] == 1
    )
    assert failed_event["failed_interface"] == "manipulation_release"
    assert failed_event["url"] == "http://robot:8084/manipulation/release"
    assert not [
        request for request in mock.requests if request.url.path == "/navigation/nudge"
    ]
    assert events[-2]["event"] == "失败回开始点"
    assert events[-2]["status"] == "succeeded"
    assert events[-1]["error_code"] == "TASK_ACTIONS_FAILED"


@pytest.mark.asyncio
async def test_task1_health_failure_returns_start() -> None:
    mock = Task1Mock()
    mock.health["navigation"] = "ERROR"
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        with pytest.raises(Task1ServiceError) as error:
            await Task1Orchestrator(settings(), client).run(Task1Request())

    assert error.value.code == "CAPABILITY_NOT_READY"
    assert "/navigation/health" in paths(mock)
    navigation_requests = [
        request for request in mock.requests if request.url.path == "/navigation/navigate"
    ]
    assert [payload(request)["target_id"] for request in navigation_requests] == ["start"]


@pytest.mark.asyncio
async def test_task1_reports_failure_recovery_error_when_start_navigation_fails() -> None:
    mock = Task1Mock()
    mock.health["perception"] = "ERROR"
    mock.navigation_failure = True
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        with pytest.raises(Task1ServiceError) as error:
            await Task1Orchestrator(settings(), client).run(Task1Request())

    assert error.value.code == "FAILURE_RECOVERY_FAILED"
    assert error.value.step == "失败回开始点"
    assert "健康检查" in error.value.message


@pytest.mark.asyncio
async def test_task1_retries_pick_with_same_idempotency_key() -> None:
    mock = Task1Mock()
    mock.pick_timeout_once = True
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        await Task1Orchestrator(settings(), client).run(Task1Request())

    pick_requests = [request for request in mock.requests if request.url.path == "/pick"]
    assert mock.pick_attempts == 3
    assert len(pick_requests) == 3
    assert pick_requests[0].headers["Idempotency-Key"] == pick_requests[1].headers["Idempotency-Key"]


@pytest.mark.asyncio
async def test_task1_writes_pickplace_style_operation_log(tmp_path) -> None:
    mock = Task1Mock()
    task_settings = settings().model_copy(update={"log_dir": str(tmp_path)})
    client = Task1Client(task_settings, transport=mock.transport)
    async with client:
        await Task1Orchestrator(task_settings, client).run(Task1Request(), "web-task1-log")

    directories = list(tmp_path.iterdir())
    assert len(directories) == 1
    assert (directories[0] / "operation.json").exists()
    events = (directories[0] / "events.jsonl").read_text(encoding="utf-8")
    assert '"event": "小票识别"' in events
    assert '"event": "SKU货位转换"' in events
    assert '"event": "抓取"' in events
    assert '"event": "operation"' in events
