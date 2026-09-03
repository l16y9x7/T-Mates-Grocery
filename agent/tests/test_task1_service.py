from __future__ import annotations

import asyncio
import json
from pathlib import Path
from random import Random

import httpx
import pytest
import yaml

import task1_service.service as task1_service_module
from task1_service.client import Task1Client
from task1_service.mock_order import MockOrderSystem
from task1_service.models import (
    ActionResponse,
    GraspOption,
    Hand,
    ProductHandOptionsFile,
    TargetItem,
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
        self.health = {"navigation": "READY", "pose": "READY", "pick_place": "READY", "sku": "READY"}
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
        self.pose_failure_keys: set[str] = set()
        self.navigation_failure = False
        self.gripper_requested_position = 255
        self.gripper_position = 216

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        service = {
            "navigation.local": "navigation",
            "pose.local": "pose",
            "pick-place.local": "pick_place",
            "sku.local": "sku",
        }[host]
        path = request.url.path
        if path == "/" + service + "/health" or (service == "pick_place" and path == "/health"):
            payload: dict[str, object] = {"status": self.health[service]}
            if service == "pose":
                payload["current_pose"] = {"pose_type": "START_POSITION"}
            return httpx.Response(200, json=payload)
        if path == "/sku/get_all_names":
            return httpx.Response(200, json=list(self.names))
        if path == "/sku/search_by_name":
            name = request.url.params["name"]
            locations = self.sku_locations.get(name, [self.names[name]])
            return httpx.Response(
                200,
                json={
                    "sku_id": "SKU",
                    "name": name,
                    "images": [],
                    "locations": locations,
                    "inventory": locations,
                },
            )
        if path == "/sku/search_by_location":
            location = request.url.params["location"]
            name = next(name for name, slot in self.names.items() if slot == location)
            return httpx.Response(
                200,
                json={
                    "sku_id": "SKU",
                    "name": name,
                    "images": [],
                    "locations": [location],
                    "inventory": [location],
                },
            )
        if path == "/sku/modify_inventory":
            request_payload = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "slot_id": request_payload["slot_id"],
                    "modification": "deplete",
                    "modified": True,
                },
            )
        if path == "/manipulation/gripper/status":
            return httpx.Response(
                200,
                json={
                    "status": "SUCCEEDED",
                    "hand": request.url.params["hand"],
                    "gripper": {
                        "requested_position": self.gripper_requested_position,
                        "position": self.gripper_position,
                    },
                },
            )
        if path == "/pick/both":
            self.pick_attempts += 2
            request_payload = json.loads(request.content)

            def succeeded(side: str) -> dict[str, object]:
                item = request_payload[side]
                return {
                    "status": "SUCCEEDED",
                    "product_name": item["product_name"],
                    "hand": side,
                    "error_code": None,
                    "message": None,
                    "failed_interface": None,
                    "url": None,
                    "pose": None,
                }

            return httpx.Response(
                200,
                json={
                    "status": "SUCCEEDED",
                    "left": succeeded("left"),
                    "right": succeeded("right"),
                },
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
        if (
            path == "/pose/prepare"
            and request.headers.get("Idempotency-Key") in self.pose_failure_keys
        ):
            return httpx.Response(500, json={"error_code": "POSE_PREPARE_FAILED"})
        if path in {"/navigation/navigate", "/pose/prepare"}:
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        return httpx.Response(404, json={"error_code": "UNKNOWN_ENDPOINT"})


class PickConcurrencyMock(Task1Mock):
    """Return configurable per-hand results for the paired initial pick."""

    def __init__(self) -> None:
        super().__init__()
        self.active_pick_requests = 0
        self.max_active_pick_requests = 0
        self.initial_pick_failures: dict[str, dict[str, object]] = {}

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/pick/both":
            self.requests.append(request)
            request_payload = payload(request)
            sides: dict[str, dict[str, object]] = {}
            for request_index, side in enumerate(("left", "right"), start=1):
                item = request_payload[side]
                failure = self.initial_pick_failures.get(item["hand"])
                if failure is None and self.pick_failure is not None and (
                    self.pick_failure_limit is None
                    or request_index <= self.pick_failure_limit
                ):
                    failure = self.pick_failure
                if failure is None:
                    sides[side] = {
                        "status": "SUCCEEDED",
                        "product_name": item["product_name"],
                        "hand": side,
                        "error_code": None,
                        "message": None,
                        "failed_interface": None,
                        "url": None,
                        "pose": None,
                    }
                    continue
                error_code = failure.get("error_code", "EXECUTION_FAILED")
                sides[side] = {
                    "status": (
                        "UNKNOWN"
                        if error_code
                        in {"ACTION_RESULT_UNKNOWN", "NETWORK_ERROR", "INVALID_RESPONSE"}
                        else "FAILED"
                    ),
                    "product_name": item["product_name"],
                    "hand": side,
                    "error_code": error_code,
                    "message": failure.get("message", "pick failed"),
                    "failed_interface": failure.get("failed_interface"),
                    "url": failure.get("url"),
                    "pose": failure.get("pose"),
                }
            self.pick_attempts += 2
            statuses = {item["status"] for item in sides.values()}
            overall = (
                "SUCCEEDED"
                if statuses == {"SUCCEEDED"}
                else "UNKNOWN"
                if "UNKNOWN" in statuses
                else "PARTIAL"
                if "SUCCEEDED" in statuses
                else "FAILED"
            )
            return httpx.Response(200, json={"status": overall, **sides})

        if request.url.path != "/pick":
            return await super().handle(request)

        self.requests.append(request)
        self.pick_attempts += 1
        request_index = self.pick_attempts
        self.active_pick_requests += 1
        self.max_active_pick_requests = max(
            self.max_active_pick_requests,
            self.active_pick_requests,
        )
        try:
            if request_index == 1:
                # Yield once: a concurrently scheduled second request enters
                # before this one completes; a serial caller remains at one.
                await asyncio.sleep(0)
            if self.pick_failure is not None and (
                self.pick_failure_limit is None
                or request_index <= self.pick_failure_limit
            ):
                return httpx.Response(502, json=self.pick_failure)
            request_hand = payload(request)["hand"]
            if ":recovery." not in request.headers["Idempotency-Key"]:
                failure = self.initial_pick_failures.get(request_hand)
                if failure is not None:
                    return httpx.Response(502, json=failure)
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        finally:
            self.active_pick_requests -= 1


def settings() -> Task1Settings:
    return Task1Settings(
        services={
            "navigation": "http://navigation.local",
            "pose": "http://pose.local",
            "pick_place": "http://pick-place.local",
            "sku": "http://sku.local",
        },
        timeouts=Task1Timeouts(
            connect_seconds=0.1,
            health_seconds=0.2,
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


@pytest.mark.asyncio
async def test_gripper_threshold_and_bagged_sku_bypass() -> None:
    mock = Task1Mock()
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        assert await client.is_object_grasped(Hand.LEFT, "SKU_001") is True
        mock.gripper_requested_position = 217
        assert await client.is_object_grasped(Hand.LEFT, "SKU_001") is False
        request_count = len(mock.requests)
        assert await client.is_object_grasped(Hand.RIGHT, "SKU_107") is True

    assert len(mock.requests) == request_count


def test_production_config_loads_complete_product_hand_options() -> None:
    options_path = CONFIG_DIR / "product-hand-options.yaml"
    with options_path.open("r", encoding="utf-8") as options_file:
        options = ProductHandOptionsFile.model_validate(yaml.safe_load(options_file))

    assert len(options.product_hand_options) == 74
    assert options.product_hand_options["H3_L01_C01"].product_name == "NFC桔汁"
    assert options.product_hand_options["H1_L04_C02"].product_name == "心相印厨房纸巾"

    task_settings = TaskServiceSettings.load(
        CONFIG_DIR / "runtime.production.yaml"
    ).tasks.task1
    assert len(task_settings.product_hand_options) == 74
    assert task_settings.product_hand_options["H3_L01_C01"] == ["LEFT", "RIGHT"]
    assert task_settings.product_target_ids["H3_L01_C01"] == "H3_INSPECT"
    assert len(task_settings.product_grasp_options["H1_L01_C04"]) == 2
    assert [
        (option.target_id, [hand.value for hand in option.hands])
        for option in task_settings.product_grasp_options["H2_L03_C03"]
    ] == [
        ("H2_INSPECT", ["LEFT"]),
        ("H12_INSPECT", ["RIGHT"]),
    ]
    assert [
        (option.target_id, [hand.value for hand in option.hands])
        for option in task_settings.product_grasp_options["H2_L04_C03"]
    ] == [
        ("H2_INSPECT", ["LEFT"]),
        ("H12_INSPECT", ["RIGHT"]),
    ]
    assert [
        (option.target_id, [hand.value for hand in option.hands])
        for option in task_settings.product_grasp_options["H2_L04_C04"]
    ] == [
        ("H2_INSPECT", ["RIGHT"]),
        ("H23_INSPECT", ["LEFT"]),
    ]
    assert task_settings.services.pose.endswith(":8084")

    catalog = json.loads(
        (CONFIG_DIR.parents[1] / "perception" / "sku" / "products.json").read_text(
            encoding="utf-8"
        )
    )["products"]
    assert len(catalog) == 43
    assert all(
        any(
            task_settings.product_grasp_options.get(location)
            for location in product["locations"]
        )
        for product in catalog
    )


def test_action_response_accepts_pose_execution_metadata() -> None:
    response = ActionResponse.model_validate(
        {
            "status": "SUCCEEDED",
            "executed": True,
            "current_pose": {"pose_type": "START_POSITION"},
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
async def test_mock_order_catalog_uses_active_sku_service() -> None:
    mock = Task1Mock()
    client = Task1Client(settings(), transport=mock.transport)

    async with client:
        names = await client.list_product_names()

    assert names == list(mock.names)
    assert paths(mock) == ["/sku/get_all_names"]


@pytest.mark.asyncio
async def test_mock_order_selects_two_distinct_products_from_43_skus() -> None:
    catalog = [f"商品-{index:02d}" for index in range(43)]

    async def load_catalog() -> list[str]:
        return catalog

    order = await MockOrderSystem(load_catalog, rng=Random(7)).create_order()

    assert order.catalog_size == 43
    assert len(order.product_names) == 2
    assert len(set(order.product_names)) == 2
    assert set(order.product_names) <= set(catalog)
    assert order.available_product_names == catalog


@pytest.mark.asyncio
async def test_interface_metrics_count_remote_protocol_errors() -> None:
    async def fail_with_protocol_error(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("connection closed", request=request)

    client = Task1Client(
        settings(), transport=httpx.MockTransport(fail_with_protocol_error)
    )
    events: list[dict[str, object]] = []
    client.set_trace_callback(events.append)
    async with client:
        with pytest.raises(Task1ServiceError) as error:
            await client.list_product_names()

    assert error.value.code == "NETWORK_ERROR"
    assert len(events) == 2
    assert [event["attempt"] for event in events] == [1, 2]
    assert len({event["call_id"] for event in events}) == 2
    assert all(event["duration_ms"] >= 0 for event in events)
    [metric] = client.interface_metrics()
    assert metric.call_count == 2
    assert metric.failure_count == 2


@pytest.mark.asyncio
async def test_interface_metrics_count_cancelled_in_flight_request() -> None:
    request_started = asyncio.Event()
    never_finish = asyncio.Event()

    async def wait_forever(request: httpx.Request) -> httpx.Response:
        request_started.set()
        await never_finish.wait()
        return httpx.Response(200, json=[])

    client = Task1Client(settings(), transport=httpx.MockTransport(wait_forever))
    events: list[dict[str, object]] = []
    client.set_trace_callback(events.append)
    async with client:
        request_task = asyncio.create_task(client.list_product_names())
        await request_started.wait()
        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task

    assert len(events) == 1
    assert events[0]["attempt"] == 1
    assert events[0]["error"] == "request cancelled"
    assert events[0]["duration_ms"] >= 0
    [metric] = client.interface_metrics()
    assert metric.call_count == 1
    assert metric.failure_count == 1


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
    pick_requests = [request for request in mock.requests if request.url.path == "/pick"]
    assert [payload(request)["slot_id"] for request in pick_requests] == [
        result.target_items[0].product_slot_id,
        result.target_items[1].product_slot_id,
    ]
    assert [payload(request)["location_id"] for request in pick_requests] == [
        result.target_items[0].target_id,
        result.target_items[1].target_id,
    ]
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
    assert result.order is not None
    assert result.order.source == "mock_random"
    assert result.order.catalog_size == 2
    assert result.order.product_names == list(mock.names)
    assert "/sku/get_all_names" in paths(mock)
    gripper_requests = [
        request
        for request in mock.requests
        if request.url.path == "/manipulation/gripper/status"
    ]
    assert len(gripper_requests) == 2
    assert {
        payload(r)["slot_id"]
        for r in mock.requests
        if r.url.path == "/sku/modify_inventory"
    } == {item.product_slot_id for item in result.target_items}


@pytest.mark.asyncio
async def test_task1_forwards_exact_connector_slot_target_and_hand() -> None:
    fixed_left_product = "固定左手商品"
    connector_product = "外星人电解质水白桃口味0糖"
    fixed_left_slot = "H2_L03_C01"
    connector_slot = "H2_L03_C03"
    mock = Task1Mock()
    mock.names = {
        fixed_left_product: fixed_left_slot,
        connector_product: connector_slot,
    }
    task_settings = settings().model_copy(
        update={
            "product_grasp_options": {
                fixed_left_slot: [
                    GraspOption(hands=[Hand.LEFT], target_id="H2_INSPECT")
                ],
                connector_slot: [
                    GraspOption(hands=[Hand.LEFT], target_id="H2_INSPECT"),
                    GraspOption(hands=[Hand.RIGHT], target_id="H12_INSPECT"),
                ],
            }
        }
    )
    client = Task1Client(task_settings, transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(task_settings, client).run(
            Task1Request(
                order_id="connector-context-test",
                product_names=[fixed_left_product, connector_product],
            )
        )

    assert [item.product_slot_id for item in result.target_items] == [
        fixed_left_slot,
        connector_slot,
    ]
    assert [item.hand for item in result.target_items] == [Hand.LEFT, Hand.RIGHT]
    assert [item.target_id for item in result.target_items] == [
        "H2_INSPECT",
        "H12_INSPECT",
    ]
    connector_pick = next(
        request
        for request in mock.requests
        if request.url.path == "/pick"
        and payload(request)["product_name"] == connector_product
    )
    assert payload(connector_pick) == {
        "task_type": "SORTING",
        "product_name": connector_product,
        "hand": "RIGHT",
        "level": "L3",
        "slot_id": connector_slot,
        "location_id": "H12_INSPECT",
    }
    assert "/perception/parse" not in paths(mock)
    assert "/camera/head/resolution" not in paths(mock)
    assert all(
        payload(request).get("target_id") != "receipt_viewpoint"
        for request in mock.requests
        if request.url.path == "/navigation/navigate"
    )
    metrics = {metric.interface: metric for metric in result.interface_metrics}
    assert metrics["sku/sku/get_all_names"].call_count == 1
    assert metrics["sku/sku/get_all_names"].success_count == 1
    assert metrics["sku/sku/get_all_names"].total_duration_ms >= 0


@pytest.mark.asyncio
async def test_task1_executes_the_order_previewed_in_the_web_console() -> None:
    mock = Task1Mock()
    client = Task1Client(settings(), transport=mock.transport)
    request = Task1Request(
        order_id="preview-order",
        product_names=["百事可乐瓶装", "可口可乐罐装"],
    )

    async with client:
        result = await Task1Orchestrator(settings(), client).run(request)

    assert result.product_names == ["百事可乐瓶装", "可口可乐罐装"]
    assert result.order is not None
    assert result.order.order_id == "preview-order"
    assert [target.product_name for target in result.target_items] == result.product_names


@pytest.mark.parametrize(
    "product_names",
    [[], ["可口可乐罐装"], ["可口可乐罐装", "可口可乐罐装"]],
)
def test_task1_request_rejects_invalid_mock_orders(product_names: list[str]) -> None:
    with pytest.raises(ValueError):
        Task1Request(product_names=product_names)


def test_task1_request_rejects_blank_mock_order_id() -> None:
    with pytest.raises(ValueError):
        Task1Request(order_id="   ")


@pytest.mark.parametrize(
    "payload",
    [
        {"order_id": "preview-order"},
        {"product_names": ["可口可乐罐装", "百事可乐瓶装"]},
    ],
)
def test_task1_request_requires_complete_preview_order(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Task1Request.model_validate(payload)


def test_task1_settings_ignore_removed_receipt_stage_keys() -> None:
    raw_settings = settings().model_dump(mode="json")
    raw_settings["receipt_viewpoint"] = "receipt_viewpoint"
    raw_settings["services"].update(
        perception="http://perception.local",
        camera="http://camera.local",
    )
    raw_settings["timeouts"].update(
        receipt_seconds=30,
        resolution_seconds=10,
    )

    migrated = Task1Settings.from_mapping(raw_settings, CONFIG_DIR)

    assert migrated == settings()


@pytest.mark.asyncio
async def test_task1_uses_first_location_when_sku_has_multiple_locations() -> None:
    mock = Task1Mock()
    mock.sku_locations["可口可乐罐装"] = ["H2_F_L1_C01", "H2_F_L1_C03"]
    client = Task1Client(settings(), transport=mock.transport)
    async with client:
        result = await Task1Orchestrator(settings(), client).run(Task1Request())

    assert result.target_items[0].product_slot_id == "H2_F_L1_C01"
    assert result.target_items[0].picked is True


def test_task1_rejects_unmapped_new_style_slot_with_business_error() -> None:
    orchestrator = Task1Orchestrator(settings(), None)  # type: ignore[arg-type]
    target = TargetItem(
        product_name="未配置商品",
        sku_id="SKU",
        product_slot_id="H1_L01_C99",
        target_id="",
        shelf_level="L1",
        hand=Hand.LEFT,
    )

    with pytest.raises(Task1ServiceError) as error:
        orchestrator._plan_grasps([target])

    assert error.value.code == "NO_FEASIBLE_HAND_ASSIGNMENT"


@pytest.mark.asyncio
async def test_task1_reselects_overlapping_sku_location_to_keep_slots_distinct(
    tmp_path: Path,
) -> None:
    mock = Task1Mock()
    mock.sku_locations = {
        "可口可乐罐装": ["H3_L01_C03", "H3_L01_C04"],
        "百事可乐瓶装": ["H3_L01_C03"],
    }
    task_settings = settings().model_copy(
        update={
            "log_dir": str(tmp_path),
            "product_grasp_options": {
                "H3_L01_C03": [
                    GraspOption(hands=[Hand.LEFT], target_id="H3_INSPECT")
                ],
                "H3_L01_C04": [
                    GraspOption(hands=[Hand.RIGHT], target_id="H3_INSPECT")
                ],
            }
        }
    )
    client = Task1Client(task_settings, transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(task_settings, client).run(Task1Request())

    assert [item.product_slot_id for item in result.target_items] == [
        "H3_L01_C04",
        "H3_L01_C03",
    ]
    assert [item.hand for item in result.target_items] == [Hand.RIGHT, Hand.LEFT]
    assert all(item.picked and item.placed for item in result.target_items)


def test_task1_rejects_two_products_when_only_same_slot_is_available() -> None:
    shared_slot = "H3_L01_C03"
    task_settings = settings().model_copy(
        update={
            "product_grasp_options": {
                shared_slot: [
                    GraspOption(
                        hands=[Hand.LEFT, Hand.RIGHT], target_id="H3_INSPECT"
                    )
                ]
            }
        }
    )
    orchestrator = Task1Orchestrator(task_settings, None)  # type: ignore[arg-type]
    targets = [
        TargetItem(
            product_name=product_name,
            sku_id="SKU",
            product_slot_id=shared_slot,
            target_id="",
            shelf_level="L1",
            hand=Hand.LEFT,
        )
        for product_name in ("商品一", "商品二")
    ]

    with pytest.raises(Task1ServiceError) as error:
        orchestrator._plan_grasps(targets, [[shared_slot], [shared_slot]])

    assert error.value.code == "NO_FEASIBLE_HAND_ASSIGNMENT"
    assert "distinct physical slot" in error.value.message


def test_task1_prefers_same_target_and_level_for_parallel_pick() -> None:
    task_settings = settings().model_copy(
        update={
            "product_grasp_options": {
                "H2_L04_C02": [
                    GraspOption(hands=[Hand.LEFT], target_id="H2_INSPECT")
                ],
                "H2_L05_C03": [
                    GraspOption(hands=[Hand.LEFT], target_id="H2_INSPECT")
                ],
                "H2_L05_C04": [
                    GraspOption(hands=[Hand.RIGHT], target_id="H2_INSPECT")
                ],
            }
        }
    )
    orchestrator = Task1Orchestrator(task_settings, None)  # type: ignore[arg-type]
    targets = [
        TargetItem(
            product_name="多层商品",
            sku_id="SKU",
            product_slot_id="H2_L04_C02",
            target_id="",
            shelf_level="L4",
            hand=Hand.LEFT,
        ),
        TargetItem(
            product_name="五层商品",
            sku_id="SKU",
            product_slot_id="H2_L05_C04",
            target_id="",
            shelf_level="L5",
            hand=Hand.LEFT,
        ),
    ]

    planned = orchestrator._plan_grasps(
        targets,
        [["H2_L04_C02", "H2_L05_C03"], ["H2_L05_C04"]],
    )

    assert [target.product_slot_id for target in targets] == [
        "H2_L05_C03",
        "H2_L05_C04",
    ]
    assert [target.shelf_level for target in targets] == ["L5", "L5"]
    assert planned == [
        ("H2_INSPECT", Hand.LEFT),
        ("H2_INSPECT", Hand.RIGHT),
    ]


def test_parallel_pick_eligibility_does_not_require_different_product_names() -> None:
    targets = [
        TargetItem(
            product_name="同款商品",
            sku_id="SKU",
            product_slot_id="H3_L01_C03",
            target_id="H3_INSPECT",
            shelf_level="L1",
            hand=Hand.LEFT,
        ),
        TargetItem(
            product_name="同款商品",
            sku_id="SKU",
            product_slot_id="H3_L01_C04",
            target_id="H3_INSPECT",
            shelf_level="L1",
            hand=Hand.RIGHT,
        ),
    ]

    assert Task1Orchestrator._can_pick_in_parallel(targets) is True


@pytest.mark.parametrize(
    ("locations", "expected_slot"),
    [
        (
            [
                "H2_F_L5_C01",
                "H2_F_L1_C01",
                "H2_F_L4_C01",
                "H2_F_L2_C01",
                "H2_F_L3_C01",
            ],
            "H2_F_L3_C01",
        ),
        (
            ["H2_F_L5_C01", "H2_F_L1_C01", "H2_F_L4_C01", "H2_F_L2_C01"],
            "H2_F_L2_C01",
        ),
        (
            ["H2_F_L5_C01", "H2_F_L1_C01", "H2_F_L4_C01"],
            "H2_F_L4_C01",
        ),
        (["H2_F_L5_C01", "H2_F_L1_C01"], "H2_F_L1_C01"),
        (["H2_F_L5_C01"], "H2_F_L5_C01"),
    ],
)
@pytest.mark.asyncio
async def test_task1_selects_sku_location_by_shelf_priority(
    locations: list[str], expected_slot: str
) -> None:
    mock = Task1Mock()
    mock.sku_locations["可口可乐罐装"] = locations
    client = Task1Client(settings(), transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(settings(), client).run(Task1Request())

    assert result.target_items[0].product_slot_id == expected_slot
    product_pick = next(
        request
        for request in mock.requests
        if request.url.path == "/pick"
        and payload(request)["product_name"] == "可口可乐罐装"
    )
    assert payload(product_pick)["level"] == expected_slot.split("_")[2]


@pytest.mark.asyncio
async def test_task1_skips_configured_product_before_sku_lookup() -> None:
    mock = Task1Mock()
    mock.names["雪碧罐装"] = "H2_F_L1_C02"
    task_settings = settings().model_copy(
        update={
            "skip_product_names": ["可口可乐罐装"],
            "product_hand_options": {
                **settings().product_hand_options,
                "H2_F_L1_C02": ["LEFT", "RIGHT"],
            },
        }
    )
    client = Task1Client(task_settings, transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(task_settings, client).run(Task1Request())

    assert result.product_names == ["百事可乐瓶装", "雪碧罐装"]
    assert [target.product_name for target in result.target_items] == result.product_names
    assert [
        request.url.params["name"]
        for request in mock.requests
        if request.url.path == "/sku/search_by_name"
    ] == result.product_names
    assert [
        payload(request)["product_name"]
        for request in mock.requests
        if request.url.path == "/pick"
    ] == result.product_names


@pytest.mark.asyncio
async def test_task1_defers_configured_product_until_after_other_product() -> None:
    mock = Task1Mock()
    task_settings = settings().model_copy(
        update={"defer_product_names": ["可口可乐罐装"]}
    )
    client = Task1Client(task_settings, transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(task_settings, client).run(Task1Request())

    assert result.product_names == ["可口可乐罐装", "百事可乐瓶装"]
    assert [target.product_name for target in result.target_items] == [
        "百事可乐瓶装",
        "可口可乐罐装",
    ]
    assert [
        payload(request)["product_name"]
        for request in mock.requests
        if request.url.path == "/pick"
    ] == ["百事可乐瓶装", "可口可乐罐装"]


@pytest.mark.asyncio
async def test_task1_mock_order_requires_at_least_two_catalog_products() -> None:
    mock = Task1Mock()
    mock.names = {"可口可乐罐装": "H2_F_L1_C01"}
    client = Task1Client(settings(), transport=mock.transport)

    async with client:
        with pytest.raises(Task1ServiceError) as error:
            await Task1Orchestrator(settings(), client).create_mock_order()

    assert error.value.code == "INVALID_RESPONSE"


@pytest.mark.parametrize(
    ("skip_names", "defer_names", "message"),
    [
        ([""], [], "must not be empty"),
        (["可口可乐罐装", "可口可乐罐装"], [], "must not contain duplicates"),
        ([" 可口可乐罐装 "], ["可口可乐罐装"], "must not overlap"),
    ],
)
def test_task1_rejects_invalid_product_policy_configuration(
    skip_names: list[str], defer_names: list[str], message: str
) -> None:
    raw_settings = settings().model_dump()
    raw_settings["skip_product_names"] = skip_names
    raw_settings["defer_product_names"] = defer_names

    with pytest.raises(ValueError, match=message):
        Task1Settings.model_validate(raw_settings)


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
async def test_task1_picks_in_parallel_at_the_same_target_and_level(
    tmp_path: Path,
) -> None:
    mock = PickConcurrencyMock()
    mock.names = {
        "同层左手商品": "H3_L01_C03",
        "同层右手商品": "H3_L01_C04",
    }
    task_settings = settings().model_copy(
        update={
            "log_dir": str(tmp_path),
            "product_grasp_options": {
                "H3_L01_C03": [
                    GraspOption(hands=[Hand.LEFT], target_id="H3_INSPECT")
                ],
                "H3_L01_C04": [
                    GraspOption(hands=[Hand.RIGHT], target_id="H3_INSPECT")
                ],
            },
        }
    )
    client = Task1Client(task_settings, transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(task_settings, client).run(Task1Request())

    assert [item.target_id for item in result.target_items] == [
        "H3_INSPECT",
        "H3_INSPECT",
    ]
    assert [item.shelf_level for item in result.target_items] == ["L1", "L1"]
    assert [item.hand for item in result.target_items] == [Hand.LEFT, Hand.RIGHT]
    pair_requests = [
        request for request in mock.requests if request.url.path == "/pick/both"
    ]
    assert len(pair_requests) == 1
    assert pair_requests[0].headers["Idempotency-Key"].endswith(
        ":task1.pick.parallel.pick_both"
    )
    pair_payload = payload(pair_requests[0])
    assert pair_payload["left"]["slot_id"] == "H3_L01_C03"
    assert pair_payload["right"]["slot_id"] == "H3_L01_C04"
    shelf_pick_poses = [
        payload(request)
        for request in mock.requests
        if request.url.path == "/pose/prepare"
        and payload(request).get("pose_type") == "SHELF_PICK_READY"
    ]
    assert shelf_pick_poses == [
        {"pose_type": "SHELF_PICK_READY", "shelf_level": "L1"}
    ]


@pytest.mark.asyncio
async def test_parallel_pick_waits_for_both_initial_calls_before_serial_recovery(
    tmp_path: Path,
) -> None:
    mock = PickConcurrencyMock()
    mock.names = {
        "并行左手商品": "H3_L01_C03",
        "并行右手商品": "H3_L01_C04",
    }
    mock.pick_failure = {
        "error_code": "EXECUTION_FAILED",
        "message": "first left pick failed",
        "failed_interface": "manipulation_grasp",
    }
    mock.pick_failure_limit = 1
    task_settings = settings().model_copy(
        update={
            "log_dir": str(tmp_path),
            "product_grasp_options": {
                "H3_L01_C03": [
                    GraspOption(hands=[Hand.LEFT], target_id="H3_INSPECT")
                ],
                "H3_L01_C04": [
                    GraspOption(hands=[Hand.RIGHT], target_id="H3_INSPECT")
                ],
            },
        }
    )
    client = Task1Client(task_settings, transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(task_settings, client).run(
            Task1Request(), "parallel-recovery"
        )

    assert [item.picked for item in result.target_items] == [True, True]
    pick_requests = [
        request
        for request in mock.requests
        if request.url.path in {"/pick", "/pick/both"}
    ]
    assert [request.headers["Idempotency-Key"] for request in pick_requests] == [
        "parallel-recovery:task1.pick.parallel.pick_both",
        "parallel-recovery:task1.pick.0.pick:recovery.retry",
    ]


@pytest.mark.asyncio
async def test_parallel_pick_does_not_recover_while_other_hand_is_unknown() -> None:
    mock = PickConcurrencyMock()
    mock.initial_pick_failures = {
        "LEFT": {
            "error_code": "ACTION_RESULT_UNKNOWN",
            "message": "left pick result unknown",
        },
        "RIGHT": {
            "error_code": "EXECUTION_FAILED",
            "message": "right pick failed",
            "failed_interface": "manipulation_grasp",
            "pose": [1, 2, 3, 4, 5, 6],
        },
    }
    task_settings = settings().model_copy(
        update={
            "product_grasp_options": {
                "H3_L01_C03": [
                    GraspOption(hands=[Hand.LEFT], target_id="H3_INSPECT")
                ],
                "H3_L01_C04": [
                    GraspOption(hands=[Hand.RIGHT], target_id="H3_INSPECT")
                ],
            }
        }
    )
    targets = [
        TargetItem(
            product_name="未知左手商品",
            sku_id="SKU",
            product_slot_id="H3_L01_C03",
            target_id="H3_INSPECT",
            shelf_level="L1",
            hand=Hand.LEFT,
        ),
        TargetItem(
            product_name="失败右手商品",
            sku_id="SKU",
            product_slot_id="H3_L01_C04",
            target_id="H3_INSPECT",
            shelf_level="L1",
            hand=Hand.RIGHT,
        ),
    ]
    client = Task1Client(task_settings, transport=mock.transport)
    held_items: dict[Hand, str] = {}
    uncertain_hands: set[Hand] = set()
    action_failures: list[dict[str, str]] = []

    async with client:
        outcomes = await Task1Orchestrator(
            task_settings, client
        )._pick_targets_parallel(
            targets,
            "parallel-unknown",
            task1_service_module._NullTaskLog(),
            {"target_id": None},
            held_items,
            uncertain_hands,
            action_failures,
        )

    assert outcomes == [False, False]
    assert uncertain_hands == {Hand.LEFT}
    assert held_items == {}
    pair_requests = [
        request for request in mock.requests if request.url.path == "/pick/both"
    ]
    assert len(pair_requests) == 1
    assert not [
        request
        for request in mock.requests
        if request.url.path == "/navigation/nudge"
        or (
            request.url.path == "/pose/prepare"
            and request.headers["Idempotency-Key"].endswith(":recovery.pose")
        )
    ]


@pytest.mark.asyncio
async def test_task1_keeps_same_target_different_levels_serial(
    tmp_path: Path,
) -> None:
    mock = PickConcurrencyMock()
    mock.names = {
        "一层左手商品": "H3_L01_C03",
        "二层右手商品": "H3_L02_C04",
    }
    task_settings = settings().model_copy(
        update={
            "log_dir": str(tmp_path),
            "product_grasp_options": {
                "H3_L01_C03": [
                    GraspOption(hands=[Hand.LEFT], target_id="H3_INSPECT")
                ],
                "H3_L02_C04": [
                    GraspOption(hands=[Hand.RIGHT], target_id="H3_INSPECT")
                ],
            },
        }
    )
    client = Task1Client(task_settings, transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(task_settings, client).run(Task1Request())

    assert [item.target_id for item in result.target_items] == [
        "H3_INSPECT",
        "H3_INSPECT",
    ]
    assert [item.shelf_level for item in result.target_items] == ["L1", "L2"]
    assert [item.hand for item in result.target_items] == [Hand.LEFT, Hand.RIGHT]
    assert mock.max_active_pick_requests == 1
    shelf_pick_poses = [
        payload(request)
        for request in mock.requests
        if request.url.path == "/pose/prepare"
        and payload(request).get("pose_type") == "SHELF_PICK_READY"
    ]
    assert shelf_pick_poses == [
        {"pose_type": "SHELF_PICK_READY", "shelf_level": "L1"},
        {"pose_type": "SHELF_PICK_READY", "shelf_level": "L2"},
    ]


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
    recovery_pose = next(
        request
        for request in mock.requests
        if request.url.path == "/pose/prepare"
        and request.headers["Idempotency-Key"].endswith(":recovery.pose")
    )
    assert len(special_picks) == 2
    assert payload(recovery_pose) == {
        "pose_type": "SHELF_PICK_READY",
        "shelf_level": "L1",
    }
    assert mock.requests.index(nudge_requests[0]) < mock.requests.index(special_picks[0])
    assert mock.requests.index(special_picks[0]) < mock.requests.index(nudge_requests[1])
    assert mock.requests.index(nudge_requests[1]) < mock.requests.index(recovery_pose)
    assert mock.requests.index(recovery_pose) < mock.requests.index(special_picks[1])
    assert mock.requests.index(special_picks[1]) < mock.requests.index(nudge_requests[2])


@pytest.mark.asyncio
async def test_task1_h2_b_l1_c01_nudges_left_before_pick_and_returns() -> None:
    mock = Task1Mock()
    special_product = "舒肤佳香皂纯白清香型"
    mock.names = {
        special_product: "H2_B_L1_C01",
        "百事可乐瓶装": "H2_F_L1_C04",
    }
    task_settings = settings().model_copy(
        update={
            "product_hand_options": {
                "H2_B_L1_C01": ["LEFT"],
                "H2_F_L1_C04": ["RIGHT"],
            }
        }
    )
    client = Task1Client(task_settings, transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(task_settings, client).run(Task1Request())

    assert result.status == "SUCCEEDED"
    nudge_requests = [
        request for request in mock.requests if request.url.path == "/navigation/nudge"
    ]
    assert [payload(request) for request in nudge_requests] == [
        {"action": "approach", "direction": "left"},
        {"action": "return"},
    ]
    special_pick = next(
        request
        for request in mock.requests
        if request.url.path == "/pick"
        and payload(request)["product_name"] == special_product
    )
    assert payload(special_pick)["hand"] == "LEFT"
    assert mock.requests.index(nudge_requests[0]) < mock.requests.index(special_pick)
    assert mock.requests.index(special_pick) < mock.requests.index(nudge_requests[1])


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
    recovery_pose = next(
        request
        for request in mock.requests
        if request.headers.get("Idempotency-Key")
        == "task1-recovery:task1.pick.0.pick:recovery.pose"
    )
    assert recovery_pose.url.path == "/pose/prepare"
    assert payload(recovery_pose) == {
        "pose_type": "SHELF_PICK_READY",
        "shelf_level": "L1",
    }
    assert mock.requests.index(nudge_requests[0]) < mock.requests.index(recovery_pose)
    assert mock.requests.index(recovery_pose) < mock.requests.index(first_product_picks[1])


@pytest.mark.asyncio
async def test_task1_retries_recovery_pose_before_retrying_pick() -> None:
    mock = Task1Mock()
    mock.pick_failure = {
        "error_code": "EXECUTION_FAILED",
        "message": "grasp failed",
        "failed_interface": "manipulation_grasp",
        "pose": [1, 2, 3, 4, 5, 6],
    }
    mock.pick_failure_limit = 1
    recovery_pose_key = "task1-pose-failure:task1.pick.0.pick:recovery.pose"
    mock.pose_failure_keys.add(recovery_pose_key)
    client = Task1Client(settings(), transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(settings(), client).run(
            Task1Request(), "task1-pose-failure"
        )

    assert result.status == "SUCCEEDED"
    assert [item.placed for item in result.target_items] == [True, True]
    first_product_picks = [
        request
        for request in mock.requests
        if request.url.path == "/pick"
        and payload(request)["product_name"] == "可口可乐罐装"
    ]
    assert [request.headers["Idempotency-Key"] for request in first_product_picks] == [
        "task1-pose-failure:task1.pick.0.pick",
        "task1-pose-failure:task1.pick.0.pick:recovery.retry",
    ]
    recovery_pose = next(
        request
        for request in mock.requests
        if request.headers.get("Idempotency-Key") == recovery_pose_key
    )
    recovery_pose_retry = next(
        request
        for request in mock.requests
        if request.headers.get("Idempotency-Key") == f"{recovery_pose_key}:retry"
    )
    nudge_requests = [
        request for request in mock.requests if request.url.path == "/navigation/nudge"
    ]
    assert [payload(request) for request in nudge_requests] == [
        {"action": "approach", "direction": "right"},
        {"action": "return"},
    ]
    assert mock.requests.index(nudge_requests[0]) < mock.requests.index(recovery_pose)
    assert mock.requests.index(recovery_pose) < mock.requests.index(recovery_pose_retry)
    assert mock.requests.index(recovery_pose_retry) < mock.requests.index(first_product_picks[1])
    assert mock.requests.index(first_product_picks[1]) < mock.requests.index(nudge_requests[1])


@pytest.mark.asyncio
async def test_task1_continues_after_two_nudge_return_failures() -> None:
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
        result = await Task1Orchestrator(settings(), client).run(
            Task1Request(), "task1-return-failure"
        )

    assert result.status == "SUCCEEDED"
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
        "百事可乐瓶装",
    ]
    navigation_targets = [
        payload(request)["target_id"]
        for request in mock.requests
        if request.url.path == "/navigation/navigate"
    ]
    assert navigation_targets[-1] == "task_boundary"


@pytest.mark.asyncio
async def test_task1_falls_back_to_individual_release_after_dual_failure(tmp_path: Path) -> None:
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
        result = await Task1Orchestrator(task_settings, client).run(Task1Request())

    assert result.status == "SUCCEEDED"
    assert [item.placed for item in result.target_items] == [True, True]
    assert len([request for request in mock.requests if request.url.path == "/place"]) == 2
    navigation_targets = [
        payload(request)["target_id"]
        for request in mock.requests
        if request.url.path == "/navigation/navigate"
    ]
    assert navigation_targets[-1] == "task_boundary"
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
    assert events[-1]["event"] == "operation"
    assert events[-1]["status"] == "succeeded"


@pytest.mark.asyncio
async def test_task1_unknown_pick_disables_only_affected_hand() -> None:
    mock = Task1Mock()
    mock.pick_failure = {
        "error_code": "ACTION_RESULT_UNKNOWN",
        "message": "pick result unknown",
    }
    mock.pick_failure_limit = 1
    client = Task1Client(settings(), transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(settings(), client).run(Task1Request())

    assert result.status == "SUCCEEDED"
    assert [item.placed for item in result.target_items] == [False, True]
    assert [
        payload(request)["target_id"]
        for request in mock.requests
        if request.url.path == "/navigation/navigate"
    ][-1] == "task_boundary"


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
    assert error.value.interface_metrics
    assert any(
        metric.interface == "navigation/navigation/navigate"
        for metric in error.value.interface_metrics
    )


@pytest.mark.asyncio
async def test_task1_reports_failure_recovery_error_when_start_navigation_fails() -> None:
    mock = Task1Mock()
    mock.health["sku"] = "ERROR"
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
        result = await Task1Orchestrator(settings(), client).run(Task1Request())

    pick_requests = [request for request in mock.requests if request.url.path == "/pick"]
    assert mock.pick_attempts == 3
    assert len(pick_requests) == 3
    assert pick_requests[0].headers["Idempotency-Key"] == pick_requests[1].headers["Idempotency-Key"]
    metrics = {metric.interface: metric for metric in result.interface_metrics}
    pick_metric = metrics["pick_place/pick"]
    assert pick_metric.call_count == 3
    assert pick_metric.success_count == 2
    assert pick_metric.failure_count == 1
    assert pick_metric.average_duration_ms >= 0


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
    assert '"event": "模拟点单"' in events
    assert '"event": "SKU货位转换"' in events
    assert '"event": "抓取"' in events
    assert '"duration_ms":' in events
    assert '"call_count":' in events
    assert '"event": "operation"' in events
    interface_events = [
        event
        for line in events.splitlines()
        if (event := json.loads(line))["event"] == "接口调用"
    ]
    assert interface_events
    assert len({event["call_id"] for event in interface_events}) == len(
        interface_events
    )
    assert all(event["duration_ms"] >= 0 for event in interface_events)


@pytest.mark.asyncio
async def test_task1_continues_when_operation_log_cannot_be_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_log_initialization(*args: object, **kwargs: object) -> None:
        raise OSError("log directory unavailable")

    monkeypatch.setattr(task1_service_module, "_Task1Log", fail_log_initialization)
    mock = Task1Mock()
    task_settings = settings()
    client = Task1Client(task_settings, transport=mock.transport)

    async with client:
        result = await Task1Orchestrator(task_settings, client).run(Task1Request())

    assert result.status == "SUCCEEDED"
    assert [item.placed for item in result.target_items] == [True, True]
