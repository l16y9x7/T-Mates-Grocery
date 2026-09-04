from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import httpx
import pytest

import task2_service.service as task2_service_module
from task2_service.client import Task2Client
from task2_service.models import InspectionPose, ProductHandOption, Task2Request, Task2ServiceError, Task2Settings, Task2Timeouts
from task2_service.service import Task2Orchestrator
from task_service.settings import TaskServiceSettings


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class Task2Mock:
    def __init__(self, inspection_results: list[list[object]]) -> None:
        self.requests: list[httpx.Request] = []
        self.inspection_results = deque(inspection_results)
        self.pick_attempts = 0
        self.pick_failure_limit = 0
        self.place_attempts = 0
        self.place_failure_limit = 0
        self.gripper_open_attempts = 0
        self.gripper_open_failure_limit = 0

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/health") or path == "/health":
            return httpx.Response(200, json={"status": "READY"})
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
        if path == "/pick":
            self.pick_attempts += 1
            if self.pick_attempts <= self.pick_failure_limit:
                return httpx.Response(
                    502,
                    json={
                        "error_code": "EXECUTION_FAILED",
                        "message": "pick failed",
                        "failed_interface": "manipulation_grasp",
                        "pose": [-1, 2, 3, 4, 5, 6],
                    },
                )
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        if path == "/place":
            self.place_attempts += 1
            if self.place_attempts <= self.place_failure_limit:
                return httpx.Response(
                    502,
                    json={
                        "error_code": "EXECUTION_FAILED",
                        "message": "place failed",
                        "failed_interface": "manipulation_release",
                        "pose": [1, 2, 3, 4, 5, 6],
                    },
                )
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        if path == "/manipulation/gripper/open":
            self.gripper_open_attempts += 1
            if self.gripper_open_attempts <= self.gripper_open_failure_limit:
                return httpx.Response(502, json={"error_code": "EXECUTION_FAILED"})
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        if path == "/navigation/nudge":
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        if path in {"/navigation/navigate", "/pose/prepare"}:
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        return httpx.Response(404, json={"error_code": "UNKNOWN_ENDPOINT"})


def settings(tmp_path: Path) -> Task2Settings:
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
        baseline_dir=str(tmp_path / "task0"),
        product_hand_options_file="unused.yaml",
        product_hand_options={
            "H1_F_L1_C01": {"product_name": "商品一", "hands": ["LEFT"], "target_id": "H1_F_L_INSPECT"},
            "H1_F_L1_C02": {"product_name": "商品二", "hands": ["LEFT"], "target_id": "H1_F_L_INSPECT"},
            "H1_F_L1_C03": {"product_name": "商品三", "hands": ["LEFT"], "target_id": "H1_F_L_INSPECT"},
            "H1_F_L3_C02": {"product_name": "商品二", "hands": ["LEFT"], "target_id": "H1_F_L_INSPECT"},
            "H1_F_R1_C04": {"product_name": "右侧商品", "hands": ["RIGHT"], "target_id": "H1_F_R_INSPECT"},
        },
        log_dir=str(tmp_path / "log"),
    )


def create_baselines(task_settings: Task2Settings) -> None:
    for target_id in task_settings.inspection_points:
        for pose in (InspectionPose.UPPER, InspectionPose.LOWER):
            path = Path(task_settings.baseline_dir) / f"{target_id}_{pose.directory_suffix}" / "rgb.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"baseline")
            (path.parent / "depth_mm.npy").write_bytes(b"depth")
            (path.parent / "meta.json").write_text("{}", encoding="utf-8")


def ready_settings(tmp_path: Path) -> Task2Settings:
    task_settings = settings(tmp_path)
    create_baselines(task_settings)
    return task_settings


def shortage(*names: str) -> list[dict[str, str]]:
    return [{"shortage_product_name": name} for name in names]


def shortage_slot(name: str, slot_id: str) -> dict[str, str]:
    return {"shortage_product_name": name, "slot_id": slot_id}


def payload(request: httpx.Request) -> dict[str, object]:
    return json.loads(request.content) if request.content else {}


def requests_for(mock: Task2Mock, path: str) -> list[httpx.Request]:
    return [request for request in mock.requests if request.url.path == path]


@pytest.mark.asyncio
async def test_task2_inspects_a_full_face_before_processing_all_findings(tmp_path: Path) -> None:
    mock = Task2Mock([shortage("商品一"), shortage("商品二"), [], []])
    task_settings = ready_settings(tmp_path)
    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.status == "SUCCEEDED"
    assert result.inspection_pass == 1
    assert result.product_names == ["商品一", "商品二"]
    assert all(item.picked and item.placed for item in result.target_items)
    inspections = requests_for(mock, "/perception/inspect")
    first_pick = next(index for index, request in enumerate(mock.requests) if request.url.path == "/pick")
    assert len(inspections) == 4
    assert all(mock.requests.index(request) < first_pick for request in inspections)
    assert [payload(request)["location_id"] for request in inspections] == [
        "H1_F_L_INSPECT",
        "H1_F_L_INSPECT",
        "H1_F_R_INSPECT",
        "H1_F_R_INSPECT",
    ]


@pytest.mark.asyncio
async def test_task2_processes_each_face_before_inspecting_the_next(tmp_path: Path) -> None:
    mock = Task2Mock(
        [shortage("商品一"), [], [], [], shortage("背面商品"), [], [], []]
    )
    task_settings = settings(tmp_path)
    task_settings.inspection_points.extend(["H1_B_L_INSPECT", "H1_B_R_INSPECT"])
    task_settings.product_hand_options["H1_B_L1_C01"] = ProductHandOption(
        product_name="背面商品",
        hands=["LEFT"],
        target_id="H1_B_L_INSPECT",
    )
    create_baselines(task_settings)
    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.product_names == ["商品一", "背面商品"]
    inspections = requests_for(mock, "/perception/inspect")
    picks = requests_for(mock, "/pick")
    assert mock.requests.index(inspections[3]) < mock.requests.index(picks[0])
    assert mock.requests.index(picks[0]) < mock.requests.index(inspections[4])
    assert mock.requests.index(inspections[7]) < mock.requests.index(picks[1])


@pytest.mark.asyncio
async def test_task2_tries_duplicate_findings_without_count_limit(tmp_path: Path) -> None:
    mock = Task2Mock([shortage("商品一", "商品一", "商品二"), [], [], []])
    mock.pick_failure_limit = 2
    task_settings = ready_settings(tmp_path)
    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.product_names == ["商品一", "商品一", "商品二"]
    assert [payload(request)["product_name"] for request in requests_for(mock, "/pick")] == [
        "商品一",
        "商品一",
        "商品一",
        "商品二",
    ]
    assert [item.placed for item in result.target_items] == [False, True, True]


@pytest.mark.asyncio
async def test_task2_skips_unknown_finding_and_continues(tmp_path: Path) -> None:
    mock = Task2Mock([shortage("未知商品", "商品一", "商品二"), [], [], []])
    task_settings = ready_settings(tmp_path)
    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.product_names == ["商品一", "商品二"]
    assert [payload(request)["product_name"] for request in requests_for(mock, "/pick")] == ["商品一", "商品二"]


@pytest.mark.asyncio
async def test_task2_retries_place_before_moving_to_the_next_finding(tmp_path: Path) -> None:
    mock = Task2Mock([shortage("商品一", "商品二", "商品三"), [], [], []])
    mock.place_failure_limit = 1
    task_settings = ready_settings(tmp_path)
    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.status == "SUCCEEDED"
    assert [item.placed for item in result.target_items] == [True, True]
    assert len(requests_for(mock, "/place")) == 3
    assert not requests_for(mock, "/manipulation/gripper/open")


@pytest.mark.asyncio
async def test_task2_restores_replenishment_pose_before_failed_pick_nudge(
    tmp_path: Path,
) -> None:
    mock = Task2Mock([shortage("商品一"), [], [], []])
    mock.pick_failure_limit = 1
    task_settings = ready_settings(tmp_path)

    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(
            Task2Request(), "task2-pick-recovery"
        )

    assert result.status == "SUCCEEDED"
    recovery_pose = next(
        request
        for request in requests_for(mock, "/pose/prepare")
        if request.headers["Idempotency-Key"].endswith(":recovery.pose")
    )
    assert payload(recovery_pose) == {
        "pose_type": "REPLENISHMENT_TABLE_PICK_READY"
    }
    nudge_requests = requests_for(mock, "/navigation/nudge")
    retry_pick = requests_for(mock, "/pick")[1]
    assert mock.requests.index(recovery_pose) < mock.requests.index(nudge_requests[0])
    assert mock.requests.index(nudge_requests[0]) < mock.requests.index(retry_pick)


@pytest.mark.asyncio
async def test_task2_stops_inspection_after_two_successful_places(tmp_path: Path) -> None:
    mock = Task2Mock([shortage("商品一", "商品二", "商品三"), [], [], []])
    task_settings = ready_settings(tmp_path)
    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.product_names == ["商品一", "商品二"]
    assert [payload(request)["product_name"] for request in requests_for(mock, "/pick")] == ["商品一", "商品二"]
    navigation_targets = [payload(request)["target_id"] for request in requests_for(mock, "/navigation/navigate")]
    assert navigation_targets[-1] == "task_boundary"


@pytest.mark.asyncio
async def test_task2_finishes_at_boundary_with_only_one_success(tmp_path: Path) -> None:
    mock = Task2Mock([shortage("商品一"), [], [], []])
    task_settings = ready_settings(tmp_path)
    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.status == "SUCCEEDED"
    assert [item.placed for item in result.target_items] == [True]
    navigation_targets = [payload(request)["target_id"] for request in requests_for(mock, "/navigation/navigate")]
    assert navigation_targets[-1] == "task_boundary"


@pytest.mark.asyncio
async def test_task2_retries_discard_and_continues(tmp_path: Path) -> None:
    mock = Task2Mock([shortage("商品一", "商品二", "商品三"), [], [], []])
    mock.place_failure_limit = 2
    mock.gripper_open_failure_limit = 1
    task_settings = ready_settings(tmp_path)
    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.status == "SUCCEEDED"
    assert len(requests_for(mock, "/manipulation/gripper/open")) == 2
    navigation_targets = [payload(request)["target_id"] for request in requests_for(mock, "/navigation/navigate")]
    assert navigation_targets[-1] == "task_boundary"


@pytest.mark.asyncio
async def test_task2_rejects_invalid_inspection_payload(tmp_path: Path) -> None:
    mock = Task2Mock([["商品一"], [], [], []])
    task_settings = ready_settings(tmp_path)
    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.status == "SUCCEEDED"
    assert not requests_for(mock, "/pick")


@pytest.mark.asyncio
async def test_task2_rejects_blank_product_name(tmp_path: Path) -> None:
    mock = Task2Mock([shortage("  "), [], [], []])
    task_settings = ready_settings(tmp_path)
    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.status == "SUCCEEDED"
    assert not requests_for(mock, "/pick")


@pytest.mark.asyncio
async def test_task2_unknown_pick_disables_hand_and_finishes_inspection(tmp_path: Path) -> None:
    mock = Task2Mock([shortage("商品一", "商品二"), [], [], []])
    task_settings = ready_settings(tmp_path)

    async def unknown_pick(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/pick":
            mock.requests.append(request)
            return httpx.Response(
                502,
                json={
                    "error_code": "ACTION_RESULT_UNKNOWN",
                    "message": "pick result unknown",
                },
            )
        return await mock.handle(request)

    async with Task2Client(
        task_settings, transport=httpx.MockTransport(unknown_pick)
    ) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.status == "SUCCEEDED"
    assert len(requests_for(mock, "/pick")) == 1
    assert [payload(request)["target_id"] for request in requests_for(mock, "/navigation/navigate")][-1] == "task_boundary"


@pytest.mark.asyncio
async def test_task2_tries_the_other_hand_when_one_hand_becomes_uncertain(
    tmp_path: Path,
) -> None:
    mock = Task2Mock(
        [
            shortage("商品一", "商品二", "商品三"),
            [],
            shortage("右侧商品"),
            [],
        ]
    )
    task_settings = ready_settings(tmp_path)
    task_settings.product_hand_options["H1_F_L1_C02"] = ProductHandOption(
        product_name="商品二",
        hands=["RIGHT"],
        target_id="H1_F_L_INSPECT",
    )
    first_left_pick = True

    async def unknown_first_left_pick(request: httpx.Request) -> httpx.Response:
        nonlocal first_left_pick
        if request.url.path == "/pick" and first_left_pick:
            first_left_pick = False
            mock.requests.append(request)
            return httpx.Response(
                502,
                json={
                    "error_code": "ACTION_RESULT_UNKNOWN",
                    "message": "left pick result unknown",
                },
            )
        return await mock.handle(request)

    async with Task2Client(
        task_settings,
        transport=httpx.MockTransport(unknown_first_left_pick),
    ) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert [item.product_name for item in result.target_items] == [
        "商品一",
        "商品二",
        "右侧商品",
    ]
    assert [item.product_name for item in result.target_items if item.placed] == [
        "商品二",
        "右侧商品",
    ]
    assert [payload(request)["product_name"] for request in requests_for(mock, "/pick")] == [
        "商品一",
        "商品二",
        "右侧商品",
    ]


@pytest.mark.asyncio
async def test_task2_matches_normalized_product_name(tmp_path: Path) -> None:
    configured_name = "Lays乐事薯片墨西哥鸡汁番茄味"
    detected_name = "Lay's乐事薯片墨西哥鸡汁番茄味"
    mock = Task2Mock([shortage(detected_name, "商品二"), [], [], []])
    task_settings = ready_settings(tmp_path)
    task_settings.product_hand_options["H1_F_L1_C01"].product_name = configured_name
    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.product_names == [detected_name, "商品二"]


@pytest.mark.asyncio
async def test_task2_requires_task0_baselines(tmp_path: Path) -> None:
    mock = Task2Mock([])
    task_settings = settings(tmp_path)
    async with Task2Client(task_settings, transport=mock.transport) as client:
        orchestrator = Task2Orchestrator(task_settings, client)
        assert not await orchestrator.ready()
        with pytest.raises(Task2ServiceError) as error:
            await orchestrator.run(Task2Request())

    assert error.value.code == "BASELINE_NOT_READY"
    assert error.value.step == "健康检查"
    assert not requests_for(mock, "/perception/inspect")


def test_task2_resolves_versioned_task0_baseline(tmp_path: Path) -> None:
    task_settings = settings(tmp_path)
    storage_root = Path(task_settings.baseline_dir)
    scan_id = "0123456789abcdef0123456789abcdef"
    scan_root = storage_root / "runs" / scan_id
    task_settings.baseline_dir = str(scan_root)
    create_baselines(task_settings)
    task_settings.baseline_dir = str(storage_root)
    (scan_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "scan_id": scan_id,
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    (storage_root / "current.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "scan_id": scan_id,
                "run_directory": f"runs/{scan_id}",
            }
        ),
        encoding="utf-8",
    )

    assert Task2Orchestrator(task_settings, object()).baselines_ready()


def test_task2_production_config_has_its_own_face_order() -> None:
    task_settings = TaskServiceSettings.load(CONFIG_DIR / "runtime.production.yaml")

    assert task_settings.tasks.task2.inspection_points == [
        "H1_INSPECT",
        "H12_INSPECT",
        "H2_INSPECT",
        "H23_INSPECT",
        "H3_INSPECT",
    ]
    assert task_settings.tasks.task0.inspection_points[0] == "H1_INSPECT"
    assert task_settings.tasks.task3.inspection_points[0].target_id == "H1_INSPECT"
    assert task_settings.tasks.task2.product_hand_options_schema_version == "2.0"


@pytest.mark.asyncio
async def test_task2_v2_picks_two_exact_slots_in_one_two_hand_batch(
    tmp_path: Path,
) -> None:
    task_settings = settings(tmp_path)
    task_settings.inspection_points = [
        "H1_INSPECT",
        "H12_INSPECT",
        "H2_INSPECT",
    ]
    task_settings.product_hand_options_schema_version = "2.0"
    task_settings.product_hand_options = {
        "H1_L01_C05": ProductHandOption.model_validate(
            {
                "product_name": "汰渍肥皂",
                "grasp_options": [
                    {"hands": ["RIGHT"], "target_id": "H1_INSPECT"},
                    {"hands": ["LEFT"], "target_id": "H12_INSPECT"},
                ],
            }
        ),
        "H2_L01_C01": ProductHandOption.model_validate(
            {
                "product_name": "可口可乐罐装",
                "grasp_options": [
                    {"hands": ["LEFT"], "target_id": "H2_INSPECT"},
                    {"hands": ["RIGHT"], "target_id": "H12_INSPECT"},
                ],
            }
        ),
    }
    create_baselines(task_settings)
    mock = Task2Mock(
        [
            [shortage_slot("汰渍肥皂", "H1_L01_C05")],
            [],
            [shortage_slot("汰渍肥皂", "H1_L01_C05")],
            [],
            [shortage_slot("可口可乐罐装", "H2_L01_C01")],
            [],
        ]
    )

    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert [item.product_slot_id for item in result.target_items] == [
        "H1_L01_C05",
        "H2_L01_C01",
    ]
    assert [item.hand.value for item in result.target_items] == ["LEFT", "RIGHT"]
    assert {
        item.inspection_target_id for item in result.target_items
    } == {"H12_INSPECT"}
    picks = requests_for(mock, "/pick")
    places = requests_for(mock, "/place")
    assert len(picks) == 2
    assert len(places) == 2
    assert max(mock.requests.index(request) for request in picks) < min(
        mock.requests.index(request) for request in places
    )
    assert [payload(request)["slot_id"] for request in places] == [
        "H1_L01_C05",
        "H2_L01_C01",
    ]
    replenishment_navigations = [
        request
        for request in requests_for(mock, "/navigation/navigate")
        if payload(request)["target_id"] == task_settings.replenishment_pickup
    ]
    assert len(replenishment_navigations) == 1


@pytest.mark.asyncio
async def test_task2_v2_rejects_a_shortage_without_exact_slot_id(
    tmp_path: Path,
) -> None:
    task_settings = settings(tmp_path)
    task_settings.product_hand_options_schema_version = "2.0"
    create_baselines(task_settings)
    mock = Task2Mock([shortage("商品一"), [], [], []])

    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.target_items == []
    assert not requests_for(mock, "/pick")


@pytest.mark.asyncio
async def test_task2_continues_when_operation_log_cannot_be_initialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_log_initialization(*args: object, **kwargs: object) -> None:
        raise OSError("log directory unavailable")

    monkeypatch.setattr(task2_service_module, "_Task2Log", fail_log_initialization)
    mock = Task2Mock([shortage("商品一", "商品二"), [], [], []])
    task_settings = ready_settings(tmp_path)

    async with Task2Client(task_settings, transport=mock.transport) as client:
        result = await Task2Orchestrator(task_settings, client).run(Task2Request())

    assert result.status == "SUCCEEDED"
    assert [item.placed for item in result.target_items] == [True, True]
