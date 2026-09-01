"""8086 取放编排服务的 API 和内部顺序测试。"""

from __future__ import annotations

import asyncio
import base64
import json
import math
from pathlib import Path

import httpx
import numpy as np
import pytest

import pick_place_service.service as service_module
from pick_place_service.app import create_app
from pick_place_service.models import (
    FrameBundle,
    LocateResponse,
    PlaceLocateResponse,
    PickPlaceRequest,
    PickPlaceSettings,
    PoseResponse,
    ServiceError,
    StatusResponse,
    action_payload,
    normalize_product_name,
)
from pick_place_service.service import (
    CameraFrameProvider,
    OperationCache,
    PickPlaceOrchestrator,
    SubagentClient,
)
from pick_place_service.place_pose import synthesize_place_pose


PICK_CAMERAS = {"left": "left_wrist", "right": "right_wrist"}


class FakeSubagents:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_execute = False
        self.executed_poses: list[list[float]] = []
        self.estimated_pose = [1, 2, 3, 4, 5, 6]
        self.estimated_poses: list[list[float]] = []
        self.place_direction = "up"
        self.pose_log_suffixes: list[str | None] = []

    async def locate(self, request: PickPlaceRequest, kind: str) -> LocateResponse:
        self.calls.append((kind, "locate"))
        return LocateResponse(product_name=request.product_name, bbox=[100, 200, 300, 500])

    async def locate_place(self, request: PickPlaceRequest) -> PlaceLocateResponse:
        self.calls.append(("place", "locate"))
        reference_count = 1 if self.place_direction == "up" else 2
        return PlaceLocateResponse(
            name=request.product_name,
            bbox=[[index * 2, 0, index * 2 + 2, 2] for index in range(reference_count)],
            mask=["bWFzaw=="] * reference_count,
            direction=self.place_direction,
            image_path="/task0/rgb.jpg",
            current_image_path="/current/rgb.jpg",
            level="L2",
        )

    async def estimate_pose(
        self,
        request: PickPlaceRequest,
        kind: str,
        frame: FrameBundle,
        *,
        log_suffix: str | None = None,
    ) -> PoseResponse:
        self.calls.append((kind, "pose"))
        self.pose_log_suffixes.append(log_suffix)
        assert frame.mask == "mask.pgm"
        pose = self.estimated_poses.pop(0) if self.estimated_poses else self.estimated_pose
        return PoseResponse(pose=pose)

    async def execute(
        self,
        request: PickPlaceRequest,
        kind: str,
        pose: PoseResponse,
        operation_key: str,
    ) -> None:
        self.calls.append((kind, "execute"))
        self.executed_poses.append(list(pose.pose))
        if self.fail_execute:
            raise ServiceError("EXECUTION_FAILED", "grasp failed")

    async def prepare_place(self, level: str, operation_key: str) -> None:
        assert level == "L2"
        self.calls.append(("place", "prepare"))

    async def check(self, request: PickPlaceRequest, kind: str) -> None:
        self.calls.append((kind, "check"))


class FakeFrames:
    def __init__(self) -> None:
        self.cameras: list[str] = []

    async def capture(
        self,
        camera: str,
        bbox: list[int | float],
        operation: str,
        mask_base64: str | None = None,
    ) -> FrameBundle:
        self.cameras.append(camera)
        return FrameBundle(rgb="rgb", depth="depth", camera="camera", mask="mask.pgm")

    async def prepare_place_references(
        self,
        located: PlaceLocateResponse,
        camera: str,
        operation: str,
    ) -> list[FrameBundle]:
        self.cameras.append(camera)
        return [
            FrameBundle(rgb="rgb", depth="depth", camera="camera", mask="mask.pgm")
            for _ in located.mask
        ]


def make_app(fake: FakeSubagents):
    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_file="camera.json",
    )
    orchestrator = PickPlaceOrchestrator(settings, fake, FakeFrames())
    return create_app(settings, orchestrator=orchestrator)


@pytest.mark.asyncio
async def test_operation_cache_reports_active_operations() -> None:
    cache = OperationCache()
    started = asyncio.Event()
    release = asyncio.Event()
    request = PickPlaceRequest(
        task_type="SORTING", product_name="可口可乐", hand="left"
    )

    async def operation():
        started.set()
        await release.wait()
        return StatusResponse(status="SUCCEEDED")

    execution = asyncio.create_task(cache.run("active-1", request, operation))
    await started.wait()
    assert await cache.active_count() == 1
    release.set()
    await execution
    assert await cache.active_count() == 0


@pytest.mark.asyncio
async def test_operation_result_reports_running_success_and_missing_key() -> None:
    fake = FakeSubagents()
    started = asyncio.Event()
    release = asyncio.Event()
    original_run = fake.execute

    async def delayed_execute(*args, **kwargs):
        started.set()
        await release.wait()
        return await original_run(*args, **kwargs)

    fake.execute = delayed_execute  # type: ignore[method-assign]
    app = make_app(fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://pick-place"
    ) as client:
        payload = {
            "task_type": "SORTING",
            "product_name": "可口可乐",
            "hand": "LEFT",
        }
        pick = asyncio.create_task(
            client.post(
                "/pick",
                json=payload,
                headers={"Idempotency-Key": "query-result"},
            )
        )
        await started.wait()

        running = await client.get(
            "/operations/result", params={"idempotency_key": "query-result"}
        )
        missing = await client.get(
            "/operations/result", params={"idempotency_key": "missing"}
        )
        release.set()
        assert (await pick).status_code == 200
        succeeded = await client.get(
            "/operations/result", params={"idempotency_key": "query-result"}
        )

    assert running.status_code == 202
    assert running.json() == {"status": "RUNNING"}
    assert missing.status_code == 404
    assert missing.json()["error_code"] == "OPERATION_NOT_FOUND"
    assert succeeded.status_code == 200
    assert succeeded.json() == {"status": "SUCCEEDED"}


@pytest.mark.asyncio
async def test_operation_result_replays_cached_failure_details() -> None:
    fake = FakeSubagents()

    async def failing_execute(*args, **kwargs):
        raise ServiceError(
            "EXECUTION_FAILED",
            "grasp failed",
            failed_interface="manipulation_grasp",
            url="http://robot/manipulation/grasp",
            pose=[1, 2, 3, 4, 5, 6],
        )

    fake.execute = failing_execute  # type: ignore[method-assign]
    app = make_app(fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://pick-place"
    ) as client:
        payload = {
            "task_type": "SORTING",
            "product_name": "可口可乐",
            "hand": "LEFT",
        }
        failed = await client.post(
            "/pick",
            json=payload,
            headers={"Idempotency-Key": "query-failure"},
        )
        queried = await client.get(
            "/operations/result", params={"idempotency_key": "query-failure"}
        )

    assert failed.status_code == queried.status_code == 502
    assert queried.json() == failed.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("hand", ["left", "right", "LEFT", "RIGHT"])
async def test_operation_started_event_uses_public_request_fields(tmp_path: Path, hand: str) -> None:
    fake = FakeSubagents()
    frames = FakeFrames()
    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        calibration_file="camera.json",
        log_dir=str(tmp_path),
        pick_cameras=PICK_CAMERAS,
    )
    orchestrator = PickPlaceOrchestrator(settings, fake, frames)
    request = PickPlaceRequest(task_type="SORTING", product_name="舒克牙膏海盐薄荷", hand=hand)

    await orchestrator.run(request, "pick", f"operation-{hand}")

    [log_dir] = list(tmp_path.iterdir())
    started = json.loads((log_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert started["event"] == "operation"
    assert started["status"] == "started"
    assert started["task_type"] == "SORTING"
    assert started["product_name"] == "舒克牙膏海盐薄荷"
    assert started["hand"] == hand.lower()
    assert "kind" not in started
    assert frames.cameras == [f"{hand.lower()}_wrist"]


@pytest.mark.asyncio
@pytest.mark.parametrize("hand", ["left", "right", "LEFT", "RIGHT"])
async def test_shortage_pick_always_uses_head_camera(tmp_path: Path, hand: str) -> None:
    fake = FakeSubagents()
    frames = FakeFrames()
    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        calibration_file="camera.json",
        log_dir=str(tmp_path),
        pick_cameras=PICK_CAMERAS,
        shortage_pick_camera="head",
    )
    orchestrator = PickPlaceOrchestrator(settings, fake, frames)
    request = PickPlaceRequest(task_type="SHORTAGE", product_name="舒克牙膏海盐薄荷", hand=hand)

    await orchestrator.run(request, "pick", f"shortage-pick-{hand}")

    assert frames.cameras == ["head"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "hand", "expected_camera"),
    [
        ("pick", "left", "left_wrist"),
        ("pick", "right", "right_wrist"),
        ("place", "left", "head"),
        ("place", "right", "head"),
    ],
)
async def test_misplaced_camera_policy(
    tmp_path: Path, kind: str, hand: str, expected_camera: str
) -> None:
    fake = FakeSubagents()
    frames = FakeFrames()
    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        calibration_file="camera.json",
        log_dir=str(tmp_path),
        pick_cameras=PICK_CAMERAS,
        place_camera="head",
    )
    request_data = {
        "task_type": "MISPLACED",
        "product_name": "舒克牙膏海盐薄荷",
        "hand": hand,
    }
    if kind == "place":
        request_data.update(
            location_id="H1_F_L_INSPECT", pose_type="SHELF_VIEW_UPPER"
        )
    request = PickPlaceRequest(**request_data)

    await PickPlaceOrchestrator(settings, fake, frames).run(
        request, kind, f"misplaced-{kind}-{hand}"
    )

    assert frames.cameras == [expected_camera]
    expected_calls = [(kind, "locate"), (kind, "pose")]
    if kind == "place":
        expected_calls.append((kind, "prepare"))
    expected_calls.append((kind, "execute"))
    assert fake.calls == expected_calls


@pytest.mark.parametrize(
    ("pick_cameras", "missing_hand"),
    [
        ({"right": "right_wrist"}, "left"),
        ({"left": "left_wrist"}, "right"),
    ],
)
def test_settings_require_both_pick_cameras(
    pick_cameras: dict[str, str], missing_hand: str
) -> None:
    with pytest.raises(ValueError, match=rf"pick_cameras 缺少配置: {missing_hand}"):
        PickPlaceSettings(
            perception_url="http://perception",
            manipulation_url="http://manipulation",
            camera_url="http://camera",
            pick_cameras=pick_cameras,
        )


def test_settings_require_pick_cameras() -> None:
    with pytest.raises(ValueError, match="pick_cameras"):
        PickPlaceSettings(
            perception_url="http://perception",
            manipulation_url="http://manipulation",
            camera_url="http://camera",
        )


@pytest.mark.asyncio
async def test_pick_runs_subagents_once_and_reuses_idempotency_result() -> None:
    fake = FakeSubagents()
    app = make_app(fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://pick-place"
    ) as client:
        payload = {"task_type": "SORTING", "product_name": "可口可乐", "hand": "LEFT"}
        first = await client.post("/pick", json=payload, headers={"Idempotency-Key": "task-1"})
        second = await client.post("/pick", json=payload, headers={"Idempotency-Key": "task-1"})

    assert first.status_code == second.status_code == 200
    assert first.json() == {"status": "SUCCEEDED"}
    assert fake.calls == [("pick", "locate"), ("pick", "pose"), ("pick", "execute")]


@pytest.mark.asyncio
async def test_pick_rejects_request_without_hand() -> None:
    fake = FakeSubagents()
    app = make_app(fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://pick-place"
    ) as client:
        response = await client.post(
            "/pick",
            json={"task_type": "SORTING", "product_name": "可口可乐"},
            headers={"Idempotency-Key": "task-without-hand"},
        )

    assert response.status_code == 422
    assert fake.calls == []


@pytest.mark.asyncio
async def test_pick_rejects_invalid_level() -> None:
    fake = FakeSubagents()
    app = make_app(fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://pick-place"
    ) as client:
        response = await client.post(
            "/pick",
            json={
                "task_type": "SORTING",
                "product_name": "可口可乐",
                "hand": "left",
                "level": "1",
            },
            headers={"Idempotency-Key": "task-invalid-level"},
        )

    assert response.status_code == 422
    assert fake.calls == []


@pytest.mark.asyncio
async def test_place_maps_internal_sequence_and_rejects_key_conflict() -> None:
    fake = FakeSubagents()
    app = make_app(fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://pick-place"
    ) as client:
        first = await client.post(
            "/place",
            json={
                "task_type": "SHORTAGE",
                "product_name": "矿泉水",
                "hand": "right",
                "location_id": "H1_F_L_INSPECT",
                "pose_type": "SHELF_VIEW_UPPER",
                "slot_id": "H1_L01_C01",
            },
            headers={"Idempotency-Key": "task-2"},
        )
        conflict = await client.post(
            "/place",
            json={
                "task_type": "SHORTAGE",
                "product_name": "牛奶",
                "hand": "right",
                "location_id": "H1_F_L_INSPECT",
                "pose_type": "SHELF_VIEW_UPPER",
                "slot_id": "H1_L01_C01",
            },
            headers={"Idempotency-Key": "task-2"},
        )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert fake.calls == [
        ("place", "locate"),
        ("place", "pose"),
        ("place", "prepare"),
        ("place", "execute"),
    ]


@pytest.mark.asyncio
async def test_place_uses_up_reference_pose_before_prepare_and_release() -> None:
    fake = FakeSubagents()
    fake.estimated_pose = [10, 20, 30, 0, 0, 0]
    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_file="camera.json",
    )
    request = PickPlaceRequest(
        task_type="SHORTAGE",
        product_name="矿泉水",
        hand="right",
        location_id="H1_F_L_INSPECT",
        pose_type="SHELF_VIEW_UPPER",
        slot_id="H1_L01_C01",
    )

    result = await PickPlaceOrchestrator(settings, fake, FakeFrames()).run(
        request, "place", "translated-place"
    )

    assert result.status == "SUCCEEDED"
    assert fake.calls == [
        ("place", "locate"),
        ("place", "pose"),
        ("place", "prepare"),
        ("place", "execute"),
    ]
    assert fake.executed_poses == [[10.0, 20.0, 30.0, 0.0, 0.0, 0.0]]
    assert fake.pose_log_suffixes == ["reference_01"]


@pytest.mark.asyncio
async def test_place_estimates_both_references_before_synthesizing_and_releasing() -> None:
    fake = FakeSubagents()
    fake.place_direction = "both"
    fake.estimated_poses = [
        [10, 20, 30, 0, 0, 0],
        [30, 40, 50, 0, 0, math.pi / 2],
    ]
    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_file="camera.json",
    )
    request = PickPlaceRequest(
        task_type="MISPLACED",
        product_name="矿泉水",
        hand="left",
        location_id="H1_F_L_INSPECT",
        pose_type="SHELF_VIEW_UPPER",
    )

    result = await PickPlaceOrchestrator(settings, fake, FakeFrames()).run(
        request, "place", "both-place"
    )

    assert result.status == "SUCCEEDED"
    assert fake.calls == [
        ("place", "locate"),
        ("place", "pose"),
        ("place", "pose"),
        ("place", "prepare"),
        ("place", "execute"),
    ]
    assert fake.pose_log_suffixes == ["reference_01", "reference_02"]
    assert fake.executed_poses == [pytest.approx([20, 30, 40, 0, 0, math.pi / 4])]


@pytest.mark.asyncio
async def test_sorting_place_skips_vision_and_uses_fixed_release_payload() -> None:
    requests: list[httpx.Request] = []

    async def handler(http_request: httpx.Request) -> httpx.Response:
        requests.append(http_request)
        return httpx.Response(
            200,
            json={
                "status": "SUCCEEDED",
                "executed": True,
                "operation": "RELEASE",
                "message": "配送桌放置、松开夹爪并返回原手臂位置完成",
            },
        )

    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_file="camera.json",
    )
    request = PickPlaceRequest(task_type="SORTING", product_name="可口可乐", hand="left", product_type="can")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        orchestrator = PickPlaceOrchestrator(settings, SubagentClient(settings, client), FakeFrames())
        result = await orchestrator.run(request, "place", "release-left-001")

    assert result.status == "SUCCEEDED"
    assert len(requests) == 1
    assert requests[0].url.path == "/manipulation/release"
    assert requests[0].headers["Idempotency-Key"] == "release-left-001:execute"
    assert json.loads(requests[0].content) == {
        "task_type": "SORTING",
        "product_name": "可口可乐",
        "hand": "LEFT",
        "pose": [0, 0, 0, 0, 0, 0],
        "frame": "camera",
        "pose_unit": "mm_rad",
        "rotation_order": "zyx",
    }


@pytest.mark.asyncio
async def test_sorting_release_preserves_robot_planning_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "error_code": "EXECUTION_FAILED",
                "message": "MoveL 规划失败，错误码 -1022",
            },
        )

    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://robot:8084",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_file="camera.json",
    )
    request = PickPlaceRequest(
        task_type="SORTING", product_name="水溶C100瓶装", hand="RIGHT"
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ServiceError) as error:
            await SubagentClient(settings, client).execute(
                request,
                "place",
                PoseResponse(pose=[0, 0, 0, 0, 0, 0]),
                "task1-place",
            )

    assert error.value.code == "EXECUTION_FAILED"
    assert error.value.message == "/manipulation/release: MoveL 规划失败，错误码 -1022"
    assert error.value.failed_interface == "manipulation_release"
    assert error.value.url == "http://robot:8084/manipulation/release"


@pytest.mark.asyncio
async def test_missing_idempotency_key_and_downstream_failure_are_reported() -> None:
    fake = FakeSubagents()
    fake.fail_execute = True
    app = make_app(fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://pick-place"
    ) as client:
        missing = await client.post(
            "/pick", json={"task_type": "SORTING", "product_name": "可口可乐", "hand": "left"}
        )
        failed = await client.post(
            "/pick",
            json={"task_type": "SORTING", "product_name": "可口可乐", "hand": "left"},
            headers={"Idempotency-Key": "task-3"},
        )

    assert missing.status_code == 400
    assert failed.status_code == 502
    assert failed.json()["error_code"] == "EXECUTION_FAILED"


@pytest.mark.asyncio
async def test_explicit_execution_failure_response_includes_pose() -> None:
    fake = FakeSubagents()
    url = "http://robot:8084/manipulation/grasp"

    async def fail_execute(
        request: PickPlaceRequest,
        kind: str,
        pose: PoseResponse,
        operation_key: str,
    ) -> None:
        raise ServiceError(
            "EXECUTION_FAILED",
            "grasp failed",
            failed_interface="manipulation_grasp",
            url=url,
        )

    fake.execute = fail_execute  # type: ignore[method-assign]
    app = make_app(fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://pick-place"
    ) as client:
        response = await client.post(
            "/pick",
            json={"task_type": "SORTING", "product_name": "可口可乐", "hand": "right"},
            headers={"Idempotency-Key": "execution-error"},
        )

    assert response.status_code == 502
    assert response.json() == {
        "error_code": "EXECUTION_FAILED",
        "message": "grasp failed",
        "failed_interface": "manipulation_grasp",
        "url": url,
        "pose": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }


@pytest.mark.asyncio
async def test_downstream_failure_response_exposes_interface_and_url() -> None:
    fake = FakeSubagents()
    url = "http://robot:8084/manipulation/grasp"

    async def fail_execute(
        request: PickPlaceRequest,
        kind: str,
        pose: PoseResponse,
        operation_key: str,
    ) -> None:
        raise ServiceError(
            "NETWORK_ERROR",
            f"{url}: All connection attempts failed",
            failed_interface="manipulation_grasp",
            url=url,
        )

    fake.execute = fail_execute  # type: ignore[method-assign]
    app = make_app(fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://pick-place"
    ) as client:
        response = await client.post(
            "/pick",
            json={"task_type": "SORTING", "product_name": "可口可乐", "hand": "right"},
            headers={"Idempotency-Key": "network-error"},
        )

    assert response.status_code == 502
    assert response.json() == {
        "error_code": "NETWORK_ERROR",
        "message": f"{url}: All connection attempts failed",
        "failed_interface": "manipulation_grasp",
        "url": url,
    }


@pytest.mark.asyncio
async def test_camera_provider_uses_color_snapshot_and_depth_stream(tmp_path: Path) -> None:
    jpeg = b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x02\x00\x03\x03\x01\x11\x00\xff\xd9"
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path + "?" + request.url.query.decode())
        if request.url.path.endswith("/snapshot"):
            return httpx.Response(200, content=jpeg)
        if request.url.path.endswith("/stream"):
            return httpx.Response(200, content=b"header" + jpeg + b"tail")
        return httpx.Response(404)

    settings = PickPlaceSettings(
        perception_url="http://perception",
        pose_estimation_url="http://pose-estimation",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_file="camera.json",
        temp_dir=str(tmp_path),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        frame = await CameraFrameProvider(settings, client).capture(
            "head", [100, 200, 500, 700], "test"
        )
    assert requests == [
        "/camera/snapshot?camera=head&type=color",
        "/camera/stream?camera=head&type=depth",
    ]
    assert Path(frame.rgb).suffix == ".jpg"
    assert Path(frame.depth).suffix == ".jpg"
    assert Path(frame.depth).read_bytes() == jpeg
    assert Path(frame.mask).read_bytes().splitlines()[1] == b"3 2"
    cleanup = Path(frame.cleanup_path or "")
    for child in cleanup.iterdir():
        child.unlink()
    cleanup.rmdir()


@pytest.mark.asyncio
async def test_camera_provider_extracts_raw_uint16_depth_multipart(tmp_path: Path) -> None:
    jpeg = b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x02\x00\x03\x03\x01\x11\x00\xff\xd9"
    # 3x2 个 little-endian uint16 深度值，模拟相机网关的 application/octet-stream 帧。
    raw_depth = b"".join(value.to_bytes(2, "little") for value in range(1, 7))
    multipart = (
        b"--depth-boundary\r\n"
        b"Content-Type: application/octet-stream\r\n"
        + f"Content-Length: {len(raw_depth)}\r\n\r\n".encode()
        + raw_depth
        + b"\r\n--depth-boundary\r\n"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/snapshot"):
            return httpx.Response(200, content=jpeg)
        return httpx.Response(
            200,
            content=multipart,
            headers={"content-type": "multipart/x-mixed-replace; boundary=depth-boundary"},
        )

    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_file="camera.json",
        temp_dir=str(tmp_path),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        frame = await CameraFrameProvider(settings, client).capture(
            "head", [0, 0, 1000, 1000], "raw-depth"
        )

    depth = Path(frame.depth).read_bytes()
    assert depth.startswith(b"\x89PNG")
    assert int.from_bytes(depth[16:20], "big") == 3
    assert int.from_bytes(depth[20:24], "big") == 2
    assert depth[24] == 16
    assert depth[25] == 0
    cleanup = Path(frame.cleanup_path or "")
    for child in cleanup.iterdir():
        child.unlink()
    cleanup.rmdir()


@pytest.mark.asyncio
async def test_camera_provider_prepares_current_rgbd_and_multiple_reference_masks(
    tmp_path: Path,
) -> None:
    current_dir = tmp_path / "place" / "run"
    current_dir.mkdir(parents=True)
    current_rgb = current_dir / "current_rgb.jpg"
    current_rgb.write_bytes(
        b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x02\x00\x03\x03\x01\x11\x00\xff\xd9"
    )
    calibration = tmp_path / "head.json"
    calibration.write_text(
        json.dumps({"cam_K": [100, 0, 1, 0, 100, 1, 0, 0, 1]}),
        encoding="utf-8",
    )
    np.save(
        current_dir / "current_depth_mm.npy",
        np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint16),
    )
    masks = [
        service_module._make_png(3, 2, pixels)
        for pixels in (b"\x00\xff\xff\xff" * 2, b"\xff\x00\xff\xff" * 2)
    ]
    located = PlaceLocateResponse(
        name="矿泉水",
        bbox=[[0, 0, 2, 2], [1, 0, 3, 2]],
        mask=[base64.b64encode(mask).decode("ascii") for mask in masks],
        direction="both",
        image_path="/task0/rgb.jpg",
        current_image_path=str(current_rgb),
        level="L1",
    )
    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_file=str(calibration),
        temp_dir=str(tmp_path / "staging"),
    )
    operation_log = tmp_path / "operation-log"
    operation_log.mkdir()
    log_token = service_module._ACTIVE_LOG_DIR.set(operation_log)
    try:
        async with httpx.AsyncClient() as client:
            frames = await CameraFrameProvider(settings, client).prepare_place_references(
                located, "head", "reference"
            )
    finally:
        service_module._ACTIVE_LOG_DIR.reset(log_token)

    assert len(frames) == 2
    assert all(Path(frame.rgb).read_bytes() == current_rgb.read_bytes() for frame in frames)
    depth = Path(frames[0].depth).read_bytes()
    assert depth.startswith(b"\x89PNG")
    assert int.from_bytes(depth[16:20], "big") == 3
    assert int.from_bytes(depth[20:24], "big") == 2
    assert depth[24:26] == bytes([16, 0])
    assert [Path(frame.mask).read_bytes() for frame in frames] == masks
    assert frames[0].cleanup_path is not None
    assert frames[1].cleanup_path is None
    assert (operation_log / "current" / "rgb.jpg").read_bytes() == current_rgb.read_bytes()
    assert (operation_log / "current" / "mask_01.png").read_bytes() == masks[0]
    assert (operation_log / "current" / "mask_02.png").read_bytes() == masks[1]
    current_input = json.loads((operation_log / "current" / "input.json").read_text())
    assert current_input["image_size"] == [3, 2]
    assert current_input["source_rgb"] == str(current_rgb)
    assert current_input["source_depth"] == str(current_dir / "current_depth_mm.npy")
    assert current_input["direction"] == "both"
    assert (operation_log / "current" / "head.json").is_file()
    cleanup = Path(frames[0].cleanup_path or "")
    for child in cleanup.iterdir():
        child.unlink()
    cleanup.rmdir()


@pytest.mark.asyncio
async def test_subagent_pose_uses_multipart_files_for_real_manipulation_api(tmp_path: Path) -> None:
    paths = {
        "rgb": tmp_path / "rgb.jpg",
        "depth": tmp_path / "depth.png",
        "camera": tmp_path / "camera.json",
        "mask": tmp_path / "mask.pgm",
    }
    contents = {
        "rgb": b"rgb-bytes",
        "depth": b"depth-bytes",
        "camera": b'{"cam_K":[1,0,0,0,1,0,0,0,1],"depth_scale":1.0}',
        "mask": b"mask-bytes",
    }
    for field, path in paths.items():
        path.write_bytes(contents[field])

    settings = PickPlaceSettings(
        perception_url="http://perception",
        pose_estimation_url="http://pose-estimation",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_file=str(paths["camera"]),
    )
    frame = FrameBundle(**{field: str(path) for field, path in paths.items()})
    request = PickPlaceRequest(task_type="SORTING", product_name="可口可乐", hand="left")
    seen: dict[str, str] = {}

    async def handler(http_request: httpx.Request) -> httpx.Response:
        body = http_request.content
        assert http_request.url.host == "pose-estimation"
        assert http_request.url.path == "/manipulation/pick_pose"
        assert http_request.headers["content-type"].startswith("multipart/form-data;")
        for field, filename in (("rgb", "rgb.jpg"), ("depth", "depth.png"), ("camera", "camera.json"), ("mask", "mask.pgm")):
            assert f'name="{field}"'.encode() in body
            assert f'filename="{filename}"'.encode() in body
            assert contents[field] in body
        assert b'name="product_name"' in body
        assert "可口可乐".encode() in body
        seen["ok"] = "yes"
        return httpx.Response(200, json={"pose": [1, 2, 3, 4, 5, 6]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await SubagentClient(settings, client).estimate_pose(request, "pick", frame)

    assert seen == {"ok": "yes"}
    assert result.pose == [1, 2, 3, 4, 5, 6]


@pytest.mark.asyncio
async def test_subagent_pose_preserves_downstream_error_message(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths = {
        field: tmp_path / filename
        for field, filename in {
            "rgb": "rgb.jpg",
            "depth": "depth.jpg",
            "camera": "camera.json",
            "mask": "mask.png",
        }.items()
    }
    for path in paths.values():
        path.write_bytes(b"test-input")

    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_file=str(paths["camera"]),
    )
    frame = FrameBundle(**{field: str(path) for field, path in paths.items()})
    request = PickPlaceRequest(task_type="SORTING", product_name="可口可乐", hand="left")

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error_code": "INVALID_REQUEST",
                "message": "No valid depth values inside the selected mask",
            },
        )

    caplog.set_level("ERROR", logger="pick_place_service.service")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ServiceError) as error:
            await SubagentClient(settings, client).estimate_pose(request, "pick", frame)

    assert error.value.code == "INVALID_REQUEST"
    assert "No valid depth values inside the selected mask" in error.value.message
    assert "No valid depth values inside the selected mask" in caplog.text


@pytest.mark.asyncio
async def test_connection_failure_records_interface_url_and_transport_error(tmp_path: Path) -> None:
    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://robot:8084",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_file="camera.json",
    )
    request = PickPlaceRequest(task_type="SORTING", product_name="可口可乐", hand="right")
    pose = PoseResponse(pose=[1, 2, 3, 4, 5, 6])

    async def handler(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=http_request)

    token = service_module._ACTIVE_LOG_DIR.set(tmp_path)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ServiceError) as error:
                await SubagentClient(settings, client).execute(request, "pick", pose, "operation")
    finally:
        service_module._ACTIVE_LOG_DIR.reset(token)

    assert error.value.code == "NETWORK_ERROR"
    assert error.value.failed_interface == "manipulation_grasp"
    assert error.value.url == "http://robot:8084/manipulation/grasp"
    assert error.value.url in error.value.message
    interface_dir = tmp_path / "interfaces" / "manipulation_grasp"
    logged_request = json.loads((interface_dir / "request.json").read_text(encoding="utf-8"))
    logged_response = json.loads((interface_dir / "response.json").read_text(encoding="utf-8"))
    assert logged_request["url"] == error.value.url
    assert logged_response["status_code"] == 502
    assert logged_response["body"]["transport_error"] is True
    assert logged_response["body"]["url"] == error.value.url


def test_action_payload_matches_8084_execution_contract() -> None:
    request = PickPlaceRequest(
        task_type="SORTING",
        product_name="可口可乐",
        hand="RIGHT",
        product_type="can",
    )
    pose = PoseResponse(
        pose=[1, 2, 3, 4, 5, 6],
        frame="camera",
        pose_unit="mm_rad",
        rotation_order="zyx",
    )

    assert action_payload(request, pose) == {
        "task_type": "SORTING",
        "product_name": "可口可乐",
        "pose": [1, 2, 3, 4, 5, 6],
        "hand": "right",
        "frame": "camera",
        "pose_unit": "mm_rad",
        "rotation_order": "zyx",
    }


def test_calibration_file_is_selected_by_camera() -> None:
    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_files={
            "head": "config/camera/head.json",
            "left_wrist": "config/camera/left_wrist.json",
            "right_wrist": "config/camera/right_wrist.json",
        },
    )

    assert settings.calibration_for("head") == "config/camera/head.json"
    assert settings.calibration_for("left_wrist") == "config/camera/left_wrist.json"
    assert settings.calibration_for("right_wrist") == "config/camera/right_wrist.json"


@pytest.mark.asyncio
async def test_subagent_health_checks_pose_and_manipulation_services() -> None:
    paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        paths.append(f"{request.url.host}{request.url.path}")
        if request.url.path == "/camera/list":
            return httpx.Response(200, json={"cameras": []})
        return httpx.Response(200, json={"status": "READY"})

    settings = PickPlaceSettings(
        perception_url="http://perception",
        pose_estimation_url="http://pose-estimation",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_file="camera.json",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await SubagentClient(settings, client).health()

    assert set(paths[:4]) == {
        "perception/perception/health",
        "pose-estimation/manipulation/health",
        "manipulation/manipulation/health",
        "camera/camera/health",
    }
    assert paths[4] == "camera/camera/list"


@pytest.mark.asyncio
async def test_subagent_locate_uses_formal_locate_contract() -> None:
    settings = PickPlaceSettings(
        perception_url="http://legacy-perception",
        locate_url="http://formal-locate",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_file="camera.json",
    )
    request = PickPlaceRequest(
        task_type="SORTING", product_name="可口可乐", hand="LEFT", level="L2"
    )

    async def handler(http_request: httpx.Request) -> httpx.Response:
        assert http_request.url.path == "/perception/pick/locate"
        assert http_request.headers["content-type"] == "application/json"
        assert http_request.content.decode("utf-8") == (
            '{"task_type":"SORTING","product_name":"可口可乐",'
            '"hand":"left","level":"L2"}'
        )
        return httpx.Response(
            200,
            json={
                "product_name": "可口可乐",
                "bbox": [100, 200, 300, 500],
                "mask": "c2VnbWVudGF0aW9u",
                "image_path": "/tmp/locate/rgb.jpg",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await SubagentClient(settings, client).locate(request, "pick")

    assert result.bbox == [100, 200, 300, 500]
    assert result.mask == "c2VnbWVudGF0aW9u"


@pytest.mark.asyncio
async def test_subagent_place_locate_sends_exact_context_and_parses_new_fields() -> None:
    request = PickPlaceRequest(
        task_type="SHORTAGE",
        product_name="商品名",
        hand="LEFT",
        location_id="H1_F_L_INSPECT",
        pose_type="SHELF_VIEW_UPPER",
        slot_id="H1_L01_C01",
    )
    async def handler(http_request: httpx.Request) -> httpx.Response:
        assert http_request.url.path == "/perception/place/locate"
        assert json.loads(http_request.content) == {
            "task_type": "SHORTAGE",
            "product_name": "商品名",
            "location_id": "H1_F_L_INSPECT",
            "pose_type": "SHELF_VIEW_UPPER",
            "slot_id": "H1_L01_C01",
        }
        return httpx.Response(
            200,
            json={
                "name": "商品名",
                "slot_id": "H1_L01_C01",
                "bbox": [[10, 20, 30, 40], [50, 20, 70, 40]],
                "mask": ["cG5nLTE=", "cG5nLTI="],
                "direction": "both",
                "image_path": "/output/task0/reference/rgb.jpg",
                "current_image_path": "/output/place/run/current_rgb.jpg",
                "level": "L3",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await SubagentClient(_locate_settings(), client).locate_place(request)

    assert result.name == "商品名"
    assert result.slot_id == "H1_L01_C01"
    assert result.direction == "both"
    assert result.bbox == [[10, 20, 30, 40], [50, 20, 70, 40]]
    assert result.mask == ["cG5nLTE=", "cG5nLTI="]
    assert result.current_image_path == "/output/place/run/current_rgb.jpg"
    assert result.level == "L3"


@pytest.mark.asyncio
async def test_subagent_prepare_place_uses_returned_level() -> None:
    async def handler(http_request: httpx.Request) -> httpx.Response:
        assert http_request.url.path == "/pose/prepare"
        assert http_request.headers["Idempotency-Key"] == "place-1:place-ready"
        assert json.loads(http_request.content) == {
            "pose_type": "SHELF_PLACE_READY",
            "shelf_level": "L4",
        }
        return httpx.Response(200, json={"status": "SUCCEEDED"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await SubagentClient(_locate_settings(), client).prepare_place("L4", "place-1")


@pytest.mark.asyncio
async def test_subagent_locate_omits_level_when_not_provided() -> None:
    request = PickPlaceRequest(
        task_type="SHORTAGE", product_name="矿泉水", hand="right"
    )

    async def handler(http_request: httpx.Request) -> httpx.Response:
        assert json.loads(http_request.content) == {
            "task_type": "SHORTAGE",
            "product_name": "矿泉水",
            "hand": "right",
        }
        return httpx.Response(
            200,
            json={"product_name": "矿泉水", "bbox": [100, 200, 300, 500]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await SubagentClient(_locate_settings(), client).locate(request, "pick")


def test_normalize_product_name_strips_spaces_and_symbols() -> None:
    assert normalize_product_name("Lay's乐事薯片意大利香浓红烩味") == "Lays乐事薯片意大利香浓红烩味"
    assert normalize_product_name("Lays乐事薯片意大利香浓红烩味") == "Lays乐事薯片意大利香浓红烩味"
    assert normalize_product_name("呀！土豆番茄酱味") == "呀土豆番茄酱味"
    assert normalize_product_name("Lays 乐事") == "Lays乐事"


def _locate_settings() -> PickPlaceSettings:
    return PickPlaceSettings(
        perception_url="http://legacy-perception",
        locate_url="http://formal-locate",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        pick_cameras=PICK_CAMERAS,
        calibration_file="camera.json",
    )


@pytest.mark.asyncio
async def test_subagent_locate_accepts_product_name_with_symbols() -> None:
    request = PickPlaceRequest(
        task_type="SORTING",
        product_name="Lays乐事薯片意大利香浓红烩味",
        hand="LEFT",
    )

    async def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "product_name": "Lay's乐事薯片意大利香浓红烩味",
                "bbox": [620, 420, 670, 455],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await SubagentClient(_locate_settings(), client).locate(request, "pick")

    assert result.product_name == "Lay's乐事薯片意大利香浓红烩味"
    assert result.bbox == [620, 420, 670, 455]


@pytest.mark.asyncio
async def test_subagent_locate_rejects_different_product_name() -> None:
    request = PickPlaceRequest(task_type="SORTING", product_name="可口可乐", hand="LEFT")

    async def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"product_name": "百事可乐", "bbox": [100, 200, 300, 500]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ServiceError, match="invalid pick locate response"):
            await SubagentClient(_locate_settings(), client).locate(request, "pick")


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("both", [20, 30, 40]),
        ("left", [50, 60, 70]),
        ("right", [-10, 0, 10]),
    ],
)
def test_synthesize_place_pose_uses_horizontal_reference_geometry(
    direction: str, expected: list[float]
) -> None:
    references = [
        PoseResponse(pose=[10, 20, 30, 0, 0, 0]),
        PoseResponse(pose=[30, 40, 50, 0, 0, 0]),
    ]

    result = synthesize_place_pose(references, direction)  # type: ignore[arg-type]

    assert result.pose == pytest.approx([*expected, 0, 0, 0])
    assert (result.frame, result.pose_unit, result.rotation_order) == (
        "camera",
        "mm_rad",
        "zyx",
    )


def test_synthesize_place_pose_uses_up_reference_directly() -> None:
    pose = PoseResponse(pose=[10, 20, 30, 0.1, -0.2, 0.3])

    assert synthesize_place_pose([pose], "up").pose == pytest.approx(pose.pose)


def test_synthesize_place_pose_averages_reference_rotations() -> None:
    references = [
        PoseResponse(pose=[0, 0, 0, 0, 0, 0]),
        PoseResponse(pose=[20, 0, 0, 0, 0, math.pi / 2]),
    ]

    assert synthesize_place_pose(references, "both").pose == pytest.approx(
        [10, 0, 0, 0, 0, math.pi / 4]
    )


@pytest.mark.parametrize(
    ("poses", "direction", "message"),
    [
        ([PoseResponse(pose=[0, 0, 0, 0, 0, 0])], "both", "requires 2"),
        ([PoseResponse(pose=[0, 0, 0, 0, 0])], "up", "six finite"),
        ([PoseResponse(pose=[0, 0, 0, 0, 0, float("nan")])], "up", "six finite"),
        ([PoseResponse(pose=[0, 0, 0, 0, 0, 0], frame="base")], "up", "frame"),
        ([PoseResponse(pose=[0, 0, 0, 0, 0, 0], pose_unit="m_rad")], "up", "unit"),
        ([PoseResponse(pose=[0, 0, 0, 0, 0, 0], rotation_order="xyz")], "up", "order"),
    ],
)
def test_synthesize_place_pose_rejects_invalid_references(
    poses: list[PoseResponse], direction: str, message: str
) -> None:
    with pytest.raises(ServiceError, match=message):
        synthesize_place_pose(poses, direction)  # type: ignore[arg-type]
