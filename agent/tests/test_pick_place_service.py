"""8086 取放编排服务的 API 和内部顺序测试。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from pick_place_service.app import create_app
from pick_place_service.models import (
    FrameBundle,
    LocateResponse,
    PickPlaceRequest,
    PickPlaceSettings,
    PoseResponse,
    ServiceError,
    action_payload,
)
from pick_place_service.service import CameraFrameProvider, PickPlaceOrchestrator, SubagentClient


class FakeSubagents:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_execute = False

    async def locate(self, request: PickPlaceRequest, kind: str) -> LocateResponse:
        self.calls.append((kind, "locate"))
        return LocateResponse(product_name=request.product_name, bbox=[100, 200, 300, 500])

    async def estimate_pose(
        self, request: PickPlaceRequest, kind: str, frame: FrameBundle
    ) -> PoseResponse:
        self.calls.append((kind, "pose"))
        assert frame.mask == "mask.pgm"
        return PoseResponse(pose=[1, 2, 3, 4, 5, 6])

    async def execute(
        self,
        request: PickPlaceRequest,
        kind: str,
        pose: PoseResponse,
        operation_key: str,
    ) -> None:
        self.calls.append((kind, "execute"))
        if self.fail_execute:
            raise ServiceError("EXECUTION_FAILED", "grasp failed")

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


def make_app(fake: FakeSubagents):
    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
        calibration_file="camera.json",
    )
    orchestrator = PickPlaceOrchestrator(settings, fake, FakeFrames())
    return create_app(settings, orchestrator=orchestrator)


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
        pick_cameras={"left": "left_wrist", "right": "right_wrist"},
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
    assert fake.calls == [("pick", "locate"), ("pick", "pose"), ("pick", "execute"), ("pick", "check")]


@pytest.mark.asyncio
async def test_place_maps_internal_sequence_and_rejects_key_conflict() -> None:
    fake = FakeSubagents()
    app = make_app(fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://pick-place"
    ) as client:
        first = await client.post(
            "/place",
            json={"task_type": "SHORTAGE", "product_name": "矿泉水", "hand": "right"},
            headers={"Idempotency-Key": "task-2"},
        )
        conflict = await client.post(
            "/place",
            json={"task_type": "SHORTAGE", "product_name": "牛奶", "hand": "right"},
            headers={"Idempotency-Key": "task-2"},
        )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert fake.calls == [("place", "locate"), ("place", "pose"), ("place", "execute"), ("place", "check")]


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
        "pose": [1, 2, 3, 4, 5, 6],
        "hand": "right",
        "product_type": "can",
        "frame": "camera",
        "pose_unit": "mm_rad",
        "rotation_order": "zyx",
    }


def test_calibration_file_is_selected_by_camera() -> None:
    settings = PickPlaceSettings(
        perception_url="http://perception",
        manipulation_url="http://manipulation",
        camera_url="http://camera",
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
        calibration_file="camera.json",
    )
    request = PickPlaceRequest(task_type="SORTING", product_name="可口可乐", hand="LEFT")

    async def handler(http_request: httpx.Request) -> httpx.Response:
        assert http_request.url.path == "/perception/pick/locate"
        assert http_request.headers["content-type"] == "application/json"
        assert http_request.content.decode("utf-8") == '{"task_type":"SORTING","product_name":"可口可乐","hand":"left"}'
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
