from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from test1_service.camera_frames import make_png
from test1_service.client import Test1Client as Client
from test1_service.models import (
    Hand,
    Test1ServiceError as ServiceError,
    Test1Settings as Settings,
    Test1Timeouts as Timeouts,
)
from test1_service.service import Test1Orchestrator as Orchestrator


CONFIG_FILE = Path(__file__).resolve().parents[1] / "config/runtime.production.yaml"


class CameraMock:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.fail_camera: str | None = None
        self.right_depth_online = True
        self.rgb = make_png(1, 1, b"\x00\x00\x01")
        self.depth = b"\x01\x00"

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in {"/navigation/health", "/pose/health", "/camera/health"}:
            return httpx.Response(200, json={"status": "READY"})
        if path == "/camera/list":
            return httpx.Response(
                200,
                json={
                    "cameras": [
                        {
                            "id": "left_wrist",
                            "online": True,
                            "streams": [
                                {"type": "color", "online": True},
                                {"type": "depth", "online": True},
                            ],
                        },
                        {
                            "id": "right_wrist",
                            "online": True,
                            "streams": [
                                {"type": "color", "online": True},
                                {
                                    "type": "depth",
                                    "online": self.right_depth_online,
                                },
                            ],
                        },
                    ]
                },
            )
        if path in {"/navigation/navigate", "/pose/prepare"}:
            return httpx.Response(200, json={"status": "SUCCEEDED"})
        camera = request.url.params.get("camera")
        if path == "/camera/snapshot":
            if camera == self.fail_camera:
                return httpx.Response(500, json={"error_code": "CAMERA_FAILED"})
            return httpx.Response(
                200, content=self.rgb, headers={"content-type": "image/png"}
            )
        if path == "/camera/stream":
            boundary = "depth-boundary"
            content = (
                f"--{boundary}\r\n"
                "Content-Type: application/octet-stream\r\n"
                f"Content-Length: {len(self.depth)}\r\n\r\n"
            ).encode() + self.depth + f"\r\n--{boundary}--\r\n".encode()
            return httpx.Response(
                200,
                content=content,
                headers={
                    "content-type": (
                        f"multipart/x-mixed-replace; boundary={boundary}"
                    )
                },
            )
        return httpx.Response(404)


def settings(tmp_path: Path, points: list[str] | None = None) -> Settings:
    return Settings(
        services={
            "navigation": "http://navigation.local",
            "pose": "http://pose.local",
            "camera": "http://camera.local",
        },
        timeouts=Timeouts(
            connect_seconds=0.1,
            health_seconds=0.2,
            navigation_seconds=0.2,
            pose_seconds=0.2,
            camera_seconds=0.2,
        ),
        inspection_points=points or ["POINT_ONE"],
        start_target_id="start",
        capture_settle_seconds=0,
        output_dir=str(tmp_path / "output"),
        log_dir=str(tmp_path / "logs"),
    )


def payload(request: httpx.Request) -> dict[str, str]:
    return json.loads(request.content) if request.content else {}


@pytest.mark.asyncio
async def test_test1_runs_fixed_levels_and_both_wrist_cameras(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("test1_service.service.asyncio.sleep", record_sleep)
    mock = CameraMock()
    task_settings = settings(tmp_path, ["POINT_ONE", "POINT_TWO"])
    task_settings.capture_settle_seconds = 2
    async with Client(task_settings, transport=mock.transport) as client:
        result = await Orchestrator(task_settings, client).run("test-run")

    assert result.status == "SUCCEEDED"
    assert len(result.captures) == 20
    assert delays == [2] * 10
    assert [
        payload(request)["target_id"]
        for request in mock.requests
        if request.url.path == "/navigation/navigate"
    ] == ["start", "POINT_ONE", "POINT_TWO", "start"]

    action_requests = [
        request
        for request in mock.requests
        if request.url.path in {"/pose/prepare", "/navigation/navigate"}
    ]
    navigation_indexes = [
        index
        for index, request in enumerate(action_requests)
        if request.url.path == "/navigation/navigate"
    ]
    assert len(navigation_indexes) == 4
    assert all(
        action_requests[index - 1].url.path == "/pose/prepare"
        and payload(action_requests[index - 1]) == {"pose_type": "START_POSITION"}
        for index in navigation_indexes
    )

    shelf_poses = [
        payload(request)
        for request in mock.requests
        if request.url.path == "/pose/prepare"
        and payload(request).get("pose_type") == "SHELF_INSPECT"
    ]
    expected_levels = ["L1", "L2", "L3", "L4", "L5"] * 2
    assert [item["shelf_level"] for item in shelf_poses] == expected_levels

    camera_requests = [
        request
        for request in mock.requests
        if request.url.path in {"/camera/snapshot", "/camera/stream"}
    ]
    expected_camera_cycle = [
        ("/camera/snapshot", "left_wrist", "color"),
        ("/camera/stream", "left_wrist", "depth"),
        ("/camera/snapshot", "right_wrist", "color"),
        ("/camera/stream", "right_wrist", "depth"),
    ]
    assert [
        (request.url.path, request.url.params["camera"], request.url.params["type"])
        for request in camera_requests
    ] == expected_camera_cycle * 10

    directories = [Path(capture.directory) for capture in result.captures]
    assert [directory.name for directory in directories[:4]] == [
        "POINT_ONE-L1-LEFT",
        "POINT_ONE-L1-RIGHT",
        "POINT_ONE-L2-LEFT",
        "POINT_ONE-L2-RIGHT",
    ]
    assert len(set(directories)) == 20
    assert all({path.name for path in directory.iterdir()} == {"rgb.png", "depth.png"} for directory in directories)
    assert all((directory / "depth.png").read_bytes().startswith(b"\x89PNG") for directory in directories)


@pytest.mark.asyncio
async def test_test1_failure_stops_without_returning_to_start(tmp_path: Path) -> None:
    mock = CameraMock()
    mock.fail_camera = "left_wrist"
    task_settings = settings(tmp_path)
    async with Client(task_settings, transport=mock.transport) as client:
        with pytest.raises(ServiceError) as error:
            await Orchestrator(task_settings, client).run("failed-run")

    assert error.value.step == "采集 POINT_ONE L1 LEFT RGB-D"
    assert [
        payload(request)["target_id"]
        for request in mock.requests
        if request.url.path == "/navigation/navigate"
    ] == ["start", "POINT_ONE"]
    batch = next((tmp_path / "output").iterdir())
    assert not [path for path in batch.iterdir() if not path.name.startswith(".")]


@pytest.mark.asyncio
async def test_test1_health_requires_both_wrist_depth_streams(tmp_path: Path) -> None:
    mock = CameraMock()
    mock.right_depth_online = False
    task_settings = settings(tmp_path)
    async with Client(task_settings, transport=mock.transport) as client:
        with pytest.raises(ServiceError) as error:
            await client.check_all_health()

    assert error.value.code == "CAPABILITY_NOT_READY"
    assert "camera.right_wrist.depth" in error.value.message


@pytest.mark.asyncio
async def test_test1_rejects_corrupt_color_without_publishing_capture(
    tmp_path: Path,
) -> None:
    mock = CameraMock()
    mock.rgb = b"\x89PNG\r\n\x1a\n" + b"broken-image-data"
    task_settings = settings(tmp_path)
    async with Client(task_settings, transport=mock.transport) as client:
        with pytest.raises(ServiceError) as error:
            await Orchestrator(task_settings, client).run("corrupt-run")

    assert error.value.code == "INVALID_CAMERA_FRAME"
    batch = next((tmp_path / "output").iterdir())
    assert not list(batch.iterdir())


@pytest.mark.asyncio
async def test_test1_keeps_each_run_in_a_separate_batch(tmp_path: Path) -> None:
    mock = CameraMock()
    task_settings = settings(tmp_path)
    async with Client(task_settings, transport=mock.transport) as client:
        orchestrator = Orchestrator(task_settings, client)
        first = await orchestrator.run("same-key")
        second = await orchestrator.run("same-key")

    assert first.batch_directory != second.batch_directory
    assert Path(first.batch_directory).is_dir()
    assert Path(second.batch_directory).is_dir()


def test_test1_loads_production_configuration() -> None:
    production = Settings.load(CONFIG_FILE)

    assert production.start_target_id == "start"
    assert production.capture_settle_seconds == 2
    assert production.inspection_points == [
        "H1_INSPECT",
        "H2_INSPECT",
        "H3_INSPECT",
    ]
    assert production.services.navigation == "http://192.168.200.66:8081"
    assert production.services.pose == "http://192.168.200.66:8084"
    assert production.services.camera == "http://192.168.200.66:8085"
