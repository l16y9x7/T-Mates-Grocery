from __future__ import annotations

import asyncio
import json
import zipfile
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from task0_service.client import Task0Client
from task0_service.models import (
    CaptureResult,
    InspectionPose,
    Task0Request,
    Task0Result,
    Task0ServiceError,
    Task0Settings,
    Task0Timeouts,
)
from task0_service.service import Task0Orchestrator
from task_service.settings import TaskServiceSettings


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def rgbd_archive(marker: str) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        bundle.writestr("rgb.jpg", b"\xff\xd8" + marker.encode() + b"\xff\xd9")
        bundle.writestr("depth_mm.npy", b"\x93NUMPY" + marker.encode())
        bundle.writestr(
            "meta.json",
            json.dumps({"camera": "head", "marker": marker}),
        )
    return output.getvalue()


class Task0Mock:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.capture_number = 0
        self.invalid_capture = False
        self.invalid_capture_number: int | None = None
        self.head_depth_online = True

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path in {
            "/navigation/health",
            "/pose/health",
            "/camera/health",
        }:
            return httpx.Response(200, json={"status": "READY"})
        if request.url.path == "/camera/list":
            return httpx.Response(
                200,
                json={
                    "cameras": [
                        {
                            "id": "head",
                            "online": True,
                            "streams": [
                                {"type": "color", "online": True},
                                {"type": "depth", "online": self.head_depth_online},
                            ],
                        }
                    ]
                },
            )
        if request.url.path in {"/navigation/navigate", "/pose/prepare"}:
            return httpx.Response(200, json={"status": "SUCCEEDED", "executed": True})
        if request.url.path == "/camera/rgbd":
            self.capture_number += 1
            if (
                self.invalid_capture
                or self.capture_number == self.invalid_capture_number
            ):
                return httpx.Response(200, content=b"not-a-zip")
            return httpx.Response(
                200,
                content=rgbd_archive(f"capture-{self.capture_number}"),
                headers={"content-type": "application/zip"},
            )
        return httpx.Response(404, json={"error_code": "UNKNOWN_ENDPOINT"})


def settings(tmp_path: Path, points: list[str] | None = None) -> Task0Settings:
    return Task0Settings(
        services={
            "navigation": "http://navigation.local",
            "pose": "http://pose.local",
            "camera": "http://camera.local",
        },
        timeouts=Task0Timeouts(
            connect_seconds=0.1,
            health_seconds=0.2,
            navigation_seconds=0.2,
            pose_seconds=0.2,
            camera_seconds=0.2,
        ),
        inspection_points=points or ["POINT_TOP", "POINT_BOTTOM"],
        start_target_id="start",
        capture_settle_seconds=0,
        camera="head",
        output_dir=str(tmp_path / "captures"),
        log_dir=str(tmp_path / "logs"),
    )


def request_payload(request: httpx.Request) -> dict[str, str]:
    return json.loads(request.content) if request.content else {}


@pytest.mark.asyncio
async def test_task0_captures_upper_and_lower_at_configured_points(tmp_path: Path) -> None:
    mock = Task0Mock()
    task_settings = settings(tmp_path)
    client = Task0Client(task_settings, transport=mock.transport)
    async with client:
        result = await Task0Orchestrator(task_settings, client).run(
            Task0Request(), "operation-1"
        )

    assert result.status == "SUCCEEDED"
    assert result.inspection_points == ["POINT_TOP", "POINT_BOTTOM"]
    assert [Path(capture.directory).name for capture in result.captures] == [
        "POINT_TOP_UPPER",
        "POINT_TOP_LOWER",
        "POINT_BOTTOM_LOWER",
        "POINT_BOTTOM_UPPER",
    ]

    task_requests = [
        request
        for request in mock.requests
        if request.url.path
        in {"/navigation/navigate", "/pose/prepare", "/camera/rgbd"}
    ]
    assert [request.url.path for request in task_requests] == [
        "/navigation/navigate",
        "/navigation/navigate",
        "/pose/prepare",
        "/camera/rgbd",
        "/pose/prepare",
        "/camera/rgbd",
        "/navigation/navigate",
        "/camera/rgbd",
        "/pose/prepare",
        "/camera/rgbd",
        "/navigation/navigate",
    ]
    assert [
        request_payload(request)["target_id"]
        for request in task_requests
        if request.url.path == "/navigation/navigate"
    ] == ["start", "POINT_TOP", "POINT_BOTTOM", "start"]
    assert [
        request_payload(request)["pose_type"]
        for request in task_requests
        if request.url.path == "/pose/prepare"
    ] == [
        "SHELF_VIEW_UPPER",
        "SHELF_VIEW_LOWER",
        "SHELF_VIEW_UPPER",
    ]
    assert all(
        dict(request.url.params) == {"camera": "head"}
        for request in task_requests
        if request.url.path == "/camera/rgbd"
    )
    assert all(
        request.headers.get("Idempotency-Key", "").startswith("operation-1:task0")
        for request in task_requests
        if request.url.path in {"/navigation/navigate", "/pose/prepare"}
    )

    for capture in result.captures:
        directory = Path(capture.directory)
        assert {path.name for path in directory.iterdir()} == {
            "rgb.jpg",
            "depth_mm.npy",
            "meta.json",
        }
        assert Path(capture.rgb_path).is_file()
        assert Path(capture.depth_path).read_bytes().startswith(b"\x93NUMPY")

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    pointer = json.loads(
        (Path(task_settings.output_dir) / "current.json").read_text(encoding="utf-8")
    )
    assert manifest["scan_id"] == result.scan_id
    assert manifest["complete"] is True
    assert manifest["capture_count"] == 4
    assert pointer["scan_id"] == result.scan_id
    assert pointer["run_directory"] == f"runs/{result.scan_id}"

    log_directories = list((tmp_path / "logs").iterdir())
    assert len(log_directories) == 1
    events = [
        json.loads(line)
        for line in (log_directories[0] / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    completed_captures = [
        event
        for event in events
        if event["event"] == "RGB-D采集" and event["status"] == "succeeded"
    ]
    assert [event["capture_number"] for event in completed_captures] == [1, 2, 3, 4]
    assert all(event["total_captures"] == 4 for event in completed_captures)
    assert any(event["event"] == "接口调用" for event in events)


@pytest.mark.asyncio
async def test_task0_waits_before_every_capture(monkeypatch, tmp_path: Path) -> None:
    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("task0_service.service.asyncio.sleep", record_sleep)
    mock = Task0Mock()
    task_settings = settings(tmp_path, ["POINT_ONE"])
    task_settings.capture_settle_seconds = 2
    client = Task0Client(task_settings, transport=mock.transport)
    async with client:
        await Task0Orchestrator(task_settings, client).run(Task0Request(), "wait-test")

    assert delays == [2, 2]


@pytest.mark.asyncio
async def test_task0_publishes_new_immutable_scan_generation(tmp_path: Path) -> None:
    mock = Task0Mock()
    task_settings = settings(tmp_path, ["POINT_ONE"])
    client = Task0Client(task_settings, transport=mock.transport)
    orchestrator = Task0Orchestrator(task_settings, client)
    async with client:
        first = await orchestrator.run(Task0Request(), "first")
        first_upper = Path(first.captures[0].directory)
        first_meta = (first_upper / "meta.json").read_text(encoding="utf-8")
        second = await orchestrator.run(Task0Request(), "second")

    second_upper = Path(second.captures[0].directory)
    assert first.scan_id != second.scan_id
    assert first_upper.is_dir()
    assert (first_upper / "meta.json").read_text(encoding="utf-8") == first_meta
    second_metadata = json.loads(
        (second_upper / "meta.json").read_text(encoding="utf-8")
    )
    assert second_metadata["marker"] == "capture-3"
    pointer = json.loads(
        (Path(task_settings.output_dir) / "current.json").read_text(encoding="utf-8")
    )
    assert pointer["scan_id"] == second.scan_id


@pytest.mark.asyncio
async def test_partial_scan_does_not_replace_published_generation(tmp_path: Path) -> None:
    mock = Task0Mock()
    task_settings = settings(tmp_path, ["POINT_ONE"])
    client = Task0Client(task_settings, transport=mock.transport)
    orchestrator = Task0Orchestrator(task_settings, client)
    async with client:
        first = await orchestrator.run(Task0Request(), "first")
        mock.invalid_capture_number = mock.capture_number + 2
        with pytest.raises(Task0ServiceError):
            await orchestrator.run(Task0Request(), "partial")

    pointer = json.loads(
        (Path(task_settings.output_dir) / "current.json").read_text(encoding="utf-8")
    )
    assert pointer["scan_id"] == first.scan_id
    runs_root = Path(task_settings.output_dir) / "runs"
    assert not any(path.name.startswith(".staging-") for path in runs_root.iterdir())


@pytest.mark.asyncio
async def test_invalid_archive_does_not_replace_existing_capture(tmp_path: Path) -> None:
    mock = Task0Mock()
    task_settings = settings(tmp_path, ["POINT_ONE"])
    upper = Path(task_settings.output_dir) / "POINT_ONE_UPPER"
    upper.mkdir(parents=True)
    (upper / "sentinel.txt").write_text("keep", encoding="utf-8")
    mock.invalid_capture = True
    client = Task0Client(task_settings, transport=mock.transport)
    async with client:
        with pytest.raises(Task0ServiceError) as error:
            await Task0Orchestrator(task_settings, client).run(Task0Request())

    assert error.value.code == "INVALID_CAMERA_RESPONSE"
    assert error.value.step == "采集 POINT_ONE UPPER RGB-D"
    assert (upper / "sentinel.txt").read_text(encoding="utf-8") == "keep"
    assert not (Path(task_settings.output_dir) / "current.json").exists()


@pytest.mark.asyncio
async def test_health_requires_head_color_and_depth_streams(tmp_path: Path) -> None:
    mock = Task0Mock()
    mock.head_depth_online = False
    task_settings = settings(tmp_path)
    client = Task0Client(task_settings, transport=mock.transport)
    async with client:
        assert not await client.health_ready()
        with pytest.raises(Task0ServiceError) as error:
            await client.check_all_health()

    assert error.value.code == "CAPABILITY_NOT_READY"
    assert "camera.head.rgbd" in error.value.message


def test_task0_settings_validate_points_and_load_production_config(tmp_path: Path) -> None:
    production = TaskServiceSettings.load(
        CONFIG_DIR / "runtime.production.yaml"
    ).tasks.task0
    assert production.services.camera.endswith(":8085")
    assert production.camera == "head"
    assert production.start_target_id == "start"
    assert production.capture_settle_seconds == 2
    assert production.inspection_points == [
        "H1_INSPECT",
        "H2_INSPECT",
        "H3_INSPECT",
    ]

    with pytest.raises(ValidationError):
        settings(tmp_path, ["POINT/UNSAFE"])
    with pytest.raises(ValidationError):
        settings(tmp_path, ["POINT_ONE", "POINT_ONE"])
    with pytest.raises(ValidationError):
        Task0Settings(
            services={
                "navigation": "http://navigation.local",
                "pose": "http://pose.local",
                "camera": "http://camera.local",
            },
            inspection_points=["POINT_ONE"],
            camera="left_wrist",
        )
