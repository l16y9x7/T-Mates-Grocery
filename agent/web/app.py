"""统一任务与取放控制台路由，并通过 SSE 转发流程事件。"""

from __future__ import annotations

import asyncio
import base64
from ipaddress import IPv4Address
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal
from uuid import uuid4

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from task_service.coordinator import TaskCoordinator, TaskServiceError
from pick_place_service.models import PickPlaceSettings
from task_service.settings import TaskServiceSettings

from .settings import LOCATE_IMAGE_ROOTS, LOG_ROOT, SETTINGS, UNIFIED_SETTINGS

WEB_ROOT = Path(__file__).resolve().parent
SERVICES = SETTINGS.services
DEFAULT_TIMEOUT = SETTINGS.request_timeout_seconds
RUNTIME_SETTINGS = UNIFIED_SETTINGS
RUNTIME_CONFIG_PATH = UNIFIED_SETTINGS.config_path
ROBOT_IP = str(UNIFIED_SETTINGS.robot.ip)
PICK_PLACE_STATUS_URL = UNIFIED_SETTINGS.pick_place_status_url
PROJECT_ROOT = WEB_ROOT.parent


class LocateRequest(BaseModel):
    """兼容旧定位工作台接口。"""

    task_type: str = Field(default="SORTING", min_length=1, max_length=64)
    product_name: str = Field(min_length=1, max_length=200)
    hand: str = Field(default="left")
    url: str | None = None

    @field_validator("task_type", "product_name", "hand")
    @classmethod
    def trim_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("字段不能为空")
        return value


class PickRequest(BaseModel):
    task_type: str = Field(default="SORTING", min_length=1, max_length=32)
    product_name: str = Field(min_length=1, max_length=200)
    hand: str = Field(default="left", min_length=1, max_length=16)

    @field_validator("task_type", "product_name", "hand")
    @classmethod
    def trim_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("字段不能为空")
        return value


class UnifiedTaskRequest(BaseModel):
    """Task0-Task3 currently share an empty request body."""

    model_config = {"extra": "forbid"}


class PosePrepareRequest(BaseModel):
    pose_type: str = Field(min_length=1, max_length=64)
    shelf_level: str | None = Field(default=None, min_length=1, max_length=16)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("pose_type", "shelf_level", "idempotency_key")
    @classmethod
    def trim_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        return value or None


class NavigationRequest(BaseModel):
    target_id: str = Field(min_length=1, max_length=200)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("target_id", "idempotency_key")
    @classmethod
    def trim_navigation_text(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        return value or None


class GripperOpenRequest(BaseModel):
    """请求机器人打开指定手的夹爪。"""

    model_config = {"extra": "forbid"}

    hand: Literal["LEFT", "RIGHT"]


class ReceiptParseRequest(BaseModel):
    """小票识别请求目前没有业务字段，保留空 JSON body 的契约。"""

    pass


class RobotIpUpdateRequest(BaseModel):
    robot_ip: IPv4Address
    force_restart: bool = False


@dataclass
class PickTask:
    task_id: str
    operation_key: str
    workflow: str = "pick_place"
    task: asyncio.Task[None] | None = None
    execution_task: asyncio.Task[object] | None = None
    finished: bool = False
    result: dict[str, object] | None = None
    created_at: float = field(default_factory=time.monotonic)


app = FastAPI(title="Pick/Place Web Console", version="1.0")
app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")
TASKS: dict[str, PickTask] = {}
COORDINATOR_GETTER: Callable[[], TaskCoordinator] | None = None
CONFIG_UPDATE_LOCK = asyncio.Lock()


def configure_runtime(
    settings: TaskServiceSettings,
    coordinator_getter: Callable[[], TaskCoordinator],
) -> None:
    global SERVICES, DEFAULT_TIMEOUT, LOG_ROOT, LOCATE_IMAGE_ROOTS, COORDINATOR_GETTER
    global RUNTIME_SETTINGS, RUNTIME_CONFIG_PATH, ROBOT_IP, PICK_PLACE_STATUS_URL
    RUNTIME_SETTINGS = settings
    RUNTIME_CONFIG_PATH = settings.config_path
    ROBOT_IP = str(settings.robot.ip)
    PICK_PLACE_STATUS_URL = settings.pick_place_status_url
    SERVICES = settings.web.services
    DEFAULT_TIMEOUT = settings.web.request_timeout_seconds
    LOG_ROOT = settings.web.paths.log_dir
    LOCATE_IMAGE_ROOTS = tuple(settings.web.paths.locate_image_roots)
    COORDINATOR_GETTER = coordinator_getter


def _coordinator() -> TaskCoordinator:
    if COORDINATOR_GETTER is None:
        raise HTTPException(status_code=503, detail="统一任务服务尚未初始化")
    return COORDINATOR_GETTER()


def _restart_preflight() -> tuple[bool, str | None]:
    required_scripts = (
        PROJECT_ROOT / "scripts" / "restart-runtime.sh",
        PROJECT_ROOT / "scripts" / "tasks.sh",
        PROJECT_ROOT / "scripts" / "pick-place.sh",
    )
    for script in required_scripts:
        if not script.is_file() or not os.access(script, os.X_OK):
            return False, f"运行控制脚本不存在或不可执行：{script.name}"
    return True, None


async def _active_operations() -> dict[str, object]:
    web_operations = [
        {
            "workflow": state.workflow,
            "operation_key": state.operation_key,
        }
        for state in TASKS.values()
        if not state.finished
    ]
    active: dict[str, object] = {
        "task_id": _coordinator().active_task_id,
        "web_operations": web_operations,
        "pick_place_active_operations": 0,
        "pick_place_status_unknown": False,
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(PICK_PLACE_STATUS_URL, timeout=1.5)
            response.raise_for_status()
            body = response.json()
        count = body.get("active_operations", 0) if isinstance(body, dict) else 0
        active["pick_place_active_operations"] = count if isinstance(count, int) else 0
    except (httpx.HTTPError, TimeoutError, ValueError):
        active["pick_place_status_unknown"] = True
    return active


def _has_active_operations(active: dict[str, object]) -> bool:
    return bool(
        active["task_id"]
        or active["web_operations"]
        or active["pick_place_active_operations"]
        or active["pick_place_status_unknown"]
    )


def _write_robot_ip(robot_ip: str) -> Path:
    config_path = RUNTIME_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("robot"), dict):
        raise RuntimeError("运行配置缺少 robot 节点")
    raw["robot"]["ip"] = robot_ip
    original_mode = config_path.stat().st_mode
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            yaml.safe_dump(raw, stream, allow_unicode=True, sort_keys=False)
            temporary_path = Path(stream.name)
        temporary_path.chmod(original_mode)
        TaskServiceSettings.load(temporary_path)
        PickPlaceSettings.load(temporary_path)
        backup_path = config_path.with_suffix(config_path.suffix + ".bak")
        shutil.copy2(config_path, backup_path)
        os.replace(temporary_path, config_path)
        return backup_path
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _schedule_runtime_restart() -> None:
    log_dir = LOG_ROOT / "process"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"runtime-restart-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    restart_script = PROJECT_ROOT / "scripts" / "restart-runtime.sh"
    environment = os.environ.copy()
    environment["RUNTIME_CONFIG_FILE"] = str(RUNTIME_CONFIG_PATH)
    with log_path.open("ab") as output:
        subprocess.Popen(
            [str(restart_script)],
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )


@app.exception_handler(TaskServiceError)
async def task_service_error_handler(_, exc: TaskServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": exc.code, "message": exc.message},
    )


def _local_image_data(image_path: str) -> str | None:
    """Encode a result image when the locate service shares a filesystem."""

    if image_path.startswith(("http://", "https://")):
        return None
    path = Path(image_path).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    if not any(resolved == root or root in resolved.parents for root in LOCATE_IMAGE_ROOTS):
        return None
    try:
        content = resolved.read_bytes()
    except OSError:
        return None
    mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "operation"


def _find_log_dir(operation_key: str) -> Path | None:
    if not LOG_ROOT.exists():
        return None
    suffix = f"-{_safe_key(operation_key)}"
    candidates = [path for path in LOG_ROOT.iterdir() if path.is_dir() and path.name.endswith(suffix)]
    return sorted(candidates, key=lambda path: path.name)[-1] if candidates else None


def _locate_response(log_dir: Path) -> tuple[str, dict[str, object], Path] | None:
    """Return the successful pick/place locate response persisted by 8086."""

    for kind in ("pick", "place"):
        path = log_dir / "interfaces" / f"perception_{kind}_locate" / "response.json"
        document = _read_json_file(path)
        if not isinstance(document, dict):
            continue
        status_code = document.get("status_code")
        if isinstance(status_code, int) and not 200 <= status_code < 300:
            continue
        body = document.get("body", document)
        if isinstance(body, dict):
            return kind, body, path
    return None


def _find_task_visual_log_dir(operation_key: str, task_id: str) -> Path | None:
    """Find the newest task child operation with a successful locate response."""

    if not LOG_ROOT.exists():
        return None
    prefix = f"-{_safe_key(operation_key)}_task{task_id}."
    candidates = [
        path
        for path in LOG_ROOT.iterdir()
        if path.is_dir() and prefix in path.name and _locate_response(path) is not None
    ]
    return max(candidates, key=lambda path: path.name) if candidates else None


def _find_task_pick_log_dir(operation_key: str, task_id: str) -> Path | None:
    """Compatibility alias for callers using the previous helper name."""

    return _find_task_visual_log_dir(operation_key, task_id)


def _log_file_data(log_dir: Path, relative_path: str) -> str | None:
    """Return a data URI for a file inside the current operation log."""

    path = (log_dir / relative_path).resolve()
    try:
        if log_dir.resolve() not in path.parents or not path.is_file():
            return None
        content = path.read_bytes()
    except OSError:
        return None
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(content).decode('ascii')}"


def _first_log_file(
    log_dir: Path, directory: str, names: tuple[str, ...]
) -> tuple[str | None, Path | None]:
    for name in names:
        relative_path = f"{directory}/{name}"
        data = _log_file_data(log_dir, relative_path)
        if data:
            return data, log_dir / relative_path
    return None, None


def _calibration_matrix(log_dir: Path, directory: str) -> list[float] | None:
    calibration_dir = log_dir / directory
    if not calibration_dir.is_dir():
        return None
    for path in sorted(calibration_dir.glob("*.json")):
        if path.name == "input.json":
            continue
        document = _read_json_file(path)
        cam_k = document.get("cam_K") if isinstance(document, dict) else None
        if (
            isinstance(cam_k, list)
            and len(cam_k) == 9
            and all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in cam_k
            )
        ):
            return [float(value) for value in cam_k]
    return None


def _project_pose_axes(
    pose: object, cam_k: object, axis_length_mm: float = 100.0
) -> dict[str, object] | None:
    """Project a camera-frame mm/rad ZYX pose into image pixel coordinates."""

    if (
        not isinstance(pose, list)
        or len(pose) != 6
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in pose
        )
        or not isinstance(cam_k, list)
        or len(cam_k) != 9
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in cam_k
        )
        or not math.isfinite(axis_length_mm)
        or axis_length_mm <= 0
    ):
        return None

    x, y, z, rx, ry, rz = (float(value) for value in pose)
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    rotation = (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )
    fx, fy, principal_x, principal_y = (
        float(cam_k[0]),
        float(cam_k[4]),
        float(cam_k[2]),
        float(cam_k[5]),
    )

    def project(point: tuple[float, float, float]) -> list[float] | None:
        point_x, point_y, point_z = point
        if point_z <= 0:
            return None
        pixel = [
            fx * point_x / point_z + principal_x,
            fy * point_y / point_z + principal_y,
        ]
        return pixel if all(math.isfinite(value) for value in pixel) else None

    origin = project((x, y, z))
    if origin is None:
        return None
    axes: dict[str, object] = {"origin": origin, "axis_length_mm": axis_length_mm}
    for index, name in enumerate(("x", "y", "z")):
        endpoint = project(
            (
                x + axis_length_mm * rotation[0][index],
                y + axis_length_mm * rotation[1][index],
                z + axis_length_mm * rotation[2][index],
            )
        )
        if endpoint is not None:
            axes[name] = endpoint
    return axes if len(axes) > 2 else None


def _visual_revision(log_dir: Path, paths: list[Path | None]) -> str:
    parts = [log_dir.name]
    for path in paths:
        if path is None:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        parts.append(f"{path.relative_to(log_dir)}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


def _operation_visual(log_dir: Path | None) -> dict[str, object]:
    """Collect the images and pose fields written by the pick/place service."""

    result: dict[str, object] = {
        "available": False,
        "visual_revision": None,
        "image_data": None,
        "mask_data": None,
        "bbox": None,
        "bbox_coordinate_width": None,
        "bbox_coordinate_height": None,
        "pose": None,
        "pose_axes": None,
        "pose_unit": None,
        "frame": None,
        "rotation_order": None,
        "corners_mm": None,
    }
    if log_dir is None:
        return result

    locate = _locate_response(log_dir)
    locate_kind = locate[0] if locate else None
    locate_body = locate[1] if locate else {}
    locate_response_path = locate[2] if locate else None
    is_place = locate_kind == "place"
    if isinstance(locate_body.get("bbox"), list) and len(locate_body["bbox"]) == 4:
        result["bbox"] = locate_body["bbox"]
    if locate_kind == "pick":
        result["bbox_coordinate_width"] = 1000
        result["bbox_coordinate_height"] = 1000

    events_path = log_dir / "events.jsonl"
    try:
        lines = (
            events_path.read_text(encoding="utf-8").splitlines()
            if events_path.is_file()
            else []
        )
    except OSError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if (
            result["bbox"] is None
            and isinstance(event.get("bbox"), list)
            and len(event["bbox"]) == 4
        ):
            result["bbox"] = event["bbox"]
        if (
            is_place
            and isinstance(event.get("current_pose"), list)
            and len(event["current_pose"]) == 6
        ):
            result["pose"] = event["current_pose"]
        elif not is_place and isinstance(event.get("pose"), list) and len(event["pose"]) == 6:
            result["pose"] = event["pose"]

    response_path = log_dir / "interfaces" / "manipulation_pick_pose" / "response.json"
    if not response_path.is_file():
        response_path = log_dir / "interfaces" / "manipulation_place_pose" / "response.json"
    try:
        response = json.loads(response_path.read_text(encoding="utf-8")) if response_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        response = {}
    if isinstance(response, dict) and isinstance(response.get("body"), dict):
        response = response["body"]
    if isinstance(response, dict):
        for key in ("pose_unit", "frame", "rotation_order", "corners_mm"):
            if response.get(key) is not None:
                result[key] = response[key]
        if (
            not is_place
            and result["pose"] is None
            and isinstance(response.get("pose"), list)
            and len(response["pose"]) == 6
        ):
            result["pose"] = response["pose"]

    image_names = ("rgb.jpg", "rgb.jpeg", "rgb.png", "rgb.webp")
    mask_names = ("mask.png", "mask.jpg", "mask.jpeg", "mask.pgm")
    image_data = None
    image_path: Path | None = None
    mask_data = None
    calibration_dir: str | None = None
    pose_is_aligned = False

    if result["pose"] is not None:
        visual_dir = "current" if is_place else "camera"
        image_data, image_path = _first_log_file(log_dir, visual_dir, image_names)
        if image_data is not None:
            calibration_dir = visual_dir
            pose_is_aligned = True
            if is_place:
                result["bbox"] = None
            else:
                mask_data, _ = _first_log_file(log_dir, "camera", mask_names)

    if image_data is None:
        locate_image_path = locate_body.get("image_path")
        if isinstance(locate_image_path, str):
            image_data = _local_image_data(locate_image_path)
        fallback_dir = "reference" if is_place else "camera"
        if image_data is None:
            image_data, image_path = _first_log_file(log_dir, fallback_dir, image_names)
        locate_mask = locate_body.get("mask")
        if isinstance(locate_mask, str) and locate_mask:
            mask_data = locate_mask if locate_mask.startswith("data:") else f"data:image/png;base64,{locate_mask}"
        if mask_data is None:
            mask_data, _ = _first_log_file(log_dir, fallback_dir, mask_names)

    if pose_is_aligned and calibration_dir is not None:
        cam_k = _calibration_matrix(log_dir, calibration_dir)
        result["pose_axes"] = _project_pose_axes(result["pose"], cam_k)

    result["image_data"] = image_data
    result["mask_data"] = mask_data
    result["available"] = any(
        value is not None
        for value in (image_data, mask_data, result["bbox"], result["pose"])
    )
    result["visual_revision"] = _visual_revision(
        log_dir,
        [locate_response_path, response_path, events_path, image_path],
    )
    return result


def _sse(event: str, payload: object) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _read_json_file(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, json.JSONDecodeError):
        return None


def _interface_events(log_dir: Path | None, emitted: set[str]) -> list[dict[str, object]]:
    """Convert persisted downstream HTTP traces into flow events."""

    if log_dir is None:
        return []
    interfaces_dir = log_dir / "interfaces"
    if not interfaces_dir.is_dir():
        return []
    events: list[dict[str, object]] = []
    try:
        interface_dirs = sorted(path for path in interfaces_dir.iterdir() if path.is_dir())
    except OSError:
        return []
    for interface_dir in interface_dirs:
        name = interface_dir.name
        if name in emitted:
            continue
        request = _read_json_file(interface_dir / "request.json")
        response = _read_json_file(interface_dir / "response.json")
        if request is None and response is None:
            continue
        if response is None:
            continue
        status_code = response.get("status_code") if isinstance(response, dict) else None
        status = "succeeded" if isinstance(status_code, int) and status_code < 400 else "failed"
        events.append({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "event": "接口调用",
            "status": status,
            "interface": name,
            "request": request,
            "response": response,
        })
        emitted.add(name)
    return events


async def _run_pick_place(task_state: PickTask, request: PickRequest, target_url: str) -> None:
    payload = request.model_dump(mode="json")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                target_url,
                json=payload,
                headers={"Idempotency-Key": task_state.operation_key},
                timeout=DEFAULT_TIMEOUT,
            )
        try:
            body: object = response.json()
        except (ValueError, json.JSONDecodeError):
            body = response.text
        task_state.result = {
            "status_code": response.status_code,
            "ok": response.is_success,
            "body": body,
        }
    except (httpx.HTTPError, TimeoutError) as exc:
        task_state.result = {"status_code": 502, "ok": False, "body": {"message": str(exc)}}
    finally:
        task_state.finished = True


async def _finish_unified_task(
    task_state: PickTask, execution: asyncio.Task[object]
) -> None:
    try:
        result = await execution
        body = result.model_dump(mode="json") if isinstance(result, BaseModel) else result
        task_state.result = {
            "status_code": 200,
            "ok": True,
            "body": body,
        }
    except asyncio.CancelledError:
        task_state.result = {
            "status_code": 499,
            "ok": False,
            "body": {
                "error_code": "TASK_TERMINATED",
                "message": "任务已由用户终止",
            },
        }
    except Exception as exc:
        content: dict[str, object] = {
            "error_code": getattr(exc, "code", "TASK_EXECUTION_ERROR"),
            "message": getattr(exc, "message", str(exc)),
        }
        if getattr(exc, "step", None):
            content["failed_step"] = exc.step
        if getattr(exc, "failed_interface", None):
            content["failed_interface"] = exc.failed_interface
        if getattr(exc, "url", None):
            content["url"] = exc.url
        task_state.result = {
            "status_code": getattr(exc, "status_code", 500),
            "ok": False,
            "body": content,
        }
    finally:
        task_state.finished = True


@app.get("/", include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse(
        (WEB_ROOT / "static" / "index.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/config")
async def config() -> dict[str, object]:
    restart_supported, restart_reason = _restart_preflight()
    return {
        "pick_url": SERVICES.pick_url,
        "place_url": SERVICES.place_url,
        "tasks_url": "/tasks/{task_id}/run",
        "navigation_url": SERVICES.navigation_url,
        "pose_url": SERVICES.pose_url,
        "perception_url": SERVICES.perception_url,
        "log_dir": str(LOG_ROOT),
        "robot_ip": ROBOT_IP,
        "restart_supported": restart_supported,
        "restart_unavailable_reason": restart_reason,
    }


@app.put("/api/system/robot-ip")
async def update_robot_ip(request: RobotIpUpdateRequest) -> JSONResponse:
    async with CONFIG_UPDATE_LOCK:
        restart_supported, restart_reason = _restart_preflight()
        if not restart_supported:
            return JSONResponse(
                status_code=409,
                content={
                    "error_code": "RESTART_UNAVAILABLE",
                    "message": restart_reason,
                },
            )

        active = await _active_operations()
        if _has_active_operations(active) and not request.force_restart:
            return JSONResponse(
                status_code=409,
                content={
                    "error_code": "OPERATIONS_IN_PROGRESS",
                    "message": "有任务正在执行，强制重启可能让机器人保持运动或持物状态",
                    "requires_force": True,
                    "active": active,
                },
            )

        backup_path: Path | None = None
        try:
            backup_path = _write_robot_ip(str(request.robot_ip))
            _schedule_runtime_restart()
        except Exception as exc:
            if backup_path is not None and backup_path.exists():
                shutil.copy2(backup_path, RUNTIME_CONFIG_PATH)
            return JSONResponse(
                status_code=500,
                content={
                    "error_code": "CONFIG_UPDATE_FAILED",
                    "message": str(exc),
                },
            )

        return JSONResponse(
            status_code=202,
            content={
                "status": "RESTARTING",
                "robot_ip": str(request.robot_ip),
                "services": ["pick-place:8086", "tasks:8108"],
                "forced": request.force_restart,
            },
        )


async def _robot_request(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, object] | None,
    operation: str,
    idempotency_key: str | None,
    timeout: float | None = None,
) -> JSONResponse:
    """调用真实机器人接口，并把请求/响应原样返回给网页。"""

    key = idempotency_key or f"web:{operation}:{uuid4().hex}"
    request_payload = payload or {}
    target_url = f"{base_url.rstrip('/')}{path}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method,
                target_url,
                json=request_payload if method != "GET" else None,
                headers={"Content-Type": "application/json", "Idempotency-Key": key},
                timeout=timeout or DEFAULT_TIMEOUT,
            )
    except (httpx.HTTPError, TimeoutError) as exc:
        return JSONResponse(
            status_code=502,
            content={
                "ok": False,
                "status_code": 502,
                "operation": operation,
                "url": target_url,
                "idempotency_key": key,
                "request": request_payload,
                "body": {"message": str(exc)},
            },
        )
    try:
        body: object = response.json()
    except (ValueError, json.JSONDecodeError):
        body = response.text
    result = {
        "ok": response.is_success,
        "status_code": response.status_code,
        "operation": operation,
        "url": target_url,
        "idempotency_key": key,
        "request": request_payload,
        "body": body,
    }
    return JSONResponse(status_code=response.status_code, content=result)


@app.get("/api/robot/health")
async def robot_health() -> dict[str, object]:
    """同时检查导航服务和位姿准备服务。"""

    async def check(name: str, base_url: str, path: str) -> dict[str, object]:
        response = await _robot_request("GET", base_url, path, None, f"{name}_health", None, timeout=10)
        try:
            body: object = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = response.body.decode("utf-8", errors="replace")
        ready = body.get("status") == "READY" if isinstance(body, dict) and "status" in body else response.status_code < 400
        return {"service": name, "ok": response.status_code < 400 and ready, "status_code": response.status_code, "body": body}

    pose, navigation = await asyncio.gather(
        check("pose", SERVICES.pose_url, "/pose/health"),
        check("navigation", SERVICES.navigation_url, "/navigation/health"),
    )
    return {"pose": pose, "navigation": navigation}


@app.post("/api/robot/prepare")
async def robot_prepare(
    request: PosePrepareRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    payload = request.model_dump(exclude={"idempotency_key"}, exclude_none=True)
    return await _robot_request(
        "POST",
        SERVICES.pose_url,
        "/pose/prepare",
        payload,
        "pose_prepare",
        request.idempotency_key or idempotency_key,
    )


@app.post("/api/robot/navigate")
async def robot_navigate(
    request: NavigationRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    payload = request.model_dump(exclude={"idempotency_key"})
    return await _robot_request(
        "POST",
        SERVICES.navigation_url,
        "/navigation/navigate",
        payload,
        "navigation",
        request.idempotency_key or idempotency_key,
    )


@app.post("/api/robot/gripper/open")
async def gripper_open(
    request: GripperOpenRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    """代理机器人夹爪松手接口。"""

    return await _robot_request(
        "POST",
        SERVICES.pose_url,
        "/manipulation/gripper/open",
        request.model_dump(mode="json"),
        "gripper_open",
        idempotency_key,
    )


@app.post("/api/perception/parse")
async def perception_parse(
    request: ReceiptParseRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    """代理视觉理解服务的小票识别接口。"""

    return await _robot_request(
        "POST",
        SERVICES.perception_url,
        "/perception/parse",
        request.model_dump(mode="json"),
        "perception_parse",
        idempotency_key,
    )


@app.post("/api/pick/start")
async def start_pick(request: PickRequest) -> dict[str, str]:
    task_id = uuid4().hex
    operation_key = f"web-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{task_id[:10]}"
    state = PickTask(task_id=task_id, operation_key=operation_key)
    state.task = asyncio.create_task(_run_pick_place(state, request, SERVICES.pick_url))
    TASKS[task_id] = state
    return {
        "task_id": task_id,
        "operation_key": operation_key,
        "events_url": f"/api/pick/{task_id}/events",
    }


async def _event_stream(state: PickTask):
    offset = 0
    emitted_interfaces: set[str] = set()
    sent_result = False
    deadline = time.monotonic() + DEFAULT_TIMEOUT + 30
    while time.monotonic() < deadline:
        log_dir = _find_log_dir(state.operation_key)
        events_file = log_dir / "events.jsonl" if log_dir else None
        if events_file and events_file.exists():
            try:
                with events_file.open("r", encoding="utf-8") as source:
                    source.seek(offset)
                    while True:
                        line = source.readline()
                        if not line:
                            break
                        offset = source.tell()
                        try:
                            yield _sse("flow", json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except OSError:
                pass
        for interface_event in _interface_events(log_dir, emitted_interfaces):
            yield _sse("flow", interface_event)
        if state.finished and not sent_result:
            sent_result = True
            yield _sse("result", state.result or {"ok": False, "body": {"message": "任务无结果"}})
            return
        yield ": heartbeat\n\n"
        await asyncio.sleep(0.35)
    if not sent_result:
        yield _sse("result", {"ok": False, "body": {"message": "网页等待任务超时"}})


def _events_response(state: PickTask) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(state),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/pick/{task_id}/events")
async def pick_events(task_id: str) -> StreamingResponse:
    state = TASKS.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return _events_response(state)


@app.get("/api/pick/{task_id}/visual")
async def pick_visual(task_id: str) -> dict[str, object]:
    state = TASKS.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return _operation_visual(_find_log_dir(state.operation_key))


@app.post("/api/place/start")
async def start_place(request: PickRequest) -> dict[str, str]:
    task_id = uuid4().hex
    operation_key = f"web-place-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{task_id[:10]}"
    state = PickTask(task_id=task_id, operation_key=operation_key)
    state.task = asyncio.create_task(_run_pick_place(state, request, SERVICES.place_url))
    TASKS[task_id] = state
    return {
        "task_id": task_id,
        "operation_key": operation_key,
        "events_url": f"/api/place/{task_id}/events",
    }


@app.get("/api/place/{task_id}/events")
async def place_events(task_id: str) -> StreamingResponse:
    state = TASKS.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return _events_response(state)


@app.get("/api/place/{task_id}/visual")
async def place_visual(task_id: str) -> dict[str, object]:
    state = TASKS.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return _operation_visual(_find_log_dir(state.operation_key))


@app.post("/api/tasks/{task_id}/start")
async def start_unified_task(
    task_id: str, request: UnifiedTaskRequest
) -> dict[str, str]:
    run_id = uuid4().hex
    operation_key = (
        f"web-task{task_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{run_id[:10]}"
    )
    execution = await _coordinator().start_background(
        task_id, request.model_dump(mode="json"), operation_key
    )
    state = PickTask(
        task_id=run_id,
        operation_key=operation_key,
        workflow=f"task{task_id}",
        execution_task=execution,
    )
    state.task = asyncio.create_task(_finish_unified_task(state, execution))
    TASKS[run_id] = state
    return {
        "run_id": run_id,
        "task_id": task_id,
        "operation_key": operation_key,
        "events_url": f"/api/task-runs/{run_id}/events",
        "visual_url": f"/api/task-runs/{run_id}/visual",
    }


@app.post("/api/task-runs/{run_id}/terminate")
async def terminate_unified_task(run_id: str) -> dict[str, object]:
    state = TASKS.get(run_id)
    if state is None or not state.workflow.startswith("task"):
        raise HTTPException(status_code=404, detail="统一任务不存在或已过期")
    if state.finished:
        return {
            "status": "ALREADY_FINISHED",
            "run_id": run_id,
            "result": state.result,
        }
    if state.task is None:
        raise HTTPException(status_code=409, detail="任务尚未建立执行句柄")

    if state.execution_task is not None and not state.execution_task.done():
        state.execution_task.cancel()
    else:
        state.task.cancel()
    try:
        await state.task
    except asyncio.CancelledError:
        # Keep the endpoint idempotent even if a custom execution wrapper
        # propagates cancellation instead of converting it to a result.
        if state.result is None:
            state.result = {
                "status_code": 499,
                "ok": False,
                "body": {
                    "error_code": "TASK_TERMINATED",
                    "message": "任务已由用户终止",
                },
            }
        state.finished = True
    return {
        "status": "TERMINATED",
        "run_id": run_id,
        "result": state.result,
    }


@app.get("/api/task-runs/{run_id}/events")
async def unified_task_events(run_id: str) -> StreamingResponse:
    state = TASKS.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return _events_response(state)


@app.get("/api/task-runs/{run_id}/visual")
async def unified_task_visual(run_id: str) -> dict[str, object]:
    state = TASKS.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    task_id = state.workflow.removeprefix("task")
    log_dir = _find_log_dir(state.operation_key)
    if task_id in {"1", "2", "3"}:
        log_dir = _find_task_visual_log_dir(state.operation_key, task_id) or log_dir
    return _operation_visual(log_dir)


@app.post("/api/locate")
async def locate(request: LocateRequest) -> dict[str, object]:
    """保留原定位工作台接口，便于已有调用方继续使用。"""

    target_url = request.url or SERVICES.locate_url
    payload = {"task_type": request.task_type, "product_name": request.product_name, "hand": request.hand.lower()}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(target_url, json=payload, timeout=DEFAULT_TIMEOUT)
    except (httpx.HTTPError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"定位接口请求失败：{exc}") from exc
    try:
        result = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="定位接口返回了非 JSON 数据") from exc
    if not response.is_success:
        detail = result.get("message", result) if isinstance(result, dict) else result
        raise HTTPException(status_code=response.status_code, detail=f"定位接口错误：{detail}")
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="定位接口返回格式不正确")
    bbox = result.get("bbox")
    mask = result.get("mask")
    image_path = result.get("image_path")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(isinstance(value, (int, float)) for value in bbox)
        or not isinstance(mask, str)
        or not mask
        or not isinstance(image_path, str)
        or not image_path
    ):
        raise HTTPException(status_code=502, detail="定位接口缺少可视化所需字段")
    result["image_data"] = _local_image_data(image_path)
    result["bbox_coordinate_space"] = 1000
    return result
