"""机器人流程网页代理：启动取放和任务一服务并通过 SSE 转发流程事件。"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .settings import LOCATE_IMAGE_ROOTS, LOG_ROOT, SETTINGS

WEB_ROOT = Path(__file__).resolve().parent
SERVICES = SETTINGS.services
DEFAULT_TIMEOUT = SETTINGS.request_timeout_seconds


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


class Task1Request(BaseModel):
    """任务一独立服务请求。"""

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


class ReceiptParseRequest(BaseModel):
    """小票识别请求目前没有业务字段，保留空 JSON body 的契约。"""

    pass


@dataclass
class PickTask:
    task_id: str
    operation_key: str
    task: asyncio.Task[None] | None = None
    finished: bool = False
    result: dict[str, object] | None = None
    created_at: float = field(default_factory=time.monotonic)


app = FastAPI(title="Pick/Place Web Console", version="1.0")
app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")
TASKS: dict[str, PickTask] = {}


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


def _find_task1_pick_log_dir(operation_key: str) -> Path | None:
    """Find the newest 8086 child operation created by a task-one run."""

    if not LOG_ROOT.exists():
        return None
    prefix = f"-{_safe_key(operation_key)}_task1.pick."
    candidates = [
        path
        for path in LOG_ROOT.iterdir()
        if path.is_dir() and prefix in path.name and path.name.endswith(".pick")
    ]
    return sorted(candidates, key=lambda path: path.name)[-1] if candidates else None


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


def _operation_visual(log_dir: Path | None) -> dict[str, object]:
    """Collect the images and pose fields written by the pick/place service."""

    result: dict[str, object] = {
        "available": False,
        "image_data": None,
        "mask_data": None,
        "bbox": None,
        "bbox_coordinate_space": 1000,
        "pose": None,
        "pose_unit": None,
        "frame": None,
        "rotation_order": None,
        "corners_mm": None,
    }
    if log_dir is None:
        return result

    events_path = log_dir / "events.jsonl"
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines() if events_path.is_file() else []
    except OSError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if isinstance(event.get("bbox"), list) and len(event["bbox"]) == 4:
            result["bbox"] = event["bbox"]
        if isinstance(event.get("pose"), list) and len(event["pose"]) == 6:
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
        if result["pose"] is None and isinstance(response.get("pose"), list) and len(response["pose"]) == 6:
            result["pose"] = response["pose"]

    image_data = None
    for name in ("rgb.jpg", "rgb.jpeg", "rgb.png", "rgb.webp"):
        image_data = _log_file_data(log_dir, f"camera/{name}")
        if image_data:
            break
    mask_data = None
    for name in ("mask.png", "mask.jpg", "mask.jpeg", "mask.pgm"):
        mask_data = _log_file_data(log_dir, f"camera/{name}")
        if mask_data:
            break

    # If capture did not finish, expose the RGB returned by the locate service.
    if image_data is None:
        locate_response_path = log_dir / "interfaces" / "perception_pick_locate" / "response.json"
        if not locate_response_path.is_file():
            locate_response_path = log_dir / "interfaces" / "perception_place_locate" / "response.json"
        try:
            locate_response = json.loads(locate_response_path.read_text(encoding="utf-8"))
            locate_body = locate_response.get("body", locate_response) if isinstance(locate_response, dict) else {}
            image_path = locate_body.get("image_path") if isinstance(locate_body, dict) else None
            if isinstance(image_path, str):
                image_data = _local_image_data(image_path)
            locate_mask = locate_body.get("mask") if isinstance(locate_body, dict) else None
            if isinstance(locate_mask, str) and locate_mask:
                mask_data = locate_mask if locate_mask.startswith("data:") else f"data:image/png;base64,{locate_mask}"
            if result["bbox"] is None and isinstance(locate_body, dict) and isinstance(locate_body.get("bbox"), list):
                result["bbox"] = locate_body["bbox"]
        except (OSError, json.JSONDecodeError):
            pass
    result["image_data"] = image_data
    result["mask_data"] = mask_data
    result["available"] = image_data is not None or mask_data is not None or result["bbox"] is not None or result["pose"] is not None
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


async def _run_task1(task_state: PickTask, request: Task1Request) -> None:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                SERVICES.task1_url,
                json=request.model_dump(mode="json"),
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
        task_state.result = {
            "status_code": 502,
            "ok": False,
            "body": {
                "error_code": "TASK1_PROXY_ERROR",
                "message": str(exc),
                "failed_step": "任务一服务请求",
            },
        }
    finally:
        task_state.finished = True


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "static" / "index.html")


@app.get("/api/config")
async def config() -> dict[str, str]:
    return {
        "pick_url": SERVICES.pick_url,
        "place_url": SERVICES.place_url,
        "task1_url": SERVICES.task1_url,
        "navigation_url": SERVICES.navigation_url,
        "pose_url": SERVICES.pose_url,
        "perception_url": SERVICES.perception_url,
        "log_dir": str(LOG_ROOT),
    }


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


@app.post("/api/task1/start")
async def start_task1(request: Task1Request) -> dict[str, str]:
    task_id = uuid4().hex
    operation_key = f"web-task1-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{task_id[:10]}"
    state = PickTask(task_id=task_id, operation_key=operation_key)
    state.task = asyncio.create_task(_run_task1(state, request))
    TASKS[task_id] = state
    return {
        "task_id": task_id,
        "operation_key": operation_key,
        "events_url": f"/api/task1/{task_id}/events",
    }


@app.get("/api/task1/{task_id}/events")
async def task1_events(task_id: str) -> StreamingResponse:
    state = TASKS.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return _events_response(state)


@app.get("/api/task1/{task_id}/visual")
async def task1_visual(task_id: str) -> dict[str, object]:
    state = TASKS.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    log_dir = _find_task1_pick_log_dir(state.operation_key) or _find_log_dir(state.operation_key)
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
