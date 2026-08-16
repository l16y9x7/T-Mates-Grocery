"""Serial task 0 shelf inspection and aligned RGB-D persistence."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from task0_service.client import Task0Client
from task0_service.models import (
    CaptureResult,
    InspectionPose,
    Task0Request,
    Task0Result,
    Task0ServiceError,
    Task0Settings,
)


LOGGER = logging.getLogger(__name__)
REQUIRED_RGBD_FILES = ("rgb.jpg", "depth_mm.npy", "meta.json")


class Task0Orchestrator:
    def __init__(self, settings: Task0Settings, client: Task0Client) -> None:
        self.settings = settings
        self.client = client

    async def run(
        self, request: Task0Request, operation_key: str | None = None
    ) -> Task0Result:
        task_run_id = operation_key or uuid4().hex
        task_log = _Task0Log(self.settings, task_run_id, request)
        self.client.set_trace_callback(task_log.interface_event)
        captures: list[CaptureResult] = []
        total_captures = len(self.settings.inspection_points) * 2
        current_pose: InspectionPose | None = None
        step = "健康检查"
        try:
            task_log.event("operation", "started", total_captures=total_captures)
            task_log.event("健康检查", "started")
            await self.client.check_all_health()
            task_log.event("健康检查", "succeeded")

            step = "导航到开始点"
            task_log.event(
                "开始点导航",
                "started",
                target_id=self.settings.start_target_id,
                phase="start",
            )
            await self.client.navigate(
                self.settings.start_target_id,
                f"{task_run_id}:task0.start.navigate",
            )
            task_log.event(
                "开始点导航",
                "succeeded",
                target_id=self.settings.start_target_id,
                phase="start",
            )

            for index, target_id in enumerate(self.settings.inspection_points):
                step = f"导航到 {target_id}"
                task_log.event(
                    "巡检点导航",
                    "started",
                    target_id=target_id,
                    point_index=index + 1,
                    total_points=len(self.settings.inspection_points),
                )
                await self.client.navigate(
                    target_id, f"{task_run_id}:task0.inspect.{index}.navigate"
                )
                task_log.event(
                    "巡检点导航",
                    "succeeded",
                    target_id=target_id,
                    point_index=index + 1,
                    total_points=len(self.settings.inspection_points),
                )
                pose_order = (
                    (InspectionPose.UPPER, InspectionPose.LOWER)
                    if index % 2 == 0
                    else (InspectionPose.LOWER, InspectionPose.UPPER)
                )
                for pose in pose_order:
                    action_prefix = (
                        f"{task_run_id}:task0.inspect.{index}."
                        f"{pose.directory_suffix.lower()}"
                    )
                    if current_pose is not pose:
                        step = f"准备 {target_id} {pose.directory_suffix} 位姿"
                        task_log.event(
                            "巡检观察位姿",
                            "started",
                            target_id=target_id,
                            pose_type=pose.value,
                        )
                        await self.client.prepare_pose(
                            pose.value, f"{action_prefix}.pose"
                        )
                        current_pose = pose
                        task_log.event(
                            "巡检观察位姿",
                            "succeeded",
                            target_id=target_id,
                            pose_type=pose.value,
                            reused=False,
                        )
                    else:
                        task_log.event(
                            "巡检观察位姿",
                            "succeeded",
                            target_id=target_id,
                            pose_type=pose.value,
                            reused=True,
                        )
                    step = f"采集 {target_id} {pose.directory_suffix} RGB-D"
                    capture_number = len(captures) + 1
                    task_log.event(
                        "RGB-D采集",
                        "started",
                        target_id=target_id,
                        pose_type=pose.value,
                        capture_number=capture_number,
                        total_captures=total_captures,
                        settle_seconds=self.settings.capture_settle_seconds,
                    )
                    await asyncio.sleep(self.settings.capture_settle_seconds)
                    archive = await self.client.capture_rgbd()
                    capture = self._store_capture(target_id, pose, archive)
                    captures.append(capture)
                    task_log.event(
                        "RGB-D采集",
                        "succeeded",
                        target_id=target_id,
                        pose_type=pose.value,
                        capture_number=capture_number,
                        total_captures=total_captures,
                        directory=capture.directory,
                    )
                    LOGGER.info(
                        "task0 capture saved target=%s pose=%s directory=%s",
                        target_id,
                        pose.value,
                        capture.directory,
                    )

            step = "返回开始点"
            task_log.event(
                "开始点导航",
                "started",
                target_id=self.settings.start_target_id,
                phase="finish",
            )
            await self.client.navigate(
                self.settings.start_target_id,
                f"{task_run_id}:task0.finish.navigate",
            )
            task_log.event(
                "开始点导航",
                "succeeded",
                target_id=self.settings.start_target_id,
                phase="finish",
            )
            result = Task0Result(
                task_run_id=task_run_id,
                task_type="PREPARATION",
                status="SUCCEEDED",
                inspection_points=list(self.settings.inspection_points),
                captures=captures,
            )
            task_log.event(
                "operation", "succeeded", captured_count=len(captures)
            )
            return result
        except Exception as exc:
            if isinstance(exc, Task0ServiceError):
                exc.step = step
            task_log.event(
                "operation",
                "failed",
                step=step,
                error_code=getattr(exc, "code", type(exc).__name__),
                message=str(exc),
            )
            LOGGER.exception("task0 failed step=%s key=%s", step, task_run_id)
            raise
        finally:
            self.client.set_trace_callback(None)

    def _store_capture(
        self, target_id: str, pose: InspectionPose, archive: bytes
    ) -> CaptureResult:
        files = _validate_rgbd_archive(archive, expected_camera=self.settings.camera)
        root = Path(self.settings.output_dir)
        directory_name = f"{target_id}_{pose.directory_suffix}"
        target = root / directory_name
        root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{directory_name}-", dir=root))
        backup: Path | None = None
        try:
            for name in REQUIRED_RGBD_FILES:
                (temporary / name).write_bytes(files[name])
            if target.exists() or target.is_symlink():
                backup = root / f".{directory_name}.backup-{uuid4().hex}"
                os.replace(target, backup)
            try:
                os.replace(temporary, target)
            except Exception:
                if backup is not None:
                    os.replace(backup, target)
                    backup = None
                raise
            if backup is not None:
                _remove_path(backup)
                backup = None
        except Task0ServiceError:
            raise
        except OSError as exc:
            raise Task0ServiceError(
                "STORAGE_ERROR",
                f"failed to store RGB-D capture for {target_id}: {exc}",
                status_code=500,
            ) from exc
        finally:
            if temporary.exists():
                _remove_path(temporary)
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)

        return CaptureResult(
            target_id=target_id,
            pose_type=pose,
            directory=str(target),
            rgb_path=str(target / "rgb.jpg"),
            depth_path=str(target / "depth_mm.npy"),
            meta_path=str(target / "meta.json"),
        )


def _validate_rgbd_archive(
    archive: bytes, *, expected_camera: str
) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(BytesIO(archive)) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)):
                raise Task0ServiceError(
                    "INVALID_CAMERA_RESPONSE", "RGB-D ZIP contains duplicate entries"
                )
            unsafe = [
                name
                for name in names
                if Path(name).is_absolute() or ".." in Path(name).parts
            ]
            if unsafe:
                raise Task0ServiceError(
                    "INVALID_CAMERA_RESPONSE", "RGB-D ZIP contains unsafe paths"
                )
            missing = [name for name in REQUIRED_RGBD_FILES if name not in names]
            if missing:
                raise Task0ServiceError(
                    "INVALID_CAMERA_RESPONSE",
                    "RGB-D ZIP is missing required files: " + ", ".join(missing),
                )
            files = {name: bundle.read(name) for name in REQUIRED_RGBD_FILES}
    except Task0ServiceError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise Task0ServiceError(
            "INVALID_CAMERA_RESPONSE", "camera response is not a valid RGB-D ZIP"
        ) from exc

    empty = [name for name, content in files.items() if not content]
    if empty:
        raise Task0ServiceError(
            "INVALID_CAMERA_RESPONSE",
            "RGB-D ZIP contains empty files: " + ", ".join(empty),
        )
    if not files["depth_mm.npy"].startswith(b"\x93NUMPY"):
        raise Task0ServiceError(
            "INVALID_CAMERA_RESPONSE", "depth_mm.npy is not a NumPy array"
        )
    try:
        metadata = json.loads(files["meta.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Task0ServiceError(
            "INVALID_CAMERA_RESPONSE", "meta.json is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(metadata, dict):
        raise Task0ServiceError(
            "INVALID_CAMERA_RESPONSE", "meta.json must contain a JSON object"
        )
    if metadata.get("camera") != expected_camera:
        raise Task0ServiceError(
            "INVALID_CAMERA_RESPONSE",
            f"meta.json camera must be {expected_camera}",
        )
    return files


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


class _Task0Log:
    def __init__(
        self, settings: Task0Settings, operation_key: str, request: Task0Request
    ) -> None:
        root = Path(settings.log_dir)
        root.mkdir(parents=True, exist_ok=True)
        safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", operation_key).strip("._")
        safe_key = safe_key or "operation"
        self.directory = root / (
            f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{safe_key}"
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        self._write_json(
            "operation.json",
            {
                "kind": "task0",
                "operation_key": operation_key,
                "request": request.model_dump(mode="json"),
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
            },
        )

    def _write_json(self, relative_path: str, payload: object) -> None:
        path = self.directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def event(self, name: str, status: str, **details: object) -> None:
        try:
            record = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "event": name,
                "status": status,
                **details,
            }
            with (self.directory / "events.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )
        except (OSError, TypeError, ValueError):
            LOGGER.exception("task0 log write failed event=%s status=%s", name, status)

    def interface_event(self, trace: dict[str, object]) -> None:
        status_code = trace.get("status_code")
        status = (
            "succeeded"
            if isinstance(status_code, int) and status_code < 400
            else "failed"
        )
        self.event(
            "接口调用",
            status,
            interface=trace.get("interface"),
            service=trace.get("service"),
            request={
                "method": trace.get("method"),
                "url": trace.get("url"),
                "params": trace.get("params") or {},
                "headers": trace.get("headers") or {},
                "body": trace.get("body"),
                "attempt": trace.get("attempt"),
            },
            response={
                "status_code": status_code,
                "headers": trace.get("response_headers") or {},
                "body": trace.get("response_body"),
                "error": trace.get("error"),
            },
        )
