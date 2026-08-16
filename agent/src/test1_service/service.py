"""Serial Test1 shelf-level RGB-D collection workflow."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from test1_service.client import Test1Client
from test1_service.models import (
    CameraFrame,
    CaptureResult,
    Hand,
    SHELF_LEVELS,
    Test1Result,
    Test1ServiceError,
    Test1Settings,
)


LOGGER = logging.getLogger(__name__)


class Test1Orchestrator:
    def __init__(self, settings: Test1Settings, client: Test1Client) -> None:
        self.settings = settings
        self.client = client

    async def run(self, operation_key: str | None = None) -> Test1Result:
        task_run_id = operation_key or uuid4().hex
        run_log = _Test1Log(self.settings, task_run_id)
        self.client.set_trace_callback(run_log.interface_event)
        batch = self._create_batch_directory(task_run_id)
        captures: list[CaptureResult] = []
        total_captures = len(self.settings.inspection_points) * len(
            SHELF_LEVELS
        ) * len(Hand)
        step = "健康检查"
        try:
            run_log.event("operation", "started", total_captures=total_captures)
            await self.client.check_all_health()
            run_log.event("健康检查", "succeeded")

            step = "导航到开始点"
            await self._reset_and_navigate(
                self.settings.start_target_id,
                f"{task_run_id}:test1.start",
                run_log,
                phase="start",
            )

            for point_index, target_id in enumerate(self.settings.inspection_points):
                step = f"导航到 {target_id}"
                await self._reset_and_navigate(
                    target_id,
                    f"{task_run_id}:test1.inspect.{point_index}",
                    run_log,
                    point_index=point_index + 1,
                )
                for level_index, shelf_level in enumerate(SHELF_LEVELS):
                    action_prefix = (
                        f"{task_run_id}:test1.inspect.{point_index}.level.{level_index}"
                    )
                    step = f"准备 {target_id} {shelf_level} 位姿"
                    run_log.event(
                        "分层拍摄位姿",
                        "started",
                        target_id=target_id,
                        shelf_level=shelf_level,
                    )
                    await self.client.prepare_pose(
                        "SHELF_INSPECT",
                        f"{action_prefix}.pose",
                        shelf_level=shelf_level,
                    )
                    run_log.event(
                        "分层拍摄位姿",
                        "succeeded",
                        target_id=target_id,
                        shelf_level=shelf_level,
                    )
                    run_log.event(
                        "拍摄前等待",
                        "started",
                        target_id=target_id,
                        shelf_level=shelf_level,
                        seconds=self.settings.capture_settle_seconds,
                    )
                    await asyncio.sleep(self.settings.capture_settle_seconds)
                    run_log.event(
                        "拍摄前等待",
                        "succeeded",
                        target_id=target_id,
                        shelf_level=shelf_level,
                    )

                    for hand in Hand:
                        step = f"采集 {target_id} {shelf_level} {hand.value} RGB-D"
                        capture_number = len(captures) + 1
                        run_log.event(
                            "腕部RGB-D采集",
                            "started",
                            target_id=target_id,
                            shelf_level=shelf_level,
                            hand=hand.value,
                            camera=hand.camera,
                            capture_number=capture_number,
                            total_captures=total_captures,
                        )
                        frame = await self.client.capture(hand)
                        capture = self._store_capture(
                            batch, target_id, shelf_level, hand, frame
                        )
                        captures.append(capture)
                        run_log.event(
                            "腕部RGB-D采集",
                            "succeeded",
                            target_id=target_id,
                            shelf_level=shelf_level,
                            hand=hand.value,
                            camera=hand.camera,
                            capture_number=capture_number,
                            total_captures=total_captures,
                            directory=capture.directory,
                        )

            step = "返回开始点"
            await self._reset_and_navigate(
                self.settings.start_target_id,
                f"{task_run_id}:test1.finish",
                run_log,
                phase="finish",
            )
            result = Test1Result(
                task_run_id=task_run_id,
                batch_directory=str(batch),
                captures=captures,
            )
            run_log.event(
                "operation",
                "succeeded",
                captured_count=len(captures),
                batch_directory=str(batch),
            )
            return result
        except Exception as exc:
            if isinstance(exc, Test1ServiceError):
                exc.step = step
            run_log.event(
                "operation",
                "failed",
                step=step,
                captured_count=len(captures),
                batch_directory=str(batch),
                error_code=getattr(exc, "code", type(exc).__name__),
                message=str(exc),
            )
            LOGGER.exception("test1 failed step=%s key=%s", step, task_run_id)
            raise
        finally:
            self.client.set_trace_callback(None)

    async def _reset_and_navigate(
        self,
        target_id: str,
        action_prefix: str,
        run_log: "_Test1Log",
        **details: object,
    ) -> None:
        run_log.event(
            "移动前复位", "started", target_id=target_id, **details
        )
        await self.client.prepare_pose(
            "START_POSITION", f"{action_prefix}.reset_pose"
        )
        run_log.event(
            "移动前复位", "succeeded", target_id=target_id, **details
        )
        run_log.event("导航", "started", target_id=target_id, **details)
        await self.client.navigate(target_id, f"{action_prefix}.navigate")
        run_log.event("导航", "succeeded", target_id=target_id, **details)

    def _create_batch_directory(self, task_run_id: str) -> Path:
        root = Path(self.settings.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        safe_key = _safe_key(task_run_id)
        prefix = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        batch = root / f"{prefix}-{safe_key}"
        try:
            batch.mkdir(parents=False, exist_ok=False)
        except OSError as exc:
            raise Test1ServiceError(
                "STORAGE_ERROR", f"failed to create batch directory: {exc}"
            ) from exc
        return batch

    def _store_capture(
        self,
        batch: Path,
        target_id: str,
        shelf_level: str,
        hand: Hand,
        frame: CameraFrame,
    ) -> CaptureResult:
        directory_name = f"{target_id}-{shelf_level}-{hand.value}"
        target = batch / directory_name
        temporary = Path(tempfile.mkdtemp(prefix=f".{directory_name}-", dir=batch))
        try:
            rgb_path = temporary / f"rgb{frame.rgb_suffix}"
            depth_path = temporary / "depth.png"
            rgb_path.write_bytes(frame.rgb)
            depth_path.write_bytes(frame.depth)
            os.replace(temporary, target)
        except OSError as exc:
            raise Test1ServiceError(
                "STORAGE_ERROR",
                f"failed to store capture {directory_name}: {exc}",
                status_code=500,
            ) from exc
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return CaptureResult(
            target_id=target_id,
            shelf_level=shelf_level,
            hand=hand,
            camera=hand.camera,
            directory=str(target),
            rgb_path=str(target / rgb_path.name),
            depth_path=str(target / depth_path.name),
        )


class _Test1Log:
    def __init__(self, settings: Test1Settings, operation_key: str) -> None:
        root = Path(settings.log_dir)
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.directory = root / f"{timestamp}-{_safe_key(operation_key)}-test1"
        self.directory.mkdir(parents=False, exist_ok=False)
        self._write_json(
            "operation.json",
            {
                "kind": "test1",
                "operation_key": operation_key,
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
            },
        )

    def _write_json(self, relative_path: str, payload: object) -> None:
        (self.directory / relative_path).write_text(
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
            LOGGER.exception("test1 log write failed event=%s status=%s", name, status)

    def interface_event(self, trace: dict[str, object]) -> None:
        status_code = trace.get("status_code")
        status = (
            "succeeded"
            if isinstance(status_code, int) and status_code < 400
            else "failed"
        )
        self.event("接口调用", status, **trace)


def _safe_key(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return normalized or "operation"
