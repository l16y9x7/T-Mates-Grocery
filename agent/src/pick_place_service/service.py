"""8086 取放流程和下游能力适配。"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextvars
from datetime import datetime
import hashlib
import json
import logging
import mimetypes
import re
import struct
import tempfile
import zlib
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

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
from pick_place_service.place_pose import synthesize_place_pose

LOGGER = logging.getLogger(__name__)
_ACTIVE_LOG_DIR: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "pick_place_active_log_dir", default=None
)


class FrameProvider(Protocol):
    async def capture(
        self,
        camera: str,
        bbox: list[int | float],
        operation: str,
        mask_base64: str | None = None,
    ) -> FrameBundle: ...

    async def prepare_place_references(
        self,
        located: PlaceLocateResponse,
        camera: str,
        operation: str,
    ) -> list[FrameBundle]: ...


class CameraFrameProvider:
    """从 8085 相机网关取 RGB-D，并准备给 8084 的文件输入。"""

    def __init__(self, settings: PickPlaceSettings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client

    async def capture(
        self,
        camera: str,
        bbox: list[int | float],
        operation: str,
        mask_base64: str | None = None,
    ) -> FrameBundle:
        LOGGER.info("相机取图开始 camera=%s operation=%s", camera, operation)
        root = Path(self.settings.temp_dir)
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f"{operation}-", dir=root))
        try:
            try:
                calibration_file = self.settings.calibration_for(camera)
            except ValueError as exc:
                raise ServiceError("CALIBRATION_UNAVAILABLE", str(exc), status_code=502) from exc
            color = await self._snapshot(camera, "color")
            LOGGER.info(
                "相机 RGB 快照成功 camera=%s bytes=%d image=%s",
                camera,
                len(color),
                _describe_image(color),
            )
            width, height = _image_size(color)
            depth = await self._depth_frame(camera)
            LOGGER.info(
                "相机 depth 流首帧成功 camera=%s bytes=%d image=%s",
                camera,
                len(depth),
                _describe_image(depth),
            )
            if depth.startswith(b"\xff\xd8"):
                # JPEG 通常是用于预览的 8 位图，无法无损承载 ROS 16UC1 毫米深度。
                # 这里先保留原始响应交给 8084 校验，同时明确提示最可能的输入问题。
                LOGGER.warning(
                    "深度输入格式可疑 camera=%s format=JPEG；位姿服务通常需要与 RGB 对齐的 "
                    "16 位单通道深度 PNG，而不是彩色/灰度预览 JPEG",
                    camera,
                )
            depth = _normalize_depth_frame(depth, width, height)
            LOGGER.info("深度输入标准化完成 camera=%s image=%s", camera, _describe_image(depth))
            rgb_path = directory / f"rgb{_image_suffix(color, '.jpg')}"
            depth_path = directory / f"depth{_image_suffix(depth, '.png')}"
            rgb_path.write_bytes(color)
            depth_path.write_bytes(depth)
            if mask_base64:
                # 正式定位服务返回的是原图尺寸的 PNG mask，优先使用它，避免
                # 用矩形 bbox 覆盖真实分割结果；没有 mask 时才走兼容兜底。
                mask_path = directory / "mask.png"
                _write_locate_mask(mask_path, mask_base64, width, height)
            else:
                mask_path = directory / "mask.pgm"
                _write_mask(mask_path, width, height, bbox)
            _save_log_file("camera/rgb" + rgb_path.suffix, color)
            _save_log_file("camera/depth" + depth_path.suffix, depth)
            _save_log_file("camera/" + mask_path.name, mask_path.read_bytes())
            calibration_path = Path(calibration_file)
            if calibration_path.is_file():
                _save_log_file("camera/" + calibration_path.name, calibration_path.read_bytes())
            _save_log_json(
                "camera/input.json",
                {
                    "camera": camera,
                    "rgb": _describe_image(color),
                    "depth": _describe_image(depth),
                    "mask": str(mask_path.name),
                    "calibration": calibration_file,
                },
            )
            LOGGER.info(
                "位姿输入文件已生成 rgb=%s depth=%s mask=%s calibration=%s",
                rgb_path,
                depth_path,
                mask_path,
                calibration_file,
            )
            frame = FrameBundle(
                rgb=str(rgb_path),
                depth=str(depth_path),
                camera=calibration_file,
                mask=str(mask_path),
                cleanup_path=str(directory),
            )
            LOGGER.info("相机输入准备完成 rgb=%s depth=%s mask=%s", rgb_path, depth_path, mask_path)
            return frame
        except Exception:
            LOGGER.exception("相机取图失败 camera=%s operation=%s", camera, operation)
            _remove_directory(directory)
            raise

    async def prepare_place_references(
        self,
        located: PlaceLocateResponse,
        camera: str,
        operation: str,
    ) -> list[FrameBundle]:
        """Prepare current RGB-D and one pose-estimator input per reference mask."""

        root = Path(self.settings.temp_dir)
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f"{operation}-", dir=root))
        try:
            try:
                calibration_file = self.settings.calibration_for(camera)
            except ValueError as exc:
                raise ServiceError(
                    "CALIBRATION_UNAVAILABLE", str(exc), status_code=502
                ) from exc
            source_rgb = Path(located.current_image_path)
            source_depth = source_rgb.with_name("current_depth_mm.npy")
            if not source_rgb.is_file() or not source_depth.is_file():
                raise ServiceError(
                    "CURRENT_INPUT_UNAVAILABLE",
                    f"current RGB-D is unavailable beside {source_rgb}",
                )
            color = source_rgb.read_bytes()
            width, height = _image_size(color)
            try:
                depth_array = np.load(source_depth, allow_pickle=False)
            except (OSError, ValueError, TypeError) as exc:
                raise ServiceError(
                    "CURRENT_INPUT_INVALID", f"invalid current depth array: {exc}"
                ) from exc
            depth = _depth_array_to_png(depth_array, width, height)
            for bbox in located.bbox:
                _validate_pixel_bbox(bbox, width, height)

            rgb_path = directory / f"rgb{_image_suffix(color, '.jpg')}"
            depth_path = directory / "depth.png"
            rgb_path.write_bytes(color)
            depth_path.write_bytes(depth)
            mask_paths: list[Path] = []
            for index, encoded_mask in enumerate(located.mask, start=1):
                mask_path = directory / f"mask_{index:02d}.png"
                _write_locate_mask(mask_path, encoded_mask, width, height)
                mask_paths.append(mask_path)

            _save_log_file("current/rgb" + rgb_path.suffix, color)
            _save_log_file("current/depth.png", depth)
            for mask_path in mask_paths:
                _save_log_file("current/" + mask_path.name, mask_path.read_bytes())
            calibration_path = Path(calibration_file)
            if calibration_path.is_file():
                _save_log_file(
                    "current/" + calibration_path.name,
                    calibration_path.read_bytes(),
                )
            _save_log_json(
                "current/input.json",
                {
                    "source_rgb": str(source_rgb),
                    "source_depth": str(source_depth),
                    "baseline_rgb": located.image_path,
                    "image_size": [width, height],
                    "bboxes": located.bbox,
                    "masks": [path.name for path in mask_paths],
                    "direction": located.direction,
                    "camera": camera,
                    "calibration": calibration_file,
                },
            )
            return [
                FrameBundle(
                    rgb=str(rgb_path),
                    depth=str(depth_path),
                    camera=calibration_file,
                    mask=str(mask_path),
                    cleanup_path=str(directory) if index == 0 else None,
                )
                for index, mask_path in enumerate(mask_paths)
            ]
        except Exception:
            _remove_directory(directory)
            raise

    async def _snapshot(self, camera: str, image_type: str) -> bytes:
        try:
            response = await self.client.get(
                f"{self.settings.camera_url.rstrip('/')}/camera/snapshot",
                params={"camera": camera, "type": image_type},
                timeout=_http_timeout(self.settings, self.settings.timeouts.camera_seconds),
            )
            response.raise_for_status()
            if not response.content:
                raise ServiceError("VISION_INPUT_UNAVAILABLE", "RGB snapshot returned no data")
            snapshot_url = f"{self.settings.camera_url.rstrip('/')}/camera/snapshot"
            _save_log_json(
                f"interfaces/camera_snapshot_{image_type}/request.json",
                {"method": "GET", "url": snapshot_url, "params": {"camera": camera, "type": image_type}},
            )
            _save_log_json(
                f"interfaces/camera_snapshot_{image_type}/response.json",
                {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "bytes": len(response.content),
                },
            )
            LOGGER.info(
                "相机快照 HTTP 响应 camera=%s type=%s content_type=%s content_length=%s",
                camera,
                image_type,
                response.headers.get("content-type", "unknown"),
                response.headers.get("content-length", "unknown"),
            )
            return response.content
        except ServiceError:
            raise
        except (httpx.HTTPError, TimeoutError) as exc:
            raise ServiceError("VISION_INPUT_UNAVAILABLE", str(exc), status_code=502) from exc

    async def _depth_frame(self, camera: str) -> bytes:
        """从长连接深度流读取一帧；兼容 JPEG 帧流和单帧二进制响应。"""

        try:
            async with self.client.stream(
                "GET",
                f"{self.settings.camera_url.rstrip('/')}/camera/stream",
                params={"camera": camera, "type": "depth"},
                headers={"Accept": "multipart/x-mixed-replace"},
                timeout=_http_timeout(self.settings, self.settings.timeouts.camera_seconds),
            ) as response:
                response.raise_for_status()
                LOGGER.info(
                    "相机流 HTTP 响应 camera=%s type=depth content_type=%s",
                    camera,
                    response.headers.get("content-type", "unknown"),
                )
                content_type = response.headers.get("content-type", "")
                data = bytearray()
                frame: bytes | None = None
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    frame = _extract_stream_frame(bytes(data), content_type)
                    if frame is not None or len(data) >= 8 * 1024 * 1024:
                        break
                if not data:
                    raise ServiceError("VISION_INPUT_UNAVAILABLE", "depth stream returned no frame")
                if frame is None:
                    frame = _extract_stream_frame(bytes(data), content_type) or bytes(data)
                _save_log_json(
                    "interfaces/camera_stream_depth/request.json",
                    {
                        "method": "GET",
                        "url": f"{self.settings.camera_url.rstrip('/')}/camera/stream",
                        "params": {"camera": camera, "type": "depth"},
                        "headers": {"Accept": "multipart/x-mixed-replace"},
                    },
                )
                _save_log_json(
                    "interfaces/camera_stream_depth/response.json",
                    {"status_code": response.status_code, "headers": dict(response.headers), "bytes": len(data)},
                )
                _save_log_file("interfaces/camera_stream_depth/frame.bin", frame)
                LOGGER.info(
                    "相机 depth 首帧提取完成 camera=%s stream_bytes=%d frame_bytes=%d",
                    camera,
                    len(data),
                    len(frame),
                )
                return frame
        except ServiceError:
            raise
        except (httpx.HTTPError, TimeoutError) as exc:
            raise ServiceError("VISION_INPUT_UNAVAILABLE", str(exc), status_code=502) from exc


class SubagentClient:
    """调用 8083/8084 的子agent接口。"""

    def __init__(self, settings: PickPlaceSettings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client

    async def health(self) -> bool:
        pose_estimation_url = self.settings.pose_estimation_url or self.settings.manipulation_url
        checks = (
            (self.settings.perception_url, "/perception/health"),
            # 物体位姿估计和机械动作部署在不同主机，但当前都使用 manipulation 命名空间。
            (pose_estimation_url, "/manipulation/health"),
            (self.settings.manipulation_url, "/manipulation/health"),
            (self.settings.camera_url, "/camera/health"),
        )
        try:
            results = await asyncio.gather(*(self._health(url, path) for url, path in checks))
            camera_list = await self._camera_list_ready()
            return all(results) and camera_list
        except (httpx.HTTPError, TimeoutError, ServiceError):
            return False

    async def _camera_list_ready(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.settings.camera_url.rstrip('/')}/camera/list",
                timeout=self.settings.timeouts.health_seconds,
            )
            return response.is_success
        except (httpx.HTTPError, TimeoutError):
            return False

    async def _health(self, base_url: str, path: str) -> bool:
        response = await self.client.get(
            f"{base_url.rstrip('/')}{path}",
            timeout=self.settings.timeouts.health_seconds,
        )
        if not response.is_success:
            return False
        try:
            return response.json().get("status") == "READY"
        except (AttributeError, TypeError, ValueError):
            return False

    async def locate(self, request: PickPlaceRequest, kind: str) -> LocateResponse:
        locate_url = self.settings.locate_url or self.settings.perception_url
        LOGGER.info(
            "定位步骤开始 kind=%s product=%s url=%s",
            kind,
            request.product_name,
            locate_url,
        )
        payload = {
            # 8083 正式定位接口使用 task_type；name 是旧版接口字段。
            "task_type": request.task_type.value,
            "product_name": request.product_name,
            "hand": request.normalized_hand,
        }
        if kind == "pick" and request.level is not None:
            payload["level"] = request.level
        response = await self._post(
            locate_url,
            f"/perception/{kind}/locate",
            payload,
            self.settings.timeouts.locate_seconds,
        )
        try:
            located = LocateResponse.model_validate(response.json())
            if normalize_product_name(located.product_name) != normalize_product_name(
                request.product_name
            ):
                raise ValueError("locate response product_name does not match request")
            if len(located.bbox) != 4:
                raise ValueError("bbox must contain four values")
            LOGGER.info(
                "定位步骤成功 kind=%s product=%s bbox=%s mask=%s",
                kind,
                request.product_name,
                located.bbox,
                bool(located.mask),
            )
            return located
        except Exception as exc:
            LOGGER.exception("定位响应解析失败 kind=%s product=%s", kind, request.product_name)
            raise ServiceError("INVALID_RESPONSE", f"invalid {kind} locate response") from exc

    async def locate_place(self, request: PickPlaceRequest) -> PlaceLocateResponse:
        if request.location_id is None or request.pose_type is None:
            raise ServiceError(
                "INVALID_REQUEST",
                "SHORTAGE/MISPLACED place requires location_id and pose_type",
                status_code=422,
            )
        locate_url = self.settings.locate_url or self.settings.perception_url
        payload = {
            "task_type": request.task_type.value,
            "product_name": request.product_name,
            "location_id": request.location_id,
            "pose_type": request.pose_type,
        }
        response = await self._post(
            locate_url,
            "/perception/place/locate",
            payload,
            self.settings.timeouts.locate_seconds,
        )
        try:
            located = PlaceLocateResponse.model_validate(response.json())
            if normalize_product_name(located.name) != normalize_product_name(
                request.product_name
            ):
                raise ValueError("locate response name does not match request")
            expected_count = 1 if located.direction == "up" else 2
            if len(located.bbox) != expected_count or len(located.mask) != expected_count:
                raise ValueError(
                    f"direction {located.direction} requires {expected_count} bbox/mask entries"
                )
            if any(len(bbox) != 4 for bbox in located.bbox):
                raise ValueError("each bbox must contain four values")
            if any(not mask.strip() for mask in located.mask):
                raise ValueError("each mask must be non-empty")
            return located
        except Exception as exc:
            LOGGER.exception(
                "放置定位响应解析失败 product=%s", request.product_name
            )
            raise ServiceError(
                "INVALID_RESPONSE", "invalid place locate response"
            ) from exc

    async def prepare_place(self, level: str, operation_key: str) -> None:
        response = await self._post(
            self.settings.manipulation_url,
            "/pose/prepare",
            {"pose_type": "SHELF_PLACE_READY", "shelf_level": level},
            self.settings.timeouts.action_seconds,
            headers={"Idempotency-Key": f"{operation_key}:place-ready"},
        )
        self._validate_status(response, "shelf place pose preparation")

    async def estimate_pose(
        self,
        request: PickPlaceRequest,
        kind: str,
        frame: FrameBundle,
        *,
        log_suffix: str | None = None,
    ) -> PoseResponse:
        pose_estimation_url = self.settings.pose_estimation_url or self.settings.manipulation_url
        LOGGER.info(
            "位姿步骤开始 kind=%s product=%s url=%s",
            kind,
            request.product_name,
            pose_estimation_url,
        )
        paths = {
            "rgb": Path(frame.rgb),
            "depth": Path(frame.depth),
            "camera": Path(frame.camera),
            "mask": Path(frame.mask),
        }
        try:
            # 在打开文件前记录实际上传内容，便于区分文件缺失、格式错误和下游算法失败。
            for field, path in paths.items():
                LOGGER.info(
                    "位姿上传文件 field=%s path=%s bytes=%d mime=%s",
                    field,
                    path,
                    path.stat().st_size,
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                )
            with ExitStack() as stack:
                files = {
                    field: (
                        path.name,
                        stack.enter_context(path.open("rb")),
                        mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                    )
                    for field, path in paths.items()
                }
                data = {"product_name": request.product_name}
                response = await self.client.post(
                    f"{pose_estimation_url.rstrip('/')}/manipulation/{kind}_pose",
                    files=files,
                    data=data,
                    timeout=self.settings.timeouts.pose_seconds,
                )
                pose_log_dir = "interfaces/manipulation_%s_pose" % kind
                if log_suffix:
                    pose_log_dir += "_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", log_suffix)
                _save_log_json(
                    f"{pose_log_dir}/request.json",
                    {
                        "method": "POST",
                        "url": f"{pose_estimation_url.rstrip('/')}/manipulation/{kind}_pose",
                        "data": data,
                        "files": {
                            field: {
                                "filename": path.name,
                                "bytes": path.stat().st_size,
                                "mime": mimetypes.guess_type(path.name)[0]
                                or "application/octet-stream",
                            }
                            for field, path in paths.items()
                        },
                    },
                )
                _save_http_response(f"{pose_log_dir}/response.json", response)
                if not response.is_success:
                    LOGGER.error(
                        "位姿 HTTP 请求失败 kind=%s status=%d response=%s",
                        kind,
                        response.status_code,
                        response.text[:2000],
                    )
                    try:
                        payload = response.json()
                        if isinstance(payload, dict):
                            code = payload.get("error_code") or "POSE_ESTIMATION_FAILED"
                            # 不同下游版本可能使用 detail 或 message；两者都保留，
                            # 否则 8084 的具体拒绝原因会被压缩成笼统的 INVALID_REQUEST。
                            detail = payload.get("detail") or payload.get("message") or code
                        else:
                            code = "POSE_ESTIMATION_FAILED"
                            detail = payload
                    except (AttributeError, TypeError, ValueError):
                        code = "POSE_ESTIMATION_FAILED"
                        detail = "POSE_ESTIMATION_FAILED"
                    raise ServiceError(str(code), f"/manipulation/{kind}_pose: {detail}", status_code=502)
                LOGGER.info("位姿 HTTP 请求成功 kind=%s status=%d", kind, response.status_code)
        except ServiceError:
            raise
        except FileNotFoundError as exc:
            raise ServiceError("POSE_INPUT_UNAVAILABLE", str(exc), status_code=502) from exc
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise ServiceError("NETWORK_ERROR", f"/manipulation/{kind}_pose request timed out", status_code=504) from exc
        except httpx.HTTPError as exc:
            raise ServiceError("NETWORK_ERROR", str(exc), status_code=502) from exc
        try:
            result = PoseResponse.model_validate(response.json())
            if len(result.pose) != 6:
                raise ValueError("pose must contain six values")
            LOGGER.info("位姿响应解析成功 kind=%s pose=%s", kind, result.pose)
            return result
        except Exception as exc:
            LOGGER.exception("位姿响应解析失败 kind=%s product=%s", kind, request.product_name)
            raise ServiceError("INVALID_RESPONSE", f"invalid {kind} pose response") from exc

    async def execute(
        self,
        request: PickPlaceRequest,
        kind: str,
        pose: PoseResponse,
        operation_key: str,
    ) -> None:
        endpoint = "/manipulation/grasp" if kind == "pick" else "/manipulation/release"
        LOGGER.info("执行步骤开始 kind=%s endpoint=%s", kind, endpoint)
        if kind == "place" and request.task_type.value == "SORTING":
            # 任务一已在交付台完成固定放置位姿准备；SORTING 放置不再经过
            # 定位、取图和位姿估计，直接按现场 release 契约释放当前手上的商品。
            payload = {
                "task_type": "SORTING",
                "product_name": request.product_name,
                "hand": request.hand.upper(),
                "pose": [0, 0, 0, 0, 0, 0],
                "frame": "camera",
                "pose_unit": "mm_rad",
                "rotation_order": "zyx",
            }
        else:
            payload = action_payload(request, pose)
        LOGGER.info("执行请求参数 kind=%s payload=%s", kind, payload)
        response = await self._post(
            self.settings.manipulation_url,
            endpoint,
            payload,
            self.settings.timeouts.action_seconds,
            headers={"Idempotency-Key": f"{operation_key}:execute"},
        )
        self._validate_status(response, f"{kind} execution")
        LOGGER.info("执行步骤成功 kind=%s endpoint=%s", kind, endpoint)

    async def check(self, request: PickPlaceRequest, kind: str) -> None:
        LOGGER.info("校验步骤开始 kind=%s product=%s", kind, request.product_name)
        response = await self._post(
            self.settings.perception_url,
            f"/perception/{kind}/check",
            {
                "task_type": request.task_type.value,
                "product_name": request.product_name,
                "hand": request.normalized_hand,
            },
            self.settings.timeouts.check_seconds,
        )
        self._validate_status(response, f"{kind} visual check")
        LOGGER.info("校验步骤成功 kind=%s product=%s", kind, request.product_name)

    async def _post(
        self,
        base_url: str,
        path: str,
        payload: dict[str, object],
        timeout_seconds: float,
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        # 所有 JSON 下游请求统一在这里处理 HTTP 错误、超时和网络异常，
        # 这样上层日志能明确知道是哪个能力接口断开，而不是只看到 502。
        url = f"{base_url.rstrip('/')}{path}"
        interface_name = path.strip("/").replace("/", "_") or "root"
        log_dir = f"interfaces/{interface_name}"
        _save_log_json(
            f"{log_dir}/request.json",
            {"method": "POST", "url": url, "headers": headers or {}, "body": payload},
        )
        try:
            response = await self.client.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout_seconds,
            )
            _save_http_response(f"{log_dir}/response.json", response)
            if not response.is_success:
                try:
                    payload = response.json()
                    if isinstance(payload, dict):
                        code = payload.get("error_code") or "EXECUTION_FAILED"
                        detail = payload.get("detail") or payload.get("message") or code
                    else:
                        code = "EXECUTION_FAILED"
                        detail = payload
                except (AttributeError, TypeError, ValueError):
                    code = "EXECUTION_FAILED"
                    detail = "EXECUTION_FAILED"
                raise ServiceError(
                    str(code),
                    f"{path}: {detail}",
                    status_code=502,
                    failed_interface=interface_name,
                    url=url,
                )
            LOGGER.info("下游请求成功 method=POST path=%s status=%d", path, response.status_code)
            return response
        except ServiceError:
            LOGGER.exception("下游请求返回失败 path=%s", path)
            raise
        except (httpx.TimeoutException, TimeoutError) as exc:
            LOGGER.exception("下游请求超时 path=%s", path)
            code = "ACTION_RESULT_UNKNOWN" if path.endswith(("/grasp", "/release")) else "NETWORK_ERROR"
            message = f"{url} request result is unknown"
            _save_http_error(f"{log_dir}/response.json", 504, code, message, url)
            raise ServiceError(
                code,
                message,
                status_code=504,
                failed_interface=interface_name,
                url=url,
            ) from exc
        except httpx.HTTPError as exc:
            LOGGER.exception("下游网络异常 path=%s", path)
            message = f"{url}: {exc}"
            _save_http_error(f"{log_dir}/response.json", 502, "NETWORK_ERROR", message, url)
            raise ServiceError(
                "NETWORK_ERROR",
                message,
                status_code=502,
                failed_interface=interface_name,
                url=url,
            ) from exc

    @staticmethod
    def _validate_status(response: httpx.Response, action: str) -> None:
        try:
            payload = response.json()
            if not isinstance(payload, dict) or payload.get("status") != "SUCCEEDED":
                raise ValueError("status must be SUCCEEDED")
        except Exception as exc:
            raise ServiceError("INVALID_RESPONSE", f"invalid {action} response") from exc


class PickPlaceOrchestrator:
    def __init__(
        self,
        settings: PickPlaceSettings,
        subagents: SubagentClient,
        frames: FrameProvider,
    ) -> None:
        self.settings = settings
        self.subagents = subagents
        self.frames = frames

    async def run(self, request: PickPlaceRequest, kind: str, operation_key: str) -> StatusResponse:
        # 用 step 标记当前阶段；任何下游异常都会在统一日志中指出具体阶段。
        step = "定位"
        execution_pose: PoseResponse | None = None
        log_dir = _create_operation_log(self.settings, request, kind, operation_key)
        log_token = _ACTIVE_LOG_DIR.set(log_dir)
        try:
            _append_log_event(
                "operation",
                "started",
                task_type=request.task_type.value,
                product_name=request.product_name,
                hand=request.normalized_hand,
            )
            _save_log_json(
                "request.json",
                {
                    "task_type": request.task_type.value,
                    "product_name": request.product_name,
                    "hand": request.normalized_hand,
                    "operation_key": operation_key,
                    "request": request.model_dump(mode="json"),
                },
            )
            if kind == "place" and request.task_type.value == "SORTING":
                step = "释放执行"
                _append_log_event("释放执行", "started", hand=request.hand.upper())
                execution_pose = PoseResponse(
                    pose=[0, 0, 0, 0, 0, 0],
                    frame="camera",
                    pose_unit="mm_rad",
                    rotation_order="zyx",
                )
                await self.subagents.execute(
                    request,
                    kind,
                    execution_pose,
                    operation_key,
                )
                _append_log_event("释放执行", "succeeded", hand=request.hand.upper())
                _append_log_event("operation", "succeeded")
                LOGGER.info(
                    "SORTING 放置直接释放完成 product=%s hand=%s key=%s",
                    request.product_name,
                    request.hand.upper(),
                    operation_key,
                )
                return StatusResponse(status="SUCCEEDED")
            camera = self.settings.camera_for(
                kind, request.normalized_hand, request.task_type
            )
            if kind == "place":
                _append_log_event("定位", "started")
                place_located = await self.subagents.locate_place(request)
                _append_log_event(
                    "定位",
                    "succeeded",
                    bboxes=place_located.bbox,
                    mask_count=len(place_located.mask),
                    direction=place_located.direction,
                    image_path=place_located.image_path,
                    current_image_path=place_located.current_image_path,
                    level=place_located.level,
                )
                step = "当前参照输入准备"
                _append_log_event("当前参照输入准备", "started", camera=camera)
                frames = await self.frames.prepare_place_references(
                    place_located, camera, operation_key
                )
                _append_log_event(
                    "当前参照输入准备",
                    "succeeded",
                    camera=camera,
                    reference_count=len(frames),
                )
                step = "参照物位姿估计"
                reference_poses: list[PoseResponse] = []
                for index, frame in enumerate(frames, start=1):
                    _append_log_event(
                        "参照物位姿估计",
                        "started",
                        reference_index=index,
                        reference_count=len(frames),
                    )
                    reference_pose = await self.subagents.estimate_pose(
                        request,
                        kind,
                        frame,
                        log_suffix=f"reference_{index:02d}",
                    )
                    reference_poses.append(reference_pose)
                    _append_log_event(
                        "参照物位姿估计",
                        "succeeded",
                        reference_index=index,
                        reference_count=len(frames),
                        pose=reference_pose.pose,
                    )
                step = "放置位姿合成"
                _append_log_event(
                    "放置位姿合成",
                    "started",
                    direction=place_located.direction,
                    reference_poses=[item.pose for item in reference_poses],
                )
                pose = synthesize_place_pose(
                    reference_poses,
                    place_located.direction,
                )
                _append_log_event(
                    "放置位姿合成",
                    "succeeded",
                    direction=place_located.direction,
                    reference_poses=[item.pose for item in reference_poses],
                    current_pose=pose.pose,
                )
                step = "放置预备位姿"
                _append_log_event(
                    "放置预备位姿", "started", level=place_located.level
                )
                await self.subagents.prepare_place(place_located.level, operation_key)
                _append_log_event(
                    "放置预备位姿", "succeeded", level=place_located.level
                )
            else:
                _append_log_event("定位", "started")
                located = await self.subagents.locate(request, kind)
                _append_log_event(
                    "定位",
                    "succeeded",
                    bbox=located.bbox,
                    has_mask=bool(located.mask),
                )
                step = "取图"
                _append_log_event("取图", "started", camera=camera)
                frame = await self.frames.capture(
                    camera,
                    located.bbox,
                    operation_key,
                    located.mask,
                )
                _append_log_event("取图", "succeeded", camera=camera)
                step = "位姿估计"
                _append_log_event("位姿估计", "started")
                pose = await self.subagents.estimate_pose(request, kind, frame)
                _append_log_event("位姿估计", "succeeded", pose=pose.pose)
            step = "抓取/释放执行"
            execution_pose = pose
            _append_log_event("抓取/释放执行", "started")
            await self.subagents.execute(request, kind, pose, operation_key)
            _append_log_event("抓取/释放执行", "succeeded")
            # 所有 pick/place 操作均跳过结果视觉校验；抓取或释放执行成功后直接返回。
            # if kind in {"pick", "place"}:
            #     step = "视觉校验"
            #     _append_log_event("视觉校验", "started")
            #     await self.subagents.check(request, kind)
            #     _append_log_event("视觉校验", "succeeded")
            _append_log_event("operation", "succeeded")
            LOGGER.info("取放流程完成 kind=%s product=%s key=%s", kind, request.product_name, operation_key)
            return StatusResponse(status="SUCCEEDED")
        except Exception as exc:
            if (
                isinstance(exc, ServiceError)
                and step in {"抓取/释放执行", "释放执行"}
                and exc.failed_interface
                in {"manipulation_grasp", "manipulation_release"}
                and exc.code not in {"ACTION_RESULT_UNKNOWN", "NETWORK_ERROR"}
                and execution_pose is not None
            ):
                exc.pose = list(execution_pose.pose)
            _append_log_event(
                "operation",
                "failed",
                step=step,
                error_code=getattr(exc, "code", type(exc).__name__),
                message=str(exc),
                failed_interface=getattr(exc, "failed_interface", None),
                url=getattr(exc, "url", None),
            )
            LOGGER.exception(
                "取放流程失败 step=%s kind=%s product=%s key=%s",
                step,
                kind,
                request.product_name,
                operation_key,
            )
            raise
        finally:
            if "frames" in locals() and frames:
                cleanup_path = next(
                    (item.cleanup_path for item in frames if item.cleanup_path),
                    None,
                )
                if cleanup_path:
                    _remove_directory(Path(cleanup_path))
            elif "frame" in locals() and frame.cleanup_path:
                _remove_directory(Path(frame.cleanup_path))
            _ACTIVE_LOG_DIR.reset(log_token)


class OperationCache:
    """单进程内按 key 去重整条复合流程。"""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[str, asyncio.Task[StatusResponse]]] = {}
        self._lock = asyncio.Lock()

    async def active_count(self) -> int:
        async with self._lock:
            return sum(not task.done() for _, task in self._entries.values())

    async def result(self, key: str) -> StatusResponse | None:
        """Return a cached terminal result, or ``None`` while it is running."""

        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise ServiceError(
                    "OPERATION_NOT_FOUND",
                    "no cached operation matches the idempotency key",
                    status_code=404,
                )
            task = entry[1]
            if not task.done():
                return None
            if task.cancelled():
                raise ServiceError(
                    "OPERATION_CANCELLED",
                    "the cached operation was cancelled",
                    status_code=409,
                )

        return await task

    async def run(
        self,
        key: str,
        request: PickPlaceRequest,
        operation: Callable[[], Awaitable[StatusResponse]],
    ) -> StatusResponse:
        fingerprint = hashlib.sha256(
            json.dumps(request.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()
        async with self._lock:
            existing = self._entries.get(key)
            if existing:
                if existing[0] != fingerprint:
                    raise ServiceError("IDEMPOTENCY_KEY_CONFLICT", "key reused with a different request", status_code=409)
                task = existing[1]
            else:
                task = asyncio.create_task(operation())
                self._entries[key] = (fingerprint, task)
        return await task


def _image_size(data: bytes) -> tuple[int, int]:
    """读取常见 JPEG/PNG 尺寸；未知格式使用稳定默认值。"""

    if data.startswith(b"\x89PNG") and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            length = int.from_bytes(data[index + 2:index + 4], "big")
            if marker in {0xC0, 0xC1, 0xC2, 0xC3} and index + 8 < len(data):
                height = int.from_bytes(data[index + 5:index + 7], "big")
                width = int.from_bytes(data[index + 7:index + 9], "big")
                return width, height
            index += max(length + 2, 2)
    raise ServiceError("INVALID_CAMERA_FRAME", "unable to determine RGB image size", status_code=502)


def _describe_image(data: bytes) -> str:
    """返回适合诊断日志的编码信息，不对图像做有损转换。"""

    if data.startswith(b"\x89PNG") and len(data) >= 26:
        width, height = _image_size(data)
        bit_depth = data[24]
        color_type = data[25]
        color_names = {
            0: "grayscale",
            2: "truecolor",
            3: "indexed",
            4: "grayscale+alpha",
            6: "truecolor+alpha",
        }
        color = color_names.get(color_type, f"unknown({color_type})")
        return f"format=PNG size={width}x{height} bit_depth={bit_depth} color={color}"
    if data.startswith(b"\xff\xd8"):
        width, height = _image_size(data)
        return f"format=JPEG size={width}x{height}"
    return f"format=unknown magic={data[:8].hex() or 'empty'}"


def _image_suffix(data: bytes, default: str) -> str:
    """Return a suffix that matches the encoded camera payload."""

    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    return default


def _http_timeout(settings: PickPlaceSettings, read_seconds: float) -> httpx.Timeout:
    return httpx.Timeout(read_seconds, connect=settings.timeouts.connect_seconds)


def _write_locate_mask(
    path: Path,
    encoded_mask: str,
    width: int,
    height: int,
) -> None:
    try:
        mask = base64.b64decode(encoded_mask.split(",", 1)[-1], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ServiceError("INVALID_MASK", "locate response mask is not valid base64") from exc
    if not mask.startswith(b"\x89PNG"):
        raise ServiceError("INVALID_MASK", "locate response mask is not a PNG image")
    if _image_size(mask) != (width, height):
        raise ServiceError("INVALID_MASK", "locate response mask size does not match RGB")
    path.write_bytes(mask)


def _extract_stream_frame(data: bytes, content_type: str) -> bytes | None:
    """提取 multipart 流中的一帧，避免把 multipart 头和多帧数据写进图像文件。"""

    boundary_match = re.search(r"boundary=\"?([^;\"]+)", content_type, re.IGNORECASE)
    if boundary_match:
        marker = b"--" + boundary_match.group(1).encode()
        start = data.find(marker)
        if start < 0:
            return None
        header_start = start + len(marker)
        header_end = data.find(b"\r\n\r\n", header_start)
        if header_end < 0:
            return None
        headers = data[header_start:header_end].decode("latin-1", errors="replace")
        body_start = header_end + 4
        length_match = re.search(r"(?:^|\r\n)Content-Length:\s*(\d+)", headers, re.IGNORECASE)
        if length_match:
            body_length = int(length_match.group(1))
            if len(data) < body_start + body_length:
                return None
            return data[body_start:body_start + body_length]
        next_boundary = data.find(b"\r\n" + marker, body_start)
        if next_boundary < 0:
            return None
        return data[body_start:next_boundary]

    # 非 multipart 的 JPEG/PNG 响应直接按完整文件处理；JPEG 流仍只取首帧。
    start = data.find(b"\xff\xd8")
    if start >= 0:
        end = data.find(b"\xff\xd9", start + 2)
        return data[start:end + 2] if end >= 0 else None
    if data.startswith(b"\x89PNG"):
        return data
    return None


def _normalize_depth_frame(data: bytes, width: int, height: int) -> bytes:
    """将相机返回的 PNG 或裸 little-endian uint16 转成标准 16 位灰度 PNG。"""

    if data.startswith(b"\x89PNG"):
        return data
    expected = width * height * 2
    if len(data) != expected:
        return data
    rows = []
    for row in range(height):
        raw_row = data[row * width * 2:(row + 1) * width * 2]
        # ROS 16UC1 常见为 little-endian；PNG 样本要求 network byte order。
        values = struct.unpack(f"<{width}H", raw_row)
        rows.append(b"\x00" + struct.pack(f">{width}H", *values))
    return _make_png(width, height, b"".join(rows))


def _depth_array_to_png(depth: np.ndarray, width: int, height: int) -> bytes:
    array = np.asarray(depth)
    if array.shape != (height, width) or not np.issubdtype(array.dtype, np.number):
        raise ServiceError(
            "REFERENCE_INPUT_INVALID",
            f"Task0 depth must be a numeric {height}x{width} array",
        )
    values = array.astype(np.float64)
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 65535)
    ):
        raise ServiceError(
            "REFERENCE_INPUT_INVALID",
            "Task0 depth contains values outside uint16 millimetres",
        )
    encoded = np.rint(values).astype(">u2", copy=False)
    rows = [b"\x00" + encoded[row].tobytes() for row in range(height)]
    return _make_png(width, height, b"".join(rows))


def _validate_pixel_bbox(bbox: list[int], width: int, height: int) -> None:
    if len(bbox) != 4:
        raise ServiceError("INVALID_BBOX", "bbox must contain four pixel values")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ServiceError(
            "INVALID_BBOX", "reference bbox does not fit the Task0 image"
        )


def _make_png(width: int, height: int, scanlines: bytes) -> bytes:
    """用标准库生成 16 位单通道 PNG，避免为运行服务引入图像依赖。"""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b"")


def _create_operation_log(
    settings: PickPlaceSettings,
    request: PickPlaceRequest,
    kind: str,
    operation_key: str,
) -> Path:
    """为一次 pick/place 创建独立的持久化日志目录。"""

    root = Path(settings.log_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", operation_key).strip("._") or "operation"
    directory = root / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{safe_key}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "operation.json").write_text(
        json.dumps(
            {
                "task_type": request.task_type.value,
                "product_name": request.product_name,
                "hand": request.normalized_hand,
                "operation_key": operation_key,
                "request": request.model_dump(mode="json"),
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return directory


def _save_log_json(relative_path: str, payload: Any) -> None:
    """保存 JSON 日志；日志失败不能中断真实机器人流程。"""

    root = _ACTIVE_LOG_DIR.get()
    if root is None:
        return
    try:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError):
        LOGGER.exception("持久化 JSON 日志失败 path=%s", relative_path)


def _append_log_event(event: str, status: str, **details: Any) -> None:
    """向当前运行的 events.jsonl 追加一条实时可读事件。"""

    root = _ACTIVE_LOG_DIR.get()
    if root is None:
        return
    try:
        path = root / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            "status": status,
            **details,
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        LOGGER.exception("持久化流程事件失败 event=%s status=%s", event, status)


def _save_log_file(relative_path: str, content: bytes) -> None:
    """保存二进制输入文件，例如 RGB、depth、mask 和标定文件。"""

    root = _ACTIVE_LOG_DIR.get()
    if root is None:
        return
    try:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    except OSError:
        LOGGER.exception("持久化二进制日志失败 path=%s", relative_path)


def _save_http_response(relative_path: str, response: httpx.Response) -> None:
    """保存下游 HTTP 状态、响应头和 JSON/文本响应。"""

    try:
        body: Any
        try:
            body = response.json()
        except (AttributeError, TypeError, ValueError):
            body = response.text
        _save_log_json(
            relative_path,
            {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": body,
            },
        )
    except Exception:
        LOGGER.exception("持久化 HTTP 响应日志失败 path=%s", relative_path)


def _save_http_error(
    relative_path: str,
    status_code: int,
    error_code: str,
    message: str,
    url: str,
) -> None:
    """保存未收到 HTTP 响应时的结构化传输错误，供 Web 接口时间线展示。"""

    _save_log_json(
        relative_path,
        {
            "status_code": status_code,
            "headers": {},
            "body": {
                "error_code": error_code,
                "message": message,
                "url": url,
                "transport_error": True,
            },
        },
    )


def _write_mask(path: Path, width: int, height: int, bbox: list[int | float]) -> None:
    if len(bbox) != 4:
        raise ServiceError("INVALID_BBOX", "bbox must contain four values", status_code=502)
    x1, y1, x2, y2 = bbox
    left, right = max(0, min(width, round(float(x1) * width / 1000))), max(0, min(width, round(float(x2) * width / 1000)))
    top, bottom = max(0, min(height, round(float(y1) * height / 1000))), max(0, min(height, round(float(y2) * height / 1000)))
    if right <= left or bottom <= top:
        raise ServiceError("INVALID_BBOX", "bbox does not intersect the image", status_code=502)
    rows = []
    for y in range(height):
        rows.append(bytes(255 if left <= x < right and top <= y < bottom else 0 for x in range(width)))
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode() + b"".join(rows))


def _remove_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()
