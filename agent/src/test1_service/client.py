"""HTTP client for Test1 navigation, pose, and wrist-camera capabilities."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import httpx
from pydantic import ValidationError

from test1_service.camera_frames import (
    extract_stream_frame,
    image_size,
    image_suffix,
    normalize_depth_frame,
)
from test1_service.models import (
    ActionResponse,
    CameraFrame,
    CameraListResponse,
    Hand,
    HealthResponse,
    Test1ServiceError,
    Test1Settings,
)


HEALTH_PATHS = {
    "navigation": "/navigation/health",
    "pose": "/pose/health",
    "camera": "/camera/health",
}
MAX_STREAM_BYTES = 8 * 1024 * 1024


class Test1Client:
    def __init__(
        self,
        settings: Test1Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(transport=transport)
        self.trace_callback: Callable[[dict[str, Any]], None] | None = None

    async def __aenter__(self) -> "Test1Client":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def set_trace_callback(
        self, callback: Callable[[dict[str, Any]], None] | None
    ) -> None:
        self.trace_callback = callback

    async def check_all_health(self) -> None:
        services = tuple(HEALTH_PATHS)
        states = await asyncio.gather(
            *(self._check_health(service) for service in services)
        )
        unavailable = [
            service for service, ready in zip(services, states) if not ready
        ]
        if not unavailable:
            unavailable.extend(await self._unavailable_wrist_streams())
        if unavailable:
            raise Test1ServiceError(
                "CAPABILITY_NOT_READY",
                "capability modules are not ready: " + ", ".join(unavailable),
                status_code=503,
            )

    async def navigate(self, target_id: str, idempotency_key: str) -> None:
        await self._physical_action(
            "navigation",
            "/navigation/navigate",
            {"target_id": target_id},
            idempotency_key,
            self.settings.timeouts.navigation_seconds,
        )

    async def prepare_pose(
        self,
        pose_type: str,
        idempotency_key: str,
        *,
        shelf_level: str | None = None,
    ) -> None:
        payload = {"pose_type": pose_type}
        if shelf_level is not None:
            payload["shelf_level"] = shelf_level
        await self._physical_action(
            "pose",
            "/pose/prepare",
            payload,
            idempotency_key,
            self.settings.timeouts.pose_seconds,
        )

    async def capture(self, hand: Hand) -> CameraFrame:
        camera = hand.camera
        response = await self._request(
            "camera",
            "GET",
            "/camera/snapshot",
            params={"camera": camera, "type": "color"},
            timeout_seconds=self.settings.timeouts.camera_seconds,
        )
        rgb = response.content
        if not rgb:
            raise Test1ServiceError(
                "INVALID_CAMERA_FRAME", f"{camera} RGB snapshot is empty"
            )
        suffix = image_suffix(rgb)
        width, height = image_size(rgb)
        depth = await self._depth_frame(camera)
        return CameraFrame(
            rgb=rgb,
            rgb_suffix=suffix,
            depth=normalize_depth_frame(depth, width, height),
        )

    async def _check_health(self, service: str) -> bool:
        response = await self._request(
            service,
            "GET",
            HEALTH_PATHS[service],
            timeout_seconds=self.settings.timeouts.health_seconds,
        )
        try:
            return HealthResponse.model_validate(response.json()).status == "READY"
        except (ValueError, ValidationError) as exc:
            raise Test1ServiceError(
                "INVALID_RESPONSE", f"invalid health response from {service}"
            ) from exc

    async def _unavailable_wrist_streams(self) -> list[str]:
        response = await self._request(
            "camera",
            "GET",
            "/camera/list",
            timeout_seconds=self.settings.timeouts.health_seconds,
        )
        try:
            cameras = CameraListResponse.model_validate(response.json()).cameras
        except (ValueError, ValidationError) as exc:
            raise Test1ServiceError(
                "INVALID_RESPONSE", "invalid camera list response"
            ) from exc
        unavailable: list[str] = []
        for hand in Hand:
            camera = next((item for item in cameras if item.id == hand.camera), None)
            if camera is None or not camera.online:
                unavailable.append(f"camera.{hand.camera}")
                continue
            online = {stream.type for stream in camera.streams if stream.online}
            for stream_type in ("color", "depth"):
                if stream_type not in online:
                    unavailable.append(f"camera.{hand.camera}.{stream_type}")
        return unavailable

    async def _depth_frame(self, camera: str) -> bytes:
        url = f"{self.settings.services.camera.rstrip('/')}/camera/stream"
        params = {"camera": camera, "type": "depth"}
        headers = {"Accept": "multipart/x-mixed-replace"}
        timeout_seconds = self.settings.timeouts.camera_seconds
        timeout = self._timeout(timeout_seconds)
        for attempt in range(2):
            try:
                async with asyncio.timeout(timeout_seconds):
                    async with self._client.stream(
                        "GET",
                        url,
                        params=params,
                        headers=headers,
                        timeout=timeout,
                    ) as response:
                        if not response.is_success:
                            self._trace(
                                "camera",
                                "GET",
                                "/camera/stream",
                                params=params,
                                headers=headers,
                                attempt=attempt + 1,
                                response=response,
                                response_bytes=0,
                            )
                            raise self._response_error("camera", response)
                        content_type = response.headers.get("content-type", "")
                        data = bytearray()
                        frame: bytes | None = None
                        async for chunk in response.aiter_bytes():
                            data.extend(chunk)
                            frame = extract_stream_frame(bytes(data), content_type)
                            if frame is not None or len(data) >= MAX_STREAM_BYTES:
                                break
                        if not data:
                            raise Test1ServiceError(
                                "INVALID_CAMERA_FRAME",
                                f"{camera} depth stream returned no frame",
                            )
                        if frame is None:
                            frame = extract_stream_frame(bytes(data), content_type)
                        if frame is None and "multipart" not in content_type.lower():
                            frame = bytes(data)
                        if frame is None:
                            raise Test1ServiceError(
                                "INVALID_CAMERA_FRAME",
                                f"{camera} depth stream did not contain a complete frame",
                            )
                        self._trace(
                            "camera",
                            "GET",
                            "/camera/stream",
                            params=params,
                            headers=headers,
                            attempt=attempt + 1,
                            response=response,
                            response_bytes=len(frame),
                        )
                        return frame
            except Test1ServiceError:
                raise
            except (TimeoutError, httpx.TransportError) as exc:
                self._trace(
                    "camera",
                    "GET",
                    "/camera/stream",
                    params=params,
                    headers=headers,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt == 0:
                    await asyncio.sleep(0.05)
                    continue
                raise Test1ServiceError(
                    "NETWORK_ERROR", f"camera depth request failed: {exc}"
                ) from exc
        raise AssertionError("unreachable")

    async def _physical_action(
        self,
        service: str,
        path: str,
        payload: dict[str, str],
        idempotency_key: str,
        timeout_seconds: float,
    ) -> None:
        response = await self._request(
            service,
            "POST",
            path,
            json=payload,
            headers={"Idempotency-Key": idempotency_key},
            timeout_seconds=timeout_seconds,
            result_unknown_on_exhaustion=True,
        )
        try:
            ActionResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise Test1ServiceError(
                "INVALID_RESPONSE", f"invalid action response from {service}"
            ) from exc

    async def _request(
        self,
        service: str,
        method: str,
        path: str,
        *,
        json: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: float,
        result_unknown_on_exhaustion: bool = False,
    ) -> httpx.Response:
        url = f"{getattr(self.settings.services, service).rstrip('/')}{path}"
        for attempt in range(2):
            try:
                async with asyncio.timeout(timeout_seconds):
                    response = await self._client.request(
                        method,
                        url,
                        json=json,
                        params=params,
                        headers=headers,
                        timeout=self._timeout(timeout_seconds),
                    )
                self._trace(
                    service,
                    method,
                    path,
                    params=params,
                    headers=headers,
                    body=json,
                    attempt=attempt + 1,
                    response=response,
                )
                if not response.is_success:
                    raise self._response_error(service, response)
                return response
            except Test1ServiceError:
                raise
            except (TimeoutError, httpx.TransportError) as exc:
                self._trace(
                    service,
                    method,
                    path,
                    params=params,
                    headers=headers,
                    body=json,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt == 0:
                    await asyncio.sleep(0.05)
                    continue
                code = (
                    "ACTION_RESULT_UNKNOWN"
                    if result_unknown_on_exhaustion
                    else "NETWORK_ERROR"
                )
                raise Test1ServiceError(
                    code, f"{service} request failed: {exc}"
                ) from exc
        raise AssertionError("unreachable")

    @staticmethod
    def _response_error(
        service: str, response: httpx.Response
    ) -> Test1ServiceError:
        payload: dict[str, object] = {}
        try:
            candidate = response.json()
            if isinstance(candidate, dict):
                payload = candidate
        except ValueError:
            pass
        code = payload.get("error_code")
        detail = payload.get("message") or payload.get("detail")
        message = f"{service} returned HTTP {response.status_code}"
        if isinstance(detail, str) and detail:
            message = f"{message}: {detail}"
        return Test1ServiceError(
            code if isinstance(code, str) else "EXECUTION_FAILED", message
        )

    def _timeout(self, read_seconds: float) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.settings.timeouts.connect_seconds,
            read=read_seconds,
            write=10,
            pool=5,
        )

    def _trace(
        self,
        service: str,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        body: dict[str, str] | None = None,
        attempt: int,
        response: httpx.Response | None = None,
        response_bytes: int | None = None,
        error: str | None = None,
    ) -> None:
        if self.trace_callback is None:
            return
        self.trace_callback(
            {
                "interface": path,
                "service": service,
                "method": method,
                "url": f"{getattr(self.settings.services, service).rstrip('/')}{path}",
                "params": params or {},
                "headers": headers or {},
                "body": body,
                "attempt": attempt,
                "status_code": response.status_code if response is not None else None,
                "response_bytes": response_bytes
                if response_bytes is not None
                else (len(response.content) if response is not None else None),
                "error": error,
            }
        )
