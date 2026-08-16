"""HTTP client for task 0 navigation, pose, and camera capabilities."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import httpx
from pydantic import ValidationError

from task0_service.models import (
    ActionResponse,
    CameraListResponse,
    HealthResponse,
    Task0ServiceError,
    Task0Settings,
)


HEALTH_PATHS = {
    "navigation": "/navigation/health",
    "pose": "/pose/health",
    "camera": "/camera/health",
}


class Task0Client:
    def __init__(
        self,
        settings: Task0Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(transport=transport)
        self.trace_callback: Callable[[dict[str, Any]], None] | None = None

    async def __aenter__(self) -> "Task0Client":
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
        ready = await asyncio.gather(
            *(self._check_health(service) for service in services)
        )
        unavailable = [
            service for service, is_ready in zip(services, ready) if not is_ready
        ]
        if not unavailable and not await self._head_rgbd_ready():
            unavailable.append("camera.head.rgbd")
        if unavailable:
            raise Task0ServiceError(
                "CAPABILITY_NOT_READY",
                f"capability modules are not ready: {', '.join(unavailable)}",
                status_code=503,
            )

    async def health_ready(self) -> bool:
        try:
            await self.check_all_health()
        except Task0ServiceError:
            return False
        return True

    async def navigate(self, target_id: str, idempotency_key: str) -> None:
        await self._physical_action(
            "navigation",
            "/navigation/navigate",
            {"target_id": target_id},
            idempotency_key,
            self.settings.timeouts.navigation_seconds,
        )

    async def prepare_pose(self, pose_type: str, idempotency_key: str) -> None:
        await self._physical_action(
            "pose",
            "/pose/prepare",
            {"pose_type": pose_type},
            idempotency_key,
            self.settings.timeouts.pose_seconds,
        )

    async def capture_rgbd(self) -> bytes:
        response = await self._request(
            "camera",
            "GET",
            "/camera/rgbd",
            params={"camera": self.settings.camera},
            timeout_seconds=self.settings.timeouts.camera_seconds,
        )
        if not response.content:
            raise Task0ServiceError(
                "INVALID_CAMERA_RESPONSE", "camera RGB-D response is empty"
            )
        return response.content

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
            raise Task0ServiceError(
                "INVALID_RESPONSE", f"invalid health response from {service}"
            ) from exc

    async def _head_rgbd_ready(self) -> bool:
        response = await self._request(
            "camera",
            "GET",
            "/camera/list",
            timeout_seconds=self.settings.timeouts.health_seconds,
        )
        try:
            cameras = CameraListResponse.model_validate(response.json()).cameras
        except (ValueError, ValidationError) as exc:
            raise Task0ServiceError(
                "INVALID_RESPONSE", "invalid camera list response"
            ) from exc
        head = next(
            (camera for camera in cameras if camera.id == self.settings.camera), None
        )
        if head is None or not head.online:
            return False
        online_streams = {stream.type for stream in head.streams if stream.online}
        return {"color", "depth"}.issubset(online_streams)

    async def _physical_action(
        self,
        service: str,
        path: str,
        payload: dict[str, Any],
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
            raise Task0ServiceError(
                "INVALID_RESPONSE", f"invalid action response from {service}"
            ) from exc

    async def _request(
        self,
        service: str,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: float,
        result_unknown_on_exhaustion: bool = False,
    ) -> httpx.Response:
        url = f"{getattr(self.settings.services, service).rstrip('/')}{path}"
        timeout = httpx.Timeout(
            connect=self.settings.timeouts.connect_seconds,
            read=timeout_seconds,
            write=10.0,
            pool=5.0,
        )
        for attempt in range(2):
            try:
                async with asyncio.timeout(timeout_seconds):
                    response = await self._client.request(
                        method,
                        url,
                        json=json,
                        params=params,
                        headers=headers,
                        timeout=timeout,
                    )
            except (TimeoutError, httpx.TimeoutException, httpx.NetworkError) as exc:
                self._trace(
                    service=service,
                    method=method,
                    path=path,
                    url=url,
                    params=params,
                    headers=headers,
                    body=json,
                    error=str(exc),
                    attempt=attempt + 1,
                )
                if attempt == 0:
                    await asyncio.sleep(0.05)
                    continue
                code = (
                    "ACTION_RESULT_UNKNOWN"
                    if result_unknown_on_exhaustion
                    else "NETWORK_ERROR"
                )
                raise Task0ServiceError(
                    code, f"{service} request result could not be determined"
                ) from exc
            if not response.is_success:
                error_payload: dict[str, Any] = {}
                try:
                    raw_payload = response.json()
                    if isinstance(raw_payload, dict):
                        error_payload = raw_payload
                except ValueError:
                    pass
                code = error_payload.get("error_code", "EXECUTION_FAILED")
                detail = error_payload.get("message") or error_payload.get("detail")
                message = f"{service} returned HTTP {response.status_code}"
                if detail:
                    message = f"{message}: {detail}"
                self._trace(
                    service=service,
                    method=method,
                    path=path,
                    url=url,
                    params=params,
                    headers=headers,
                    body=json,
                    response=response,
                    attempt=attempt + 1,
                )
                raise Task0ServiceError(
                    code if isinstance(code, str) else "EXECUTION_FAILED", message
                )
            self._trace(
                service=service,
                method=method,
                path=path,
                url=url,
                params=params,
                headers=headers,
                body=json,
                response=response,
                attempt=attempt + 1,
            )
            return response
        raise AssertionError("unreachable")

    def _trace(
        self,
        *,
        service: str,
        method: str,
        path: str,
        url: str,
        params: dict[str, str] | None,
        headers: dict[str, str] | None,
        body: dict[str, Any] | None,
        attempt: int,
        response: httpx.Response | None = None,
        error: str | None = None,
    ) -> None:
        if self.trace_callback is None:
            return
        response_body: object | None = None
        if response is not None:
            content_type = response.headers.get("content-type", "")
            if "json" in content_type:
                try:
                    response_body = response.json()
                except ValueError:
                    response_body = response.text
            else:
                response_body = {
                    "content_type": content_type or "application/octet-stream",
                    "bytes": len(response.content),
                }
        try:
            self.trace_callback(
                {
                    "interface": path,
                    "service": service,
                    "method": method,
                    "url": url,
                    "params": params or {},
                    "headers": headers or {},
                    "body": body,
                    "attempt": attempt,
                    "status_code": response.status_code if response is not None else None,
                    "response_headers": dict(response.headers) if response is not None else {},
                    "response_body": response_body,
                    "error": error,
                }
            )
        except Exception:
            return
