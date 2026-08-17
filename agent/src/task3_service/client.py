"""HTTP client for Task 3 capability services."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import httpx
from pydantic import ValidationError

from task3_service.models import (
    ActionResponse,
    CameraListResponse,
    Hand,
    HealthResponse,
    InspectionPoint,
    InspectionPose,
    InspectionResponse,
    MisplacedFinding,
    SkuResponse,
    Task3ServiceError,
    Task3Settings,
    TaskType,
)


HEALTH_PATHS = {
    "navigation": "/navigation/health",
    "perception": "/perception/health",
    "pose": "/pose/health",
    "pick_place": "/health",
    "sku": "/sku/health",
    "camera": "/camera/health",
}


def _execution_pose(value: object) -> list[float] | None:
    if (
        not isinstance(value, list)
        or len(value) != 6
        or any(
            not isinstance(item, (int, float)) or isinstance(item, bool)
            for item in value
        )
    ):
        return None
    return [float(item) for item in value]


class Task3Client:
    def __init__(
        self,
        settings: Task3Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(transport=transport)
        self.trace_callback: Callable[[dict[str, Any]], None] | None = None

    async def __aenter__(self) -> "Task3Client":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def set_trace_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self.trace_callback = callback

    async def check_all_health(self) -> None:
        services = tuple(HEALTH_PATHS)
        results = await asyncio.gather(*(self._check_health(service) for service in services))
        not_ready = [service for service, ready in zip(services, results) if not ready]
        if not not_ready and not await self._head_color_ready():
            not_ready.append("camera.head.color")
        if not_ready:
            raise Task3ServiceError(
                "CAPABILITY_NOT_READY",
                f"capability modules are not ready: {', '.join(not_ready)}",
                status_code=503,
            )

    async def health_ready(self) -> bool:
        try:
            await self.check_all_health()
        except Task3ServiceError:
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

    async def nudge(self, direction: str, idempotency_key: str) -> None:
        await self._physical_action(
            "navigation",
            "/navigation/nudge",
            {"action": "approach", "direction": direction},
            idempotency_key,
            self.settings.timeouts.navigation_seconds,
        )

    async def nudge_return(self, idempotency_key: str) -> None:
        await self._physical_action(
            "navigation",
            "/navigation/nudge",
            {"action": "return"},
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

    async def inspect(
        self,
        point: InspectionPoint,
        pose: InspectionPose,
    ) -> list[MisplacedFinding]:
        response = await self._request(
            "perception",
            "POST",
            "/perception/inspect",
            json={
                "task_type": TaskType.MISPLACED.value,
                "location_id": point.target_id,
                "pose_type": pose.value,
            },
            timeout_seconds=self.settings.timeouts.inspection_seconds,
        )
        try:
            return InspectionResponse.model_validate(response.json()).findings
        except (ValueError, ValidationError) as exc:
            raise Task3ServiceError(
                "INVALID_RESPONSE", "inspection response must contain misplaced findings"
            ) from exc

    async def search_by_name(self, name: str) -> SkuResponse:
        response = await self._request(
            "sku",
            "GET",
            "/sku/search_by_name",
            params={"name": name},
            timeout_seconds=self.settings.timeouts.sku_seconds,
        )
        try:
            result = SkuResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise Task3ServiceError("INVALID_RESPONSE", "SKU response is invalid") from exc
        if result.name != name:
            raise Task3ServiceError(
                "INVALID_RESPONSE", "SKU response name does not match request"
            )
        return result

    async def pick(
        self, product_name: str, hand: Hand, level: str, idempotency_key: str
    ) -> None:
        await self._pick_place_action(
            "/pick",
            product_name,
            hand,
            idempotency_key,
            self.settings.timeouts.pick_seconds,
            level=level,
        )

    async def place(
        self,
        product_name: str,
        hand: Hand,
        location_id: str,
        pose_type: str,
        idempotency_key: str,
    ) -> None:
        await self._pick_place_action(
            "/place",
            product_name,
            hand,
            idempotency_key,
            self.settings.timeouts.place_seconds,
            location_id=location_id,
            pose_type=pose_type,
        )

    async def _pick_place_action(
        self,
        path: str,
        product_name: str,
        hand: Hand,
        idempotency_key: str,
        timeout_seconds: float,
        *,
        level: str | None = None,
        location_id: str | None = None,
        pose_type: str | None = None,
    ) -> None:
        payload = {
            "task_type": TaskType.MISPLACED.value,
            "product_name": product_name,
            "hand": hand.value,
        }
        if level is not None:
            payload["level"] = level
        if location_id is not None:
            payload["location_id"] = location_id
        if pose_type is not None:
            payload["pose_type"] = pose_type
        await self._physical_action(
            "pick_place",
            path,
            payload,
            idempotency_key,
            timeout_seconds,
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
            raise Task3ServiceError(
                "INVALID_RESPONSE", f"invalid health response from {service}"
            ) from exc

    async def _head_color_ready(self) -> bool:
        response = await self._request(
            "camera",
            "GET",
            "/camera/list",
            timeout_seconds=self.settings.timeouts.health_seconds,
        )
        try:
            cameras = CameraListResponse.model_validate(response.json()).cameras
        except (ValueError, ValidationError) as exc:
            raise Task3ServiceError(
                "INVALID_RESPONSE", "invalid camera list response"
            ) from exc
        head = next((camera for camera in cameras if camera.id == self.settings.camera), None)
        if head is None or not head.online:
            return False
        return any(stream.type == "color" and stream.online for stream in head.streams)

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
            raise Task3ServiceError(
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
            write=max(10.0, min(timeout_seconds, 60.0)),
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
                    headers=headers,
                    query=params,
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
                raise Task3ServiceError(
                    code, f"{service} request result could not be determined"
                ) from exc
            if not response.is_success:
                payload: dict[str, Any] = {}
                try:
                    raw_payload = response.json()
                    if isinstance(raw_payload, dict):
                        payload = raw_payload
                except ValueError:
                    pass
                code = payload.get("error_code", "EXECUTION_FAILED")
                detail = payload.get("message") or payload.get("detail")
                failed_interface = payload.get("failed_interface")
                failed_url = payload.get("url")
                failed_pose = _execution_pose(payload.get("pose"))
                message = f"{service} returned HTTP {response.status_code}"
                if detail:
                    message = f"{message}: {detail}"
                self._trace(
                    service=service,
                    method=method,
                    path=path,
                    url=url,
                    headers=headers,
                    query=params,
                    body=json,
                    response=response,
                    attempt=attempt + 1,
                )
                raise Task3ServiceError(
                    code if isinstance(code, str) else "EXECUTION_FAILED",
                    message,
                    failed_interface=(
                        failed_interface if isinstance(failed_interface, str) else None
                    ),
                    url=failed_url if isinstance(failed_url, str) else None,
                    pose=failed_pose,
                )
            self._trace(
                service=service,
                method=method,
                path=path,
                url=url,
                headers=headers,
                query=params,
                body=json,
                response=response,
                attempt=attempt + 1,
            )
            return response
        raise AssertionError("unreachable retry state")

    def _trace(
        self,
        *,
        service: str,
        method: str,
        path: str,
        url: str,
        headers: dict[str, str] | None,
        query: dict[str, str] | None,
        body: dict[str, Any] | None,
        response: httpx.Response | None = None,
        error: str | None = None,
        attempt: int,
    ) -> None:
        if self.trace_callback is None:
            return
        response_body: object = None
        response_headers: dict[str, str] = {}
        status_code: int | None = None
        if response is not None:
            status_code = response.status_code
            response_headers = dict(response.headers)
            content_type = response.headers.get("content-type", "")
            if "json" in content_type:
                try:
                    response_body = response.json()
                except (ValueError, json.JSONDecodeError):
                    response_body = response.text
            else:
                response_body = f"<binary:{len(response.content)} bytes>"
        traced_body = dict(body) if body is not None else None
        self.trace_callback(
            {
                "interface": f"{service}{path}",
                "service": service,
                "method": method,
                "url": url,
                "headers": headers or {},
                "query": query or {},
                "body": traced_body,
                "attempt": attempt,
                "status_code": status_code,
                "response_headers": response_headers,
                "response_body": response_body,
                "error": error,
            }
        )
