"""任务二依赖的能力模块 HTTP 客户端。"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import httpx
from pydantic import ValidationError

from task2_service.models import (
    ActionResponse,
    CameraListResponse,
    Hand,
    HealthResponse,
    ShortageProductFinding,
    Task2ServiceError,
    Task2Settings,
    TaskType,
)


HEALTH_PATHS = {
    "navigation": "/navigation/health",
    "perception": "/perception/health",
    "pose": "/pose/health",
    "pick_place": "/health",
    "camera": "/camera/health",
}
ACTION_RECONCILIATION_SECONDS = 15.0
ACTION_RECONCILIATION_INTERVAL_SECONDS = 0.5


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


class Task2Client:
    def __init__(
        self,
        settings: Task2Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(transport=transport)
        self.trace_callback: Callable[[dict[str, Any]], None] | None = None

    async def __aenter__(self) -> "Task2Client":
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
            raise Task2ServiceError(
                "CAPABILITY_NOT_READY",
                f"capability modules are not ready: {', '.join(not_ready)}",
                status_code=503,
            )

    async def health_ready(self) -> bool:
        try:
            await self.check_all_health()
        except Task2ServiceError:
            return False
        return True

    async def navigate(self, target_id: str, idempotency_key: str) -> None:
        for attempt in (1, 2):
            key = idempotency_key if attempt == 1 else f"{idempotency_key}:retry"
            try:
                await self._physical_action(
                    "navigation",
                    "/navigation/navigate",
                    {"target_id": target_id},
                    key,
                    self.settings.timeouts.navigation_seconds,
                )
            except Task2ServiceError as exc:
                if attempt == 2 or exc.code in {
                    "ACTION_RESULT_UNKNOWN",
                    "NETWORK_ERROR",
                    "INVALID_RESPONSE",
                }:
                    raise
                continue
            return

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

    async def prepare_pose(self, pose_type: str, idempotency_key: str) -> None:
        for attempt in (1, 2):
            key = idempotency_key if attempt == 1 else f"{idempotency_key}:retry"
            try:
                await self._physical_action(
                    "pose",
                    "/pose/prepare",
                    {"pose_type": pose_type},
                    key,
                    self.settings.timeouts.pose_seconds,
                )
            except Task2ServiceError as exc:
                if attempt == 2 or exc.code in {
                    "ACTION_RESULT_UNKNOWN",
                    "NETWORK_ERROR",
                    "INVALID_RESPONSE",
                }:
                    raise
                continue
            return

    async def open_gripper(self, hand: Hand, idempotency_key: str) -> None:
        """Release a held item at the replenishment table."""
        await self._physical_action(
            "pose",
            "/manipulation/gripper/open",
            {"hand": hand.value},
            idempotency_key,
            self.settings.timeouts.pose_seconds,
        )

    async def inspect(
        self,
        location_id: str,
        pose_type: str,
    ) -> list[ShortageProductFinding]:
        last_error: Task2ServiceError | None = None
        for attempt in (1, 2):
            try:
                response = await self._request(
                    "perception",
                    "POST",
                    "/perception/inspect",
                    json={
                        "task_type": TaskType.SHORTAGE.value,
                        "location_id": location_id,
                        "pose_type": pose_type,
                    },
                    timeout_seconds=self.settings.timeouts.inspection_seconds,
                )
                try:
                    raw = response.json()
                    raw_findings = raw["findings"]
                    if not isinstance(raw_findings, list):
                        raise TypeError("findings must be a list")
                except (KeyError, TypeError, ValueError) as exc:
                    raise Task2ServiceError(
                        "INVALID_RESPONSE", "inspection response must contain findings"
                    ) from exc
                findings: list[ShortageProductFinding] = []
                for raw_finding in raw_findings:
                    try:
                        finding = ShortageProductFinding.model_validate(raw_finding)
                    except ValidationError as exc:
                        if self.settings.product_hand_options_schema_version == "2.0":
                            raise Task2ServiceError(
                                "INVALID_RESPONSE",
                                "schema 2.0 inspection finding is malformed",
                            ) from exc
                        continue
                    name = finding.shortage_product_name.strip()
                    if not name:
                        if self.settings.product_hand_options_schema_version == "2.0":
                            raise Task2ServiceError(
                                "INVALID_RESPONSE",
                                "schema 2.0 inspection finding has a blank product name",
                            )
                        continue
                    if (
                        self.settings.product_hand_options_schema_version == "2.0"
                        and finding.slot_id is None
                    ):
                        raise Task2ServiceError(
                            "INVALID_RESPONSE",
                            "schema 2.0 inspection finding is missing slot_id",
                        )
                    findings.append(
                        finding.model_copy(
                            update={"shortage_product_name": name}
                        )
                    )
                return findings
            except Task2ServiceError as exc:
                last_error = exc
                if attempt == 2 or exc.code in {"NETWORK_ERROR", "INVALID_RESPONSE"}:
                    raise
        assert last_error is not None
        raise last_error

    async def pick(self, product_name: str, hand: Hand, idempotency_key: str) -> None:
        await self._physical_action(
            "pick_place",
            "/pick",
            {
                "task_type": TaskType.SHORTAGE.value,
                "product_name": product_name,
                "hand": hand.value,
            },
            idempotency_key,
            self.settings.timeouts.pick_seconds,
        )

    async def place(
        self,
        product_name: str,
        hand: Hand,
        location_id: str,
        pose_type: str,
        idempotency_key: str,
        slot_id: str | None = None,
    ) -> None:
        payload = {
            "task_type": TaskType.SHORTAGE.value,
            "product_name": product_name,
            "hand": hand.value,
            "location_id": location_id,
            "pose_type": pose_type,
        }
        if slot_id is not None:
            payload["slot_id"] = slot_id
        await self._physical_action(
            "pick_place",
            "/place",
            payload,
            idempotency_key,
            self.settings.timeouts.place_seconds,
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
            raise Task2ServiceError(
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
            raise Task2ServiceError(
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
        try:
            response = await self._request(
                service,
                "POST",
                path,
                json=payload,
                headers={"Idempotency-Key": idempotency_key},
                timeout_seconds=timeout_seconds,
                result_unknown_on_exhaustion=True,
            )
            ActionResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            error = Task2ServiceError(
                "INVALID_RESPONSE", f"invalid action response from {service}"
            )
            if service == "pick_place":
                await self._reconcile_pick_place_action(idempotency_key, error)
                return
            raise error from exc
        except Task2ServiceError as exc:
            if service == "pick_place" and exc.code == "ACTION_RESULT_UNKNOWN":
                await self._reconcile_pick_place_action(idempotency_key, exc)
                return
            raise

    async def _reconcile_pick_place_action(
        self, idempotency_key: str, original_error: Task2ServiceError
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + ACTION_RECONCILIATION_SECONDS
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise original_error
            try:
                response = await self._request(
                    "pick_place",
                    "GET",
                    "/operations/result",
                    params={"idempotency_key": idempotency_key},
                    timeout_seconds=min(1.0, remaining),
                )
            except Task2ServiceError as query_error:
                if query_error.code in {
                    "OPERATION_NOT_FOUND",
                    "UNKNOWN_ENDPOINT",
                    "NETWORK_ERROR",
                    "INVALID_RESPONSE",
                }:
                    raise original_error from query_error
                raise
            if response.status_code == 200:
                try:
                    ActionResponse.model_validate(response.json())
                except (ValueError, ValidationError) as exc:
                    raise original_error from exc
                return
            try:
                status = response.json().get("status")
            except (AttributeError, ValueError):
                raise original_error
            if response.status_code != 202 or status != "RUNNING":
                raise original_error
            await asyncio.sleep(
                min(ACTION_RECONCILIATION_INTERVAL_SECONDS, max(0.0, remaining))
            )

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
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        for attempt in range(2):
            remaining = max(0.001, deadline - loop.time())
            timeout = httpx.Timeout(
                connect=min(self.settings.timeouts.connect_seconds, remaining),
                read=remaining,
                write=min(max(10.0, min(timeout_seconds, 60.0)), remaining),
                pool=min(5.0, remaining),
            )
            try:
                async with asyncio.timeout(remaining):
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
                retry_remaining = deadline - loop.time()
                if attempt == 0 and retry_remaining > 0:
                    await asyncio.sleep(min(0.05, retry_remaining))
                    continue
                code = "ACTION_RESULT_UNKNOWN" if result_unknown_on_exhaustion else "NETWORK_ERROR"
                raise Task2ServiceError(
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
                raise Task2ServiceError(
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
