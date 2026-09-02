"""任务一依赖的能力模块 HTTP 客户端。"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
from typing import Any, Callable
from uuid import uuid4

import httpx
from pydantic import ValidationError

from task1_service.models import (
    ActionResponse,
    Hand,
    HealthResponse,
    InterfaceMetric,
    SkuResponse,
    Task1ServiceError,
    Task1Settings,
    TaskType,
)


HEALTH_PATHS = {
    "navigation": "/navigation/health",
    "pose": "/pose/health",
    "pick_place": "/health",
    "sku": "/sku/health",
}
ACTION_RECONCILIATION_SECONDS = 15.0
ACTION_RECONCILIATION_INTERVAL_SECONDS = 0.5


@dataclass
class _InterfaceTotals:
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration_ms: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class _TraceContext:
    callback: Callable[[dict[str, Any]], None]
    session_id: str = field(default_factory=lambda: uuid4().hex)
    next_call_index: int = 0
    totals: dict[str, _InterfaceTotals] = field(default_factory=dict)


class Task1Client:
    """封装任务一所需的导航、位姿、取放和 SKU HTTP 服务。"""

    def __init__(
        self,
        settings: Task1Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(transport=transport)
        # The application owns one shared client. A ContextVar keeps metrics from
        # concurrent readiness probes out of the currently running Task1 trace,
        # while asyncio.gather children created by that run inherit its context.
        self._trace_context: ContextVar[_TraceContext | None] = ContextVar(
            f"task1_trace_context_{id(self)}", default=None
        )

    async def __aenter__(self) -> "Task1Client":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def set_trace_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self._trace_context.set(_TraceContext(callback) if callback is not None else None)

    def interface_metrics(self) -> list[InterfaceMetric]:
        """Return the aggregate collected in the current Task1 trace context."""

        trace_context = self._trace_context.get()
        if trace_context is None:
            return []
        metrics: list[InterfaceMetric] = []
        for interface, totals in trace_context.totals.items():
            metadata = totals.metadata
            metrics.append(
                InterfaceMetric(
                    interface=interface,
                    service=metadata["service"],
                    method=metadata["method"],
                    url=metadata["url"],
                    call_count=totals.call_count,
                    success_count=totals.success_count,
                    failure_count=totals.failure_count,
                    total_duration_ms=totals.total_duration_ms,
                    average_duration_ms=(
                        totals.total_duration_ms / totals.call_count
                    ),
                )
            )
        return metrics

    def _trace(
        self,
        *,
        service: str,
        method: str,
        path: str,
        url: str,
        headers: dict[str, str] | None,
        query: dict[str, Any] | None,
        body: dict[str, Any] | None,
        response: httpx.Response | None = None,
        error: str | None = None,
        attempt: int,
        duration_ms: float,
    ) -> None:
        trace_context = self._trace_context.get()
        if trace_context is None:
            return
        interface = f"{service}{path}"
        succeeded = response is not None and response.is_success
        totals = trace_context.totals.setdefault(interface, _InterfaceTotals())
        totals.metadata = {"service": service, "method": method, "url": url}
        totals.call_count += 1
        if succeeded:
            totals.success_count += 1
        else:
            totals.failure_count += 1
        totals.total_duration_ms += duration_ms
        trace_context.next_call_index += 1
        response_body: object = None
        response_headers: dict[str, str] = {}
        status_code: int | None = None
        if response is not None:
            status_code = response.status_code
            response_headers = dict(response.headers)
            try:
                response_body = response.json()
            except (ValueError, json.JSONDecodeError):
                response_body = response.text
        trace_context.callback({
            "call_id": f"{trace_context.session_id}:{trace_context.next_call_index}",
            "interface": interface,
            "service": service,
            "method": method,
            "url": url,
            "headers": headers or {},
            "query": query or {},
            "body": body,
            "attempt": attempt,
            "status_code": status_code,
            "response_headers": response_headers,
            "response_body": response_body,
            "error": error,
            "duration_ms": duration_ms,
            "call_count": totals.call_count,
            "success_count": totals.success_count,
            "failure_count": totals.failure_count,
            "total_duration_ms": totals.total_duration_ms,
            "average_duration_ms": totals.total_duration_ms / totals.call_count,
        })

    async def check_all_health(self) -> None:
        services = tuple(HEALTH_PATHS)
        # Wait for every probe so all calls are measured before the run either
        # proceeds or returns a health failure.
        results = await asyncio.gather(
            *(self._check_health(service) for service in services),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            first_error = errors[0]
            if isinstance(first_error, asyncio.CancelledError):
                raise first_error
            if isinstance(first_error, Task1ServiceError):
                raise first_error
            raise Task1ServiceError(
                "HEALTH_CHECK_FAILED",
                f"capability health check failed: {first_error}",
            ) from first_error
        not_ready = [service for service, ready in zip(services, results) if not ready]
        if not_ready:
            raise Task1ServiceError(
                "CAPABILITY_NOT_READY",
                f"capability modules are not ready: {', '.join(not_ready)}",
                status_code=503,
            )

    async def health_ready(self) -> bool:
        try:
            await self.check_all_health()
        except Task1ServiceError:
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
            except Task1ServiceError as exc:
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

    async def prepare_pose(
        self,
        pose_type: str,
        idempotency_key: str,
        *,
        shelf_level: str | None = None,
    ) -> None:
        payload: dict[str, str] = {"pose_type": pose_type}
        if shelf_level is not None:
            payload["shelf_level"] = shelf_level
        for attempt in (1, 2):
            key = idempotency_key if attempt == 1 else f"{idempotency_key}:retry"
            try:
                await self._physical_action(
                    "pose",
                    "/pose/prepare",
                    payload,
                    key,
                    self.settings.timeouts.pose_seconds,
                )
            except Task1ServiceError as exc:
                if attempt == 2 or exc.code in {
                    "ACTION_RESULT_UNKNOWN",
                    "NETWORK_ERROR",
                    "INVALID_RESPONSE",
                }:
                    raise
                continue
            return

    async def list_product_names(self) -> list[str]:
        """Return the selectable POS mock catalog from the active SKU service."""

        response = await self._request(
            "sku",
            "GET",
            "/sku/get_all_names",
            timeout_seconds=self.settings.timeouts.sku_seconds,
        )
        try:
            raw_names = response.json()
        except ValueError as exc:
            raise Task1ServiceError(
                "INVALID_RESPONSE", "SKU name list response must be a JSON array"
            ) from exc
        if not isinstance(raw_names, list) or any(
            not isinstance(name, str) or not name.strip() for name in raw_names
        ):
            raise Task1ServiceError(
                "INVALID_RESPONSE", "SKU name list must contain only non-empty strings"
            )
        names = list(dict.fromkeys(name.strip() for name in raw_names))
        if len(names) < 2:
            raise Task1ServiceError(
                "INVALID_RESPONSE", "SKU name list must contain at least two unique names"
            )
        return names

    async def search_by_name(self, name: str) -> SkuResponse:
        last_error: Task1ServiceError | None = None
        for attempt in (1, 2):
            try:
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
                    raise Task1ServiceError(
                        "INVALID_RESPONSE", "SKU name response is invalid"
                    ) from exc
                if result.name != name:
                    raise Task1ServiceError(
                        "INVALID_RESPONSE", "SKU response name does not match request"
                    )
                return result
            except Task1ServiceError as exc:
                last_error = exc
                if attempt == 2 or exc.code == "NETWORK_ERROR":
                    raise
        assert last_error is not None
        raise last_error

    async def search_by_location(self, location: str) -> SkuResponse:
        response = await self._request(
            "sku",
            "GET",
            "/sku/search_by_location",
            params={"location": location},
            timeout_seconds=self.settings.timeouts.sku_seconds,
        )
        try:
            result = SkuResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise Task1ServiceError("INVALID_RESPONSE", "SKU location response is invalid") from exc
        if location not in result.locations:
            raise Task1ServiceError("INVALID_RESPONSE", "SKU response location does not match request")
        return result

    async def pick(
        self,
        product_name: str,
        hand: Hand,
        level: str,
        idempotency_key: str,
        *,
        slot_id: str | None = None,
        target_id: str | None = None,
    ) -> None:
        payload = {
            "task_type": TaskType.SORTING.value,
            "product_name": product_name,
            "hand": hand.value,
            "level": level,
        }
        if slot_id is not None:
            payload["slot_id"] = slot_id
        if target_id is not None:
            # 8086 沿用 PickPlaceRequest.location_id 承载实际导航点。
            payload["location_id"] = target_id
        await self._physical_action(
            "pick_place",
            "/pick",
            payload,
            idempotency_key,
            self.settings.timeouts.pick_seconds,
        )

    async def place(
        self,
        product_name: str,
        hand: Hand,
        idempotency_key: str,
    ) -> None:
        await self._physical_action(
            "pick_place",
            "/place",
            {"task_type": TaskType.SORTING.value, "product_name": product_name, "hand": hand.value},
            idempotency_key,
            self.settings.timeouts.place_seconds,
        )

    async def place_both(
        self,
        left_product_name: str,
        right_product_name: str,
        idempotency_key: str,
    ) -> None:
        await self._physical_action(
            "pose",
            "/manipulation/release/both",
            {
                "task_type": TaskType.SORTING.value,
                "left": {"product_name": left_product_name},
                "right": {"product_name": right_product_name},
            },
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
            raise Task1ServiceError("INVALID_RESPONSE", f"invalid health response from {service}") from exc

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
            error = Task1ServiceError(
                "INVALID_RESPONSE", f"invalid action response from {service}"
            )
            if service == "pick_place":
                await self._reconcile_pick_place_action(idempotency_key, error)
                return
            raise error from exc
        except Task1ServiceError as exc:
            if service == "pick_place" and exc.code == "ACTION_RESULT_UNKNOWN":
                await self._reconcile_pick_place_action(idempotency_key, exc)
                return
            raise

    async def _reconcile_pick_place_action(
        self, idempotency_key: str, original_error: Task1ServiceError
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
            except Task1ServiceError as query_error:
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
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
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
                write=min(10.0, remaining),
                pool=min(5.0, remaining),
            )
            attempt_started_at = loop.time()
            try:
                async with asyncio.timeout(remaining):
                    response = await self._client.request(
                        method,
                        url,
                        params=params,
                        json=json,
                        headers=headers,
                        timeout=timeout,
                    )
            except asyncio.CancelledError:
                duration_ms = max(0.0, loop.time() - attempt_started_at) * 1000.0
                self._trace(
                    service=service,
                    method=method,
                    path=path,
                    url=url,
                    headers=headers,
                    query=params,
                    body=json,
                    error="request cancelled",
                    attempt=attempt + 1,
                    duration_ms=duration_ms,
                )
                raise
            except (TimeoutError, httpx.RequestError) as exc:
                duration_ms = max(0.0, loop.time() - attempt_started_at) * 1000.0
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
                    duration_ms=duration_ms,
                )
                retry_remaining = deadline - loop.time()
                if attempt == 0 and retry_remaining > 0:
                    await asyncio.sleep(min(0.05, retry_remaining))
                    continue
                code = "ACTION_RESULT_UNKNOWN" if result_unknown_on_exhaustion else "NETWORK_ERROR"
                raise Task1ServiceError(code, f"{service} request result could not be determined") from exc
            duration_ms = max(0.0, loop.time() - attempt_started_at) * 1000.0
            if not response.is_success:
                payload: dict[str, Any] = {}
                try:
                    raw_payload = response.json()
                    if isinstance(raw_payload, dict):
                        payload = raw_payload
                except ValueError:
                    payload = {}
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
                    duration_ms=duration_ms,
                )
                raise Task1ServiceError(
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
                duration_ms=duration_ms,
            )
            return response
        raise AssertionError("unreachable retry state")


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
