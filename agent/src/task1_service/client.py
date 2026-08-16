"""任务一依赖的能力模块 HTTP 客户端。"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import httpx
from pydantic import ValidationError

from task1_service.models import (
    ActionResponse,
    Hand,
    HealthResponse,
    ParseReceiptResponse,
    SkuResponse,
    Task1ServiceError,
    Task1Settings,
    TaskType,
)


HEALTH_PATHS = {
    "navigation": "/navigation/health",
    "perception": "/perception/health",
    "pose": "/pose/health",
    "pick_place": "/health",
    "sku": "/sku/health",
}


class Task1Client:
    """封装任务一所需的五个 HTTP 服务。"""

    def __init__(
        self,
        settings: Task1Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(transport=transport)
        self.trace_callback: Callable[[dict[str, Any]], None] | None = None

    async def __aenter__(self) -> "Task1Client":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def set_trace_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self.trace_callback = callback

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
    ) -> None:
        if self.trace_callback is None:
            return
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
        self.trace_callback({
            "interface": f"{service}{path}",
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
        })

    async def check_all_health(self) -> None:
        services = tuple(HEALTH_PATHS)
        results = await asyncio.gather(*(self._check_health(service) for service in services))
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
        await self._physical_action(
            "navigation",
            "/navigation/navigate",
            {"target_id": target_id},
            idempotency_key,
            self.settings.timeouts.navigation_seconds,
        )

    async def nudge_back(self, idempotency_key: str) -> None:
        await self._physical_action(
            "navigation",
            "/navigation/nudge",
            {"action": "approach", "direction": "back"},
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
        await self._physical_action(
            "pose",
            "/pose/prepare",
            payload,
            idempotency_key,
            self.settings.timeouts.pose_seconds,
        )

    async def parse_receipt(self) -> list[str]:
        response = await self._request(
            "perception",
            "POST",
            "/perception/parse",
            timeout_seconds=self.settings.timeouts.receipt_seconds,
        )
        try:
            result = ParseReceiptResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise Task1ServiceError(
                "INVALID_RESPONSE", "receipt response must contain product_names"
            ) from exc
        names = [name.strip() for name in result.product_names if isinstance(name, str)]
        if len(names) != 2 or len(set(names)) != 2 or any(not name for name in names):
            raise Task1ServiceError(
                "INVALID_RECEIPT", "receipt must contain two different non-empty product names",
                status_code=422,
            )
        return names

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
            raise Task1ServiceError("INVALID_RESPONSE", "SKU name response is invalid") from exc
        if result.name != name:
            raise Task1ServiceError("INVALID_RESPONSE", "SKU response name does not match request")
        return result

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
    ) -> None:
        await self._physical_action(
            "pick_place",
            "/pick",
            {
                "task_type": TaskType.SORTING.value,
                "product_name": product_name,
                "hand": hand.value,
                "level": level,
            },
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
            raise Task1ServiceError("INVALID_RESPONSE", f"invalid action response from {service}") from exc

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
                        params=params,
                        json=json,
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
                code = "ACTION_RESULT_UNKNOWN" if result_unknown_on_exhaustion else "NETWORK_ERROR"
                raise Task1ServiceError(code, f"{service} request result could not be determined") from exc
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
                raise Task1ServiceError(
                    code if isinstance(code, str) else "EXECUTION_FAILED",
                    message,
                    failed_interface=(
                        failed_interface if isinstance(failed_interface, str) else None
                    ),
                    url=failed_url if isinstance(failed_url, str) else None,
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
