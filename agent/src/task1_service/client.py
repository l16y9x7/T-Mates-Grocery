"""任务一依赖的能力模块 HTTP 客户端。"""

from __future__ import annotations

import asyncio
from typing import Any

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

    async def __aenter__(self) -> "Task1Client":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

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
            json={"name": name},
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
            json={"location": location},
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
        idempotency_key: str,
    ) -> None:
        await self._physical_action(
            "pick_place",
            "/pick",
            {"task_type": TaskType.SORTING.value, "product_name": product_name, "hand": hand.value},
            idempotency_key,
            self.settings.timeouts.pick_seconds,
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
                        method, url, json=json, headers=headers, timeout=timeout
                    )
            except (TimeoutError, httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == 0:
                    await asyncio.sleep(0.05)
                    continue
                code = "ACTION_RESULT_UNKNOWN" if result_unknown_on_exhaustion else "NETWORK_ERROR"
                raise Task1ServiceError(code, f"{service} request result could not be determined") from exc
            if not response.is_success:
                try:
                    code = response.json().get("error_code", "EXECUTION_FAILED")
                except ValueError:
                    code = "EXECUTION_FAILED"
                raise Task1ServiceError(
                    code if isinstance(code, str) else "EXECUTION_FAILED",
                    f"{service} returned HTTP {response.status_code}",
                )
            return response
        raise AssertionError("unreachable retry state")
