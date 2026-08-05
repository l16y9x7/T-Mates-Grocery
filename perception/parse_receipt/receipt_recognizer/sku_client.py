"""Client for validating recognized receipt names against the SKU service."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import Settings
from .errors import SKUConnectionError, SKUResponseError


@dataclass(frozen=True)
class SKUValidation:
    name: str
    matched: bool
    locations: list[str]
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "matched": self.matched,
            "locations": self.locations,
        }
        if self.error_code is not None:
            result["error_code"] = self.error_code
        return result


class SkuLookupClient:
    """Call the team's SKU lookup API after Qwen receipt parsing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def validate_items(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            self.locations_for_name(_item_name(item)).to_dict()
            for item in items
        ]

    def locations_for_name(self, name: str) -> SKUValidation:
        normalized_name = name.strip()
        if not normalized_name:
            return SKUValidation(
                name=normalized_name,
                matched=False,
                locations=[],
                error_code="EMPTY_NAME",
            )

        url = (
            f"{self.settings.sku_base_url}/sku/locations?"
            f"{urlencode({'name': normalized_name})}"
        )
        request = Request(
            url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(
                request,
                timeout=self.settings.sku_timeout_seconds,
            ) as response:
                raw_body = response.read()
        except HTTPError as exc:
            if exc.code == 404:
                return SKUValidation(
                    name=normalized_name,
                    matched=False,
                    locations=[],
                    error_code=_error_code_from_http_error(exc)
                    or "SKU_NOT_FOUND",
                )
            raise SKUResponseError(
                f"SKU 服务返回 HTTP {exc.code}: {_summarize_http_error(exc)}",
                status_code=exc.code,
            ) from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            reason = getattr(exc, "reason", exc)
            raise SKUConnectionError(
                f"无法连接 SKU 服务：{reason}"
            ) from exc

        payload = _decode_json_object(raw_body)
        response_name = payload.get("name")
        locations = payload.get("locations")
        if not isinstance(response_name, str) or not isinstance(locations, list):
            raise SKUResponseError("SKU 服务响应缺少 name 或 locations。")
        if not all(isinstance(location, str) for location in locations):
            raise SKUResponseError("SKU 服务响应 locations 必须是字符串数组。")

        return SKUValidation(
            name=response_name,
            matched=True,
            locations=locations,
        )


def _item_name(item: dict[str, Any]) -> str:
    name = item.get("name")
    return name if isinstance(name, str) else ""


def _decode_json_object(raw_body: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SKUResponseError("SKU 服务返回的内容不是有效 UTF-8 JSON。") from exc
    if not isinstance(decoded, dict):
        raise SKUResponseError("SKU 服务顶层响应不是 JSON 对象。")
    return decoded


def _error_code_from_http_error(exc: HTTPError) -> str | None:
    body = exc.read().decode("utf-8", errors="replace")
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    error_code = decoded.get("error_code")
    return error_code if isinstance(error_code, str) else None


def _summarize_http_error(exc: HTTPError, limit: int = 800) -> str:
    body = exc.read().decode("utf-8", errors="replace").strip()
    if not body:
        return "响应体为空"
    return body[:limit]
