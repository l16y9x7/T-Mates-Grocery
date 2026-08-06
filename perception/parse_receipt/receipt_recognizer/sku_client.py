"""Client for validating recognized receipt names against the SKU service."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import Settings
from .errors import SKUConnectionError, SKUNotFoundError, SKUResponseError


@dataclass(frozen=True)
class SKULocation:
    name: str
    locations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "locations": self.locations,
        }


@dataclass(frozen=True)
class SKUCandidate:
    name: str
    locations: list[str]
    distance: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "locations": self.locations,
            "distance": self.distance,
        }


class SkuLookupClient:
    """Call the team's SKU lookup API after Qwen receipt parsing."""

    _all_names_cache: ClassVar[dict[str, list[str]]] = {}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def lookup_items(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [self.lookup_item(item) for item in items]

    def lookup_item(self, item: dict[str, Any]) -> dict[str, Any]:
        name = _item_name(item)
        try:
            return self.locations_for_name(name).to_dict()
        except SKUNotFoundError:
            candidates = self.edit_distance_candidates(name)
            if not candidates:
                raise
            return {
                "recognized_name": name.strip(),
                "match_type": "edit_distance",
                "candidates": [candidate.to_dict() for candidate in candidates],
            }

    def locations_for_name(self, name: str) -> SKULocation:
        normalized_name = name.strip()
        if not normalized_name:
            raise SKUNotFoundError(normalized_name, "EMPTY_NAME")

        url = (
            f"{self.settings.sku_base_url}/sku/search_by_name?"
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
                raise SKUNotFoundError(
                    normalized_name,
                    _error_code_from_http_error(exc) or "SKU_NOT_FOUND",
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

        return SKULocation(
            name=response_name,
            locations=locations,
        )

    def edit_distance_candidates(self, name: str) -> list[SKUCandidate]:
        normalized_name = name.strip()
        if not normalized_name:
            return []

        ranked_names = sorted(
            (
                (_edit_distance(normalized_name, sku_name), sku_name)
                for sku_name in self.all_names()
            ),
            key=lambda value: (value[0], value[1]),
        )

        candidates: list[SKUCandidate] = []
        for distance, sku_name in ranked_names:
            if distance > self.settings.sku_edit_distance_max:
                break
            try:
                location = self.locations_for_name(sku_name)
            except SKUNotFoundError:
                continue
            candidates.append(
                SKUCandidate(
                    name=location.name,
                    locations=location.locations,
                    distance=distance,
                )
            )
            if len(candidates) >= self.settings.sku_fuzzy_limit:
                break
        return candidates

    def all_names(self) -> list[str]:
        cached = self._all_names_cache.get(self.settings.sku_base_url)
        if cached is not None:
            return list(cached)

        url = f"{self.settings.sku_base_url}/sku/get_all_names"
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
            raise SKUResponseError(
                f"SKU 服务返回 HTTP {exc.code}: {_summarize_http_error(exc)}",
                status_code=exc.code,
            ) from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            reason = getattr(exc, "reason", exc)
            raise SKUConnectionError(
                f"无法连接 SKU 服务：{reason}"
            ) from exc

        names = _decode_json_string_list(raw_body)
        self._all_names_cache[self.settings.sku_base_url] = names
        return list(names)


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


def _decode_json_string_list(raw_body: bytes) -> list[str]:
    try:
        decoded = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SKUResponseError("SKU 服务返回的内容不是有效 UTF-8 JSON。") from exc
    if not isinstance(decoded, list) or not all(
        isinstance(name, str) for name in decoded
    ):
        raise SKUResponseError("SKU 名称列表响应必须是字符串数组。")
    return [name.strip() for name in decoded if name.strip()]


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


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
