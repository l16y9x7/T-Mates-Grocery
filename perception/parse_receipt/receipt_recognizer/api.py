"""Small dependency-free client for the OpenAI-compatible HTTP protocol."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .config import Settings
from .errors import APIConnectionError, APIResponseError


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ChatResponse:
    content: str
    finish_reason: str | None
    usage: JsonObject | None
    raw: JsonObject


class OpenAICompatibleClient:
    """Call only the endpoints needed by this project."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def list_models(self) -> JsonObject:
        return self._request_json("GET", f"{self.settings.base_url}/models")

    def get_openapi(self) -> JsonObject:
        parts = urlsplit(self.settings.base_url)
        path = parts.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        openapi_url = urlunsplit(
            (parts.scheme, parts.netloc, f"{path}/openapi.json", "", "")
        )
        return self._request_json("GET", openapi_url)

    def create_chat_completion(
        self,
        messages: list[JsonObject],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        payload: JsonObject = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        raw = self._request_json(
            "POST",
            f"{self.settings.base_url}/chat/completions",
            payload,
        )

        try:
            choice = raw["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise APIResponseError(
                "Chat Completions 响应缺少 choices[0].message.content。"
            ) from exc

        if not isinstance(content, str):
            raise APIResponseError(
                "choices[0].message.content 不是字符串。"
            )

        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = str(finish_reason)

        usage = raw.get("usage")
        if usage is not None and not isinstance(usage, dict):
            usage = None

        return ChatResponse(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            raw=raw,
        )

    def _request_json(
        self,
        method: str,
        url: str,
        payload: JsonObject | None = None,
    ) -> JsonObject:
        headers = {"Accept": "application/json"}
        body: bytes | None = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(
                request,
                timeout=self.settings.timeout_seconds,
            ) as response:
                raw_body = response.read()
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            summary = _summarize_error_body(error_body)
            raise APIResponseError(
                f"模型 API 返回 HTTP {exc.code}: {summary}",
                status_code=exc.code,
            ) from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            reason = getattr(exc, "reason", exc)
            raise APIConnectionError(
                f"无法连接模型 API：{reason}"
            ) from exc

        try:
            decoded = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise APIResponseError(
                "模型 API 返回的内容不是有效 UTF-8 JSON。"
            ) from exc
        if not isinstance(decoded, dict):
            raise APIResponseError("模型 API 顶层响应不是 JSON 对象。")
        return decoded


def _summarize_error_body(body: str, limit: int = 800) -> str:
    stripped = body.strip()
    if not stripped:
        return "响应体为空"
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped[:limit]

    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message[:limit]
        detail = parsed.get("detail")
        if detail is not None:
            return json.dumps(detail, ensure_ascii=False)[:limit]
    return json.dumps(parsed, ensure_ascii=False)[:limit]

