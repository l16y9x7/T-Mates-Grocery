"""Agent-facing HTTP client for TianJi navigation gateway.

Talks to retail_nav_http_gateway (default http://127.0.0.1:8081):

- GET  /navigation/health
- POST /navigation/navigate
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional
from urllib.parse import urljoin


class NavigationError(Exception):
    """Raised when the navigation gateway returns a non-success response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_code: Optional[str] = None,
        body: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.body = dict(body or {})


class NavigationClient:
    """Thin client for Agent scheduling to call navigation health / navigate."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081",
        *,
        timeout_sec: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_sec = timeout_sec

    def health(self) -> str:
        """Return gateway status string: READY / STARTING / ERROR."""
        body = self._request_json("GET", "navigation/health")
        status = body.get("status")
        if not isinstance(status, str) or not status:
            raise NavigationError(
                "health response missing status",
                status_code=200,
                body=body,
            )
        return status

    def is_ready(self) -> bool:
        return self.health() == "READY"

    def navigate(
        self,
        target_id: str,
        *,
        idempotency_key: str,
    ) -> dict[str, str]:
        """Block until navigation finishes; return {"status": "SUCCEEDED"}."""
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("target_id must be a non-empty string")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError("idempotency_key must be a non-empty string")

        body = self._request_json(
            "POST",
            "navigation/navigate",
            payload={"target_id": target_id.strip()},
            headers={"Idempotency-Key": idempotency_key.strip()},
        )
        status = body.get("status")
        if status != "SUCCEEDED":
            raise NavigationError(
                f"unexpected navigate status: {status!r}",
                status_code=200,
                body=body,
            )
        return {"status": "SUCCEEDED"}

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> dict[str, Any]:
        url = urljoin(self.base_url, path)
        req_headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req_headers["Content-Type"] = "application/json; charset=utf-8"
        if headers:
            req_headers.update(headers)

        request = urllib.request.Request(
            url,
            data=data,
            headers=req_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_sec
            ) as response:
                raw = response.read().decode("utf-8")
                status_code = response.getcode()
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            status_code = exc.code
            body = self._parse_body(raw)
            error_code = body.get("error_code")
            if not isinstance(error_code, str):
                error_code = None
            raise NavigationError(
                f"navigation HTTP {status_code}: {error_code or raw}",
                status_code=status_code,
                error_code=error_code,
                body=body,
            ) from exc
        except urllib.error.URLError as exc:
            raise NavigationError(
                f"cannot reach navigation gateway: {exc.reason}",
                status_code=0,
            ) from exc

        body = self._parse_body(raw)
        if status_code >= 400:
            error_code = body.get("error_code")
            if not isinstance(error_code, str):
                error_code = None
            raise NavigationError(
                f"navigation HTTP {status_code}: {error_code or raw}",
                status_code=status_code,
                error_code=error_code,
                body=body,
            )
        return body

    @staticmethod
    def _parse_body(raw: str) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"raw": parsed}
