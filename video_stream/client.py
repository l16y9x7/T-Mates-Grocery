"""Agent-facing HTTP client for TianJi camera / video gateway.

Talks to retail_camera_http_gateway (default http://127.0.0.1:8085):

- GET /camera/health
- GET /camera/list
- GET /camera/snapshot
- GET /camera/stream
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator, Mapping, Optional
from urllib.parse import urljoin


class VideoStreamError(Exception):
    """Raised when the camera gateway returns a non-success response."""

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


class VideoStreamClient:
    """Thin client for Agent scheduling to call camera health / list / snapshot / stream."""

    VALID_TYPES = ("color", "depth")

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8085",
        *,
        timeout_sec: float = 30.0,
        stream_timeout_sec: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_sec = timeout_sec
        self.stream_timeout_sec = stream_timeout_sec

    def health(self) -> str:
        """Return gateway status string: READY / STARTING / ERROR."""
        body = self._request_json("GET", "camera/health")
        status = body.get("status")
        if not isinstance(status, str) or not status:
            raise VideoStreamError(
                "health response missing status",
                status_code=200,
                body=body,
            )
        return status

    def is_ready(self) -> bool:
        return self.health() == "READY"

    def list_cameras(self) -> dict[str, Any]:
        """Return {"cameras": [...]} from /camera/list."""
        body = self._request_json("GET", "camera/list")
        if "cameras" not in body:
            raise VideoStreamError(
                "list response missing cameras",
                status_code=200,
                body=body,
            )
        return body

    def snapshot(self, camera: str, stream_type: str = "color") -> bytes:
        """Fetch one JPEG frame from /camera/snapshot."""
        camera, stream_type = self._normalize_camera_args(camera, stream_type)
        path = "camera/snapshot?" + urllib.parse.urlencode(
            {"camera": camera, "type": stream_type}
        )
        return self._request_bytes("GET", path, accept="image/jpeg")

    def stream_url(self, camera: str, stream_type: str = "color") -> str:
        """Build absolute MJPEG URL for /camera/stream (does not open the connection)."""
        camera, stream_type = self._normalize_camera_args(camera, stream_type)
        query = urllib.parse.urlencode({"camera": camera, "type": stream_type})
        return urljoin(self.base_url, f"camera/stream?{query}")

    def iter_stream_jpegs(
        self,
        camera: str,
        stream_type: str = "color",
        *,
        max_frames: Optional[int] = None,
    ) -> Iterator[bytes]:
        """Yield JPEG frames from the MJPEG multipart stream."""
        url = self.stream_url(camera, stream_type)
        request = urllib.request.Request(
            url,
            headers={"Accept": "multipart/x-mixed-replace"},
            method="GET",
        )
        try:
            response = urllib.request.urlopen(
                request, timeout=self.stream_timeout_sec
            )
        except urllib.error.HTTPError as exc:
            raise self._http_error_to_exception(exc) from exc
        except urllib.error.URLError as exc:
            raise VideoStreamError(
                f"cannot reach camera gateway: {exc.reason}",
                status_code=0,
            ) from exc

        content_type = response.headers.get("Content-Type", "")
        boundary = self._parse_boundary(content_type)
        yielded = 0
        try:
            for jpeg in self._iter_multipart_jpegs(response, boundary):
                yield jpeg
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
        finally:
            response.close()

    def _normalize_camera_args(
        self, camera: str, stream_type: str
    ) -> tuple[str, str]:
        if not isinstance(camera, str) or not camera.strip():
            raise ValueError("camera must be a non-empty string")
        if not isinstance(stream_type, str) or not stream_type.strip():
            raise ValueError("stream_type must be a non-empty string")
        normalized_type = stream_type.strip().lower()
        if normalized_type not in self.VALID_TYPES:
            raise ValueError(
                f"stream_type must be one of {self.VALID_TYPES}, got {stream_type!r}"
            )
        return camera.strip(), normalized_type

    def _request_json(self, method: str, path: str) -> dict[str, Any]:
        raw = self._request_bytes(method, path, accept="application/json")
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VideoStreamError(
                "camera gateway returned non-JSON body",
                status_code=200,
                body={"raw": raw[:200].decode("utf-8", errors="replace")},
            ) from exc
        if not isinstance(parsed, dict):
            raise VideoStreamError(
                "camera gateway JSON root is not an object",
                status_code=200,
                body={"raw": parsed},
            )
        return parsed

    def _request_bytes(
        self,
        method: str,
        path: str,
        *,
        accept: str,
    ) -> bytes:
        url = urljoin(self.base_url, path)
        request = urllib.request.Request(
            url,
            headers={"Accept": accept},
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_sec
            ) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise self._http_error_to_exception(exc) from exc
        except urllib.error.URLError as exc:
            raise VideoStreamError(
                f"cannot reach camera gateway: {exc.reason}",
                status_code=0,
            ) from exc

    @staticmethod
    def _http_error_to_exception(exc: urllib.error.HTTPError) -> VideoStreamError:
        raw = exc.read()
        body: dict[str, Any] = {}
        error_code: Optional[str] = None
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict):
                body = parsed
                code = parsed.get("error_code")
                if isinstance(code, str):
                    error_code = code
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = {"raw": raw[:200].decode("utf-8", errors="replace")}
        return VideoStreamError(
            f"camera HTTP {exc.code}: {error_code or body}",
            status_code=exc.code,
            error_code=error_code,
            body=body,
        )

    @staticmethod
    def _parse_boundary(content_type: str) -> bytes:
        marker = "boundary="
        lower = content_type.lower()
        idx = lower.find(marker)
        if idx < 0:
            return b"tianjiframe"
        value = content_type[idx + len(marker) :].strip().strip('"')
        if ";" in value:
            value = value.split(";", 1)[0].strip()
        return value.encode("ascii", errors="ignore") or b"tianjiframe"

    @staticmethod
    def _iter_multipart_jpegs(response, boundary: bytes) -> Iterator[bytes]:
        delimiter = b"--" + boundary
        buffer = b""
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            buffer += chunk
            while True:
                start = buffer.find(delimiter)
                if start < 0:
                    # Keep a small tail in case delimiter straddles chunks.
                    if len(buffer) > len(delimiter) + 64:
                        buffer = buffer[-(len(delimiter) + 64) :]
                    break
                next_start = buffer.find(delimiter, start + len(delimiter))
                if next_start < 0:
                    buffer = buffer[start:]
                    break
                part = buffer[start + len(delimiter) : next_start]
                buffer = buffer[next_start:]
                jpeg = VideoStreamClient._extract_jpeg_from_part(part)
                if jpeg is not None:
                    yield jpeg

    @staticmethod
    def _extract_jpeg_from_part(part: bytes) -> Optional[bytes]:
        # Parts look like: \r\nContent-Type: image/jpeg\r\nContent-Length: N\r\n\r\n<bytes>\r\n
        if part.startswith(b"--"):
            return None
        header_end = part.find(b"\r\n\r\n")
        if header_end < 0:
            return None
        body = part[header_end + 4 :]
        if body.endswith(b"\r\n"):
            body = body[:-2]
        if body.startswith(b"\xff\xd8"):
            return body
        return None
