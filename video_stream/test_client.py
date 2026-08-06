"""Unit tests for VideoStreamClient (no robot / ROS required)."""

from __future__ import annotations

import json
import unittest
from io import BytesIO
from typing import Any, Optional
from unittest import mock
from urllib.error import HTTPError, URLError

from client import VideoStreamClient, VideoStreamError


class _FakeResponse:
    def __init__(
        self,
        payload: bytes | dict[str, Any],
        status: int = 200,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        if isinstance(payload, dict):
            self._raw = json.dumps(payload).encode("utf-8")
        else:
            self._raw = payload
        self._status = status
        self.headers = headers or {}
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            data = self._raw[self._offset :]
            self._offset = len(self._raw)
            return data
        data = self._raw[self._offset : self._offset + size]
        self._offset += len(data)
        return data

    def getcode(self) -> int:
        return self._status

    def close(self) -> None:
        return None

    def __enter__(self) -> "_FakeResponse":
        self._offset = 0
        return self

    def __exit__(self, *args: object) -> None:
        return None


class VideoStreamClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = VideoStreamClient("http://127.0.0.1:8085")

    def test_health_ready(self) -> None:
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _FakeResponse({"status": "READY"})
            self.assertEqual(self.client.health(), "READY")
            self.assertTrue(self.client.is_ready())
            request = urlopen.call_args.args[0]
            self.assertEqual(
                request.full_url, "http://127.0.0.1:8085/camera/health"
            )
            self.assertEqual(request.get_method(), "GET")

    def test_list_cameras(self) -> None:
        payload = {
            "cameras": [
                {
                    "id": "head",
                    "online": True,
                    "streams": [{"type": "color", "online": True}],
                }
            ]
        }
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _FakeResponse(payload)
            self.assertEqual(self.client.list_cameras(), payload)
            request = urlopen.call_args.args[0]
            self.assertEqual(
                request.full_url, "http://127.0.0.1:8085/camera/list"
            )

    def test_snapshot(self) -> None:
        jpeg = b"\xff\xd8\xfffakejpeg\xff\xd9"
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _FakeResponse(jpeg)
            self.assertEqual(self.client.snapshot("head", "color"), jpeg)
            request = urlopen.call_args.args[0]
            self.assertEqual(
                request.full_url,
                "http://127.0.0.1:8085/camera/snapshot?camera=head&type=color",
            )

    def test_snapshot_rejects_invalid_type(self) -> None:
        with self.assertRaises(ValueError):
            self.client.snapshot("head", "rgb")

    def test_stream_url(self) -> None:
        self.assertEqual(
            self.client.stream_url("head", "depth"),
            "http://127.0.0.1:8085/camera/stream?camera=head&type=depth",
        )

    def test_iter_stream_jpegs(self) -> None:
        jpeg = b"\xff\xd8\xffabc\xff\xd9"
        body = (
            b"--tianjiframe\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: %d\r\n\r\n" % len(jpeg)
            + jpeg
            + b"\r\n"
            b"--tianjiframe\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: %d\r\n\r\n" % len(jpeg)
            + jpeg
            + b"\r\n"
            b"--tianjiframe--"
        )
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _FakeResponse(
                body,
                headers={
                    "Content-Type": "multipart/x-mixed-replace; boundary=tianjiframe"
                },
            )
            frames = list(
                self.client.iter_stream_jpegs("head", "color", max_frames=2)
            )
        self.assertEqual(frames, [jpeg, jpeg])

    def test_stream_not_ready(self) -> None:
        error = HTTPError(
            url="http://127.0.0.1:8085/camera/snapshot?camera=head&type=color",
            code=503,
            msg="Service Unavailable",
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(b'{"error_code":"STREAM_NOT_READY"}'),
        )
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(VideoStreamError) as caught:
                self.client.snapshot("head", "color")
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.error_code, "STREAM_NOT_READY")

    def test_gateway_unreachable(self) -> None:
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=URLError("connection refused"),
        ):
            with self.assertRaises(VideoStreamError) as caught:
                self.client.health()
        self.assertEqual(caught.exception.status_code, 0)


if __name__ == "__main__":
    unittest.main()
