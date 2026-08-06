"""Unit tests for NavigationClient (no robot / ROS required)."""

from __future__ import annotations

import json
import unittest
from io import BytesIO
from typing import Any
from unittest import mock
from urllib.error import HTTPError, URLError

from client import NavigationClient, NavigationError


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._raw = json.dumps(payload).encode("utf-8")
        self._status = status

    def read(self) -> bytes:
        return self._raw

    def getcode(self) -> int:
        return self._status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class NavigationClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = NavigationClient("http://127.0.0.1:8081")

    def test_health_ready(self) -> None:
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _FakeResponse({"status": "READY"})
            self.assertEqual(self.client.health(), "READY")
            self.assertTrue(self.client.is_ready())
            request = urlopen.call_args.args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:8081/navigation/health")
            self.assertEqual(request.get_method(), "GET")

    def test_navigate_success(self) -> None:
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _FakeResponse({"status": "SUCCEEDED"})
            result = self.client.navigate(
                "H1_F_L1_C01",
                idempotency_key="agent:nav-001",
            )
            self.assertEqual(result, {"status": "SUCCEEDED"})
            request = urlopen.call_args.args[0]
            self.assertEqual(
                request.full_url, "http://127.0.0.1:8081/navigation/navigate"
            )
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.get_header("Idempotency-key"), "agent:nav-001")
            body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(body, {"target_id": "H1_F_L1_C01"})

    def test_navigate_rejects_empty_idempotency_key(self) -> None:
        with self.assertRaises(ValueError):
            self.client.navigate("H1_F_L1_C01", idempotency_key="  ")

    def test_module_not_ready(self) -> None:
        error = HTTPError(
            url="http://127.0.0.1:8081/navigation/navigate",
            code=503,
            msg="Service Unavailable",
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(b'{"error_code":"MODULE_NOT_READY"}'),
        )
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(NavigationError) as caught:
                self.client.navigate(
                    "delivery_place",
                    idempotency_key="agent:nav-002",
                )
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(caught.exception.error_code, "MODULE_NOT_READY")

    def test_gateway_unreachable(self) -> None:
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=URLError("connection refused"),
        ):
            with self.assertRaises(NavigationError) as caught:
                self.client.health()
        self.assertEqual(caught.exception.status_code, 0)


if __name__ == "__main__":
    unittest.main()
