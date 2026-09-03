#!/usr/bin/env python3
"""Small same-origin gateway for the external task API demo."""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
ROBOT_TASK_URL = os.environ.get("ROBOT_TASK_URL", "http://127.0.0.1:8108").rstrip("/")
HOST = os.environ.get("DEMO_HOST", "127.0.0.1")
PORT = int(os.environ.get("DEMO_PORT", "8765"))
CALLBACK_PATH = "/api/callback"
CALLBACK_URL = os.environ.get("DEMO_CALLBACK_URL", f"http://127.0.0.1:{PORT}{CALLBACK_PATH}")
events: dict[str, list[dict]] = {}
events_lock = threading.Lock()


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "ExternalTaskDemo/1.0"

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {format % args}")

    def send_json(self, status: int, payload: object) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, filename: str) -> None:
        path = ROOT / filename
        if not path.is_file():
            self.send_error(404)
            return
        content_type = "text/html; charset=utf-8" if filename.endswith(".html") else "text/css; charset=utf-8" if filename.endswith(".css") else "application/javascript; charset=utf-8"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.serve_static("index.html")
            return
        if parsed.path == "/static/styles.css":
            self.serve_static("styles.css")
            return
        if parsed.path == "/static/app.js":
            self.serve_static("app.js")
            return
        if parsed.path == "/api/config":
            self.send_json(200, {"robot_task_url": ROBOT_TASK_URL, "callback_url": CALLBACK_URL})
            return
        if parsed.path == "/api/events":
            task_run_id = parse_qs(parsed.query).get("task_run_id", [""])[0]
            with events_lock:
                payload = list(events.get(task_run_id, []))
            self.send_json(200, {"events": payload})
            return
        if parsed.path.startswith("/api/external/"):
            self.proxy("GET", parsed.path + (f"?{parsed.query}" if parsed.query else ""), None)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_json(400, {"error_code": "INVALID_JSON", "message": "请求体不是有效 JSON"})
            return
        if parsed.path == CALLBACK_PATH:
            task_run_id = body.get("task_run_id")
            if not task_run_id:
                self.send_json(400, {"error_code": "INVALID_CALLBACK", "message": "缺少 task_run_id"})
                return
            with events_lock:
                events.setdefault(task_run_id, []).append(body)
            self.send_json(200, {"accepted": True})
            return
        if parsed.path.startswith("/api/external/"):
            self.proxy("POST", parsed.path, raw)
            return
        self.send_error(404)

    def proxy(self, method: str, path: str, body: bytes | None) -> None:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        for name in ("Authorization", "Idempotency-Key", "X-Request-Id"):
            value = self.headers.get(name)
            if value:
                headers[name] = value
        request = Request(f"{ROBOT_TASK_URL}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=12) as response:
                response_body = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
        except HTTPError as error:
            response_body = error.read()
            self.send_response(error.code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        except (URLError, TimeoutError, OSError) as error:
            self.send_json(502, {"error_code": "ROBOT_SERVICE_UNREACHABLE", "message": f"无法连接任务服务：{error}"})


if __name__ == "__main__":
    print(f"External task demo: http://{HOST}:{PORT}")
    print(f"Robot task service: {ROBOT_TASK_URL}")
    ThreadingHTTPServer((HOST, PORT), DemoHandler).serve_forever()
