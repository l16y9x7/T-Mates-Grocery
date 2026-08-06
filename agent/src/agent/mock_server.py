from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


PORTS = {
    "navigation": 8101,
    "perception": 8102,
    "pose": 8103,
    "manipulation": 8104,
}
SERVICE_NAMES = {
    "navigation": "导航模块",
    "perception": "场景理解模块",
    "pose": "位姿控制模块",
    "manipulation": "抓放模块",
}
ACTION_NAMES = {
    "receipt": "小票识别",
    "inspection": "货架巡检",
    "navigation": "导航",
    "pose": "位姿准备",
    "pick": "抓取",
    "place": "放置",
}
LOG_LOCK = threading.Lock()
PHYSICAL_ENDPOINTS = {
    "/navigation/navigate": "navigation",
    "/pose/prepare": "pose",
    "/manipulation/pick": "pick",
    "/manipulation/place": "place",
}
SCENARIOS = (
    "success",
    "slow",
    "health-error",
    "navigation-failure",
    "late-findings",
    "timeout-recovery",
    "timeout-unknown",
)


@dataclass
class ActionRecord:
    endpoint: str
    event: threading.Event = field(default_factory=threading.Event)
    status_code: int = 200
    response: dict[str, str] = field(default_factory=lambda: {"status": "SUCCEEDED"})


def scenario_config(name: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "health": {
            "navigation": "READY",
            "perception": "READY",
            "pose": "READY",
            "manipulation": "READY",
        },
        "receipt": ["H1_F_L1_C01", "H1_F_L1_C02"],
        "inspections": {
            "SHORTAGE": [["H1_F_L2_C01", "H2_B_L3_C02"]],
            "MISPLACED": [["H1_F_L1_C01", "H1_F_L1_C02"]],
        },
        "delays": {
            "receipt": [0.05],
            "inspection": [0.05],
            "navigation": [0.05],
            "pose": [0.05],
            "pick": [0.05],
            "place": [0.05],
        },
        "failures": {},
    }

    if name == "slow":
        config["delays"].update(
            {
                "receipt": [1.0],
                "inspection": [1.0],
                "navigation": [2.0],
                "pose": [1.0],
                "pick": [2.0],
                "place": [2.0],
            }
        )
    elif name == "health-error":
        config["health"]["pose"] = "ERROR"
    elif name == "navigation-failure":
        config["failures"]["navigation"] = [500]
    elif name == "late-findings":
        config["inspections"]["SHORTAGE"] = [
            *[[] for _ in range(8)],
            ["H1_F_L2_C01", "H2_B_L3_C02"],
        ]
    elif name == "timeout-recovery":
        config["delays"]["navigation"] = [4.0, 0.05]
    elif name == "timeout-unknown":
        config["delays"]["navigation"] = [7.0, 0.05]
    elif name != "success":
        raise ValueError(f"unknown scenario: {name}")
    return config


class MockState:
    def __init__(self, scenario: str) -> None:
        config = scenario_config(scenario)
        self.scenario = scenario
        self.health: dict[str, str] = config["health"]
        self.receipt: list[str] = config["receipt"]
        self.inspections = {
            task_type: deque(results) for task_type, results in config["inspections"].items()
        }
        self.delays = {
            action: deque(values) for action, values in config["delays"].items()
        }
        self.failures = {
            action: deque(values) for action, values in config["failures"].items()
        }
        self.lock = threading.Lock()
        self.actions: dict[str, ActionRecord] = {}
        self.request_counts: Counter[str] = Counter()
        self.actual_action_counts: Counter[str] = Counter()

    def record_request(self, service: str, method: str, path: str) -> None:
        with self.lock:
            self.request_counts[f"{service}:{method}:{path}"] += 1

    def next_inspection(self, task_type: str) -> list[str]:
        with self.lock:
            results = self.inspections.get(task_type)
            if not results:
                return []
            if len(results) == 1:
                return list(results[0])
            return list(results.popleft())

    def delay(self, action: str) -> float:
        with self.lock:
            values = self.delays[action]
            if len(values) == 1:
                return values[0]
            return values.popleft()

    def failure(self, action: str) -> int | None:
        with self.lock:
            values = self.failures.get(action)
            return values.popleft() if values else None

    def execute_action(
        self,
        endpoint: str,
        key: str,
        on_start: Callable[[bool, float], None],
    ) -> tuple[int, dict[str, str], bool]:
        with self.lock:
            record = self.actions.get(key)
            if record is not None and record.endpoint != endpoint:
                return 409, {"error_code": "IDEMPOTENCY_KEY_CONFLICT"}, False
            is_new = record is None
            if is_new:
                record = ActionRecord(endpoint=endpoint)
                self.actions[key] = record
                self.actual_action_counts[endpoint] += 1

        assert record is not None
        if is_new:
            delay = self.delay(endpoint)
            on_start(True, delay)
            if delay:
                time.sleep(delay)
            status_code = self.failure(endpoint)
            if status_code is not None:
                record.status_code = status_code
                record.response = {"error_code": "EXECUTION_FAILED"}
            record.event.set()
        else:
            on_start(False, 0.0)
            record.event.wait()
        return record.status_code, record.response, is_new

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "scenario": self.scenario,
                "request_counts": dict(self.request_counts),
                "actual_action_counts": dict(self.actual_action_counts),
                "idempotency_keys": len(self.actions),
            }


class MockHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], service: str, state: MockState) -> None:
        self.service = service
        self.state = state
        super().__init__(address, MockRequestHandler)


class MockRequestHandler(BaseHTTPRequestHandler):
    server: MockHTTPServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        self.server.state.record_request(self.server.service, "GET", path)
        self._log_request("GET", path, None)
        if path == f"/{self.server.service}/health":
            self._send(200, {"status": self.server.state.health[self.server.service]})
            return
        if path == "/mock/state":
            self._send(200, self.server.state.snapshot())
            return
        self._send(404, {"error_code": "UNKNOWN_ENDPOINT"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        self.server.state.record_request(self.server.service, "POST", path)
        try:
            payload = self._read_json()
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._log_request("POST", path, {"错误": "请求体不是合法 JSON"})
            self._send(400, {"error_code": "INVALID_JSON"})
            return
        self._log_request("POST", path, payload)

        if path == "/receipt/parse" and self.server.service == "perception":
            delay = self.server.state.delay("receipt")
            self._log("模拟处理", 动作=ACTION_NAMES["receipt"], 延迟秒=delay)
            time.sleep(delay)
            self._send(200, self.server.state.receipt)
            return
        if path == "/areas/inspect" and self.server.service == "perception":
            delay = self.server.state.delay("inspection")
            self._log("模拟处理", 动作=ACTION_NAMES["inspection"], 延迟秒=delay)
            time.sleep(delay)
            task_type = payload.get("task_type", "")
            self._send(200, {"findings": self.server.state.next_inspection(task_type)})
            return

        endpoint = PHYSICAL_ENDPOINTS.get(path)
        if endpoint is None or not self._endpoint_belongs_to_service(endpoint):
            self._send(404, {"error_code": "UNKNOWN_ENDPOINT"})
            return
        key = self.headers.get("Idempotency-Key")
        if not key:
            self._send(400, {"error_code": "MISSING_IDEMPOTENCY_KEY"})
            return

        status_code, response, _ = self.server.state.execute_action(
            endpoint,
            key,
            lambda is_new, delay: self._log_action(endpoint, key, is_new, delay),
        )
        self._send(status_code, response)

    def _endpoint_belongs_to_service(self, endpoint: str) -> bool:
        return (
            endpoint == self.server.service
            or endpoint in {"pick", "place"}
            and self.server.service == "manipulation"
        )

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        return payload if isinstance(payload, dict) else {}

    def _send(self, status_code: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._log("发送响应", 状态码=status_code, 响应体=payload)
        try:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self._log("发送中断", 原因="客户端已断开连接")

    def _log_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> None:
        details: dict[str, Any] = {"方法": method, "路径": path}
        if payload is not None:
            details["请求体"] = payload
        key = self.headers.get("Idempotency-Key")
        if key:
            details["幂等键"] = key
        self._log("收到请求", **details)

    def _log_action(self, endpoint: str, key: str, is_new: bool, delay: float) -> None:
        if is_new:
            self._log(
                "开始模拟动作",
                动作=ACTION_NAMES[endpoint],
                处理方式="首次执行",
                延迟秒=delay,
                幂等键=key,
            )
            return
        self._log(
            "等待原动作",
            动作=ACTION_NAMES[endpoint],
            处理方式="重复请求，不重复执行",
            幂等键=key,
        )

    def _log(self, event: str, **details: Any) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        module_name = SERVICE_NAMES[self.server.service]
        detail_text = " | ".join(
            f"{key}={self._format_value(value)}" for key, value in details.items()
        )
        line = f"{timestamp} | Mock {module_name} | {event}"
        if detail_text:
            line = f"{line} | {detail_text}"
        with LOG_LOCK:
            print(line, flush=True)

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return str(value)

    def log_message(self, format_string: str, *args: object) -> None:
        return


def serve(host: str, scenario: str) -> None:
    state = MockState(scenario)
    servers = [
        MockHTTPServer((host, port), service, state) for service, port in PORTS.items()
    ]
    threads = [
        threading.Thread(target=server.serve_forever, name=server.service, daemon=True)
        for server in servers
    ]
    for thread in threads:
        thread.start()

    print(f"Mock 场景：{scenario}", flush=True)
    for service, port in PORTS.items():
        print(f"  {SERVICE_NAMES[service]}：http://{host}:{port}", flush=True)
    print("按 Ctrl+C 停止服务。", flush=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("正在停止 Mock 服务...", flush=True)
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run four standalone capability mock services")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--scenario", choices=SCENARIOS, default="success")
    args = parser.parse_args()
    serve(args.host, args.scenario)


if __name__ == "__main__":
    cli()
