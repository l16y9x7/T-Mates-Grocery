"""四个外部能力模块的内存 Mock 实现。

MockTransport 让测试完整经过 ``CapabilityClient`` 的 HTTP 序列化、超时与重试逻辑，
但不需要启动真实服务。这里还模拟服务端幂等去重，以验证“请求可以重试，机器人
动作只能执行一次”这一关键安全约束。
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class RecordedRequest:
    """一条已接收 HTTP 请求的规范化记录，供测试检查顺序、参数和请求头。"""

    service: str
    method: str
    path: str
    payload: dict[str, Any] | None
    headers: dict[str, str]


class MockServices:
    """可编程的导航、感知、位姿和抓放服务集合。"""

    # Host 与能力模块的映射和生产配置结构一致，用于模拟四个独立服务。
    HOSTS = {
        "navigation.local": "navigation",
        "perception.local": "perception",
        "pose.local": "pose",
        "manipulation.local": "manipulation",
    }
    PORTS = {
        8101: "navigation",
        8102: "perception",
        8103: "pose",
        8104: "manipulation",
    }

    def __init__(self) -> None:
        """初始化默认成功场景及用于故障注入、调用追踪的内存状态。"""

        self.transport = httpx.MockTransport(self._handle)
        self.health = {service: "READY" for service in self.HOSTS.values()}
        # 默认小票对应两个不同商品；巡检默认没有发现项，可由具体测试覆盖。
        self.receipt_result = ["H1_F_L1_C01", "H1_F_L1_C02"]
        self.inspection_results: list[list[str]] = []
        self.requests: list[RecordedRequest] = []
        # requests 统计 HTTP 尝试；actual_action_counts 统计去重后的真实物理动作。
        self.actual_action_counts: Counter[str] = Counter()
        # 相同幂等键共享同一个 Task，模拟能力服务等待并返回原动作结果。
        self._action_tasks: dict[str, asyncio.Task[dict[str, str]]] = {}
        self._delays: dict[str, float] = defaultdict(float)
        self._timeouts: Counter[str] = Counter()
        self._failures: dict[str, list[int]] = defaultdict(list)

    def set_health(self, service: str, status: str) -> None:
        """设置单个能力模块的健康状态。"""

        self.health[service] = status

    def set_delay(self, action: str, seconds: float) -> None:
        """为某类物理动作注入执行延迟，用于验证长动作超时边界。"""

        self._delays[action] = seconds

    def timeout_next(self, action: str, *, times: int = 1) -> None:
        """让某类动作接下来的若干 HTTP 请求抛出读取超时。"""

        self._timeouts[action] += times

    def fail_next(self, action: str, *, status_code: int = 500) -> None:
        """让某类动作下一次请求返回指定的非成功 HTTP 状态。"""

        self._failures[action].append(status_code)

    def calls(self, *, path: str | None = None, service: str | None = None) -> list[RecordedRequest]:
        """按路径和/或服务筛选已记录请求；参数为空时返回全部请求。"""

        return [
            request
            for request in self.requests
            if (path is None or request.path == path)
            and (service is None or request.service == service)
        ]

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        """根据 host 和 path 分派请求，并应用测试预设的响应或故障。"""

        service = self.HOSTS.get(request.url.host) or self.PORTS.get(request.url.port)
        if service is None:
            return httpx.Response(404, json={"error_code": "UNKNOWN_SERVICE"})

        # 在响应前先记录请求，因此即使随后超时，测试也能看到真实发送次数。
        payload = json.loads(request.content) if request.content else None
        self.requests.append(
            RecordedRequest(
                service=service,
                method=request.method,
                path=request.url.path,
                payload=payload,
                headers={key.lower(): value for key, value in request.headers.items()},
            )
        )

        # 查询类接口没有物理副作用，直接从当前脚本化状态生成响应。
        if request.url.path == f"/{service}/health":
            return httpx.Response(200, json={"status": self.health[service]})
        if request.url.path == "/receipt/parse":
            return httpx.Response(200, json=self.receipt_result)
        if request.url.path == "/areas/inspect":
            findings = self.inspection_results.pop(0) if self.inspection_results else []
            return httpx.Response(200, json={"findings": findings})

        # 其余已知端点都是物理动作，必须走幂等键与执行任务去重逻辑。
        action = self._action_name(request.url.path)
        if action is None:
            return httpx.Response(404, json={"error_code": "UNKNOWN_ENDPOINT"})
        if self._failures[action]:
            status_code = self._failures[action].pop(0)
            return httpx.Response(status_code, json={"error_code": "EXECUTION_FAILED"})

        key = request.headers.get("Idempotency-Key")
        if not key:
            return httpx.Response(400, json={"error_code": "MISSING_IDEMPOTENCY_KEY"})
        if key not in self._action_tasks:
            # 只有首次看到幂等键才启动真实动作；重试只复用已有 Task。
            self.actual_action_counts[key] += 1
            self._action_tasks[key] = asyncio.create_task(self._complete_action(action))

        if self._timeouts[action] > 0:
            # 超时只丢失本次 HTTP 响应，不取消已经启动的机器人动作。
            self._timeouts[action] -= 1
            raise httpx.ReadTimeout("scripted read timeout", request=request)

        # shield 模拟服务端动作不受客户端取消或超时影响。
        result = await asyncio.shield(self._action_tasks[key])
        return httpx.Response(200, json=result)

    async def _complete_action(self, action: str) -> dict[str, str]:
        """模拟一个可延迟、最终成功的真实物理动作。"""

        delay = self._delays[action]
        if delay:
            await asyncio.sleep(delay)
        return {"status": "SUCCEEDED"}

    @staticmethod
    def _action_name(path: str) -> str | None:
        """将 HTTP 端点映射为故障注入使用的简短动作名。"""

        return {
            "/navigation/navigate": "navigation",
            "/pose/prepare": "pose",
            "/manipulation/pick": "pick",
            "/manipulation/place": "place",
        }.get(path)
