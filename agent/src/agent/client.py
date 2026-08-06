"""机器人能力服务的异步 HTTP 客户端。

编排层只调用本类提供的语义化方法，不直接拼接 URL 或处理 HTTP 错误。本模块统一
负责响应校验、超时、一次重试、物理动作幂等键，以及将底层异常转换成 ``AgentError``。
"""

from __future__ import annotations

import asyncio
import logging
from time import monotonic
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from agent.models import AgentError, AgentSettings, Hand, TaskType


LOGGER = logging.getLogger(__name__)
SERVICE_NAMES = {
    "navigation": "导航",
    "perception": "场景感知",
    "pose": "位姿控制",
    "manipulation": "抓放操作",
}
ACTION_NAMES = {
    "/navigation/navigate": "导航",
    "/pose/prepare": "位姿准备",
    "/manipulation/pick": "抓取",
    "/manipulation/place": "放置",
}
HEALTH_PATHS = {
    "navigation": "/navigation/health",
    "perception": "/perception/health",
    "pose": "/pose/health",
    "manipulation": "/manipulation/health",
}


class HealthResponse(BaseModel):
    """能力模块健康检查的严格响应结构。"""

    model_config = ConfigDict(extra="forbid")
    status: Literal["STARTING", "READY", "ERROR"]


class ActionResponse(BaseModel):
    """导航、姿态、抓取和放置动作成功时的严格响应结构。"""

    model_config = ConfigDict(extra="forbid")
    status: Literal["SUCCEEDED"]


class InspectionResponse(BaseModel):
    """货架巡检返回的异常货位列表。"""

    model_config = ConfigDict(extra="forbid")
    findings: list[str]


# 小票识别接口直接返回 JSON 字符串数组，不需要为此额外声明 BaseModel。
STRING_LIST = TypeAdapter(list[str])


def action_key(task_run_id: str, action_id: str) -> str:
    """生成物理动作幂等键，确保同一任务步骤重试时不会被机器人重复执行。"""

    return f"{task_run_id}:{action_id}"


class CapabilityClient:
    """封装导航、感知、姿态和操作四类能力服务。"""

    def __init__(
        self,
        settings: AgentSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """创建客户端；测试可注入 ``MockTransport``，生产环境使用默认网络传输。"""

        self.settings = settings
        self._client = httpx.AsyncClient(transport=transport)

    async def __aenter__(self) -> CapabilityClient:
        """支持 ``async with`` 管理底层连接池的生命周期。"""

        return self

    async def __aexit__(self, *_: object) -> None:
        """退出异步上下文时关闭客户端，异常也不会导致连接资源泄漏。"""

        await self.aclose()

    async def aclose(self) -> None:
        """释放 HTTP 连接池。"""

        await self._client.aclose()

    async def check_all_health(self) -> None:
        """并发检查所有能力模块；任一模块未就绪都禁止启动任务。"""

        services = ("navigation", "perception", "pose", "manipulation")
        # 四项检查彼此独立，并发执行可缩短启动等待时间。
        results = await asyncio.gather(*(self._check_health(service) for service in services))
        not_ready = [service for service, status in zip(services, results) if status != "READY"]
        if not_ready:
            raise AgentError(
                "CAPABILITY_NOT_READY",
                f"capability modules are not ready: {', '.join(not_ready)}",
            )

    async def navigate(self, target_id: str, task_run_id: str, action_id: str) -> None:
        """命令导航模块移动到命名点位。"""

        await self._physical_action(
            "navigation",
            "/navigation/navigate",
            {"target_id": target_id},
            task_run_id,
            action_id,
            self.settings.timeouts.navigation_seconds,
        )

    async def prepare_pose(
        self,
        pose_type: str,
        shelf_level: str | None,
        task_run_id: str,
        action_id: str,
    ) -> None:
        """命令姿态模块切换到指定动作姿态；货架动作可额外指定层号。"""

        payload: dict[str, str] = {"pose_type": pose_type}
        if shelf_level is not None:
            payload["shelf_level"] = shelf_level
        await self._physical_action(
            "pose",
            "/pose/prepare",
            payload,
            task_run_id,
            action_id,
            self.settings.timeouts.pose_seconds,
        )

    async def parse_receipt(self) -> list[str]:
        """识别小票上的两个目标货位，并校验响应的基本结构与数量。"""

        response = await self._request(
            "perception",
            "POST",
            "/receipt/parse",
            timeout_seconds=self.settings.timeouts.receipt_seconds,
        )
        try:
            slots = STRING_LIST.validate_python(response.json())
        except (ValueError, ValidationError) as exc:
            raise AgentError("INVALID_RESPONSE", "receipt response must be a JSON string array") from exc
        if len(slots) != 2:
            raise AgentError("INVALID_RESPONSE", "receipt response must contain exactly two slots")
        return slots

    async def inspect(self, task_type: TaskType) -> list[str]:
        """请求感知模块检查当前货架面，返回缺货或错放货位。"""

        response = await self._request(
            "perception",
            "POST",
            "/areas/inspect",
            json={"task_type": task_type.value},
            timeout_seconds=self.settings.timeouts.inspection_seconds,
        )
        try:
            return InspectionResponse.model_validate(response.json()).findings
        except (ValueError, ValidationError) as exc:
            raise AgentError("INVALID_RESPONSE", "inspection response is invalid") from exc

    async def pick(
        self,
        task_type: TaskType,
        product_name: str,
        hand: Hand,
        task_run_id: str,
        action_id: str,
    ) -> None:
        """使用指定手抓取商品。"""

        await self._physical_action(
            "manipulation",
            "/manipulation/pick",
            {"task_type": task_type.value, "product_name": product_name, "hand": hand},
            task_run_id,
            action_id,
            self.settings.timeouts.pick_seconds,
        )

    async def place(
        self,
        task_type: TaskType,
        product_name: str,
        hand: Hand,
        task_run_id: str,
        action_id: str,
    ) -> None:
        """使用指定手放置商品。"""

        await self._physical_action(
            "manipulation",
            "/manipulation/place",
            {"task_type": task_type.value, "product_name": product_name, "hand": hand},
            task_run_id,
            action_id,
            self.settings.timeouts.place_seconds,
        )

    async def _check_health(self, service: str) -> str:
        """检查单个服务，并拒绝字段缺失、值非法或包含多余字段的响应。"""

        response = await self._request(
            service,
            "GET",
            HEALTH_PATHS[service],
            timeout_seconds=self.settings.timeouts.health_seconds,
        )
        try:
            status = HealthResponse.model_validate(response.json()).status
        except (ValueError, ValidationError) as exc:
            raise AgentError("INVALID_RESPONSE", f"invalid health response from {service}") from exc
        LOGGER.info(
            "健康检查结果 | service=%s(%s) | status=%s",
            SERVICE_NAMES[service],
            service,
            status,
        )
        return status

    async def _physical_action(
        self,
        service: str,
        path: str,
        payload: dict[str, str],
        task_run_id: str,
        action_id: str,
        timeout_seconds: float,
    ) -> None:
        """执行一个具有副作用的物理动作并严格校验成功响应。

        幂等键在重试前保持不变。能力服务应缓存该键的执行结果，从而做到 HTTP
        请求可以重发，但真实机器人动作只执行一次。
        """

        action_name = ACTION_NAMES[path]
        LOGGER.info(
            "能力动作开始 | action=%s | service=%s | action_id=%s | payload=%s",
            action_name,
            SERVICE_NAMES[service],
            action_id,
            payload,
        )
        try:
            response = await self._request(
                service,
                "POST",
                path,
                json=payload,
                headers={"Idempotency-Key": action_key(task_run_id, action_id)},
                timeout_seconds=timeout_seconds,
                result_unknown_on_exhaustion=True,
            )
            ActionResponse.model_validate(response.json())
        except AgentError as exc:
            LOGGER.error(
                "能力动作失败 | action=%s | action_id=%s | error_code=%s | result_unknown=%s",
                action_name,
                action_id,
                exc.code,
                exc.result_unknown,
            )
            raise
        except (ValueError, ValidationError) as exc:
            error = AgentError("INVALID_RESPONSE", f"invalid action response from {service}")
            LOGGER.error(
                "能力动作失败 | action=%s | action_id=%s | error_code=%s",
                action_name,
                action_id,
                error.code,
            )
            raise error from exc
        LOGGER.info("能力动作成功 | action=%s | action_id=%s", action_name, action_id)

    async def _request(
        self,
        service: str,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: float,
        result_unknown_on_exhaustion: bool = False,
    ) -> httpx.Response:
        """发送请求，处理超时/网络错误、一次重试和非 2xx 响应。

        ``result_unknown_on_exhaustion`` 只对有物理副作用的调用启用。两次网络失败
        后，普通查询可报告网络错误；物理动作则必须报告结果未知，因为请求可能已经
        到达服务端并开始执行，只是响应未能返回。
        """

        url = f"{getattr(self.settings.services, service).rstrip('/')}{path}"
        # httpx 负责分阶段超时；外层 asyncio.timeout 限制整个单次请求的总时长。
        timeout = httpx.Timeout(
            connect=self.settings.timeouts.connect_seconds,
            read=timeout_seconds,
            write=10.0,
            pool=5.0,
        )

        # 最多尝试两次。这里只重试传输层故障，不重试明确返回的业务/HTTP 失败。
        for attempt in range(2):
            attempt_number = attempt + 1
            started_at = monotonic()
            LOGGER.debug(
                "HTTP 请求开始 | service=%s | method=%s | path=%s | attempt=%d/2 | timeout=%.2fs",
                SERVICE_NAMES[service],
                method,
                path,
                attempt_number,
                timeout_seconds,
            )
            try:
                async with asyncio.timeout(timeout_seconds):
                    response = await self._client.request(
                        method,
                        url,
                        json=json,
                        headers=headers,
                        timeout=timeout,
                    )
            except (TimeoutError, httpx.TimeoutException, httpx.NetworkError) as exc:
                elapsed = monotonic() - started_at
                if attempt == 0:
                    # 短暂退避，且第二次请求继续携带完全相同的幂等键。
                    LOGGER.warning(
                        "HTTP 请求异常，准备重试 | service=%s | path=%s | attempt=%d/2 "
                        "| elapsed=%.3fs | error=%s",
                        SERVICE_NAMES[service],
                        path,
                        attempt_number,
                        elapsed,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(0.05)
                    continue
                code = "ACTION_RESULT_UNKNOWN" if result_unknown_on_exhaustion else "NETWORK_ERROR"
                LOGGER.error(
                    "HTTP 请求连续失败 | service=%s | path=%s | attempt=%d/2 "
                    "| elapsed=%.3fs | error_code=%s | error=%s",
                    SERVICE_NAMES[service],
                    path,
                    attempt_number,
                    elapsed,
                    code,
                    type(exc).__name__,
                )
                raise AgentError(
                    code,
                    f"{service} request result could not be determined",
                    result_unknown=result_unknown_on_exhaustion,
                ) from exc

            if not response.is_success:
                error_message = self._error_message(service, response)
                LOGGER.error(
                    "HTTP 响应失败 | service=%s | path=%s | status_code=%d | elapsed=%.3fs",
                    SERVICE_NAMES[service],
                    path,
                    response.status_code,
                    monotonic() - started_at,
                )
                raise AgentError(
                    "EXECUTION_FAILED",
                    error_message,
                )
            LOGGER.debug(
                "HTTP 请求成功 | service=%s | path=%s | status_code=%d | attempt=%d/2 "
                "| elapsed=%.3fs",
                SERVICE_NAMES[service],
                path,
                response.status_code,
                attempt_number,
                monotonic() - started_at,
            )
            return response

        raise AssertionError("unreachable retry state")

    @staticmethod
    def _error_message(service: str, response: httpx.Response) -> str:
        """从失败响应提取能力模块错误码；非 JSON 响应使用通用错误码。"""

        try:
            error_code = response.json().get("error_code", "EXECUTION_FAILED")
        except ValueError:
            error_code = "EXECUTION_FAILED"
        return f"{service} returned HTTP {response.status_code}: {error_code}"
