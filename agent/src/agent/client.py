"""机器人能力服务的异步 HTTP 客户端。

编排层只调用本类提供的语义化方法，不直接拼接 URL 或处理 HTTP 错误。本模块统一
负责响应校验、超时、一次重试、物理动作幂等键，以及将底层异常转换成 ``AgentError``。
"""

from __future__ import annotations

import asyncio
import json as jsonlib
import logging
from time import monotonic
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from agent.models import AgentError, AgentSettings, Hand, TaskType


LOGGER = logging.getLogger(__name__)
SERVICE_NAMES = {
    "navigation": "导航",
    "perception": "视觉理解",
    "pose": "躯干控制",
    "manipulation": "抓放操作",
    "pick_place": "取放编排",
    "sku": "商品库",
}
HEALTH_PATHS = {
    "navigation": "/navigation/health",
    "perception": "/perception/health",
    "pose": "/pose/health",
    "pick_place": "/health",
    "sku": "/sku/health",
}
ENDPOINT_PURPOSES = {
    "/navigation/health": "检查导航服务是否就绪",
    "/perception/health": "检查视觉理解服务是否就绪",
    "/pose/health": "检查躯干控制服务是否就绪",
    "/sku/health": "检查商品库是否就绪",
    "/health": "检查取放编排服务是否就绪",
    "/navigation/navigate": "导航到目标点",
    "/pose/prepare": "准备机器人躯干姿态",
    "/perception/parse": "识别小票中的商品货位",
    "/perception/inspect": "识别货架上的缺货或乱放位置",
    "/sku/search_by_name": "根据商品名查询标准货位",
    "/sku/search_by_location": "根据货位查询商品信息",
    "/sku/name": "兼容旧版接口：根据货位查询商品名",
    "/sku/locations": "兼容旧版接口：根据商品名查询标准货位",
    "/sku/images": "查询商品图片",
    "/pick": "执行完整抓取流程",
    "/place": "执行完整放置流程",
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


class SkuNameResponse(BaseModel):
    """商品库按货位返回的标准商品名。"""

    model_config = ConfigDict(extra="forbid")
    location: str = Field(min_length=1)
    name: str = Field(min_length=1)


class SkuSearchResponse(BaseModel):
    """商品库按商品名查询的实际响应。"""

    model_config = ConfigDict(extra="forbid")
    sku_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    images: list[str]
    locations: list[str]


class SkuLocationsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    locations: list[str]


class SkuImagesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    images: list[str]


# 小票识别接口直接返回 JSON 字符串数组，不需要为此额外声明 BaseModel。
STRING_LIST = TypeAdapter(list[str])


def action_key(task_run_id: str, action_id: str) -> str:
    """生成物理动作幂等键，确保同一任务步骤重试时不会被机器人重复执行。"""

    return f"{task_run_id}:{action_id}"


class CapabilityClient:
    """封装导航、感知、姿态和 8086 取放编排服务。"""

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
        """并发检查主流程依赖；任一服务未就绪都禁止启动任务。"""

        services = ("navigation", "perception", "pose", "sku", "pick_place")
        # 各项检查彼此独立，并发执行可缩短启动等待时间。
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
        """识别小票上的两个商品名，并兼容旧版货位数组响应。"""

        response = await self._request(
            "perception",
            "POST",
            "/perception/parse",
            timeout_seconds=self.settings.timeouts.receipt_seconds,
        )
        try:
            payload = response.json()
            if isinstance(payload, dict) and "product_names" in payload:
                names = STRING_LIST.validate_python(payload["product_names"])
                if len(names) != 2 or len(set(names)) != 2 or any(not name.strip() for name in names):
                    raise ValueError("receipt must contain two different non-empty product names")
                return names
            # 本地旧 Mock 曾直接返回货位数组，保留读取能力以便联调期间平滑升级。
            slots = STRING_LIST.validate_python(payload)
        except (ValueError, ValidationError) as exc:
            raise AgentError(
                "INVALID_RESPONSE",
                "receipt response must contain product_names or a legacy slot array",
            ) from exc
        if len(slots) != 2:
            raise AgentError("INVALID_RESPONSE", "receipt response must contain exactly two slots")
        return slots

    async def sku_search_by_name(self, name: str) -> SkuSearchResponse:
        """按商品名查询 SKU；优先使用商品库当前的 ``/sku/search_by_name``。"""

        try:
            response = await self._request(
                "sku",
                "GET",
                "/sku/search_by_name",
                json={"name": name},
                timeout_seconds=self.settings.timeouts.sku_seconds,
            )
            result = SkuSearchResponse.model_validate(response.json())
            if result.name != name:
                raise ValueError("sku response name does not match request")
            return result
        except AgentError as exc:
            # 兼容仍提供旧 /sku/name 的现场网关和 Mock；正式环境应使用新接口。
            if exc.code != "EXECUTION_FAILED":
                raise
            response = await self._request(
                "sku",
                "GET",
                "/sku/name",
                json={"location": name},
                timeout_seconds=self.settings.timeouts.sku_seconds,
            )
            try:
                legacy = SkuNameResponse.model_validate(response.json())
                return SkuSearchResponse(
                    sku_id=legacy.location,
                    name=legacy.name,
                    images=[],
                    locations=[legacy.location],
                )
            except (ValueError, ValidationError) as legacy_exc:
                raise exc from legacy_exc
        except (ValueError, ValidationError) as exc:
            raise AgentError("INVALID_RESPONSE", "sku search response is invalid") from exc

    async def inspect(self, task_type: TaskType) -> list[str]:
        """请求感知模块检查当前货架面，返回缺货或错放货位。"""

        response = await self._request(
            "perception",
            "POST",
            "/perception/inspect",
            json={"task_type": task_type.value},
            timeout_seconds=self.settings.timeouts.inspection_seconds,
        )
        try:
            return InspectionResponse.model_validate(response.json()).findings
        except (ValueError, ValidationError) as exc:
            raise AgentError("INVALID_RESPONSE", "inspection response is invalid") from exc

    async def sku_name(self, location: str) -> str:
        """通过商品库查询货位对应的标准商品名。"""

        response = await self._request(
            "sku",
            "GET",
            "/sku/name",
            json={"location": location},
            timeout_seconds=self.settings.timeouts.sku_seconds,
        )
        try:
            result = SkuNameResponse.model_validate(response.json())
            if result.location.upper() != location.upper():
                raise ValueError("sku response location does not match request")
            return result.name
        except (ValueError, ValidationError) as exc:
            raise AgentError("INVALID_RESPONSE", "sku name response is invalid") from exc

    async def sku_locations(self, name: str) -> list[str]:
        """通过商品库查询商品的标准货位。"""

        response = await self._request(
            "sku",
            "GET",
            "/sku/locations",
            json={"name": name},
            timeout_seconds=self.settings.timeouts.sku_seconds,
        )
        try:
            return SkuLocationsResponse.model_validate(response.json()).locations
        except (ValueError, ValidationError) as exc:
            raise AgentError("INVALID_RESPONSE", "sku locations response is invalid") from exc

    async def sku_images(self, name: str) -> list[str]:
        """通过商品库查询商品图片引用。"""

        response = await self._request(
            "sku",
            "GET",
            "/sku/images",
            json={"name": name},
            timeout_seconds=self.settings.timeouts.sku_seconds,
        )
        try:
            return SkuImagesResponse.model_validate(response.json()).images
        except (ValueError, ValidationError) as exc:
            raise AgentError("INVALID_RESPONSE", "sku images response is invalid") from exc

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
            "pick_place",
            "/pick",
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
            "pick_place",
            "/place",
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
        LOGGER.debug(
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
        payload: dict[str, Any],
        task_run_id: str,
        action_id: str,
        timeout_seconds: float,
    ) -> None:
        """执行一个具有副作用的物理动作并严格校验成功响应。

        幂等键在重试前保持不变。能力服务应缓存该键的执行结果，从而做到 HTTP
        请求可以重发，但真实机器人动作只执行一次。
        """

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
        except AgentError:
            raise
        except (ValueError, ValidationError) as exc:
            raise AgentError(
                "INVALID_RESPONSE",
                f"{SERVICE_NAMES[service]}返回的动作结果格式无效",
            ) from exc

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
            started_at = monotonic()
            retry_text = " | 重试=第2次" if attempt else ""
            LOGGER.info(
                "调用接口 | 用途=%s | 服务=%s | 接口=%s %s | 入参=%s%s",
                ENDPOINT_PURPOSES.get(path, f"调用{SERVICE_NAMES[service]}服务"),
                SERVICE_NAMES[service],
                method,
                path,
                self._format_log_value(json),
                retry_text,
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
                        "接口调用异常，准备重试 | 服务=%s | 接口=%s %s | 耗时=%.2f秒 | 异常=%s",
                        SERVICE_NAMES[service],
                        method,
                        path,
                        elapsed,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(0.05)
                    continue
                code = "ACTION_RESULT_UNKNOWN" if result_unknown_on_exhaustion else "NETWORK_ERROR"
                LOGGER.error(
                    "接口连续两次调用异常 | 服务=%s | 接口=%s %s | 耗时=%.2f秒 "
                    "| 错误码=%s | 异常=%s",
                    SERVICE_NAMES[service],
                    method,
                    path,
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
                    "接口返回失败 | 服务=%s | 接口=%s %s | HTTP状态=%d | 返回=%s | 耗时=%.2f秒",
                    SERVICE_NAMES[service],
                    method,
                    path,
                    response.status_code,
                    self._format_response(response),
                    monotonic() - started_at,
                )
                raise AgentError(
                    "EXECUTION_FAILED",
                    error_message,
                )
            LOGGER.info(
                "接口返回 | 服务=%s | 接口=%s %s | HTTP状态=%d | 返回=%s | 耗时=%.2f秒",
                SERVICE_NAMES[service],
                method,
                path,
                response.status_code,
                self._format_response(response),
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

    @staticmethod
    def _format_log_value(value: Any) -> str:
        """把请求或响应整理为单行中文日志可读的紧凑 JSON。"""

        if value is None:
            return "无"
        return jsonlib.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _format_response(cls, response: httpx.Response) -> str:
        """优先记录 JSON 响应，非 JSON 内容限制长度后记录。"""

        try:
            return cls._format_log_value(response.json())
        except ValueError:
            text = response.text.strip()
            return text[:500] if text else "无"
