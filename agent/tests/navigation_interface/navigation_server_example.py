"""导航模块接口写法示例。

注意：本文件只说明 HTTP 接口的路径、参数和返回格式，不指导导航模块内部实现。
示例采用 FastAPI 语法便于阅读，Nora 可以按自己项目使用的 Web 框架实现同一协议。

Idempotency-Key 原理：
1. Agent 为每个逻辑动作生成唯一键，并放在 HTTP 请求头中。
2. 网络超时时，Agent 会使用相同请求体和相同键重试一次。
3. 导航模块第一次收到键时执行导航；再次收到相同键时不能重复移动。
4. 重复请求应等待第一次执行结束，或直接返回第一次保存的最终结果。
5. 相同键对应不同 target_id 时，应返回 HTTP 409。

幂等记录使用内存、数据库还是其他方式，由导航模块自行决定。
"""

from typing import Literal

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


app = FastAPI()


class HealthResponse(BaseModel):
    status: Literal["STARTING", "READY", "ERROR"]


class NavigateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_id: str = Field(min_length=1)


class NavigateResponse(BaseModel):
    status: Literal["SUCCEEDED"]


class IdempotencyKeyConflict(Exception):
    """相同 Idempotency-Key 被用于不同导航目标。"""


async def navigate_once(target_id: str, idempotency_key: str) -> None:
    """由导航模块自行实现。

    必须阻塞等待机器人真正到达目标后再返回，并保证相同 Idempotency-Key 只执行
    一次真实导航。导航失败时抛出异常；幂等键冲突时抛出 IdempotencyKeyConflict。
    """

    raise NotImplementedError


@app.get("/navigation/health", response_model=HealthResponse)
async def navigation_health() -> HealthResponse:
    """模块完成初始化且能够执行导航时返回 READY。"""

    return HealthResponse(status="READY")


@app.post("/navigation/navigate", response_model=NavigateResponse)
async def navigation_navigate(
    request: NavigateRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> NavigateResponse | JSONResponse:
    """接收一次阻塞式导航请求。"""

    if not idempotency_key:
        return JSONResponse(
            {"error_code": "MISSING_IDEMPOTENCY_KEY"},
            status_code=400,
        )

    try:
        await navigate_once(request.target_id, idempotency_key)
    except IdempotencyKeyConflict:
        return JSONResponse(
            {"error_code": "IDEMPOTENCY_KEY_CONFLICT"},
            status_code=409,
        )
    except Exception:
        return JSONResponse(
            {"error_code": "EXECUTION_FAILED"},
            status_code=500,
        )

    return NavigateResponse(status="SUCCEEDED")
