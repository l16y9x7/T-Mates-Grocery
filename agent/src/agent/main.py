"""Agent 的程序入口。

``run_task`` 供其他 Python 代码调用，``cli`` 提供命令行启动方式。入口只负责加载
配置、创建能力客户端和执行已编译的工作流，不承载具体比赛规则。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from sys import maxsize

import httpx

from agent.client import CapabilityClient
from agent.logging_config import configure_logging
from agent.models import AgentSettings, TaskType, WorkflowState
from agent.workflow import WorkflowBuilder, initial_state


# 使用相对源码位置推导默认配置，避免依赖启动命令所在的当前工作目录。
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "agent.production.yaml"
LOGGER = logging.getLogger(__name__)
# LangGraph 要求提供有限整数；使用系统最大整数，实际业务不设置巡检轮数上限。
GRAPH_RECURSION_LIMIT = maxsize
TASK_NAMES = {
    TaskType.SORTING: "商品拣选",
    TaskType.SHORTAGE: "货架补货",
    TaskType.MISPLACED: "乱放归位",
}


async def run_task(
    task_type: TaskType | str,
    *,
    settings: AgentSettings | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    transport: httpx.AsyncBaseTransport | None = None,
) -> WorkflowState:
    """运行一个完整任务并返回最终状态。

    ``task_type`` 同时接受枚举和字符串；``settings`` 便于测试直接注入短超时配置，
    ``transport`` 便于使用 httpx MockTransport。未注入配置时从 ``config_path`` 加载。
    客户端通过异步上下文管理器保证任务成功或异常退出后都能关闭连接池。
    """

    resolved_type = TaskType(task_type)
    resolved_settings = settings or AgentSettings.load(config_path)
    state = initial_state(resolved_settings, resolved_type)

    LOGGER.info("任务开始 | 任务=%s（%s）", TASK_NAMES[resolved_type], resolved_type.value)
    try:
        async with CapabilityClient(resolved_settings, transport=transport) as client:
            graph = WorkflowBuilder(resolved_settings, client).build(resolved_type)
            result = await graph.ainvoke(
                state,
                config={"recursion_limit": GRAPH_RECURSION_LIMIT},
            )
        final_state = WorkflowState(**result)
        LOGGER.info(
            "任务结束 | 任务=%s | 结果=%s | 作业数=%d | 错误码=%s",
            TASK_NAMES[resolved_type],
            "成功" if final_state["status"] == "SUCCEEDED" else "失败",
            len(final_state["jobs"]),
            final_state["error_code"] or "-",
        )
        return final_state
    except BaseException:
        # 业务错误会在工作流内转换成 FAILED；这里只记录真正逃出状态图的意外异常。
        LOGGER.exception("任务异常中止 | 任务=%s", TASK_NAMES[resolved_type])
        raise


def cli() -> None:
    """解析命令行参数，执行任务，并以可读 JSON 输出最终状态。"""

    parser = argparse.ArgumentParser(description="Run one robot task workflow")
    parser.add_argument("task_type", choices=[task.value for task in TaskType])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="控制台日志级别（默认：INFO）",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    result = asyncio.run(run_task(args.task_type, config_path=args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 仅在 ``python -m agent.main`` 或直接运行本文件时启动 CLI；导入时不产生副作用。
    cli()
