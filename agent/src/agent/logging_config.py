"""Agent 运行日志的统一配置和任务上下文。

业务模块只负责产生日志，不直接创建 Handler。命令行入口调用 ``configure_logging``
后，所有 ``agent.*`` Logger 都会使用相同格式输出到标准错误。ContextVar 会自动把
当前任务 ID 传递给异步子任务，因此并发执行的导航和位姿日志仍能关联到同一任务。
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token


LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | task=%(task_run_id)s | "
    "%(name)s | %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 未进入任务上下文的启动日志使用短横线，避免格式化阶段缺少字段。
_TASK_RUN_ID: ContextVar[str] = ContextVar("agent_task_run_id", default="-")


class TaskContextFilter(logging.Filter):
    """把当前异步上下文中的任务 ID 注入每条日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        """补充格式化所需字段，并始终允许日志继续输出。"""

        record.task_run_id = _TASK_RUN_ID.get()
        return True


def bind_task_run_id(task_run_id: str) -> Token[str]:
    """绑定当前任务 ID，并返回供调用方恢复上下文的 Token。"""

    return _TASK_RUN_ID.set(task_run_id)


def reset_task_run_id(token: Token[str]) -> None:
    """任务结束后恢复此前的日志上下文。"""

    _TASK_RUN_ID.reset(token)


def configure_logging(level: str = "INFO") -> None:
    """为 ``agent`` Logger 配置可重复调用且不会重复输出的控制台日志。

    日志写入 ``StreamHandler`` 的默认目标 stderr，使 CLI 的 stdout 只保留最终 JSON。
    应用若已有自己的日志体系，可以不调用本函数，直接接管 ``agent`` Logger。
    """

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"unsupported log level: {level}")

    agent_logger = logging.getLogger("agent")
    agent_logger.setLevel(numeric_level)

    # CLI 可能在同一进程中被测试多次，通过自定义标记避免重复添加 Handler。
    handler = next(
        (
            existing
            for existing in agent_logger.handlers
            if getattr(existing, "_agent_console_handler", False)
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler._agent_console_handler = True  # type: ignore[attr-defined]
        handler.addFilter(TaskContextFilter())
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        agent_logger.addHandler(handler)

    handler.setLevel(numeric_level)
    # 已由专用 Handler 输出，禁止继续冒泡到 root 造成一条日志打印两次。
    agent_logger.propagate = False
