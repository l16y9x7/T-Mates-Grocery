"""Agent 运行日志的统一配置。

业务模块只负责产生日志，不直接创建 Handler。命令行入口调用 ``configure_logging``
后，所有 ``agent.*`` Logger 都会使用相同格式输出到标准错误。
"""

from __future__ import annotations

import logging


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


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
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        agent_logger.addHandler(handler)

    handler.setLevel(numeric_level)
    # 已由专用 Handler 输出，禁止继续冒泡到 root 造成一条日志打印两次。
    agent_logger.propagate = False
