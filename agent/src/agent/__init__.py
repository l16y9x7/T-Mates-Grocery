"""机器人零售任务编排包的公共接口。

包顶层导出调用方最常用的任务类型、最终状态类型和日志配置函数；HTTP 客户端、
配置模型及工作流构建器保留在各自模块中，避免无意扩大公共 API。
"""

from agent.logging_config import configure_logging
from agent.models import TaskType, WorkflowState

__all__ = ["TaskType", "WorkflowState", "configure_logging"]
