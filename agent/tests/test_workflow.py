"""三个比赛工作流的端到端编排测试。

每个用例都从 ``run_task`` 入口执行完整 LangGraph，并通过 Mock HTTP 服务观察外部
行为。测试不仅检查最终状态，还覆盖动作顺序、双手分配、失败后停止和幂等重试。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from agent.main import run_task
from agent.models import AgentSettings, TaskType, TimeoutSettings
from tests.mock_services import MockServices


# 测试复用生产配置的数据结构与货位表，只缩短耗时相关配置。
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "agent.mock.yaml"


def make_test_settings() -> AgentSettings:
    """加载真实配置，并用亚秒级超时替换实机长超时以加快测试。"""

    settings = AgentSettings.load(CONFIG_PATH)
    timeouts = TimeoutSettings(
        connect_seconds=0.1,
        health_seconds=0.2,
        receipt_seconds=0.2,
        inspection_seconds=0.2,
        navigation_seconds=0.2,
        pose_seconds=0.2,
        pick_seconds=0.2,
        place_seconds=0.2,
    )
    return settings.model_copy(update={"timeouts": timeouts})


def test_run_task_works_with_cli_event_loop() -> None:
    """直接使用 asyncio.run 时状态图条件路由也必须正常推进，不能卡在线程池。"""

    mock = MockServices()

    result = asyncio.run(
        run_task(
            TaskType.SORTING,
            settings=make_test_settings(),
            transport=mock.transport,
        )
    )

    assert result["status"] == "SUCCEEDED"


@pytest.mark.asyncio
async def test_sorting_success() -> None:
    """商品拣选应左右手各取一件、全部交付，并最终返回任务判定区。"""

    mock = MockServices()

    result = await run_task(TaskType.SORTING, settings=make_test_settings(), transport=mock.transport)

    assert result["status"] == "SUCCEEDED"
    assert result["held_items"] == {}
    assert all(job["picked"] and job["placed"] for job in result["jobs"])
    pick_calls = mock.calls(path="/pick")
    place_calls = mock.calls(path="/place")
    assert [call.payload["hand"] for call in pick_calls] == ["LEFT", "RIGHT"]
    assert [call.payload["hand"] for call in place_calls] == ["LEFT", "RIGHT"]
    assert len(mock.calls(path="/perception/parse")) == 1
    assert [call.payload["location"] for call in mock.calls(path="/sku/name")] == [
        "H1_F_L1_C01",
        "H1_F_L1_C02",
    ]
    assert mock.calls(path="/navigation/navigate")[-1].payload == {"target_id": "task_boundary"}

    receipt_navigation_index = next(
        index
        for index, call in enumerate(mock.requests)
        if call.path == "/navigation/navigate"
        and call.payload == {"target_id": "receipt_viewpoint"}
    )
    receipt_pose_index = next(
        index
        for index, call in enumerate(mock.requests)
        if call.path == "/pose/prepare"
        and call.payload == {"pose_type": "RECEIPT_VIEW"}
    )
    assert receipt_navigation_index < receipt_pose_index


@pytest.mark.asyncio
async def test_shortage_success_across_inspection_points() -> None:
    """补货任务应跨货架面累计两处缺货，再完成两项补货作业。"""

    mock = MockServices()
    mock.inspection_results = [
        [], [],  # 第一个巡检点：上半部、下半部
        ["H1_F_L2_C01"], [],  # 第二个巡检点
        ["H2_B_L3_C02"], [],  # 第三个巡检点
    ]

    result = await run_task(TaskType.SHORTAGE, settings=make_test_settings(), transport=mock.transport)

    assert result["status"] == "SUCCEEDED"
    assert result["findings"] == ["H1_F_L2_C01", "H2_B_L3_C02"]
    assert [job["hand"] for job in result["jobs"]] == ["LEFT", "RIGHT"]
    assert len(mock.calls(path="/perception/inspect")) == 6
    assert all(job["placed"] for job in result["jobs"])

    first_point_events = [
        (index, call)
        for index, call in enumerate(mock.requests)
        if (
            call.path == "/navigation/navigate"
            and call.payload == {"target_id": "H1_F_INSPECT"}
        )
        or (
            call.path == "/pose/prepare"
            and call.payload["pose_type"] in {"SHELF_VIEW_UPPER", "SHELF_VIEW_LOWER"}
        )
        or call.path == "/perception/inspect"
    ]
    first_nav = next(index for index, call in first_point_events if call.path == "/navigation/navigate")
    first_upper_pose = next(
        index
        for index, call in first_point_events
        if call.path == "/pose/prepare"
        and call.payload["pose_type"] == "SHELF_VIEW_UPPER"
    )
    first_upper_capture = next(
        index for index, call in first_point_events if call.path == "/perception/inspect"
    )
    first_lower_pose = next(
        index
        for index, call in first_point_events
        if call.path == "/pose/prepare"
        and call.payload["pose_type"] == "SHELF_VIEW_LOWER"
    )
    first_lower_capture = [
        index for index, call in first_point_events if call.path == "/perception/inspect"
    ][1]
    assert first_nav < first_upper_pose < first_upper_capture < first_lower_pose < first_lower_capture


@pytest.mark.asyncio
async def test_misplaced_success_uses_fixed_swap_order() -> None:
    """乱放归位必须按左抓、右抓、左放、右放完成两件商品互换。"""

    mock = MockServices()
    mock.inspection_results = [[], [], ["H1_F_L1_C01"], ["H1_F_L1_C02"]]

    result = await run_task(TaskType.MISPLACED, settings=make_test_settings(), transport=mock.transport)

    assert result["status"] == "SUCCEEDED"
    manipulation_calls = [
        call
        for call in mock.requests
        if call.path in {"/pick", "/place"}
    ]
    assert [(call.path, call.payload["hand"]) for call in manipulation_calls] == [
        ("/pick", "LEFT"),
        ("/pick", "RIGHT"),
        ("/place", "LEFT"),
        ("/place", "RIGHT"),
    ]
    assert manipulation_calls[0].payload["product_name"] == "矿泉水"
    assert manipulation_calls[1].payload["product_name"] == "可口可乐"
    assert len(mock.calls(path="/perception/inspect")) == 4
    navigation_targets = [
        call.payload["target_id"]
        for call in mock.calls(path="/navigation/navigate")
    ]
    # P2 抓取后机器人已经在 P2，左手放置前只准备位姿，不应重复导航。
    assert navigation_targets.count("H1_F_L1_C02") == 1
    # P1 第一次用于左手抓取，第二次用于右手放回。
    assert navigation_targets.count("H1_F_L1_C01") == 2


@pytest.mark.asyncio
async def test_not_ready_fails_without_physical_actions() -> None:
    """任一能力模块未就绪时应在起步前失败，不能发送物理动作。"""

    mock = MockServices()
    mock.set_health("pose", "ERROR")

    result = await run_task(TaskType.SORTING, settings=make_test_settings(), transport=mock.transport)

    assert result["status"] == "FAILED"
    assert result["error_code"] == "CAPABILITY_NOT_READY"
    assert not any("idempotency-key" in call.headers for call in mock.requests)


@pytest.mark.asyncio
async def test_health_checks_use_module_specific_paths() -> None:
    """健康检查必须使用各模块最新约定的带模块名前缀路径。"""

    mock = MockServices()

    result = await run_task(TaskType.SORTING, settings=make_test_settings(), transport=mock.transport)

    assert result["status"] == "SUCCEEDED"
    health_calls = {
        (call.service, call.path)
        for call in mock.requests
        if call.method == "GET" and call.path.endswith("/health")
    }
    assert health_calls == {
        ("navigation", "/navigation/health"),
        ("pick_place", "/health"),
        ("perception", "/perception/health"),
        ("pose", "/pose/health"),
        ("sku", "/sku/health"),
    }


@pytest.mark.asyncio
async def test_non_2xx_fails_without_retry() -> None:
    """明确的非 2xx 响应是执行失败，不应像网络超时一样自动重试。"""

    mock = MockServices()
    mock.fail_next("navigation", status_code=500)

    result = await run_task(TaskType.SORTING, settings=make_test_settings(), transport=mock.transport)

    assert result["status"] == "FAILED"
    assert result["error_code"] == "EXECUTION_FAILED"
    assert len(mock.calls(path="/navigation/navigate")) == 1
    assert not mock.calls(path="/pose/prepare")
    assert not mock.calls(path="/pick")


@pytest.mark.asyncio
async def test_timeout_retry_reuses_key_and_action() -> None:
    """一次超时后应复用幂等键重试，且底层真实导航只执行一次。"""

    mock = MockServices()
    mock.timeout_next("navigation")

    result = await run_task(TaskType.SORTING, settings=make_test_settings(), transport=mock.transport)

    assert result["status"] == "SUCCEEDED"
    navigation_calls = mock.calls(path="/navigation/navigate")
    first_key = navigation_calls[0].headers["idempotency-key"]
    matching_calls = [
        call for call in navigation_calls if call.headers["idempotency-key"] == first_key
    ]
    assert len(matching_calls) == 2
    assert mock.actual_action_counts[first_key] == 1


@pytest.mark.asyncio
async def test_two_timeouts_stop_with_unknown_result() -> None:
    """连续两次超时后动作结果未知，工作流必须停止且不得继续抓取。"""

    mock = MockServices()
    mock.timeout_next("navigation", times=2)

    result = await run_task(TaskType.SORTING, settings=make_test_settings(), transport=mock.transport)

    assert result["status"] == "FAILED"
    assert result["error_code"] == "ACTION_RESULT_UNKNOWN"
    assert result["current_action_status"] == "UNKNOWN"
    navigation_calls = mock.calls(path="/navigation/navigate")
    assert len(navigation_calls) == 2
    first_key = navigation_calls[0].headers["idempotency-key"]
    assert {call.headers["idempotency-key"] for call in navigation_calls} == {first_key}
    assert mock.actual_action_counts[first_key] == 1
    assert not mock.calls(path="/pose/prepare")
    assert not mock.calls(path="/pick")


@pytest.mark.asyncio
async def test_shortage_keeps_inspecting_beyond_two_passes() -> None:
    """两轮仍无结果时继续第三轮，且每次换轮的重合点只识别。"""

    mock = MockServices()
    mock.inspection_results = [
        *[[] for _ in range(12)],
        ["H1_F_L2_C01", "H2_B_L3_C02"], [],
    ]
    raw_settings = make_test_settings().model_dump()
    raw_settings["inspection_points"] = ["POINT_A", "POINT_B", "POINT_C"]
    settings = AgentSettings.model_validate(raw_settings)

    result = await run_task(TaskType.SHORTAGE, settings=settings, transport=mock.transport)

    assert result["status"] == "SUCCEEDED"
    assert result["inspection_pass"] == 3
    assert len(mock.calls(path="/perception/inspect")) == 14
    inspection_navigation_targets = [
        call.payload["target_id"]
        for call in mock.calls(path="/navigation/navigate")
        if call.payload["target_id"].startswith("POINT_")
    ]
    assert inspection_navigation_targets == [
        "POINT_A",
        "POINT_B",
        "POINT_C",
        "POINT_B",
        "POINT_A",
    ]
    upper_view_calls = [
        call
        for call in mock.calls(path="/pose/prepare")
        if call.payload["pose_type"] == "SHELF_VIEW_UPPER"
    ]
    lower_view_calls = [
        call
        for call in mock.calls(path="/pose/prepare")
        if call.payload["pose_type"] == "SHELF_VIEW_LOWER"
    ]
    assert len(upper_view_calls) == 7
    assert len(lower_view_calls) == 7
    assert mock.calls(path="/pick")


@pytest.mark.asyncio
async def test_long_action_finishes_before_configured_timeout() -> None:
    """耗时但未超过配置上限的动作应正常完成，不能被过早取消或重复执行。"""

    mock = MockServices()
    mock.set_delay("navigation", seconds=0.1)

    result = await run_task(TaskType.SORTING, settings=make_test_settings(), transport=mock.transport)

    assert result["status"] == "SUCCEEDED"
    assert all(mock.actual_action_counts[key] == 1 for key in mock.actual_action_counts)


@pytest.mark.asyncio
async def test_success_emits_clear_progress_logs(caplog: pytest.LogCaptureFixture) -> None:
    """成功任务应记录任务边界、业务步骤、能力动作和最终结果。"""

    mock = MockServices()
    caplog.set_level(logging.INFO, logger="agent")

    result = await run_task(
        TaskType.SORTING,
        settings=make_test_settings(),
        transport=mock.transport,
    )

    assert result["status"] == "SUCCEEDED"
    messages = [record.getMessage() for record in caplog.records if record.name.startswith("agent")]
    assert any("任务开始 | 任务=商品拣选（SORTING）" in message for message in messages)
    assert any("检查主流程依赖的5个服务" in message for message in messages)
    assert any(
        "调用接口 | 用途=执行完整抓取流程" in message
        and "接口=POST /pick" in message
        and "入参=" in message
        for message in messages
    )
    assert any(
        "接口返回 | 服务=取放编排 | 接口=POST /place" in message and "返回=" in message
        for message in messages
    )
    assert any("右手已放置" in message for message in messages)
    assert any("任务结束 | 任务=商品拣选 | 结果=成功" in message for message in messages)


@pytest.mark.asyncio
async def test_retry_and_failure_emit_warning_and_error_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """网络重试和结果未知必须分别留下 WARNING 与 ERROR 级别日志。"""

    mock = MockServices()
    mock.timeout_next("navigation", times=2)
    caplog.set_level(logging.INFO, logger="agent")

    result = await run_task(
        TaskType.SORTING,
        settings=make_test_settings(),
        transport=mock.transport,
    )

    assert result["status"] == "FAILED"
    assert any(
        record.levelno == logging.WARNING and "准备重试" in record.getMessage()
        for record in caplog.records
    )
    assert any(
        record.levelno == logging.ERROR and "ACTION_RESULT_UNKNOWN" in record.getMessage()
        for record in caplog.records
    )
