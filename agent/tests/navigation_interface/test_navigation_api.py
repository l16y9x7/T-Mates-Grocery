"""模拟 Agent 调用导航模块的独立联调脚本，仅使用 Python 标准库。"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4


NETWORK_ERRORS = (TimeoutError, socket.timeout, URLError, ConnectionError, HTTPException)


class InterfaceTestError(Exception):
    """表示接口响应不符合约定。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试导航模块 HTTP 接口")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8101",
        help="导航模块基础地址，默认 http://127.0.0.1:8101",
    )
    parser.add_argument(
        "--target-id",
        required=True,
        help="由导航模块测试人员填写的安全导航目标",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600.0,
        help="单次导航请求超时，默认与 Agent 导航读取超时一致为 600 秒",
    )
    parser.add_argument(
        "--verify-idempotency",
        action="store_true",
        help="成功后使用相同键重放一次；启用前确认现场安全",
    )
    return parser.parse_args()


def validate_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise InterfaceTestError("--base-url 必须是完整的 http:// 或 https:// 地址")
    return normalized


def request_json(
    method: str,
    url: str,
    *,
    timeout_seconds: float,
    payload: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, Any, float]:
    """发送一次 JSON 请求；非 2xx 也作为明确 HTTP 响应返回。"""

    request_headers = {"Accept": "application/json", **(headers or {})}
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json; charset=utf-8"
    request = Request(url, data=body, headers=request_headers, method=method)

    started_at = time.monotonic()
    try:
        response = urlopen(request, timeout=timeout_seconds)
    except HTTPError as exc:
        response = exc

    try:
        status_code = response.status
        content_type = response.headers.get_content_type()
        response_body = response.read()
    finally:
        response.close()
    elapsed = time.monotonic() - started_at

    if content_type != "application/json":
        raise InterfaceTestError(
            f"响应 Content-Type 必须是 application/json，实际为 {content_type!r}"
        )
    try:
        response_payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InterfaceTestError(
            f"响应不是合法 UTF-8 JSON：status={status_code}, body={response_body!r}"
        ) from exc
    return status_code, response_payload, elapsed


def check_health(base_url: str) -> None:
    url = f"{base_url}/navigation/health"
    print(f"[1/2] 调用健康检查：GET {url}")
    try:
        status_code, payload, _ = request_json(
            "GET",
            url,
            timeout_seconds=5.0,
        )
    except NETWORK_ERRORS as exc:
        raise InterfaceTestError(f"健康检查连接失败：{exc}") from exc

    print(f"      HTTP {status_code}，响应={payload}")
    if status_code != 200:
        raise InterfaceTestError("健康检查必须返回 HTTP 200")
    if payload != {"status": "READY"}:
        raise InterfaceTestError('健康检查必须严格返回 {"status":"READY"}')


def call_navigation(
    base_url: str,
    target_id: str,
    idempotency_key: str,
    timeout_seconds: float,
) -> float:
    url = f"{base_url}/navigation/navigate"
    headers = {"Idempotency-Key": idempotency_key}
    payload = {"target_id": target_id}

    for attempt in range(1, 3):
        started_at = time.monotonic()
        print(
            f"      第 {attempt}/2 次请求：POST {url}\n"
            f"      请求体={payload}\n"
            f"      Idempotency-Key={idempotency_key}"
        )
        try:
            status_code, response_payload, elapsed = request_json(
                "POST",
                url,
                timeout_seconds=timeout_seconds,
                payload=payload,
                headers=headers,
            )
        except NETWORK_ERRORS as exc:
            elapsed = time.monotonic() - started_at
            print(f"      网络异常：{type(exc).__name__}，耗时={elapsed:.3f}s")
            if attempt == 1:
                print("      按 Agent 规则复用原幂等键重试一次。")
                continue
            raise InterfaceTestError("连续两次网络异常，导航动作结果未知") from exc

        print(f"      HTTP {status_code}，耗时={elapsed:.3f}s，响应={response_payload}")
        if not 200 <= status_code < 300:
            raise InterfaceTestError("导航明确返回非 2xx；按 Agent 规则不重试")
        if response_payload != {"status": "SUCCEEDED"}:
            raise InterfaceTestError('导航成功响应必须严格为 {"status":"SUCCEEDED"}')
        return elapsed

    raise AssertionError("unreachable retry state")


def main() -> int:
    args = parse_args()
    if args.timeout_seconds <= 0:
        print("测试失败：--timeout-seconds 必须大于 0", file=sys.stderr)
        return 2

    try:
        base_url = validate_base_url(args.base_url)
        task_run_id = uuid4().hex
        action_id = "navigation.interface_test.navigate"
        idempotency_key = f"{task_run_id}:{action_id}"

        check_health(base_url)
        print(f"[2/2] 调用导航接口，target_id={args.target_id}")
        call_navigation(
            base_url,
            args.target_id,
            idempotency_key,
            args.timeout_seconds,
        )

        if args.verify_idempotency:
            print("[附加] 使用相同请求体和 Idempotency-Key 重放请求")
            replay_elapsed = call_navigation(
                base_url,
                args.target_id,
                idempotency_key,
                args.timeout_seconds,
            )
            print(
                "      重放响应正确。请结合服务端日志确认真实导航只执行了一次，"
                f"重放耗时={replay_elapsed:.3f}s。"
            )
    except InterfaceTestError as exc:
        print(f"测试失败：{exc}", file=sys.stderr)
        return 1

    print("测试通过：导航模块接口符合当前 Agent 调用协议。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
