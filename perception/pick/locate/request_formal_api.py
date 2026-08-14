from __future__ import annotations

import argparse
import json
import os
import sys

import requests


DEFAULT_LOCATE_URL = os.getenv(
    "LOCATE_FORMAL_API_URL",
    "http://127.0.0.1:8083/perception/pick/locate",
)


def request_locate(
    task_type: str,
    product_name: str,
    level: str,
    hand: str,
    url: str = DEFAULT_LOCATE_URL,
    timeout: float = 600,
) -> dict:
    payload = {
        "task_type": task_type,
        "product_name": product_name,
        "level": level,
        "hand": hand,
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as error:
        raise RuntimeError(f"Locate API 请求失败: {error}") from error

    try:
        result = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"Locate API 返回非 JSON，HTTP {response.status_code}: {response.text[:500]}"
        ) from error

    if not response.ok:
        raise RuntimeError(
            f"Locate API 返回 HTTP {response.status_code}: "
            f"{json.dumps(result, ensure_ascii=False)}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="请求正式商品定位接口")
    parser.add_argument("task_type", help="请求的 task_type，例如 SORTING")
    parser.add_argument("product_name", help="完整商品名称，例如 可口可乐")
    parser.add_argument("level", help="目标所在层，例如 L4")
    parser.add_argument("hand", help="使用的手，例如 left 或 right")
    parser.add_argument("--url", default=DEFAULT_LOCATE_URL, help="正式接口地址")
    parser.add_argument("--timeout", type=float, default=600, help="请求超时秒数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = request_locate(
            args.task_type.strip(),
            args.product_name.strip(),
            args.level.strip().upper(),
            args.hand.strip(),
            url=args.url,
            timeout=args.timeout,
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
