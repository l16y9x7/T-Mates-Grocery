"""Read-only probes for discovering the deployed API behavior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import OpenAICompatibleClient
from .config import Settings
from .errors import ReceiptRecognizerError
from .service import ReceiptRecognizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读检查 Qwen OpenAI-compatible API。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("models", help="GET /v1/models")
    subparsers.add_parser("openapi", help="GET /openapi.json")
    subparsers.add_parser("text", help="发送最小纯文字 Chat 请求")

    image_parser = subparsers.add_parser(
        "receipt",
        help="用本地图片或 PDF 验证小票识别",
    )
    image_parser.add_argument("input", type=Path)
    image_parser.add_argument(
        "--diagnostics",
        action="store_true",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        client = OpenAICompatibleClient(settings)

        if args.command == "models":
            result = client.list_models()
        elif args.command == "openapi":
            result = client.get_openapi()
        elif args.command == "text":
            response = client.create_chat_completion(
                [
                    {
                        "role": "user",
                        "content": "只回复：连接成功",
                    }
                ],
                temperature=0.0,
                max_tokens=32,
            )
            result = {
                "content": response.content,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
            }
        else:
            recognition = ReceiptRecognizer(
                settings,
                client=client,
            ).recognize_file(args.input)
            result = (
                recognition.diagnostics
                if args.diagnostics
                else recognition.business_items
            )
    except ReceiptRecognizerError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

