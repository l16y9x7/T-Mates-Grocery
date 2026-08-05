"""Command-line entrypoint for one local image or PDF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .errors import ReceiptRecognizerError
from .service import ReceiptRecognizer
from .sku_client import SkuLookupClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 Qwen3-VL 识别一张本地购物小票图片或 PDF。"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="本地 .jpg、.jpeg、.png 或 .pdf 文件",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="将行级诊断 JSON 输出到 stderr",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=2200,
        help="发送前图片最长边，默认 2200",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=180,
        help="PDF 渲染 DPI，默认 180",
    )
    parser.add_argument(
        "--max-pdf-pages",
        type=int,
        default=1,
        help="最多读取 PDF 前几页，默认 1；多图验证后再增加",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Qwen 采样温度，默认 0.0；实验稳定性时可试 0.2、0.5",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        recognizer = ReceiptRecognizer(settings)
        result = recognizer.recognize_file(
            args.input,
            max_edge=args.max_edge,
            pdf_dpi=args.pdf_dpi,
            max_pdf_pages=args.max_pdf_pages,
            temperature=args.temperature,
        )
        sku_validation = SkuLookupClient(settings).validate_items(
            result.business_items
        )
    except ReceiptRecognizerError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    if args.diagnostics:
        print(
            json.dumps(
                result.diagnostics,
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
    print(
        json.dumps(
            {
                "items": result.business_items,
                "sku_validation": sku_validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
