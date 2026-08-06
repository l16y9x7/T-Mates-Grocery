"""PaddleOCR-only receipt text experiment.

This module intentionally does not call Qwen and does not produce the final
business receipt JSON.  It only records OCR text evidence that can later be
compared with Qwen output or inventory SKU names.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import InputFileError, ReceiptRecognizerError
from .media import SUPPORTED_IMAGE_SUFFIXES


DEFAULT_OCR_DEVICE = "cpu"
OCR_DEVICE_ENV = "RECEIPT_OCR_DEVICE"
_OCR_DEVICE_PATTERN = re.compile(r"^(?:cpu|gpu(?::\d+)?)$")


@dataclass(frozen=True)
class OCRLine:
    text: str
    score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
        }


@dataclass(frozen=True)
class OCRResult:
    image: str
    ocr_lines: list[OCRLine]

    @property
    def full_text(self) -> str:
        return " ".join(line.text for line in self.ocr_lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "ocr_lines": [line.to_dict() for line in self.ocr_lines],
            "full_text": self.full_text,
        }


def recognize_image_with_paddleocr(
    input_path: Path,
    *,
    lang: str = "ch",
    use_angle_cls: bool = True,
    device: str = DEFAULT_OCR_DEVICE,
) -> OCRResult:
    """Run PaddleOCR on one local JPEG/PNG image."""

    path = input_path.expanduser().resolve()
    _validate_image_path(path)
    engine = _create_paddleocr_engine(
        lang=lang,
        use_angle_cls=use_angle_cls,
        device=_parse_ocr_device(device),
    )
    try:
        raw_result = _run_paddleocr_engine(
            engine,
            path,
            use_textline_orientation=use_angle_cls,
        )
    except Exception as exc:  # pragma: no cover - depends on local OCR runtime
        raise ReceiptRecognizerError(f"PaddleOCR 识别失败：{exc}") from exc

    return OCRResult(
        image=path.name,
        ocr_lines=extract_ocr_lines(raw_result),
    )


def extract_ocr_lines(raw_result: Any) -> list[OCRLine]:
    """Extract text lines from common PaddleOCR result shapes.

    PaddleOCR 2.x usually returns line records shaped like:
    ``[[box, (text, score)], ...]`` wrapped in an outer image/page list.  Some
    versions expose the line list directly.

    PaddleOCR 3.x returns result objects/dicts with ``rec_texts`` and
    ``rec_scores``.  This parser supports both shapes and keeps only text plus
    confidence score, because that is all the receipt experiment needs.
    """

    direct_lines = _extract_v3_result_lines(raw_result)
    if direct_lines:
        return direct_lines

    lines: list[OCRLine] = []
    for candidate in _walk_paddleocr_records(raw_result):
        parsed = _parse_line_record(candidate)
        if parsed is not None:
            lines.append(parsed)
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 PaddleOCR 输出一张本地小票图片的原始 OCR 文本。"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="本地 .jpg、.jpeg 或 .png 文件",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选：保存 OCR JSON 到指定文件",
    )
    parser.add_argument(
        "--lang",
        default="ch",
        help="PaddleOCR 语言参数，默认 ch",
    )
    parser.add_argument(
        "--no-angle-cls",
        action="store_true",
        help="关闭文字方向分类；默认开启",
    )
    parser.add_argument(
        "--device",
        type=_parse_ocr_device,
        default=_parse_ocr_device(
            os.getenv(OCR_DEVICE_ENV, DEFAULT_OCR_DEVICE)
        ),
        help=(
            "OCR 推理设备，例如 cpu 或 gpu:0；默认读取 "
            f"{OCR_DEVICE_ENV}，未设置时使用 {DEFAULT_OCR_DEVICE}"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = recognize_image_with_paddleocr(
            args.input,
            lang=args.lang,
            use_angle_cls=not args.no_angle_cls,
            device=args.device,
        )
    except ReceiptRecognizerError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    text = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _validate_image_path(path: Path) -> None:
    if not path.is_file():
        raise InputFileError(f"输入文件不存在：{path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise InputFileError(f"OCR 实验仅支持 {supported}，实际为：{suffix or '无扩展名'}")


def _create_paddleocr_engine(
    *,
    lang: str,
    use_angle_cls: bool,
    device: str,
):
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:  # pragma: no cover - exercised without dep
        raise ReceiptRecognizerError(
            "当前环境未安装 PaddleOCR。请先运行："
            'python -m pip install -e ".[ocr]"；如果仍提示缺少 paddle，'
            "请按 PaddleOCR 官方说明安装 paddlepaddle。"
        ) from exc
    try:
        return PaddleOCR(
            use_textline_orientation=use_angle_cls,
            lang=lang,
            device=device,
        )
    except TypeError:
        # PaddleOCR 2.x used the old ``use_angle_cls`` name.
        return PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=device.startswith("gpu"),
        )
    except RuntimeError as exc:
        if "paddlepaddle" in str(exc).lower():
            raise ReceiptRecognizerError(
                "当前环境已安装 paddleocr，但还缺少 paddlepaddle。"
                "请在 receipt-qwen-vl 环境里运行：python -m pip install paddlepaddle"
            ) from exc
        raise ReceiptRecognizerError(f"PaddleOCR 初始化失败：{exc}") from exc


def _parse_ocr_device(value: str) -> str:
    normalized = value.strip().lower()
    if not _OCR_DEVICE_PATTERN.fullmatch(normalized):
        raise argparse.ArgumentTypeError(
            "OCR device 只支持 cpu、gpu 或 gpu:<非负整数>。"
        )
    return normalized


def _run_paddleocr_engine(
    engine: Any,
    path: Path,
    *,
    use_textline_orientation: bool,
) -> Any:
    if hasattr(engine, "predict"):
        return engine.predict(
            str(path),
            use_textline_orientation=use_textline_orientation,
        )
    return engine.ocr(str(path), cls=use_textline_orientation)


def _extract_v3_result_lines(value: Any) -> list[OCRLine]:
    lines: list[OCRLine] = []
    for result in _walk_v3_results(value):
        texts = _get_result_field(result, "rec_texts") or []
        scores = _get_result_field(result, "rec_scores") or []
        for index, text in enumerate(texts):
            clean = str(text).strip()
            if not clean:
                continue
            score = scores[index] if index < len(scores) else None
            lines.append(OCRLine(text=clean, score=_parse_score(score)))
    return lines


def _walk_v3_results(value: Any) -> Iterable[Any]:
    if _get_result_field(value, "rec_texts") is not None:
        yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from _walk_v3_results(item)


def _get_result_field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    if hasattr(value, "get"):
        try:
            return value.get(key)
        except Exception:
            return None
    return getattr(value, key, None)


def _walk_paddleocr_records(value: Any) -> Iterable[Any]:
    if _looks_like_line_record(value):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from _walk_paddleocr_records(item)


def _looks_like_line_record(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[1], (tuple, list))
        and len(value[1]) >= 1
        and isinstance(value[1][0], str)
    )


def _parse_line_record(value: Any) -> OCRLine | None:
    if not _looks_like_line_record(value):
        return None
    text = str(value[1][0]).strip()
    if not text:
        return None
    score = _parse_score(value[1][1] if len(value[1]) > 1 else None)
    return OCRLine(text=text, score=score)


def _parse_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
