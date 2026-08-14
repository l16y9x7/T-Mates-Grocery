"""Parse a receipt image from disk without starting an HTTP server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__:
    from . import server
else:
    PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
    if str(PERCEPTION_ROOT) not in sys.path:
        sys.path.insert(0, str(PERCEPTION_ROOT))
    from parse_receipt import server


def read_image_file(image_path: str | Path) -> bytes:
    """Read one local image while enforcing the service's input size limit."""

    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在：{path}")
    if not path.is_file():
        raise ValueError(f"图片路径不是文件：{path}")

    try:
        with path.open("rb") as image_file:
            raw = image_file.read(server.MAX_CAMERA_BYTES + 1)
    except OSError as exc:
        raise OSError(f"无法读取图片文件 {path}：{exc}") from exc

    if not raw:
        raise ValueError(f"图片文件为空：{path}")
    if len(raw) > server.MAX_CAMERA_BYTES:
        raise ValueError(
            f"图片文件超过 {server.MAX_CAMERA_BYTES // (1024 * 1024)}MB 限制：{path}"
        )
    return raw


def parse_image_file(
    image_path: str | Path,
    settings: server.Settings | None = None,
) -> server.ParseReceiptResponse:
    """Run the same recognition and SKU matching stages as /perception/parse."""

    configured = settings or server.Settings.from_env()
    recognized_items = server.recognize_frame(
        read_image_file(image_path),
        configured,
    )
    if len(recognized_items) != 2:
        raise server.ServiceError(
            502,
            "qwen_output_error",
            f"Qwen 必须识别出两个商品，当前得到 {len(recognized_items)} 个。",
        )
    return server.ParseReceiptResponse(
        product_names=server.lookup_sku_items(recognized_items, configured)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取本地小票图片并输出两个标准 SKU 商品名称（不会启动接口）。"
    )
    parser.add_argument("image", type=Path, help="本地图片文件路径")
    return parser


def _error_payload(error: Exception) -> dict[str, object]:
    if isinstance(error, server.ServiceError):
        detail: dict[str, object] = {
            "type": error.error_type,
            "message": error.message,
            "stage": error.stage,
            "retryable": error.retryable,
            "hint": error.hint,
        }
        if error.upstream_status_code is not None:
            detail["upstream_status_code"] = error.upstream_status_code
        if error.upstream is not None:
            detail["upstream"] = error.upstream
        if error.elapsed_ms is not None:
            detail["elapsed_ms"] = error.elapsed_ms
        if error.timeout_seconds is not None:
            detail["timeout_seconds"] = error.timeout_seconds
        return {"error": detail}
    return {
        "error": {
            "type": "image_file_error",
            "message": str(error),
            "stage": "image_read",
            "retryable": False,
        }
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = parse_image_file(args.image)
    except (OSError, ValueError, server.ServiceError) as error:
        print(
            json.dumps(_error_payload(error), ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {"product_names": result.product_names},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
