from __future__ import annotations

import argparse
import io
import json
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError


DEFAULT_URL = "http://192.168.130.50:8085/camera/snapshot?camera=head&type=color"


def test_snapshot(url: str, timeout: float, save_path: Path | None = None) -> dict:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(f"URL 无效: {url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    tcp_started = time.perf_counter()
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            pass
    except OSError as error:
        raise RuntimeError(
            f"TCP 连接失败: {parsed.hostname}:{port}: {error}"
        ) from error
    tcp_ms = round((time.perf_counter() - tcp_started) * 1000, 2)

    http_started = time.perf_counter()
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as error:
        raise RuntimeError(f"HTTP 请求失败: {error}") from error
    http_ms = round((time.perf_counter() - http_started) * 1000, 2)
    if not response.ok:
        body_preview = response.text[:500]
        raise RuntimeError(
            f"HTTP {response.status_code}，响应内容: {body_preview}"
        )

    image_bytes = response.content
    if not image_bytes:
        raise RuntimeError("HTTP 请求成功，但响应图片为空")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = (image.format or "").upper()
            image_size = list(image.size)
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise RuntimeError(f"响应无法解码为图片: {error}") from error
    if image_format not in {"JPEG", "PNG"}:
        raise RuntimeError(f"图片格式不是 JPG/PNG: {image_format or 'UNKNOWN'}")

    saved_to = None
    if save_path is not None:
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(image_bytes)
        except OSError as error:
            raise RuntimeError(f"保存快照失败: {error}") from error
        saved_to = str(save_path.resolve())

    return {
        "ok": True,
        "url": url,
        "tcp": {
            "host": parsed.hostname,
            "port": port,
            "latency_ms": tcp_ms,
        },
        "http": {
            "status_code": response.status_code,
            "latency_ms": http_ms,
            "content_type": response.headers.get("Content-Type"),
            "content_length": len(image_bytes),
        },
        "image": {
            "format": image_format,
            "size": image_size,
        },
        "saved_to": saved_to,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试相机 snapshot 接口连通性")
    parser.add_argument("--url", default=DEFAULT_URL, help="snapshot 接口地址")
    parser.add_argument("--timeout", type=float, default=10, help="TCP/HTTP 超时秒数")
    parser.add_argument("--save", type=Path, help="可选的快照保存路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = test_snapshot(args.url, args.timeout, args.save)
    except RuntimeError as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
