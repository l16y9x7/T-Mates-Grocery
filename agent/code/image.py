"""检查 8085 相机服务，并保存一张 RGB 快照和视频流样本。

示例：
    python code/image.py
    python code/image.py --camera head --output-dir /tmp/camera-test
    python code/image.py --skip-stream
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests


DEFAULT_CAMERA_URL = "http://192.168.130.50:8085"
DEFAULT_CAMERA = "head"
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "camera_results"


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _print_payload(label: str, payload: Any) -> None:
    if isinstance(payload, (dict, list)):
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        rendered = str(payload)
    print(f"[{label}]\n{rendered}")


def get_json(
    client: requests.Session,
    url: str,
    *,
    timeout: tuple[float, float],
) -> Any:
    response = client.get(url, timeout=timeout)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as error:
        raise RuntimeError(f"{url} 返回的不是有效 JSON") from error


def save_snapshot(
    client: requests.Session,
    base_url: str,
    camera: str,
    output_path: Path,
    *,
    timeout: tuple[float, float],
) -> int:
    response = client.get(
        _url(base_url, "/camera/snapshot"),
        params={"camera": camera, "type": "color"},
        timeout=timeout,
    )
    response.raise_for_status()
    if not response.content:
        raise RuntimeError("RGB 快照响应为空")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    content_type = response.headers.get("content-type", "unknown")
    print(
        f"[snapshot] camera={camera} type=color "
        f"content-type={content_type} bytes={len(response.content)} "
        f"saved={output_path.resolve()}"
    )
    return len(response.content)


def save_first_stream_frame(
    client: requests.Session,
    base_url: str,
    camera: str,
    stream_type: str,
    output_path: Path,
    *,
    timeout: tuple[float, float],
    max_bytes: int,
) -> int:
    """读取流中的第一帧；JPEG 流按 SOI/EOI 截取，其他流保存首个数据块。"""

    response = client.get(
        _url(base_url, "/camera/stream"),
        params={"camera": camera, "type": stream_type},
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()
    data = bytearray()
    try:
        for chunk in response.iter_content(chunk_size=4 * 1024):
            if not chunk:
                continue
            data.extend(chunk)
            if len(data) >= max_bytes:
                break
            # 相机服务通常返回连续 JPEG 帧；拿到第一帧即可证明接口可用。
            if stream_type == "color" and b"\xff\xd9" in data:
                break
    finally:
        response.close()

    if not data:
        raise RuntimeError("视频流没有返回数据")

    raw = bytes(data)
    if stream_type == "color":
        start = raw.find(b"\xff\xd8")
        end = raw.find(b"\xff\xd9", start + 2) if start >= 0 else -1
        if start >= 0 and end >= 0:
            raw = raw[start : end + 2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    content_type = response.headers.get("content-type", "unknown")
    print(
        f"[stream] camera={camera} type={stream_type} "
        f"content-type={content_type} bytes={len(raw)} "
        f"saved={output_path.resolve()}"
    )
    return len(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断 8085 相机接口")
    parser.add_argument(
        "--url",
        default=DEFAULT_CAMERA_URL,
        help=f"相机服务地址，默认：{DEFAULT_CAMERA_URL}",
    )
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="相机名称，默认：head")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"输出目录，默认：{DEFAULT_OUTPUT_DIRECTORY}",
    )
    parser.add_argument("--connect-timeout", type=float, default=5.0)
    parser.add_argument("--read-timeout", type=float, default=15.0)
    parser.add_argument(
        "--skip-stream",
        action="store_true",
        help="只检查 health/list 和 RGB 快照，不读取长连接视频流",
    )
    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="跳过 RGB 快照，只检查 health/list 和视频流",
    )
    parser.add_argument(
        "--stream-type",
        choices=("color", "depth"),
        default="color",
        help="要验证的流类型，默认：color",
    )
    parser.add_argument(
        "--max-stream-bytes",
        type=int,
        default=4 * 1024 * 1024,
        help="最多读取的流字节数，默认 4 MiB",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timeout = (args.connect_timeout, args.read_timeout)
    base_url = args.url.rstrip("/")

    try:
        with requests.Session() as client:
            _print_payload("health", get_json(client, _url(base_url, "/camera/health"), timeout=timeout))
            _print_payload("list", get_json(client, _url(base_url, "/camera/list"), timeout=timeout))
            if not args.skip_snapshot:
                save_snapshot(
                    client,
                    base_url,
                    args.camera,
                    args.output_dir / f"{args.camera}.jpg",
                    timeout=timeout,
                )
            if not args.skip_stream:
                suffix = "jpg" if args.stream_type == "color" else "bin"
                save_first_stream_frame(
                    client,
                    base_url,
                    args.camera,
                    args.stream_type,
                    args.output_dir / f"{args.camera}_{args.stream_type}_stream.{suffix}",
                    timeout=timeout,
                    max_bytes=args.max_stream_bytes,
                )
    except (requests.RequestException, RuntimeError, OSError) as error:
        print(f"相机接口检查失败: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
