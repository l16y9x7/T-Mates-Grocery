"""调用位姿估计接口：POST /manipulation/pick_pose。
python pick_pose_request.py --rgb test\\2026-08-04\\record_20260804_144405_673341_rgb.jpg --depth test
\\2026-08-04\\record_20260804_144405_673341_depth_mm.png --camera test\\camera.json --mask test\\record_20260804_144405_673341_rgb.png
"""

import argparse
import json
import mimetypes
from contextlib import ExitStack
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="请求目标物体的 6D 位姿")
    parser.add_argument("--url", default="http://192.168.130.59:8084/manipulation/pick_pose", help="接口地址，例如 http://127.0.0.1:8000/manipulation/pick_pose")
    parser.add_argument("--rgb", required=True, type=Path, help="RGB 图像路径")
    parser.add_argument("--depth", required=True, type=Path, help="对齐的深度图路径")
    parser.add_argument("--camera", required=True, type=Path, help="相机参数 JSON 路径")
    parser.add_argument("--mask", required=True, type=Path, help="掩码图路径")
    parser.add_argument("--product-name", help="产品名称（可选）")
    parser.add_argument("--timeout", type=float, default=60, help="超时时间，默认 60 秒")
    args = parser.parse_args()

    paths = {
        "rgb": args.rgb,
        "depth": args.depth,
        "camera": args.camera,
        "mask": args.mask,
    }

    with ExitStack() as stack:
        files = {
            field: (
                path.name,
                stack.enter_context(path.open("rb")),
                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            )
            for field, path in paths.items()
        }
        data = {"product_name": args.product_name} if args.product_name else None
        try:
            with requests.Session() as session:
                session.trust_env = False  # 局域网服务不使用系统代理
                response = session.post(args.url, files=files, data=data, timeout=args.timeout)
        except requests.RequestException as exc:
            raise SystemExit(f"请求失败：{exc}") from exc

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        result = response.text

    if not response.ok:
        print(f"请求失败：HTTP {response.status_code}")
        print(json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, (dict, list)) else result)
        raise SystemExit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
