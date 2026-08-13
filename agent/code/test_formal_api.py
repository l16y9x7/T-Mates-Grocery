from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIRECTORY = ROOT.parents[1] / "test_data" / "2026-08-04"
DEFAULT_IMAGE_MAPPING_PATH = DEFAULT_IMAGE_DIRECTORY / "image_name_mapping.json"
SKU_API_URL = os.getenv("SKU_API_URL", "http://192.168.130.59:25540").rstrip("/")
LOCATE_API_URL = os.getenv("LOCATE_API_URL", "http://192.168.130.59:8083").rstrip("/")
SKU_REQUEST_TIMEOUT_SECONDS = float(os.getenv("SKU_REQUEST_TIMEOUT_SECONDS", "120"))
LOCATE_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("LOCATE_REQUEST_TIMEOUT_SECONDS", "600")
)


def lookup_sku(product_name: str) -> dict[str, Any]:
    try:
        response = requests.get(
            f"{SKU_API_URL}/sku/search_by_name",
            params={"name": product_name},
            timeout=SKU_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise RuntimeError(f"SKU API 请求失败: {error}") from error

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError("SKU API 响应不是有效 JSON") from error
    if not response.ok:
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        raise RuntimeError(f"SKU API 返回 {response.status_code}: {detail}")
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("sku_id"), str)
        or not isinstance(payload.get("name"), str)
    ):
        raise RuntimeError("SKU API 响应缺少 sku_id 或 name")
    return payload


def load_image_mapping(path: Path = DEFAULT_IMAGE_MAPPING_PATH) -> dict[str, list[str]]:
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"图片映射文件不存在: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"读取图片映射失败: {error}") from error
    if not isinstance(mapping, dict):
        raise RuntimeError("图片映射必须是 JSON 对象")
    for image_name, sku_ids in mapping.items():
        if (
            not isinstance(image_name, str)
            or Path(image_name).name != image_name
            or not isinstance(sku_ids, list)
            or not all(isinstance(sku_id, str) for sku_id in sku_ids)
        ):
            raise RuntimeError("图片映射格式必须是 image_name 到 SKU ID 列表的映射")
    return mapping


def find_local_images(
    sku_id: str,
    mapping_path: Path = DEFAULT_IMAGE_MAPPING_PATH,
    image_directory: Path = DEFAULT_IMAGE_DIRECTORY,
) -> list[Path]:
    normalized_sku_id = sku_id.strip().upper()
    mapping = load_image_mapping(mapping_path)
    images = [
        image_directory / image_name
        for image_name, sku_ids in mapping.items()
        if normalized_sku_id in {value.strip().upper() for value in sku_ids}
    ]
    if not images:
        raise RuntimeError(f"图片映射中没有找到 {normalized_sku_id}")
    missing = [path.name for path in images if not path.is_file()]
    if missing:
        raise RuntimeError(f"映射中的本地图片不存在: {', '.join(missing)}")
    return images


def validate_formal_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("Locate API 响应必须是 JSON 对象")
    expected_fields = {"product_name", "bbox", "mask", "image_path"}
    if set(payload) != expected_fields:
        raise RuntimeError(
            "Locate API 响应字段不符合正式接口: "
            f"期望 {sorted(expected_fields)}，实际 {sorted(payload)}"
        )

    bbox = payload.get("bbox")
    if (
        not isinstance(payload.get("product_name"), str)
        or not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(isinstance(value, (int, float)) and 1 <= value <= 1000 for value in bbox)
        or not isinstance(payload.get("mask"), str)
        or not payload["mask"]
        or not isinstance(payload.get("image_path"), str)
        or not payload["image_path"]
        or not Path(payload["image_path"]).is_absolute()
    ):
        raise RuntimeError("Locate API 正式响应字段类型或取值无效")
    try:
        base64.b64decode(payload["mask"].split(",", 1)[-1], validate=True)
    except (ValueError, binascii.Error) as error:
        raise RuntimeError("Locate API mask 不是有效 base64") from error
    return payload


def call_formal_locate(
    task_type: str,
    product_name: str,
    hand: str,
    image_path: Path,
) -> dict[str, Any]:
    request_payload = {
        "task_type": task_type,
        "product_name": product_name,
        "hand": hand,
        "image_name": image_path.name,
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
    }
    try:
        response = requests.post(
            f"{LOCATE_API_URL}/perception/pick/locate",
            json=request_payload,
            timeout=LOCATE_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise RuntimeError(f"Locate API 请求失败: {error}") from error
    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError("Locate API 响应不是有效 JSON") from error
    if not response.ok:
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        raise RuntimeError(f"Locate API 返回 {response.status_code}: {detail}")
    return validate_formal_response(payload)


def run_formal_test(task_type: str, product_name: str, hand: str) -> dict[str, Any]:
    normalized_inputs = [value.strip() for value in (task_type, product_name, hand)]
    if not all(normalized_inputs):
        raise RuntimeError("task_type、product_name、hand 都不能为空")
    normalized_task_type, normalized_product_name, normalized_hand = normalized_inputs

    product = lookup_sku(normalized_product_name)
    image_paths = find_local_images(product["sku_id"])
    results: dict[str, Any] = {}
    for image_path in image_paths:
        results[image_path.name] = call_formal_locate(
            normalized_task_type,
            product["name"],
            normalized_hand,
            image_path,
        )
    return {
        "input": {
            "task_type": normalized_task_type,
            "product_name": normalized_product_name,
            "hand": normalized_hand,
        },
        "sku_id": product["sku_id"],
        "matched_images": [path.name for path in image_paths],
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只输入 task_type、product_name、hand，自动匹配本地图片并测试正式 Locate API"
    )
    parser.add_argument("task_type", help="请求的 task_type 字段")
    parser.add_argument("product_name", help="完整商品名称")
    parser.add_argument("hand", help="请求的 hand 字段，例如 left 或 right")
    parser.add_argument("--output", type=Path, help="可选 JSON 结果保存路径")
    return parser.parse_args()


def main_cli() -> None:
    args = parse_args()
    try:
        result = run_formal_test(args.task_type, args.product_name, args.hand)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
        return
    args.output.write_text(serialized, encoding="utf-8")
    print(f"正式接口测试结果已保存到: {args.output.resolve()}")


if __name__ == "__main__":
    main_cli()
