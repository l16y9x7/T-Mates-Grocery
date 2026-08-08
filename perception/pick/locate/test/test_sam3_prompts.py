from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import requests


LOCATE_ROOT = Path(__file__).resolve().parents[1]
PERCEPTION_ROOT = LOCATE_ROOT.parents[1]
SKU_ROOT = PERCEPTION_ROOT / "sku"
CATALOG_PATH = SKU_ROOT / "products.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "sam3_prompt_test_results.json"
DEFAULT_SAM3_URL = os.getenv(
    "SAM3_URL",
    "http://211.137.21.33:25541/api/v1/segment",
)
PROMPT_PATHS = {
    "SORTING": LOCATE_ROOT / "qwen_sam_prompt_mapping.json",
    "SHORTAGE": LOCATE_ROOT / "qwen_sam_prompt_mapping_shortage.json",
    "MISPLACED": LOCATE_ROOT / "qwen_sam_prompt_mapping_misplaced.json",
}


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"文件不存在: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"读取 JSON 失败 {path}: {error}") from error


def load_products() -> list[dict[str, Any]]:
    payload = load_json(CATALOG_PATH)
    products = payload.get("products") if isinstance(payload, dict) else None
    if not isinstance(products, list):
        raise RuntimeError("SKU products.json 缺少 products 数组")
    return [product for product in products if isinstance(product, dict)]


def resolve_sku_image(relative_path: str) -> Path:
    posix_path = PurePosixPath(relative_path.replace("\\", "/"))
    image_path = SKU_ROOT.joinpath(*posix_path.parts).resolve()
    try:
        image_path.relative_to(SKU_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"SKU 图片路径越界: {relative_path}") from error
    return image_path


def test_one_prompt(
    *,
    task_type: str,
    product: dict[str, Any],
    prompt_mapping: dict[str, Any],
    sam3_url: str,
    timeout: float,
) -> dict[str, Any]:
    name = str(product.get("name", "")).strip()
    sku_id = str(product.get("sku_id", "")).strip()
    pair = prompt_mapping.get(name)
    prompt = pair.get("sam3_prompt", "") if isinstance(pair, dict) else ""
    prompt = prompt.strip() if isinstance(prompt, str) else ""
    result: dict[str, Any] = {
        "task_type": task_type,
        "sku_id": sku_id,
        "name": name,
        "sam3_prompt": prompt,
        "images": [],
        "instance_count": 0,
    }
    if not prompt:
        result["status"] = "empty_prompt"
        result["error"] = "sam3_prompt 为空或商品 key 不存在"
        return result

    image_names = product.get("images")
    if not isinstance(image_names, list) or not image_names:
        result["status"] = "missing_image"
        result["error"] = "SKU 没有参考图片"
        return result

    request_errors: list[str] = []
    invalid_responses: list[str] = []
    for relative_path in image_names:
        if not isinstance(relative_path, str):
            invalid_responses.append("SKU 图片路径不是字符串")
            continue
        try:
            image_path = resolve_sku_image(relative_path)
        except RuntimeError as error:
            request_errors.append(str(error))
            continue
        if not image_path.is_file():
            request_errors.append(f"图片不存在: {image_path}")
            continue

        image_result: dict[str, Any] = {"image": relative_path}
        try:
            with image_path.open("rb") as image_file:
                response = requests.post(
                    sam3_url,
                    files={"image": (image_path.name, image_file)},
                    # 不发送 threshold/mask_threshold，使用 SAM3 服务端默认阈值。
                    data={"prompt": prompt},
                    timeout=timeout,
                )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            message = f"{relative_path}: {error}"
            request_errors.append(message)
            image_result["error"] = message
            result["images"].append(image_result)
            continue
        except ValueError as error:
            message = f"{relative_path}: SAM3 响应不是有效 JSON: {error}"
            invalid_responses.append(message)
            image_result["error"] = message
            result["images"].append(image_result)
            continue

        instances = payload.get("instances") if isinstance(payload, dict) else None
        if not isinstance(instances, list):
            message = f"{relative_path}: SAM3 响应缺少 instances 数组"
            invalid_responses.append(message)
            image_result["error"] = message
            result["images"].append(image_result)
            continue
        image_result["instance_count"] = len(instances)
        result["instance_count"] += len(instances)
        result["images"].append(image_result)

    if result["instance_count"] > 0:
        result["status"] = "detected"
    elif request_errors and not result["images"]:
        result["status"] = "request_error"
        result["error"] = "; ".join(request_errors)
    elif invalid_responses:
        result["status"] = "invalid_response"
        result["error"] = "; ".join(invalid_responses)
    elif request_errors:
        result["status"] = "request_error"
        result["error"] = "; ".join(request_errors)
    else:
        result["status"] = "no_detection"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 SKU 参考图片验证三类 task_type 的 SAM3 Prompt",
    )
    parser.add_argument("--sam3-url", default=DEFAULT_SAM3_URL)
    parser.add_argument(
        "--task-type",
        action="append",
        choices=tuple(PROMPT_PATHS),
        help="仅测试指定类型；可重复传入。默认测试三种类型",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers 必须大于等于 1")

    task_types = args.task_type or list(PROMPT_PATHS)
    products = load_products()
    prompt_mappings = {
        task_type: load_json(PROMPT_PATHS[task_type]) for task_type in task_types
    }
    jobs = [
        (task_type, product)
        for task_type in task_types
        for product in products
    ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                test_one_prompt,
                task_type=task_type,
                product=product,
                prompt_mapping=prompt_mappings[task_type],
                sam3_url=args.sam3_url,
                timeout=args.timeout,
            ): (task_type, product)
            for task_type, product in jobs
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if result["status"] != "detected":
                print(
                    f"[{result['status'].upper()}] "
                    f"{result['task_type']} / {result['sku_id']} / {result['name']}"
                )
            print(f"\r进度: {completed}/{len(jobs)}", end="", flush=True)
    print()

    task_order = {task_type: index for index, task_type in enumerate(task_types)}
    sku_order = {
        str(product.get("sku_id", "")): index for index, product in enumerate(products)
    }
    results.sort(
        key=lambda item: (
            task_order[item["task_type"]],
            sku_order.get(item["sku_id"], 10**9),
        )
    )
    failures = [result for result in results if result["status"] != "detected"]
    summary = {
        task_type: {
            status: sum(
                result["task_type"] == task_type and result["status"] == status
                for result in results
            )
            for status in (
                "detected",
                "no_detection",
                "empty_prompt",
                "missing_image",
                "request_error",
                "invalid_response",
            )
        }
        for task_type in task_types
    }
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "sam3_url": args.sam3_url,
        "threshold": "SAM3 server default (threshold fields were not sent)",
        "summary": summary,
        "failure_count": len(failures),
        "failures": failures,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"未检测成功: {len(failures)} / {len(results)}")
    print(f"完整报告: {args.output.resolve()}")


if __name__ == "__main__":
    main()
