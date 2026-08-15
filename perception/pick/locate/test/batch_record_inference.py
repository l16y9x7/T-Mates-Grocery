from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
PERCEPTION_ROOT = ROOT.parents[2]
DEFAULT_MAPPING_PATH = (
    PERCEPTION_ROOT / "test_data" / "2026-08-13" / "sorting_pick_locate_batch.json"
)
DEFAULT_API_URL = os.getenv("LOCATE_API_URL", "http://127.0.0.1:8083").rstrip("/")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("LOCATE_REQUEST_TIMEOUT_SECONDS", "600"))
DEFAULT_WORKERS = int(os.getenv("LOCATE_BATCH_WORKERS", "4"))
INVALID_WINDOWS_FILENAME_CHARS = set('<>:"/\\|?*')
LOCATION_LEVEL_PATTERN = re.compile(r"_L([1-5])_")

# These records contain SKUs stocked on more than one level. Their level is
# fixed by the capture sequence: paired left/right wrist records observe the
# same shelf level, while the next pair observes the next level.
RECORD_LEVEL_OVERRIDES = {
    "record_20260813_081100_910887": "L2",
    "record_20260813_083528_841922": "L4",
    "record_20260813_094432_071832": "L4",
    "record_20260813_094447_638495": "L4",
    "record_20260813_094539_955139": "L5",
    "record_20260813_094546_000550": "L5",
    "record_20260813_094957_167471": "L5",
    "record_20260813_095209_878438": "L4",
}


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"文件不存在: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"读取 JSON 失败 {path}: {error}") from error


def write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_catalog() -> dict[str, dict[str, Any]]:
    payload = load_json(PERCEPTION_ROOT / "sku" / "products.json")
    products = payload.get("products") if isinstance(payload, dict) else payload
    if not isinstance(products, list):
        raise RuntimeError("SKU products.json 格式错误")
    by_name = {
        product["name"]: product
        for product in products
        if isinstance(product, dict) and isinstance(product.get("name"), str)
    }
    return by_name


def product_levels(product: dict[str, Any]) -> set[str]:
    levels: set[str] = set()
    locations = product.get("locations")
    if not isinstance(locations, list):
        return levels
    for location in locations:
        if not isinstance(location, str):
            continue
        match = LOCATION_LEVEL_PATTERN.search(location.upper())
        if match is not None:
            levels.add(f"L{match.group(1)}")
    return levels


def infer_record_level(
    record_name: str,
    product_names: list[str],
    catalog: dict[str, dict[str, Any]],
) -> str:
    override = RECORD_LEVEL_OVERRIDES.get(record_name)
    if override is not None:
        return override
    level_sets = [product_levels(catalog[name]) for name in product_names]
    if not level_sets or any(not levels for levels in level_sets):
        raise RuntimeError(f"{record_name} 无法从 SKU locations 推导层级")
    common_levels = set.intersection(*level_sets)
    if len(common_levels) != 1:
        raise RuntimeError(
            f"{record_name} 层级不唯一: {sorted(common_levels)}，请增加 override"
        )
    return next(iter(common_levels))


def infer_record_hand(record_directory: Path) -> str:
    state = load_json(record_directory / "robot_state.json")
    camera = state.get("camera") if isinstance(state, dict) else None
    camera_id = camera.get("id") if isinstance(camera, dict) else None
    if not isinstance(camera_id, str):
        raise RuntimeError(f"{record_directory.name} robot_state 缺少 camera.id")
    normalized = camera_id.strip().lower()
    if normalized.startswith("left"):
        return "left"
    if normalized.startswith("right"):
        return "right"
    raise RuntimeError(f"{record_directory.name} 无法识别相机手腕: {camera_id}")


def validate_output_name(product_name: str) -> None:
    if (
        not product_name
        or product_name in {".", ".."}
        or any(character in INVALID_WINDOWS_FILENAME_CHARS for character in product_name)
    ):
        raise RuntimeError(f"商品名不能安全用作结果文件名: {product_name}")


def load_and_validate_mapping(
    mapping_path: Path,
) -> tuple[Path, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = load_json(mapping_path)
    if not isinstance(payload, dict) or payload.get("task_type") != "SORTING":
        raise RuntimeError("映射文件 task_type 必须是 SORTING")
    entries = payload.get("records")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("映射文件缺少 records")
    record_root_value = payload.get("record_root", ".")
    if not isinstance(record_root_value, str):
        raise RuntimeError("record_root 必须是字符串")
    record_root = (mapping_path.parent / record_root_value).resolve()
    catalog = load_catalog()
    prompt_mapping = load_json(
        PERCEPTION_ROOT / "pick" / "locate" / "qwen_sam_prompt_mapping.json"
    )
    if not isinstance(prompt_mapping, dict):
        raise RuntimeError("SORTING Prompt 映射格式错误")

    seen_records: set[str] = set()
    validated: list[dict[str, Any]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise RuntimeError("record 条目必须是对象")
        record_name = raw_entry.get("record")
        product_names = raw_entry.get("product_names")
        if (
            not isinstance(record_name, str)
            or not record_name.startswith("record_")
            or Path(record_name).name != record_name
            or record_name in seen_records
        ):
            raise RuntimeError(f"record 名称无效或重复: {record_name}")
        if (
            not isinstance(product_names, list)
            or not product_names
            or not all(isinstance(name, str) and name.strip() for name in product_names)
        ):
            raise RuntimeError(f"{record_name} product_names 格式错误")
        normalized_names = [name.strip() for name in product_names]
        if len(set(normalized_names)) != len(normalized_names):
            raise RuntimeError(f"{record_name} 存在重复商品名")
        for product_name in normalized_names:
            validate_output_name(product_name)
            if product_name not in catalog:
                raise RuntimeError(f"SKU 表不存在商品: {product_name}")
            if product_name not in prompt_mapping:
                raise RuntimeError(f"SORTING Prompt 未配置商品: {product_name}")

        record_directory = record_root / record_name
        rgb_path = record_directory / str(payload.get("rgb_file", "rgb.jpg"))
        depth_path = record_directory / str(payload.get("depth_file", "depth_mm.npy"))
        if not rgb_path.is_file() or not depth_path.is_file():
            raise RuntimeError(f"{record_name} 缺少 RGB 或深度文件")
        hand = raw_entry.get("hand") or infer_record_hand(record_directory)
        level = raw_entry.get("level") or infer_record_level(
            record_name,
            normalized_names,
            catalog,
        )
        if hand not in {"left", "right"} or not re.fullmatch(r"L[1-5]", level):
            raise RuntimeError(f"{record_name} 的 hand/level 无效: {hand}/{level}")
        seen_records.add(record_name)
        validated.append(
            {
                "record": record_name,
                "record_directory": record_directory,
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "hand": hand,
                "level": level,
                "product_names": normalized_names,
            }
        )
    return record_root, validated, catalog


def instance_summary(instance: Any) -> dict[str, Any] | None:
    if not isinstance(instance, dict):
        return None
    return {
        key: instance.get(key)
        for key in (
            "bbox",
            "score",
            "depth_mm",
            "source_qwen_index",
            "hard_case_group_index",
            "mapped_product_name",
            "is_selected",
        )
    }


def compact_response(payload: dict[str, Any]) -> dict[str, Any]:
    selected = instance_summary(payload.get("selected_instance"))
    return {
        "sku_id": payload.get("sku_id"),
        "product_name": payload.get("product_name"),
        "image_name": payload.get("image_name"),
        "image_size": payload.get("image_size"),
        "inference_image_size": payload.get("inference_image_size"),
        "qwen3_prompt_used": payload.get("qwen3_prompt_used"),
        "sam3_prompt_used": payload.get("sam3_prompt_used"),
        "raw_qwen_bboxes": payload.get("raw_qwen_bboxes", []),
        "qwen_bboxes": payload.get("qwen_bboxes", []),
        "raw_sam_instances": [
            summary
            for item in payload.get("raw_sam_instances", [])
            if (summary := instance_summary(item)) is not None
        ],
        "instances": [
            summary
            for item in payload.get("instances", [])
            if (summary := instance_summary(item)) is not None
        ],
        "selected_instance_index": payload.get("selected_instance_index"),
        "selected_instance": selected,
        "hard_case": payload.get("hard_case"),
        "error": payload.get("error"),
        "error_status_code": payload.get("error_status_code"),
    }


def normalized_bbox(bbox: Any, image_size: Any) -> list[int] | None:
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(isinstance(value, (int, float)) for value in bbox)
        or not isinstance(image_size, list)
        or len(image_size) != 2
    ):
        return None
    width, height = image_size
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        return None
    scales = (width, height, width, height)
    return [
        max(1, min(1000, round(1 + max(0.0, min(scale, value)) / scale * 999)))
        for value, scale in zip(bbox, scales)
    ]


def draw_result(
    rgb_path: Path,
    payload: dict[str, Any],
    output_path: Path,
    *,
    status: str,
) -> None:
    try:
        with Image.open(rgb_path) as source:
            canvas = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise RuntimeError(f"读取 RGB 结果图失败: {error}") from error
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(canvas.width / 320))
    instances = payload.get("instances")
    if not isinstance(instances, list):
        instances = []
    for index, instance in enumerate(instances, start=1):
        if not isinstance(instance, dict):
            continue
        bbox = instance.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            continue
        box = tuple(round(float(value)) for value in bbox)
        draw.rectangle(box, outline="#2dd4bf", width=line_width)
        depth = instance.get("depth_mm")
        depth_label = "n/a" if not isinstance(depth, (int, float)) else f"{depth:.0f}mm"
        group = instance.get("hard_case_group_index")
        group_label = "" if not isinstance(group, int) else f" G{group}"
        label = f"#{index}{group_label} {depth_label}"
        draw.text(
            (box[0] + 2, max(0, box[1] - 13)),
            label,
            fill="#2dd4bf",
            stroke_width=2,
            stroke_fill="black",
        )

    selected = payload.get("selected_instance")
    selected_bbox = selected.get("bbox") if isinstance(selected, dict) else None
    if (
        isinstance(selected_bbox, list)
        and len(selected_bbox) == 4
        and all(isinstance(value, (int, float)) for value in selected_bbox)
    ):
        box = tuple(round(float(value)) for value in selected_bbox)
        draw.rectangle(box, outline="#ef4444", width=max(6, line_width * 3))
        depth = selected.get("depth_mm")
        depth_label = "" if not isinstance(depth, (int, float)) else f" {depth:.0f}mm"
        draw.text(
            (box[0] + 2, box[1] + 2),
            f"PICK{depth_label}",
            fill="#ef4444",
            stroke_width=3,
            stroke_fill="white",
        )
    elif status != "success":
        qwen_bboxes = payload.get("qwen_bboxes")
        if isinstance(qwen_bboxes, list):
            for record in qwen_bboxes:
                bbox = record.get("bbox_original") if isinstance(record, dict) else None
                if isinstance(bbox, list) and len(bbox) == 4:
                    draw.rectangle(
                        tuple(round(float(value)) for value in bbox),
                        outline="#f59e0b",
                        width=line_width,
                    )
        draw.rectangle((0, 0, canvas.width - 1, canvas.height - 1), outline="#ef4444", width=6)
        draw.text((8, 8), "ERROR", fill="#ef4444", stroke_width=2, stroke_fill="white")
    canvas.save(output_path, format="JPEG", quality=95)


def request_locate(
    api_url: str,
    request_payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> tuple[int, dict[str, Any]]:
    response = requests.post(
        f"{api_url}/perception/pick/locate/debug",
        json=request_payload,
        timeout=timeout_seconds,
    )
    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(f"Locate 返回非 JSON: HTTP {response.status_code}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Locate 返回值不是 JSON 对象")
    return response.status_code, payload


def run_batch_job(
    *,
    job_number: int,
    total: int,
    entry: dict[str, Any],
    product_name: str,
    sku_id: Any,
    rgb_base64: str,
    depth_base64: str,
    api_url: str,
    timeout_seconds: float,
    retries: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Run one independent record/product inference and persist its item files."""
    output_image_path = entry["record_directory"] / f"{product_name}.jpg"
    output_json_path = entry["record_directory"] / f"{product_name}.json"
    if not overwrite and output_image_path.is_file() and output_json_path.is_file():
        try:
            existing_result = load_json(output_json_path)
        except RuntimeError:
            existing_result = None
        if (
            isinstance(existing_result, dict)
            and existing_result.get("status") == "success"
        ):
            return {
                "record": entry["record"],
                "product_name": product_name,
                "skipped": True,
                "success": True,
                "summary": {
                    key: value
                    for key, value in existing_result.items()
                    if key != "response"
                },
                "is_system_error": False,
                "combined_error": "",
                "elapsed_seconds": 0.0,
            }

    request_payload = {
        "task_type": "SORTING",
        "product_name": product_name,
        "level": entry["level"],
        "hand": entry["hand"],
        "image_name": "rgb.jpg",
        "image_base64": rgb_base64,
        "depth_image_name": "depth_mm.npy",
        "depth_image_base64": depth_base64,
    }
    started = time.perf_counter()
    response_payload: dict[str, Any] = {}
    http_status = 0
    request_error: str | None = None
    attempts = 0
    for attempt in range(max(0, retries) + 1):
        attempts = attempt + 1
        try:
            http_status, response_payload = request_locate(
                api_url,
                request_payload,
                timeout_seconds=timeout_seconds,
            )
            request_error = None
        except (requests.RequestException, RuntimeError) as error:
            request_error = str(error)
            response_payload = {}
        inference_error = response_payload.get("error")
        selected = response_payload.get("selected_instance")
        if (
            request_error is None
            and 200 <= http_status < 300
            and not inference_error
            and isinstance(selected, dict)
        ):
            break
        if attempt < max(0, retries):
            print(
                f"[{job_number}/{total}] RETRY {entry['record']} {product_name} "
                f"attempt={attempts}",
                flush=True,
            )

    elapsed = round(time.perf_counter() - started, 3)
    selected = response_payload.get("selected_instance")
    inference_error = response_payload.get("error")
    success = (
        request_error is None
        and 200 <= http_status < 300
        and not inference_error
        and isinstance(selected, dict)
    )
    status = "success" if success else "error"
    draw_result(
        entry["rgb_path"],
        response_payload,
        output_image_path,
        status=status,
    )
    compact = compact_response(response_payload)
    selected_bbox = selected.get("bbox") if isinstance(selected, dict) else None
    item_result = {
        "record": entry["record"],
        "product_name": product_name,
        "sku_id": sku_id,
        "level": entry["level"],
        "hand": entry["hand"],
        "status": status,
        "http_status": http_status,
        "attempts": attempts,
        "elapsed_seconds": elapsed,
        "error": request_error or inference_error,
        "error_status_code": response_payload.get("error_status_code"),
        "selected_bbox_pixel": selected_bbox,
        "selected_bbox_normalized": normalized_bbox(
            selected_bbox,
            response_payload.get("image_size"),
        ),
        "selected_depth_mm": (
            selected.get("depth_mm") if isinstance(selected, dict) else None
        ),
        "output_image": str(output_image_path.resolve()),
        "output_json": str(output_json_path.resolve()),
        "response": compact,
    }
    write_json(output_json_path, item_result)
    combined_error = str(request_error or inference_error or "")
    is_system_error = request_error is not None or any(
        marker in combined_error
        for marker in (
            "WinError 10013",
            "Failed to establish a new connection",
            "Locate API 请求失败",
        )
    )
    return {
        "record": entry["record"],
        "product_name": product_name,
        "skipped": False,
        "success": success,
        "summary": {
            key: value for key, value in item_result.items() if key != "response"
        },
        "is_system_error": is_system_error,
        "combined_error": combined_error,
        "elapsed_seconds": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 record→商品映射批量运行 SORTING pick/locate RGB+depth 测试"
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-consecutive-system-errors", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--record")
    parser.add_argument("--product-name")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise RuntimeError("--workers 必须大于等于 1")
    mapping_path = args.mapping.resolve()
    _, all_entries, catalog = load_and_validate_mapping(mapping_path)
    targeted = args.record is not None or args.product_name is not None
    if targeted and (not args.record or not args.product_name):
        raise RuntimeError("--record 和 --product-name 必须同时提供")
    entries = all_entries
    if targeted:
        entries = [
            {
                **entry,
                "product_names": [args.product_name],
            }
            for entry in all_entries
            if entry["record"] == args.record
            and args.product_name in entry["product_names"]
        ]
        if not entries:
            raise RuntimeError(
                f"映射中不存在目标组合: {args.record} / {args.product_name}"
            )
    total = sum(len(entry["product_names"]) for entry in entries)
    print(f"validated records={len(entries)} detections={total}", flush=True)
    for entry in entries:
        print(
            f"MAP {entry['record']} {entry['level']} {entry['hand']} "
            f"{','.join(entry['product_names'])}",
            flush=True,
        )
    if args.dry_run:
        return

    batch_result_path = mapping_path.with_name("sorting_pick_locate_batch_results.json")
    if targeted and batch_result_path.is_file():
        batch_results = load_json(batch_result_path)
        if not isinstance(batch_results.get("results"), list):
            raise RuntimeError("现有批测汇总缺少 results 数组")
        batch_results["api_url"] = args.api_url
        batch_results["workers"] = args.workers
        batch_results["started_at_unix"] = time.time()
        batch_results["last_targeted_run"] = {
            "record": args.record,
            "product_name": args.product_name,
        }
        batch_results.pop("finished_at_unix", None)
        batch_results.pop("aborted_reason", None)
    else:
        batch_results = {
            "schema_version": 1,
            "mapping_file": str(mapping_path),
            "api_url": args.api_url,
            "workers": args.workers,
            "task_type": "SORTING",
            "started_at_unix": time.time(),
            "total_records": len(all_entries),
            "total_detections": sum(
                len(entry["product_names"]) for entry in all_entries
            ),
            "completed": 0,
            "successes": 0,
            "failures": 0,
            "skipped": 0,
            "results": [],
        }
    jobs: list[dict[str, Any]] = []
    result_order: dict[tuple[str, str], int] = {}
    order_index = 0
    for all_entry in all_entries:
        for all_product_name in all_entry["product_names"]:
            result_order[(all_entry["record"], all_product_name)] = order_index
            order_index += 1
    for entry in entries:
        rgb_base64 = base64.b64encode(entry["rgb_path"].read_bytes()).decode("ascii")
        depth_base64 = base64.b64encode(entry["depth_path"].read_bytes()).decode("ascii")
        for product_name in entry["product_names"]:
            jobs.append(
                {
                    "job_number": len(jobs) + 1,
                    "entry": entry,
                    "product_name": product_name,
                    "sku_id": catalog[product_name].get("sku_id"),
                    "rgb_base64": rgb_base64,
                    "depth_base64": depth_base64,
                }
            )

    completed = 0
    consecutive_system_errors = 0
    print(f"running workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_jobs = {
            executor.submit(
                run_batch_job,
                job_number=job["job_number"],
                total=total,
                entry=job["entry"],
                product_name=job["product_name"],
                sku_id=job["sku_id"],
                rgb_base64=job["rgb_base64"],
                depth_base64=job["depth_base64"],
                api_url=args.api_url,
                timeout_seconds=args.timeout,
                retries=args.retries,
                overwrite=args.overwrite,
            ): job
            for job in jobs
        }
        for future in as_completed(future_jobs):
            completed += 1
            try:
                result = future.result()
            except Exception as error:
                abort_reason = f"批测 worker 异常，任务已中止: {error}"
                batch_results["aborted_reason"] = abort_reason
                batch_results["finished_at_unix"] = time.time()
                write_json(batch_result_path, batch_results)
                for pending_future in future_jobs:
                    pending_future.cancel()
                raise RuntimeError(abort_reason) from error
            record = result["record"]
            product_name = result["product_name"]
            if result["skipped"]:
                print(
                    f"[{completed}/{total}] SKIP-SUCCESS {record} {product_name}",
                    flush=True,
                )
                if targeted:
                    continue
                batch_results["successes"] += 1
                batch_results["skipped"] += 1
            elif not targeted:
                if result["success"]:
                    batch_results["successes"] += 1
                else:
                    batch_results["failures"] += 1

            summary_item = result["summary"]
            if targeted and not result["skipped"]:
                existing_index = next(
                    (
                        index
                        for index, existing in enumerate(batch_results["results"])
                        if existing.get("record") == record
                        and existing.get("product_name") == product_name
                    ),
                    None,
                )
                if existing_index is None:
                    batch_results["results"].append(summary_item)
                else:
                    batch_results["results"][existing_index] = summary_item
                batch_results["successes"] = sum(
                    item.get("status") == "success"
                    for item in batch_results["results"]
                )
                batch_results["failures"] = sum(
                    item.get("status") != "success"
                    for item in batch_results["results"]
                )
                batch_results["completed"] = len(batch_results["results"])
            elif not targeted:
                batch_results["results"].append(summary_item)
                batch_results["completed"] = completed

            batch_results["results"].sort(
                key=lambda item: result_order.get(
                    (item.get("record"), item.get("product_name")),
                    len(result_order),
                )
            )
            batch_results["updated_at_unix"] = time.time()
            write_json(batch_result_path, batch_results)
            if not result["skipped"]:
                status = "SUCCESS" if result["success"] else "ERROR"
                print(
                    f"[{completed}/{total}] {status} {record} {product_name} "
                    f"{result['elapsed_seconds']:.3f}s",
                    flush=True,
                )

            if result["skipped"]:
                continue
            consecutive_system_errors = (
                consecutive_system_errors + 1
                if result["is_system_error"]
                else 0
            )
            error_limit = max(1, args.max_consecutive_system_errors)
            if consecutive_system_errors >= error_limit:
                abort_reason = (
                    f"连续 {consecutive_system_errors} 个系统连接错误，批测已中止: "
                    f"{result['combined_error']}"
                )
                batch_results["aborted_reason"] = abort_reason
                batch_results["finished_at_unix"] = time.time()
                write_json(batch_result_path, batch_results)
                for pending_future in future_jobs:
                    pending_future.cancel()
                raise RuntimeError(abort_reason)

    batch_results["finished_at_unix"] = time.time()
    write_json(batch_result_path, batch_results)
    print(
        f"DONE run_total={total} global_success={batch_results['successes']} "
        f"global_failure={batch_results['failures']} skipped={batch_results['skipped']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
