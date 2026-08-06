from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIRECTORY = ROOT.parents[1] / "test_data" / "2026-08-04"
DEFAULT_IMAGE_MAPPING_PATH = DEFAULT_IMAGE_DIRECTORY / "image_name_mapping.json"
DEFAULT_RESULT_DIRECTORY = DEFAULT_IMAGE_DIRECTORY / "locate_results"
SKU_API_URL = os.getenv("SKU_API_URL", "http://192.168.130.59:25540").rstrip("/")
LOCATE_API_URL = os.getenv("LOCATE_API_URL", "http://192.168.130.59:8081").rstrip("/")
SKU_REQUEST_TIMEOUT_SECONDS = float(os.getenv("SKU_REQUEST_TIMEOUT_SECONDS", "120"))
LOCATE_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("LOCATE_REQUEST_TIMEOUT_SECONDS", "600")
)
MASK_COLORS = [
    (45, 212, 191),
    (245, 158, 11),
    (96, 165, 250),
    (244, 114, 182),
    (167, 139, 250),
    (251, 113, 133),
]
QWEN_BOX_COLORS = [
    (239, 68, 68),
    (37, 99, 235),
    (234, 88, 12),
    (147, 51, 234),
]


def load_image_mapping(path: Path = DEFAULT_IMAGE_MAPPING_PATH) -> dict[str, list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"图片映射文件不存在: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"读取图片映射失败: {error}") from error

    if not isinstance(payload, dict):
        raise RuntimeError("图片映射必须是 JSON 对象")
    mapping: dict[str, list[str]] = {}
    for image_name, sku_ids in payload.items():
        if (
            not isinstance(image_name, str)
            or Path(image_name).name != image_name
            or not isinstance(sku_ids, list)
            or not all(isinstance(sku_id, str) for sku_id in sku_ids)
        ):
            raise RuntimeError("图片映射格式错误，必须是 image_name 到 SKU ID 列表的映射")
        mapping[image_name] = sku_ids
    return mapping


def find_test_images(
    sku_id: str,
    mapping_path: Path = DEFAULT_IMAGE_MAPPING_PATH,
    image_directory: Path = DEFAULT_IMAGE_DIRECTORY,
) -> list[Path]:
    normalized_sku_id = sku_id.strip().upper()
    mapping = load_image_mapping(mapping_path)
    image_paths = [
        image_directory / image_name
        for image_name, sku_ids in mapping.items()
        if normalized_sku_id in (item.strip().upper() for item in sku_ids)
    ]
    if not image_paths:
        raise RuntimeError(f"图片映射中没有找到 {normalized_sku_id}")

    missing_images = [path.name for path in image_paths if not path.is_file()]
    if missing_images:
        raise RuntimeError(f"映射中的 RGB 图片不存在: {', '.join(missing_images)}")
    return image_paths


def lookup_sku_by_name(name: str) -> dict[str, Any]:
    try:
        response = requests.get(
            f"{SKU_API_URL}/sku/search_by_name",
            params={"name": name},
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


def save_result_visualization(
    image_path: Path,
    response_payload: dict[str, Any],
    output_directory: Path,
    sku_id: str,
) -> Path:
    instances = response_payload.get("instances")
    if not isinstance(instances, list):
        raise RuntimeError("Locate 响应缺少 instances 数组")
    try:
        with Image.open(image_path) as source_image:
            canvas = source_image.convert("RGBA")
    except (UnidentifiedImageError, OSError) as error:
        raise RuntimeError(f"读取结果原图失败: {error}") from error

    line_width = max(3, round(canvas.width / 420))
    for index, instance in enumerate(instances, start=1):
        if not isinstance(instance, dict):
            continue
        bbox = instance.get("bbox")
        mask_base64 = instance.get("mask")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
            or not isinstance(mask_base64, str)
        ):
            continue

        try:
            mask_bytes = base64.b64decode(mask_base64.split(",", 1)[-1], validate=True)
            with Image.open(io.BytesIO(mask_bytes)) as source_mask:
                mask = source_mask.convert("L")
        except (ValueError, OSError, UnidentifiedImageError):
            continue
        if mask.size != canvas.size:
            mask = mask.resize(canvas.size, resample=Image.Resampling.NEAREST)

        color = MASK_COLORS[(index - 1) % len(MASK_COLORS)]
        alpha = mask.point(lambda value: 96 if value > 127 else 0)
        overlay = Image.new("RGBA", canvas.size, (*color, 0))
        overlay.putalpha(alpha)
        canvas = Image.alpha_composite(canvas, overlay)

        draw = ImageDraw.Draw(canvas)
        x1, y1, x2, y2 = (round(float(value)) for value in bbox)
        draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=line_width)
        score = instance.get("score")
        label = f"#{index}" if not isinstance(score, (int, float)) else f"#{index} {score:.3f}"
        text_box = draw.textbbox((x1, y1), label)
        label_height = text_box[3] - text_box[1] + 6
        label_width = text_box[2] - text_box[0] + 8
        label_y = max(0, y1 - label_height)
        draw.rectangle(
            (x1, label_y, x1 + label_width, label_y + label_height),
            fill=(*color, 255),
        )
        draw.text((x1 + 4, label_y + 3), label, fill=(7, 16, 23, 255))

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{image_path.stem}_{sku_id}_locate.png"
    canvas.convert("RGB").save(output_path, format="PNG")
    return output_path


def save_qwen_visualization(
    image_path: Path,
    response_payload: dict[str, Any],
    output_directory: Path,
    sku_id: str,
) -> Path:
    qwen_bboxes = response_payload.get("qwen_bboxes")
    if not isinstance(qwen_bboxes, list):
        raise RuntimeError("Locate 响应缺少 qwen_bboxes 数组，请更新远端 Locate API")
    try:
        with Image.open(image_path) as source_image:
            canvas = source_image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise RuntimeError(f"读取 Qwen 结果原图失败: {error}") from error

    draw = ImageDraw.Draw(canvas)
    line_width = max(3, round(canvas.width / 420))
    for index, record in enumerate(qwen_bboxes, start=1):
        bbox = record.get("bbox_original") if isinstance(record, dict) else None
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            continue
        x1, y1, x2, y2 = (round(float(value)) for value in bbox)
        color = QWEN_BOX_COLORS[(index - 1) % len(QWEN_BOX_COLORS)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = f"Qwen #{index}"
        text_box = draw.textbbox((x1, y1), label)
        label_height = text_box[3] - text_box[1] + 6
        label_width = text_box[2] - text_box[0] + 8
        label_y = max(0, y1 - label_height)
        draw.rectangle(
            (x1, label_y, x1 + label_width, label_y + label_height),
            fill=color,
        )
        draw.text((x1 + 4, label_y + 3), label, fill=(255, 255, 255))

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{image_path.stem}_{sku_id}_qwen.png"
    canvas.save(output_path, format="PNG")
    return output_path


def run_test_inference(
    name: str,
    output_directory: Path = DEFAULT_RESULT_DIRECTORY,
) -> dict[str, dict[str, Any]]:
    normalized_name = name.strip()
    if not normalized_name:
        raise RuntimeError("name 不能为空")

    product = lookup_sku_by_name(normalized_name)
    image_paths = find_test_images(product["sku_id"])
    results: dict[str, dict[str, Any]] = {}
    for image_path in image_paths:
        try:
            response = requests.post(
                f"{LOCATE_API_URL}/visual/pick/locate",
                json={
                    "name": product["name"],
                    "image_name": image_path.name,
                    "image_base64": base64.b64encode(image_path.read_bytes()).decode(
                        "ascii"
                    ),
                },
                timeout=LOCATE_REQUEST_TIMEOUT_SECONDS,
            )
            payload = response.json()
            if response.ok:
                result_directory = output_directory / product["sku_id"]
                try:
                    qwen_result_image = save_qwen_visualization(
                        image_path,
                        payload,
                        result_directory,
                        product["sku_id"],
                    )
                    payload["qwen_result_image"] = str(qwen_result_image.resolve())
                except RuntimeError as error:
                    payload["qwen_visualization_error"] = str(error)
                try:
                    result_image = save_result_visualization(
                        image_path,
                        payload,
                        result_directory,
                        product["sku_id"],
                    )
                    payload["result_image"] = str(result_image.resolve())
                except RuntimeError as error:
                    payload["visualization_error"] = str(error)
                results[image_path.name] = payload
            else:
                results[image_path.name] = {
                    "error": payload.get("detail", payload),
                    "status_code": response.status_code,
                }
        except requests.RequestException as error:
            results[image_path.name] = {
                "error": f"Locate API 请求失败: {error}",
                "status_code": 502,
            }
        except ValueError:
            results[image_path.name] = {
                "error": "Locate API 响应不是有效 JSON",
                "status_code": 502,
            }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据商品名、SKU 和 image_name_mapping.json 运行测试图片推理"
    )
    parser.add_argument("name", help="SKU 商品名称，例如：蒙牛纯牛奶")
    parser.add_argument(
        "--output",
        type=Path,
        help="可选的 JSON 输出路径；未提供时打印到终端",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULT_DIRECTORY,
        help=f"结果图片目录，默认：{DEFAULT_RESULT_DIRECTORY}",
    )
    return parser.parse_args()


def main_cli() -> None:
    args = parse_args()
    try:
        results = run_test_inference(args.name, output_directory=args.output_dir)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    serialized = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
        return
    args.output.write_text(serialized, encoding="utf-8")
    print(f"推理结果已保存到: {args.output.resolve()}")


if __name__ == "__main__":
    main_cli()
