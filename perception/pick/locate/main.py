from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import mimetypes
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
PROMPT_MAPPING_PATH = ROOT / "qwen_sam_prompt_mapping.json"
MONITOR_IMAGE_DIR = Path(
    os.getenv("LOCATE_MONITOR_IMAGE_DIR", str(ROOT / "monitor_images"))
)
CAMERA_SNAPSHOT_URL = os.getenv(
    "CAMERA_SNAPSHOT_URL",
    "http://192.168.130.50:8085/camera/snapshot?camera=left_wrist&type=color",
)
CAMERA_SNAPSHOT_TIMEOUT_SECONDS = float(
    os.getenv("CAMERA_SNAPSHOT_TIMEOUT_SECONDS", "5")
)
CAMERA_SNAPSHOT_CACHE_DIR = Path(
    os.getenv("CAMERA_SNAPSHOT_CACHE_DIR", str(ROOT / "camera_snapshots"))
)

SKU_API_URL = os.getenv("SKU_API_URL", "http://127.0.0.1:25540").rstrip("/")
SAM3_URL = os.getenv(
    "SAM3_URL",
    "http://211.137.21.33:25541/api/v1/segment",
)
QWEN3_URL = os.getenv(
    "QWEN3_URL",
    "http://211.137.21.33:25542/v1/chat/completions",
)
QWEN3_MODEL = os.getenv("QWEN3_MODEL", "Qwen3-VL-4B-Instruct")

QWEN_SAMPLE_COUNT = 3
QWEN_TEMPERATURE = 0.5
QWEN_CONSENSUS_IOU = 0.85
CROP_PADDING_RATIO = 0.1
SAM3_THRESHOLD = 0.5
SAM3_MASK_THRESHOLD = 0.5
SAM_BBOX_OVERLAP_MIN_RATIO = float(
    os.getenv("SAM_BBOX_OVERLAP_MIN_RATIO", "0.2")
)
SAM_FRONT_AREA_DOMINANCE_RATIO = float(
    os.getenv("SAM_FRONT_AREA_DOMINANCE_RATIO", "2.0")
)
SAM_SMALLEST_MASK_MAX_RATIO = float(
    os.getenv("SAM_SMALLEST_MASK_MAX_RATIO", "0.5")
)
REQUEST_TIMEOUT_SECONDS = 120

app = FastAPI(title="Sorting Pick Locate", version="2.0.0")


class LocateRequest(BaseModel):
    name: str
    product_name: str
    hand: str
    image_name: str | None = None
    image_base64: str | None = None
    qwen3_prompt: str | None = None
    sam3_prompt: str | None = None


class LocatedInstance(BaseModel):
    bbox: list[float]
    mask: str
    score: float | None = None


class QwenBBoxRecord(BaseModel):
    bbox_normalized: list[float]
    bbox_original: list[float]
    crop_box_original: list[int]


class RawQwenBBoxRecord(BaseModel):
    sample_index: int
    name: str
    bbox_normalized: list[float]
    bbox_original: list[float]


class LocateDebugResponse(BaseModel):
    sku_id: str
    product_name: str
    image_name: str
    image_path: str
    image_base64: str
    image_media_type: str
    image_size: list[int]
    raw_qwen_bboxes: list[RawQwenBBoxRecord] = Field(default_factory=list)
    qwen_bboxes: list[QwenBBoxRecord] = Field(default_factory=list)
    instances: list[LocatedInstance] = Field(default_factory=list)
    error: str | None = None
    error_status_code: int | None = None


class LocateResponse(BaseModel):
    product_name: str
    bbox: list[int]
    mask: str
    image_path: str


def get_latest_rgb() -> Path:
    """只读取相机快照接口；不可用或内容无效时返回 HTTP 400。"""
    camera_snapshot = fetch_camera_snapshot()
    if camera_snapshot is not None:
        return camera_snapshot
    raise HTTPException(
        status_code=400,
        detail="未提供图片，且相机快照接口读取失败或未返回有效 JPG/PNG",
    )


def fetch_camera_snapshot() -> Path | None:
    """获取并验证相机快照；任何读取错误都返回 None。"""
    try:
        response = requests.get(
            CAMERA_SNAPSHOT_URL,
            timeout=CAMERA_SNAPSHOT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        image_bytes = response.content
    except requests.RequestException:
        return None

    if not image_bytes or len(image_bytes) > 20 * 1024 * 1024:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as source_image:
            image_format = (source_image.format or "").upper()
            source_image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return None

    suffix = {"JPEG": ".jpg", "PNG": ".png"}.get(image_format)
    if suffix is None:
        return None
    snapshot_path = CAMERA_SNAPSHOT_CACHE_DIR / f"latest_camera_rgb{suffix}"
    temporary_path = snapshot_path.with_suffix(f"{suffix}.tmp")
    try:
        CAMERA_SNAPSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temporary_path.write_bytes(image_bytes)
        temporary_path.replace(snapshot_path)
    except OSError:
        return None
    return snapshot_path


def store_monitor_image(image_path: Path) -> str:
    """按内容哈希持久化原图，返回监控系统可读取的本地绝对路径。"""
    try:
        image_bytes = image_path.read_bytes()
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"读取待存储图片失败: {error}") from error
    if not image_bytes:
        raise HTTPException(status_code=500, detail="待存储图片为空")

    suffix = image_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        suffix = ".jpg"
    digest = hashlib.sha256(image_bytes).hexdigest()[:24]
    stored_name = f"{digest}{suffix}"
    stored_path = MONITOR_IMAGE_DIR / stored_name
    try:
        MONITOR_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        if not stored_path.exists():
            temporary_path = stored_path.with_suffix(f"{stored_path.suffix}.tmp")
            temporary_path.write_bytes(image_bytes)
            temporary_path.replace(stored_path)
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"存储监控图片失败: {error}") from error
    return str(stored_path.resolve())


def decode_uploaded_image(image_base64: str) -> bytes:
    encoded = image_base64.split(",", 1)[-1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise HTTPException(status_code=400, detail="image_base64 格式错误") from error
    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传图片不能为空")
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="上传图片不能超过 20 MB")
    return image_bytes


def uploaded_image_name(image_name: str | None) -> str:
    normalized_name = (image_name or "uploaded_rgb.jpg").strip()
    if (
        not normalized_name
        or Path(normalized_name).name != normalized_name
        or Path(normalized_name).suffix.lower() not in {".jpg", ".jpeg", ".png"}
    ):
        raise HTTPException(status_code=400, detail="image_name 不是合法的图片文件名")
    return normalized_name


@app.get("/video/frame")
def get_video_frame() -> FileResponse:
    image_path = get_latest_rgb()
    media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    return FileResponse(image_path, media_type=media_type)


def lookup_sku_by_name(name: str) -> dict[str, Any]:
    try:
        response = requests.get(
            f"{SKU_API_URL}/sku/search_by_name",
            params={"name": name},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"SKU 查询请求失败: {error}") from error

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail=f"SKU 中不存在商品: {name}")
    try:
        response.raise_for_status()
        product = response.json()
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"SKU 查询请求失败: {error}") from error
    except ValueError as error:
        raise HTTPException(status_code=502, detail="SKU 查询响应不是有效 JSON") from error

    if (
        not isinstance(product, dict)
        or not isinstance(product.get("sku_id"), str)
        or not isinstance(product.get("name"), str)
    ):
        raise HTTPException(status_code=502, detail="SKU 查询响应缺少 sku_id 或 name")
    return product


def load_prompt_pair(name: str) -> tuple[str, str]:
    try:
        mapping = json.loads(PROMPT_MAPPING_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail="Prompt 配对文件不存在") from error
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"读取 Prompt 配对失败: {error}") from error

    pair = mapping.get(name) if isinstance(mapping, dict) else None
    if not isinstance(pair, dict):
        raise HTTPException(status_code=400, detail=f"商品尚未配置配对 Prompt: {name}")
    qwen_prompt = pair.get("qwen3_prompt")
    sam_prompt = pair.get("sam3_prompt")
    if not isinstance(qwen_prompt, str) or not qwen_prompt.strip():
        raise HTTPException(status_code=500, detail=f"商品缺少 Qwen3 Prompt: {name}")
    if not isinstance(sam_prompt, str) or not sam_prompt.strip():
        raise HTTPException(status_code=500, detail=f"商品缺少 SAM3 Prompt: {name}")
    return qwen_prompt.strip(), sam_prompt.strip()


def call_qwen3(prompt: str, image_path: Path) -> str:
    media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = requests.post(
        QWEN3_URL,
        json={
            "model": QWEN3_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            "temperature": QWEN_TEMPERATURE,
            "max_tokens": 1024,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("Qwen3 message content 不是字符串")
    return content


def parse_qwen_detections(content: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    decoded: dict[str, Any] | list[Any] | None = None
    for match in re.finditer(r"[\[{]", content):
        try:
            candidate, _ = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) or (
            isinstance(candidate, list)
            and all(isinstance(item, dict) for item in candidate)
        ):
            decoded = candidate
            break
    if decoded is None:
        raise ValueError("Qwen3 输出中没有找到 JSON 对象或数组")

    items = [decoded] if isinstance(decoded, dict) else decoded
    detections: list[dict[str, Any]] = []
    for item in items:
        bbox = item.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in bbox
            )
        ):
            raise ValueError("Qwen3 bbox 必须由四个有限数字组成")
        x1, y1, x2, y2 = (float(value) for value in bbox)
        normalized_bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        if normalized_bbox[2] <= normalized_bbox[0] or normalized_bbox[3] <= normalized_bbox[1]:
            raise ValueError("Qwen3 bbox 面积必须大于 0")
        detections.append(
            {
                "name": str(item.get("name", "")).strip(),
                "bbox": normalized_bbox,
            }
        )
    return detections


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    intersection_width = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    intersection_height = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def consensus_qwen_bboxes(
    samples: list[tuple[int, list[dict[str, Any]]]],
    iou_threshold: float = QWEN_CONSENSUS_IOU,
) -> list[list[float]]:
    """聚合跨采样检测，仅保留至少两个采样以 IoU>阈值共同支持的框。"""
    detections = [
        (sample_index, detection["bbox"])
        for sample_index, sample_detections in samples
        for detection in sample_detections
    ]
    parents = list(range(len(detections)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(detections)):
        for right in range(left + 1, len(detections)):
            if detections[left][0] == detections[right][0]:
                continue
            if bbox_iou(detections[left][1], detections[right][1]) > iou_threshold:
                union(left, right)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(detections)):
        components[find(index)].append(index)

    consensus: list[list[float]] = []
    for component in components.values():
        by_sample: dict[int, list[int]] = defaultdict(list)
        for index in component:
            by_sample[detections[index][0]].append(index)
        if len(by_sample) < 2:
            continue

        selected_indices: list[int] = []
        for candidates in by_sample.values():
            selected_indices.append(
                max(
                    candidates,
                    key=lambda candidate: sum(
                        bbox_iou(detections[candidate][1], detections[other][1])
                        for other in component
                        if detections[candidate][0] != detections[other][0]
                    ),
                )
            )

        consensus.append(
            [
                sum(detections[index][1][coordinate] for index in selected_indices)
                / len(selected_indices)
                for coordinate in range(4)
            ]
        )

    return sorted(consensus, key=lambda box: (box[1], box[0], box[3], box[2]))


class QwenConsensusBBoxes(list[list[float]]):
    def __init__(
        self,
        bboxes: list[list[float]],
        samples: list[tuple[int, list[dict[str, Any]]]],
    ) -> None:
        super().__init__(bboxes)
        self.samples = samples


def get_stable_qwen_bboxes(prompt: str, image_path: Path) -> list[list[float]]:
    samples: list[tuple[int, list[dict[str, Any]]]] = []
    errors: list[str] = []
    for sample_index in range(1, QWEN_SAMPLE_COUNT + 1):
        try:
            content = call_qwen3(prompt, image_path)
            samples.append((sample_index, parse_qwen_detections(content)))
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
            errors.append(f"第 {sample_index} 次: {error}")

    if len(samples) < 2:
        detail = "; ".join(errors) or "成功采样不足两次"
        raise HTTPException(status_code=502, detail=f"Qwen3 无法形成跨采样共识: {detail}")

    bboxes = consensus_qwen_bboxes(samples)
    if not bboxes:
        raise HTTPException(
            status_code=404,
            detail=f"Qwen3 没有产生跨采样 IoU > {QWEN_CONSENSUS_IOU} 的稳定 bbox",
        )
    return QwenConsensusBBoxes(bboxes, samples)


def qwen_bbox_to_crop(
    bbox: list[float], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """按网页默认规则将 [0,1000] Qwen 坐标转为像素并外扩 10%。"""
    image_width, image_height = image_size
    x1, y1, x2, y2 = qwen_bbox_to_original(bbox, image_size)
    padding_x = (x2 - x1) * CROP_PADDING_RATIO
    padding_y = (y2 - y1) * CROP_PADDING_RATIO
    crop_box = (
        max(0, math.floor(x1 - padding_x)),
        max(0, math.floor(y1 - padding_y)),
        min(image_width, math.ceil(x2 + padding_x)),
        min(image_height, math.ceil(y2 + padding_y)),
    )
    if crop_box[2] - crop_box[0] < 2 or crop_box[3] - crop_box[1] < 2:
        raise ValueError("Qwen3 bbox 无法生成有效 crop")
    return crop_box


def qwen_bbox_to_original(
    bbox: list[float], image_size: tuple[int, int]
) -> list[float]:
    """将 Qwen [0,1000] bbox 转换为未外扩的原图像素坐标。"""
    image_width, image_height = image_size
    return [
        round(min(1000.0, max(0.0, bbox[0])) / 1000.0 * image_width, 6),
        round(min(1000.0, max(0.0, bbox[1])) / 1000.0 * image_height, 6),
        round(min(1000.0, max(0.0, bbox[2])) / 1000.0 * image_width, 6),
        round(min(1000.0, max(0.0, bbox[3])) / 1000.0 * image_height, 6),
    ]


def call_sam3(prompt: str, crop_image: Image.Image) -> list[dict[str, Any]]:
    buffer = io.BytesIO()
    crop_image.save(buffer, format="JPEG", quality=95)
    try:
        response = requests.post(
            SAM3_URL,
            files={"image": ("qwen_crop.jpg", buffer.getvalue(), "image/jpeg")},
            data={
                "prompt": prompt,
                "threshold": SAM3_THRESHOLD,
                "mask_threshold": SAM3_MASK_THRESHOLD,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"SAM3 请求失败: {error}") from error
    except ValueError as error:
        raise HTTPException(status_code=502, detail="SAM3 响应不是有效 JSON") from error

    instances = payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(instances, list):
        raise HTTPException(status_code=502, detail="SAM3 响应缺少 instances 数组")
    return instances


def map_mask_to_original(
    mask_base64: str,
    crop_box: tuple[int, int, int, int],
    original_size: tuple[int, int],
) -> str:
    encoded = mask_base64.split(",", 1)[-1]
    try:
        mask_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(mask_bytes)) as source_mask:
            crop_mask = source_mask.convert("L")
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=502, detail=f"SAM3 mask PNG 无效: {error}") from error

    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]
    if crop_mask.size != (crop_width, crop_height):
        crop_mask = crop_mask.resize(
            (crop_width, crop_height),
            resample=Image.Resampling.NEAREST,
        )

    original_mask = Image.new("L", original_size, 0)
    original_mask.paste(crop_mask, (crop_box[0], crop_box[1]))
    output = io.BytesIO()
    original_mask.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def map_sam_instance_to_original(
    instance: dict[str, Any],
    crop_box: tuple[int, int, int, int],
    original_size: tuple[int, int],
) -> LocatedInstance:
    bbox = instance.get("bbox_xyxy")
    mask = instance.get("mask_png_base64")
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(isinstance(value, (int, float)) for value in bbox)
    ):
        raise HTTPException(status_code=502, detail="SAM3 实例 bbox_xyxy 格式错误")
    if not isinstance(mask, str) or not mask:
        raise HTTPException(status_code=502, detail="SAM3 实例缺少 mask_png_base64")

    crop_x1, crop_y1, _, _ = crop_box
    image_width, image_height = original_size
    original_bbox = [
        max(0.0, min(float(image_width), float(bbox[0]) + crop_x1)),
        max(0.0, min(float(image_height), float(bbox[1]) + crop_y1)),
        max(0.0, min(float(image_width), float(bbox[2]) + crop_x1)),
        max(0.0, min(float(image_height), float(bbox[3]) + crop_y1)),
    ]
    score_value = instance.get("score")
    score = float(score_value) if isinstance(score_value, (int, float)) else None
    return LocatedInstance(
        bbox=original_bbox,
        mask=map_mask_to_original(mask, crop_box, original_size),
        score=score,
    )


def bbox_overlap_by_smaller_area(
    first_bbox: list[float], second_bbox: list[float]
) -> float:
    """返回交集占较小 bbox 的比例，适合发现嵌套或大小不同的重复框。"""
    first_area = max(0.0, first_bbox[2] - first_bbox[0]) * max(
        0.0, first_bbox[3] - first_bbox[1]
    )
    second_area = max(0.0, second_bbox[2] - second_bbox[0]) * max(
        0.0, second_bbox[3] - second_bbox[1]
    )
    smaller_area = min(first_area, second_area)
    if smaller_area <= 0:
        return 0.0

    intersection_width = max(
        0.0,
        min(first_bbox[2], second_bbox[2]) - max(first_bbox[0], second_bbox[0]),
    )
    intersection_height = max(
        0.0,
        min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]),
    )
    return intersection_width * intersection_height / smaller_area


def mask_foreground_pixel_count(mask_base64: str) -> int:
    encoded = mask_base64.split(",", 1)[-1]
    try:
        mask_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(mask_bytes)) as source_mask:
            histogram = source_mask.convert("L").histogram()
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=502, detail=f"SAM3 mask PNG 无效: {error}") from error
    return sum(histogram[128:])


def select_frontmost_instance(
    instances: list[LocatedInstance],
) -> LocatedInstance:
    """从一条重叠链中按 mask 面积悬殊规则和 mask 密度选出最前方实例。"""
    if len(instances) == 1:
        return instances[0]

    metrics: list[tuple[LocatedInstance, int, float, float]] = []
    for instance in instances:
        bbox_width = max(0.0, instance.bbox[2] - instance.bbox[0])
        bbox_height = max(0.0, instance.bbox[3] - instance.bbox[1])
        bbox_area = bbox_width * bbox_height
        mask_area = mask_foreground_pixel_count(instance.mask)
        mask_density = mask_area / bbox_area if bbox_area > 0 else 0.0
        metrics.append((instance, mask_area, mask_density, bbox_area))

    by_mask_area = sorted(metrics, key=lambda item: item[1], reverse=True)
    if (
        by_mask_area[0][1] > 0
        and by_mask_area[0][1]
        >= SAM_FRONT_AREA_DOMINANCE_RATIO * by_mask_area[1][1]
    ):
        return by_mask_area[0][0]

    return max(
        metrics,
        key=lambda item: (
            item[2],
            item[1],
            item[0].score if item[0].score is not None else -1.0,
            item[3],
        ),
    )[0]


def keep_frontmost_in_overlap_chains(
    instances: list[LocatedInstance],
) -> list[LocatedInstance]:
    """每个 bbox 重叠连通分量只保留一个最前方实例。"""
    if len(instances) < 2:
        return instances

    parents = list(range(len(instances)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(instances):
        for second_index in range(first_index + 1, len(instances)):
            overlap = bbox_overlap_by_smaller_area(
                first.bbox,
                instances[second_index].bbox,
            )
            if overlap >= SAM_BBOX_OVERLAP_MIN_RATIO:
                union(first_index, second_index)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(instances)):
        components[find(index)].append(index)

    selected: list[tuple[int, LocatedInstance]] = []
    for indices in components.values():
        component_instances = [instances[index] for index in indices]
        frontmost = select_frontmost_instance(component_instances)
        selected_index = next(
            index
            for index in indices
            if instances[index] is frontmost
        )
        selected.append((selected_index, frontmost))
    return [instance for _, instance in sorted(selected, key=lambda item: item[0])]


def drop_smallest_mask_area_outlier(
    instances: list[LocatedInstance],
) -> list[LocatedInstance]:
    """最小 mask 不超过第二小 mask 的指定比例时，只删除该最小离群项一次。"""
    if len(instances) < 2:
        return instances

    by_mask_area = sorted(
        (
            (index, mask_foreground_pixel_count(instance.mask))
            for index, instance in enumerate(instances)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    smallest_index, smallest_area = by_mask_area[-1]
    second_smallest_area = by_mask_area[-2][1]
    if (
        second_smallest_area > 0
        and smallest_area
        <= SAM_SMALLEST_MASK_MAX_RATIO * second_smallest_area
    ):
        return [
            instance
            for index, instance in enumerate(instances)
            if index != smallest_index
        ]
    return instances


def locate_product_in_image(
    product: dict[str, Any],
    image_path: Path,
    *,
    qwen_prompt_override: str | None = None,
    sam_prompt_override: str | None = None,
) -> LocateDebugResponse:
    """使用已查询的 SKU 信息，在指定 RGB 图片上运行完整定位流程。"""
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"测试图片不存在: {image_path.name}")
    monitor_image_path = store_monitor_image(image_path)
    canonical_name = product["name"].strip()
    qwen_prompt = (qwen_prompt_override or "").strip()
    sam_prompt = (sam_prompt_override or "").strip()
    if not qwen_prompt or not sam_prompt:
        stored_qwen_prompt, stored_sam_prompt = load_prompt_pair(canonical_name)
        qwen_prompt = qwen_prompt or stored_qwen_prompt
        sam_prompt = sam_prompt or stored_sam_prompt
    qwen_bboxes = get_stable_qwen_bboxes(qwen_prompt, image_path)

    try:
        with Image.open(image_path) as source_image:
            original_image = source_image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=500, detail=f"读取 RGB 图片失败: {error}") from error

    raw_qwen_bbox_records = [
        RawQwenBBoxRecord(
            sample_index=sample_index,
            name=detection["name"],
            bbox_normalized=detection["bbox"],
            bbox_original=qwen_bbox_to_original(detection["bbox"], original_image.size),
        )
        for sample_index, detections in getattr(qwen_bboxes, "samples", [])
        for detection in detections
    ]
    qwen_bbox_records: list[QwenBBoxRecord] = []
    located_instances: list[LocatedInstance] = []
    for qwen_bbox in qwen_bboxes:
        try:
            crop_box = qwen_bbox_to_crop(qwen_bbox, original_image.size)
        except ValueError:
            continue
        qwen_bbox_records.append(
            QwenBBoxRecord(
                bbox_normalized=qwen_bbox,
                bbox_original=qwen_bbox_to_original(qwen_bbox, original_image.size),
                crop_box_original=list(crop_box),
            )
        )
        crop_image = original_image.crop(crop_box)
        for instance in call_sam3(sam_prompt, crop_image):
            if isinstance(instance, dict):
                located_instances.append(
                    map_sam_instance_to_original(instance, crop_box, original_image.size)
                )

    if not located_instances:
        raise HTTPException(status_code=404, detail="SAM3 没有找到目标商品实例")

    located_instances = keep_frontmost_in_overlap_chains(located_instances)
    located_instances = drop_smallest_mask_area_outlier(located_instances)

    return LocateDebugResponse(
        sku_id=product["sku_id"],
        product_name=canonical_name,
        image_name=image_path.name,
        image_path=monitor_image_path,
        image_base64=base64.b64encode(image_path.read_bytes()).decode("ascii"),
        image_media_type=mimetypes.guess_type(image_path.name)[0] or "image/jpeg",
        image_size=list(original_image.size),
        raw_qwen_bboxes=raw_qwen_bbox_records,
        qwen_bboxes=qwen_bbox_records,
        instances=located_instances,
    )


def make_locate_debug_error_response(
    product: dict[str, Any], image_path: Path, error: HTTPException
) -> LocateDebugResponse:
    """Return the actual input image together with an inference error for debugging."""
    try:
        with Image.open(image_path) as source_image:
            image_size = list(source_image.size)
        image_bytes = image_path.read_bytes()
    except (UnidentifiedImageError, OSError) as image_error:
        raise error from image_error

    detail = error.detail
    if not isinstance(detail, str):
        detail = json.dumps(detail, ensure_ascii=False)
    return LocateDebugResponse(
        sku_id=product["sku_id"],
        product_name=product["name"].strip(),
        image_name=image_path.name,
        image_path=store_monitor_image(image_path),
        image_base64=base64.b64encode(image_bytes).decode("ascii"),
        image_media_type=mimetypes.guess_type(image_path.name)[0] or "image/jpeg",
        image_size=image_size,
        error=detail,
        error_status_code=error.status_code,
    )


def locate_product_debug(
    request: LocateRequest,
    *,
    capture_inference_errors: bool = False,
    allow_prompt_overrides: bool = False,
) -> LocateDebugResponse:
    product_name = request.product_name.strip()
    if not product_name:
        raise HTTPException(status_code=400, detail="product_name 不能为空")
    product = lookup_sku_by_name(product_name)
    prompt_overrides = (
        {
            "qwen_prompt_override": request.qwen3_prompt,
            "sam_prompt_override": request.sam3_prompt,
        }
        if allow_prompt_overrides
        else {}
    )
    if request.image_base64 is None:
        if request.image_name is not None:
            raise HTTPException(
                status_code=400,
                detail="指定 image_name 时必须同时提供 image_base64",
            )
        image_path = get_latest_rgb()
        try:
            return locate_product_in_image(product, image_path, **prompt_overrides)
        except HTTPException as error:
            if not capture_inference_errors:
                raise
            return make_locate_debug_error_response(product, image_path, error)

    image_bytes = decode_uploaded_image(request.image_base64)
    image_name = uploaded_image_name(request.image_name)
    with tempfile.TemporaryDirectory(prefix="locate-upload-") as temporary_directory:
        image_path = Path(temporary_directory) / image_name
        image_path.write_bytes(image_bytes)
        try:
            return locate_product_in_image(product, image_path, **prompt_overrides)
        except HTTPException as error:
            if not capture_inference_errors:
                raise
            return make_locate_debug_error_response(product, image_path, error)


def normalize_bbox_to_1_1000(
    bbox: list[float], image_size: list[int]
) -> list[int]:
    """把原图像素 bbox 映射到接口约定的闭区间 [1,1000]。"""
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise HTTPException(status_code=500, detail="原图尺寸无效")
    scales = (image_width, image_height, image_width, image_height)
    return [
        max(
            1,
            min(
                1000,
                round(1 + max(0.0, min(float(scale), float(value))) / scale * 999),
            ),
        )
        for value, scale in zip(bbox, scales)
    ]


def make_locate_response(debug_response: LocateDebugResponse) -> LocateResponse:
    if not debug_response.instances:
        raise HTTPException(status_code=404, detail="SAM3 没有找到目标商品实例")
    frontmost = select_frontmost_instance(debug_response.instances)
    return LocateResponse(
        product_name=debug_response.product_name,
        bbox=normalize_bbox_to_1_1000(frontmost.bbox, debug_response.image_size),
        mask=frontmost.mask,
        image_path=debug_response.image_path,
    )


@app.post("/perception/pick/locate", response_model=LocateResponse)
def locate_product(request: LocateRequest) -> LocateResponse:
    return make_locate_response(locate_product_debug(request))


@app.post("/perception/pick/locate/debug", response_model=LocateDebugResponse)
def locate_product_debug_api(request: LocateRequest) -> LocateDebugResponse:
    return locate_product_debug(
        request,
        capture_inference_errors=True,
        allow_prompt_overrides=True,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
