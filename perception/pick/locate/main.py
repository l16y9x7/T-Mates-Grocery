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
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
if __package__ and __package__.startswith("perception."):
    from ...config import (
        QWEN3_MODEL as CONFIG_QWEN3_MODEL,
        QWEN3_URL as CONFIG_QWEN3_URL,
        SAM3_URL as CONFIG_SAM3_URL,
        SKU_API_URL as CONFIG_SKU_API_URL,
        camera_depth_snapshot_url,
        camera_snapshot_url,
    )
else:
    PERCEPTION_ROOT = ROOT.parents[1]
    if str(PERCEPTION_ROOT) not in sys.path:
        sys.path.insert(0, str(PERCEPTION_ROOT))
    from config import (
        QWEN3_MODEL as CONFIG_QWEN3_MODEL,
        QWEN3_URL as CONFIG_QWEN3_URL,
        SAM3_URL as CONFIG_SAM3_URL,
        SKU_API_URL as CONFIG_SKU_API_URL,
        camera_depth_snapshot_url,
        camera_snapshot_url,
    )


PROMPT_MAPPING_PATH = ROOT / "qwen_sam_prompt_mapping.json"
HARD_CASE_PROMPT_MAPPING_PATH = ROOT / "qwen_sam_prompt_mapping_hard_case.json"
HARD_CASE_SCOPE_PATH = ROOT.parents[1] / "hard_case_config.json"
SUPPORTED_TASK_TYPES = ("SORTING", "SHORTAGE", "MISPLACED")
PROMPT_MAPPING_PATHS = {
    "SORTING": PROMPT_MAPPING_PATH,
    "SHORTAGE": ROOT / "qwen_sam_prompt_mapping_shortage.json",
    "MISPLACED": ROOT / "qwen_sam_prompt_mapping_misplaced.json",
}
MONITOR_IMAGE_DIR = Path(
    os.getenv("LOCATE_MONITOR_IMAGE_DIR", str(ROOT / "monitor_images"))
)
LEFT_CAMERA_SNAPSHOT_URL = camera_snapshot_url("left")
RIGHT_CAMERA_SNAPSHOT_URL = camera_snapshot_url("right")
LEFT_CAMERA_DEPTH_SNAPSHOT_URL = camera_depth_snapshot_url("left")
RIGHT_CAMERA_DEPTH_SNAPSHOT_URL = camera_depth_snapshot_url("right")
# 保留旧常量名称，兼容现有左手相机配置与测试。
CAMERA_SNAPSHOT_URL = LEFT_CAMERA_SNAPSHOT_URL
CAMERA_SNAPSHOT_URLS = {
    "left": LEFT_CAMERA_SNAPSHOT_URL,
    "right": RIGHT_CAMERA_SNAPSHOT_URL,
}
CAMERA_DEPTH_SNAPSHOT_URLS = {
    "left": LEFT_CAMERA_DEPTH_SNAPSHOT_URL,
    "right": RIGHT_CAMERA_DEPTH_SNAPSHOT_URL,
}
CAMERA_SNAPSHOT_TIMEOUT_SECONDS = float(
    os.getenv("CAMERA_SNAPSHOT_TIMEOUT_SECONDS", "5")
)
CAMERA_SNAPSHOT_CACHE_DIR = Path(
    os.getenv("CAMERA_SNAPSHOT_CACHE_DIR", str(ROOT / "camera_snapshots"))
)

SKU_API_URL = CONFIG_SKU_API_URL
SAM3_URL = CONFIG_SAM3_URL
QWEN3_URL = CONFIG_QWEN3_URL
QWEN3_MODEL = CONFIG_QWEN3_MODEL

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
PICK_MIN_ASPECT_RATIO_TO_BEST = float(
    os.getenv("PICK_MIN_ASPECT_RATIO_TO_BEST", "0.75")
)
PICK_OCCLUSION_DEPTH_MARGIN_MM = float(
    os.getenv("PICK_OCCLUSION_DEPTH_MARGIN_MM", "30")
)
PICK_FRONT_ROW_DEPTH_TOLERANCE_MM = float(
    os.getenv("PICK_FRONT_ROW_DEPTH_TOLERANCE_MM", "30")
)
PICK_OCCLUSION_MAX_NEIGHBOR_GAP_RATIO = float(
    os.getenv("PICK_OCCLUSION_MAX_NEIGHBOR_GAP_RATIO", "0.25")
)
PICK_OCCLUSION_MIN_VERTICAL_OVERLAP_RATIO = float(
    os.getenv("PICK_OCCLUSION_MIN_VERTICAL_OVERLAP_RATIO", "0.5")
)
PICK_MIN_VALID_DEPTH_PIXELS = int(
    os.getenv("PICK_MIN_VALID_DEPTH_PIXELS", "50")
)
CAMERA_DEPTH_UNIT_MM = float(os.getenv("CAMERA_DEPTH_UNIT_MM", "1.0"))
REQUEST_TIMEOUT_SECONDS = 120

LOCATION_PATTERN = re.compile(
    r"^H(?P<shelf>[12])_(?P<face>[FB])_L(?P<level>[1-5])_C(?P<column>\d{2})$"
)


@dataclass(frozen=True)
class HardCaseGroupConfig:
    members: tuple[str, ...]
    preferred_hand: str
    hand_overrides: tuple[tuple[str, str], ...] = ()

    def hand_for_product(self, product_name: str) -> str:
        return dict(self.hand_overrides).get(product_name, self.preferred_hand)


# This allow-list is deliberately narrow. Non-SORTING tasks and every SKU not
# listed here continue through the original prompt and center-selection path.
HARD_CASE_GROUPS: dict[str, HardCaseGroupConfig] = {
    "maiydong": HardCaseGroupConfig(
        members=(
            "脉动观梅止渴饮",
            "脉动芒果口味",
            "脉动菠萝口味",
            "脉动猫薄荷瓶",
        ),
        preferred_hand="left",
        hand_overrides=(("脉动猫薄荷瓶", "right"),),
    ),
    "alien_energy": HardCaseGroupConfig(
        members=(
            "外星人电解质水椰子口味",
            "外星人电解质水青柠口味",
            "外星人电解质水白桃口味0糖",
            "外星人电解质水西柚口味",
            "外星人电解质水青柠口味0糖",
        ),
        preferred_hand="left",
        hand_overrides=(("外星人电解质水青柠口味0糖", "right"),),
    ),
    "bbq_seasoning": HardCaseGroupConfig(
        members=(
            "草原红太阳烧烤料原味",
            "草原红太阳烧烤料香辣味",
        ),
        preferred_hand="left",
    ),
    "bbq_sauce_spicy": HardCaseGroupConfig(
        members=("草原红太阳烧烤酱香辣",),
        preferred_hand="left",
    ),
    "bbq_sauce_original": HardCaseGroupConfig(
        members=("草原红太阳烧烤酱原味",),
        preferred_hand="left",
    ),
    "soy_vinegar_fish_soy": HardCaseGroupConfig(
        members=(
            "镇江香醋",
            "蒸鱼豉油",
            "薄盐生抽",
        ),
        preferred_hand="right",
    ),
}

app = FastAPI(title="Sorting Pick Locate", version="2.0.0")
router = APIRouter()


class LocateRequest(BaseModel):
    task_type: str
    product_name: str
    level: str
    hand: str
    image_name: str | None = None
    image_base64: str | None = None
    depth_image_name: str | None = None
    depth_image_base64: str | None = None
    depth_is_bigendian: bool = False
    qwen3_prompt: str | None = None
    sam3_prompt: str | None = None


class LocatedInstance(BaseModel):
    bbox: list[float]
    mask: str
    score: float | None = None
    depth_mm: float | None = None
    source_qwen_index: int | None = None
    hard_case_group_index: int | None = None
    mapped_product_name: str | None = None
    is_selected: bool = False


class HardCaseGroupResult(BaseModel):
    index: int
    mapped_product_name: str
    bbox: list[float]
    instance_count: int


class HardCaseDebugInfo(BaseModel):
    group_id: str
    preferred_hand: str
    actual_hand: str
    order_direction: str
    target_location: str
    target_level: int
    target_column: int
    standard_order: list[str]
    front_row_only: bool = True
    groups: list[HardCaseGroupResult] = Field(default_factory=list)
    selected_group_index: int


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
    qwen3_prompt_used: str | None = None
    sam3_prompt_used: str | None = None
    raw_qwen_bboxes: list[RawQwenBBoxRecord] = Field(default_factory=list)
    qwen_bboxes: list[QwenBBoxRecord] = Field(default_factory=list)
    raw_sam_instances: list[LocatedInstance] = Field(default_factory=list)
    instances: list[LocatedInstance] = Field(default_factory=list)
    selected_instance: LocatedInstance | None = None
    selected_instance_index: int | None = None
    hard_case: HardCaseDebugInfo | None = None
    error: str | None = None
    error_status_code: int | None = None


class LocateResponse(BaseModel):
    product_name: str
    bbox: list[int]
    mask: str
    image_path: str


def get_latest_rgb(hand: str = "left") -> Path:
    """只读取相机快照接口；不可用或内容无效时返回 HTTP 400。"""
    normalized_hand = hand.strip().lower()
    if normalized_hand not in CAMERA_SNAPSHOT_URLS:
        raise HTTPException(status_code=400, detail="hand 只能是 left 或 right")
    camera_snapshot = fetch_camera_snapshot(normalized_hand)
    if camera_snapshot is not None:
        return camera_snapshot
    raise HTTPException(
        status_code=400,
        detail="未提供图片，且相机快照接口读取失败或未返回有效 JPG/PNG",
    )


def fetch_camera_snapshot(hand: str = "left") -> Path | None:
    """获取并验证相机快照；任何读取错误都返回 None。"""
    camera_url = CAMERA_SNAPSHOT_URLS.get(hand.strip().lower())
    print("CAMERA_URL:", camera_url)
    if camera_url is None:
        raise HTTPException(status_code=400, detail="hand 只能是 left 或 right")
    try:
        response = requests.get(
            camera_url,
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


def fetch_camera_depth(
    hand: str,
    expected_size: tuple[int, int],
) -> Image.Image | None:
    """读取相机服务返回的原始 16UC1 深度帧；不可用时静默回退。"""
    camera_url = CAMERA_DEPTH_SNAPSHOT_URLS.get(hand.strip().lower())
    if camera_url is None:
        raise HTTPException(status_code=400, detail="hand 只能是 left 或 right")
    try:
        response = requests.get(
            camera_url,
            timeout=CAMERA_SNAPSHOT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        depth_bytes = response.content
        width = int(response.headers["X-Image-Width"])
        height = int(response.headers["X-Image-Height"])
        encoding = response.headers["X-Image-Encoding"].strip().upper()
        step = int(response.headers["X-Image-Step"])
        is_bigendian = int(response.headers["X-Image-Is-Bigendian"])
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return None

    if (
        encoding != "16UC1"
        or (width, height) != expected_size
        or width <= 0
        or height <= 0
        or step != width * 2
        or len(depth_bytes) != step * height
        or is_bigendian not in {0, 1}
    ):
        return None

    raw_mode = "I;16B" if is_bigendian else "I;16L"
    try:
        return Image.frombytes(raw_mode, (width, height), depth_bytes)
    except (OSError, ValueError):
        return None


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


def decode_uploaded_depth_image(
    depth_image_base64: str,
    depth_image_name: str,
    expected_size: tuple[int, int],
    *,
    is_bigendian: bool = False,
) -> Image.Image:
    """Decode an uploaded 16-bit depth PNG/TIFF or headerless 16UC1 frame."""
    encoded = depth_image_base64.split(",", 1)[-1]
    try:
        depth_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise HTTPException(status_code=400, detail="depth_image_base64 格式错误") from error
    if not depth_bytes:
        raise HTTPException(status_code=400, detail="上传深度数据不能为空")
    if len(depth_bytes) > 40 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="上传深度数据不能超过 40 MB")

    normalized_name = depth_image_name.strip()
    if not normalized_name or Path(normalized_name).name != normalized_name:
        raise HTTPException(status_code=400, detail="depth_image_name 不是合法文件名")
    suffix = Path(normalized_name).suffix.lower()
    if suffix == ".npy":
        try:
            depth_array = np.load(io.BytesIO(depth_bytes), allow_pickle=False)
        except (OSError, ValueError, TypeError) as error:
            raise HTTPException(status_code=400, detail=f"读取 NPY 深度数据失败: {error}") from error
        expected_shape = (expected_size[1], expected_size[0])
        if not isinstance(depth_array, np.ndarray) or depth_array.ndim != 2:
            raise HTTPException(status_code=400, detail="NPY 深度数据必须是二维数组")
        if depth_array.shape != expected_shape:
            raise HTTPException(
                status_code=400,
                detail=(
                    "NPY 深度数据尺寸必须与 RGB 图片一致: "
                    f"rgb={expected_size}, depth_shape={depth_array.shape}"
                ),
            )
        if depth_array.nbytes > 40 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="NPY 深度数组不能超过 40 MB")
        if np.issubdtype(depth_array.dtype, np.integer):
            if np.any(depth_array < 0) or np.any(depth_array > np.iinfo(np.int32).max):
                raise HTTPException(status_code=400, detail="NPY 整数深度值超出有效范围")
            return Image.fromarray(depth_array.astype(np.int32, copy=False))
        if np.issubdtype(depth_array.dtype, np.floating):
            return Image.fromarray(depth_array.astype(np.float32, copy=False))
        raise HTTPException(status_code=400, detail="NPY 深度数组必须是整数或浮点数类型")

    if suffix in {".raw", ".bin"}:
        width, height = expected_size
        expected_bytes = width * height * 2
        if width <= 0 or height <= 0 or len(depth_bytes) != expected_bytes:
            raise HTTPException(
                status_code=400,
                detail=(
                    "16UC1 RAW 深度数据尺寸不匹配: "
                    f"expected={expected_bytes} bytes ({width}x{height}), "
                    f"actual={len(depth_bytes)} bytes"
                ),
            )
        raw_mode = "I;16B" if is_bigendian else "I;16L"
        try:
            return Image.frombytes(raw_mode, expected_size, depth_bytes)
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=f"解析 RAW 深度数据失败: {error}") from error

    if suffix not in {".png", ".tif", ".tiff"}:
        raise HTTPException(
            status_code=400,
            detail="深度数据只支持 NPY、16 位 PNG/TIFF 或 16UC1 RAW/BIN",
        )
    try:
        with Image.open(io.BytesIO(depth_bytes)) as source_depth:
            if source_depth.size != expected_size:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "深度图尺寸必须与 RGB 图片一致: "
                        f"rgb={expected_size}, depth={source_depth.size}"
                    ),
                )
            if source_depth.mode != "I" and not source_depth.mode.startswith("I;16"):
                raise HTTPException(status_code=400, detail="深度图必须是 16 位单通道图片")
            return source_depth.convert("I")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"读取深度图失败: {error}") from error


@router.get("/video/frame")
def get_video_frame(hand: str = "left") -> FileResponse:
    image_path = get_latest_rgb(hand)
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


def lookup_sku_row(location_id: str) -> list[dict[str, Any]]:
    """Return the standard row containing location_id, ordered left to right."""
    try:
        response = requests.request(
            "GET",
            f"{SKU_API_URL}/sku/get_candidate_SKU",
            json={"location_id": location_id, "pose_type": ""},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        rows = response.json()
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"SKU 行查询请求失败: {error}") from error
    except ValueError as error:
        raise HTTPException(status_code=502, detail="SKU 行查询响应不是有效 JSON") from error
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], list)
        or not all(isinstance(product, dict) for product in rows[0])
    ):
        raise HTTPException(status_code=502, detail="SKU 行查询响应格式错误")
    return rows[0]


def normalize_level(level: str) -> str:
    normalized = level.strip().upper()
    if not re.fullmatch(r"L[1-5]", normalized):
        raise HTTPException(status_code=400, detail="level 必须是 L1 到 L5")
    return normalized


def load_hard_case_scope() -> set[tuple[str, str, str]]:
    try:
        payload = json.loads(HARD_CASE_SCOPE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail="hard case 范围配置不存在") from error
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"读取 hard case 范围配置失败: {error}") from error
    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail="hard case 范围配置必须是数组")
    scope: set[tuple[str, str, str]] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise HTTPException(status_code=500, detail="hard case 范围条目格式错误")
        name = item.get("product_name")
        level = item.get("level")
        hand = item.get("hand")
        if not all(isinstance(value, str) and value.strip() for value in (name, level, hand)):
            raise HTTPException(status_code=500, detail="hard case 范围条目缺少字段")
        normalized_level = level.strip().upper()
        normalized_hand = hand.strip().lower()
        if not re.fullmatch(r"L[1-5]", normalized_level) or normalized_hand not in CAMERA_SNAPSHOT_URLS:
            raise HTTPException(status_code=500, detail="hard case 范围条目的 level 或 hand 无效")
        key = (name.strip(), normalized_level, normalized_hand)
        if key in scope:
            raise HTTPException(status_code=500, detail=f"hard case 范围存在重复条目: {key}")
        scope.add(key)
    return scope


def hard_case_group_for_product(
    product_name: str,
    task_type: str,
    level: str,
    hand: str,
) -> tuple[str, HardCaseGroupConfig] | None:
    if task_type != "SORTING" or (
        product_name.strip(), level.strip().upper(), hand.strip().lower()
    ) not in load_hard_case_scope():
        return None
    for group_id, config in HARD_CASE_GROUPS.items():
        if product_name in config.members:
            return group_id, config
    return None


def select_location_for_level(
    product: dict[str, Any], level: str
) -> tuple[str, re.Match[str]]:
    locations = product.get("locations")
    if not isinstance(locations, list) or not locations:
        raise HTTPException(status_code=502, detail="hard case SKU 缺少 locations")
    normalized_level = normalize_level(level)
    parsed: list[tuple[str, re.Match[str]]] = []
    for value in locations:
        if not isinstance(value, str):
            continue
        match = LOCATION_PATTERN.fullmatch(value.strip().upper())
        if match is not None and f"L{match.group('level')}" == normalized_level:
            parsed.append((value.strip().upper(), match))
    if not parsed:
        raise HTTPException(status_code=404, detail=f"hard case SKU 在 {normalized_level} 没有 location")
    if len(parsed) != 1:
        raise HTTPException(status_code=502, detail=f"hard case SKU 在 {normalized_level} 存在多个 location")
    return parsed[0]


def hard_case_standard_order(
    target_location: str,
    config: HardCaseGroupConfig,
) -> list[str]:
    row = lookup_sku_row(target_location)
    order = [
        product["name"]
        for product in row
        if isinstance(product.get("name"), str)
        and product["name"] in config.members
    ]
    if not order:
        raise HTTPException(status_code=502, detail="目标层没有 hard case 品牌成员")
    return order


def normalize_task_type(task_type: str) -> str:
    normalized = task_type.strip().upper()
    if normalized not in PROMPT_MAPPING_PATHS:
        raise HTTPException(
            status_code=400,
            detail=f"task_type 只能是: {', '.join(SUPPORTED_TASK_TYPES)}",
        )
    return normalized


def prompt_mapping_path(task_type: str) -> Path:
    normalized = normalize_task_type(task_type)
    if normalized == "SORTING":
        return PROMPT_MAPPING_PATH
    return PROMPT_MAPPING_PATHS[normalized]


def load_prompt_pair(
    name: str, task_type: str = "SORTING", *, hard_case: bool = False
) -> tuple[str, str]:
    normalized_task_type = normalize_task_type(task_type)
    mapping_path = (
        HARD_CASE_PROMPT_MAPPING_PATH
        if hard_case and normalized_task_type == "SORTING"
        else prompt_mapping_path(normalized_task_type)
    )
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=f"{normalized_task_type} Prompt 配对文件不存在",
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"读取 Prompt 配对失败: {error}") from error

    pair = mapping.get(name) if isinstance(mapping, dict) else None
    if not isinstance(pair, dict):
        raise HTTPException(
            status_code=400,
            detail=f"{normalized_task_type} 尚未配置商品 Prompt: {name}",
        )
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
    print(f"[Locate Qwen3] prompt before request:\n{prompt}", flush=True)
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
    source_qwen_index: int | None = None,
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
        source_qwen_index=source_qwen_index,
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


def estimate_instance_depth_mm(
    instance: LocatedInstance,
    depth_image: Image.Image,
    *,
    min_valid_pixels: int = PICK_MIN_VALID_DEPTH_PIXELS,
) -> float | None:
    """计算实例 mask 内非零深度的中位数，单位为毫米。"""
    encoded = instance.mask.split(",", 1)[-1]
    try:
        mask_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(mask_bytes)) as source_mask:
            mask_image = source_mask.convert("L")
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError):
        return None
    if mask_image.size != depth_image.size:
        return None

    valid_depths = sorted(
        float(depth_value) * CAMERA_DEPTH_UNIT_MM
        for mask_value, depth_value in zip(
            mask_image.getdata(),
            depth_image.getdata(),
        )
        if mask_value >= 128
        and isinstance(depth_value, (int, float))
        and depth_value > 0
    )
    if len(valid_depths) < max(1, min_valid_pixels):
        return None

    middle = len(valid_depths) // 2
    if len(valid_depths) % 2:
        return valid_depths[middle]
    return (valid_depths[middle - 1] + valid_depths[middle]) / 2


def select_frontmost_instance(
    instances: list[LocatedInstance],
) -> LocatedInstance:
    """重叠链优先选择深度前排，再按 mask 面积和密度选择实例。"""
    if len(instances) == 1:
        return instances[0]

    instances = keep_front_row_pick_candidates(instances)
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


def keep_frontmost_in_source_groups(
    instances: list[LocatedInstance],
) -> list[LocatedInstance]:
    """Apply overlap-chain filtering independently inside each Qwen display crop."""
    by_source: dict[int | None, list[LocatedInstance]] = defaultdict(list)
    for instance in instances:
        by_source[instance.source_qwen_index].append(instance)
    filtered = [
        item
        for source_instances in by_source.values()
        for item in keep_frontmost_in_overlap_chains(source_instances)
    ]
    return sorted(filtered, key=lambda item: (item.source_qwen_index or 0, item.bbox[0]))


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


def instance_center_x(instance: LocatedInstance) -> float:
    return (instance.bbox[0] + instance.bbox[2]) / 2


def union_instance_bboxes(instances: list[LocatedInstance]) -> list[float]:
    return [
        min(instance.bbox[0] for instance in instances),
        min(instance.bbox[1] for instance in instances),
        max(instance.bbox[2] for instance in instances),
        max(instance.bbox[3] for instance in instances),
    ]


def split_instances_into_display_groups(
    instances: list[LocatedInstance],
    shelf_front_line: tuple[float, float] | None = None,
) -> list[list[LocatedInstance]]:
    """Return the visible product columns from left to right.

    Hard-case Qwen prompts request one bbox per display column, so source crops
    are the primary column boundary. If Qwen returned one combined crop, each
    SAM frontmost instance is conservatively treated as one visible column.
    """
    if not instances:
        raise HTTPException(status_code=422, detail="没有可用于排序的第一排实例")

    by_source: dict[int, list[LocatedInstance]] = defaultdict(list)
    for instance in instances:
        if instance.source_qwen_index is not None:
            by_source[instance.source_qwen_index].append(instance)
    nonempty_source_groups = [group for group in by_source.values() if group]
    if len(nonempty_source_groups) > 1:
        return sorted(
            nonempty_source_groups,
            key=lambda group: sum(instance_center_x(item) for item in group) / len(group),
        )

    front_row = keep_front_depth_row(instances, shelf_front_line)
    # With one broad hard-case Qwen crop, rear-row and partial masks can bridge
    # otherwise independent front products into one overlap chain. Determine
    # depth first, then remove duplicate SAM masks only among front-row items.
    ordered = sorted(
        keep_frontmost_in_overlap_chains(front_row),
        key=instance_center_x,
    )
    return [[instance] for instance in ordered]


def detect_red_shelf_front_line(
    image: Image.Image,
) -> tuple[float, float] | None:
    """Detect the red shelf-edge strip as ``y = slope * x + intercept``.

    A small Hough-like search is used instead of assuming a horizontal shelf:
    wrist-camera perspective commonly makes the red strip slope across the image.
    The winning line must be supported across a meaningful horizontal span, which
    prevents isolated red packaging from becoming the shelf reference.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 20 or height < 20:
        return None
    step = max(1, min(width, height) // 360)
    x_bucket_size = max(12, width // 48)
    intercept_bin_size = max(5, height // 140)
    candidates: list[tuple[float, float]] = []
    pixels = rgb.load()
    for y in range(int(height * 0.55), height, step):
        for x in range(0, width, step):
            red, green, blue = pixels[x, y]
            if red >= 90 and red - green >= 35 and red - blue >= 25:
                candidates.append((float(x), float(y)))
    if not candidates:
        return None

    best: tuple[int, float, int] | None = None
    for slope_index in range(-10, 11):
        slope = slope_index * 0.025
        bins: dict[int, set[int]] = defaultdict(set)
        for x, y in candidates:
            intercept_bin = round((y - slope * x) / intercept_bin_size)
            bins[intercept_bin].add(int(x) // x_bucket_size)
        for intercept_bin, x_buckets in bins.items():
            candidate = (len(x_buckets), slope, intercept_bin)
            if best is None or candidate[0] > best[0]:
                best = candidate
    minimum_support = max(6, int(width / x_bucket_size * 0.18))
    if best is None or best[0] < minimum_support:
        return None
    _, slope, intercept_bin = best
    approximate_intercept = intercept_bin * intercept_bin_size
    residuals = sorted(
        y - slope * x
        for x, y in candidates
        if abs((y - slope * x) - approximate_intercept) <= intercept_bin_size
    )
    if not residuals:
        return None
    return slope, residuals[len(residuals) // 2]


def keep_front_depth_row(
    instances: list[LocatedInstance],
    shelf_front_line: tuple[float, float] | None = None,
) -> list[LocatedInstance]:
    """Keep instances whose lower edge reaches the detected shelf front."""
    if len(instances) <= 1:
        return instances
    heights = [max(0.0, item.bbox[3] - item.bbox[1]) for item in instances]
    if shelf_front_line is not None:
        slope, intercept = shelf_front_line
        front = []
        for item, height in zip(instances, heights):
            shelf_y = slope * instance_center_x(item) + intercept
            # The mask/bbox may end slightly above the visible red edge or extend
            # through it. Perspective and image-edge clipping therefore scale the
            # tolerance with each individual object, not the largest object.
            upper_tolerance = max(12.0, height * 0.25)
            lower_tolerance = max(12.0, height * 0.35)
            if shelf_y - upper_tolerance <= item.bbox[3] <= shelf_y + lower_tolerance:
                front.append(item)
        if front:
            return front

    # Conservative fallback for shelves where no red strip is detectable. Size
    # and mask area are deliberately not hard rejection criteria because an edge
    # SKU can be truncated or smaller under perspective.
    max_bottom = max(item.bbox[3] for item in instances)
    max_height = max(heights)
    bottom_tolerance = max(8.0, max_height * 0.25)
    front = [
        item
        for item in instances
        if item.bbox[3] >= max_bottom - bottom_tolerance
    ]
    return front or [max(instances, key=lambda item: item.bbox[3])]


def apply_hard_case_ordering(
    instances: list[LocatedInstance],
    *,
    product: dict[str, Any],
    task_type: str,
    level: str,
    hand: str,
    shelf_front_line: tuple[float, float] | None = None,
) -> tuple[list[LocatedInstance], HardCaseDebugInfo | None]:
    hard_case = hard_case_group_for_product(
        product["name"].strip(), task_type, level, hand
    )
    if hard_case is None:
        return instances, None
    group_id, config = hard_case
    actual_hand = hand.strip().lower()
    preferred_hand = actual_hand

    target_location, target_match = select_location_for_level(product, level)
    standard_order = hard_case_standard_order(target_location, config)
    target_name = product["name"].strip()
    if target_name not in standard_order:
        raise HTTPException(status_code=502, detail="目标 SKU 不在指定层品牌顺序中")

    display_groups = split_instances_into_display_groups(instances, shelf_front_line)
    visible_count = len(display_groups)
    if visible_count > len(standard_order):
        raise HTTPException(
            status_code=422,
            detail=(
                f"检测列数超过标准品牌列数: "
                f"visible={visible_count}, standard={len(standard_order)}"
            ),
        )
    if actual_hand == "left":
        directional_groups = display_groups
        directional_order = standard_order[:visible_count]
    else:
        directional_groups = list(reversed(display_groups))
        directional_order = list(reversed(standard_order[-visible_count:]))

    if target_name not in directional_order:
        raise HTTPException(
            status_code=404,
            detail=(
                f"目标 SKU 不在当前相机可见列范围内: {target_name}; "
                f"visible_order={directional_order}"
            ),
        )

    annotated_instances: list[LocatedInstance] = []
    debug_groups: list[HardCaseGroupResult] = []
    selected_group_index = -1
    selected_instance: LocatedInstance | None = None
    for group_index, (group, mapped_name) in enumerate(
        zip(directional_groups, directional_order),
        start=1,
    ):
        front_instances = keep_front_depth_row(group, shelf_front_line)
        if not front_instances:
            raise HTTPException(status_code=422, detail=f"{mapped_name} 没有第一排实例")
        if mapped_name == target_name:
            selected_group_index = group_index
            selected_instance = select_frontmost_instance(front_instances)
        updated_group: list[LocatedInstance] = []
        for instance in front_instances:
            updated = instance.model_copy(
                update={
                    "hard_case_group_index": group_index,
                    "mapped_product_name": mapped_name,
                    "is_selected": instance is selected_instance,
                }
            )
            updated_group.append(updated)
            annotated_instances.append(updated)
        debug_groups.append(
            HardCaseGroupResult(
                index=group_index,
                mapped_product_name=mapped_name,
                bbox=union_instance_bboxes(updated_group),
                instance_count=len(updated_group),
            )
        )

    if selected_group_index < 0 or selected_instance is None:
        raise HTTPException(status_code=422, detail="无法在第一排陈列组中定位目标 SKU")
    return annotated_instances, HardCaseDebugInfo(
        group_id=group_id,
        preferred_hand=preferred_hand,
        actual_hand=actual_hand,
        order_direction="left_to_right" if actual_hand == "left" else "right_to_left",
        target_location=target_location,
        target_level=int(target_match.group("level")),
        target_column=int(target_match.group("column")),
        standard_order=standard_order,
        groups=debug_groups,
        selected_group_index=selected_group_index,
    )


def locate_product_in_image(
    product: dict[str, Any],
    image_path: Path,
    *,
    task_type: str = "SORTING",
    level: str,
    hand: str = "left",
    qwen_prompt_override: str | None = None,
    sam_prompt_override: str | None = None,
    depth_image: Image.Image | None = None,
    depth_image_provider: Callable[[tuple[int, int]], Image.Image | None] | None = None,
    capture_postprocess_errors: bool = False,
) -> LocateDebugResponse:
    """使用已查询的 SKU 信息，在指定 RGB 图片上运行完整定位流程。"""
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"测试图片不存在: {image_path.name}")
    monitor_image_path = store_monitor_image(image_path)
    canonical_name = product["name"].strip()
    normalized_level = normalize_level(level)
    hard_case = hard_case_group_for_product(
        canonical_name, task_type, normalized_level, hand
    )
    qwen_prompt = (qwen_prompt_override or "").strip()
    sam_prompt = (sam_prompt_override or "").strip()
    if not qwen_prompt or not sam_prompt:
        stored_qwen_prompt, stored_sam_prompt = load_prompt_pair(
            canonical_name,
            task_type,
            hard_case=hard_case is not None,
        )
        qwen_prompt = qwen_prompt or stored_qwen_prompt
        sam_prompt = sam_prompt or stored_sam_prompt
    if hard_case is not None:
        hard_case_group_id, _ = hard_case
        target_location, target_match = select_location_for_level(
            product, normalized_level
        )
        if hard_case_group_id == "bbq_sauce_spicy":
            bbox_instruction = "每个红色烧烤酱袋子堆分别输出 bbox。"
        elif hard_case_group_id == "bbq_sauce_original":
            bbox_instruction = "每个绿色烧烤酱袋子堆分别输出 bbox。"
        else:
            bbox_instruction = "把该层所有同组商品合并在一个完整 bbox 中，只输出这一个 bbox。"
        qwen_prompt = (
            f"{qwen_prompt}\n"
            f"该 hard case 商品本次只处理标准库位置 {target_location} "
            f"对应的 L{target_match.group('level')} 层；如果该 SKU 有多个位置，"
            "这里使用调用方指定的层。不要输出同品牌在其他货架层的商品；"
            f"{bbox_instruction}"
        )
    qwen_bboxes = get_stable_qwen_bboxes(qwen_prompt, image_path)
    if hard_case is not None and len(qwen_bboxes) > 1:
        # Hard cases use Qwen only to obtain one broad brand-row crop. Product
        # columns/flavours are determined from SAM instances afterwards.
        qwen_bboxes = QwenConsensusBBoxes(
            [[
                min(box[0] for box in qwen_bboxes),
                min(box[1] for box in qwen_bboxes),
                max(box[2] for box in qwen_bboxes),
                max(box[3] for box in qwen_bboxes),
            ]],
            getattr(qwen_bboxes, "samples", []),
        )

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
    for qwen_index, qwen_bbox in enumerate(qwen_bboxes):
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
                    map_sam_instance_to_original(
                        instance,
                        crop_box,
                        original_image.size,
                        source_qwen_index=qwen_index,
                    )
                )

    if not located_instances:
        raise HTTPException(status_code=404, detail="SAM3 没有找到目标商品实例")

<<<<<<< HEAD
=======
    raw_sam_instances = list(located_instances)
>>>>>>> 398ebec0f8eeb004d4de9c0078cbc4b561a51d0a
    depth_enabled_for_task = task_type.strip().upper() != "SHORTAGE"
    if hard_case is None and depth_enabled_for_task and len(located_instances) > 1:
        if depth_image is None and depth_image_provider is not None:
            depth_image = depth_image_provider(original_image.size)
        if depth_image is not None and depth_image.size == original_image.size:
            located_instances = [
                instance.model_copy(
                    update={
                        "depth_mm": estimate_instance_depth_mm(instance, depth_image)
                    }
                )
                for instance in located_instances
            ]
<<<<<<< HEAD
    raw_sam_instances = list(located_instances)
    located_instances = keep_frontmost_in_overlap_chains(located_instances)
    located_instances = drop_smallest_mask_area_outlier(located_instances)
    selected_instance = select_pick_instance(
        located_instances,
        list(original_image.size),
=======
            raw_sam_instances = list(located_instances)
            # Depth must remove the rear row before overlap-chain de-duplication;
            # otherwise a rear/partial mask can bridge several front products and
            # make the entire component collapse to the wrong instance.
            located_instances = keep_front_row_pick_candidates(located_instances)
    if hard_case is None:
        located_instances = keep_frontmost_in_overlap_chains(located_instances)
        # Preserve the original global outlier rule for normal SKUs. A hard-case
        # row can contain a legitimately smaller perspective-edge display group,
        # so its rear-row filtering is performed inside each mapped group below.
        located_instances = drop_smallest_mask_area_outlier(located_instances)
    else:
        # Hard cases now use one broad Qwen crop. Preserve all SAM instances here;
        # first-row filtering must run before overlap de-duplication so rear/partial
        # masks cannot transitively merge separate front-row product columns.
        located_instances = list(located_instances)
    shelf_front_line = (
        detect_red_shelf_front_line(original_image)
        if hard_case is not None
        else None
>>>>>>> 398ebec0f8eeb004d4de9c0078cbc4b561a51d0a
    )
    hard_case_debug: HardCaseDebugInfo | None = None
    postprocess_error: HTTPException | None = None
    try:
        located_instances, hard_case_debug = apply_hard_case_ordering(
            located_instances,
            product=product,
            task_type=task_type,
            level=normalized_level,
            hand=hand,
            shelf_front_line=shelf_front_line,
        )
    except HTTPException as error:
        if not capture_postprocess_errors:
            raise
        postprocess_error = error

    if hard_case is not None and depth_enabled_for_task and len(located_instances) > 1:
        if depth_image is None and depth_image_provider is not None:
            depth_image = depth_image_provider(original_image.size)
        if depth_image is not None and depth_image.size == original_image.size:
            located_instances = [
                instance.model_copy(
                    update={
                        "depth_mm": estimate_instance_depth_mm(instance, depth_image)
                    }
                )
                for instance in located_instances
            ]

    if hard_case_debug is not None:
        hard_case_selected = [
            instance for instance in located_instances if instance.is_selected
        ]
        if len(hard_case_selected) != 1:
            raise HTTPException(status_code=500, detail="hard case 目标实例标记无效")
        selected_instance = hard_case_selected[0]
    else:
        selected_instance = select_pick_instance(
            located_instances,
            list(original_image.size),
        )
    selected_instance_index = next(
        index
        for index, instance in enumerate(located_instances, start=1)
        if instance is selected_instance
    )

    return LocateDebugResponse(
        sku_id=product["sku_id"],
        product_name=canonical_name,
        image_name=image_path.name,
        image_path=monitor_image_path,
        image_base64=base64.b64encode(image_path.read_bytes()).decode("ascii"),
        image_media_type=mimetypes.guess_type(image_path.name)[0] or "image/jpeg",
        image_size=list(original_image.size),
        qwen3_prompt_used=qwen_prompt,
        sam3_prompt_used=sam_prompt,
        raw_qwen_bboxes=raw_qwen_bbox_records,
        qwen_bboxes=qwen_bbox_records,
        raw_sam_instances=raw_sam_instances,
        instances=located_instances,
        selected_instance=selected_instance,
        selected_instance_index=selected_instance_index,
        hard_case=hard_case_debug,
        error=(
            str(postprocess_error.detail)
            if postprocess_error is not None
            else None
        ),
        error_status_code=(
            postprocess_error.status_code
            if postprocess_error is not None
            else None
        ),
    )


def make_locate_debug_error_response(
    product: dict[str, Any],
    image_path: Path,
    error: HTTPException,
    *,
    qwen3_prompt_used: str | None = None,
    sam3_prompt_used: str | None = None,
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
        qwen3_prompt_used=qwen3_prompt_used,
        sam3_prompt_used=sam3_prompt_used,
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
    task_type = normalize_task_type(request.task_type)
    level = normalize_level(request.level)
    if not product_name:
        raise HTTPException(status_code=400, detail="product_name 不能为空")
    product = lookup_sku_by_name(product_name)
    prompt_overrides = (
        {
            "task_type": task_type,
            "qwen_prompt_override": request.qwen3_prompt,
            "sam_prompt_override": request.sam3_prompt,
        }
        if allow_prompt_overrides
        else {"task_type": task_type}
    )
    prompt_overrides["hand"] = request.hand
    prompt_overrides["level"] = level
    prompt_overrides["capture_postprocess_errors"] = capture_inference_errors
    has_depth_name = request.depth_image_name is not None
    has_depth_base64 = request.depth_image_base64 is not None
    if has_depth_name != has_depth_base64:
        raise HTTPException(
            status_code=400,
            detail="depth_image_name 和 depth_image_base64 必须同时提供或同时省略",
        )
    if request.image_base64 is None:
        if request.image_name is not None:
            raise HTTPException(
                status_code=400,
                detail="指定 image_name 时必须同时提供 image_base64",
            )
        if has_depth_name:
            raise HTTPException(
                status_code=400,
                detail="离线深度数据必须与离线 RGB 图片同时提供",
            )
        image_path = get_latest_rgb(request.hand)
        try:
            return locate_product_in_image(
                product,
                image_path,
                depth_image_provider=lambda expected_size: fetch_camera_depth(
                    request.hand,
                    expected_size,
                ),
                **prompt_overrides,
            )
        except HTTPException as error:
            if not capture_inference_errors:
                raise
            return make_locate_debug_error_response(
                product,
                image_path,
                error,
                qwen3_prompt_used=request.qwen3_prompt if allow_prompt_overrides else None,
                sam3_prompt_used=request.sam3_prompt if allow_prompt_overrides else None,
            )

    image_bytes = decode_uploaded_image(request.image_base64)
    image_name = uploaded_image_name(request.image_name)
    with tempfile.TemporaryDirectory(prefix="locate-upload-") as temporary_directory:
        image_path = Path(temporary_directory) / image_name
        image_path.write_bytes(image_bytes)
        depth_image: Image.Image | None = None
        if has_depth_name and has_depth_base64:
            try:
                with Image.open(image_path) as source_image:
                    expected_size = source_image.size
            except (UnidentifiedImageError, OSError) as error:
                raise HTTPException(status_code=400, detail=f"读取上传 RGB 图片失败: {error}") from error
            depth_image = decode_uploaded_depth_image(
                request.depth_image_base64 or "",
                request.depth_image_name or "",
                expected_size,
                is_bigendian=request.depth_is_bigendian,
            )
        try:
            return locate_product_in_image(
                product,
                image_path,
                depth_image=depth_image,
                **prompt_overrides,
            )
        except HTTPException as error:
            if not capture_inference_errors:
                raise
            return make_locate_debug_error_response(
                product,
                image_path,
                error,
                qwen3_prompt_used=request.qwen3_prompt if allow_prompt_overrides else None,
                sam3_prompt_used=request.sam3_prompt if allow_prompt_overrides else None,
            )


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


def bbox_width_height_ratio(instance: LocatedInstance) -> float:
    """返回 bbox 宽高比；退化 bbox 视为不可见。"""
    bbox_width = max(0.0, instance.bbox[2] - instance.bbox[0])
    bbox_height = max(0.0, instance.bbox[3] - instance.bbox[1])
    return bbox_width / bbox_height if bbox_height > 0 else 0.0


def keep_visibly_complete_pick_candidates(
    instances: list[LocatedInstance],
    min_ratio_to_best: float = PICK_MIN_ASPECT_RATIO_TO_BEST,
) -> list[LocatedInstance]:
    """过滤相对同类最佳外观明显过窄、通常被左右遮挡的实例。"""
    if len(instances) < 2:
        return instances

    aspect_ratios = [bbox_width_height_ratio(instance) for instance in instances]
    best_aspect_ratio = max(aspect_ratios)
    if best_aspect_ratio <= 0:
        return instances

    relative_threshold = max(0.0, min(1.0, min_ratio_to_best))
    minimum_aspect_ratio = best_aspect_ratio * relative_threshold
    filtered = [
        instance
        for instance, aspect_ratio in zip(instances, aspect_ratios)
        if aspect_ratio >= minimum_aspect_ratio
    ]
    return filtered or instances


def bbox_vertical_overlap_by_smaller_height(
    first_bbox: list[float],
    second_bbox: list[float],
) -> float:
    first_height = max(0.0, first_bbox[3] - first_bbox[1])
    second_height = max(0.0, second_bbox[3] - second_bbox[1])
    smaller_height = min(first_height, second_height)
    if smaller_height <= 0:
        return 0.0
    overlap = max(
        0.0,
        min(first_bbox[3], second_bbox[3])
        - max(first_bbox[1], second_bbox[1]),
    )
    return overlap / smaller_height


def bbox_horizontal_gap(first_bbox: list[float], second_bbox: list[float]) -> float:
    """返回两个 bbox 的水平净间距；水平重叠时为 0。"""
    if first_bbox[2] < second_bbox[0]:
        return second_bbox[0] - first_bbox[2]
    if second_bbox[2] < first_bbox[0]:
        return first_bbox[0] - second_bbox[2]
    return 0.0


def keep_depth_unoccluded_pick_candidates(
    instances: list[LocatedInstance],
    *,
    depth_margin_mm: float = PICK_OCCLUSION_DEPTH_MARGIN_MM,
    max_neighbor_gap_ratio: float = PICK_OCCLUSION_MAX_NEIGHBOR_GAP_RATIO,
    min_vertical_overlap_ratio: float = PICK_OCCLUSION_MIN_VERTICAL_OVERLAP_RATIO,
) -> list[LocatedInstance]:
    """移除左右两侧都存在更近邻居的后排实例。"""
    if len(instances) < 3:
        return instances

    selected: list[LocatedInstance] = []
    for candidate in instances:
        candidate_depth = candidate.depth_mm
        if (
            candidate_depth is None
            or not math.isfinite(candidate_depth)
            or candidate_depth <= 0
        ):
            selected.append(candidate)
            continue

        candidate_width = max(0.0, candidate.bbox[2] - candidate.bbox[0])
        candidate_center_x = (candidate.bbox[0] + candidate.bbox[2]) / 2
        closer_sides: set[str] = set()
        for neighbor in instances:
            if neighbor is candidate:
                continue
            neighbor_depth = neighbor.depth_mm
            if (
                neighbor_depth is None
                or not math.isfinite(neighbor_depth)
                or neighbor_depth <= 0
                or neighbor_depth + max(0.0, depth_margin_mm) >= candidate_depth
            ):
                continue
            if (
                bbox_vertical_overlap_by_smaller_height(
                    candidate.bbox,
                    neighbor.bbox,
                )
                < min_vertical_overlap_ratio
            ):
                continue
            neighbor_width = max(0.0, neighbor.bbox[2] - neighbor.bbox[0])
            allowed_gap = (
                max(0.0, max_neighbor_gap_ratio)
                * min(candidate_width, neighbor_width)
            )
            if bbox_horizontal_gap(candidate.bbox, neighbor.bbox) > allowed_gap:
                continue

            neighbor_center_x = (neighbor.bbox[0] + neighbor.bbox[2]) / 2
            if neighbor_center_x < candidate_center_x:
                closer_sides.add("left")
            elif neighbor_center_x > candidate_center_x:
                closer_sides.add("right")

        if closer_sides != {"left", "right"}:
            selected.append(candidate)

    return selected or instances


def keep_front_row_pick_candidates(
    instances: list[LocatedInstance],
    *,
    depth_tolerance_mm: float = PICK_FRONT_ROW_DEPTH_TOLERANCE_MM,
) -> list[LocatedInstance]:
    """保留深度最小的一层候选；完全无有效深度时沿用原候选。"""
    candidates_with_depth = [
        instance
        for instance in instances
        if instance.depth_mm is not None
        and math.isfinite(instance.depth_mm)
        and instance.depth_mm > 0
    ]
    if not candidates_with_depth:
        return instances

    nearest_depth_mm = min(
        instance.depth_mm for instance in candidates_with_depth
    )
    maximum_front_depth_mm = nearest_depth_mm + max(0.0, depth_tolerance_mm)
    return [
        instance
        for instance in candidates_with_depth
        if instance.depth_mm <= maximum_front_depth_mm
    ]


def select_pick_instance(
    instances: list[LocatedInstance],
    image_size: list[int],
) -> LocatedInstance:
    """先排除遮挡并保留最前排候选，再选择最靠近画面中心的实例。"""
    if not instances:
        raise HTTPException(status_code=404, detail="SAM3 没有找到目标商品实例")

    candidates = keep_depth_unoccluded_pick_candidates(instances)
    candidates = keep_visibly_complete_pick_candidates(candidates)
    candidates = keep_front_row_pick_candidates(candidates)
    image_center_x = image_size[0] / 2
    image_center_y = image_size[1] / 2
    return min(
        candidates,
        key=lambda instance: (
            ((instance.bbox[0] + instance.bbox[2]) / 2 - image_center_x) ** 2
            + ((instance.bbox[1] + instance.bbox[3]) / 2 - image_center_y) ** 2
        ),
    )


def make_locate_response(debug_response: LocateDebugResponse) -> LocateResponse:
    hard_case_selected = [
        instance for instance in debug_response.instances if instance.is_selected
    ]
    if debug_response.hard_case is not None:
        if len(hard_case_selected) != 1:
            raise HTTPException(status_code=500, detail="hard case 目标实例标记无效")
        selected_instance = hard_case_selected[0]
    else:
        selected_instance = (
            debug_response.selected_instance
            or select_pick_instance(
                debug_response.instances,
                debug_response.image_size,
            )
        )
    return LocateResponse(
        product_name=debug_response.product_name,
        bbox=normalize_bbox_to_1_1000(
            selected_instance.bbox,
            debug_response.image_size,
        ),
        mask=selected_instance.mask,
        image_path=debug_response.image_path,
    )


@router.post("/perception/pick/locate", response_model=LocateResponse)
def locate_product(request: LocateRequest) -> LocateResponse:
    return make_locate_response(locate_product_debug(request))


@router.post("/perception/pick/locate/debug", response_model=LocateDebugResponse)
def locate_product_debug_api(request: LocateRequest) -> LocateDebugResponse:
    return locate_product_debug(
        request,
        capture_inference_errors=True,
        allow_prompt_overrides=True,
    )


app.include_router(router)
