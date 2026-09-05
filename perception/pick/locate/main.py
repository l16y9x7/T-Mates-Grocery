from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import logging
import math
import mimetypes
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import cv2
import requests
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask


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
HARD_CASE_LAYOUT_OVERRIDES_PATH = (
    ROOT.parents[1] / "hard_case_layout_overrides.json"
)
HARD_CASE_VIEW_LAYOUT_PATH = ROOT.parents[1] / "hard_case_view_layout.json"
MULTI_ROW_PRODUCTS_PATH = ROOT / "multi_row_products.json"
MOCK_SKU_CATALOG_PATH = ROOT.parents[1] / "sku" / "products.json"
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
HEAD_CAMERA_SNAPSHOT_URL = camera_snapshot_url("head")
LEFT_CAMERA_DEPTH_SNAPSHOT_URL = camera_depth_snapshot_url("left")
RIGHT_CAMERA_DEPTH_SNAPSHOT_URL = camera_depth_snapshot_url("right")
HEAD_CAMERA_DEPTH_SNAPSHOT_URL = camera_depth_snapshot_url("head")
# 保留旧常量名称，兼容现有左手相机配置与测试。
CAMERA_SNAPSHOT_URL = LEFT_CAMERA_SNAPSHOT_URL
CAMERA_SNAPSHOT_URLS = {
    "left": LEFT_CAMERA_SNAPSHOT_URL,
    "right": RIGHT_CAMERA_SNAPSHOT_URL,
    "head": HEAD_CAMERA_SNAPSHOT_URL,
}
CAMERA_DEPTH_SNAPSHOT_URLS = {
    "left": LEFT_CAMERA_DEPTH_SNAPSHOT_URL,
    "right": RIGHT_CAMERA_DEPTH_SNAPSHOT_URL,
    "head": HEAD_CAMERA_DEPTH_SNAPSHOT_URL,
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
RGB_INFERENCE_SIZE = (1280, 720)
RGB_INFERENCE_JPEG_QUALITY = 95
CROP_PADDING_RATIO = 0.1
QWEN_CROP_PADDING_RATIO_OVERRIDES = {
    "小鹿妈妈牙线": 0.5,
}
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
PICK_MIN_SAM_QWEN_BBOX_COVERAGE = float(
    os.getenv(
        "PICK_MIN_SAM_QWEN_BBOX_COVERAGE",
        os.getenv("SHORTAGE_MIN_SAM_QWEN_BBOX_COVERAGE", "0.25"),
    )
)
PICK_MIN_SAM_TO_LARGEST_BBOX_AREA_RATIO = float(
    os.getenv("PICK_MIN_SAM_TO_LARGEST_BBOX_AREA_RATIO", "0.35")
)
PICK_MIN_ASPECT_RATIO_TO_BEST = float(
    os.getenv("PICK_MIN_ASPECT_RATIO_TO_BEST", "0.75")
)
PICK_MIN_HEIGHT_RATIO_TO_TALLEST = float(
    os.getenv("PICK_MIN_HEIGHT_RATIO_TO_TALLEST", "0.60")
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
PICK_UPPER_CONFIDENCE_SCORE_MARGIN = float(
    os.getenv("PICK_UPPER_CONFIDENCE_SCORE_MARGIN", "0.10")
)
PICK_UPPER_VERTICAL_TIE_TOLERANCE_RATIO = float(
    os.getenv("PICK_UPPER_VERTICAL_TIE_TOLERANCE_RATIO", "0.10")
)
PICK_MIN_MASK_BBOX_FILL_RATIO = float(
    os.getenv("PICK_MIN_MASK_BBOX_FILL_RATIO", "0.15")
)
PICK_MIN_OVERLAP_MASK_AREA_RATIO = float(
    os.getenv("PICK_MIN_OVERLAP_MASK_AREA_RATIO", "0.20")
)
HARD_CASE_FRONT_UPPER_TOLERANCE_RATIO = float(
    os.getenv("HARD_CASE_FRONT_UPPER_TOLERANCE_RATIO", "0.25")
)
HARD_CASE_FRONT_MAX_UPPER_TOLERANCE_RATIO = float(
    os.getenv("HARD_CASE_FRONT_MAX_UPPER_TOLERANCE_RATIO", "0.35")
)
HARD_CASE_FRONT_LOWER_TOLERANCE_RATIO = float(
    os.getenv("HARD_CASE_FRONT_LOWER_TOLERANCE_RATIO", "0.35")
)
HARD_CASE_FRONT_DISTANCE_GAP_RATIO = float(
    os.getenv("HARD_CASE_FRONT_DISTANCE_GAP_RATIO", "0.10")
)
HARD_CASE_MASK_LOWER_CONTACT_QUANTILE = float(
    os.getenv("HARD_CASE_MASK_LOWER_CONTACT_QUANTILE", "0.80")
)
# Only excess edge columns may be removed; compare shape, not absolute size.
HARD_CASE_EDGE_MAX_ASPECT_RATIO_TO_MEDIAN = 0.60
MULTI_ROW_CONTACT_GAP_RATIO = float(
    os.getenv("MULTI_ROW_CONTACT_GAP_RATIO", "0.05")
)
# Boundary contact tolerates a 3 px segmentation gap on a 640x480 image.
PICK_MASK_CONTACT_RADIUS = 3
PICK_MIN_MASK_CONTACT_RATIO = 0.15
PICK_DUPLICATE_MASK_IOU = 0.85
MULTI_ROW_MASK_LOWER_CONTACT_QUANTILE = float(
    os.getenv("MULTI_ROW_MASK_LOWER_CONTACT_QUANTILE", "0.50")
)
MULTI_ROW_DEPTH_GAP_MM = float(
    os.getenv("MULTI_ROW_DEPTH_GAP_MM", "45")
)
PICK_HISTORY_OVERLAP_RATIO = float(
    os.getenv("PICK_HISTORY_OVERLAP_RATIO", "0.50")
)
CAMERA_DEPTH_UNIT_MM = float(os.getenv("CAMERA_DEPTH_UNIT_MM", "1.0"))
REQUEST_TIMEOUT_SECONDS = 120

# These products are presented as vertically stacked packs/cups. SAM3 often
# returns both the exposed top item and a larger mask spanning the lower stack.
# For SORTING, pick the upper near-best-score instance directly and do not use
# depth, which describes shelf frontage rather than the top item to grasp.
UPPER_CONFIDENCE_PICK_PRODUCTS = frozenset(
    {
        "得宝纸巾",
        "海氏海诺创口贴",
        "德佑湿巾",
        "心相印纸巾",
        "农心碗面",
        "妙洁海绵百洁布",
        "康师傅香辣牛肉面",
        "康师傅鲜虾鱼板面",
        "康师傅老坛酸菜牛肉面",
        "纯棉酒店大毛巾",
        "京东京造毛巾",
        "心相印厨房纸巾",
        "拖鞋",
    }
)

# These bagged products frequently produce several overlapping top/edge
# fragments. Their complete pick region is represented most reliably by the
# actual SAM foreground-mask area, not bbox area, score, height, or depth.
MAX_MASK_AREA_PICK_PRODUCTS = frozenset({"中盐精制盐", "小苏打"})

LOCATION_PATTERN = re.compile(
    r"^H(?P<shelf>[1-3])_L(?P<level>0[1-5])_C(?P<column>\d{2})$"
)
INSPECTION_TARGET_PATTERN = re.compile(r"^H(?:1|12|2|23|3)_INSPECT$")


@dataclass(frozen=True)
class HardCaseGroupConfig:
    members: tuple[str, ...]
    preferred_hand: str
    hand_overrides: tuple[tuple[str, str], ...] = ()

    def hand_for_product(self, product_name: str) -> str:
        return dict(self.hand_overrides).get(product_name, self.preferred_hand)


@dataclass(frozen=True)
class HardCaseViewLayout:
    target_id: str
    hand: str
    level: str
    group_id: str
    visible_slot_order: tuple[str, ...]


# This allow-list is deliberately narrow. Non-SORTING tasks and every SKU not
# listed here continue through the original prompt and center-selection path.
HARD_CASE_GROUPS: dict[str, HardCaseGroupConfig] = {
    "maiydong": HardCaseGroupConfig(
        members=(
            "脉动观梅止渴饮",
            "脉动芒果口味",
            "脉动菠萝口味",
        ),
        preferred_hand="left",
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
    "shuke_toothpaste": HardCaseGroupConfig(
        members=(
            "舒克牙膏竹炭薄荷",
            "舒克牙膏柠檬百香果",
            "舒克牙膏海盐薄荷",
        ),
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
logger = logging.getLogger("uvicorn.error")


class LocateRequest(BaseModel):
    task_type: str
    product_name: str
    level: str | None = None
    hand: str
    slot_id: str | None = None
    target_id: str | None = None
    image_name: str | None = None
    image_base64: str | None = None
    depth_image_name: str | None = None
    depth_image_base64: str | None = None
    depth_is_bigendian: bool = False
    qwen3_prompt: str | None = None
    sam3_prompt: str | None = None
    previous_picked_bboxes: list[list[int]] = Field(default_factory=list)


class LocateDebugRequest(LocateRequest):
    mock_inventory: list[str] | None = None


class LocatedInstance(BaseModel):
    bbox: list[float]
    mask: str
    score: float | None = None
    depth_mm: float | None = None
    source_qwen_index: int | None = None
    hard_case_group_index: int | None = None
    mapped_product_name: str | None = None
    mapped_slot_id: str | None = None
    display_row_index: int | None = None
    display_position_in_row: int | None = None
    display_row_source: str | None = None
    shelf_front_distance_ratio: float | None = None
    history_overlap_count: int = 0
    is_selected: bool = False


class HardCaseGroupResult(BaseModel):
    index: int
    mapped_product_name: str
    mapped_slot_id: str | None = None
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
    layout_override_applied: bool = False
    groups: list[HardCaseGroupResult] = Field(default_factory=list)
    selected_group_index: int
    target_slot_id: str | None = None
    target_id: str | None = None
    visible_slot_order: list[str] = Field(default_factory=list)
    slot_view_applied: bool = False


class QwenBBoxRecord(BaseModel):
    bbox_normalized: list[float]
    bbox_original: list[float]
    crop_box_original: list[int]


class RawQwenBBoxRecord(BaseModel):
    sample_index: int
    name: str
    bbox_normalized: list[float]
    bbox_original: list[float]


@dataclass(frozen=True)
class QwenReferenceImage:
    logical_name: str
    media_type: str
    content: bytes


class LocateDebugResponse(BaseModel):
    sku_id: str
    product_name: str
    image_name: str
    image_path: str
    image_base64: str
    image_media_type: str
    image_size: list[int]
    inference_image_size: list[int] | None = None
    qwen3_prompt_used: str | None = None
    sam3_prompt_used: str | None = None
    qwen_reference_image_used: bool = False
    qwen_reference_image_name: str | None = None
    qwen_reference_image_media_type: str | None = None
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
    slot_id: str | None = None
    bbox: list[int]
    mask: str
    image_path: str


def locate_request_log_summary(request: LocateRequest) -> dict[str, Any]:
    """Return request parameters without writing image or prompt bodies to logs."""
    summary: dict[str, Any] = {
        "task_type": request.task_type,
        "product_name": request.product_name,
        "level": request.level,
        "hand": request.hand,
        "slot_id": request.slot_id,
        "target_id": request.target_id,
        "image_name": request.image_name,
        "image_provided": request.image_base64 is not None,
        "image_base64_chars": len(request.image_base64 or ""),
        "depth_image_name": request.depth_image_name,
        "depth_image_provided": request.depth_image_base64 is not None,
        "depth_image_base64_chars": len(request.depth_image_base64 or ""),
        "depth_is_bigendian": request.depth_is_bigendian,
        "qwen3_prompt_override": bool((request.qwen3_prompt or "").strip()),
        "qwen3_prompt_chars": len(request.qwen3_prompt or ""),
        "sam3_prompt_override": bool((request.sam3_prompt or "").strip()),
        "sam3_prompt_chars": len(request.sam3_prompt or ""),
        "previous_picked_bboxes": request.previous_picked_bboxes,
    }
    if isinstance(request, LocateDebugRequest):
        summary["mock_inventory"] = request.mock_inventory
    return summary


def locate_debug_response_log_summary(
    response: LocateDebugResponse,
) -> dict[str, Any]:
    """Return the useful locate result fields without masks, images, or prompts."""
    selected = response.selected_instance
    return {
        "sku_id": response.sku_id,
        "product_name": response.product_name,
        "image_name": response.image_name,
        "image_size": response.image_size,
        "raw_qwen_bbox_count": len(response.raw_qwen_bboxes),
        "qwen_bbox_count": len(response.qwen_bboxes),
        "raw_sam_instance_count": len(response.raw_sam_instances),
        "final_instance_count": len(response.instances),
        "selected_instance_index": response.selected_instance_index,
        "selected_slot_id": selected.mapped_slot_id if selected is not None else None,
        "selected_bbox": selected.bbox if selected is not None else None,
        "hard_case": response.hard_case is not None,
        "error": response.error,
        "error_status_code": response.error_status_code,
    }


def get_latest_rgb(camera: str = "left") -> Path:
    """只读取相机快照接口；不可用或内容无效时返回 HTTP 400。"""
    normalized_camera = camera.strip().lower()
    if normalized_camera not in CAMERA_SNAPSHOT_URLS:
        raise HTTPException(status_code=400, detail="camera 只能是 left、right 或 head")
    camera_snapshot = fetch_camera_snapshot(normalized_camera)
    if camera_snapshot is not None:
        return camera_snapshot
    raise HTTPException(
        status_code=400,
        detail="未提供图片，且相机快照接口读取失败或未返回有效 JPG/PNG",
    )


def fetch_camera_snapshot(camera: str = "left") -> Path | None:
    """获取并验证相机快照；任何读取错误都返回 None。"""
    normalized_camera = camera.strip().lower()
    camera_url = CAMERA_SNAPSHOT_URLS.get(normalized_camera)
    if camera_url is None:
        raise HTTPException(status_code=400, detail="camera 只能是 left、right 或 head")
    logger.info(
        "Pick Locate camera snapshot request camera=%s url=%s",
        normalized_camera,
        camera_url,
    )
    try:
        response = requests.get(
            camera_url,
            timeout=CAMERA_SNAPSHOT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        image_bytes = response.content
    except requests.RequestException as error:
        logger.warning(
            "Pick Locate camera snapshot failed camera=%s error=%s",
            normalized_camera,
            error,
        )
        return None

    if not image_bytes or len(image_bytes) > 20 * 1024 * 1024:
        logger.warning(
            "Pick Locate camera snapshot invalid size camera=%s bytes=%s",
            normalized_camera,
            len(image_bytes),
        )
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as source_image:
            image_format = (source_image.format or "").upper()
            source_image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        logger.warning(
            "Pick Locate camera snapshot invalid image camera=%s bytes=%s error=%s",
            normalized_camera,
            len(image_bytes),
            error,
        )
        return None

    suffix = {"JPEG": ".jpg", "PNG": ".png"}.get(image_format)
    if suffix is None:
        return None
    snapshot_path = CAMERA_SNAPSHOT_CACHE_DIR / (
        f"camera_rgb_{normalized_camera}_{uuid4().hex}{suffix}"
    )
    try:
        CAMERA_SNAPSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        write_bytes_atomically(snapshot_path, image_bytes)
    except OSError as error:
        logger.warning(
            "Pick Locate camera snapshot save failed camera=%s path=%s error=%s",
            normalized_camera,
            snapshot_path,
            error,
        )
        return None
    logger.info(
        "Pick Locate camera snapshot succeeded camera=%s bytes=%s format=%s path=%s",
        normalized_camera,
        len(image_bytes),
        image_format,
        snapshot_path,
    )
    return snapshot_path


def write_bytes_atomically(destination: Path, content: bytes) -> None:
    """Atomically write bytes without sharing a temporary path across callers."""

    temporary_path = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(destination)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def remove_camera_snapshot(snapshot_path: Path) -> None:
    """Best-effort cleanup for snapshots created in this process's cache."""

    try:
        cache_directory = CAMERA_SNAPSHOT_CACHE_DIR.resolve()
        resolved_path = snapshot_path.resolve()
        resolved_path.relative_to(cache_directory)
    except (OSError, ValueError):
        return
    if not resolved_path.name.startswith("camera_rgb_"):
        return
    try:
        resolved_path.unlink(missing_ok=True)
    except OSError:
        pass


def fetch_camera_depth(
    camera: str,
    expected_size: tuple[int, int],
) -> Image.Image | None:
    """读取相机服务返回的原始 16UC1 深度帧；不可用时静默回退。"""
    camera_url = CAMERA_DEPTH_SNAPSHOT_URLS.get(camera.strip().lower())
    if camera_url is None:
        raise HTTPException(status_code=400, detail="camera 只能是 left、right 或 head")
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
            write_bytes_atomically(stored_path, image_bytes)
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
def get_video_frame(hand: str = "left", task_type: str = "SORTING") -> FileResponse:
    image_path = get_latest_rgb(camera_for_task(task_type, hand))
    media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    return FileResponse(
        image_path,
        media_type=media_type,
        background=BackgroundTask(remove_camera_snapshot, image_path),
    )


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


def load_mock_sku_catalog() -> list[dict[str, Any]]:
    """Load the repository catalog used by offline/debug tooling."""

    try:
        payload = json.loads(MOCK_SKU_CATALOG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail="mock SKU 商品库不存在") from error
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"读取 mock SKU 商品库失败: {error}",
        ) from error
    products = payload.get("products") if isinstance(payload, dict) else None
    if not isinstance(products, list):
        raise HTTPException(status_code=500, detail="mock SKU 商品库缺少 products 数组")
    return [product for product in products if isinstance(product, dict)]


def lookup_mock_sku_by_name(
    name: str,
    mock_inventory: list[str],
) -> dict[str, Any]:
    """Build one SKU record with request-scoped inventory and no SKU API call."""

    product = next(
        (
            item
            for item in load_mock_sku_catalog()
            if isinstance(item.get("name"), str) and item["name"].strip() == name
        ),
        None,
    )
    if product is None:
        raise HTTPException(status_code=404, detail=f"mock SKU 中不存在商品: {name}")
    if not isinstance(product.get("sku_id"), str):
        raise HTTPException(status_code=500, detail="mock SKU 商品缺少 sku_id")

    raw_locations = product.get("locations")
    locations = {
        value.strip().upper()
        for value in raw_locations
        if isinstance(value, str) and LOCATION_PATTERN.fullmatch(value.strip().upper())
    } if isinstance(raw_locations, list) else set()
    normalized_inventory: list[str] = []
    for value in mock_inventory:
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail="mock_inventory 必须是槽位字符串数组")
        normalized = value.strip().upper()
        if LOCATION_PATTERN.fullmatch(normalized) is None:
            raise HTTPException(status_code=400, detail=f"mock 库存槽位格式无效: {normalized}")
        if normalized not in locations:
            raise HTTPException(
                status_code=400,
                detail=f"mock 库存槽位与商品不一致: {normalized}",
            )
        if normalized not in normalized_inventory:
            normalized_inventory.append(normalized)

    mocked_product = dict(product)
    mocked_product["inventory"] = normalized_inventory
    return mocked_product


def lookup_mock_sku_row(location_id: str) -> list[dict[str, Any]]:
    """Resolve a physical row from the local catalog for mock hard cases."""

    normalized_location = location_id.strip().upper()
    target_match = LOCATION_PATTERN.fullmatch(normalized_location)
    if target_match is None:
        raise HTTPException(status_code=400, detail="mock SKU 行 location_id 格式无效")
    row: list[dict[str, Any]] = []
    for product in load_mock_sku_catalog():
        raw_locations = product.get("locations")
        if not isinstance(raw_locations, list):
            continue
        for value in raw_locations:
            normalized = value.strip().upper() if isinstance(value, str) else ""
            match = LOCATION_PATTERN.fullmatch(normalized)
            if (
                match is None
                or match.group("shelf") != target_match.group("shelf")
                or match.group("level") != target_match.group("level")
            ):
                continue
            item = dict(product)
            item["location_id"] = normalized
            row.append(item)
    row.sort(
        key=lambda item: int(
            LOCATION_PATTERN.fullmatch(item["location_id"]).group("column")  # type: ignore[union-attr]
        )
    )
    if not row:
        raise HTTPException(status_code=404, detail=f"mock SKU 中不存在货架行: {normalized_location}")
    return row


def fetch_sku_reference_image(product: dict[str, Any]) -> QwenReferenceImage:
    """Fetch the first catalog image for a SHORTAGE target exactly once."""
    images = product.get("images")
    image_path = images[0] if isinstance(images, list) and images else None
    if not isinstance(image_path, str) or not image_path.strip():
        raise HTTPException(status_code=502, detail="SHORTAGE 商品缺少 SKU 样例图")

    logical_path = PurePosixPath(image_path.strip().replace("\\", "/"))
    if logical_path.is_absolute() or ".." in logical_path.parts:
        raise HTTPException(status_code=502, detail="SHORTAGE SKU 样例图路径不合法")
    try:
        response = requests.get(
            f"{SKU_API_URL}/{quote(logical_path.as_posix(), safe='/')}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"SHORTAGE SKU 样例图读取失败: {error}",
        ) from error
    if not response.content:
        raise HTTPException(status_code=502, detail="SHORTAGE SKU 样例图为空")
    media_type = response.headers.get("Content-Type", "image/jpeg").split(";", 1)[0]
    if not media_type.startswith("image/"):
        media_type = "image/jpeg"
    return QwenReferenceImage(
        logical_name=logical_path.as_posix(),
        media_type=media_type,
        content=response.content,
    )


def lookup_sku_row(location_id: str) -> list[dict[str, Any]]:
    """Return every physical column in a standard row, ordered left to right."""
    try:
        response = requests.request(
            "GET",
            f"{SKU_API_URL}/sku/get_row_layout",
            json={"location_id": location_id, "pose_type": ""},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        row = response.json()
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"SKU 行查询请求失败: {error}") from error
    except ValueError as error:
        raise HTTPException(status_code=502, detail="SKU 行查询响应不是有效 JSON") from error
    if (
        not isinstance(row, list)
        or not row
        or not all(
            isinstance(product, dict)
            and isinstance(product.get("location_id"), str)
            for product in row
        )
    ):
        raise HTTPException(status_code=502, detail="SKU 行查询响应格式错误")
    return row


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
        if (
            not re.fullmatch(r"L[1-5]", normalized_level)
            or normalized_hand not in {"left", "right"}
        ):
            raise HTTPException(status_code=500, detail="hard case 范围条目的 level 或 hand 无效")
        key = (name.strip(), normalized_level, normalized_hand)
        if key in scope:
            raise HTTPException(status_code=500, detail=f"hard case 范围存在重复条目: {key}")
        scope.add(key)
    return scope


def load_hard_case_view_layout(
) -> tuple[dict[str, str], dict[tuple[str, str, str, str], HardCaseViewLayout]]:
    """Load exact hard-case slots and their wrist-camera left-to-right order."""
    try:
        payload = json.loads(HARD_CASE_VIEW_LAYOUT_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail="hard case 视角配置不存在") from error
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"读取 hard case 视角配置失败: {error}",
        ) from error
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="hard case 视角配置必须是对象")

    raw_slot_groups = payload.get("slot_groups")
    raw_views = payload.get("views")
    if not isinstance(raw_slot_groups, dict) or not isinstance(raw_views, list):
        raise HTTPException(
            status_code=500,
            detail="hard case 视角配置缺少 slot_groups 或 views",
        )

    slot_to_group: dict[str, str] = {}
    for group_id, raw_slots in raw_slot_groups.items():
        if group_id not in {"alien_energy", "maiydong"}:
            raise HTTPException(
                status_code=500,
                detail=f"hard case 视角配置包含不支持的槽位组: {group_id}",
            )
        if not (
            isinstance(raw_slots, list)
            and raw_slots
            and all(isinstance(value, str) and value.strip() for value in raw_slots)
        ):
            raise HTTPException(
                status_code=500,
                detail=f"hard case 槽位组 {group_id} 必须是非空数组",
            )
        for raw_slot in raw_slots:
            slot_id = raw_slot.strip().upper()
            if LOCATION_PATTERN.fullmatch(slot_id) is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"hard case 槽位格式无效: {slot_id}",
                )
            if slot_id in slot_to_group:
                raise HTTPException(
                    status_code=500,
                    detail=f"hard case 槽位重复配置: {slot_id}",
                )
            slot_to_group[slot_id] = group_id

    views: dict[tuple[str, str, str, str], HardCaseViewLayout] = {}
    covered_slots: set[str] = set()
    for raw_view in raw_views:
        if not isinstance(raw_view, dict):
            raise HTTPException(status_code=500, detail="hard case 视角条目格式错误")
        target_id = raw_view.get("target_id")
        hand = raw_view.get("hand")
        level = raw_view.get("level")
        group_id = raw_view.get("group_id")
        visible_slots = raw_view.get("visible_slot_order")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (target_id, hand, level, group_id)
        ) or not (
            isinstance(visible_slots, list)
            and visible_slots
            and all(isinstance(value, str) and value.strip() for value in visible_slots)
        ):
            raise HTTPException(status_code=500, detail="hard case 视角条目缺少字段")

        normalized_target = target_id.strip().upper()
        normalized_hand = hand.strip().lower()
        normalized_level = level.strip().upper()
        normalized_group = group_id.strip()
        normalized_slots = tuple(value.strip().upper() for value in visible_slots)
        if (
            INSPECTION_TARGET_PATTERN.fullmatch(normalized_target) is None
            or normalized_hand not in {"left", "right"}
            or re.fullmatch(r"L[1-5]", normalized_level) is None
            or normalized_group not in raw_slot_groups
        ):
            raise HTTPException(status_code=500, detail="hard case 视角键无效")
        if len(set(normalized_slots)) != len(normalized_slots):
            raise HTTPException(
                status_code=500,
                detail=f"hard case 视角包含重复槽位: {normalized_target}/{normalized_level}",
            )
        for slot_id in normalized_slots:
            match = LOCATION_PATTERN.fullmatch(slot_id)
            if (
                match is None
                or f"L{int(match.group('level'))}" != normalized_level
                or slot_to_group.get(slot_id) != normalized_group
            ):
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "hard case 视角槽位与 group/level 不一致: "
                        f"{normalized_target}/{normalized_hand}/{normalized_level}/{slot_id}"
                    ),
                )
        key = (
            normalized_target,
            normalized_hand,
            normalized_level,
            normalized_group,
        )
        if key in views:
            raise HTTPException(status_code=500, detail=f"hard case 视角重复配置: {key}")
        views[key] = HardCaseViewLayout(
            target_id=normalized_target,
            hand=normalized_hand,
            level=normalized_level,
            group_id=normalized_group,
            visible_slot_order=normalized_slots,
        )
        covered_slots.update(normalized_slots)

    missing_slots = sorted(set(slot_to_group) - covered_slots)
    if missing_slots:
        raise HTTPException(
            status_code=500,
            detail=f"hard case 槽位没有任何可用视角: {missing_slots}",
        )
    return slot_to_group, views


def validate_slot_hard_case_context(
    product: dict[str, Any],
    task_type: str,
    level: str | None,
    hand: str,
    slot_id: str | None,
    target_id: str | None,
) -> tuple[str, HardCaseViewLayout] | None:
    """Validate exact slot/view context for the connector hard-case groups."""
    if task_type.strip().upper() != "SORTING":
        return None
    product_name = str(product.get("name", "")).strip()
    connector_group_ids = {"alien_energy", "maiydong"}
    configured_product_groups = {
        group_id
        for group_id in connector_group_ids
        if product_name in HARD_CASE_GROUPS[group_id].members
    }
    # Ordinary products, catnip Maiydong and the legacy Shuke hard case do not
    # depend on the connector-view file.
    if not configured_product_groups:
        return None

    normalized_hand = hand.strip().lower()
    if normalized_hand not in {"left", "right"}:
        raise HTTPException(status_code=400, detail="hand 只能是 left 或 right")
    if not level:
        raise HTTPException(
            status_code=400,
            detail="电解质和非猫薄荷脉动 hard case 必须提供 level",
        )
    normalized_level = normalize_level(level)
    scope_key = (product_name, normalized_level, normalized_hand)
    if scope_key not in load_hard_case_scope():
        return None

    slot_to_group, views = load_hard_case_view_layout()
    raw_locations = product.get("locations")
    product_locations = {
        value.strip().upper()
        for value in raw_locations
        if isinstance(value, str) and LOCATION_PATTERN.fullmatch(value.strip().upper())
    } if isinstance(raw_locations, list) else set()
    product_groups = {slot_to_group[value] for value in product_locations if value in slot_to_group}

    normalized_slot = (slot_id or "").strip().upper()
    requested_group = slot_to_group.get(normalized_slot)
    if not product_groups and requested_group is None:
        return None
    if not product_groups:
        raise HTTPException(
            status_code=400,
            detail=f"slot_id 与商品不一致: {normalized_slot}",
        )
    if len(product_groups) != 1:
        raise HTTPException(
            status_code=500,
            detail="SKU hard case 槽位组配置不唯一",
        )
    group_id = next(iter(product_groups))
    if group_id not in configured_product_groups:
        raise HTTPException(status_code=400, detail="商品不属于该 hard case 商品组")
    if not normalized_slot or not (target_id or "").strip():
        raise HTTPException(
            status_code=400,
            detail="电解质和非猫薄荷脉动 hard case 必须提供精确 slot_id 和 target_id",
        )
    slot_match = LOCATION_PATTERN.fullmatch(normalized_slot)
    if slot_match is None:
        raise HTTPException(status_code=400, detail="slot_id 格式无效")
    if normalized_slot not in product_locations:
        raise HTTPException(
            status_code=400,
            detail=f"slot_id 与商品不一致: {normalized_slot}",
        )

    if requested_group != group_id:
        raise HTTPException(status_code=400, detail="slot_id 与 hard case 商品组不一致")
    config = HARD_CASE_GROUPS[group_id]
    if product_name not in config.members:
        raise HTTPException(status_code=400, detail="商品不属于该 hard case 商品组")

    slot_level = f"L{int(slot_match.group('level'))}"
    if normalized_level != slot_level:
        raise HTTPException(
            status_code=400,
            detail=f"level 与 slot_id 不一致: level={normalized_level}, slot={normalized_slot}",
        )
    normalized_target = (target_id or "").strip().upper()
    if INSPECTION_TARGET_PATTERN.fullmatch(normalized_target) is None:
        raise HTTPException(status_code=400, detail="target_id 格式无效")
    view = views.get((normalized_target, normalized_hand, normalized_level, group_id))
    if view is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "当前 target_id、hand、level 没有 hard case 腕部视角配置: "
                f"{normalized_target}/{normalized_hand}/{normalized_level}/{group_id}"
            ),
        )
    if normalized_slot not in view.visible_slot_order:
        raise HTTPException(
            status_code=400,
            detail=f"目标槽位不在当前腕部视角中: {normalized_slot}",
        )
    return group_id, view


def load_hard_case_layout_overrides(
) -> dict[tuple[str, str, str], tuple[str, ...]]:
    """Load tuple-scoped hard-case orders, preserving intentional duplicates."""
    try:
        payload = json.loads(
            HARD_CASE_LAYOUT_OVERRIDES_PATH.read_text(encoding="utf-8")
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail="hard case 布局覆盖配置不存在",
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"读取 hard case 布局覆盖配置失败: {error}",
        ) from error
    if not isinstance(payload, list):
        raise HTTPException(
            status_code=500,
            detail="hard case 布局覆盖配置必须是数组",
        )

    overrides: dict[tuple[str, str, str], tuple[str, ...]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=500,
                detail="hard case 布局覆盖条目格式错误",
            )
        name = item.get("product_name")
        level = item.get("level")
        hand = item.get("hand")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (name, level, hand)
        ):
            raise HTTPException(
                status_code=500,
                detail="hard case 布局覆盖条目缺少字段",
            )

        normalized_level = level.strip().upper()
        normalized_hand = hand.strip().lower()
        if (
            not re.fullmatch(r"L[1-5]", normalized_level)
            or normalized_hand not in {"left", "right"}
        ):
            raise HTTPException(
                status_code=500,
                detail="hard case 布局覆盖条目的 level 或 hand 无效",
            )

        visible_order = item.get(f"visible_order_from_{normalized_hand}")
        if not (
            isinstance(visible_order, list)
            and visible_order
            and all(
                isinstance(value, str) and value.strip()
                for value in visible_order
            )
        ):
            raise HTTPException(
                status_code=500,
                detail=(
                    "hard case 布局覆盖条目缺少与 hand 对应的 "
                    f"visible_order_from_{normalized_hand}"
                ),
            )
        key = (
            name.strip(),
            normalized_level,
            normalized_hand,
        )
        if key in overrides:
            raise HTTPException(
                status_code=500,
                detail=f"hard case 布局覆盖存在重复条目: {key}",
            )
        # Do not de-duplicate this sequence: repeated physical columns are the
        # reason this tuple-scoped override exists.
        overrides[key] = tuple(value.strip() for value in visible_order)
    return overrides


def hard_case_layout_order_for_request(
    product_name: str,
    level: str | None,
    hand: str,
) -> tuple[str, ...] | None:
    if not level:
        return None
    key = (
        product_name.strip(),
        level.strip().upper(),
        hand.strip().lower(),
    )
    return load_hard_case_layout_overrides().get(key)


def hard_case_group_for_product(
    product_name: str,
    task_type: str,
    level: str | None,
    hand: str,
    slot_id: str | None = None,
    target_id: str | None = None,
) -> tuple[str, HardCaseGroupConfig] | None:
    del target_id  # View validity is checked once the full SKU record is available.
    if task_type.strip().upper() != "SORTING" or not level:
        return None
    normalized_name = product_name.strip()
    scope_key = (
        normalized_name,
        level.strip().upper(),
        hand.strip().lower(),
    )
    if scope_key not in load_hard_case_scope():
        return None
    normalized_slot = (slot_id or "").strip().upper()
    if normalized_slot:
        slot_to_group, _ = load_hard_case_view_layout()
        group_id = slot_to_group.get(normalized_slot)
        if group_id is not None:
            config = HARD_CASE_GROUPS[group_id]
            return (group_id, config) if normalized_name in config.members else None
    for group_id, config in HARD_CASE_GROUPS.items():
        if normalized_name in config.members:
            return group_id, config
    return None


def hard_case_level_required(
    product_name: str,
    task_type: str,
    hand: str,
) -> bool:
    """Whether this SORTING product/hand has any level-scoped hard case."""
    if task_type.strip().upper() != "SORTING":
        return False
    normalized_name = product_name.strip()
    normalized_hand = hand.strip().lower()
    return any(
        scoped_name == normalized_name and scoped_hand == normalized_hand
        for scoped_name, _scoped_level, scoped_hand in load_hard_case_scope()
    )


def load_multi_row_products(
    path: str | Path | None = None,
) -> frozenset[str]:
    """Load product names whose SORTING candidates may span several display rows.

    The special ``*`` entry enables the behavior for every SORTING product, so
    newly added SKUs do not silently miss repeated-pick/history handling.
    """

    config_path = Path(path) if path is not None else MULTI_ROW_PRODUCTS_PATH
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail="多排商品配置不存在") from error
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"读取多排商品配置失败: {error}",
        ) from error
    products = payload.get("products") if isinstance(payload, dict) else None
    if not (
        isinstance(products, list)
        and all(isinstance(name, str) and name.strip() for name in products)
    ):
        raise HTTPException(
            status_code=500,
            detail="多排商品配置 products 必须是非空字符串数组",
        )
    return frozenset(name.strip() for name in products)


def uses_multi_row_pick(product_name: str, task_type: str) -> bool:
    if task_type.strip().upper() != "SORTING":
        return False
    configured_products = load_multi_row_products()
    return "*" in configured_products or product_name.strip() in configured_products


def valid_product_locations(product: dict[str, Any]) -> frozenset[str]:
    raw_locations = product.get("locations")
    if not isinstance(raw_locations, list):
        return frozenset()
    return frozenset(
        value.strip().upper()
        for value in raw_locations
        if isinstance(value, str)
        and LOCATION_PATTERN.fullmatch(value.strip().upper())
    )


def supports_inventory_row_mapping(product: dict[str, Any]) -> bool:
    """Only products with several catalog slots need row/column inference."""

    return len(valid_product_locations(product)) > 1


def inventory_slots_for_pick(
    product: dict[str, Any],
    task_type: str,
    slot_id: str | None,
    level: str | None,
) -> list[str]:
    """Return current same-shelf-row inventory ordered by physical column."""

    if task_type.strip().upper() != "SORTING" or not (slot_id or "").strip():
        return []
    normalized_slot = (slot_id or "").strip().upper()
    target_match = LOCATION_PATTERN.fullmatch(normalized_slot)
    if target_match is None:
        raise HTTPException(status_code=400, detail="slot_id 格式无效")
    if level is not None and normalize_level(level) != f"L{int(target_match.group('level'))}":
        raise HTTPException(
            status_code=400,
            detail=f"level 与 slot_id 不一致: level={level}, slot={normalized_slot}",
        )

    locations = valid_product_locations(product)
    if normalized_slot not in locations:
        raise HTTPException(status_code=400, detail="slot_id 与商品不一致")

    raw_inventory = product.get("inventory")
    if not isinstance(raw_inventory, list):
        raw_inventory = list(locations)
    inventory = {
        value.strip().upper()
        for value in raw_inventory
        if isinstance(value, str) and LOCATION_PATTERN.fullmatch(value.strip().upper())
    }
    if normalized_slot not in inventory:
        raise HTTPException(status_code=409, detail=f"slot_id 已不在库存中: {normalized_slot}")

    shelf = target_match.group("shelf")
    row_level = target_match.group("level")
    row_slots = [
        value
        for value in inventory
        if (
            (match := LOCATION_PATTERN.fullmatch(value)) is not None
            and match.group("shelf") == shelf
            and match.group("level") == row_level
        )
    ]
    row_slots.sort(
        key=lambda value: int(LOCATION_PATTERN.fullmatch(value).group("column"))  # type: ignore[union-attr]
    )
    return row_slots


def select_location_for_level(
    product: dict[str, Any], level: str, hand: str = "left"
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
        if match is not None and f"L{int(match.group('level'))}" == normalized_level:
            parsed.append((value.strip().upper(), match))
    if not parsed:
        raise HTTPException(status_code=404, detail=f"hard case SKU 在 {normalized_level} 没有 location")
    parsed.sort(key=lambda item: int(item[1].group("column")))
    return parsed[-1] if hand.strip().lower() == "right" else parsed[0]


def hard_case_standard_order(
    target_location: str,
    config: HardCaseGroupConfig,
    sku_row_lookup: Callable[[str], list[dict[str, Any]]] | None = None,
) -> list[str]:
    row = (sku_row_lookup or lookup_sku_row)(target_location)
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


def camera_for_task(task_type: str, hand: str) -> str:
    """Route live locate input to the task's physical camera."""
    normalized_task_type = normalize_task_type(task_type)
    if normalized_task_type in {"SHORTAGE", "MISPLACED"}:
        return "head"
    normalized_hand = hand.strip().lower()
    if normalized_hand not in {"left", "right"}:
        raise HTTPException(status_code=400, detail="hand 只能是 left 或 right")
    return normalized_hand


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


def prepare_rgb_inference_image(
    original_image: Image.Image,
) -> tuple[Image.Image, bytes]:
    """Use the same RGB inference canvas size as qwen-debug."""
    resized_image = original_image.resize(
        RGB_INFERENCE_SIZE,
        resample=Image.Resampling.BILINEAR,
    )
    buffer = io.BytesIO()
    resized_image.save(
        buffer,
        format="JPEG",
        quality=RGB_INFERENCE_JPEG_QUALITY,
    )
    inference_bytes = buffer.getvalue()
    # qwen-debug crops from the decoded JPEG canvas, not from its pre-encode
    # canvas pixels. Decode once here so Qwen and SAM3 observe the same RGB.
    with Image.open(io.BytesIO(inference_bytes)) as encoded_image:
        inference_image = encoded_image.convert("RGB")
    return inference_image, inference_bytes


def qwen_image_content(image_bytes: bytes, media_type: str) -> dict[str, Any]:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
    }


def call_qwen3(
    prompt: str,
    image_source: Path | bytes,
    *,
    reference_image: QwenReferenceImage | None = None,
) -> str:
    if isinstance(image_source, Path):
        media_type = mimetypes.guess_type(image_source.name)[0] or "image/jpeg"
        image_bytes = image_source.read_bytes()
    else:
        media_type = "image/jpeg"
        image_bytes = image_source
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if reference_image is not None:
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "下面第一张图是目标商品的标准 SKU 样例图，仅用于识别商品外观；"
                        "不要输出这张样例图中的 bbox。"
                    ),
                },
                qwen_image_content(reference_image.content, reference_image.media_type),
                {
                    "type": "text",
                    "text": (
                        "下面第二张图是待定位货架图。只输出第二张图中目标商品的 bbox，"
                        "bbox 坐标必须以第二张图为准。"
                    ),
                },
            ]
        )
    content.append(qwen_image_content(image_bytes, media_type))
    logger.info(
        "Pick Locate Qwen3 request model=%s prompt_chars=%s image_bytes=%s "
        "reference_image=%s",
        QWEN3_MODEL,
        len(prompt),
        len(image_bytes),
        reference_image.logical_name if reference_image is not None else None,
    )
    response = requests.post(
        QWEN3_URL,
        json={
            "model": QWEN3_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": content,
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
    logger.info(
        "Pick Locate Qwen3 response received content_chars=%s",
        len(content),
    )
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


def get_stable_qwen_bboxes(
    prompt: str,
    image_source: Path | bytes,
    *,
    reference_image: QwenReferenceImage | None = None,
) -> list[list[float]]:
    samples: list[tuple[int, list[dict[str, Any]]]] = []
    errors: list[str] = []
    for sample_index in range(1, QWEN_SAMPLE_COUNT + 1):
        try:
            content = (
                call_qwen3(
                    prompt,
                    image_source,
                    reference_image=reference_image,
                )
                if reference_image is not None
                else call_qwen3(prompt, image_source)
            )
            detections = parse_qwen_detections(content)
            samples.append((sample_index, detections))
            logger.info(
                "Pick Locate Qwen3 sample succeeded sample=%s detection_count=%s",
                sample_index,
                len(detections),
            )
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
            errors.append(f"第 {sample_index} 次: {error}")
            logger.warning(
                "Pick Locate Qwen3 sample failed sample=%s error=%s",
                sample_index,
                error,
            )

    if len(samples) < 2:
        detail = "; ".join(errors) or "成功采样不足两次"
        raise HTTPException(status_code=502, detail=f"Qwen3 无法形成跨采样共识: {detail}")

    bboxes = consensus_qwen_bboxes(samples)
    if not bboxes:
        raise HTTPException(
            status_code=404,
            detail=f"Qwen3 没有产生跨采样 IoU > {QWEN_CONSENSUS_IOU} 的稳定 bbox",
        )
    logger.info(
        "Pick Locate Qwen3 consensus completed successful_samples=%s "
        "failed_samples=%s bbox_count=%s",
        len(samples),
        len(errors),
        len(bboxes),
    )
    return QwenConsensusBBoxes(bboxes, samples)


def qwen_bbox_to_crop(
    bbox: list[float],
    image_size: tuple[int, int],
    *,
    padding_ratio: float = CROP_PADDING_RATIO,
) -> tuple[int, int, int, int]:
    """将 [0,1000] Qwen 坐标转为像素，并按指定比例向四周外扩。"""
    image_width, image_height = image_size
    x1, y1, x2, y2 = qwen_bbox_to_original(bbox, image_size)
    padding_x = (x2 - x1) * padding_ratio
    padding_y = (y2 - y1) * padding_ratio
    crop_box = (
        max(0, math.floor(x1 - padding_x)),
        max(0, math.floor(y1 - padding_y)),
        min(image_width, math.ceil(x2 + padding_x)),
        min(image_height, math.ceil(y2 + padding_y)),
    )
    if crop_box[2] - crop_box[0] < 2 or crop_box[3] - crop_box[1] < 2:
        raise ValueError("Qwen3 bbox 无法生成有效 crop")
    return crop_box


def qwen_crop_padding_ratio(product_name: str) -> float:
    """返回商品对应的 Qwen crop 外扩比例，未配置商品保持默认 10%。"""
    return QWEN_CROP_PADDING_RATIO_OVERRIDES.get(
        product_name.strip(),
        CROP_PADDING_RATIO,
    )


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


def map_bbox_between_sizes(
    bbox: list[float],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[float]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return [
        max(0.0, min(float(target_width), float(bbox[0]) * scale_x)),
        max(0.0, min(float(target_height), float(bbox[1]) * scale_y)),
        max(0.0, min(float(target_width), float(bbox[2]) * scale_x)),
        max(0.0, min(float(target_height), float(bbox[3]) * scale_y)),
    ]


def map_crop_box_between_sizes(
    crop_box: tuple[int, int, int, int],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[int]:
    mapped = map_bbox_between_sizes(list(crop_box), source_size, target_size)
    return [
        math.floor(mapped[0]),
        math.floor(mapped[1]),
        math.ceil(mapped[2]),
        math.ceil(mapped[3]),
    ]


def call_sam3(prompt: str, crop_image: Image.Image) -> list[dict[str, Any]]:
    buffer = io.BytesIO()
    crop_image.save(buffer, format="JPEG", quality=95)
    image_bytes = buffer.getvalue()
    logger.info(
        "Pick Locate SAM3 request crop_size=%sx%s image_bytes=%s prompt_chars=%s",
        crop_image.width,
        crop_image.height,
        len(image_bytes),
        len(prompt),
    )
    try:
        response = requests.post(
            SAM3_URL,
            files={"image": ("qwen_crop.jpg", image_bytes, "image/jpeg")},
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
    logger.info(
        "Pick Locate SAM3 response received instance_count=%s",
        len(instances),
    )
    return instances


def map_mask_to_original(
    mask_base64: str,
    crop_box: tuple[int, int, int, int],
    inference_size: tuple[int, int],
    original_size: tuple[int, int] | None = None,
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

    inference_mask = Image.new("L", inference_size, 0)
    inference_mask.paste(crop_mask, (crop_box[0], crop_box[1]))
    output_size = original_size or inference_size
    original_mask = (
        inference_mask.resize(output_size, resample=Image.Resampling.NEAREST)
        if output_size != inference_size
        else inference_mask
    )
    output = io.BytesIO()
    original_mask.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def map_sam_instance_to_original(
    instance: dict[str, Any],
    crop_box: tuple[int, int, int, int],
    inference_size: tuple[int, int],
    original_size: tuple[int, int] | None = None,
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
    output_size = original_size or inference_size
    inference_bbox = [
        float(bbox[0]) + crop_x1,
        float(bbox[1]) + crop_y1,
        float(bbox[2]) + crop_x1,
        float(bbox[3]) + crop_y1,
    ]
    original_bbox = map_bbox_between_sizes(
        inference_bbox,
        inference_size,
        output_size,
    )
    score_value = instance.get("score")
    score = float(score_value) if isinstance(score_value, (int, float)) else None
    return LocatedInstance(
        bbox=original_bbox,
        mask=map_mask_to_original(
            mask,
            crop_box,
            inference_size,
            output_size,
        ),
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


def bbox_coverage_ratio(
    candidate_bbox: list[float], reference_bbox: list[float]
) -> float:
    """Return the fraction of the reference bbox covered by the candidate."""
    reference_width = max(0.0, reference_bbox[2] - reference_bbox[0])
    reference_height = max(0.0, reference_bbox[3] - reference_bbox[1])
    reference_area = reference_width * reference_height
    if reference_area <= 0:
        return 0.0
    intersection_width = max(
        0.0,
        min(candidate_bbox[2], reference_bbox[2])
        - max(candidate_bbox[0], reference_bbox[0]),
    )
    intersection_height = max(
        0.0,
        min(candidate_bbox[3], reference_bbox[3])
        - max(candidate_bbox[1], reference_bbox[1]),
    )
    return intersection_width * intersection_height / reference_area


def keep_sam_instances_with_qwen_coverage(
    instances: list[LocatedInstance],
    qwen_bboxes_by_source: dict[int, list[float]],
    *,
    minimum_coverage: float = PICK_MIN_SAM_QWEN_BBOX_COVERAGE,
    minimum_relative_area: float = PICK_MIN_SAM_TO_LARGEST_BBOX_AREA_RATIO,
) -> list[LocatedInstance]:
    """Drop SAM bboxes tiny relative to both Qwen and the largest peer bbox."""
    threshold = max(0.0, min(1.0, minimum_coverage))
    relative_area_threshold = max(0.0, min(1.0, minimum_relative_area))
    indices_by_source: dict[int, list[int]] = defaultdict(list)
    for index, instance in enumerate(instances):
        if instance.source_qwen_index in qwen_bboxes_by_source:
            indices_by_source[instance.source_qwen_index].append(index)
    largest_index_by_source = {
        source: max(
            source_indices,
            key=lambda index: (
                max(0.0, instances[index].bbox[2] - instances[index].bbox[0])
                * max(0.0, instances[index].bbox[3] - instances[index].bbox[1]),
                instances[index].score
                if instances[index].score is not None
                else -1.0,
            ),
        )
        for source, source_indices in indices_by_source.items()
    }
    protected_indices = set(largest_index_by_source.values())
    largest_area_by_source = {
        source: max(
            0.0,
            instances[index].bbox[2] - instances[index].bbox[0],
        )
        * max(
            0.0,
            instances[index].bbox[3] - instances[index].bbox[1],
        )
        for source, index in largest_index_by_source.items()
    }
    return [
        instance
        for index, instance in enumerate(instances)
        if index in protected_indices
        or instance.source_qwen_index not in qwen_bboxes_by_source
        or bbox_coverage_ratio(
            instance.bbox,
            qwen_bboxes_by_source[instance.source_qwen_index],
        )
        >= threshold
        or (
            largest_area_by_source[instance.source_qwen_index] > 0
            and max(0.0, instance.bbox[2] - instance.bbox[0])
            * max(0.0, instance.bbox[3] - instance.bbox[1])
            / largest_area_by_source[instance.source_qwen_index]
            >= relative_area_threshold
        )
    ]


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


def uses_upper_confidence_pick(product_name: str, task_type: str) -> bool:
    """Whether SORTING should pick an upper stacked item without depth."""
    return (
        task_type.strip().upper() == "SORTING"
        and product_name.strip() in UPPER_CONFIDENCE_PICK_PRODUCTS
    )


def keep_instances_from_nearest_qwen_shelf_row(
    instances: list[LocatedInstance],
    qwen_bboxes_by_source: dict[int, list[float]],
    image_height: int,
    *,
    row_gap_tolerance_ratio: float = 0.05,
) -> list[LocatedInstance]:
    """Keep the Qwen shelf row nearest the image's vertical center.

    Upper-confidence products may appear on two visible shelf levels. Their
    final "upper item" rule must run inside the intended shelf row instead of
    treating the row above as another item in the same vertical stack.
    """
    if len(instances) < 2 or image_height <= 0:
        return instances

    instance_sources = {
        instance.source_qwen_index
        for instance in instances
        if instance.source_qwen_index in qwen_bboxes_by_source
    }
    source_intervals = sorted(
        (
            source,
            max(0.0, min(float(image_height), float(qwen_bboxes_by_source[source][1]))),
            max(0.0, min(float(image_height), float(qwen_bboxes_by_source[source][3]))),
        )
        for source in instance_sources
    )
    if len(source_intervals) < 2:
        return instances

    tolerance = image_height * max(0.0, row_gap_tolerance_ratio)
    rows: list[dict[str, Any]] = []
    for source, first_y, second_y in sorted(
        source_intervals,
        key=lambda item: (min(item[1], item[2]), max(item[1], item[2])),
    ):
        top = min(first_y, second_y)
        bottom = max(first_y, second_y)
        if rows and top <= float(rows[-1]["bottom"]) + tolerance:
            rows[-1]["bottom"] = max(float(rows[-1]["bottom"]), bottom)
            rows[-1]["sources"].add(source)
        else:
            rows.append({"top": top, "bottom": bottom, "sources": {source}})

    if len(rows) < 2:
        return instances

    image_center_y = image_height / 2.0

    def row_distance(row: dict[str, Any]) -> tuple[float, float]:
        top = float(row["top"])
        bottom = float(row["bottom"])
        if top <= image_center_y <= bottom:
            edge_distance = 0.0
        else:
            edge_distance = min(
                abs(image_center_y - top),
                abs(image_center_y - bottom),
            )
        return edge_distance, abs(image_center_y - (top + bottom) / 2.0)

    selected_sources = min(rows, key=row_distance)["sources"]
    filtered = [
        instance
        for instance in instances
        if instance.source_qwen_index in selected_sources
    ]
    return filtered or instances


def select_upper_high_confidence_instance(
    instances: list[LocatedInstance],
    *,
    score_margin: float = PICK_UPPER_CONFIDENCE_SCORE_MARGIN,
    vertical_tie_tolerance_ratio: float = PICK_UPPER_VERTICAL_TIE_TOLERANCE_RATIO,
    minimum_mask_bbox_fill_ratio: float = PICK_MIN_MASK_BBOX_FILL_RATIO,
    minimum_overlap_mask_area_ratio: float = PICK_MIN_OVERLAP_MASK_AREA_RATIO,
) -> LocatedInstance:
    """Pick an upper complete mask among candidates close to the best score."""
    if not instances:
        raise HTTPException(status_code=404, detail="SAM3 没有找到目标商品实例")
    if len(instances) == 1:
        return instances[0]

    # Remove visibly sparse/small fragments before their score or vertical
    # position can define the candidate band. The largest mask and largest bbox
    # in every overlap component remain protected by the helper.
    quality_candidates = keep_mask_area_quality_candidates(
        instances,
        minimum_fill_ratio=minimum_mask_bbox_fill_ratio,
        minimum_relative_mask_area=minimum_overlap_mask_area_ratio,
    )
    scored = [
        instance
        for instance in quality_candidates
        if instance.score is not None and math.isfinite(instance.score)
    ]
    if scored:
        best_score = max(instance.score for instance in scored)
        minimum_score = best_score - max(0.0, score_margin)
        candidates = [
            instance
            for instance in scored
            if instance.score >= minimum_score
        ]
    else:
        candidates = quality_candidates

    top_center_y = min(
        (instance.bbox[1] + instance.bbox[3]) / 2
        for instance in candidates
    )
    tallest_height = max(
        max(0.0, instance.bbox[3] - instance.bbox[1])
        for instance in candidates
    )
    vertical_tolerance = tallest_height * max(
        0.0,
        vertical_tie_tolerance_ratio,
    )
    upper_band = [
        instance
        for instance in candidates
        if (instance.bbox[1] + instance.bbox[3]) / 2
        <= top_center_y + vertical_tolerance
    ]
    return max(
        upper_band,
        key=lambda instance: (
            mask_foreground_pixel_count(instance.mask),
            max(0.0, instance.bbox[2] - instance.bbox[0])
            * max(0.0, instance.bbox[3] - instance.bbox[1]),
            instance.score if instance.score is not None else -1.0,
            -((instance.bbox[1] + instance.bbox[3]) / 2),
        ),
    )


def uses_max_mask_area_pick(product_name: str, task_type: str) -> bool:
    return (
        task_type.strip().upper() == "SORTING"
        and product_name.strip() in MAX_MASK_AREA_PICK_PRODUCTS
    )


def select_largest_mask_area_instance(
    instances: list[LocatedInstance],
) -> LocatedInstance:
    """Select the instance with the largest actual SAM foreground mask."""
    if not instances:
        raise HTTPException(status_code=404, detail="SAM3 没有找到目标商品实例")
    return max(
        instances,
        key=lambda instance: (
            mask_foreground_pixel_count(instance.mask),
            instance.score if instance.score is not None else -1.0,
            (instance.bbox[2] - instance.bbox[0])
            * (instance.bbox[3] - instance.bbox[1]),
        ),
    )


def bbox_overlap_components(
    instances: list[LocatedInstance],
) -> list[list[int]]:
    """Return bbox-overlap connected components without reading SAM masks."""
    if not instances:
        return []

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
    return list(components.values())


def keep_mask_area_quality_candidates(
    instances: list[LocatedInstance],
    *,
    minimum_fill_ratio: float = PICK_MIN_MASK_BBOX_FILL_RATIO,
    minimum_relative_mask_area: float = PICK_MIN_OVERLAP_MASK_AREA_RATIO,
) -> list[LocatedInstance]:
    """Drop sparse/small masks inside overlap groups while protecting maxima."""
    if len(instances) < 2:
        return instances

    fill_threshold = max(0.0, min(1.0, minimum_fill_ratio))
    relative_area_threshold = max(
        0.0,
        min(1.0, minimum_relative_mask_area),
    )
    mask_areas = [
        mask_foreground_pixel_count(instance.mask) for instance in instances
    ]
    bbox_areas = [
        max(0.0, instance.bbox[2] - instance.bbox[0])
        * max(0.0, instance.bbox[3] - instance.bbox[1])
        for instance in instances
    ]
    kept_indices: set[int] = set()
    for component in bbox_overlap_components(instances):
        if len(component) == 1:
            kept_indices.update(component)
            continue

        largest_mask_area = max(mask_areas[index] for index in component)
        largest_bbox_area = max(bbox_areas[index] for index in component)
        protected_indices = {
            index
            for index in component
            if mask_areas[index] == largest_mask_area
            or bbox_areas[index] == largest_bbox_area
        }

        for index in component:
            if index in protected_indices:
                kept_indices.add(index)
                continue
            bbox_area = bbox_areas[index]
            fill_ratio = mask_areas[index] / bbox_area if bbox_area > 0 else 0.0
            relative_mask_area = (
                mask_areas[index] / largest_mask_area
                if largest_mask_area > 0
                else 0.0
            )
            if (
                fill_ratio >= fill_threshold
                and relative_mask_area >= relative_area_threshold
            ):
                kept_indices.add(index)

    filtered = [
        instance
        for index, instance in enumerate(instances)
        if index in kept_indices
    ]
    return filtered or instances


def keep_frontmost_in_overlap_chains(
    instances: list[LocatedInstance],
) -> list[LocatedInstance]:
    """每个 bbox 重叠连通分量只保留一个最前方实例。"""
    if len(instances) < 2:
        return instances

    selected: list[tuple[int, LocatedInstance]] = []
    for indices in bbox_overlap_components(instances):
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
    # otherwise independent front products into one overlap chain. Filter the
    # geometric front row first, then remove duplicate SAM masks only within it.
    ordered = sorted(
        keep_frontmost_in_overlap_chains(front_row),
        key=instance_center_x,
    )
    return [[instance] for instance in ordered]


def drop_excess_truncated_edge_groups(
    display_groups: list[list[LocatedInstance]],
    expected_count: int,
    image_width: int | None,
) -> list[list[LocatedInstance]]:
    """Drop unusually narrow image-edge columns only to resolve a slot overflow."""
    if (
        expected_count < 1
        or len(display_groups) <= expected_count
        or image_width is None
        or image_width <= 0
    ):
        return display_groups

    bboxes = [union_instance_bboxes(group) for group in display_groups]
    if any(right <= left or bottom <= top for left, top, right, bottom in bboxes):
        return display_groups
    aspect_ratios = [
        (right - left) / (bottom - top) for left, top, right, bottom in bboxes
    ]
    # Use the original image boundary, never the Qwen crop boundary. A few
    # pixels of SAM boundary error are tolerated, scaled with image resolution.
    edge_margin = max(2.0, image_width * 0.005)
    reference_ratios = [
        ratio
        for bbox, ratio in zip(bboxes, aspect_ratios)
        if bbox[0] > edge_margin and bbox[2] < image_width - edge_margin
    ]
    if not reference_ratios:
        return display_groups
    reference_ratio = float(np.median(reference_ratios))
    candidates: list[tuple[float, int, str]] = []
    for index, side in ((0, "left"), (len(display_groups) - 1, "right")):
        touches_edge = (
            bboxes[index][0] <= edge_margin
            if side == "left"
            else bboxes[index][2] >= image_width - edge_margin
        )
        if touches_edge and (
            aspect_ratios[index]
            < reference_ratio * HARD_CASE_EDGE_MAX_ASPECT_RATIO_TO_MEDIAN
        ):
            candidates.append((aspect_ratios[index], index, side))

    # Both ends can be truncated. Remove the narrower end first, and never
    # remove more than the overflow or delete a narrow interior column.
    removed_indices: set[int] = set()
    for ratio, index, side in sorted(candidates)[: len(display_groups) - expected_count]:
        removed_indices.add(index)
        logger.info(
            "pick/locate hard case 剔除贴边窄列 side=%s bbox=%s "
            "aspect_ratio=%.4f reference_aspect_ratio=%.4f relative_ratio=%.4f "
            "image_width=%s visible_before=%s visible_after=%s configured=%s",
            side, bboxes[index], ratio, reference_ratio, ratio / reference_ratio,
            image_width, len(display_groups), len(display_groups) - len(removed_indices),
            expected_count,
        )
    return [
        group for index, group in enumerate(display_groups) if index not in removed_indices
    ]


def detect_red_shelf_front_line(
    image: Image.Image,
) -> tuple[float, float] | None:
    """Detect the upper edge of the red shelf strip as a perspective line.

    A small Hough-like search is used instead of assuming a horizontal shelf:
    wrist-camera perspective commonly makes the red strip slope across the image.
    The upper color transition is preferred over the middle/lower part of the
    strip because product bottoms align with the shelf surface. The winning line
    must be supported across a meaningful horizontal span, which prevents
    isolated red packaging from becoming the shelf reference.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 20 or height < 20:
        return None
    step = max(1, min(width, height) // 360)
    x_bucket_size = max(12, width // 48)
    intercept_bin_size = max(5, height // 140)
    candidates: list[tuple[float, float]] = []
    upper_edge_candidates: list[tuple[float, float]] = []
    pixels = rgb.load()

    def is_shelf_red(x: int, y: int) -> bool:
        red, green, blue = pixels[x, y]
        return red >= 90 and red - green >= 35 and red - blue >= 25

    edge_probe_gap = max(2, height // 160)
    for y in range(int(height * 0.55), height, step):
        for x in range(0, width, step):
            if not is_shelf_red(x, y):
                continue
            point = (float(x), float(y))
            candidates.append(point)
            probe_y = max(0, y - edge_probe_gap)
            if not is_shelf_red(x, probe_y):
                upper_edge_candidates.append(point)
    if not candidates:
        return None

    minimum_support = max(6, int(width / x_bucket_size * 0.18))

    def fit_supported_line(
        points: list[tuple[float, float]],
        *,
        prefer_upper: bool,
    ) -> tuple[float, float] | None:
        line_candidates: list[tuple[int, float, int]] = []
        for slope_index in range(-10, 11):
            slope = slope_index * 0.025
            bins: dict[int, set[int]] = defaultdict(set)
            for x, y in points:
                intercept_bin = round((y - slope * x) / intercept_bin_size)
                bins[intercept_bin].add(int(x) // x_bucket_size)
            line_candidates.extend(
                (len(x_buckets), slope, intercept_bin)
                for intercept_bin, x_buckets in bins.items()
                if len(x_buckets) >= minimum_support
            )
        if not line_candidates:
            return None

        maximum_support = max(candidate[0] for candidate in line_candidates)
        strongly_supported = [
            candidate
            for candidate in line_candidates
            if candidate[0] >= max(minimum_support, math.ceil(maximum_support * 0.85))
        ]
        if prefer_upper:
            _, slope, intercept_bin = min(
                strongly_supported,
                key=lambda candidate: (
                    candidate[1] * width / 2
                    + candidate[2] * intercept_bin_size,
                    -candidate[0],
                ),
            )
        else:
            _, slope, intercept_bin = max(
                strongly_supported,
                key=lambda candidate: candidate[0],
            )
        approximate_intercept = intercept_bin * intercept_bin_size
        supporting_points = [
            (x, y)
            for x, y in points
            if abs((y - slope * x) - approximate_intercept)
            <= intercept_bin_size
        ]
        if not supporting_points:
            return None

        # Refine the quantized Hough slope with a continuous least-squares fit
        # over its supported red-edge points. The Hough stage supplies the
        # robust inlier set; this stage removes the 0.025 slope quantization that
        # otherwise becomes visible as row drift toward the image edges.
        mean_x = sum(x for x, _y in supporting_points) / len(supporting_points)
        mean_y = sum(y for _x, y in supporting_points) / len(supporting_points)
        denominator = sum((x - mean_x) ** 2 for x, _y in supporting_points)
        if denominator > 0:
            refined_slope = sum(
                (x - mean_x) * (y - mean_y)
                for x, y in supporting_points
            ) / denominator
            if -0.30 <= refined_slope <= 0.30:
                slope = refined_slope
        residuals = sorted(y - slope * x for x, y in supporting_points)
        return slope, residuals[len(residuals) // 2]

    upper_edge_line = fit_supported_line(
        upper_edge_candidates,
        prefer_upper=True,
    )

    def is_plausible_shelf_line(line: tuple[float, float] | None) -> bool:
        if line is None:
            return False
        slope, intercept = line
        center_y = slope * width / 2 + intercept
        # A close wrist-camera view can put the shelf surface in the last few
        # image rows. Keep a small one-pixel guard, but do not reject a valid
        # interrupted shelf edge merely because it lies below 98% of the frame.
        return height * 0.55 <= center_y <= height * 0.995

    if is_plausible_shelf_line(upper_edge_line):
        assert upper_edge_line is not None
        return upper_edge_line
    fallback_line = fit_supported_line(candidates, prefer_upper=False)
    return fallback_line if is_plausible_shelf_line(fallback_line) else None


def instance_lower_contour_points(
    instance: LocatedInstance,
) -> list[tuple[float, float]]:
    """Return one lower-mask point per sufficiently supported image column."""

    fallback = [
        (
            instance_center_x(instance),
            float(instance.bbox[3]),
        )
    ]
    encoded = instance.mask.split(",", 1)[-1]
    if not encoded:
        return fallback
    try:
        mask_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(mask_bytes)) as source_mask:
            mask = source_mask.convert("L")
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError):
        return fallback

    left = max(0, int(math.floor(instance.bbox[0])))
    top = max(0, int(math.floor(instance.bbox[1])))
    right = min(mask.width, int(math.ceil(instance.bbox[2])))
    bottom = min(mask.height, int(math.ceil(instance.bbox[3])))
    if left >= right or top >= bottom:
        return fallback

    pixels = mask.load()
    contour_points: list[tuple[float, float]] = []
    for x in range(left, right):
        for y in range(bottom - 1, top - 1, -1):
            if pixels[x, y] >= 128:
                contour_points.append((x + 0.5, float(y + 1)))
                break
    minimum_columns = max(3, int((right - left) * 0.20))
    return contour_points if len(contour_points) >= minimum_columns else fallback


def instance_lower_contact_y(instance: LocatedInstance) -> float:
    """Estimate the object's main lower contour without trusting mask tails."""

    column_bottoms = sorted(
        y for _x, y in instance_lower_contour_points(instance)
    )

    quantile = min(1.0, max(0.0, HARD_CASE_MASK_LOWER_CONTACT_QUANTILE))
    quantile_index = min(
        len(column_bottoms) - 1,
        max(0, math.ceil(quantile * len(column_bottoms)) - 1),
    )
    return min(instance.bbox[3], float(column_bottoms[quantile_index]))


def instance_shelf_front_distance(
    instance: LocatedInstance,
    shelf_front_line: tuple[float, float],
    *,
    lower_contact_quantile: float = HARD_CASE_MASK_LOWER_CONTACT_QUANTILE,
) -> tuple[float, float]:
    """Return robust signed perpendicular distance to the shelf-front line.

    Positive values place the mask above/behind the shelf front; negative values
    mean that the mask crosses below it. Sampling the full lower contour avoids
    the perspective error caused by comparing one global bottom Y with the line
    only at the bbox center.
    """

    slope, intercept = shelf_front_line
    normal_scale = math.sqrt(slope * slope + 1.0)
    perpendicular_distances = sorted(
        (slope * x - y + intercept) / normal_scale
        for x, y in instance_lower_contour_points(instance)
    )
    # A lower contour corresponds to a smaller distance. This is the
    # perpendicular-distance equivalent of the previous 80th-percentile Y.
    lower_distance_quantile = 1.0 - min(
        1.0,
        max(0.0, lower_contact_quantile),
    )
    quantile_index = min(
        len(perpendicular_distances) - 1,
        max(
            0,
            math.ceil(
                lower_distance_quantile * len(perpendicular_distances)
            )
            - 1,
        ),
    )
    distance_pixels = perpendicular_distances[quantile_index]
    perpendicular_height = max(
        1.0,
        (instance.bbox[3] - instance.bbox[1]) / normal_scale,
    )
    return distance_pixels, distance_pixels / perpendicular_height


def cluster_display_row_measurements(
    measurements: list[tuple[int, float]],
    *,
    row_gap_ratio: float,
) -> dict[int, int]:
    """Assign rows by gaps between sorted shelf-front distances."""

    row_by_index: dict[int, int] = {}
    previous_value: float | None = None
    row_index = 1
    for instance_index, value in sorted(measurements, key=lambda item: item[1]):
        if previous_value is not None and value - previous_value > row_gap_ratio:
            row_index += 1
        row_by_index[instance_index] = row_index
        previous_value = value
    return row_by_index


def assign_display_row_indices(
    instances: list[LocatedInstance],
    *,
    shelf_front_line: tuple[float, float] | None,
    contact_gap_ratio: float = MULTI_ROW_CONTACT_GAP_RATIO,
) -> list[LocatedInstance]:
    """Annotate visible depth rows, where row 1 is nearest to the shelf front."""

    if not instances:
        return []

    if shelf_front_line is None:
        return [
            instance.model_copy(
                update={
                    "display_row_index": 1,
                    "display_row_source": "unresolved",
                }
            )
            for instance in instances
        ]

    measurements: list[tuple[int, float]] = []
    for index, instance in enumerate(instances):
        _distance_pixels, signed_distance_ratio = instance_shelf_front_distance(
            instance,
            shelf_front_line,
            lower_contact_quantile=MULTI_ROW_MASK_LOWER_CONTACT_QUANTILE,
        )
        measurements.append((index, max(0.0, signed_distance_ratio)))
    source = "shelf_front"
    row_by_index = cluster_display_row_measurements(
        measurements,
        row_gap_ratio=max(0.0, contact_gap_ratio),
    )

    annotated: list[LocatedInstance] = []
    ratio_by_index = dict(measurements) if source == "shelf_front" else {}
    for index, instance in enumerate(instances):
        annotated.append(
            instance.model_copy(
                update={
                    "display_row_index": row_by_index.get(index, 1),
                    "display_row_source": source,
                    "shelf_front_distance_ratio": (
                        round(ratio_by_index[index], 4)
                        if index in ratio_by_index
                        else None
                    ),
                }
            )
        )
    return annotated


def keep_nearest_display_row(
    instances: list[LocatedInstance],
) -> list[LocatedInstance]:
    """Keep the nearest assigned row while guaranteeing at least one candidate."""

    assigned = [
        instance.display_row_index
        for instance in instances
        if instance.display_row_index is not None
    ]
    if not assigned:
        return instances
    nearest_row = min(assigned)
    selected = [
        instance
        for instance in instances
        if instance.display_row_index == nearest_row
    ]
    return selected or instances


def keep_frontmost_by_mask_contact(
    instances: list[LocatedInstance],
) -> list[LocatedInstance]:
    """Keep side-by-side masks; suppress rear masks only with contact evidence."""
    if len(instances) < 2:
        return instances

    geometry: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = {}

    def mask_geometry(
        index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        if index not in geometry:
            try:
                encoded = instances[index].mask.split(",", 1)[-1]
                mask_bytes = base64.b64decode(encoded, validate=True)
                with Image.open(io.BytesIO(mask_bytes)) as image:
                    mask = np.asarray(image.convert("L")) >= 128
            except (ValueError, binascii.Error, UnidentifiedImageError, OSError) as error:
                raise HTTPException(
                    status_code=502, detail=f"SAM3 mask PNG 无效: {error}"
                ) from error
            radius = max(1, round(PICK_MASK_CONTACT_RADIUS * min(mask.shape) / 480))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
            )
            binary = mask.astype(np.uint8)
            boundary = mask & ~(
                cv2.erode(
                    binary,
                    np.ones((3, 3), np.uint8),
                    borderType=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ) > 0
            )
            geometry[index] = (
                mask,
                boundary,
                cv2.dilate(binary, kernel) > 0,
                int(np.count_nonzero(mask)),
                radius,
            )
        return geometry[index]

    def distance(index: int) -> float:
        value = instances[index].shelf_front_distance_ratio
        return value if value is not None and math.isfinite(value) else math.inf

    # Only a nearer mask can occlude another. A rear mask touching two front
    # bottles must never merge those front bottles into one connected component.
    ordered = sorted(
        range(len(instances)),
        key=lambda index: (
            distance(index),
            -(instances[index].score if instances[index].score is not None else -1.0),
            instance_center_x(instances[index]),
        ),
    )
    selected: list[int] = []
    for position, index in enumerate(ordered):
        mask, boundary, dilated, area, radius = mask_geometry(index)
        candidate = instances[index]
        # A suppressed middle bottle still provides evidence that a third
        # touching bottle is farther back in the same occlusion chain.
        for previous in ordered[:position]:
            existing = instances[previous]
            # Bboxes only bound the search. Their intersection alone never
            # establishes either duplicate detections or front/back occlusion.
            if (
                max(candidate.bbox[0], existing.bbox[0])
                > min(candidate.bbox[2], existing.bbox[2]) + radius
                or max(candidate.bbox[1], existing.bbox[1])
                > min(candidate.bbox[3], existing.bbox[3]) + radius
            ):
                continue
            other, other_boundary, other_dilated, other_area, _ = mask_geometry(
                previous
            )
            if mask.shape != other.shape:
                continue
            intersection = int(np.count_nonzero(mask & other))
            union = area + other_area - intersection
            mask_iou = intersection / union if union else 0.0
            contact_pixels = min(
                int(np.count_nonzero(boundary & other_dilated)),
                int(np.count_nonzero(other_boundary & dilated)),
            )
            shorter_boundary = min(
                int(np.count_nonzero(boundary)),
                int(np.count_nonzero(other_boundary)),
            )
            contact_ratio = (
                contact_pixels / shorter_boundary if shorter_boundary else 0.0
            )
            gap = (
                distance(index) - distance(previous)
                if math.isfinite(distance(index)) and math.isfinite(distance(previous))
                else None
            )
            duplicate = mask_iou >= PICK_DUPLICATE_MASK_IOU
            occluded = (
                gap is not None
                and gap > MULTI_ROW_CONTACT_GAP_RATIO
                and contact_pixels >= 8
                and contact_ratio >= PICK_MIN_MASK_CONTACT_RATIO
            )
            logger.info(
                "Pick Locate mask relation candidate_bbox=%s previous_bbox=%s "
                "mask_iou=%.4f contact_pixels=%s contact_ratio=%.4f "
                "shelf_distance_gap=%s decision=%s",
                candidate.bbox,
                existing.bbox,
                mask_iou,
                contact_pixels,
                contact_ratio,
                gap,
                "duplicate" if duplicate else "occluded" if occluded else "keep",
            )
            if duplicate or occluded:
                break
        else:
            selected.append(index)
    return [instances[index] for index in sorted(selected)]


def keep_display_rows_for_inventory(
    instances: list[LocatedInstance],
    required_count: int,
) -> list[LocatedInstance]:
    """Fill slots by shelf-front distance after removing tiny mask outliers."""

    if not instances or required_count <= 0:
        return []
    rows = sorted({instance.display_row_index or 1 for instance in instances})
    quality_candidates: list[LocatedInstance] = []
    for row_index in rows:
        quality_candidates.extend(
            drop_smallest_mask_area_outlier(
                [
                    instance
                    for instance in instances
                    if (instance.display_row_index or 1) == row_index
                ]
            )
        )
    # Use the same mask-contact rule within and across display rows. In
    # particular, tilted neighboring bottles may have overlapping bboxes even
    # though their masks are separate and both sit at the shelf front.
    candidates = keep_frontmost_by_mask_contact(quality_candidates)
    selected: list[LocatedInstance] = []
    for row_index in rows:
        available = [
            instance
            for instance in candidates
            if (instance.display_row_index or 1) == row_index
        ]
        needed = required_count - len(selected)
        if len(available) > needed:
            distances_available = all(
                instance.shelf_front_distance_ratio is not None
                and math.isfinite(instance.shelf_front_distance_ratio)
                for instance in available
            )
            if distances_available:
                # 同一候选排超出库存数量时，优先保留 mask 底部最靠近红色货架前沿的实例。
                available = sorted(
                    available,
                    key=lambda instance: (
                        instance.shelf_front_distance_ratio,
                        -(instance.score if instance.score is not None else -1.0),
                        instance_center_x(instance),
                    ),
                )[:needed]
            else:
                # 红色货架前沿不可用时，才回退到原来的 mask 面积排序。
                available = sorted(
                    available,
                    key=lambda instance: (
                        mask_foreground_pixel_count(instance.mask),
                        instance.score if instance.score is not None else -1.0,
                        (instance.bbox[2] - instance.bbox[0])
                        * (instance.bbox[3] - instance.bbox[1]),
                    ),
                    reverse=True,
                )[:needed]
        selected.extend(available)
        if len(selected) >= required_count:
            break
    return sorted(selected, key=instance_center_x)


def map_inventory_slots_to_instances(
    instances: list[LocatedInstance],
    inventory_slots: list[str],
    target_slot_id: str,
    image_width: float,
) -> tuple[list[LocatedInstance], LocatedInstance]:
    """Map a visible left/right inventory slice to image-left-to-right boxes."""

    ordered_instances = sorted(instances, key=instance_center_x)
    visible_count = min(len(ordered_instances), len(inventory_slots))
    ordered_instances = ordered_instances[:visible_count]
    if visible_count == 0:
        visible_inventory_slots: list[str] = []
    elif visible_count >= len(inventory_slots):
        visible_inventory_slots = list(inventory_slots)
    else:
        detected_left = min(instance.bbox[0] for instance in ordered_instances)
        detected_right = max(instance.bbox[2] for instance in ordered_instances)
        detected_group_center = (detected_left + detected_right) / 2
        if detected_group_center <= image_width / 2:
            visible_inventory_slots = inventory_slots[-visible_count:]
        else:
            visible_inventory_slots = inventory_slots[:visible_count]
    mapped_instances: list[LocatedInstance] = []
    selected_instance: LocatedInstance | None = None
    for position, (instance, mapped_slot) in enumerate(
        zip(ordered_instances, visible_inventory_slots),
        start=1,
    ):
        updated = instance.model_copy(
            update={
                "mapped_slot_id": mapped_slot,
                "display_position_in_row": position,
                "is_selected": mapped_slot == target_slot_id,
            }
        )
        mapped_instances.append(updated)
        if mapped_slot == target_slot_id:
            selected_instance = updated
    if selected_instance is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"目标槽位不在当前可见库存范围: {target_slot_id}; "
                f"visible={visible_inventory_slots}"
            ),
        )
    return mapped_instances, selected_instance


def assign_display_positions(
    instances: list[LocatedInstance],
) -> list[LocatedInstance]:
    """Annotate the left-to-right position inside each assigned display row."""

    positions: dict[int, int] = {}
    by_row: dict[int, list[int]] = defaultdict(list)
    for index, instance in enumerate(instances):
        by_row[instance.display_row_index or 1].append(index)
    for indices in by_row.values():
        for position, index in enumerate(
            sorted(indices, key=lambda value: instance_center_x(instances[value])),
            start=1,
        ):
            positions[index] = position
    return [
        instance.model_copy(
            update={"display_position_in_row": positions[index]}
        )
        for index, instance in enumerate(instances)
    ]


def apply_pick_bbox_history(
    instances: list[LocatedInstance],
    previous_bboxes_normalized: list[list[int]],
    image_size: tuple[int, int],
    *,
    overlap_threshold: float = PICK_HISTORY_OVERLAP_RATIO,
) -> list[LocatedInstance]:
    """Raise row indices for candidates exposed behind previously picked boxes."""

    if not instances or not previous_bboxes_normalized:
        return instances
    width, height = image_size
    history_boxes = [
        [
            float(bbox[0]) / 1000.0 * width,
            float(bbox[1]) / 1000.0 * height,
            float(bbox[2]) / 1000.0 * width,
            float(bbox[3]) / 1000.0 * height,
        ]
        for bbox in previous_bboxes_normalized
        if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]
    ]
    threshold = max(0.0, min(1.0, overlap_threshold))
    annotated: list[LocatedInstance] = []
    for instance in instances:
        overlap_count = sum(
            bbox_overlap_by_smaller_area(instance.bbox, history_bbox) >= threshold
            for history_bbox in history_boxes
        )
        geometry_row = instance.display_row_index or 1
        history_row = overlap_count + 1
        row_index = max(geometry_row, history_row)
        source = instance.display_row_source or "unresolved"
        if overlap_count:
            source = f"{source}+history"
        annotated.append(
            instance.model_copy(
                update={
                    "display_row_index": row_index,
                    "display_row_source": source,
                    "history_overlap_count": overlap_count,
                }
            )
        )
    return annotated


def keep_front_depth_row(
    instances: list[LocatedInstance],
    shelf_front_line: tuple[float, float] | None = None,
) -> list[LocatedInstance]:
    """Keep instances whose lower edge reaches the detected shelf front."""
    if len(instances) <= 1:
        return instances
    heights = [max(0.0, item.bbox[3] - item.bbox[1]) for item in instances]
    if shelf_front_line is not None:
        slope, _intercept = shelf_front_line
        normal_scale = math.sqrt(slope * slope + 1.0)
        measurements: list[tuple[LocatedInstance, float, float, float]] = []
        for item, height in zip(instances, heights):
            _distance_pixels, signed_distance_ratio = instance_shelf_front_distance(
                item,
                shelf_front_line,
            )
            # The mask/bbox may end slightly above the visible red edge or extend
            # through it. Use the same line-normal coordinate as multi-row
            # assignment so shelf slope cannot shift candidates between rows.
            perpendicular_height = max(1.0, height / normal_scale)
            lower_tolerance_ratio = max(
                12.0 / perpendicular_height,
                HARD_CASE_FRONT_LOWER_TOLERANCE_RATIO,
            )
            if signed_distance_ratio < -lower_tolerance_ratio:
                continue
            # Masks that slightly cross the shelf line belong to the same
            # zero-distance front cluster; how far they cross is handled by the
            # lower-tolerance rejection above.
            distance_ratio = max(0.0, signed_distance_ratio)
            base_upper_tolerance_ratio = max(
                12.0 / perpendicular_height,
                HARD_CASE_FRONT_UPPER_TOLERANCE_RATIO,
            )
            maximum_upper_tolerance_ratio = max(
                12.0 / perpendicular_height,
                HARD_CASE_FRONT_MAX_UPPER_TOLERANCE_RATIO,
            )
            measurements.append(
                (
                    item,
                    distance_ratio,
                    signed_distance_ratio - base_upper_tolerance_ratio,
                    signed_distance_ratio - maximum_upper_tolerance_ratio,
                )
            )

        base_measurements = sorted(
            (
                measurement
                for measurement in measurements
                if measurement[2] <= 0
            ),
            key=lambda measurement: measurement[1],
        )
        # The 25% threshold is a candidate ceiling, not an unconditional row.
        # Keep only the nearest continuous distance cluster so rear bottles whose
        # lower edge still happens to fall inside 25% cannot become extra columns.
        front_measurements: list[
            tuple[LocatedInstance, float, float, float]
        ] = []
        for measurement in base_measurements:
            if (
                front_measurements
                and measurement[1] - front_measurements[-1][1]
                > HARD_CASE_FRONT_DISTANCE_GAP_RATIO
            ):
                break
            front_measurements.append(measurement)
        # Borderline masks can end a few pixels above the baseline tolerance.
        # Expand only through nearby normalized distances and stop at a visible
        # distance gap, with a hard cap at the configured 0.35 ratio.
        relaxed_measurements = sorted(
            (
                measurement
                for measurement in measurements
                if measurement[2] > 0 and measurement[3] <= 0
            ),
            key=lambda measurement: measurement[1],
        )
        base_instances = [measurement[0] for measurement in front_measurements]
        allow_relaxation = len(bbox_overlap_components(base_instances)) <= 1
        if front_measurements and allow_relaxation:
            current_ratio = max(
                measurement[1] for measurement in front_measurements
            )
            for measurement in relaxed_measurements:
                if (
                    measurement[1] - current_ratio
                    > HARD_CASE_FRONT_DISTANCE_GAP_RATIO
                ):
                    break
                front_measurements.append(measurement)
                current_ratio = measurement[1]
        elif not front_measurements and relaxed_measurements:
            front_measurements.append(relaxed_measurements[0])
            current_ratio = relaxed_measurements[0][1]
            for measurement in relaxed_measurements[1:]:
                if (
                    measurement[1] - current_ratio
                    > HARD_CASE_FRONT_DISTANCE_GAP_RATIO
                ):
                    break
                front_measurements.append(measurement)
                current_ratio = measurement[1]

        front_ids = {id(measurement[0]) for measurement in front_measurements}
        front = [item for item in instances if id(item) in front_ids]
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
    level: str | None,
    hand: str,
    slot_id: str | None = None,
    target_id: str | None = None,
    shelf_front_line: tuple[float, float] | None = None,
    sku_row_lookup: Callable[[str], list[dict[str, Any]]] | None = None,
    image_width: int | None = None,
) -> tuple[list[LocatedInstance], HardCaseDebugInfo | None]:
    resolved_sku_row_lookup = sku_row_lookup or lookup_sku_row
    slot_context = validate_slot_hard_case_context(
        product,
        task_type,
        level,
        hand,
        slot_id,
        target_id,
    )
    hard_case = hard_case_group_for_product(
        product["name"].strip(),
        task_type,
        level,
        hand,
        slot_id=slot_id,
        target_id=target_id,
    )
    if hard_case is None:
        return instances, None
    group_id, config = hard_case
    actual_hand = hand.strip().lower()
    preferred_hand = actual_hand

    target_name = product["name"].strip()
    display_groups = split_instances_into_display_groups(instances, shelf_front_line)
    visible_count = len(display_groups)

    visible_slot_order: list[str] = []
    slot_view_applied = slot_context is not None
    normalized_target_id: str | None = None
    layout_override: tuple[str, ...] | None = None
    if slot_context is not None:
        context_group_id, view = slot_context
        if context_group_id != group_id:
            raise HTTPException(status_code=500, detail="hard case 商品组判定不一致")
        target_location = (slot_id or "").strip().upper()
        target_match = LOCATION_PATTERN.fullmatch(target_location)
        assert target_match is not None
        normalized_target_id = view.target_id
        visible_slot_order = list(view.visible_slot_order)
        row = resolved_sku_row_lookup(target_location)
        row_by_slot = {
            str(item.get("location_id", "")).strip().upper(): item
            for item in row
            if isinstance(item, dict)
        }
        missing_view_slots = [
            value for value in visible_slot_order if value not in row_by_slot
        ]
        if missing_view_slots:
            raise HTTPException(
                status_code=502,
                detail=(
                    "SKU 行数据缺少 hard case 视角槽位: "
                    f"{missing_view_slots}"
                ),
            )
        directional_order = [
            str(row_by_slot[value].get("name", "")).strip()
            for value in visible_slot_order
        ]
        if any(name not in config.members for name in directional_order):
            raise HTTPException(
                status_code=502,
                detail="hard case 视角槽位对应了非同组商品",
            )
        if row_by_slot[target_location].get("name", "").strip() != target_name:
            raise HTTPException(status_code=502, detail="SKU 行数据中的商品与目标槽位不一致")
        standard_order = [
            str(item.get("name", "")).strip()
            for item in row
            if str(item.get("name", "")).strip() in config.members
        ]
        display_groups = drop_excess_truncated_edge_groups(
            display_groups, len(visible_slot_order), image_width
        )
        visible_count = len(display_groups)
        if visible_count > len(visible_slot_order):
            # After confident edge trimming, take N columns from the hand's
            # side. Keep their image-left-to-right order for the slot view map.
            if actual_hand == "left":
                selection_direction = "left_to_right"
                kept_groups = display_groups[: len(visible_slot_order)]
                discarded_groups = display_groups[len(visible_slot_order) :]
            else:
                selection_direction = "right_to_left"
                kept_groups = display_groups[-len(visible_slot_order) :]
                discarded_groups = display_groups[: -len(visible_slot_order)]
            logger.warning(
                "pick/locate hard case 超列兜底保留前N列 target_id=%s hand=%s "
                "selection_direction=%s "
                "slot_id=%s visible=%s configured=%s visible_slot_order=%s "
                "kept_bboxes=%s discarded_bboxes=%s",
                normalized_target_id, actual_hand, selection_direction, target_location,
                visible_count, len(visible_slot_order), visible_slot_order,
                [union_instance_bboxes(group) for group in kept_groups],
                [union_instance_bboxes(group) for group in discarded_groups],
            )
            display_groups = kept_groups
            visible_count = len(display_groups)
        if visible_count != len(visible_slot_order):
            raise HTTPException(
                status_code=422,
                detail=(
                    "检测列数与 hard case 视角槽位数不一致: "
                    f"visible={visible_count}, configured={len(visible_slot_order)}"
                ),
            )
        # The view config is always image-left-to-right. In particular, a right
        # wrist at H12_INSPECT sees H2's left columns and must not use a suffix.
        directional_groups = display_groups
    else:
        target_location, target_match = select_location_for_level(product, level, hand)
        standard_order = hard_case_standard_order(
            target_location,
            config,
            resolved_sku_row_lookup,
        )
        if target_name not in standard_order:
            raise HTTPException(status_code=502, detail="目标 SKU 不在指定层品牌顺序中")
        layout_override = hard_case_layout_order_for_request(
            target_name,
            level,
            actual_hand,
        )
        if layout_override is not None:
            unknown_products = [
                name for name in layout_override if name not in config.members
            ]
            if unknown_products:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "hard case 布局覆盖包含非同组商品: "
                        f"{unknown_products}"
                    ),
                )
            if visible_count != len(layout_override):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "检测列数与图片布局覆盖不一致: "
                        f"visible={visible_count}, override={len(layout_override)}"
                    ),
                )
            directional_groups = (
                display_groups
                if actual_hand == "left"
                else list(reversed(display_groups))
            )
            directional_order = list(layout_override)
        else:
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

    if slot_context is None and target_name not in directional_order:
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
    mapped_slots: list[str | None] = (
        list(visible_slot_order)
        if slot_context is not None
        else [None] * len(directional_order)
    )
    for group_index, (group, mapped_name, mapped_slot) in enumerate(
        zip(directional_groups, directional_order, mapped_slots),
        start=1,
    ):
        front_instances = keep_front_depth_row(group, shelf_front_line)
        if not front_instances:
            raise HTTPException(status_code=422, detail=f"{mapped_name} 没有第一排实例")
        is_target_group = (
            mapped_slot == target_location
            if slot_context is not None
            else mapped_name == target_name and selected_instance is None
        )
        if is_target_group:
            selected_group_index = group_index
            selected_instance = select_frontmost_instance(front_instances)
        updated_group: list[LocatedInstance] = []
        for instance in front_instances:
            updated = instance.model_copy(
                update={
                    "hard_case_group_index": group_index,
                    "mapped_product_name": mapped_name,
                    "mapped_slot_id": mapped_slot,
                    "is_selected": instance is selected_instance,
                }
            )
            updated_group.append(updated)
            annotated_instances.append(updated)
        debug_groups.append(
            HardCaseGroupResult(
                index=group_index,
                mapped_product_name=mapped_name,
                mapped_slot_id=mapped_slot,
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
        order_direction=(
            "left_to_right"
            if slot_context is not None or actual_hand == "left"
            else "right_to_left"
        ),
        target_location=target_location,
        target_level=int(target_match.group("level")),
        target_column=int(target_match.group("column")),
        standard_order=standard_order,
        layout_override_applied=layout_override is not None,
        groups=debug_groups,
        selected_group_index=selected_group_index,
        target_slot_id=target_location if slot_context is not None else None,
        target_id=normalized_target_id,
        visible_slot_order=visible_slot_order,
        slot_view_applied=slot_view_applied,
    )


def locate_product_in_image(
    product: dict[str, Any],
    image_path: Path,
    *,
    task_type: str = "SORTING",
    level: str | None = None,
    hand: str = "left",
    slot_id: str | None = None,
    target_id: str | None = None,
    qwen_prompt_override: str | None = None,
    sam_prompt_override: str | None = None,
    depth_image: Image.Image | None = None,
    depth_image_provider: Callable[[tuple[int, int]], Image.Image | None] | None = None,
    capture_postprocess_errors: bool = False,
    previous_picked_bboxes: list[list[int]] | None = None,
    sku_row_lookup: Callable[[str], list[dict[str, Any]]] | None = None,
) -> LocateDebugResponse:
    """使用已查询的 SKU 信息，在指定 RGB 图片上运行完整定位流程。"""
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"测试图片不存在: {image_path.name}")
    monitor_image_path = store_monitor_image(image_path)
    canonical_name = product["name"].strip()
    normalized_level = normalize_level(level) if level else None
    slot_context = validate_slot_hard_case_context(
        product,
        task_type,
        normalized_level,
        hand,
        slot_id,
        target_id,
    )
    hard_case = hard_case_group_for_product(
        canonical_name,
        task_type,
        normalized_level,
        hand,
        slot_id=slot_id,
        target_id=target_id,
    )
    inventory_slots = (
        inventory_slots_for_pick(
            product,
            task_type,
            slot_id,
            normalized_level,
        )
        if hard_case is None
        else []
    )
    inventory_target_slot = (slot_id or "").strip().upper()
    inventory_row_mapping = supports_inventory_row_mapping(product)
    inventory_row_slots = inventory_slots if inventory_row_mapping else []
    single_location_target_slot = (
        inventory_target_slot
        if inventory_slots and not inventory_row_mapping
        else None
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
        assert normalized_level is not None
        hard_case_group_id, _ = hard_case
        if slot_context is not None:
            target_location = (slot_id or "").strip().upper()
            target_match = LOCATION_PATTERN.fullmatch(target_location)
            assert target_match is not None
        else:
            target_location, target_match = select_location_for_level(
                product, normalized_level, hand
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
            f"对应的 L{int(target_match.group('level'))} 层；如果该 SKU 有多个位置，"
            "这里使用调用方指定的层。不要输出同品牌在其他货架层的商品；"
            f"{bbox_instruction}"
        )
    try:
        with Image.open(image_path) as source_image:
            original_image = source_image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=500, detail=f"读取 RGB 图片失败: {error}") from error
    inference_image, inference_image_bytes = prepare_rgb_inference_image(
        original_image
    )
    logger.info(
        "Pick Locate inference started task_type=%s product_name=%s sku_id=%s "
        "level=%s hand=%s slot_id=%s target_id=%s image_name=%s "
        "image_size=%sx%s hard_case=%s inventory_slots=%s "
        "inventory_row_mapping=%s",
        task_type,
        canonical_name,
        product.get("sku_id"),
        normalized_level,
        hand,
        slot_id,
        target_id,
        image_path.name,
        original_image.width,
        original_image.height,
        hard_case is not None,
        inventory_slots,
        inventory_row_mapping,
    )

    qwen_reference_image = (
        fetch_sku_reference_image(product)
        if task_type.strip().upper() == "SHORTAGE"
        else None
    )
    qwen_bboxes = (
        get_stable_qwen_bboxes(
            qwen_prompt,
            inference_image_bytes,
            reference_image=qwen_reference_image,
        )
        if qwen_reference_image is not None
        else get_stable_qwen_bboxes(qwen_prompt, inference_image_bytes)
    )
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
    qwen_bboxes_by_source: dict[int, list[float]] = {}
    located_instances: list[LocatedInstance] = []
    crop_padding_ratio = qwen_crop_padding_ratio(canonical_name)
    for qwen_index, qwen_bbox in enumerate(qwen_bboxes):
        try:
            crop_box = qwen_bbox_to_crop(
                qwen_bbox,
                inference_image.size,
                padding_ratio=crop_padding_ratio,
            )
        except ValueError:
            continue
        qwen_bbox_original = qwen_bbox_to_original(
            qwen_bbox,
            original_image.size,
        )
        qwen_bboxes_by_source[qwen_index] = qwen_bbox_original
        qwen_bbox_records.append(
            QwenBBoxRecord(
                bbox_normalized=qwen_bbox,
                bbox_original=qwen_bbox_original,
                crop_box_original=map_crop_box_between_sizes(
                    crop_box,
                    inference_image.size,
                    original_image.size,
                ),
            )
        )
        crop_image = inference_image.crop(crop_box)
        for instance in call_sam3(sam_prompt, crop_image):
            if isinstance(instance, dict):
                located_instances.append(
                    map_sam_instance_to_original(
                        instance,
                        crop_box,
                        inference_image.size,
                        original_image.size,
                        source_qwen_index=qwen_index,
                    )
                )

    if not located_instances:
        raise HTTPException(status_code=404, detail="SAM3 没有找到目标商品实例")

    raw_sam_instances = list(located_instances)
    logger.info(
        "Pick Locate detections collected product_name=%s stable_qwen_bbox_count=%s "
        "raw_qwen_detection_count=%s raw_sam_instance_count=%s",
        canonical_name,
        len(qwen_bbox_records),
        len(raw_qwen_bbox_records),
        len(raw_sam_instances),
    )
    upper_confidence_pick = (
        hard_case is None
        and uses_upper_confidence_pick(canonical_name, task_type)
    )
    max_mask_area_pick = (
        hard_case is None
        and uses_max_mask_area_pick(canonical_name, task_type)
    )
    multi_row_pick = hard_case is None and inventory_row_mapping and (
        bool(inventory_row_slots)
        or uses_multi_row_pick(canonical_name, task_type)
    )
    if upper_confidence_pick:
        located_instances = keep_instances_from_nearest_qwen_shelf_row(
            located_instances,
            qwen_bboxes_by_source,
            original_image.height,
        )
        if not located_instances:
            raise HTTPException(
                status_code=404,
                detail="目标货架行没有可用的 SAM3 候选",
            )
    special_no_depth_pick = upper_confidence_pick or max_mask_area_pick
    if hard_case is None and not special_no_depth_pick:
        # This is the first normal-candidate filter. Remove SAM fragments that
        # occupy very little of their unpadded Qwen bbox before aspect, depth,
        # overlap-chain, or center-distance decisions can favor them.
        located_instances = keep_sam_instances_with_qwen_coverage(
            located_instances,
            qwen_bboxes_by_source,
        )
        if not located_instances:
            raise HTTPException(
                status_code=404,
                detail="SAM3 候选占原始 Qwen bbox 的面积均过小",
            )
    if hard_case is None and not special_no_depth_pick:
        # Reject visibly incomplete/sliver masks before row and overlap rules.
        located_instances = keep_visibly_complete_pick_candidates(
            located_instances
        )
    shelf_front_line = (
        detect_red_shelf_front_line(original_image)
        if hard_case is not None or multi_row_pick
        else None
    )
    if multi_row_pick:
        located_instances = assign_display_row_indices(
            located_instances,
            shelf_front_line=shelf_front_line,
        )
        if not inventory_row_slots:
            located_instances = apply_pick_bbox_history(
                located_instances,
                previous_picked_bboxes or [],
                original_image.size,
            )
        located_instances = assign_display_positions(located_instances)
        # Keep all annotated rows auditable. Inventory-aware requests admit
        # progressively farther rows until every remaining physical slot can be
        # mapped; legacy requests still keep only the nearest display row.
        raw_sam_instances = list(located_instances)
        located_instances = (
            keep_display_rows_for_inventory(
                located_instances,
                len(inventory_row_slots),
            )
            if inventory_row_slots
            else keep_nearest_display_row(located_instances)
        )
    if (
        hard_case is None
        and not special_no_depth_pick
        and not inventory_row_slots
    ):
        located_instances = keep_frontmost_in_overlap_chains(located_instances)
        # Preserve the original global outlier rule for normal SKUs. A hard-case
        # row can contain a legitimately smaller perspective-edge display group,
        # so its rear-row filtering is performed inside each mapped group below.
        located_instances = drop_smallest_mask_area_outlier(located_instances)
    elif hard_case is not None:
        # Hard cases now use one broad Qwen crop. Preserve all SAM instances here;
        # first-row filtering must run before overlap de-duplication so rear/partial
        # masks cannot transitively merge separate front-row product columns.
        located_instances = list(located_instances)
    inventory_selected_instance: LocatedInstance | None = None
    if inventory_row_slots:
        located_instances, inventory_selected_instance = map_inventory_slots_to_instances(
            located_instances,
            inventory_row_slots,
            inventory_target_slot,
            original_image.width,
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
            slot_id=slot_id,
            target_id=target_id,
            shelf_front_line=shelf_front_line,
            sku_row_lookup=sku_row_lookup,
            image_width=original_image.width,
        )
    except HTTPException as error:
        if not capture_postprocess_errors:
            raise
        postprocess_error = error

    if hard_case_debug is not None:
        hard_case_selected = [
            instance for instance in located_instances if instance.is_selected
        ]
        if len(hard_case_selected) != 1:
            raise HTTPException(status_code=500, detail="hard case 目标实例标记无效")
        selected_instance = hard_case_selected[0]
    elif inventory_selected_instance is not None:
        selected_instance = inventory_selected_instance
    elif max_mask_area_pick:
        # Preserve all candidates so the selected mask area remains auditable
        # against SAM3's original numbering.
        selected_instance = select_largest_mask_area_instance(
            located_instances
        )
    elif upper_confidence_pick:
        # Preserve all candidates in the Debug response so the selected upper
        # bbox remains auditable against SAM3's original candidate numbering.
        selected_instance = select_upper_high_confidence_instance(
            located_instances
        )
    else:
        selected_instance = select_pick_instance(
            located_instances,
            list(original_image.size),
        )
    if single_location_target_slot is not None and hard_case_debug is None:
        # The caller still receives the validated slot, but a one-location SKU
        # does not need shelf-line detection, display-row inference, or column
        # mapping to arrive at that already-unambiguous result.
        selected_instance.mapped_slot_id = single_location_target_slot
    selected_instance_index = next(
        index
        for index, instance in enumerate(located_instances, start=1)
        if instance is selected_instance
    )
    logger.info(
        "Pick Locate selection completed product_name=%s final_instance_count=%s "
        "selected_instance_index=%s selected_slot_id=%s selected_bbox=%s "
        "selected_depth_mm=%s shelf_front_distance_ratio=%s",
        canonical_name,
        len(located_instances),
        selected_instance_index,
        selected_instance.mapped_slot_id,
        selected_instance.bbox,
        selected_instance.depth_mm,
        selected_instance.shelf_front_distance_ratio,
    )

    return LocateDebugResponse(
        sku_id=product["sku_id"],
        product_name=canonical_name,
        image_name=image_path.name,
        image_path=monitor_image_path,
        image_base64=base64.b64encode(image_path.read_bytes()).decode("ascii"),
        image_media_type=mimetypes.guess_type(image_path.name)[0] or "image/jpeg",
        image_size=list(original_image.size),
        inference_image_size=list(inference_image.size),
        qwen3_prompt_used=qwen_prompt,
        sam3_prompt_used=sam_prompt,
        qwen_reference_image_used=qwen_reference_image is not None,
        qwen_reference_image_name=(
            qwen_reference_image.logical_name
            if qwen_reference_image is not None
            else None
        ),
        qwen_reference_image_media_type=(
            qwen_reference_image.media_type
            if qwen_reference_image is not None
            else None
        ),
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
        inference_image_size=list(RGB_INFERENCE_SIZE),
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
    mock_inventory: list[str] | None = None,
) -> LocateDebugResponse:
    product_name = request.product_name.strip()
    task_type = normalize_task_type(request.task_type)
    if not product_name:
        raise HTTPException(status_code=400, detail="product_name 不能为空")
    if mock_inventory is not None:
        if task_type != "SORTING":
            raise HTTPException(status_code=400, detail="mock_inventory 仅支持 SORTING 调试")
        product = lookup_mock_sku_by_name(product_name, mock_inventory)
    else:
        product = lookup_sku_by_name(product_name)
    requested_level = (request.level or "").strip()
    level = normalize_level(requested_level) if requested_level else None
    if level is None and hard_case_level_required(
        product["name"],
        task_type,
        request.hand,
    ):
        raise HTTPException(
            status_code=400,
            detail="该 SORTING 特例必须提供 level",
        )
    validate_slot_hard_case_context(
        product,
        task_type,
        level,
        request.hand,
        request.slot_id,
        request.target_id,
    )
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
    prompt_overrides["slot_id"] = request.slot_id
    prompt_overrides["target_id"] = request.target_id
    prompt_overrides["capture_postprocess_errors"] = capture_inference_errors
    prompt_overrides["previous_picked_bboxes"] = request.previous_picked_bboxes
    if mock_inventory is not None:
        prompt_overrides["sku_row_lookup"] = lookup_mock_sku_row
    if request.image_base64 is None:
        if request.image_name is not None:
            raise HTTPException(
                status_code=400,
                detail="指定 image_name 时必须同时提供 image_base64",
            )
        live_camera = camera_for_task(task_type, request.hand)
        image_path = get_latest_rgb(live_camera)
        try:
            return locate_product_in_image(
                product,
                image_path,
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
        finally:
            remove_camera_snapshot(image_path)

    image_bytes = decode_uploaded_image(request.image_base64)
    image_name = uploaded_image_name(request.image_name)
    with tempfile.TemporaryDirectory(prefix="locate-upload-") as temporary_directory:
        image_path = Path(temporary_directory) / image_name
        image_path.write_bytes(image_bytes)
        try:
            return locate_product_in_image(
                product,
                image_path,
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
    min_height_ratio_to_tallest: float = PICK_MIN_HEIGHT_RATIO_TO_TALLEST,
) -> list[LocatedInstance]:
    """过滤明显过矮或相对同类完整外观过窄的实例。"""
    if len(instances) < 2:
        return instances

    bbox_heights = [
        max(0.0, instance.bbox[3] - instance.bbox[1])
        for instance in instances
    ]
    tallest_height = max(bbox_heights)
    if tallest_height <= 0:
        return instances
    height_threshold = tallest_height * max(
        0.0,
        min(1.0, min_height_ratio_to_tallest),
    )
    aspect_ratios = [bbox_width_height_ratio(instance) for instance in instances]
    best_aspect_ratio = max(
        aspect_ratio
        for aspect_ratio, bbox_height in zip(aspect_ratios, bbox_heights)
        if bbox_height >= height_threshold
    )
    if best_aspect_ratio <= 0:
        return instances

    relative_threshold = max(0.0, min(1.0, min_ratio_to_best))
    minimum_aspect_ratio = best_aspect_ratio * relative_threshold
    filtered = [
        instance
        for instance, aspect_ratio, bbox_height in zip(
            instances,
            aspect_ratios,
            bbox_heights,
        )
        if bbox_height >= height_threshold
        and aspect_ratio >= minimum_aspect_ratio
    ]
    return filtered or instances


def select_pick_instance(
    instances: list[LocatedInstance],
    image_size: list[int],
) -> LocatedInstance:
    """Filter unusable candidates, then choose the instance nearest image center."""
    if not instances:
        raise HTTPException(status_code=404, detail="SAM3 没有找到目标商品实例")

    candidates = keep_visibly_complete_pick_candidates(instances)
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
        slot_id=selected_instance.mapped_slot_id,
        bbox=normalize_bbox_to_1_1000(
            selected_instance.bbox,
            debug_response.image_size,
        ),
        mask=selected_instance.mask,
        image_path=debug_response.image_path,
    )


@router.post("/perception/pick/locate", response_model=LocateResponse)
def locate_product(request: LocateRequest) -> LocateResponse:
    endpoint = "/perception/pick/locate"
    request_summary = locate_request_log_summary(request)
    started_at = time.perf_counter()
    logger.info(
        "Pick Locate request received endpoint=%s input=%s",
        endpoint,
        json.dumps(request_summary, ensure_ascii=False),
    )
    try:
        debug_response = locate_product_debug(request)
        response = make_locate_response(debug_response)
    except HTTPException as error:
        logger.warning(
            "Pick Locate request failed endpoint=%s duration_ms=%.1f "
            "status_code=%s detail=%s",
            endpoint,
            (time.perf_counter() - started_at) * 1000,
            error.status_code,
            error.detail,
        )
        raise
    except Exception:
        logger.exception(
            "Pick Locate request crashed endpoint=%s duration_ms=%.1f",
            endpoint,
            (time.perf_counter() - started_at) * 1000,
        )
        raise
    logger.info(
        "Pick Locate request succeeded endpoint=%s duration_ms=%.1f result=%s",
        endpoint,
        (time.perf_counter() - started_at) * 1000,
        json.dumps(
            {
                **locate_debug_response_log_summary(debug_response),
                "response_slot_id": response.slot_id,
                "response_bbox": response.bbox,
                "image_path": response.image_path,
            },
            ensure_ascii=False,
        ),
    )
    return response


@router.post("/perception/pick/locate/debug", response_model=LocateDebugResponse)
def locate_product_debug_api(request: LocateDebugRequest) -> LocateDebugResponse:
    endpoint = "/perception/pick/locate/debug"
    request_summary = locate_request_log_summary(request)
    started_at = time.perf_counter()
    logger.info(
        "Pick Locate request received endpoint=%s input=%s",
        endpoint,
        json.dumps(request_summary, ensure_ascii=False),
    )
    try:
        response = locate_product_debug(
            request,
            capture_inference_errors=True,
            allow_prompt_overrides=True,
            mock_inventory=request.mock_inventory,
        )
    except HTTPException as error:
        logger.warning(
            "Pick Locate request failed endpoint=%s duration_ms=%.1f "
            "status_code=%s detail=%s",
            endpoint,
            (time.perf_counter() - started_at) * 1000,
            error.status_code,
            error.detail,
        )
        raise
    except Exception:
        logger.exception(
            "Pick Locate request crashed endpoint=%s duration_ms=%.1f",
            endpoint,
            (time.perf_counter() - started_at) * 1000,
        )
        raise
    logger.info(
        "Pick Locate request completed endpoint=%s duration_ms=%.1f result=%s",
        endpoint,
        (time.perf_counter() - started_at) * 1000,
        json.dumps(locate_debug_response_log_summary(response), ensure_ascii=False),
    )
    return response


app.include_router(router)
