import base64
import binascii
import hashlib
import importlib.util
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from types import ModuleType
from urllib.parse import quote

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
PERCEPTION_ROOT = ROOT.parent
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))
from config import QWEN3_MODEL, QWEN3_URL, SAM3_URL, SERVICE_BIND_HOST  # noqa: E402
from initial_scan import InitialScanError, initial_scan_root, load_initial_scan  # noqa: E402
from place.locate.reference_mask import (  # noqa: E402
    ReferenceMaskError,
    generate_reference_mask,
)
from row_detection import RowDetectionConfig, detect_rows  # noqa: E402

LOCATE_ROOT = PERCEPTION_ROOT / "pick" / "locate"
INSPECT_ROOT = PERCEPTION_ROOT / "inspect"
DATA_ROOT = PERCEPTION_ROOT / "test_data"
STATIC_DIR = ROOT / "static"
RGB_DIR = DATA_ROOT / "2026-08-04"
SORTING_BATCH_ROOT = DATA_ROOT / "2026-08-13"
SORTING_BATCH_RESULTS_PATH = SORTING_BATCH_ROOT / "sorting_pick_locate_batch_results.json"
SORTING_BATCH_RESULTS_FILENAME = "sorting_pick_locate_batch_results.json"
SORTING_BATCH_MAPPING_FILENAME = "sorting_pick_locate_batch.json"
SORTING_BATCH_RUNNER_PATH = LOCATE_ROOT / "test" / "batch_record_inference.py"
SORTING_BATCH_RERUN_LOG_PATH = SORTING_BATCH_ROOT / "sorting_pick_locate_batch_rerun.log"
SORTING_BATCH_PROCESS: subprocess.Popen | None = None
SORTING_BATCH_PROCESS_LOCK = threading.Lock()
SORTING_BATCH_PROCESS_STARTED_AT: float | None = None
SORTING_BATCH_PROCESS_TARGET: dict[str, str] | None = None
SKU_CATALOG_PATH = PERCEPTION_ROOT / "sku" / "products.json"
PROMPT_PAIR_MAPPING_PATH = LOCATE_ROOT / "qwen_sam_prompt_mapping.json"
HARD_CASE_PROMPT_MAPPING_PATH = LOCATE_ROOT / "qwen_sam_prompt_mapping_hard_case.json"
HARD_CASE_SCOPE_PATH = PERCEPTION_ROOT / "hard_case_config.json"
SUPPORTED_TASK_TYPES = ("SORTING", "SHORTAGE", "MISPLACED")
PROMPT_PAIR_MAPPING_PATHS = {
    "SORTING": PROMPT_PAIR_MAPPING_PATH,
    "SHORTAGE": LOCATE_ROOT / "qwen_sam_prompt_mapping_shortage.json",
    "MISPLACED": LOCATE_ROOT / "qwen_sam_prompt_mapping_misplaced.json",
}

LOCATE_DEBUG_URL = os.getenv(
    "LOCATE_DEBUG_URL",
    "http://127.0.0.1:8083/perception/pick/locate/debug",
)
QWEN_SAMPLE_COUNT = 3
QWEN_TEMPERATURE = 0.7
QWEN_REQUEST_TIMEOUT_SECONDS = float(os.getenv("QWEN_REQUEST_TIMEOUT_SECONDS", "30"))
QWEN_REQUEST_MAX_ATTEMPTS = max(
    1, int(os.getenv("QWEN_REQUEST_MAX_ATTEMPTS", "3"))
)
QWEN_INFER_DATASETS = {
    "shortage": DATA_ROOT / "inspect_shortage_paired",
    "misplaced": DATA_ROOT / "inspect_misplaced_paired",
}
INITIAL_SCAN_ROW_RESULTS_ROOT = DATA_ROOT / "initial_scan_row_detection"
ROW_DETECTOR_SOURCE_PATH = PERCEPTION_ROOT / "row_detection" / "detector.py"
SHORTAGE_BATCH_ROOT = DATA_ROOT / "2026-08-16-self-collect-shortage-grouped"
SHORTAGE_BATCH_SUMMARY_PATH = (
    SHORTAGE_BATCH_ROOT / "shortage_inspection_batch_results.json"
)
REAL_SHORTAGE_BATCH_ROOT = DATA_ROOT / "real_shortage_regression"
SHORTAGE_BATCH_DATASETS = {
    "self_collect": {
        "label": "自采缺货批测",
        "root": SHORTAGE_BATCH_ROOT,
    },
    "real_shortage": {
        "label": "真实数据测试",
        "root": REAL_SHORTAGE_BATCH_ROOT,
    },
}
SHORTAGE_DEBUG_INDEX_CACHE_KEY: tuple[str, int] | None = None
SHORTAGE_DEBUG_INDEX_CACHE: dict[
    tuple[str, str, tuple[tuple[int, ...], ...]],
    tuple[dict[str, object], ...],
] = {}
INITIAL_SCAN_DIRECTORY_PATTERN = re.compile(
    r"^(?P<target>H[12]_[FB]_[LR]_INSPECT)_(?P<pose>UPPER|LOWER)$"
)


def load_inspect_api() -> ModuleType:
    """Load the real inspection entry point for full-chain web measurements."""

    module_name = "perception_inspect_test_web_api"
    existing = sys.modules.get(module_name)
    if isinstance(existing, ModuleType):
        return existing
    if str(INSPECT_ROOT) not in sys.path:
        sys.path.insert(0, str(INSPECT_ROOT))
    spec = importlib.util.spec_from_file_location(module_name, INSPECT_ROOT / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load inspection API from {INSPECT_ROOT / 'main.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


INSPECT_API = load_inspect_api()

app = FastAPI(title="Qwen3 / SAM3 Prompt Test Web")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PromptRequest(BaseModel):
    image_name: str
    prompt: str


class DirectQwenRequest(BaseModel):
    prompt: str
    image_base64: str
    temperature: float = Field(default=0.5, ge=0, le=2)


class SavedQwenInferRequest(BaseModel):
    dataset: str
    pair_number: int = Field(ge=1)
    region_index: int = Field(ge=1)
    stage: str | None = None
    prompt: str
    temperature: float = Field(default=0.0, ge=0, le=2)


class ShortageBatchQwenRetryRequest(BaseModel):
    dataset: str = "self_collect"
    group: str
    record: str
    region_index: int = Field(ge=1)
    prompt: str
    temperature: float = Field(default=0.0, ge=0, le=2)


class ShortageReferenceMaskRequest(BaseModel):
    dataset: str = "self_collect"
    group: str
    record: str
    region_index: int = Field(ge=1)


class FullInspectRunRequest(BaseModel):
    dataset: str
    pair_number: int = Field(ge=1)


class SaveSavedQwenPromptRequest(BaseModel):
    dataset: str
    pair_number: int = Field(ge=1)
    region_index: int = Field(ge=1)
    stage: str | None = None
    prompt: str


class SaveQwenPromptRequest(BaseModel):
    task_type: str
    sku_name: str
    prompt: str
    level: str | None = None
    hand: str | None = None


class SavePromptPairRequest(BaseModel):
    task_type: str
    sku_name: str
    qwen3_prompt: str
    sam3_prompt: str
    level: str | None = None
    hand: str | None = None


class LocateDebugProxyRequest(BaseModel):
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


class SortingBatchRerunRequest(BaseModel):
    record: str
    product_name: str
    result_file: str | None = None


class SamCropRequest(BaseModel):
    prompt: str
    image_base64: str
    crop_box_original: list[float]


class QwenDetection(BaseModel):
    name: str
    bbox: list[float]


class QwenSample(BaseModel):
    sample_index: int
    detections: list[QwenDetection] = Field(default_factory=list)
    raw_output: str = ""
    error: str | None = None


class QwenResponse(BaseModel):
    temperature: float
    samples: list[QwenSample]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/qwen-debug")
def qwen_debug_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "qwen_debug.html")


@app.get("/qwen-review")
@app.get("/qwen-infer")
def qwen_review_page() -> FileResponse:
    """Serve the canonical inspection review page; keep the old URL as an alias."""

    return FileResponse(STATIC_DIR / "qwen_review.html")


def resolve_initial_scan_for_web(scan_name: str) -> Path:
    """Resolve one task0 scan without allowing paths outside the scan root."""

    normalized = scan_name.strip().upper()
    if INITIAL_SCAN_DIRECTORY_PATTERN.fullmatch(normalized) is None:
        raise HTTPException(status_code=400, detail="初始扫描目录名称不合法")
    root = initial_scan_root().resolve()
    directory = (root / normalized).resolve()
    if directory.parent != root or not directory.is_dir():
        raise HTTPException(status_code=404, detail="初始扫描目录不存在")
    return directory


def build_initial_scan_row_sample(scan_directory: Path) -> dict:
    """Run or reuse row detection and return one browser-ready scan record."""

    match = INITIAL_SCAN_DIRECTORY_PATTERN.fullmatch(scan_directory.name)
    if match is None:
        raise HTTPException(status_code=400, detail="初始扫描目录名称不合法")
    source_path = scan_directory / "rgb.jpg"
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail=f"缺少初始扫描 RGB: {source_path}")

    pose_type = f"SHELF_VIEW_{match.group('pose')}"
    output_directory = INITIAL_SCAN_ROW_RESULTS_ROOT / scan_directory.name / "rgb"
    result_path = output_directory / "result.json"
    overlay_path = output_directory / "04_rows_and_rails.jpg"
    dependency_mtime_ns = max(
        source_path.stat().st_mtime_ns,
        ROW_DETECTOR_SOURCE_PATH.stat().st_mtime_ns,
    )
    cache_is_current = (
        result_path.is_file()
        and overlay_path.is_file()
        and min(result_path.stat().st_mtime_ns, overlay_path.stat().st_mtime_ns)
        >= dependency_mtime_ns
    )
    result_data: dict | None = None
    if cache_is_current:
        result_data = load_json_file(result_path, "初始扫描 row_detection 结果")
        cache_is_current = result_data.get("pose_type") == pose_type
    if not cache_is_current:
        detection = detect_rows(
            source_path,
            RowDetectionConfig(target_size=None, pose_type=pose_type),
        )
        detection.save_debug(output_directory)
        result_data = load_json_file(result_path, "初始扫描 row_detection 结果")
    assert result_data is not None

    version = max(source_path.stat().st_mtime_ns, overlay_path.stat().st_mtime_ns)
    rails = result_data.get("rails", [])
    rows = result_data.get("rows", [])
    return {
        "scan_name": scan_directory.name,
        "inspection_target_id": match.group("target"),
        "pose_type": pose_type,
        "image_size": result_data.get("image_size", []),
        "rail_count": len(rails),
        "row_count": len(rows),
        "rails": rails,
        "rows": rows,
        "source_url": (
            f"/api/qwen-review/initial-scan/{scan_directory.name}/source?v={version}"
        ),
        "overlay_url": (
            f"/api/qwen-review/initial-scan/{scan_directory.name}/overlay?v={version}"
        ),
    }


@app.get("/api/qwen-review/initial-scans")
def list_initial_scan_rows() -> dict:
    root = initial_scan_root()
    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"初始扫描目录不存在: {root}")
    samples = [
        build_initial_scan_row_sample(directory)
        for directory in sorted(root.iterdir(), key=lambda path: path.name)
        if directory.is_dir()
        and INITIAL_SCAN_DIRECTORY_PATTERN.fullmatch(directory.name) is not None
    ]
    if not samples:
        raise HTTPException(status_code=404, detail="没有找到 Upper/Lower 初始扫描")
    return {"samples": samples}


@app.get("/api/qwen-review/initial-scan/{scan_name}/source")
def get_initial_scan_source(scan_name: str) -> FileResponse:
    path = resolve_initial_scan_for_web(scan_name) / "rgb.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="初始扫描 RGB 不存在")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/qwen-review/initial-scan/{scan_name}/overlay")
def get_initial_scan_row_overlay(scan_name: str) -> FileResponse:
    scan_directory = resolve_initial_scan_for_web(scan_name)
    build_initial_scan_row_sample(scan_directory)
    path = (
        INITIAL_SCAN_ROW_RESULTS_ROOT
        / scan_directory.name
        / "rgb"
        / "04_rows_and_rails.jpg"
    )
    return FileResponse(path, media_type="image/jpeg")


def resolve_shortage_batch_dataset(dataset: str) -> tuple[Path, Path, str]:
    """Resolve one review dataset without allowing arbitrary filesystem roots."""

    normalized = dataset.strip().lower()
    if normalized == "self_collect":
        # Keep these aliases so local tests and deployments can override the
        # long-running self-collect dataset independently.
        return SHORTAGE_BATCH_ROOT, SHORTAGE_BATCH_SUMMARY_PATH, "自采缺货批测"
    if normalized == "real_shortage":
        root = REAL_SHORTAGE_BATCH_ROOT
        return root, root / "shortage_inspection_batch_results.json", "真实数据测试"
    config = SHORTAGE_BATCH_DATASETS.get(normalized)
    if config is None:
        raise HTTPException(status_code=400, detail=f"不支持的 shortage 数据集: {dataset}")
    root = Path(config["root"])
    return root, root / "shortage_inspection_batch_results.json", str(config["label"])


def shortage_batch_file_url(
    relative_path: object,
    version: int,
    *,
    batch_root: Path | None = None,
    dataset: str = "self_collect",
) -> str | None:
    batch_root = SHORTAGE_BATCH_ROOT if batch_root is None else batch_root
    if not isinstance(relative_path, str) or not relative_path.strip():
        return None
    path = resolve_descendant(batch_root, relative_path)
    if not path.is_file():
        return None
    normalized = path.relative_to(batch_root.resolve()).as_posix()
    return (
        f"/api/qwen-review/shortage-batch/file/{quote(normalized, safe='/')}"
        f"?dataset={quote(dataset, safe='')}&v={version}"
    )


def shortage_batch_baseline_url(
    group: object,
    record: object = None,
    version: int = 0,
    *,
    batch_root: Path | None = None,
    dataset: str = "self_collect",
) -> str | None:
    """Return the fixed task0 RGB paired with a shortage collection group."""

    if not isinstance(group, str):
        return None
    batch_root = SHORTAGE_BATCH_ROOT if batch_root is None else batch_root
    if isinstance(record, str):
        local_baseline = batch_root / group / record / "baseline_rgb.jpg"
        if local_baseline.is_file():
            relative_path = local_baseline.relative_to(batch_root.resolve()).as_posix()
            return shortage_batch_file_url(
                relative_path,
                version or local_baseline.stat().st_mtime_ns,
                batch_root=batch_root,
                dataset=dataset,
            )
    try:
        baseline_path = resolve_initial_scan_for_web(group) / "rgb.jpg"
    except HTTPException:
        return None
    if not baseline_path.is_file():
        return None
    return (
        f"/api/qwen-review/initial-scan/{quote(group, safe='')}/source"
        f"?v={baseline_path.stat().st_mtime_ns}"
    )


def shortage_prompt_key(
    location_id: object,
    pose_type: object,
    bboxes: object,
) -> tuple[str, str, tuple[tuple[int, ...], ...]] | None:
    if not isinstance(location_id, str) or not isinstance(pose_type, str):
        return None
    if not isinstance(bboxes, list):
        return None
    try:
        normalized_bboxes = tuple(
            tuple(int(value) for value in bbox)
            for bbox in bboxes
            if isinstance(bbox, list)
        )
    except (TypeError, ValueError):
        return None
    if len(normalized_bboxes) != len(bboxes):
        return None
    return location_id, pose_type, normalized_bboxes


def load_shortage_debug_index(
    batch_root: Path | None = None,
) -> dict[
    tuple[str, str, tuple[tuple[int, ...], ...]],
    tuple[dict[str, object], ...],
]:
    """Index saved Qwen prompts and ordered image inputs for batch review."""

    batch_root = SHORTAGE_BATCH_ROOT if batch_root is None else batch_root
    debug_root = batch_root / "qwen_debug"
    if not debug_root.is_dir():
        return {}
    global SHORTAGE_DEBUG_INDEX_CACHE_KEY, SHORTAGE_DEBUG_INDEX_CACHE
    cache_key = (str(debug_root.resolve()), debug_root.stat().st_mtime_ns)
    if cache_key == SHORTAGE_DEBUG_INDEX_CACHE_KEY:
        return SHORTAGE_DEBUG_INDEX_CACHE
    debug_index: dict[
        tuple[str, str, tuple[tuple[int, ...], ...]],
        tuple[dict[str, object], ...],
    ] = {}
    for request_path in sorted(debug_root.glob("*/request.json")):
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if request.get("task_type") != "SHORTAGE":
            continue
        bboxes = request.get("bboxes")
        key = shortage_prompt_key(
            request.get("location_id"),
            request.get("pose_type"),
            bboxes,
        )
        if key is None or not isinstance(bboxes, list):
            continue
        try:
            candidate_payload = json.loads(
                (request_path.parent / "candidates.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError):
            candidate_payload = {}
        all_candidates = candidate_payload.get("candidates", [])
        if not isinstance(all_candidates, list):
            all_candidates = []
        row_constraints = request.get("row_constraints", [])
        if not isinstance(row_constraints, list):
            row_constraints = []
        regions: list[dict[str, object]] = []
        for region_index in range(1, len(bboxes) + 1):
            region_directory = request_path.parent / f"region_{region_index:02d}"
            prompt_path = region_directory / "prompt.txt"
            try:
                prompt = prompt_path.read_text(encoding="utf-8")
            except OSError:
                prompt = None
            try:
                raw_output = (region_directory / "qwen_raw.txt").read_text(
                    encoding="utf-8"
                )
            except OSError:
                raw_output = None
            try:
                parsed_result = json.loads(
                    (region_directory / "parsed_result.json").read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, json.JSONDecodeError):
                parsed_result = None
            constraint = (
                row_constraints[region_index - 1]
                if region_index <= len(row_constraints)
                and isinstance(row_constraints[region_index - 1], dict)
                else {}
            )
            row_index = constraint.get("row_index")
            candidates = [
                dict(candidate)
                for candidate in all_candidates
                if isinstance(candidate, dict)
                and (
                    not isinstance(row_index, int)
                    or row_index in candidate.get("row_numbers", [])
                )
            ]
            exact_images = sorted(region_directory.glob("qwen_image_*.*"))
            regions.append(
                {
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "parsed_result": parsed_result,
                    "region_image": (
                        exact_images[0]
                        if exact_images
                        else region_directory / "bbox_expanded.jpg"
                    ),
                    "candidate_images": exact_images[1:],
                    "candidates": candidates,
                }
            )
        debug_index[key] = tuple(regions)
    SHORTAGE_DEBUG_INDEX_CACHE_KEY = cache_key
    SHORTAGE_DEBUG_INDEX_CACHE = debug_index
    return debug_index


def shortage_sku_image_url(sku_id: object) -> str | None:
    if not isinstance(sku_id, str) or not sku_id.strip():
        return None
    return (
        "/api/qwen-review/shortage-batch/sku-image/"
        f"{quote(sku_id.strip(), safe='')}"
        f"?v={SKU_CATALOG_PATH.stat().st_mtime_ns}"
    )


def resolve_shortage_sku_image(sku_id: str) -> Path:
    catalog = load_json_file(SKU_CATALOG_PATH, "SKU 商品目录")
    products = catalog.get("products", [])
    product = next(
        (
            item
            for item in products
            if isinstance(item, dict)
            and str(item.get("sku_id", "")).upper() == sku_id.upper()
        ),
        None,
    )
    images = product.get("images", []) if isinstance(product, dict) else []
    if not images or not isinstance(images[0], str):
        raise HTTPException(status_code=404, detail="SKU 候选图片不存在")
    logical_path = PureWindowsPath(images[0].replace("/", "\\"))
    if logical_path.is_absolute() or ".." in logical_path.parts:
        raise HTTPException(status_code=500, detail="SKU 候选图片路径不合法")
    relative_parts = list(logical_path.parts)
    if not relative_parts or relative_parts[0].lower() != "images":
        raise HTTPException(status_code=500, detail="SKU 候选图片路径不属于 images")
    filename = Path(*relative_parts[1:]).with_suffix(".jpg")
    images_root = (SKU_CATALOG_PATH.parent / "images_new").resolve()
    image_path = (images_root / filename).resolve()
    if not image_path.is_relative_to(images_root) or not image_path.is_file():
        raise HTTPException(status_code=404, detail="SKU 候选图片文件不存在")
    return image_path


def shortage_qwen_images(
    debug_region: dict[str, object],
    version: int,
    *,
    batch_root: Path | None = None,
    dataset: str = "self_collect",
) -> list[dict[str, object]]:
    batch_root = SHORTAGE_BATCH_ROOT if batch_root is None else batch_root
    images: list[dict[str, object]] = []
    region_image = debug_region.get("region_image")
    if isinstance(region_image, Path) and region_image.is_file():
        relative_path = region_image.relative_to(batch_root.resolve())
        images.append(
            {
                "image_index": 1,
                "kind": "reference_region",
                "label": "原始放置局部图",
                "description": "原始扫描图中的商品区域",
                "url": shortage_batch_file_url(
                    relative_path.as_posix(),
                    version,
                    batch_root=batch_root,
                    dataset=dataset,
                ),
            }
        )
    candidates = debug_region.get("candidates", [])
    exact_candidate_images = debug_region.get("candidate_images", [])
    if not isinstance(candidates, list):
        candidates = []
    if not isinstance(exact_candidate_images, list):
        exact_candidate_images = []
    valid_exact_images = [
        image
        for image in exact_candidate_images
        if isinstance(image, Path) and image.is_file()
    ]
    if valid_exact_images:
        for exact_index, exact_image in enumerate(valid_exact_images, start=1):
            relative_path = exact_image.relative_to(batch_root.resolve())
            images.append(
                {
                    "image_index": exact_index + 1,
                    "kind": "candidate_sheet",
                    "label": (
                        "候选 SKU 编号拼图"
                        if len(valid_exact_images) == 1
                        else f"候选 SKU 编号拼图 {exact_index}"
                    ),
                    "description": "",
                    "url": shortage_batch_file_url(
                        relative_path.as_posix(),
                        version,
                        batch_root=batch_root,
                        dataset=dataset,
                    ),
                }
            )
        return images
    for candidate_index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        image_url = shortage_sku_image_url(candidate.get("sku_id"))
        images.append(
            {
                "image_index": candidate_index + 1,
                "kind": "candidate",
                "label": candidate.get("name") or "未知候选",
                "description": candidate.get("sku_id") or "",
                "url": image_url,
            }
        )
    return images


def shortage_qwen_image_paths(debug_region: dict[str, object]) -> list[Path]:
    image_paths: list[Path] = []
    region_image = debug_region.get("region_image")
    if isinstance(region_image, Path) and region_image.is_file():
        image_paths.append(region_image)
    candidates = debug_region.get("candidates", [])
    exact_candidate_images = debug_region.get("candidate_images", [])
    if not isinstance(candidates, list):
        candidates = []
    if not isinstance(exact_candidate_images, list):
        exact_candidate_images = []
    valid_exact_images = [
        image
        for image in exact_candidate_images
        if isinstance(image, Path) and image.is_file()
    ]
    if valid_exact_images:
        image_paths.extend(valid_exact_images)
        return image_paths
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        exact_image = (
            exact_candidate_images[candidate_index]
            if candidate_index < len(exact_candidate_images)
            else None
        )
        if isinstance(exact_image, Path) and exact_image.is_file():
            image_paths.append(exact_image)
            continue
        sku_id = candidate.get("sku_id")
        if not isinstance(sku_id, str):
            raise HTTPException(status_code=500, detail="Qwen 候选缺少 sku_id")
        image_paths.append(resolve_shortage_sku_image(sku_id))
    return image_paths


def shortage_batch_result(
    group: str,
    record: str,
    dataset: str = "self_collect",
) -> dict:
    _, summary_path, _ = resolve_shortage_batch_dataset(dataset)
    if not summary_path.is_file():
        raise HTTPException(status_code=404, detail="shortage 批测汇总不存在")
    summary = load_json_file(summary_path, "shortage 批测汇总")
    result = next(
        (
            item
            for item in summary.get("results", [])
            if isinstance(item, dict)
            and item.get("group") == group
            and item.get("record") == record
        ),
        None,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="shortage 批测 record 不存在")
    return result


def shortage_batch_finding(result: dict, region_index: int) -> dict:
    finding = next(
        (
            item
            for item in result.get("findings", [])
            if isinstance(item, dict) and item.get("region_index") == region_index
        ),
        None,
    )
    if finding is None:
        raise HTTPException(status_code=404, detail="shortage finding 不存在")
    return finding


def decode_shortage_mask(
    relative_path: object,
    batch_root: Path | None = None,
) -> np.ndarray:
    if not isinstance(relative_path, str):
        raise HTTPException(status_code=422, detail="shortage finding 缺少 mask 文件")
    batch_root = SHORTAGE_BATCH_ROOT if batch_root is None else batch_root
    path = resolve_descendant(batch_root, relative_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="shortage finding mask 不存在")
    try:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"读取 shortage mask 失败: {error}") from error
    mask = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(status_code=500, detail="shortage finding mask 不是有效图片")
    return mask


def shortage_finding_row_bbox(
    result: dict,
    finding_bbox: list[float],
) -> list[float] | None:
    """Select the shelf row with the largest overlap with one reference bbox."""

    row_detection = result.get("row_detection")
    rows = row_detection.get("rows", []) if isinstance(row_detection, dict) else []
    if not isinstance(rows, list):
        return None
    x, y, width, height = finding_bbox
    right = x + width
    bottom = y + height
    candidates: list[tuple[float, list[float]]] = []
    for row in rows:
        row_bbox = row.get("bbox") if isinstance(row, dict) else None
        if not isinstance(row_bbox, list) or len(row_bbox) != 4:
            continue
        try:
            row_x, row_y, row_width, row_height = [float(value) for value in row_bbox]
        except (TypeError, ValueError):
            continue
        if row_width <= 0 or row_height <= 0:
            continue
        overlap_width = max(0.0, min(right, row_x + row_width) - max(x, row_x))
        overlap_height = max(0.0, min(bottom, row_y + row_height) - max(y, row_y))
        overlap_area = overlap_width * overlap_height
        if overlap_area > 0:
            candidates.append((overlap_area, [row_x, row_y, row_width, row_height]))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def encode_web_image(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", np.asarray(image))
    if not success:
        raise HTTPException(status_code=500, detail="reference mask 调试图编码失败")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def render_reference_mask_overlay(
    reference_image: np.ndarray,
    reference_mask: np.ndarray,
    reference_bbox: list[float],
    row_bbox: list[float] | None,
    crop_box: tuple[int, int, int, int],
    selected_bbox: tuple[float, float, float, float],
) -> np.ndarray:
    overlay = np.asarray(reference_image).copy()
    mask = np.asarray(reference_mask) > 0
    if np.any(mask):
        green = np.full((int(np.count_nonzero(mask)), 3), (40, 220, 80), dtype=np.uint8)
        overlay[mask] = cv2.addWeighted(overlay[mask], 0.45, green, 0.55, 0)

    def draw_xyxy(box: list[float] | tuple[float, ...], color: tuple[int, int, int], label: str) -> None:
        left, top, right, bottom = [int(round(value)) for value in box]
        cv2.rectangle(overlay, (left, top), (right, bottom), color, 3)
        cv2.putText(
            overlay,
            label,
            (max(0, left), max(18, top - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

    x, y, width, height = reference_bbox
    draw_xyxy((x, y, x + width, y + height), (0, 255, 255), "SHORTAGE BBOX")
    if row_bbox is not None:
        row_x, row_y, row_width, row_height = row_bbox
        draw_xyxy(
            (row_x, row_y, row_x + row_width, row_y + row_height),
            (255, 0, 255),
            "ROW",
        )
    draw_xyxy(crop_box, (255, 255, 0), "SAM CROP")
    draw_xyxy(selected_bbox, (0, 0, 255), "SAM INSTANCE")
    return overlay


def shortage_debug_region(
    result: dict,
    region_index: int,
    batch_root: Path | None = None,
) -> dict[str, object]:
    findings = [
        finding
        for finding in result.get("findings", [])
        if isinstance(finding, dict)
    ]
    if not any(finding.get("region_index") == region_index for finding in findings):
        raise HTTPException(status_code=404, detail="shortage finding 不存在")
    key = shortage_prompt_key(
        result.get("location_id"),
        result.get("pose_type"),
        [finding.get("bbox") for finding in findings],
    )
    regions = load_shortage_debug_index(batch_root).get(key, ())
    if not 0 < region_index <= len(regions):
        raise HTTPException(status_code=404, detail="找不到该 finding 的 Qwen 调试输入")
    return regions[region_index - 1]


def shortage_retry_result_path(
    result: dict,
    region_index: int,
    batch_root: Path | None = None,
) -> Path:
    group = result.get("group")
    record = result.get("record")
    if not isinstance(group, str) or not isinstance(record, str):
        raise HTTPException(status_code=500, detail="shortage 批测结果缺少 group/record")
    relative_path = (
        f"{group}/{record}/shortage_inspection/"
        f"qwen_retry_region_{region_index:02d}.json"
    )
    batch_root = SHORTAGE_BATCH_ROOT if batch_root is None else batch_root
    return resolve_descendant(batch_root, relative_path)


def build_shortage_retry_messages(
    prompt: str,
    image_paths: list[Path],
) -> tuple[str, list[dict]]:
    system_prompt, user_prompt = validate_saved_prompt(prompt)
    validate_saved_prompt_images(prompt, len(image_paths))
    content: list[dict] = []
    for part in re.split(r"(\[IMAGE\s+\d+\])", user_prompt):
        marker = re.fullmatch(r"\[IMAGE\s+(\d+)\]", part)
        if marker is None:
            if part:
                content.append({"type": "text", "text": part})
            continue
        path = image_paths[int(marker.group(1)) - 1]
        media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{encoded}"},
            }
        )
    return system_prompt, content


@app.get("/api/qwen-review/shortage-batch")
def list_shortage_batch_results(dataset: str = "self_collect") -> dict:
    batch_root, summary_path, dataset_label = resolve_shortage_batch_dataset(dataset)
    if not summary_path.is_file():
        return {
            "dataset": dataset,
            "dataset_label": dataset_label,
            "summary": {
                "total_records": 0,
                "completed_records": 0,
                "status_counts": {},
            },
            "samples": [],
        }
    summary = load_json_file(summary_path, "shortage 批测汇总")
    version = summary_path.stat().st_mtime_ns
    samples = []
    raw_results = summary.get("results", [])
    if not isinstance(raw_results, list):
        raise HTTPException(status_code=500, detail="shortage 批测汇总缺少 results")
    has_findings = any(
        isinstance(raw_result, dict)
        and any(
            isinstance(finding, dict)
            for finding in raw_result.get("findings", [])
        )
        for raw_result in raw_results
    )
    debug_index = load_shortage_debug_index(batch_root) if has_findings else {}
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue
        result = dict(raw_result)
        artifacts = result.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        findings = []
        fallback_key = shortage_prompt_key(
            result.get("location_id"),
            result.get("pose_type"),
            [
                finding.get("bbox")
                for finding in result.get("findings", [])
                if isinstance(finding, dict)
            ],
        )
        debug_regions = debug_index.get(fallback_key, ())
        for raw_finding in result.get("findings", []):
            if not isinstance(raw_finding, dict):
                continue
            finding = dict(raw_finding)
            finding["mask_url"] = shortage_batch_file_url(
                finding.get("mask"),
                version,
                batch_root=batch_root,
                dataset=dataset,
            )
            region_index = finding.get("region_index")
            debug_region = (
                debug_regions[region_index - 1]
                if isinstance(region_index, int)
                and 0 < region_index <= len(debug_regions)
                else None
            )
            if (
                not finding.get("qwen_prompt")
                and isinstance(debug_region, dict)
                and isinstance(debug_region.get("prompt"), str)
            ):
                finding["qwen_prompt"] = debug_region["prompt"]
            finding["qwen_images"] = (
                shortage_qwen_images(
                    debug_region,
                    version,
                    batch_root=batch_root,
                    dataset=dataset,
                )
                if isinstance(debug_region, dict)
                else []
            )
            finding["qwen_original_raw_output"] = (
                debug_region.get("raw_output")
                if isinstance(debug_region, dict)
                else None
            )
            finding["qwen_original_parsed_result"] = (
                debug_region.get("parsed_result")
                if isinstance(debug_region, dict)
                else None
            )
            retry_path = (
                shortage_retry_result_path(result, region_index, batch_root)
                if isinstance(region_index, int)
                else None
            )
            finding["last_qwen_retry"] = (
                load_json_file(retry_path, "Qwen 重试结果")
                if retry_path is not None and retry_path.is_file()
                else None
            )
            findings.append(finding)
        result["findings"] = findings
        result["source_rgb_url"] = shortage_batch_file_url(
            result.get("source_rgb"),
            version,
            batch_root=batch_root,
            dataset=dataset,
        )
        result["baseline_rgb_url"] = shortage_batch_baseline_url(
            result.get("group"),
            result.get("record"),
            version,
            batch_root=batch_root,
            dataset=dataset,
        )
        result["aligned_current_url"] = shortage_batch_file_url(
            artifacts.get("aligned_current"),
            version,
            batch_root=batch_root,
            dataset=dataset,
        )
        result["combined_mask_url"] = shortage_batch_file_url(
            artifacts.get("combined_mask"),
            version,
            batch_root=batch_root,
            dataset=dataset,
        )
        result["overlay_url"] = shortage_batch_file_url(
            artifacts.get("overlay"),
            version,
            batch_root=batch_root,
            dataset=dataset,
        )
        result["row_detection_url"] = shortage_batch_file_url(
            artifacts.get("row_detection"),
            version,
            batch_root=batch_root,
            dataset=dataset,
        )
        result["dataset"] = dataset
        samples.append(result)
    samples.sort(key=lambda item: (str(item.get("group")), str(item.get("record"))))
    return {
        "dataset": dataset,
        "dataset_label": dataset_label,
        "summary": {
            key: summary.get(key)
            for key in (
                "generated_at",
                "detection_only",
                "total_records",
                "completed_records",
                "status_counts",
            )
        },
        "samples": samples,
    }


@app.post("/api/qwen-review/shortage-batch/retry")
def retry_shortage_batch_qwen(request: ShortageBatchQwenRetryRequest) -> dict:
    request_started_at = time.perf_counter()
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt 不能为空")
    batch_root, _, _ = resolve_shortage_batch_dataset(request.dataset)
    result = shortage_batch_result(request.group, request.record, request.dataset)
    debug_region = shortage_debug_region(result, request.region_index, batch_root)
    image_paths = shortage_qwen_image_paths(debug_region)
    if not image_paths:
        raise HTTPException(status_code=404, detail="找不到该 finding 的 Qwen 输入图片")
    system_prompt, user_content = build_shortage_retry_messages(
        prompt,
        image_paths,
    )
    qwen_started_at = time.perf_counter()
    try:
        raw_output = call_qwen_messages(
            system_prompt,
            user_content,
            temperature=request.temperature,
        )
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"Qwen3 请求失败: {error}") from error
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise HTTPException(status_code=502, detail=f"Qwen3 响应格式错误: {error}") from error
    qwen_finished_at = time.perf_counter()
    try:
        parsed_result = parse_first_json(raw_output)
        parse_error = None
    except (TypeError, ValueError) as error:
        parsed_result = None
        parse_error = str(error)
    response_ready_at = time.perf_counter()
    retry_result = {
        "dataset": request.dataset,
        "group": request.group,
        "record": request.record,
        "region_index": request.region_index,
        "temperature": request.temperature,
        "prompt_used": prompt,
        "image_count": len(image_paths),
        "parsed_result": parsed_result,
        "raw_output": raw_output,
        "parse_error": parse_error,
        "qwen_elapsed_ms": round(
            (qwen_finished_at - qwen_started_at) * 1000,
            1,
        ),
        "backend_elapsed_ms": round(
            (response_ready_at - request_started_at) * 1000,
            1,
        ),
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(
        shortage_retry_result_path(result, request.region_index, batch_root),
        retry_result,
        "Qwen shortage 重试结果",
    )
    return retry_result


@app.post("/api/qwen-review/shortage-batch/reference-mask")
def generate_shortage_reference_mask(
    request: ShortageReferenceMaskRequest,
) -> dict:
    """Generate a Task0-aligned product mask from one reviewed shortage finding."""

    started_at = time.perf_counter()
    batch_root, _, _ = resolve_shortage_batch_dataset(request.dataset)
    result = shortage_batch_result(request.group, request.record, request.dataset)
    finding = shortage_batch_finding(result, request.region_index)
    product_name = finding.get("product_name")
    if not isinstance(product_name, str) or not product_name.strip():
        raise HTTPException(
            status_code=422,
            detail="该 shortage finding 尚未识别出商品名称",
        )
    raw_bbox = finding.get("bbox")
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise HTTPException(status_code=422, detail="shortage finding bbox 无效")
    try:
        reference_bbox = [float(value) for value in raw_bbox]
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="shortage finding bbox 无效") from error
    if not np.isfinite(reference_bbox).all() or reference_bbox[2] <= 0 or reference_bbox[3] <= 0:
        raise HTTPException(status_code=422, detail="shortage finding bbox 无效")

    location_id = result.get("location_id")
    pose_type = result.get("pose_type")
    if not isinstance(location_id, str) or not isinstance(pose_type, str):
        raise HTTPException(
            status_code=422,
            detail="shortage 批测结果缺少 location_id 或 pose_type",
        )
    try:
        initial_scan = load_initial_scan(location_id, pose_type)
    except InitialScanError as error:
        raise HTTPException(status_code=422, detail=f"读取 Task0 完整图失败: {error}") from error

    component_mask = decode_shortage_mask(finding.get("mask"), batch_root)
    if component_mask.shape != initial_scan.rgb.shape[:2]:
        raise HTTPException(
            status_code=422,
            detail=(
                "shortage mask 与 Task0 RGB 尺寸不一致: "
                f"mask={component_mask.shape}, rgb={initial_scan.rgb.shape[:2]}"
            ),
        )
    row_bbox = shortage_finding_row_bbox(result, reference_bbox)
    try:
        generated = generate_reference_mask(
            initial_scan.rgb,
            reference_bbox,
            product_name.strip(),
            component_mask=component_mask,
            row_bbox=row_bbox,
        )
    except ReferenceMaskError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error

    binary_mask = np.where(generated.mask > 0, 255, 0).astype(np.uint8)
    overlay = render_reference_mask_overlay(
        initial_scan.rgb,
        binary_mask,
        reference_bbox,
        row_bbox,
        generated.crop_box,
        generated.selected_bbox,
    )
    return {
        "dataset": request.dataset,
        "group": request.group,
        "record": request.record,
        "region_index": request.region_index,
        "product_name": product_name.strip(),
        "inspection_target_id": initial_scan.inspection_target_id,
        "pose_type": initial_scan.pose_type,
        "reference_image_size": [
            int(initial_scan.rgb.shape[1]),
            int(initial_scan.rgb.shape[0]),
        ],
        "reference_bbox": reference_bbox,
        "row_bbox": row_bbox,
        "sam3_prompt": generated.sam_prompt,
        "crop_box": list(generated.crop_box),
        "selected_bbox": list(generated.selected_bbox),
        "selected_score": generated.selected_score,
        "candidate_count": generated.candidate_count,
        "mask_pixels": int(np.count_nonzero(binary_mask)),
        "reference_mask_data_url": encode_web_image(binary_mask),
        "reference_overlay_data_url": encode_web_image(overlay),
        "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
    }


@app.get("/api/qwen-review/shortage-batch/file/{relative_path:path}")
def get_shortage_batch_file(
    relative_path: str,
    dataset: str = "self_collect",
) -> FileResponse:
    batch_root, _, _ = resolve_shortage_batch_dataset(dataset)
    path = resolve_descendant(batch_root, relative_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="shortage 批测文件不存在")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@app.get("/api/qwen-review/shortage-batch/sku-image/{sku_id}")
def get_shortage_batch_sku_image(sku_id: str) -> FileResponse:
    image_path = resolve_shortage_sku_image(sku_id)
    return FileResponse(image_path, media_type="image/jpeg")


@app.get("/api/qwen-review/samples")
@app.get("/api/qwen-infer/samples")
def list_qwen_infer_samples() -> dict:
    samples = []
    for dataset, dataset_root in QWEN_INFER_DATASETS.items():
        prompt_root = dataset_root / "qwen_prompt_samples"
        for manifest_path in sorted(prompt_root.glob("pair_*/manifest.json")):
            manifest = load_json_file(manifest_path, "Qwen 样例 manifest")
            sample_version = manifest_path.stat().st_mtime_ns
            pair_number = int(manifest.get("pair_number", 0))
            candidate_images = add_candidate_urls(
                dataset,
                pair_number,
                manifest.get("candidate_images", []),
                sample_version,
            )
            regions = []
            for region in manifest.get("regions", []):
                if not isinstance(region, dict):
                    continue
                region_index = int(region.get("region_index", 0))
                region_root = manifest_path.parent / f"region_{region_index:02d}"
                stage_views = []
                for stage in prompt_stages_for_region(manifest, region):
                    (
                        generated_path,
                        _input_path,
                        override_path,
                        result_path,
                    ) = resolve_prompt_stage_paths(
                        manifest_path.parent,
                        region_root,
                        stage,
                    )
                    stage_candidate_images = add_candidate_urls(
                        dataset,
                        pair_number,
                        candidate_images_for_stage(stage),
                        sample_version,
                    )
                    stage_candidate_sheets = add_candidate_urls(
                        dataset,
                        pair_number,
                        candidate_sheets_for_stage(stage),
                        sample_version,
                    )
                    stage_reference_images = (
                        stage_candidate_sheets or stage_candidate_images
                    )
                    prompt_path = generated_path
                    prompt_source = "generated"
                    prompt_warning = None
                    if override_path.is_file():
                        override_prompt = read_text_file(
                            override_path,
                            "Qwen 样例 override Prompt",
                        )
                        if (
                            prompt_has_expected_images(
                                override_prompt,
                                len(stage_reference_images) + 1,
                            )
                            and prompt_has_expected_candidates(
                                override_prompt,
                                candidate_names_from_manifest(
                                    stage_candidate_images
                                ),
                            )
                        ):
                            prompt_path = override_path
                            prompt_source = "override"
                        else:
                            prompt_warning = (
                                "已保留旧 override，但其图片、候选名称或顺序与"
                                "当前审核阶段的实际输入不一致；本次使用重新生成的 Prompt。"
                            )
                    prompt_text = read_text_file(prompt_path, "Qwen 样例 Prompt")
                    saved_result = (
                        load_json_file(result_path, "Qwen 样例推理结果")
                        if result_path.is_file()
                        else None
                    )
                    current_signature = qwen_stage_input_signature(
                        prompt_text, pair_root=manifest_path.parent, stage=stage
                    )
                    saved_signature = (
                        saved_result.get("input_signature")
                        if isinstance(saved_result, dict)
                        else None
                    )
                    # New results are invalidated by actual prompt/image content,
                    # not by manifest regeneration time. Keep the mtime fallback
                    # only for legacy result files without a signature.
                    result_stale = bool(
                        saved_result
                        and (
                            saved_signature != current_signature
                            if saved_signature
                            else result_path.stat().st_mtime_ns < sample_version
                        )
                    )
                    last_result = saved_result if saved_result and not result_stale else None
                    stage_views.append(
                        {
                            **stage,
                            "prompt": prompt_text,
                            "prompt_source": prompt_source,
                            "prompt_warning": prompt_warning,
                            "result_stale": result_stale,
                            "input_image_url": qwen_infer_file_url(
                                dataset,
                                pair_number,
                                str(stage.get("prompt_image_1", "")),
                                sample_version,
                            ),
                            "candidate_images": stage_candidate_images,
                            "candidate_sheets": stage_candidate_sheets,
                            "last_result": last_result,
                        }
                    )
                primary_stage = stage_views[0]
                regions.append(
                    {
                        "region_index": region_index,
                        "bbox": region.get("bbox", []),
                        "prompt": primary_stage["prompt"],
                        "prompt_source": primary_stage["prompt_source"],
                        "prompt_warning": primary_stage["prompt_warning"],
                        "result_stale": primary_stage["result_stale"],
                        "expanded_image_url": primary_stage["input_image_url"],
                        "row_constraint": region.get("row_constraint"),
                        "candidate_count_before": region.get(
                            "candidate_count_before",
                            len(candidate_images),
                        ),
                        "candidate_count_after": primary_stage.get(
                            "candidate_count_after",
                            len(primary_stage["candidate_images"]),
                        ),
                        "expected_candidate_names": region.get(
                            "expected_candidate_names",
                            [],
                        ),
                        "candidate_images": primary_stage["candidate_images"],
                        "candidate_sheets": primary_stage["candidate_sheets"],
                        "last_result": primary_stage["last_result"],
                        "prompt_stages": stage_views,
                    }
                )
            samples.append(
                {
                    "dataset": dataset,
                    "task_type": manifest.get("task_type"),
                    "pair_number": pair_number,
                    "location_id": manifest.get("location_id"),
                    "pose_type": manifest.get("pose_type"),
                    "regions": regions,
                    "candidate_images": candidate_images,
                    "comparison": manifest.get("comparison", {}),
                    "row_detection": manifest.get("row_detection", {}),
                    "baseline_url": f"/api/qwen-review/source/{dataset}/{pair_number}/baseline",
                    "current_url": f"/api/qwen-review/source/{dataset}/{pair_number}/current",
                    "aligned_current_url": optional_qwen_artifact_url(
                        dataset,
                        pair_number,
                        manifest.get("aligned_current"),
                        sample_version,
                    ),
                    "row_overlay_url": optional_qwen_artifact_url(
                        dataset,
                        pair_number,
                        manifest.get("row_overlay"),
                        sample_version,
                    ),
                }
            )
    return {"samples": samples}


@app.get("/api/qwen-review/file/{dataset}/{pair_number}/{relative_path:path}")
@app.get("/api/qwen-infer/file/{dataset}/{pair_number}/{relative_path:path}")
def get_qwen_infer_file(
    dataset: str,
    pair_number: int,
    relative_path: str,
) -> FileResponse:
    pair_root, _ = load_qwen_sample(dataset, pair_number)
    path = resolve_descendant(pair_root, relative_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Qwen 样例文件不存在")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@app.get("/api/qwen-review/source/{dataset}/{pair_number}/{version}")
@app.get("/api/qwen-infer/source/{dataset}/{pair_number}/{version}")
def get_qwen_infer_source(
    dataset: str,
    pair_number: int,
    version: str,
) -> FileResponse:
    dataset_root = qwen_infer_dataset_root(dataset)
    suffix = {"baseline": 1, "current": 2}.get(version)
    if suffix is None:
        raise HTTPException(status_code=400, detail="version 只能是 baseline 或 current")
    path = dataset_root / f"{pair_number}_{suffix}.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="样例原图不存在")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/qwen-review/prompt")
@app.post("/api/qwen-infer/prompt")
def save_qwen_infer_prompt(request: SaveSavedQwenPromptRequest) -> dict:
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt 不能为空")
    manifest, region_root, region, pair_root = load_qwen_region(
        request.dataset,
        request.pair_number,
        request.region_index,
    )
    stage = select_prompt_stage(manifest, region, request.stage)
    _, _, override_path, _ = resolve_prompt_stage_paths(
        pair_root,
        region_root,
        stage,
    )
    validate_saved_prompt(prompt)
    candidate_images = candidate_images_for_stage(stage)
    reference_images = prompt_reference_images_for_stage(stage)
    validate_saved_prompt_images(prompt, len(reference_images) + 1)
    validate_saved_prompt_candidates(prompt, candidate_images)
    write_text_atomic(
        override_path,
        prompt.rstrip() + "\n",
        "Qwen 样例 Prompt",
    )
    return {
        "saved": True,
        "stage": stage.get("stage"),
        "prompt_source": "override",
        "path": str(override_path),
    }


@app.post("/api/qwen-review/run")
@app.post("/api/qwen-infer/run")
def run_saved_qwen_infer(request: SavedQwenInferRequest) -> dict:
    request_started_at = time.perf_counter()
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt 不能为空")
    manifest, region_root, region, pair_root = load_qwen_region(
        request.dataset,
        request.pair_number,
        request.region_index,
    )
    stage = select_prompt_stage(manifest, region, request.stage)
    _, _, _, result_path = resolve_prompt_stage_paths(
        pair_root,
        region_root,
        stage,
    )
    system_prompt, user_content = build_saved_qwen_messages(
        prompt,
        pair_root,
        stage,
        candidate_images_for_stage(stage),
    )
    qwen_started_at = time.perf_counter()
    try:
        raw_output = call_qwen_messages(
            system_prompt,
            user_content,
            temperature=request.temperature,
        )
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"Qwen3 请求失败: {error}") from error
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise HTTPException(status_code=502, detail=f"Qwen3 响应格式错误: {error}") from error
    qwen_finished_at = time.perf_counter()
    try:
        parsed_result = parse_first_json(raw_output)
        parse_error = None
    except (TypeError, ValueError) as error:
        parsed_result = None
        parse_error = str(error)
    response_ready_at = time.perf_counter()
    prepare_inputs_ms = round((qwen_started_at - request_started_at) * 1000, 1)
    qwen_elapsed_ms = round((qwen_finished_at - qwen_started_at) * 1000, 1)
    parse_result_ms = round((response_ready_at - qwen_finished_at) * 1000, 1)
    backend_elapsed_ms = round((response_ready_at - request_started_at) * 1000, 1)
    result = {
        "dataset": request.dataset,
        "task_type": manifest.get("task_type"),
        "pair_number": request.pair_number,
        "region_index": request.region_index,
        "stage": stage.get("stage"),
        "temperature": request.temperature,
        # Keep elapsed_ms for older saved results/frontends; it has always meant
        # the outbound Qwen request rather than the whole browser round trip.
        "elapsed_ms": qwen_elapsed_ms,
        "qwen_elapsed_ms": qwen_elapsed_ms,
        "backend_elapsed_ms": backend_elapsed_ms,
        "timings": {
            "prepare_inputs_ms": prepare_inputs_ms,
            "qwen_request_ms": qwen_elapsed_ms,
            "parse_result_ms": parse_result_ms,
            "backend_processing_ms": backend_elapsed_ms,
        },
        "created_at": datetime.now(UTC).isoformat(),
        "prompt_used": prompt,
        "input_signature": qwen_stage_input_signature(
            prompt, pair_root=pair_root, stage=stage
        ),
        "parsed_result": parsed_result,
        "raw_output": raw_output,
        "parse_error": parse_error,
    }
    write_json_atomic(
        result_path,
        result,
        "Qwen 样例推理结果",
    )
    return result


@app.post("/api/qwen-review/run-full")
def run_full_inspect(request: FullInspectRunRequest) -> dict:
    """Run baseline-to-product inspection for one bundled pair and time it."""

    request_started_at = time.perf_counter()
    _, manifest = load_qwen_sample(request.dataset, request.pair_number)
    task_type = manifest.get("task_type")
    location_id = manifest.get("location_id")
    pose_type = manifest.get("pose_type")
    baseline_relative = manifest.get("baseline")
    current_relative = manifest.get("current")
    if task_type not in {"SHORTAGE", "MISPLACED"}:
        raise HTTPException(status_code=500, detail="Qwen 样例 task_type 无效")
    if not isinstance(location_id, str) or not location_id.strip():
        raise HTTPException(status_code=500, detail="Qwen 样例缺少 location_id")
    if pose_type not in {"", "SHELF_VIEW_UPPER", "SHELF_VIEW_LOWER"}:
        raise HTTPException(status_code=500, detail="Qwen 样例 pose_type 无效")
    if not isinstance(baseline_relative, str) or not isinstance(current_relative, str):
        raise HTTPException(status_code=500, detail="Qwen 样例缺少 baseline/current")

    baseline_path = resolve_sample_data_path(baseline_relative)
    current_path = resolve_sample_data_path(current_relative)
    try:
        baseline_base64 = base64.b64encode(baseline_path.read_bytes()).decode("ascii")
        current_base64 = base64.b64encode(current_path.read_bytes()).decode("ascii")
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"读取巡检样例图失败: {error}") from error
    baseline = INSPECT_API.decode_image(baseline_base64, "baseline")
    current = INSPECT_API.decode_image(current_base64, "current")

    inspect_started_at = time.perf_counter()
    inspect_response = INSPECT_API.inspect_supplied_images(
        task_type=task_type,
        location_id=location_id,
        pose_type=pose_type,
        baseline=baseline,
        current=current,
    )
    inspect_finished_at = time.perf_counter()
    inspect_findings = (
        inspect_response
        if isinstance(inspect_response, list)
        else inspect_response.findings
    )
    serialized_findings = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        for item in inspect_findings
    ]
    serialized_response = {"findings": serialized_findings}
    response_ready_at = time.perf_counter()
    prepare_inputs_ms = round((inspect_started_at - request_started_at) * 1000, 1)
    inspect_elapsed_ms = round((inspect_finished_at - inspect_started_at) * 1000, 1)
    backend_elapsed_ms = round((response_ready_at - request_started_at) * 1000, 1)
    return {
        "dataset": request.dataset,
        "task_type": task_type,
        "pair_number": request.pair_number,
        "location_id": location_id,
        "pose_type": pose_type,
        "created_at": datetime.now(UTC).isoformat(),
        "backend_elapsed_ms": backend_elapsed_ms,
        "inspect_elapsed_ms": inspect_elapsed_ms,
        "timings": {
            "prepare_inputs_ms": prepare_inputs_ms,
            "inspect_pipeline_ms": inspect_elapsed_ms,
            "backend_processing_ms": backend_elapsed_ms,
        },
        "finding_count": len(serialized_findings),
        "result": serialized_response,
    }


@app.get("/api/images")
def list_images() -> dict:
    images = sorted(path.name for path in RGB_DIR.glob("*_rgb.*"))
    return {"images": images, "default": images[-1] if images else None}


def sorting_batch_result_id(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(DATA_ROOT.resolve()).as_posix()
    except ValueError:
        return None


def sorting_batch_context(
    result_file: str | None = None,
) -> tuple[Path, Path, Path, Path, str | None]:
    if result_file is None:
        return (
            SORTING_BATCH_ROOT,
            SORTING_BATCH_RESULTS_PATH,
            SORTING_BATCH_ROOT / SORTING_BATCH_MAPPING_FILENAME,
            SORTING_BATCH_RERUN_LOG_PATH,
            sorting_batch_result_id(SORTING_BATCH_RESULTS_PATH),
        )

    normalized = result_file.strip().replace("\\", "/")
    if not normalized or normalized.startswith("/"):
        raise HTTPException(status_code=400, detail="结果文件路径不合法")
    result_path = (DATA_ROOT / normalized).resolve()
    if (
        not result_path.is_relative_to(DATA_ROOT.resolve())
        or result_path.name != SORTING_BATCH_RESULTS_FILENAME
        or not result_path.is_file()
    ):
        raise HTTPException(status_code=404, detail="sorting 批测结果文件不存在")
    batch_root = result_path.parent
    return (
        batch_root,
        result_path,
        batch_root / SORTING_BATCH_MAPPING_FILENAME,
        batch_root / "sorting_pick_locate_batch_rerun.log",
        result_path.relative_to(DATA_ROOT.resolve()).as_posix(),
    )


def sorting_batch_mapping_files(mapping_path: Path) -> tuple[str, str]:
    rgb_file = "rgb.jpg"
    depth_file = "depth_mm.npy"
    if mapping_path.is_file():
        try:
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            mapping = None
        if isinstance(mapping, dict):
            candidate_rgb = mapping.get("rgb_file")
            candidate_depth = mapping.get("depth_file")
            if (
                isinstance(candidate_rgb, str)
                and candidate_rgb
                and Path(candidate_rgb).name == candidate_rgb
            ):
                rgb_file = candidate_rgb
            if (
                isinstance(candidate_depth, str)
                and candidate_depth
                and Path(candidate_depth).name == candidate_depth
            ):
                depth_file = candidate_depth
    return rgb_file, depth_file


@app.get("/api/sorting-batch-results/files")
def list_sorting_batch_result_files() -> dict:
    files: list[dict] = []
    for path in DATA_ROOT.glob(f"*/{SORTING_BATCH_RESULTS_FILENAME}"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(summary, dict) or not isinstance(summary.get("results"), list):
            continue
        result_id = sorting_batch_result_id(path)
        if result_id is None:
            continue
        files.append(
            {
                "id": result_id,
                "label": path.parent.name,
                "updated_at_unix": path.stat().st_mtime,
                "total_records": summary.get("total_records", 0),
                "total_detections": summary.get("total_detections", 0),
                "completed": summary.get("completed", 0),
                "successes": summary.get("successes", 0),
                "failures": summary.get("failures", 0),
            }
        )
    files.sort(key=lambda item: (item["label"], item["id"]))
    default = (
        max(files, key=lambda item: item["updated_at_unix"])["id"]
        if files
        else None
    )
    return {"files": files, "default": default}


def sorting_batch_record_directory(
    record: str,
    batch_root: Path | None = None,
) -> Path:
    normalized_record = record.strip()
    if (
        not normalized_record
        or normalized_record in {".", ".."}
        or Path(normalized_record).name != normalized_record
        or "/" in normalized_record
        or "\\" in normalized_record
    ):
        raise HTTPException(status_code=400, detail="record 名称不合法")
    root = (batch_root or SORTING_BATCH_ROOT).resolve()
    directory = (root / normalized_record).resolve()
    if not directory.is_relative_to(root):
        raise HTTPException(status_code=400, detail="record 路径不合法")
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="record 目录不存在")
    return directory


def sorting_batch_asset_url(
    record: str,
    kind: str,
    path: Path,
    product_name: str | None = None,
    result_file: str | None = None,
) -> str:
    version = path.stat().st_mtime_ns if path.is_file() else 0
    url = f"/api/sorting-batch-results/image/{quote(record, safe='')}/{kind}?v={version}"
    if product_name is not None:
        url += f"&product_name={quote(product_name, safe='')}"
    if result_file is not None:
        url += f"&result_file={quote(result_file, safe='')}"
    return url


def sorting_batch_prompt_usage(result_path: Path) -> tuple[str | None, str | None]:
    result_json_path = result_path.with_suffix(".json")
    if not result_json_path.is_file():
        return None, None
    try:
        result = json.loads(result_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(result, dict):
        return None, None
    response = result.get("response")
    if not isinstance(response, dict):
        return None, None
    qwen_prompt = response.get("qwen3_prompt_used")
    sam_prompt = response.get("sam3_prompt_used")
    return (
        qwen_prompt if isinstance(qwen_prompt, str) else None,
        sam_prompt if isinstance(sam_prompt, str) else None,
    )


@app.get("/api/sorting-batch-results")
def list_sorting_batch_results(result_file: str | None = None) -> dict:
    batch_root, results_path, mapping_path, _, result_id = sorting_batch_context(
        result_file
    )
    if not results_path.is_file():
        raise HTTPException(status_code=404, detail="尚未生成 sorting 批测结果")
    summary = load_json_file(results_path, "sorting 批测结果")
    raw_results = summary.get("results")
    if not isinstance(raw_results, list):
        raise HTTPException(status_code=500, detail="sorting 批测结果缺少 results")

    rgb_file, depth_file = sorting_batch_mapping_files(mapping_path)
    records_by_name: dict[str, dict] = {}
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue
        record = str(raw_result.get("record", "")).strip()
        product_name = str(raw_result.get("product_name", "")).strip()
        if not record or not product_name:
            continue
        record_directory = sorting_batch_record_directory(record, batch_root)
        rgb_path = record_directory / rgb_file
        depth_path = record_directory / depth_file
        result_path = record_directory / f"{product_name}.jpg"
        level = raw_result.get("level")
        hand = raw_result.get("hand")
        hard_case = is_hard_case_request(product_name, "SORTING", level, hand)
        mapping_path = prompt_mapping_path("SORTING", hard_case=hard_case)
        mapping = load_prompt_pair_mapping("SORTING", hard_case=hard_case)
        current_prompt_pair = mapping.get(product_name)
        qwen_prompt_used, sam_prompt_used = sorting_batch_prompt_usage(result_path)
        prompt_matches_current_mapping = None
        if current_prompt_pair is not None and qwen_prompt_used is not None:
            current_qwen_prompt = current_prompt_pair["qwen3_prompt"]
            qwen_matches = (
                qwen_prompt_used.startswith(current_qwen_prompt)
                if hard_case
                else qwen_prompt_used == current_qwen_prompt
            )
            prompt_matches_current_mapping = (
                qwen_matches
                and sam_prompt_used == current_prompt_pair["sam3_prompt"]
            )
        record_result = records_by_name.setdefault(
            record,
            {
                "record": record,
                "rgb_url": (
                    sorting_batch_asset_url(
                        record,
                        "rgb",
                        rgb_path,
                        result_file=result_id,
                    )
                    if rgb_path.is_file()
                    else None
                ),
                "depth_url": (
                    sorting_batch_asset_url(
                        record,
                        "depth",
                        depth_path,
                        result_file=result_id,
                    )
                    if depth_path.is_file()
                    else None
                ),
                "items": [],
            },
        )
        record_result["items"].append(
            {
                "product_name": product_name,
                "sku_id": raw_result.get("sku_id"),
                "level": level,
                "hand": hand,
                "status": raw_result.get("status", "error"),
                "selected_depth_mm": raw_result.get("selected_depth_mm"),
                "selected_bbox_pixel": raw_result.get("selected_bbox_pixel"),
                "elapsed_seconds": raw_result.get("elapsed_seconds"),
                "error": raw_result.get("error"),
                "qwen3_prompt_used": qwen_prompt_used,
                "sam3_prompt_used": sam_prompt_used,
                "prompt_mapping_file": mapping_path.name,
                "prompt_matches_current_mapping": prompt_matches_current_mapping,
                "result_url": (
                    sorting_batch_asset_url(
                        record,
                        "result",
                        result_path,
                        product_name=product_name,
                        result_file=result_id,
                    )
                    if result_path.is_file()
                    else None
                ),
            }
        )

    return {
        "result_file": result_id,
        "dataset": batch_root.name,
        "summary": {
            "total_records": summary.get("total_records", len(records_by_name)),
            "total_detections": summary.get("total_detections", len(raw_results)),
            "completed": summary.get("completed", len(raw_results)),
            "successes": summary.get("successes", 0),
            "failures": summary.get("failures", 0),
            "updated_at_unix": summary.get("updated_at_unix"),
        },
        "records": list(records_by_name.values()),
    }


def sorting_batch_progress(result_file: str | None = None) -> dict:
    _, results_path, _, _, _ = sorting_batch_context(result_file)
    if not results_path.is_file():
        return {
            "completed": 0,
            "total_detections": 0,
            "successes": 0,
            "failures": 0,
        }
    try:
        summary = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "completed": 0,
            "total_detections": 0,
            "successes": 0,
            "failures": 0,
        }
    return {
        key: summary.get(key, 0)
        for key in ("completed", "total_detections", "successes", "failures")
    }


def sorting_batch_last_log_line(result_file: str | None = None) -> str | None:
    _, _, _, log_path, _ = sorting_batch_context(result_file)
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    return lines[-1] if lines else None


def sorting_batch_rerun_status(result_file: str | None = None) -> dict:
    process = SORTING_BATCH_PROCESS
    exit_code = process.poll() if process is not None else None
    running = process is not None and exit_code is None
    if running:
        state = "running"
    elif process is None:
        state = "idle"
    elif exit_code == 0:
        state = "succeeded"
    else:
        state = "failed"
    active_result_file = (
        SORTING_BATCH_PROCESS_TARGET.get("result_file")
        if SORTING_BATCH_PROCESS_TARGET is not None
        else None
    )
    selected_result_file = active_result_file or result_file
    progress = sorting_batch_progress(selected_result_file)
    if SORTING_BATCH_PROCESS_TARGET is not None:
        progress = {
            "completed": 0 if running else 1,
            "total_detections": 1,
            "successes": 1 if state == "succeeded" else 0,
            "failures": 1 if state == "failed" else 0,
        }
    return {
        "state": state,
        "running": running,
        "pid": process.pid if process is not None else None,
        "exit_code": exit_code,
        "started_at_unix": SORTING_BATCH_PROCESS_STARTED_AT,
        "target": SORTING_BATCH_PROCESS_TARGET,
        "progress": progress,
        "result_file": selected_result_file,
        "last_log_line": sorting_batch_last_log_line(selected_result_file),
    }


@app.get("/api/sorting-batch-results/rerun")
def get_sorting_batch_rerun_status(result_file: str | None = None) -> dict:
    return sorting_batch_rerun_status(result_file)


@app.post("/api/sorting-batch-results/rerun", status_code=202)
def start_sorting_batch_rerun(request: SortingBatchRerunRequest) -> dict:
    global SORTING_BATCH_PROCESS, SORTING_BATCH_PROCESS_STARTED_AT
    global SORTING_BATCH_PROCESS_TARGET
    with SORTING_BATCH_PROCESS_LOCK:
        batch_root, _, mapping_path, log_path, result_id = sorting_batch_context(
            request.result_file
        )
        if (
            SORTING_BATCH_PROCESS is not None
            and SORTING_BATCH_PROCESS.poll() is None
        ):
            return sorting_batch_rerun_status(result_id)
        if not SORTING_BATCH_RUNNER_PATH.is_file():
            raise HTTPException(status_code=500, detail="sorting 批测脚本不存在")
        record = request.record.strip()
        product_name = request.product_name.strip()
        record_directory = sorting_batch_record_directory(record, batch_root)
        if not product_name or "/" in product_name or "\\" in product_name:
            raise HTTPException(status_code=400, detail="商品名称不合法")
        if not (record_directory / f"{product_name}.json").is_file():
            raise HTTPException(status_code=404, detail="当前单项批测结果不存在")
        batch_root.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        command = [
            sys.executable,
            str(SORTING_BATCH_RUNNER_PATH),
        ]
        if mapping_path.is_file():
            command.extend(["--mapping", str(mapping_path)])
        command.extend(
            [
                "--overwrite",
                "--record",
                record,
                "--product-name",
                product_name,
            ]
        )
        try:
            with log_path.open(
                "w",
                encoding="utf-8",
            ) as log_file:
                SORTING_BATCH_PROCESS = subprocess.Popen(
                    command,
                    cwd=str(PERCEPTION_ROOT),
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except OSError as error:
            raise HTTPException(
                status_code=500,
                detail=f"启动 sorting 批测失败: {error}",
            ) from error
        SORTING_BATCH_PROCESS_STARTED_AT = time.time()
        SORTING_BATCH_PROCESS_TARGET = {
            "record": record,
            "product_name": product_name,
            "result_file": result_id or "",
        }
        return sorting_batch_rerun_status(result_id)


def render_depth_preview(depth_path: Path) -> bytes:
    try:
        if depth_path.suffix.lower() == ".npy":
            depth = np.load(depth_path, allow_pickle=False)
        else:
            encoded_depth = np.frombuffer(depth_path.read_bytes(), dtype=np.uint8)
            depth = cv2.imdecode(encoded_depth, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError("OpenCV 无法解码深度图片")
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail=f"读取深度数据失败: {error}") from error
    depth = np.asarray(depth).squeeze()
    if depth.ndim != 2:
        raise HTTPException(status_code=500, detail="深度数据必须是二维数组")
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise HTTPException(status_code=422, detail="深度数据没有有效像素")

    valid_values = depth[valid].astype(np.float32)
    near_mm, far_mm = np.percentile(valid_values, [2, 98])
    if far_mm <= near_mm:
        near_mm = float(valid_values.min())
        far_mm = float(valid_values.max())
    if far_mm <= near_mm:
        far_mm = near_mm + 1.0
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (depth[valid] - near_mm) * 255.0 / (far_mm - near_mm),
        0,
        255,
    ).astype(np.uint8)
    # Invert before TURBO so nearer pixels are warm and farther pixels are cool.
    preview = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    ok, encoded = cv2.imencode(".png", preview)
    if not ok:
        raise HTTPException(status_code=500, detail="深度预览编码失败")
    return encoded.tobytes()


@app.get("/api/sorting-batch-results/image/{record}/{kind}")
def get_sorting_batch_image(
    record: str,
    kind: str,
    product_name: str | None = None,
    result_file: str | None = None,
) -> Response:
    batch_root, _, mapping_path, _, _ = sorting_batch_context(result_file)
    rgb_file, depth_file = sorting_batch_mapping_files(mapping_path)
    record_directory = sorting_batch_record_directory(record, batch_root)
    if kind == "rgb":
        path = record_directory / rgb_file
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    elif kind == "depth":
        path = record_directory / depth_file
        if not path.is_file():
            raise HTTPException(status_code=404, detail="深度数据不存在")
        return Response(
            content=render_depth_preview(path),
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=3600"},
        )
    elif kind == "result":
        product_name = (product_name or "").strip()
        if not product_name or "/" in product_name or "\\" in product_name:
            raise HTTPException(status_code=400, detail="商品名称不合法")
        path = (record_directory / f"{product_name}.jpg").resolve()
        if not path.is_relative_to(record_directory):
            raise HTTPException(status_code=400, detail="结果图片路径不合法")
        media_type = "image/jpeg"
    else:
        raise HTTPException(status_code=400, detail="图片类型不合法")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="批测图片不存在")
    return FileResponse(path, media_type=media_type)


@app.get("/api/skus")
def list_skus(
    task_type: str = "SORTING",
    level: str | None = None,
    hand: str | None = None,
) -> dict:
    normalized_task_type = normalize_task_type(task_type)
    ordinary_mapping = load_prompt_pair_mapping(normalized_task_type)
    hard_mapping = load_prompt_pair_mapping(normalized_task_type, hard_case=True)
    skus = []
    for sku in load_skus():
        prompt_pair = (
            hard_mapping.get(sku["name"])
            if is_hard_case_request(sku["name"], normalized_task_type, level, hand)
            else ordinary_mapping.get(sku["name"])
        )
        skus.append(
            {
                "name": sku["name"],
                "qwen3_prompt": (
                    prompt_pair["qwen3_prompt"] if prompt_pair is not None else None
                ),
                "sam3_prompt": (
                    prompt_pair["sam3_prompt"] if prompt_pair is not None else None
                ),
            }
        )
    return {"task_type": normalized_task_type, "skus": skus}


@app.post("/api/qwen-prompts")
def save_qwen_prompt(request: SaveQwenPromptRequest) -> dict:
    task_type = normalize_task_type(request.task_type)
    sku_name = request.sku_name.strip()
    prompt = request.prompt.strip()
    if not sku_name:
        raise HTTPException(status_code=400, detail="SKU 不能为空")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt 不能为空")

    valid_names = {sku["name"] for sku in load_skus()}
    if sku_name not in valid_names:
        raise HTTPException(status_code=400, detail=f"商品库中不存在 SKU：{sku_name}")

    hard_case = is_hard_case_request(
        sku_name, task_type, request.level, request.hand
    )
    mapping = load_prompt_pair_mapping(task_type, hard_case=hard_case)
    current_pair = mapping.get(sku_name)
    if current_pair is None or not current_pair["sam3_prompt"].strip():
        raise HTTPException(
            status_code=409,
            detail="该 SKU 尚未保存 SAM3 Prompt，请填写右侧 Prompt 后保存完整配对",
        )

    overwritten = bool(current_pair["qwen3_prompt"].strip())
    mapping[sku_name] = {
        "qwen3_prompt": prompt,
        "sam3_prompt": current_pair["sam3_prompt"],
    }
    write_json_mapping(
        prompt_mapping_path(task_type, hard_case=hard_case),
        mapping,
        f"{task_type} 配对 Prompt",
    )
    return {
        "task_type": task_type,
        "sku_name": sku_name,
        "prompt": prompt,
        "qwen3_prompt": prompt,
        "sam3_prompt": current_pair["sam3_prompt"],
        "overwritten": overwritten,
    }


@app.post("/api/prompt-pairs")
def save_prompt_pair(request: SavePromptPairRequest) -> dict:
    task_type = normalize_task_type(request.task_type)
    sku_name = request.sku_name.strip()
    qwen3_prompt = request.qwen3_prompt.strip()
    sam3_prompt = request.sam3_prompt.strip()
    if not sku_name:
        raise HTTPException(status_code=400, detail="SKU 不能为空")
    if not qwen3_prompt:
        raise HTTPException(status_code=400, detail="左侧 Qwen3 Prompt 不能为空")
    if not sam3_prompt:
        raise HTTPException(status_code=400, detail="SAM3 Prompt 不能为空")

    valid_names = {sku["name"] for sku in load_skus()}
    if sku_name not in valid_names:
        raise HTTPException(status_code=400, detail=f"商品库中不存在 SKU：{sku_name}")

    hard_case = is_hard_case_request(
        sku_name, task_type, request.level, request.hand
    )
    mapping = load_prompt_pair_mapping(task_type, hard_case=hard_case)
    overwritten = sku_name in mapping
    mapping[sku_name] = {
        "qwen3_prompt": qwen3_prompt,
        "sam3_prompt": sam3_prompt,
    }
    write_json_mapping(
        prompt_mapping_path(task_type, hard_case=hard_case),
        mapping,
        f"{task_type} 配对 Prompt",
    )
    return {
        "task_type": task_type,
        "sku_name": sku_name,
        "qwen3_prompt": qwen3_prompt,
        "sam3_prompt": sam3_prompt,
        "overwritten": overwritten,
    }


@app.post("/api/locate-debug")
def run_locate_debug(request: LocateDebugProxyRequest) -> dict:
    payload = {
        "task_type": request.task_type.strip(),
        "product_name": request.product_name.strip(),
        "level": request.level.strip().upper(),
        "hand": request.hand.strip(),
    }
    if not all(payload.values()):
        raise HTTPException(
            status_code=400,
            detail="task_type、product_name、level、hand 都不能为空",
        )
    has_image_name = request.image_name is not None
    has_image_base64 = request.image_base64 is not None
    if has_image_name != has_image_base64:
        raise HTTPException(
            status_code=400,
            detail="image_name 和 image_base64 必须同时提供或同时省略",
        )
    if has_image_name and has_image_base64:
        payload["image_name"] = request.image_name.strip()
        payload["image_base64"] = request.image_base64
        if not payload["image_name"] or not payload["image_base64"]:
            raise HTTPException(status_code=400, detail="离线图片名称和内容不能为空")
    has_depth_name = request.depth_image_name is not None
    has_depth_base64 = request.depth_image_base64 is not None
    if has_depth_name != has_depth_base64:
        raise HTTPException(
            status_code=400,
            detail="depth_image_name 和 depth_image_base64 必须同时提供或同时省略",
        )
    if has_depth_name and not has_image_name:
        raise HTTPException(status_code=400, detail="离线深度数据必须与离线 RGB 图片同时提供")
    if has_depth_name and has_depth_base64:
        payload["depth_image_name"] = request.depth_image_name.strip()
        payload["depth_image_base64"] = request.depth_image_base64
        payload["depth_is_bigendian"] = request.depth_is_bigendian
        if not payload["depth_image_name"] or not payload["depth_image_base64"]:
            raise HTTPException(status_code=400, detail="离线深度数据名称和内容不能为空")
    if request.qwen3_prompt is not None:
        payload["qwen3_prompt"] = request.qwen3_prompt
    if request.sam3_prompt is not None:
        payload["sam3_prompt"] = request.sam3_prompt
    try:
        response = requests.post(
            LOCATE_DEBUG_URL,
            json=payload,
            timeout=600,
        )
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"Locate Debug 请求失败: {error}") from error
    try:
        result = response.json()
    except ValueError as error:
        raise HTTPException(status_code=502, detail="Locate Debug 响应不是有效 JSON") from error
    if not response.ok:
        detail = result.get("detail", result) if isinstance(result, dict) else result
        raise HTTPException(status_code=response.status_code, detail=detail)
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("image_base64"), str)
        or not isinstance(result.get("qwen_bboxes"), list)
        or not isinstance(result.get("instances"), list)
    ):
        raise HTTPException(status_code=502, detail="Locate Debug 响应缺少原图或检测结果")
    return result


@app.get("/api/image/{image_name}")
def get_image(image_name: str) -> FileResponse:
    image_path = resolve_image(image_name)
    media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    return FileResponse(image_path, media_type=media_type)


@app.post("/api/qwen", response_model=QwenResponse)
def run_qwen(request: PromptRequest) -> QwenResponse:
    image_path = resolve_image(request.image_name)
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Qwen prompt 不能为空")

    media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    samples = []
    for sample_index in range(1, QWEN_SAMPLE_COUNT + 1):
        content = ""
        try:
            content = call_qwen(prompt, media_type, image_base64)
            detections = parse_qwen_json(content)
            samples.append(
                QwenSample(
                    sample_index=sample_index,
                    detections=detections,
                    raw_output=content,
                )
            )
        except requests.RequestException as error:
            samples.append(
                QwenSample(
                    sample_index=sample_index,
                    error=f"Qwen 请求失败: {error}",
                )
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            samples.append(
                QwenSample(
                    sample_index=sample_index,
                    raw_output=content,
                    error=f"Qwen 输出处理失败: {error}",
                )
            )

    return QwenResponse(temperature=QWEN_TEMPERATURE, samples=samples)


@app.post("/api/qwen-direct")
def run_qwen_direct(request: DirectQwenRequest) -> dict:
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Qwen Prompt 不能为空")

    encoded = request.image_base64.split(",", 1)[-1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise HTTPException(status_code=400, detail="粘贴的图片不是有效 base64") from error
    if not image_bytes:
        raise HTTPException(status_code=400, detail="请先粘贴图片")
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="图片不能超过 20 MB")

    header = request.image_base64.split(",", 1)[0]
    media_match = re.match(r"data:(image/(?:jpeg|png|webp));base64", header)
    media_type = media_match.group(1) if media_match else "image/jpeg"
    try:
        raw_output = call_qwen(
            prompt,
            media_type,
            encoded,
            temperature=request.temperature,
        )
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"Qwen3 请求失败: {error}") from error
    try:
        detections = parse_qwen_json(raw_output)
        parse_error = None
    except (TypeError, ValueError) as error:
        detections = []
        parse_error = str(error)
    return {
        "prompt_used": prompt,
        "temperature": request.temperature,
        "detections": detections,
        "raw_output": raw_output,
        "parse_error": parse_error,
    }


def call_qwen(
    prompt: str,
    media_type: str,
    image_base64: str,
    *,
    temperature: float = QWEN_TEMPERATURE,
) -> str:
    print(f"[Qwen3 Direct] prompt before request:\n{prompt}", flush=True)
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
            "temperature": temperature,
            "max_tokens": 1024,
        },
        timeout=120,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("choices[0].message.content 不是字符串")
    return content


def qwen_infer_dataset_root(dataset: str) -> Path:
    try:
        return QWEN_INFER_DATASETS[dataset.strip().lower()]
    except KeyError as error:
        raise HTTPException(
            status_code=400,
            detail=f"dataset 只能是: {', '.join(QWEN_INFER_DATASETS)}",
        ) from error


def load_qwen_sample(dataset: str, pair_number: int) -> tuple[Path, dict]:
    dataset_root = qwen_infer_dataset_root(dataset)
    pair_root = dataset_root / "qwen_prompt_samples" / f"pair_{pair_number}"
    manifest_path = pair_root / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Qwen pair 样例不存在")
    return pair_root, load_json_file(manifest_path, "Qwen 样例 manifest")


def load_qwen_region(
    dataset: str,
    pair_number: int,
    region_index: int,
) -> tuple[dict, Path, dict, Path]:
    pair_root, manifest = load_qwen_sample(dataset, pair_number)
    regions = manifest.get("regions")
    if not isinstance(regions, list):
        raise HTTPException(status_code=500, detail="Qwen 样例 manifest 缺少 regions")
    region = next(
        (
            item
            for item in regions
            if isinstance(item, dict) and item.get("region_index") == region_index
        ),
        None,
    )
    if region is None:
        raise HTTPException(status_code=404, detail="Qwen region 样例不存在")
    region_root = pair_root / f"region_{region_index:02d}"
    if not region_root.is_dir():
        raise HTTPException(status_code=404, detail="Qwen region 目录不存在")
    return manifest, region_root, region, pair_root


def resolve_descendant(root: Path, relative_path: str) -> Path:
    normalized = relative_path.strip().replace("\\", "/")
    if not normalized:
        raise HTTPException(status_code=400, detail="样例文件路径不能为空")
    relative = Path(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(status_code=400, detail="样例文件路径不合法")
    root = root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise HTTPException(status_code=400, detail="样例文件路径不合法")
    return resolved


def resolve_sample_data_path(stored_path: str) -> Path:
    """Resolve current relative paths and legacy absolute test_data paths safely."""

    normalized = stored_path.strip().replace("\\", "/")
    if not normalized:
        return resolve_descendant(DATA_ROOT, normalized)

    path = Path(normalized)
    if not path.is_absolute() and ".." not in path.parts:
        return resolve_descendant(DATA_ROOT, normalized)

    # Older manifests stored the generator machine's full Windows path.  Never
    # access that location directly: retain only the suffix below test_data and
    # anchor it to this checkout's DATA_ROOT.
    windows_parts = PureWindowsPath(stored_path).parts
    test_data_index = next(
        (
            index
            for index, part in enumerate(windows_parts)
            if part.casefold() == "test_data"
        ),
        None,
    )
    if test_data_index is None or test_data_index + 1 >= len(windows_parts):
        raise HTTPException(status_code=400, detail="样例文件路径不合法")
    relative = Path(*windows_parts[test_data_index + 1 :]).as_posix()
    return resolve_descendant(DATA_ROOT, relative)


def qwen_infer_file_url(
    dataset: str,
    pair_number: int,
    relative_path: str,
    version: int | None = None,
) -> str:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    url = f"/api/qwen-review/file/{dataset}/{pair_number}/{normalized}"
    return f"{url}?v={version}" if version is not None else url


def optional_qwen_artifact_url(
    dataset: str,
    pair_number: int,
    relative_path: object,
    version: int,
) -> str | None:
    if not isinstance(relative_path, str) or not relative_path.strip():
        return None
    return qwen_infer_file_url(dataset, pair_number, relative_path, version)


def candidate_images_for_region(manifest: dict, region: dict) -> list[dict]:
    """Return the exact candidate sequence represented by a region Prompt."""

    region_candidates = region.get("candidate_images")
    candidates = (
        region_candidates
        if isinstance(region_candidates, list)
        else manifest.get("candidate_images", [])
    )
    return [dict(candidate) for candidate in candidates if isinstance(candidate, dict)]


def prompt_stages_for_region(manifest: dict, region: dict) -> list[dict]:
    """Normalize new staged manifests and legacy single-Prompt samples."""

    raw_stages = region.get("prompt_stages")
    if isinstance(raw_stages, list) and raw_stages:
        stages = [dict(stage) for stage in raw_stages if isinstance(stage, dict)]
        if stages:
            return stages
    task_type = str(manifest.get("task_type", ""))
    default_stage = (
        "shortage_product" if task_type == "SHORTAGE" else "misplaced_product"
    )
    return [
        {
            "stage": default_stage,
            "label": "当前区域 Prompt",
            "candidate_scope": "legacy",
            "prompt_image_1": region.get("prompt_image_1", ""),
            "prompt": region.get("prompt", ""),
            "candidate_count_before": region.get("candidate_count_before"),
            "candidate_count_after": region.get("candidate_count_after"),
            "candidate_images": candidate_images_for_region(manifest, region),
            "candidate_sheets": region.get("candidate_sheets", []),
        }
    ]


def select_prompt_stage(
    manifest: dict,
    region: dict,
    requested_stage: str | None,
) -> dict:
    stages = prompt_stages_for_region(manifest, region)
    normalized = (requested_stage or "").strip()
    if not normalized:
        return stages[0]
    stage = next((item for item in stages if item.get("stage") == normalized), None)
    if stage is None:
        valid = ", ".join(str(item.get("stage")) for item in stages)
        raise HTTPException(status_code=400, detail=f"stage 只能是: {valid}")
    return stage


def candidate_images_for_stage(stage: dict) -> list[dict]:
    candidates = stage.get("candidate_images", [])
    if not isinstance(candidates, list):
        raise HTTPException(status_code=500, detail="Prompt stage 候选图片格式错误")
    return [dict(candidate) for candidate in candidates if isinstance(candidate, dict)]


def candidate_sheets_for_stage(stage: dict) -> list[dict]:
    sheets = stage.get("candidate_sheets", [])
    if sheets is None:
        return []
    if not isinstance(sheets, list):
        raise HTTPException(status_code=500, detail="Prompt stage 候选拼图格式错误")
    return [dict(sheet) for sheet in sheets if isinstance(sheet, dict)]


def prompt_reference_images_for_stage(stage: dict) -> list[dict]:
    """Return physical reference images while preserving legacy manifests."""

    sheets = candidate_sheets_for_stage(stage)
    return sheets if sheets else candidate_images_for_stage(stage)


def qwen_stage_input_signature(prompt: str, *, pair_root: Path, stage: dict) -> str:
    """Fingerprint exactly the prompt and image bytes sent for one stage."""

    digest = hashlib.sha256()
    # The run endpoint intentionally strips surrounding whitespace before
    # sending; normalize the displayed file the same way for comparison.
    digest.update(prompt.strip().encode("utf-8"))
    relative_paths = [str(stage.get("prompt_image_1", ""))]
    relative_paths.extend(
        str(item.get("path", "")) for item in prompt_reference_images_for_stage(stage)
    )
    for relative_path in relative_paths:
        path = resolve_descendant(pair_root, relative_path)
        try:
            digest.update(path.read_bytes())
        except OSError:
            # Message construction performs the authoritative existence check.
            # Retain a deterministic signature here so legacy/tests can still
            # render their status instead of failing the whole samples API.
            digest.update(f"missing:{relative_path}".encode("utf-8"))
    return digest.hexdigest()


def resolve_prompt_stage_paths(
    pair_root: Path,
    region_root: Path,
    stage: dict,
) -> tuple[Path, Path, Path, Path]:
    generated_path = resolve_descendant(pair_root, str(stage.get("prompt", "")))
    input_path = resolve_descendant(
        pair_root,
        str(stage.get("prompt_image_1", "")),
    )
    stage_root = generated_path.parent
    if not stage_root.resolve().is_relative_to(region_root.resolve()):
        raise HTTPException(status_code=400, detail="Prompt stage 路径不属于当前 region")
    return (
        generated_path,
        input_path,
        stage_root / "prompt_override.txt",
        stage_root / "qwen_infer_result.json",
    )


def add_candidate_urls(
    dataset: str,
    pair_number: int,
    candidates: object,
    version: int,
) -> list[dict]:
    if not isinstance(candidates, list):
        return []
    return [
        {
            **candidate,
            "url": qwen_infer_file_url(
                dataset,
                pair_number,
                str(candidate.get("path", "")),
                version,
            ),
        }
        for candidate in candidates
        if isinstance(candidate, dict)
    ]


def prompt_image_markers(prompt: str) -> list[int]:
    return [int(value) for value in re.findall(r"\[IMAGE\s+(\d+)\]", prompt)]


def prompt_has_expected_images(prompt: str, image_count: int) -> bool:
    return prompt_image_markers(prompt) == list(range(1, image_count + 1))


def candidate_names_from_manifest(candidates: object) -> list[str]:
    if not isinstance(candidates, list):
        raise HTTPException(status_code=500, detail="候选图片 manifest 格式错误")
    names: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("name"), str):
            raise HTTPException(status_code=500, detail="候选图片 manifest 缺少商品名")
        names.append(candidate["name"])
    return names


def prompt_has_expected_candidates(prompt: str, expected_names: list[str]) -> bool:
    legacy_entries = [
        (int(index), name.strip())
        for index, name in re.findall(
            r"(?m)^\s*CANDIDATE\s+(\d+)\s*:\s*(.*?)\s*;\s*$",
            prompt,
        )
    ]
    numbered_entries = [
        (int(index), name.strip())
        for index, name in re.findall(
            r"(?m)^\s*SKU\s+(\d+)\s*:\s*(.*?)\s*$",
            prompt,
        )
    ]
    plain_numbered_entries = [
        (int(index), name.strip())
        for index, name in re.findall(
            r"(?m)^\s*(\d+)\s*:\s*(.*?)\s*$",
            prompt,
        )
    ]
    expected_entries = list(enumerate(expected_names, start=1))
    summaries = [
        value.strip()
        for value in re.findall(r"(?m)^\s*候选商品：(.*?)\s*$", prompt)
    ]
    if numbered_entries:
        return (
            not legacy_entries
            and not plain_numbered_entries
            and not summaries
            and numbered_entries == expected_entries
        )
    if plain_numbered_entries:
        return (
            not legacy_entries
            and not summaries
            and plain_numbered_entries == expected_entries
        )
    return (
        legacy_entries == expected_entries
        and summaries == ["、".join(expected_names)]
    )


def read_text_file(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"读取{label}失败: {error}") from error


def load_json_file(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"读取{label}失败: {error}") from error
    if not isinstance(value, dict):
        raise HTTPException(status_code=500, detail=f"{label}必须是 JSON 对象")
    return value


def write_text_atomic(path: Path, value: str, label: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(value, encoding="utf-8")
        temporary_path.replace(path)
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"保存{label}失败: {error}") from error


def write_json_atomic(path: Path, value: dict, label: str) -> None:
    write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        label,
    )


def validate_saved_prompt(prompt: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"\s*=== SYSTEM ===\s*\n(.*?)\n\s*=== USER ===\s*\n(.*?)\s*",
        prompt,
        flags=re.DOTALL,
    )
    if match is None:
        raise HTTPException(
            status_code=400,
            detail="Prompt 必须保留 === SYSTEM === 和 === USER === 两个区块",
        )
    system_prompt = match.group(1).strip()
    user_prompt = match.group(2).strip()
    if not system_prompt or not user_prompt:
        raise HTTPException(status_code=400, detail="SYSTEM/USER Prompt 不能为空")
    return system_prompt, user_prompt


def validate_saved_prompt_images(prompt: str, image_count: int) -> None:
    if not prompt_has_expected_images(prompt, image_count):
        raise HTTPException(
            status_code=400,
            detail=(
                "Prompt 图片标记必须依次为 "
                f"[IMAGE 1] 到 [IMAGE {image_count}]，并与当前审核阶段的输入一致"
            ),
        )


def validate_saved_prompt_candidates(prompt: str, candidate_images: list) -> None:
    expected_names = candidate_names_from_manifest(candidate_images)
    if not prompt_has_expected_candidates(prompt, expected_names):
        raise HTTPException(
            status_code=400,
            detail=(
                "Prompt 候选商品名称和编号顺序必须与当前审核阶段的"
                "实际输入一致"
            ),
        )


def build_saved_qwen_messages(
    prompt: str,
    pair_root: Path,
    region: dict,
    candidate_images: list,
) -> tuple[str, list[dict]]:
    system_prompt, user_prompt = validate_saved_prompt(prompt)
    image_paths = [
        resolve_descendant(pair_root, str(region.get("prompt_image_1", "")))
    ]
    for reference_image in prompt_reference_images_for_stage(region):
        if not isinstance(reference_image, dict):
            raise HTTPException(status_code=500, detail="候选图片 manifest 格式错误")
        image_paths.append(
            resolve_descendant(
                pair_root,
                str(reference_image.get("path", "")),
            )
        )
    if any(not path.is_file() for path in image_paths):
        raise HTTPException(status_code=404, detail="Prompt 引用的输入图片不存在")

    validate_saved_prompt_images(prompt, len(image_paths))
    validate_saved_prompt_candidates(prompt, candidate_images)

    content: list[dict] = []
    for part in re.split(r"(\[IMAGE\s+\d+\])", user_prompt):
        marker = re.fullmatch(r"\[IMAGE\s+(\d+)\]", part)
        if marker is None:
            if part:
                content.append({"type": "text", "text": part})
            continue
        path = image_paths[int(marker.group(1)) - 1]
        media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{encoded}"},
            }
        )
    return system_prompt, content


def call_qwen_messages(
    system_prompt: str,
    user_content: list[dict],
    *,
    temperature: float,
) -> str:
    payload = {
            "model": QWEN3_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": 800,
        }
    response = None
    for attempt in range(1, QWEN_REQUEST_MAX_ATTEMPTS + 1):
        try:
            response = requests.post(
                QWEN3_URL, json=payload, timeout=QWEN_REQUEST_TIMEOUT_SECONDS
            )
            break
        except (requests.Timeout, requests.ConnectionError):
            if attempt >= QWEN_REQUEST_MAX_ATTEMPTS:
                raise
            time.sleep(min(0.5 * attempt, 1.0))
    assert response is not None
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        detail = response.text.strip()[:1200]
        suffix = f"；上游响应: {detail}" if detail else ""
        raise requests.HTTPError(
            f"HTTP {response.status_code} {response.reason}{suffix}",
            response=response,
        ) from error
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("choices[0].message.content 不是字符串")
    return content


def parse_first_json(content: str) -> dict | list:
    if not isinstance(content, str):
        raise TypeError("Qwen 输出不是字符串")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", content):
        try:
            value, _ = decoder.raw_decode(content[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    raise ValueError("Qwen 输出中没有找到 JSON 对象或数组")


@app.post("/api/sam3")
def run_sam3(request: PromptRequest) -> dict:
    image_path = resolve_image(request.image_name)
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="SAM3 prompt 不能为空")

    media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    try:
        with image_path.open("rb") as image_file:
            response = requests.post(
                SAM3_URL,
                files={"image": (image_path.name, image_file, media_type)},
                data={
                    "prompt": prompt,
                    "threshold": 0.5,
                    "mask_threshold": 0.5,
                },
                timeout=120,
            )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"SAM3 请求失败: {error}",
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=502,
            detail=f"SAM3 返回格式错误: {error}",
        ) from error

    if not isinstance(result, dict) or not isinstance(result.get("instances"), list):
        raise HTTPException(status_code=502, detail="SAM3 响应缺少 instances")
    return result


@app.post("/api/sam3-crop")
def run_sam3_crop(request: SamCropRequest) -> dict:
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="SAM3 prompt 不能为空")
    if (
        len(request.crop_box_original) != 4
        or not all(
            isinstance(value, (int, float)) for value in request.crop_box_original
        )
    ):
        raise HTTPException(status_code=400, detail="crop_box_original 格式错误")

    try:
        image_bytes = base64.b64decode(request.image_base64, validate=True)
    except (ValueError, binascii.Error) as error:
        raise HTTPException(status_code=400, detail="crop 图片 Base64 格式错误") from error
    if not image_bytes:
        raise HTTPException(status_code=400, detail="crop 图片不能为空")
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="crop 图片不能超过 20 MB")

    try:
        response = requests.post(
            SAM3_URL,
            files={"image": ("qwen_crop.jpg", image_bytes, "image/jpeg")},
            data={
                "prompt": prompt,
                "threshold": 0.5,
                "mask_threshold": 0.5,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=f"SAM3 crop 请求失败: {error}",
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=502,
            detail=f"SAM3 crop 返回格式错误: {error}",
        ) from error

    if not isinstance(result, dict) or not isinstance(result.get("instances"), list):
        raise HTTPException(status_code=502, detail="SAM3 crop 响应缺少 instances")

    crop_x1, crop_y1, _, _ = request.crop_box_original
    for instance in result["instances"]:
        bbox = instance.get("bbox_xyxy") if isinstance(instance, dict) else None
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(value, (int, float)) for value in bbox)
        ):
            instance["bbox_original_xyxy"] = [
                bbox[0] + crop_x1,
                bbox[1] + crop_y1,
                bbox[2] + crop_x1,
                bbox[3] + crop_y1,
            ]
    result["crop_box_original"] = request.crop_box_original
    return result


def resolve_image(image_name: str) -> Path:
    if Path(image_name).name != image_name:
        raise HTTPException(status_code=400, detail="图片文件名不合法")
    image_path = RGB_DIR / image_name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="图片不存在")
    return image_path


def load_skus() -> list[dict]:
    try:
        catalog = json.loads(SKU_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"读取 SKU 商品库失败: {error}",
        ) from error

    products = catalog.get("products")
    if not isinstance(products, list):
        raise HTTPException(status_code=500, detail="SKU 商品库缺少 products 数组")

    skus = []
    for product in products:
        if not isinstance(product, dict):
            continue
        name = product.get("name")
        if isinstance(name, str) and name.strip():
            skus.append({"name": name.strip()})
    return skus


def normalize_task_type(task_type: str) -> str:
    normalized = task_type.strip().upper()
    if normalized not in PROMPT_PAIR_MAPPING_PATHS:
        raise HTTPException(
            status_code=400,
            detail=f"task_type 只能是: {', '.join(SUPPORTED_TASK_TYPES)}",
        )
    return normalized


def is_hard_case_request(
    sku_name: str,
    task_type: str,
    level: str | None,
    hand: str | None,
) -> bool:
    if task_type != "SORTING" or level is None or hand is None:
        return False
    try:
        scope = json.loads(HARD_CASE_SCOPE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"读取 hard case 范围失败: {error}") from error
    key = (sku_name.strip(), level.strip().upper(), hand.strip().lower())
    return any(
        isinstance(item, dict)
        and (
            str(item.get("product_name", "")).strip(),
            str(item.get("level", "")).strip().upper(),
            str(item.get("hand", "")).strip().lower(),
        ) == key
        for item in scope
    )


def prompt_mapping_path(task_type: str, *, hard_case: bool = False) -> Path:
    normalized = normalize_task_type(task_type)
    if normalized == "SORTING" and hard_case:
        return HARD_CASE_PROMPT_MAPPING_PATH
    if normalized == "SORTING":
        return PROMPT_PAIR_MAPPING_PATH
    return PROMPT_PAIR_MAPPING_PATHS[normalized]


def load_prompt_pair_mapping(
    task_type: str = "SORTING", *, hard_case: bool = False
) -> dict[str, dict[str, str]]:
    mapping_path = prompt_mapping_path(task_type, hard_case=hard_case)
    if not mapping_path.exists():
        return {}
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"读取配对 Prompt 映射失败: {error}",
        ) from error
    if not isinstance(mapping, dict) or not all(
        isinstance(name, str)
        and isinstance(prompts, dict)
        and isinstance(prompts.get("qwen3_prompt"), str)
        and isinstance(prompts.get("sam3_prompt"), str)
        for name, prompts in mapping.items()
    ):
        raise HTTPException(
            status_code=500,
            detail="配对 Prompt 映射必须是 SKU 到 Qwen3/SAM3 Prompt 的映射",
        )
    return mapping


def write_json_mapping(path: Path, mapping: dict, label: str) -> None:
    ordered_mapping = dict(sorted(mapping.items(), key=lambda item: item[0]))
    temporary_path = path.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(ordered_mapping, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"保存 {label} 失败: {error}",
        ) from error


def parse_qwen_json(content: str) -> list[dict]:
    if not isinstance(content, str):
        raise TypeError("choices[0].message.content 不是字符串")

    decoded = None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", content):
        try:
            candidate, _ = decoder.raw_decode(content[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) or (
            isinstance(candidate, list)
            and all(isinstance(item, dict) for item in candidate)
        ):
            decoded = candidate
            break

    if decoded is None:
        raise ValueError("没有找到 JSON 对象或数组")

    items = [decoded] if isinstance(decoded, dict) else decoded
    detections = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 项必须是 JSON 对象")
        name = item.get("name")
        bbox = item.get("bbox")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"第 {index} 项的 name 必须是非空字符串")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, (int, float)) for value in bbox)
        ):
            raise ValueError(f"第 {index} 项的 bbox 必须是四个数字组成的数组")
        detections.append({"name": name.strip(), "bbox": bbox})

    return detections


if __name__ == "__main__":
    uvicorn.run(app, host=SERVICE_BIND_HOST, port=8082)
