"""Use shelf candidates and localized change regions to ask Qwen for SKU names."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence
from urllib.parse import quote

import cv2
import numpy as np
import requests


PERCEPTION_ROOT = Path(__file__).resolve().parents[3]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from config import QWEN3_MODEL, QWEN3_URL, SKU_API_URL  # noqa: E402
from .visual_retrieval import (  # noqa: E402
    RetrievalMatch,
    VisualRetrievalError,
    VisualSkuRetriever,
)


TaskType = Literal["SHORTAGE", "MISPLACED"]
MisplacedStage = Literal["misplaced_product", "expected_product"]
PoseType = Literal["", "SHELF_VIEW_UPPER", "SHELF_VIEW_LOWER"]
TARGET_SIZE = (1280, 720)
SKU_TIMEOUT_SECONDS = 8.0
QWEN_TIMEOUT_SECONDS = float(os.getenv("QWEN_REQUEST_TIMEOUT_SECONDS", "30"))
QWEN_MAX_ATTEMPTS = max(1, int(os.getenv("QWEN_REQUEST_MAX_ATTEMPTS", "3")))
MAX_REFERENCE_IMAGE_BYTES = 20 * 1024 * 1024
MAX_REFERENCE_IMAGE_SIDE = 1024
CONTACT_SHEET_COLUMNS = 5
CONTACT_SHEET_MAX_ROWS = 4
CONTACT_SHEET_TILE_WIDTH = 184
CONTACT_SHEET_IMAGE_HEIGHT = 176
CONTACT_SHEET_LABEL_HEIGHT = 36
CONTACT_SHEET_GAP = 10
CONTACT_SHEET_MARGIN = 12
UNKNOWN_NAMES = {"", "UNKNOWN", "无法确认", "不确定"}
logger = logging.getLogger("uvicorn.error")
PROMPT_ROOT = Path(__file__).resolve().parent
PROMPT_PATHS: dict[TaskType, Path] = {
    "SHORTAGE": PROMPT_ROOT / "shortage_prompt.txt",
    "MISPLACED": PROMPT_ROOT / "misplaced_prompt.txt",
}
MISPLACED_PROMPT_PATHS: dict[MisplacedStage, Path] = {
    "misplaced_product": PROMPT_ROOT / "misplaced_prompt.txt",
    "expected_product": PROMPT_ROOT / "misplaced_expected_prompt.txt",
}
DEFAULT_DEBUG_ROOT = Path(
    os.getenv("INSPECT_QWEN_DEBUG_DIR", str(PROMPT_ROOT / "debug"))
)
_ENV_RETRIEVER = object()


class QwenReviewError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class CandidateProduct:
    sku_id: str
    name: str
    row_numbers: tuple[int, ...]
    image: bytes
    media_type: str


@dataclass(frozen=True)
class CandidateContactSheet:
    """One numbered grid image containing a contiguous candidate range."""

    first_candidate_number: int
    last_candidate_number: int
    image: bytes
    media_type: str = "image/jpeg"


@dataclass(frozen=True)
class ReviewRowConstraint:
    """Reliable mapping between one finding and a visible shelf row."""

    row_index: int
    row_bbox: tuple[int, int, int, int]
    overlap_ratio: float
    detected_row_index: int | None = None


@dataclass(frozen=True)
class ReviewedFinding:
    region_index: int
    confidence: float
    shortage_product_name: str | None = None
    misplaced_product_name: str | None = None
    gt_product_name: str | None = None


@dataclass(frozen=True)
class QwenReviewResult:
    findings: tuple[ReviewedFinding, ...]
    raw_response: str
    candidate_names: tuple[str, ...]
    debug_directory: Path | None = None


class QwenReviewer:
    """Fetch candidate references, assemble visual evidence, and call Qwen."""

    def __init__(
        self,
        *,
        sku_base_url: str = SKU_API_URL,
        qwen_url: str = QWEN3_URL,
        qwen_model: str = QWEN3_MODEL,
        sku_timeout: float = SKU_TIMEOUT_SECONDS,
        qwen_timeout: float = QWEN_TIMEOUT_SECONDS,
        qwen_max_attempts: int = QWEN_MAX_ATTEMPTS,
        api_key: str | None = None,
        session: Any | None = None,
        debug_root: str | Path | None = None,
        visual_retriever: Any = _ENV_RETRIEVER,
    ) -> None:
        self.sku_base_url = sku_base_url.rstrip("/")
        self.qwen_url = _chat_completions_url(qwen_url)
        self.qwen_model = qwen_model
        self.sku_timeout = sku_timeout
        self.qwen_timeout = qwen_timeout
        self.qwen_max_attempts = max(1, qwen_max_attempts)
        self.api_key = api_key or os.getenv("QWEN_API_KEY", "").strip() or None
        self.session = session or requests.Session()
        self.debug_root = Path(debug_root) if debug_root is not None else None
        self.visual_retriever = (
            VisualSkuRetriever.from_environment()
            if visual_retriever is _ENV_RETRIEVER
            else visual_retriever
        )
        self._retrieval_candidate_cache: dict[str, CandidateProduct] = {}

    def review(
        self,
        *,
        task_type: TaskType,
        location_id: str,
        pose_type: PoseType,
        current: np.ndarray,
        baseline: np.ndarray | None = None,
        bboxes: Sequence[Sequence[int]],
        row_constraints: Sequence[ReviewRowConstraint | None] | None = None,
    ) -> QwenReviewResult:
        if task_type not in {"SHORTAGE", "MISPLACED"}:
            raise ValueError("task_type must be SHORTAGE or MISPLACED")
        if not bboxes:
            return QwenReviewResult((), "", ())
        if row_constraints is None:
            normalized_constraints: list[ReviewRowConstraint | None] = [
                None
            ] * len(bboxes)
        else:
            if len(row_constraints) != len(bboxes):
                raise ValueError("row_constraints must match bboxes")
            normalized_constraints = list(row_constraints)

        current = _resize_image(current)
        baseline = _resize_image(baseline) if baseline is not None else current
        normalized_bboxes = [_normalize_bbox(bbox) for bbox in bboxes]
        debug_directory = self._create_debug_directory(
            task_type,
            location_id,
            pose_type,
            normalized_bboxes,
            normalized_constraints,
        )
        rows = self._fetch_candidate_rows(location_id, pose_type)
        normalized_constraints = [
            constraint
            if constraint is not None
            and 1 <= constraint.row_index <= len(rows)
            else None
            for constraint in normalized_constraints
        ]
        included_row_numbers: set[int] | None = None
        if task_type == "SHORTAGE" and all(
            constraint is not None for constraint in normalized_constraints
        ):
            requested_rows = {
                constraint.row_index
                for constraint in normalized_constraints
                if constraint is not None
            }
            if all(rows[row_number - 1] for row_number in requested_rows):
                included_row_numbers = requested_rows
        candidates = self._fetch_candidate_images(
            rows,
            included_row_numbers=included_row_numbers,
        )
        if not candidates:
            raise QwenReviewError("candidate_lookup", "SKU 服务未返回候选商品")

        candidate_names = {candidate.name for candidate in candidates}
        if debug_directory is not None:
            _write_json(
                debug_directory / "candidates.json",
                {
                    "rows": rows,
                    "candidates": [
                        {
                            "sku_id": candidate.sku_id,
                            "name": candidate.name,
                            "row_numbers": list(candidate.row_numbers),
                            "media_type": candidate.media_type,
                        }
                        for candidate in candidates
                    ],
                    "row_constraints": [
                        _row_constraint_dict(constraint)
                        for constraint in normalized_constraints
                    ],
                },
            )
        findings: list[ReviewedFinding] = []
        raw_responses: list[Any] = []
        # Intentionally review each bbox in an independent Qwen request. This
        # keeps the model's task local and avoids cross-region bookkeeping.
        for region_index, (bbox, row_constraint) in enumerate(
            zip(normalized_bboxes, normalized_constraints),
            start=1,
        ):
            expected_candidates = (
                [
                    candidate
                    for candidate in candidates
                    if row_constraint.row_index in candidate.row_numbers
                ]
                if row_constraint is not None
                else list(candidates)
            )
            if not expected_candidates:
                row_constraint = None
                expected_candidates = list(candidates)
            region_directory = (
                debug_directory / f"region_{region_index:02d}"
                if debug_directory is not None
                else None
            )
            if task_type == "SHORTAGE":
                region_image = crop_review_region(
                    baseline,
                    bbox,
                    task_type,
                    row_bbox=(
                        row_constraint.row_bbox
                        if row_constraint is not None
                        else None
                    ),
                )
                payload = build_qwen_payload(
                    task_type=task_type,
                    location_id=location_id,
                    pose_type=pose_type,
                    region_image=region_image,
                    candidate_rows=rows,
                    candidates=expected_candidates,
                    model=self.qwen_model,
                    expected_row_index=(
                        row_constraint.row_index
                        if row_constraint is not None
                        else None
                    ),
                    detected_row_index=(
                        row_constraint.detected_row_index
                        if row_constraint is not None
                        else None
                    ),
                )
                if region_directory is not None:
                    region_directory.mkdir(parents=True, exist_ok=True)
                    _write_image(region_directory / "bbox_expanded.jpg", region_image)
                    _write_text(
                        region_directory / "prompt.txt",
                        _payload_as_readable_prompt(payload),
                    )
                raw = self._request_qwen(payload)
                raw_responses.append(raw)
                if region_directory is not None:
                    _write_text(region_directory / "qwen_raw.txt", raw)
                finding = parse_qwen_review(
                    raw,
                    task_type=task_type,
                    candidate_names=candidate_names,
                    expected_names={
                        candidate.name for candidate in expected_candidates
                    },
                    region_index=region_index,
                )
                if region_directory is not None:
                    _write_json(
                        region_directory / "parsed_result.json",
                        _reviewed_finding_dict(finding),
                    )
            else:
                misplaced_image = crop_review_region(
                    current,
                    bbox,
                    "MISPLACED",
                    pose_type=pose_type,
                    row_bbox=(
                        row_constraint.row_bbox
                        if row_constraint is not None
                        else None
                    ),
                )
                retrieval_matches: list[RetrievalMatch] = []
                misplaced_candidates = list(candidates)
                if self.visual_retriever is not None:
                    try:
                        retrieval_matches = self.visual_retriever.retrieve(
                            misplaced_image
                        )
                    except VisualRetrievalError as error:
                        raise QwenReviewError("visual_retrieval", str(error)) from error
                    if not retrieval_matches:
                        raise QwenReviewError(
                            "visual_retrieval", "全量 SKU 特征检索未返回候选商品"
                        )
                    misplaced_candidates = self._fetch_retrieval_candidate_images(
                        retrieval_matches
                    )
                misplaced_candidate_names = {
                    candidate.name for candidate in misplaced_candidates
                }
                expected_image = build_expected_product_reference_image(
                    baseline,
                    bbox,
                    pose_type=pose_type,
                    row_bbox=(
                        row_constraint.row_bbox
                        if row_constraint is not None
                        else None
                    ),
                )
                misplaced_payload = build_qwen_payload(
                    task_type="MISPLACED",
                    location_id=location_id,
                    pose_type=pose_type,
                    region_image=misplaced_image,
                    candidate_rows=rows,
                    candidates=misplaced_candidates,
                    model=self.qwen_model,
                    misplaced_stage="misplaced_product",
                )
                expected_payload = build_qwen_payload(
                    task_type="MISPLACED",
                    location_id=location_id,
                    pose_type=pose_type,
                    region_image=expected_image,
                    candidate_rows=rows,
                    candidates=expected_candidates,
                    model=self.qwen_model,
                    expected_row_index=(
                        row_constraint.row_index
                        if row_constraint is not None
                        else None
                    ),
                    detected_row_index=(
                        row_constraint.detected_row_index
                        if row_constraint is not None
                        else None
                    ),
                    misplaced_stage="expected_product",
                )
                if region_directory is not None:
                    _write_stage_debug_input(
                        region_directory,
                        "misplaced_product",
                        misplaced_image,
                        misplaced_payload,
                    )
                    if retrieval_matches:
                        _write_json(
                            region_directory / "misplaced_product" / "retrieval.json",
                            {
                                "scope": "full_catalog_top_k",
                                "matches": [
                                    {
                                        "rank": rank,
                                        "sku_id": match.sku_id,
                                        "name": match.name,
                                        "score": match.score,
                                    }
                                    for rank, match in enumerate(
                                        retrieval_matches, start=1
                                    )
                                ],
                            },
                        )
                misplaced_raw = self._request_qwen(misplaced_payload)
                misplaced_finding = parse_qwen_review(
                    misplaced_raw,
                    task_type="MISPLACED",
                    candidate_names=misplaced_candidate_names,
                    region_index=region_index,
                    misplaced_stage="misplaced_product",
                )
                if region_directory is not None:
                    _write_stage_debug_output(
                        region_directory,
                        "misplaced_product",
                        misplaced_raw,
                        misplaced_finding,
                    )

                if region_directory is not None:
                    _write_stage_debug_input(
                        region_directory,
                        "expected_product",
                        expected_image,
                        expected_payload,
                    )
                expected_raw = self._request_qwen(expected_payload)
                expected_finding = parse_qwen_review(
                    expected_raw,
                    task_type="MISPLACED",
                    candidate_names=candidate_names,
                    expected_names={
                        candidate.name for candidate in expected_candidates
                    },
                    region_index=region_index,
                    misplaced_stage="expected_product",
                )
                if region_directory is not None:
                    _write_stage_debug_output(
                        region_directory,
                        "expected_product",
                        expected_raw,
                        expected_finding,
                    )
                raw_responses.append(
                    {
                        "misplaced_product": misplaced_raw,
                        "expected_product": expected_raw,
                    }
                )
                finding = combine_misplaced_stage_findings(
                    misplaced_finding,
                    expected_finding,
                    region_index=region_index,
                )
            if finding is not None:
                findings.append(finding)
        if debug_directory is not None:
            _write_json(
                debug_directory / "result.json",
                {
                    "findings": [
                        _reviewed_finding_dict(finding) for finding in findings
                    ],
                    "raw_response_count": sum(
                        len(response) if isinstance(response, dict) else 1
                        for response in raw_responses
                    ),
                },
            )
        return QwenReviewResult(
            findings=tuple(findings),
            raw_response=json.dumps(raw_responses, ensure_ascii=False),
            candidate_names=tuple(candidate.name for candidate in candidates),
            debug_directory=debug_directory,
        )

    def _fetch_retrieval_candidate_images(
        self,
        matches: Sequence[RetrievalMatch],
    ) -> list[CandidateProduct]:
        candidates: list[CandidateProduct] = []
        for match in matches:
            cached = self._retrieval_candidate_cache.get(match.sku_id)
            if cached is not None:
                candidates.append(cached)
                continue
            product_response = self._request(
                "GET",
                f"{self.sku_base_url}/sku/search_by_SKU",
                stage="candidate_lookup",
                params={"sku": match.sku_id},
                timeout=self.sku_timeout,
            )
            try:
                product = product_response.json()
                paths = product["images"]
                returned_name = product["name"]
                returned_sku = product["sku_id"]
                path = paths[0]
            except (ValueError, KeyError, IndexError, TypeError) as error:
                raise QwenReviewError(
                    "candidate_lookup",
                    f"商品 {match.sku_id} 的 SKU 接口返回格式无效",
                ) from error
            if returned_sku != match.sku_id or returned_name != match.name:
                raise QwenReviewError(
                    "candidate_lookup",
                    f"特征索引与 SKU 服务不一致: {match.sku_id}/{match.name}",
                )
            image_response = self._request(
                "GET",
                f"{self.sku_base_url}/{quote(path.lstrip('/'), safe='/')}",
                stage="candidate_image_download",
                timeout=self.sku_timeout,
            )
            image, media_type = _validated_reference_response(
                image_response,
                match.name,
            )
            candidate = CandidateProduct(
                sku_id=match.sku_id,
                name=match.name,
                row_numbers=(),
                image=image,
                media_type=media_type,
            )
            self._retrieval_candidate_cache[match.sku_id] = candidate
            candidates.append(candidate)
        return candidates

    def _create_debug_directory(
        self,
        task_type: TaskType,
        location_id: str,
        pose_type: PoseType,
        bboxes: Sequence[Sequence[int]],
        row_constraints: Sequence[ReviewRowConstraint | None],
    ) -> Path | None:
        if self.debug_root is None:
            return None
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        safe_location = re.sub(r"[^A-Za-z0-9_.-]+", "_", location_id.strip())
        directory = (
            self.debug_root
            / f"{timestamp}_{safe_location}_{task_type}_{uuid.uuid4().hex[:8]}"
        )
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            raise QwenReviewError(
                "debug_artifact",
                f"无法创建 Qwen 调试目录: {directory}",
            ) from error
        _write_json(
            directory / "request.json",
            {
                "task_type": task_type,
                "location_id": location_id,
                "pose_type": pose_type,
                "bbox_format": ["x", "y", "width", "height"],
                "bboxes": [list(bbox) for bbox in bboxes],
                "row_constraints": [
                    _row_constraint_dict(constraint)
                    for constraint in row_constraints
                ],
            },
        )
        logger.info(
            "Qwen review debug artifacts: task_type=%s location_id=%s directory=%s",
            task_type,
            location_id,
            directory,
        )
        return directory

    def _fetch_candidate_rows(
        self,
        location_id: str,
        pose_type: PoseType,
    ) -> list[list[dict[str, str]]]:
        response = self._request(
            "GET",
            f"{self.sku_base_url}/sku/get_candidate_SKU",
            stage="candidate_lookup",
            json={"location_id": location_id, "pose_type": pose_type},
            timeout=self.sku_timeout,
        )
        try:
            payload = response.json()
        except ValueError as error:
            raise QwenReviewError(
                "candidate_lookup",
                "候选 SKU 接口返回的不是 JSON",
            ) from error
        if not isinstance(payload, list) or not all(
            isinstance(row, list) for row in payload
        ):
            raise QwenReviewError(
                "candidate_lookup",
                "候选 SKU 接口必须返回二维数组",
            )

        rows: list[list[dict[str, str]]] = []
        for row_index, raw_row in enumerate(payload, start=1):
            row: list[dict[str, str]] = []
            for item in raw_row:
                if not isinstance(item, dict):
                    raise QwenReviewError(
                        "candidate_lookup",
                        f"候选 SKU 第 {row_index} 行包含无效元素",
                    )
                sku_id = item.get("sku_id")
                name = item.get("name")
                if not isinstance(sku_id, str) or not isinstance(name, str):
                    raise QwenReviewError(
                        "candidate_lookup",
                        f"候选 SKU 第 {row_index} 行缺少 sku_id/name",
                    )
                row.append({"sku_id": sku_id, "name": name})
            rows.append(row)
        return rows

    def _fetch_candidate_images(
        self,
        rows: Sequence[Sequence[dict[str, str]]],
        *,
        included_row_numbers: set[int] | None = None,
    ) -> list[CandidateProduct]:
        row_numbers_by_name: dict[str, list[int]] = {}
        sku_by_name: dict[str, str] = {}
        ordered_names: list[str] = []
        for row_number, row in enumerate(rows, start=1):
            if (
                included_row_numbers is not None
                and row_number not in included_row_numbers
            ):
                continue
            for item in row:
                name = item["name"]
                if name not in row_numbers_by_name:
                    ordered_names.append(name)
                    row_numbers_by_name[name] = []
                    sku_by_name[name] = item["sku_id"]
                row_numbers_by_name[name].append(row_number)

        candidates: list[CandidateProduct] = []
        for name in ordered_names:
            paths_response = self._request(
                "GET",
                f"{self.sku_base_url}/sku/get_image",
                stage="candidate_image_lookup",
                params={"name": name},
                timeout=self.sku_timeout,
            )
            try:
                paths = paths_response.json()
            except ValueError as error:
                raise QwenReviewError(
                    "candidate_image_lookup",
                    f"商品 {name} 的图片接口返回的不是 JSON",
                ) from error

            path = paths[0].lstrip("/")
            image_response = self._request(
                "GET",
                f"{self.sku_base_url}/{quote(path, safe='/')}",
                stage="candidate_image_download",
                timeout=self.sku_timeout,
            )
            image, media_type = _validated_reference_response(image_response, name)
            candidates.append(
                CandidateProduct(
                    sku_id=sku_by_name[name],
                    name=name,
                    row_numbers=tuple(row_numbers_by_name[name]),
                    image=image,
                    media_type=media_type,
                )
            )
        return candidates

    def _request_qwen(self, payload: dict[str, Any]) -> str:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self._request(
            "POST",
            self.qwen_url,
            stage="qwen_review",
            json=payload,
            headers=headers,
            timeout=self.qwen_timeout,
        )
        try:
            value = response.json()
            content = value["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise QwenReviewError(
                "qwen_review",
                f"Qwen 返回格式无效: {response.text[:300]}",
            ) from error
        if not isinstance(content, str):
            raise QwenReviewError("qwen_review", "Qwen message.content 不是字符串")
        return content

    def _request(self, method: str, url: str, *, stage: str, **kwargs: Any) -> Any:
        attempts = self.qwen_max_attempts if stage == "qwen_review" else 1
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.request(method, url, **kwargs)
                break
            except (requests.Timeout, requests.ConnectionError) as error:
                if attempt >= attempts:
                    raise QwenReviewError(
                        stage,
                        f"无法连接上游服务（已尝试 {attempts} 次）: {error}",
                    ) from error
                logger.warning(
                    "Qwen request attempt %s/%s failed: %s",
                    attempt,
                    attempts,
                    error,
                )
                time.sleep(min(0.5 * attempt, 1.0))
            except requests.RequestException as error:
                raise QwenReviewError(stage, f"无法连接上游服务: {error}") from error
        if not response.ok:
            raise QwenReviewError(
                stage,
                f"上游服务返回 HTTP {response.status_code}: {response.text[:300]}",
            )
        return response


def _validated_reference_response(response: Any, name: str) -> tuple[bytes, str]:
    image = response.content
    if not image or len(image) > MAX_REFERENCE_IMAGE_BYTES:
        raise QwenReviewError(
            "candidate_image_download",
            f"商品 {name} 的标准图片为空或过大",
        )
    media_type = response.headers.get("Content-Type", "image/jpeg")
    media_type = media_type.split(";", 1)[0]
    return normalize_reference_image(image, media_type)


def build_qwen_payload(
    *,
    task_type: TaskType,
    location_id: str,
    pose_type: PoseType,
    region_image: np.ndarray,
    candidate_rows: Sequence[Sequence[dict[str, str]]],
    candidates: Sequence[CandidateProduct],
    model: str,
    expected_row_index: int | None = None,
    detected_row_index: int | None = None,
    misplaced_stage: MisplacedStage | None = None,
    candidate_sheets: Sequence[CandidateContactSheet] | None = None,
) -> dict[str, Any]:
    row_location = (
        f"画面检测第 {detected_row_index} 行，对应 SKU 候选第 {expected_row_index} 行"
        if detected_row_index is not None
        and expected_row_index is not None
        and detected_row_index != expected_row_index
        else f"当前画面从上到下第 {expected_row_index} 行"
    )
    if task_type == "SHORTAGE":
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "请只审核下面这一张货架局部图："
                    "缺货商品只能从以下候选商品中选择。"
                ),
            },
            _numpy_image_content(region_image),
            {"type": "text", "text": _candidate_names_text(candidates)},
        ]
    elif misplaced_stage == "misplaced_product":
        content = [
            {
                "type": "text",
                "text": "任务:识别局部图红色 bbox 中当前实际放置的商品。",
            },
            _numpy_image_content(region_image),
            {
                "type": "text",
                "text": _candidate_number_mapping_text(candidates),
            },
        ]
    elif misplaced_stage == "expected_product":
        content = [
            {
                "type": "text",
                "text": (
                    "标准放置组合图：上方是带红色 bbox 的完整标准放置行，"
                    "下方是 bbox 内物体的原图抠图；请识别该商品。"
                ),
            },
            _numpy_image_content(region_image),
            {
                "type": "text",
                "text": _expected_candidate_number_mapping_text(candidates),
            },
        ]
    else:
        raise ValueError(
            "MISPLACED requires misplaced_stage=misplaced_product or expected_product"
        )

    if task_type == "SHORTAGE":
        content.append({"type": "text", "text": "下面是候选 SKU 标准图："})
        for candidate_index, candidate in enumerate(candidates, start=1):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"CANDIDATE {candidate_index}: {candidate.name};",
                    },
                    _bytes_image_content(candidate.image, candidate.media_type),
                ]
            )
    else:
        sheets = (
            list(candidate_sheets)
            if candidate_sheets is not None
            else build_candidate_contact_sheets(candidates)
        )
        if candidates and not sheets:
            raise QwenReviewError("payload_assembly", "候选 SKU 拼图为空")
        content.append(
            {
                "type": "text",
                "text": (
                    "下面是候选 SKU 标准图拼图；每格上方数字与候选 SKU 编号一致。"
                ),
            }
        )
        for sheet_index, sheet in enumerate(sheets, start=1):
            if len(sheets) > 1:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"拼图 {sheet_index}：SKU {sheet.first_candidate_number}"
                            f"-{sheet.last_candidate_number}"
                        ),
                    }
                )
            content.append(_bytes_image_content(sheet.image, sheet.media_type))
    content.append(
        {
            "type": "text",
            "text": (
                "请按系统消息规定的简单 JSON 格式返回标准放置图中"
                "红色 bbox 内物体的识别结果。"
                if misplaced_stage == "expected_product"
                else "请按系统消息规定的简单 JSON 格式返回这一张局部图的结果。"
            ),
        }
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": load_system_prompt(
                    task_type,
                    misplaced_stage=misplaced_stage,
                ),
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": 800,
    }


def load_system_prompt(
    task_type: TaskType,
    *,
    misplaced_stage: MisplacedStage | None = None,
) -> str:
    try:
        path = (
            MISPLACED_PROMPT_PATHS[misplaced_stage or "misplaced_product"]
            if task_type == "MISPLACED"
            else PROMPT_PATHS[task_type]
        )
    except KeyError as error:
        raise ValueError("task_type must be SHORTAGE or MISPLACED") from error
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise QwenReviewError(
            "payload_assembly",
            f"无法读取 {task_type} Prompt: {path}",
        ) from error
    if not prompt:
        raise QwenReviewError(
            "payload_assembly",
            f"{task_type} Prompt 为空: {path}",
        )
    return prompt


def _payload_as_readable_prompt(payload: dict[str, Any]) -> str:
    """Render the exact text sequence with image placeholders, never base64."""

    messages = payload["messages"]
    lines = ["=== SYSTEM ===", str(messages[0]["content"]), "", "=== USER ==="]
    image_index = 0
    for item in messages[1]["content"]:
        if item.get("type") == "text":
            lines.append(str(item.get("text", "")))
        elif item.get("type") == "image_url":
            image_index += 1
            lines.append(f"[IMAGE {image_index}]")
    return "\n".join(lines).rstrip() + "\n"


def _reviewed_finding_dict(finding: ReviewedFinding | None) -> dict[str, Any]:
    if finding is None:
        return {"accepted": False}
    result: dict[str, Any] = {
        "accepted": True,
        "region_index": finding.region_index,
        "confidence": finding.confidence,
    }
    if finding.shortage_product_name is not None:
        result["shortage_product_name"] = finding.shortage_product_name
    if finding.misplaced_product_name is not None:
        result["misplaced_product_name"] = finding.misplaced_product_name
    if finding.gt_product_name is not None:
        result["gt_product_name"] = finding.gt_product_name
    return result


def _row_constraint_dict(
    constraint: ReviewRowConstraint | None,
) -> dict[str, Any] | None:
    if constraint is None:
        return None
    value = {
        "row_index": constraint.row_index,
        "row_bbox": list(constraint.row_bbox),
        "overlap_ratio": constraint.overlap_ratio,
    }
    if constraint.detected_row_index is not None:
        value["detected_row_index"] = constraint.detected_row_index
    return value


def _write_text(path: Path, value: str) -> None:
    try:
        path.write_text(value, encoding="utf-8")
    except OSError as error:
        raise QwenReviewError(
            "debug_artifact",
            f"无法写入 Qwen 调试文件: {path}",
        ) from error


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def _write_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(
        path.suffix or ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not success:
        raise QwenReviewError("debug_artifact", f"无法编码 Qwen 调试图片: {path}")
    try:
        encoded.tofile(path)
    except OSError as error:
        raise QwenReviewError(
            "debug_artifact",
            f"无法写入 Qwen 调试图片: {path}",
        ) from error


def _write_stage_debug_input(
    region_directory: Path,
    stage: MisplacedStage,
    image: np.ndarray,
    payload: dict[str, Any],
) -> None:
    stage_directory = region_directory / stage
    try:
        stage_directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise QwenReviewError(
            "debug_artifact",
            f"无法创建 Qwen 阶段调试目录: {stage_directory}",
        ) from error
    _write_image(stage_directory / "input.jpg", image)
    _write_text(
        stage_directory / "prompt.txt",
        _payload_as_readable_prompt(payload),
    )


def _write_stage_debug_output(
    region_directory: Path,
    stage: MisplacedStage,
    raw: str,
    finding: ReviewedFinding | None,
) -> None:
    stage_directory = region_directory / stage
    _write_text(stage_directory / "qwen_raw.txt", raw)
    _write_json(
        stage_directory / "parsed_result.json",
        _reviewed_finding_dict(finding),
    )


def parse_qwen_review(
    content: str,
    *,
    task_type: TaskType,
    candidate_names: set[str],
    region_index: int,
    expected_names: set[str] | None = None,
    misplaced_stage: MisplacedStage | None = None,
) -> ReviewedFinding | None:
    normalized = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", normalized, re.DOTALL)
    if fenced:
        normalized = fenced.group(1)
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as error:
        raise QwenReviewError(
            "qwen_output_validation",
            f"Qwen 输出不是合法 JSON: {content[:300]}",
        ) from error
    if not isinstance(payload, dict):
        raise QwenReviewError("qwen_output_validation", "Qwen 输出必须是对象")
    confidence = payload.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise QwenReviewError(
            "qwen_output_validation",
            "confidence 必须是数字",
        )
    confidence = min(1.0, max(0.0, float(confidence)))

    if task_type == "SHORTAGE":
        name = _validated_name(
            payload.get("shortage_product_name"),
            expected_names or candidate_names,
            "shortage_product_name",
        )
        if name is None:
            return None
        return ReviewedFinding(
            region_index=region_index,
            confidence=confidence,
            shortage_product_name=name,
        )

    if misplaced_stage == "misplaced_product":
        misplaced_name = _validated_name(
            payload.get("misplaced_product_name"),
            candidate_names,
            "misplaced_product_name",
        )
        if misplaced_name is None:
            return None
        return ReviewedFinding(
            region_index=region_index,
            confidence=confidence,
            misplaced_product_name=misplaced_name,
        )
    if misplaced_stage == "expected_product":
        gt_name = _validated_name(
            payload.get("gt_product_name"),
            expected_names or candidate_names,
            "gt_product_name",
        )
        if gt_name is None:
            return None
        return ReviewedFinding(
            region_index=region_index,
            confidence=confidence,
            gt_product_name=gt_name,
        )

    misplaced_name = _validated_name(
        payload.get("misplaced_product_name"),
        candidate_names,
        "misplaced_product_name",
    )
    gt_name = _validated_name(
        payload.get("gt_product_name"),
        expected_names or candidate_names,
        "gt_product_name",
    )
    if misplaced_name is None or gt_name is None or misplaced_name == gt_name:
        return None
    return ReviewedFinding(
        region_index=region_index,
        confidence=confidence,
        misplaced_product_name=misplaced_name,
        gt_product_name=gt_name,
    )


def combine_misplaced_stage_findings(
    misplaced: ReviewedFinding | None,
    expected: ReviewedFinding | None,
    *,
    region_index: int,
) -> ReviewedFinding | None:
    """Join independently validated MISPLACED stages into one public finding."""

    if misplaced is None or expected is None:
        return None
    misplaced_name = misplaced.misplaced_product_name
    gt_name = expected.gt_product_name
    if not misplaced_name or not gt_name or misplaced_name == gt_name:
        return None
    return ReviewedFinding(
        region_index=region_index,
        confidence=min(misplaced.confidence, expected.confidence),
        misplaced_product_name=misplaced_name,
        gt_product_name=gt_name,
    )


def _validated_name(value: Any, candidates: set[str], field: str) -> str | None:
    if not isinstance(value, str):
        raise QwenReviewError("qwen_output_validation", f"{field} 必须是字符串")
    name = value.strip()
    if name.upper() in UNKNOWN_NAMES or name in UNKNOWN_NAMES:
        return None
    if name not in candidates:
        raise QwenReviewError(
            "qwen_output_validation",
            f"{field} 不在候选 SKU 中: {name}",
        )
    return name


def _candidate_rows_text(rows: Sequence[Sequence[dict[str, str]]]) -> str:
    lines = ["候选商品按画面货架层从上到下分组："]
    for row_number, row in enumerate(rows, start=1):
        names = "、".join(item["name"] for item in row) or "（空）"
        lines.append(f"第 {row_number} 行：{names}")
    lines.append("所有输出商品名必须从以上名称中逐字选择。")
    return "\n".join(lines)


def _candidate_names_text(candidates: Sequence[CandidateProduct]) -> str:
    names = "、".join(candidate.name for candidate in candidates)
    return (
        f"候选商品：{names or '（空）'}\n"
        "所有输出商品名必须从以上名称中逐字选择。"
    )


def _candidate_number_mapping_text(
    candidates: Sequence[CandidateProduct],
) -> str:
    lines = ["候选 SKU 编号（与下方标准图拼图上方数字一致）："]
    lines.extend(
        f"SKU {candidate_number}: {candidate.name}"
        for candidate_number, candidate in enumerate(candidates, start=1)
    )
    lines.append("所有输出商品名必须从以上名称中逐字选择。")
    return "\n".join(lines)


def _expected_candidate_number_mapping_text(
    candidates: Sequence[CandidateProduct],
) -> str:
    lines = [
        "这一层从左到右SKU标准放置编号如下"
        "（与下方标准图拼图上方数字一致）："
    ]
    lines.extend(
        f"SKU {candidate_number}: {candidate.name}"
        for candidate_number, candidate in enumerate(candidates, start=1)
    )
    lines.append("输出商品名必须从以上名称中逐字选择。")
    return "\n".join(lines)


def build_candidate_contact_sheets(
    candidates: Sequence[CandidateProduct],
) -> list[CandidateContactSheet]:
    """Pack MISPLACED reference images into numbered, bounded-size grids."""

    max_candidates_per_sheet = CONTACT_SHEET_COLUMNS * CONTACT_SHEET_MAX_ROWS
    sheets: list[CandidateContactSheet] = []
    for chunk_start in range(0, len(candidates), max_candidates_per_sheet):
        chunk = candidates[chunk_start : chunk_start + max_candidates_per_sheet]
        column_count = min(CONTACT_SHEET_COLUMNS, len(chunk))
        row_count = (len(chunk) + column_count - 1) // column_count
        cell_height = CONTACT_SHEET_LABEL_HEIGHT + CONTACT_SHEET_IMAGE_HEIGHT
        canvas_width = (
            CONTACT_SHEET_MARGIN * 2
            + column_count * CONTACT_SHEET_TILE_WIDTH
            + (column_count - 1) * CONTACT_SHEET_GAP
        )
        canvas_height = (
            CONTACT_SHEET_MARGIN * 2
            + row_count * cell_height
            + (row_count - 1) * CONTACT_SHEET_GAP
        )
        canvas = np.full(
            (canvas_height, canvas_width, 3),
            246,
            dtype=np.uint8,
        )

        for chunk_index, candidate in enumerate(chunk):
            candidate_number = chunk_start + chunk_index + 1
            row_index, column_index = divmod(
                chunk_index,
                CONTACT_SHEET_COLUMNS,
            )
            left = (
                CONTACT_SHEET_MARGIN
                + column_index
                * (CONTACT_SHEET_TILE_WIDTH + CONTACT_SHEET_GAP)
            )
            top = (
                CONTACT_SHEET_MARGIN
                + row_index * (cell_height + CONTACT_SHEET_GAP)
            )
            right = left + CONTACT_SHEET_TILE_WIDTH
            bottom = top + cell_height
            image_top = top + CONTACT_SHEET_LABEL_HEIGHT

            cv2.rectangle(
                canvas,
                (left, top),
                (right - 1, image_top - 1),
                (230, 238, 247),
                -1,
            )
            cv2.rectangle(
                canvas,
                (left, image_top),
                (right - 1, bottom - 1),
                (255, 255, 255),
                -1,
            )
            cv2.rectangle(
                canvas,
                (left, top),
                (right - 1, bottom - 1),
                (90, 100, 112),
                1,
            )

            label = str(candidate_number)
            (label_width, label_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                2,
            )
            cv2.putText(
                canvas,
                label,
                (
                    left + (CONTACT_SHEET_TILE_WIDTH - label_width) // 2,
                    top
                    + (CONTACT_SHEET_LABEL_HEIGHT + label_height) // 2
                    - 2,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (25, 31, 38),
                2,
                cv2.LINE_AA,
            )

            decoded = cv2.imdecode(
                np.frombuffer(candidate.image, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if decoded is None:
                raise QwenReviewError(
                    "payload_assembly",
                    f"候选商品 {candidate.name} 的标准图无法生成拼图",
                )
            source_height, source_width = decoded.shape[:2]
            available_width = CONTACT_SHEET_TILE_WIDTH - 12
            available_height = CONTACT_SHEET_IMAGE_HEIGHT - 12
            scale = min(
                available_width / source_width,
                available_height / source_height,
            )
            resized_width = max(1, round(source_width * scale))
            resized_height = max(1, round(source_height * scale))
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            resized = cv2.resize(
                decoded,
                (resized_width, resized_height),
                interpolation=interpolation,
            )
            image_left = left + (CONTACT_SHEET_TILE_WIDTH - resized_width) // 2
            target_top = image_top + (
                CONTACT_SHEET_IMAGE_HEIGHT - resized_height
            ) // 2
            canvas[
                target_top : target_top + resized_height,
                image_left : image_left + resized_width,
            ] = resized

        success, encoded = cv2.imencode(
            ".jpg",
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        if not success:
            raise QwenReviewError("payload_assembly", "无法编码候选 SKU 拼图")
        sheets.append(
            CandidateContactSheet(
                first_candidate_number=chunk_start + 1,
                last_candidate_number=chunk_start + len(chunk),
                image=encoded.tobytes(),
            )
        )
    return sheets


def _crop(
    image: np.ndarray,
    bbox: Sequence[int],
    *,
    x_scale: float,
    y_scale: float,
    max_y_padding: int | None = None,
    row_bbox: Sequence[int] | None = None,
    row_context: int = 12,
) -> np.ndarray:
    x, y, width, height = bbox
    x_padding = round(width * x_scale)
    y_padding = round(height * y_scale)
    if max_y_padding is not None:
        y_padding = min(y_padding, max_y_padding)
    left = max(0, x - x_padding)
    top = max(0, y - y_padding)
    right = min(image.shape[1], x + width + x_padding)
    bottom = min(image.shape[0], y + height + y_padding)
    if row_bbox is not None:
        if len(row_bbox) != 4:
            raise ValueError("row_bbox must be [x, y, width, height]")
        _, row_y, _, row_height = (int(value) for value in row_bbox)
        top = max(top, row_y - row_context)
        bottom = min(bottom, row_y + row_height + row_context)
        if bottom <= top:
            raise ValueError("row_bbox does not overlap the review crop")
    return image[top:bottom, left:right]


def crop_review_region(
    image: np.ndarray,
    bbox: Sequence[int],
    task_type: TaskType,
    *,
    row_bbox: Sequence[int] | None = None,
    pose_type: PoseType = "",
) -> np.ndarray:
    """Expand a detector bbox into the local image actually sent to Qwen."""

    if task_type == "SHORTAGE":
        return _crop(
            image,
            bbox,
            x_scale=0.3,
            y_scale=1.5,
            max_y_padding=100,
            row_bbox=row_bbox,
        )
    if task_type == "MISPLACED":
        if row_bbox is not None:
            if len(row_bbox) != 4:
                raise ValueError("row_bbox must be [x, y, width, height]")
            x, _, width, _ = _normalize_bbox(bbox)
            horizontal_padding = round(width * 0.5)
            left = max(0, x - horizontal_padding)
            right = min(image.shape[1], x + width + horizontal_padding)
            top, bottom, effective_row_top, row_bottom = _misplaced_row_vertical_bounds(
                image.shape[0],
                bbox,
                row_bbox,
                pose_type=pose_type,
            )
            if right <= left or bottom <= top:
                raise ValueError("row_bbox does not define a visible misplaced crop")
            region_image = image[top:bottom, left:right].copy()
            box_left = max(0, x - left)
            box_right = min(region_image.shape[1] - 1, x + width - left)
            box_top = max(0, effective_row_top - top)
            box_bottom = min(region_image.shape[0] - 1, row_bottom - top)
            if box_right > box_left and box_bottom > box_top:
                # Preserve the detector's exact horizontal span and make its
                # target column unambiguous within the detected shelf row.
                cv2.rectangle(
                    region_image,
                    (box_left, box_top),
                    (box_right, box_bottom),
                    (0, 0, 255),
                    4,
                )
            return region_image
        return _crop(
            image,
            bbox,
            x_scale=0.5,
            y_scale=0.5,
            max_y_padding=80,
        )
    raise ValueError("task_type must be SHORTAGE or MISPLACED")


def crop_expected_reference_region(
    image: np.ndarray,
    bbox: Sequence[int],
    *,
    row_bbox: Sequence[int] | None,
    pose_type: PoseType = "",
    row_context: int = 12,
) -> np.ndarray:
    """Crop the complete baseline shelf row and mark the target bbox."""

    x, y, width, height = _normalize_bbox(bbox)
    image_height, image_width = image.shape[:2]
    if row_bbox is not None:
        if len(row_bbox) != 4:
            raise ValueError("row_bbox must be [x, y, width, height]")
        row_x, _, row_width, _ = (int(value) for value in row_bbox)
        top, bottom, effective_row_top, row_bottom = (
            _misplaced_row_vertical_bounds(
                image_height,
                bbox,
                row_bbox,
                pose_type=pose_type,
                row_context=row_context,
            )
        )
        left = max(0, row_x)
        right = min(image_width, row_x + row_width)
    else:
        left = 0
        top = max(0, y - height)
        right = image_width
        bottom = min(image_height, y + height * 2)
    if right <= left or bottom <= top:
        raise ValueError("row_bbox does not define a visible shelf row")

    row_image = image[top:bottom, left:right].copy()
    box_left = max(0, x - left)
    box_right = min(row_image.shape[1] - 1, x + width - left)
    if row_bbox is not None:
        # Keep the detector's horizontal target column, but extend it from the
        # upper shelf boundary to the lower shelf boundary of this row.
        box_top = max(0, effective_row_top - top)
        box_bottom = min(
            row_image.shape[0] - 1,
            row_bottom - top,
        )
    else:
        box_top = max(0, y - top)
        box_bottom = min(row_image.shape[0] - 1, y + height - top)
    if box_right > box_left and box_bottom > box_top:
        cv2.rectangle(
            row_image,
            (box_left, box_top),
            (box_right, box_bottom),
            (0, 0, 255),
            4,
        )
    return row_image


def build_expected_product_reference_image(
    baseline: np.ndarray,
    bbox: Sequence[int],
    *,
    row_bbox: Sequence[int] | None,
    pose_type: PoseType = "",
) -> np.ndarray:
    """Stack the marked standard row above its unmarked bbox cutout."""

    row_image = crop_expected_reference_region(
        baseline,
        bbox,
        row_bbox=row_bbox,
        pose_type=pose_type,
    )
    x, y, width, height = _normalize_bbox(bbox)
    if row_bbox is not None:
        _, _, effective_row_top, row_bottom = _misplaced_row_vertical_bounds(
            baseline.shape[0],
            bbox,
            row_bbox,
            pose_type=pose_type,
        )
        crop_top = max(0, effective_row_top)
        crop_bottom = min(baseline.shape[0], row_bottom)
    else:
        crop_top = max(0, y)
        crop_bottom = min(baseline.shape[0], y + height)
    crop_left = max(0, x)
    crop_right = min(baseline.shape[1], x + width)
    if crop_right <= crop_left or crop_bottom <= crop_top:
        raise ValueError("bbox does not define a visible expected-product cutout")

    cutout = baseline[crop_top:crop_bottom, crop_left:crop_right].copy()
    lower_panel = np.full(
        (cutout.shape[0] + 24, row_image.shape[1], 3),
        (245, 245, 245),
        dtype=np.uint8,
    )
    paste_x = max(0, (lower_panel.shape[1] - cutout.shape[1]) // 2)
    lower_panel[12 : 12 + cutout.shape[0], paste_x : paste_x + cutout.shape[1]] = cutout
    divider = np.full(
        (8, row_image.shape[1], 3),
        (210, 210, 210),
        dtype=np.uint8,
    )
    return np.vstack((row_image, divider, lower_panel))


def _misplaced_row_vertical_bounds(
    image_height: int,
    bbox: Sequence[int],
    row_bbox: Sequence[int],
    *,
    pose_type: PoseType,
    row_context: int = 12,
) -> tuple[int, int, int, int]:
    """Return crop and red-box bounds for one misplaced shelf row.

    The upper camera's first detected row starts at y=0 because there is no
    rail above it.  Using that synthetic boundary would include all ceiling
    space.  For only that row, start one detector-box height above the target.
    """

    _, target_y, _, target_height = _normalize_bbox(bbox)
    _, row_y, _, row_height = (int(value) for value in row_bbox)
    effective_row_top = row_y
    if pose_type == "SHELF_VIEW_UPPER" and row_y <= 0:
        effective_row_top = max(row_y, target_y - target_height)
    row_bottom = row_y + row_height
    top = max(0, effective_row_top - row_context)
    bottom = min(image_height, row_bottom + row_context)
    return top, bottom, effective_row_top, row_bottom


def _numpy_image_content(image: np.ndarray) -> dict[str, Any]:
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    if not success:
        raise QwenReviewError("payload_assembly", "无法编码审核图片")
    return _bytes_image_content(encoded.tobytes(), "image/jpeg")


def _bytes_image_content(image: bytes, media_type: str) -> dict[str, Any]:
    encoded = base64.b64encode(image).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
    }


def normalize_reference_image(
    image: bytes,
    media_type: str,
    *,
    max_side: int = MAX_REFERENCE_IMAGE_SIDE,
) -> tuple[bytes, str]:
    """Downscale oversized SKU references before embedding them as base64."""

    decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise QwenReviewError("candidate_image_download", "候选商品标准图无法解码")
    height, width = decoded.shape[:2]
    longest_side = max(width, height)
    if longest_side <= max_side:
        return image, media_type

    scale = max_side / longest_side
    resized = cv2.resize(
        decoded,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    success, encoded = cv2.imencode(
        ".jpg",
        resized,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    if not success:
        raise QwenReviewError("candidate_image_download", "候选商品标准图缩放失败")
    return encoded.tobytes(), "image/jpeg"


def _resize_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("review images must be uint8 numpy arrays")
    return cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)


def _normalize_bbox(bbox: Sequence[int]) -> list[int]:
    if len(bbox) != 4:
        raise ValueError("bbox must be [x, y, width, height]")
    x, y, width, height = (int(value) for value in bbox)
    x = min(TARGET_SIZE[0] - 1, max(0, x))
    y = min(TARGET_SIZE[1] - 1, max(0, y))
    width = min(TARGET_SIZE[0] - x, max(1, width))
    height = min(TARGET_SIZE[1] - y, max(1, height))
    return [x, y, width, height]


def _chat_completions_url(url: str) -> str:
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    return endpoint
