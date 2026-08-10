"""Use shelf candidates and localized change regions to ask Qwen for SKU names."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
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


TaskType = Literal["SHORTAGE", "MISPLACED"]
PoseType = Literal["", "SHELF_VIEW_UPPER", "SHELF_VIEW_LOWER"]
TARGET_SIZE = (1280, 720)
SKU_TIMEOUT_SECONDS = 8.0
QWEN_TIMEOUT_SECONDS = 120.0
MAX_REFERENCE_IMAGE_BYTES = 20 * 1024 * 1024
MAX_REFERENCE_IMAGE_SIDE = 1024
UNKNOWN_NAMES = {"", "UNKNOWN", "无法确认", "不确定"}
logger = logging.getLogger("uvicorn.error")
PROMPT_ROOT = Path(__file__).resolve().parent
PROMPT_PATHS: dict[TaskType, Path] = {
    "SHORTAGE": PROMPT_ROOT / "shortage_prompt.txt",
    "MISPLACED": PROMPT_ROOT / "misplaced_prompt.txt",
}
DEFAULT_DEBUG_ROOT = Path(
    os.getenv("INSPECT_QWEN_DEBUG_DIR", str(PROMPT_ROOT / "debug"))
)


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
        api_key: str | None = None,
        session: Any | None = None,
        debug_root: str | Path | None = None,
    ) -> None:
        self.sku_base_url = sku_base_url.rstrip("/")
        self.qwen_url = _chat_completions_url(qwen_url)
        self.qwen_model = qwen_model
        self.sku_timeout = sku_timeout
        self.qwen_timeout = qwen_timeout
        self.api_key = api_key or os.getenv("QWEN_API_KEY", "").strip() or None
        self.session = session or requests.Session()
        self.debug_root = Path(debug_root) if debug_root is not None else None

    def review(
        self,
        *,
        task_type: TaskType,
        location_id: str,
        pose_type: PoseType,
        current: np.ndarray,
        bboxes: Sequence[Sequence[int]],
    ) -> QwenReviewResult:
        if task_type not in {"SHORTAGE", "MISPLACED"}:
            raise ValueError("task_type must be SHORTAGE or MISPLACED")
        if not bboxes:
            return QwenReviewResult((), "", ())

        current = _resize_image(current)
        normalized_bboxes = [_normalize_bbox(bbox) for bbox in bboxes]
        debug_directory = self._create_debug_directory(
            task_type,
            location_id,
            pose_type,
            normalized_bboxes,
        )
        rows = self._fetch_candidate_rows(location_id, pose_type)
        candidates = self._fetch_candidate_images(rows)
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
                },
            )
        findings: list[ReviewedFinding] = []
        raw_responses: list[str] = []
        # Intentionally review each bbox in an independent Qwen request. This
        # keeps the model's task local and avoids cross-region bookkeeping.
        for region_index, bbox in enumerate(normalized_bboxes, start=1):
            region_image = crop_review_region(current, bbox, task_type)
            payload = build_qwen_payload(
                task_type=task_type,
                location_id=location_id,
                pose_type=pose_type,
                region_image=region_image,
                candidate_rows=rows,
                candidates=candidates,
                model=self.qwen_model,
            )
            region_directory = (
                debug_directory / f"region_{region_index:02d}"
                if debug_directory is not None
                else None
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
                region_index=region_index,
            )
            if finding is not None:
                findings.append(finding)
            if region_directory is not None:
                _write_json(
                    region_directory / "parsed_result.json",
                    _reviewed_finding_dict(finding),
                )
        if debug_directory is not None:
            _write_json(
                debug_directory / "result.json",
                {
                    "findings": [
                        _reviewed_finding_dict(finding) for finding in findings
                    ],
                    "raw_response_count": len(raw_responses),
                },
            )
        return QwenReviewResult(
            findings=tuple(findings),
            raw_response=json.dumps(raw_responses, ensure_ascii=False),
            candidate_names=tuple(candidate.name for candidate in candidates),
            debug_directory=debug_directory,
        )

    def _create_debug_directory(
        self,
        task_type: TaskType,
        location_id: str,
        pose_type: PoseType,
        bboxes: Sequence[Sequence[int]],
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
    ) -> list[CandidateProduct]:
        row_numbers_by_name: dict[str, list[int]] = {}
        sku_by_name: dict[str, str] = {}
        ordered_names: list[str] = []
        for row_number, row in enumerate(rows, start=1):
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
            image = image_response.content
            if not image or len(image) > MAX_REFERENCE_IMAGE_BYTES:
                raise QwenReviewError(
                    "candidate_image_download",
                    f"商品 {name} 的标准图片为空或过大",
                )
            media_type = image_response.headers.get("Content-Type", "image/jpeg")
            media_type = media_type.split(";", 1)[0]
            image, media_type = normalize_reference_image(image, media_type)
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
        try:
            response = self.session.request(method, url, **kwargs)
        except requests.RequestException as error:
            raise QwenReviewError(stage, f"无法连接上游服务: {error}") from error
        if not response.ok:
            raise QwenReviewError(
                stage,
                f"上游服务返回 HTTP {response.status_code}: {response.text[:300]}",
            )
        return response


def build_qwen_payload(
    *,
    task_type: TaskType,
    location_id: str,
    pose_type: PoseType,
    region_image: np.ndarray,
    candidate_rows: Sequence[Sequence[dict[str, str]]],
    candidates: Sequence[CandidateProduct],
    model: str,
) -> dict[str, Any]:
    if task_type == "SHORTAGE":
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "请只审核下面这一张扩展后的货架局部图："},
            _numpy_image_content(region_image),
            {"type": "text", "text": _candidate_names_text(candidates)},
        ]
    else:
        content = [
            {
                "type": "text",
                "text": (
                    f"任务={task_type}，location_id={location_id}，pose_type={pose_type!r}。"
                    "请只审核下面这一张扩展后的货架局部图。"
                ),
            },
            {"type": "text", "text": _candidate_rows_text(candidate_rows)},
            {
                "type": "text",
                "text": "扩展后的当前货架局部图；重点查看中心商品及左右相邻商品：",
            },
            _numpy_image_content(region_image),
        ]

    content.append({"type": "text", "text": "下面是候选 SKU 标准图："})
    for candidate_index, candidate in enumerate(candidates, start=1):
        if task_type == "SHORTAGE":
            candidate_label = f"CANDIDATE {candidate_index}: {candidate.name};"
        else:
            rows = ",".join(str(value) for value in candidate.row_numbers)
            candidate_label = (
                f"CANDIDATE {candidate_index}: sku_id={candidate.sku_id}; "
                f"name={candidate.name}; 可见行序号={rows}"
            )
        content.extend(
            [
                {
                    "type": "text",
                    "text": candidate_label,
                },
                _bytes_image_content(candidate.image, candidate.media_type),
            ]
        )
    content.append(
        {
            "type": "text",
            "text": "请按系统消息规定的简单 JSON 格式返回这一张局部图的结果。",
        }
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": load_system_prompt(task_type)},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": 800,
    }


def load_system_prompt(task_type: TaskType) -> str:
    try:
        path = PROMPT_PATHS[task_type]
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


def parse_qwen_review(
    content: str,
    *,
    task_type: TaskType,
    candidate_names: set[str],
    region_index: int,
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
            candidate_names,
            "shortage_product_name",
        )
        if name is None:
            return None
        return ReviewedFinding(
            region_index=region_index,
            confidence=confidence,
            shortage_product_name=name,
        )

    misplaced_name = _validated_name(
        payload.get("misplaced_product_name"),
        candidate_names,
        "misplaced_product_name",
    )
    gt_name = _validated_name(
        payload.get("gt_product_name"),
        candidate_names,
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


def _crop(
    image: np.ndarray,
    bbox: Sequence[int],
    *,
    x_scale: float,
    y_scale: float,
    max_y_padding: int | None = None,
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
    return image[top:bottom, left:right]


def crop_review_region(
    image: np.ndarray,
    bbox: Sequence[int],
    task_type: TaskType,
) -> np.ndarray:
    """Expand a detector bbox into the local image actually sent to Qwen."""

    if task_type == "SHORTAGE":
        return _crop(
            image,
            bbox,
            x_scale=0.3,
            y_scale=1.5,
            max_y_padding=100,
        )
    if task_type == "MISPLACED":
        return _crop(
            image,
            bbox,
            x_scale=1.0,
            y_scale=0.5,
            max_y_padding=80,
        )
    raise ValueError("task_type must be SHORTAGE or MISPLACED")


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
