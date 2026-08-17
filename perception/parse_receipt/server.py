"""Receipt parsing service: head-camera frame -> Qwen -> canonical SKU names."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, FastAPI, Request as FastAPIRequest
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]

if __package__ and __package__.startswith("perception."):
    from ..config import (
        QWEN3_URL as CONFIG_QWEN3_URL,
        SKU_API_URL as CONFIG_SKU_API_URL,
        camera_snapshot_url,
    )
else:
    if str(PERCEPTION_ROOT) not in sys.path:
        sys.path.insert(0, str(PERCEPTION_ROOT))
    from config import (
        QWEN3_URL as CONFIG_QWEN3_URL,
        SKU_API_URL as CONFIG_SKU_API_URL,
        camera_snapshot_url,
    )


MAX_CAMERA_BYTES = 20 * 1024 * 1024
MAX_IMAGE_EDGE = 2200
DEFAULT_CAMERA_URL = camera_snapshot_url("head")
DEFAULT_QWEN_BASE_URL = CONFIG_QWEN3_URL
DEFAULT_QWEN_MODEL = "Qwen3-VL-4B-Instruct"
DEFAULT_CAMERA_TIMEOUT_SECONDS = 5.0
DEFAULT_QWEN_TIMEOUT_SECONDS = 120.0
DEFAULT_SKU_BASE_URL = CONFIG_SKU_API_URL
DEFAULT_SKU_TIMEOUT_SECONDS = 3.0
DEFAULT_RECEIPT_CAPTURE_DIR = PERCEPTION_ROOT / "test_data" / "receipt_captures"

# Reuse Uvicorn's configured error logger so diagnostics always reach the same
# terminal/file handler as the HTTP access log.
logger = logging.getLogger("uvicorn.error")

ERROR_CONTEXT = {
    "camera_connection_error": (
        "camera_capture",
        True,
        "检查相机服务是否启动，并确认相机主机、端口和网络可达。",
    ),
    "camera_response_error": (
        "camera_capture",
        True,
        "检查相机接口是否返回非空的 JPEG/PNG 图片。",
    ),
    "image_save_error": (
        "image_persistence",
        True,
        "检查 RECEIPT_CAPTURE_DIR 是否存在可写磁盘空间及目录权限。",
    ),
    "qwen_connection_error": (
        "qwen_recognition",
        True,
        "检查 Qwen 服务地址、端口和网络连接。",
    ),
    "qwen_response_error": (
        "qwen_recognition",
        True,
        "检查 Qwen 服务状态及 OpenAI 兼容响应格式。",
    ),
    "qwen_output_error": (
        "qwen_output_validation",
        False,
        "确认小票画面清晰且包含两个商品，并检查模型原始输出格式。",
    ),
    "sku_connection_error": (
        "sku_lookup",
        True,
        "检查 SKU 服务是否已在配置端口启动。",
    ),
    "sku_response_error": (
        "sku_lookup",
        True,
        "检查 /sku/get_all_names 是否返回非空 JSON 字符串数组。",
    ),
}


SYSTEM_PROMPT = """你是零售购物小票商品解析器，解析图片中小票的文本信息。

小票中有两种商品，每种商品解析两个字段：
- 商品名称：票面商品名称的原文，注意可能有换行。
- 规格：票面商品名称的原文，没有打印或无法确认时为 null。

规则：
1. 商品名称跨行打印时，按阅读顺序合并为完整名称，不能丢字。
2. 不合并小票中的商品行。
3. 确保输出的是长度为2的列表，如果只能识别一行，第二个输出空列表。
4. 不输出 Markdown、说明、代码块或任何其他字段。

输出示例：
[{"name":"商品1","specification":"500g"}, {"name":"商品2","specification":"白桃乌龙味"}]
"""

USER_PROMPT = """解析这张购物小票。"""


class ServiceError(Exception):
    """An expected service failure with an HTTP representation."""

    def __init__(
        self,
        status_code: int,
        error_type: str,
        message: str,
        *,
        upstream_status_code: int | None = None,
        stage: str | None = None,
        upstream: str | None = None,
        retryable: bool | None = None,
        hint: str | None = None,
        elapsed_ms: float | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        default_stage, default_retryable, default_hint = ERROR_CONTEXT.get(
            error_type,
            ("unknown", False, "查看服务端日志和异常堆栈。"),
        )
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.upstream_status_code = upstream_status_code
        self.stage = stage or default_stage
        self.upstream = upstream
        self.retryable = default_retryable if retryable is None else retryable
        self.hint = hint or default_hint
        self.elapsed_ms = elapsed_ms
        self.timeout_seconds = timeout_seconds


@dataclass(frozen=True)
class Settings:
    camera_url: str = DEFAULT_CAMERA_URL
    camera_timeout_seconds: float = DEFAULT_CAMERA_TIMEOUT_SECONDS
    qwen_base_url: str = DEFAULT_QWEN_BASE_URL
    qwen_model: str = DEFAULT_QWEN_MODEL
    qwen_api_key: str | None = None
    qwen_timeout_seconds: float = DEFAULT_QWEN_TIMEOUT_SECONDS
    sku_base_url: str = DEFAULT_SKU_BASE_URL
    sku_timeout_seconds: float = DEFAULT_SKU_TIMEOUT_SECONDS
    receipt_capture_dir: Path = DEFAULT_RECEIPT_CAPTURE_DIR

    @classmethod
    def from_env(cls) -> "Settings":
        camera_url = camera_snapshot_url("head")
        qwen_base_url = (
            os.getenv("QWEN_BASE_URL", "").strip()
            or os.getenv("QWEN3_URL", "").strip()
            or DEFAULT_QWEN_BASE_URL
        )
        sku_base_url = os.getenv("SKU_BASE_URL", "").strip() or DEFAULT_SKU_BASE_URL
        qwen_model = os.getenv("QWEN_MODEL", "").strip() or DEFAULT_QWEN_MODEL
        configured_capture_dir = os.getenv("RECEIPT_CAPTURE_DIR", "").strip()
        receipt_capture_dir = Path(
            configured_capture_dir or DEFAULT_RECEIPT_CAPTURE_DIR
        ).expanduser()
        if not receipt_capture_dir.is_absolute():
            receipt_capture_dir = PERCEPTION_ROOT / receipt_capture_dir

        api_key = os.getenv("QWEN_API_KEY")
        return cls(
            camera_url=camera_url,
            camera_timeout_seconds=float(
                os.getenv("CAMERA_TIMEOUT_SECONDS", DEFAULT_CAMERA_TIMEOUT_SECONDS)
            ),
            qwen_base_url=qwen_base_url.rstrip("/"),
            qwen_model=qwen_model,
            qwen_api_key=api_key.strip() if api_key and api_key.strip() else None,
            qwen_timeout_seconds=float(
                os.getenv("QWEN_TIMEOUT_SECONDS", DEFAULT_QWEN_TIMEOUT_SECONDS)
            ),
            sku_base_url=sku_base_url.rstrip("/"),
            sku_timeout_seconds=float(
                os.getenv("SKU_TIMEOUT_SECONDS", DEFAULT_SKU_TIMEOUT_SECONDS)
            ),
            receipt_capture_dir=receipt_capture_dir,
        )


app = FastAPI(
    title="Receipt Parser",
    version="1.0.0",
    description="Fetch one head-camera frame and return two canonical SKU names.",
)
router = APIRouter()


class ParseReceiptResponse(BaseModel):
    product_names: list[str]


def _request_id(request: FastAPIRequest) -> str:
    supplied = request.headers.get("X-Request-ID", "").strip()
    if supplied and len(supplied) <= 64 and re.fullmatch(r"[A-Za-z0-9._-]+", supplied):
        return supplied
    return uuid.uuid4().hex


def _safe_upstream_url(url: str) -> str:
    """Return an upstream URL without query parameters or credentials."""

    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 1)


@app.exception_handler(ServiceError)
async def handle_service_error(
    request: FastAPIRequest,
    error: ServiceError,
) -> JSONResponse:
    request_id = _request_id(request)
    detail: dict[str, Any] = {
        "type": error.error_type,
        "message": error.message,
        "stage": error.stage,
        "retryable": error.retryable,
        "hint": error.hint,
        "request_id": request_id,
    }
    if error.upstream_status_code is not None:
        detail["upstream_status_code"] = error.upstream_status_code
    if error.upstream is not None:
        detail["upstream"] = error.upstream
    if error.elapsed_ms is not None:
        detail["elapsed_ms"] = error.elapsed_ms
    if error.timeout_seconds is not None:
        detail["timeout_seconds"] = error.timeout_seconds

    client = request.client.host if request.client else "unknown"
    logger.error(
        "request_id=%s client=%s method=%s path=%s status=%s "
        "type=%s stage=%s upstream=%s upstream_status=%s retryable=%s "
        "elapsed_ms=%s timeout_seconds=%s message=%s hint=%s",
        request_id,
        client,
        request.method,
        request.url.path,
        error.status_code,
        error.error_type,
        error.stage,
        error.upstream or "-",
        error.upstream_status_code or "-",
        error.retryable,
        error.elapsed_ms if error.elapsed_ms is not None else "-",
        error.timeout_seconds if error.timeout_seconds is not None else "-",
        error.message,
        error.hint,
    )
    return JSONResponse(
        status_code=error.status_code,
        content={"error": detail},
        headers={"X-Request-ID": request_id},
    )


@router.post("/perception/parse", response_model=ParseReceiptResponse)
def parse_receipt() -> ParseReceiptResponse:
    """Capture one frame and return exactly two canonical SKU names."""

    settings = Settings.from_env()
    frame = capture_one_frame(settings)
    saved_path = save_receipt_frame(frame, settings.receipt_capture_dir)
    logger.info(
        "receipt_image_saved path=%s bytes=%s",
        saved_path,
        len(frame),
    )
    recognized_items = recognize_frame(frame, settings)
    if len(recognized_items) != 2:
        raise ServiceError(
            502,
            "qwen_output_error",
            f"Qwen 必须识别出两个商品，当前得到 {len(recognized_items)} 个。",
        )
    return ParseReceiptResponse(
        product_names=lookup_sku_items(recognized_items, settings)
    )


def capture_one_frame(settings: Settings) -> bytes:
    """GET one current camera snapshot without writing it to disk."""

    started_at = time.perf_counter()
    upstream = _safe_upstream_url(settings.camera_url)
    request = Request(
        settings.camera_url,
        headers={"Accept": "image/jpeg,image/png,image/*"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=settings.camera_timeout_seconds) as response:
            raw = response.read(MAX_CAMERA_BYTES + 1)
    except HTTPError as exc:
        raise ServiceError(
            502,
            "camera_response_error",
            f"相机接口返回 HTTP {exc.code}。",
            upstream_status_code=exc.code,
            upstream=upstream,
            retryable=exc.code >= 500,
            elapsed_ms=_elapsed_ms(started_at),
            timeout_seconds=settings.camera_timeout_seconds,
        ) from exc
    except (URLError, TimeoutError, socket.timeout) as exc:
        reason = getattr(exc, "reason", exc)
        raise ServiceError(
            502,
            "camera_connection_error",
            f"无法连接相机接口：{reason}",
            upstream=upstream,
            elapsed_ms=_elapsed_ms(started_at),
            timeout_seconds=settings.camera_timeout_seconds,
        ) from exc

    if not raw:
        raise ServiceError(
            502,
            "camera_response_error",
            "相机接口返回空图片。",
            upstream=upstream,
            elapsed_ms=_elapsed_ms(started_at),
            timeout_seconds=settings.camera_timeout_seconds,
        )
    if len(raw) > MAX_CAMERA_BYTES:
        raise ServiceError(
            502,
            "camera_response_error",
            "相机图片超过 20MB 限制。",
            upstream=upstream,
            elapsed_ms=_elapsed_ms(started_at),
            timeout_seconds=settings.camera_timeout_seconds,
        )
    return raw


def save_receipt_frame(frame: bytes, capture_root: str | Path) -> Path:
    """Atomically persist one validated camera frame and return its path."""

    try:
        with Image.open(io.BytesIO(frame)) as image:
            image_format = (image.format or "").upper()
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ServiceError(
            502,
            "camera_response_error",
            "相机返回的内容不是有效 JPEG/PNG 图片。",
        ) from exc

    suffix = {"JPEG": ".jpg", "PNG": ".png"}.get(image_format)
    if suffix is None:
        raise ServiceError(
            502,
            "camera_response_error",
            f"相机图片格式不受支持：{image_format or 'unknown'}，仅支持 JPEG/PNG。",
        )

    captured_at = datetime.now(UTC)
    directory = Path(capture_root).expanduser() / captured_at.strftime("%Y-%m-%d")
    filename = (
        f"receipt_{captured_at.strftime('%Y%m%dT%H%M%S_%fZ')}_"
        f"{uuid.uuid4().hex[:8]}{suffix}"
    )
    target = directory / filename
    temporary = directory / f".{filename}.{uuid.uuid4().hex}.tmp"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with temporary.open("xb") as file:
            file.write(frame)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(target)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ServiceError(
            500,
            "image_save_error",
            f"无法保存小票图片：{exc}",
        ) from exc
    return target.resolve()


def recognize_frame(
    frame: bytes, settings: Settings
) -> list[dict[str, str | None]]:
    """Recognize one receipt from exactly one in-memory image frame."""

    return parse_qwen_items(recognize_frame_raw(frame, settings))


def recognize_frame_raw(frame: bytes, settings: Settings) -> str:
    """Return Qwen's unmodified message.content for one receipt image."""

    content: list[dict[str, Any]] = [
        {"type": "text", "text": USER_PROMPT},
        {
            "type": "image_url",
            "image_url": {"url": image_bytes_to_data_url(frame)},
        },
    ]

    payload = {
        "model": settings.qwen_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.3,
        "max_tokens": 512,
    }
    response = _request_qwen(payload, settings)
    try:
        model_content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ServiceError(
            502,
            "qwen_response_error",
            "Qwen 响应缺少 choices[0].message.content。",
        ) from exc
    if not isinstance(model_content, str):
        raise ServiceError(
            502,
            "qwen_response_error",
            "Qwen 的 message.content 不是字符串。",
        )
    return model_content


def image_bytes_to_data_url(raw: bytes) -> str:
    """Normalize an image in memory and encode it as a JPEG data URL."""

    if not raw:
        raise ServiceError(
            502, "camera_response_error", "图片内容为空。"
        )
    try:
        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source)
            image = image.convert("RGB")
            image.thumbnail(
                (MAX_IMAGE_EDGE, MAX_IMAGE_EDGE),
                Image.Resampling.LANCZOS,
            )
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=90, optimize=True)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ServiceError(
            502,
            "camera_response_error",
            "相机返回的内容不是有效 JPEG/PNG 图片。",
        ) from exc

    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def parse_qwen_items(content: str) -> list[dict[str, str | None]]:
    """Validate Qwen's intentionally minimal name/specification array."""

    try:
        value = json.loads(content.strip())
    except json.JSONDecodeError as exc:
        raise ServiceError(
            502, "qwen_output_error", "Qwen 输出不是严格 JSON 数组。"
        ) from exc
    if not isinstance(value, list):
        raise ServiceError(
            502, "qwen_output_error", "Qwen 输出顶层必须是 JSON 数组。"
        )

    items: list[dict[str, str | None]] = []
    for index, item in enumerate(value):
        label = f"items[{index}]"
        if item == []:
            # Prompt 约定：只识别出一行时，第二项可以是空列表。
            continue
        if not isinstance(item, dict) or set(item) != {"name", "specification"}:
            raise ServiceError(
                502,
                "qwen_output_error",
                f"{label} 只能包含 name 和 specification。",
            )
        name = item["name"]
        specification = item["specification"]
        if not isinstance(name, str) or not name.strip():
            raise ServiceError(
                502, "qwen_output_error", f"{label}.name 必须是非空字符串。"
            )
        if specification is not None and (
            not isinstance(specification, str) or not specification.strip()
        ):
            raise ServiceError(
                502,
                "qwen_output_error",
                f"{label}.specification 必须是非空字符串或 null。",
            )
        items.append(
            {
                "name": name.strip(),
                "specification": (
                    specification.strip()
                    if isinstance(specification, str)
                    else None
                ),
            }
        )
    return items


def lookup_sku_items(
    items: Sequence[dict[str, str | None]], settings: Settings
) -> list[str]:
    """Match Qwen items against the complete SKU name list in priority order."""

    sku_names = _all_sku_names(settings)
    if not sku_names:
        raise ServiceError(502, "sku_response_error", "SKU 名称列表为空。")
    results: list[str] = []
    for item in items:
        recognized_name = item["name"]
        assert isinstance(recognized_name, str)
        matched_name = match_sku_name(
            recognized_name,
            item.get("specification"),
            sku_names,
        )
        results.append(matched_name)
    return results


_NUMERIC_UNIT_SPECIFICATION = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*"
    r"(?:ml|毫升|l|升|g|克|kg|千克|斤|两|mm|毫米|cm|厘米|m|米|oz|盎司|"
    r"片|包|袋|盒|瓶|罐|支|个|枚|卷|抽)"
    r"(?:\s*[x×*]\s*\d+(?:\s*(?:片|包|袋|盒|瓶|罐|支|个|枚|卷|抽))?)?\s*$",
    re.IGNORECASE,
)


def specification_for_matching(specification: str | None) -> str:
    """Exclude pure numeric-unit specifications such as 500ml or 55g."""

    if not isinstance(specification, str):
        return ""
    normalized = specification.strip()
    if not normalized or _NUMERIC_UNIT_SPECIFICATION.fullmatch(normalized):
        return ""
    return normalized


def match_sku_name(
    name: str,
    specification: str | None,
    sku_names: Sequence[str],
) -> str:
    """Apply exact name, exact name+specification, then nearest edit distance."""

    normalized_name = name.strip()
    available_names = [sku_name.strip() for sku_name in sku_names if sku_name.strip()]
    if not available_names:
        raise ServiceError(502, "sku_response_error", "SKU 名称列表为空。")

    # Priority 1: the recognized name itself is already a canonical SKU name.
    if normalized_name in available_names:
        return normalized_name

    # Priority 2: append a meaningful flavor/style specification, but ignore
    # simple size specifications such as 500ml, 55g, or 2盒.
    effective_specification = specification_for_matching(specification)
    combined_name = normalized_name + effective_specification
    if combined_name in available_names:
        return combined_name

    # Priority 3: always choose the shortest edit distance using the same
    # specification rule. Lexicographic name is a deterministic tie breaker.
    return min(
        available_names,
        key=lambda sku_name: (_edit_distance(combined_name, sku_name), sku_name),
    )


def _all_sku_names(settings: Settings) -> list[str]:
    value = _request_sku_json("/sku/get_all_names", settings)
    if not isinstance(value, list) or not all(
        isinstance(name, str) for name in value
    ):
        raise ServiceError(
            502, "sku_response_error", "SKU 名称列表必须是字符串数组。"
        )
    names = [name.strip() for name in value if name.strip()]
    return names


def _request_qwen(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if settings.qwen_api_key:
        headers["Authorization"] = f"Bearer {settings.qwen_api_key}"
    qwen_url = settings.qwen_base_url.rstrip("/")
    if not qwen_url.endswith("/chat/completions"):
        qwen_url += "/chat/completions"
    request = Request(
        qwen_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    value = _read_json(
        request,
        settings.qwen_timeout_seconds,
        connection_type="qwen_connection_error",
        response_type="qwen_response_error",
        service_name="Qwen",
    )
    if not isinstance(value, dict):
        raise ServiceError(
            502,
            "qwen_response_error",
            "Qwen 响应顶层必须是 JSON 对象。",
            upstream=_safe_upstream_url(qwen_url),
        )
    return value


def _request_sku_json(path: str, settings: Settings) -> Any:
    request = Request(
        f"{settings.sku_base_url}{path}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    return _read_json(
        request,
        settings.sku_timeout_seconds,
        connection_type="sku_connection_error",
        response_type="sku_response_error",
        service_name="SKU",
    )


def _read_json(
    request: Request,
    timeout: float,
    *,
    connection_type: str,
    response_type: str,
    service_name: str,
) -> Any:
    started_at = time.perf_counter()
    upstream = _safe_upstream_url(request.full_url)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        raise ServiceError(
            502,
            response_type,
            f"{service_name} 返回 HTTP {exc.code}。",
            upstream_status_code=exc.code,
            upstream=upstream,
            retryable=exc.code >= 500,
            elapsed_ms=_elapsed_ms(started_at),
            timeout_seconds=timeout,
        ) from exc
    except (URLError, TimeoutError, socket.timeout) as exc:
        reason = getattr(exc, "reason", exc)
        raise ServiceError(
            502,
            connection_type,
            f"无法连接 {service_name}：{reason}",
            upstream=upstream,
            elapsed_ms=_elapsed_ms(started_at),
            timeout_seconds=timeout,
        ) from exc

    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServiceError(
            502,
            response_type,
            f"{service_name} 返回的内容不是有效 UTF-8 JSON。",
            upstream=upstream,
            retryable=False,
            elapsed_ms=_elapsed_ms(started_at),
            timeout_seconds=timeout,
        ) from exc


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


# Keep a standalone app for isolated development and unit tests. Production
# registers the router on perception/pick/locate/main.py's shared 8083 app.
app.include_router(router)
