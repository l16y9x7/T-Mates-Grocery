"""Receipt parsing service: head-camera frame -> Qwen -> canonical SKU names."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

if __package__ and __package__.startswith("perception."):
    from ..config import (
        QWEN3_URL as CONFIG_QWEN3_URL,
        SKU_API_URL as CONFIG_SKU_API_URL,
        camera_snapshot_url,
    )
else:
    PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
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
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.upstream_status_code = upstream_status_code


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
        )


app = FastAPI(
    title="Receipt Parser",
    version="1.0.0",
    description="Fetch one head-camera frame and return two canonical SKU names.",
)
router = APIRouter()


class ParseReceiptResponse(BaseModel):
    product_names: list[str]


@app.exception_handler(ServiceError)
async def handle_service_error(_: Any, error: ServiceError) -> JSONResponse:
    detail: dict[str, Any] = {
        "type": error.error_type,
        "message": error.message,
    }
    if error.upstream_status_code is not None:
        detail["upstream_status_code"] = error.upstream_status_code
    return JSONResponse(
        status_code=error.status_code,
        content={"error": detail},
    )


@router.post("/perception/parse", response_model=ParseReceiptResponse)
def parse_receipt() -> ParseReceiptResponse:
    """Capture one frame and return exactly two canonical SKU names."""

    settings = Settings.from_env()
    frame = capture_one_frame(settings)
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
        ) from exc
    except (URLError, TimeoutError, socket.timeout) as exc:
        reason = getattr(exc, "reason", exc)
        raise ServiceError(
            502, "camera_connection_error", f"无法连接相机接口：{reason}"
        ) from exc

    if not raw:
        raise ServiceError(502, "camera_response_error", "相机接口返回空图片。")
    if len(raw) > MAX_CAMERA_BYTES:
        raise ServiceError(
            502, "camera_response_error", "相机图片超过 20MB 限制。"
        )
    return raw


def recognize_frame(
    frame: bytes, settings: Settings
) -> list[dict[str, str | None]]:
    """Recognize one receipt from exactly one in-memory image frame."""

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
        "temperature": 0,
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
    return parse_qwen_items(model_content)


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
            502, "qwen_response_error", "Qwen 响应顶层必须是 JSON 对象。"
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
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        raise ServiceError(
            502,
            response_type,
            f"{service_name} 返回 HTTP {exc.code}。",
            upstream_status_code=exc.code,
        ) from exc
    except (URLError, TimeoutError, socket.timeout) as exc:
        reason = getattr(exc, "reason", exc)
        raise ServiceError(
            502, connection_type, f"无法连接 {service_name}：{reason}"
        ) from exc

    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ServiceError(
            502, response_type, f"{service_name} 返回的内容不是有效 UTF-8 JSON。"
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
