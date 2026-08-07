"""Receipt parsing service: camera frame -> Qwen -> SKU locations."""

from __future__ import annotations

import base64
import io
import json
import os
import socket
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_CAMERA_BYTES = 20 * 1024 * 1024
MAX_IMAGE_EDGE = 2200
MAX_FRAMES = 3
DEFAULT_QWEN_BASE_URL = "http://127.0.0.1:8102/v1"
DEFAULT_QWEN_MODEL = "Qwen3-VL-4B-Instruct"
DEFAULT_QWEN_TIMEOUT_SECONDS = 120.0
DEFAULT_SKU_BASE_URL = "http://127.0.0.1:25540"
DEFAULT_SKU_TIMEOUT_SECONDS = 3.0
DEFAULT_SKU_EDIT_DISTANCE_MAX = 3


SYSTEM_PROMPT = """你是零售购物小票商品解析器。
请综合输入的同一张小票的一至三张图片，只输出严格 JSON 数组。

每个商品只能包含两个字段：
- name：完整票面商品名称。
- specification：票面规格原文；没有打印或无法确认时为 null。

规则：
1. 只识别商品明细，忽略店名、日期、时间、数量、价格、金额、折扣、合计和支付方式。
2. 商品名称跨行打印时，按阅读顺序合并为完整名称，不能丢字。
3. name 保留品牌、系列、品类、口味、香型和型号，不拆分 flavor，不改写或猜测。
4. specification 只保留重量、容量、尺寸或包装规格，例如 65g、500ml、60g*10。
5. 不合并小票中的商品行；多张图片中的同一商品不能重复输出。
6. 整张小票无法识别时输出 []。
7. 不输出 Markdown、说明、代码块或任何其他字段。

输出示例：
[{"name":"康师傅香辣牛肉面","specification":"500g"}]
"""

USER_PROMPT = """读取这张购物小票，输出商品名称和规格组成的 JSON 数组。所有图片属于同一张小票。"""


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


class SKUNotFoundError(ServiceError):
    def __init__(self, name: str) -> None:
        super().__init__(404, "sku_not_found", f"SKU 中不存在商品：{name}")


@dataclass(frozen=True)
class Settings:
    camera_url: str
    qwen_base_url: str = DEFAULT_QWEN_BASE_URL
    qwen_model: str = DEFAULT_QWEN_MODEL
    qwen_api_key: str | None = None
    qwen_timeout_seconds: float = DEFAULT_QWEN_TIMEOUT_SECONDS
    sku_base_url: str = DEFAULT_SKU_BASE_URL
    sku_timeout_seconds: float = DEFAULT_SKU_TIMEOUT_SECONDS
    sku_edit_distance_max: int = DEFAULT_SKU_EDIT_DISTANCE_MAX

    @classmethod
    def from_env(cls) -> "Settings":
        camera_url = _required_url("RECEIPT_CAMERA_URL")
        qwen_base_url = _optional_url(
            "QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL
        )
        sku_base_url = _optional_url("SKU_BASE_URL", DEFAULT_SKU_BASE_URL)
        qwen_model = os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL).strip()
        if not qwen_model:
            raise ServiceError(
                500, "configuration_error", "QWEN_MODEL 不能为空。"
            )

        api_key = os.getenv("QWEN_API_KEY")
        return cls(
            camera_url=camera_url,
            qwen_base_url=qwen_base_url.rstrip("/"),
            qwen_model=qwen_model,
            qwen_api_key=api_key.strip() if api_key and api_key.strip() else None,
            qwen_timeout_seconds=_positive_float(
                "QWEN_TIMEOUT_SECONDS", DEFAULT_QWEN_TIMEOUT_SECONDS
            ),
            sku_base_url=sku_base_url.rstrip("/"),
            sku_timeout_seconds=_positive_float(
                "SKU_TIMEOUT_SECONDS", DEFAULT_SKU_TIMEOUT_SECONDS
            ),
            sku_edit_distance_max=_nonnegative_int(
                "SKU_EDIT_DISTANCE_MAX", DEFAULT_SKU_EDIT_DISTANCE_MAX
            ),
        )


app = FastAPI(
    title="Receipt Parser",
    version="1.0.0",
    description="Fetch one camera frame, recognize receipt items, and return SKU locations.",
)


@app.exception_handler(ServiceError)
async def handle_service_error(_: Any, error: ServiceError) -> JSONResponse:
    if isinstance(error, SKUNotFoundError):
        return JSONResponse(
            status_code=404,
            content={"error_code": "SKU_NOT_FOUND"},
        )

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


@app.get("/health")
def health() -> dict[str, str]:
    settings = Settings.from_env()
    return {"status": "ok", "model": settings.qwen_model}


@app.post("/receipt/parse")
def parse_receipt() -> list[dict[str, Any]]:
    """Capture the current receipt frame and return canonical SKU locations."""

    settings = Settings.from_env()
    frame = capture_one_frame(settings)
    recognized_items = recognize_frames([frame], settings)
    if not recognized_items:
        return []
    return lookup_sku_items(recognized_items, settings)


def capture_one_frame(settings: Settings) -> bytes:
    """GET one current camera snapshot without writing it to disk."""

    request = Request(
        settings.camera_url,
        headers={"Accept": "image/jpeg,image/png,image/*"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=settings.qwen_timeout_seconds) as response:
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


def recognize_frames(
    frames: Sequence[bytes], settings: Settings
) -> list[dict[str, str | None]]:
    """Recognize one receipt from one to three in-memory image frames."""

    if not 1 <= len(frames) <= MAX_FRAMES:
        raise ServiceError(
            400,
            "invalid_frame_count",
            f"必须提供 1 至 {MAX_FRAMES} 张同一小票图片。",
        )

    content: list[dict[str, Any]] = [
        {"type": "text", "text": USER_PROMPT}
    ]
    for frame in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_bytes_to_data_url(frame)},
            }
        )

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
) -> list[dict[str, Any]]:
    """Map each Qwen name to one canonical SKU name and its locations."""

    results: list[dict[str, Any]] = []
    for item in items:
        recognized_name = item["name"]
        assert isinstance(recognized_name, str)
        try:
            product = _sku_product_for_name(recognized_name, settings)
        except SKUNotFoundError:
            candidate = _best_sku_name(recognized_name, settings)
            if candidate is None:
                raise SKUNotFoundError(recognized_name)
            product = _sku_product_for_name(candidate, settings)
        results.append(
            {"name": product["name"], "locations": product["locations"]}
        )
    return results


_SKU_NAMES_CACHE: dict[str, list[str]] = {}


def _best_sku_name(name: str, settings: Settings) -> str | None:
    ranked = sorted(
        (_edit_distance(name, sku_name), sku_name)
        for sku_name in _all_sku_names(settings)
    )
    if not ranked or ranked[0][0] > settings.sku_edit_distance_max:
        return None
    return ranked[0][1]


def _all_sku_names(settings: Settings) -> list[str]:
    cached = _SKU_NAMES_CACHE.get(settings.sku_base_url)
    if cached is not None:
        return list(cached)
    value = _request_sku_json("/sku/get_all_names", settings)
    if not isinstance(value, list) or not all(
        isinstance(name, str) for name in value
    ):
        raise ServiceError(
            502, "sku_response_error", "SKU 名称列表必须是字符串数组。"
        )
    names = [name.strip() for name in value if name.strip()]
    _SKU_NAMES_CACHE[settings.sku_base_url] = names
    return list(names)


def _sku_product_for_name(name: str, settings: Settings) -> dict[str, Any]:
    value = _request_sku_json(
        f"/sku/search_by_name?{urlencode({'name': name})}", settings
    )
    if not isinstance(value, dict):
        raise ServiceError(
            502, "sku_response_error", "SKU 查询响应必须是 JSON 对象。"
        )
    product_name = value.get("name")
    locations = value.get("locations")
    if not isinstance(product_name, str) or not product_name.strip():
        raise ServiceError(
            502, "sku_response_error", "SKU 响应缺少有效 name。"
        )
    if (
        not isinstance(locations, list)
        or not locations
        or not all(
            isinstance(location, str) and location
            for location in locations
        )
    ):
        raise ServiceError(
            502, "sku_response_error", "SKU 响应缺少有效 locations。"
        )
    return {"name": product_name.strip(), "locations": locations}


def _request_qwen(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if settings.qwen_api_key:
        headers["Authorization"] = f"Bearer {settings.qwen_api_key}"
    request = Request(
        f"{settings.qwen_base_url}/chat/completions",
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
    try:
        return _read_json(
            request,
            settings.sku_timeout_seconds,
            connection_type="sku_connection_error",
            response_type="sku_response_error",
            service_name="SKU",
        )
    except ServiceError as exc:
        if exc.upstream_status_code == 404:
            name = path.split("name=", 1)[-1]
            raise SKUNotFoundError(name) from exc
        raise


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


def _required_url(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ServiceError(500, "configuration_error", f"缺少环境变量 {name}。")
    return _validate_url(name, value)


def _optional_url(name: str, default: str) -> str:
    return _validate_url(name, os.getenv(name, default).strip())


def _validate_url(name: str, value: str) -> str:
    if not value.startswith(("http://", "https://")):
        raise ServiceError(
            500, "configuration_error", f"{name} 必须以 http:// 或 https:// 开头。"
        )
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ServiceError(
            500, "configuration_error", f"{name} 必须是数字。"
        ) from exc
    if value <= 0:
        raise ServiceError(500, "configuration_error", f"{name} 必须大于 0。")
    return value


def _nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ServiceError(500, "configuration_error", f"{name} 必须是整数。") from exc
    if value < 0:
        raise ServiceError(500, "configuration_error", f"{name} 不能小于 0。")
    return value
