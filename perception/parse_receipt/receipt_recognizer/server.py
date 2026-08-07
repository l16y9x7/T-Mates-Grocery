"""FastAPI HTTP service for receipt recognition."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from .config import Settings
from .errors import (
    APIConnectionError,
    APIResponseError,
    ConfigurationError,
    InputFileError,
    ModelOutputError,
    ReceiptRecognizerError,
    SKUConnectionError,
    SKUNotFoundError,
    SKUResponseError,
)
from .media import image_bytes_to_data_url
from .service import ReceiptRecognizer
from .sku_client import SkuLookupClient


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_RECEIPT_FRAMES = 4

app = FastAPI(
    title="Qwen3-VL Receipt Recognizer",
    version="0.1.0",
    description=(
        "Recognize one receipt from one or more clear image frames and "
        "return structured item JSON."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    settings = Settings.from_env()
    return {
        "status": "ok",
        "model": settings.model,
    }


@app.post("/receipt/parse", response_model=None)
async def parse_receipt(
    file: UploadFile | None = File(
        None,
        description="Backward-compatible single receipt image field.",
    ),
    files: list[UploadFile] | None = File(
        None,
        description="One to four frames of the same receipt.",
    ),
    diagnostics: bool = Query(
        False,
        description="Return line-level diagnostics together with items.",
    ),
    max_edge: int = Query(
        2200,
        ge=256,
        le=4096,
        description="Resize longest image edge before sending to Qwen.",
    ),
) -> list[dict[str, Any]] | dict[str, Any] | JSONResponse:
    uploaded_files = _uploaded_receipt_files(file, files)
    if not uploaded_files:
        return _error_response(
            400,
            "missing_file",
            "请上传 file 或 files 字段。",
        )
    if len(uploaded_files) > MAX_RECEIPT_FRAMES:
        return _error_response(
            400,
            "too_many_files",
            f"同一张小票最多上传 {MAX_RECEIPT_FRAMES} 张图片。",
        )

    try:
        settings = Settings.from_env()
        data_urls = []
        for upload in uploaded_files:
            raw = await upload.read(MAX_UPLOAD_BYTES + 1)
            if not raw:
                return _error_response(
                    400,
                    "empty_file",
                    "上传文件不能为空。",
                )
            if len(raw) > MAX_UPLOAD_BYTES:
                return _error_response(
                    413,
                    "file_too_large",
                    "单张图片不能超过 20MB。",
                )
            data_urls.append(image_bytes_to_data_url(raw, max_edge=max_edge))

        recognizer = ReceiptRecognizer(settings)
        result = recognizer.recognize_data_urls(data_urls)
        sku_items = SkuLookupClient(settings).lookup_items(
            result.business_items
        )
    except InputFileError as exc:
        return _error_response(400, "invalid_input", str(exc))
    except ConfigurationError as exc:
        return _error_response(500, "configuration_error", str(exc))
    except APIConnectionError as exc:
        return _error_response(502, "upstream_connection_error", str(exc))
    except APIResponseError as exc:
        return _error_response(
            502,
            "upstream_response_error",
            str(exc),
            upstream_status_code=exc.status_code,
        )
    except ModelOutputError as exc:
        return _error_response(502, "model_output_error", str(exc))
    except SKUConnectionError as exc:
        return _error_response(502, "sku_connection_error", str(exc))
    except SKUNotFoundError as exc:
        return JSONResponse(
            status_code=404,
            content={"error_code": exc.error_code},
        )
    except SKUResponseError as exc:
        return _error_response(
            502,
            "sku_response_error",
            str(exc),
            upstream_status_code=exc.status_code,
        )
    except ReceiptRecognizerError as exc:
        return _error_response(500, "internal_error", str(exc))

    if diagnostics:
        return {
            "items": sku_items,
            "diagnostics": result.diagnostics,
        }
    return sku_items


def _uploaded_receipt_files(
    file: UploadFile | None,
    files: list[UploadFile] | None,
) -> list[UploadFile]:
    uploaded_files: list[UploadFile] = []
    if file is not None:
        uploaded_files.append(file)
    if files:
        uploaded_files.extend(files)
    return uploaded_files


def _error_response(
    status_code: int,
    error_type: str,
    message: str,
    **extra: Any,
) -> JSONResponse:
    error: dict[str, Any] = {
        "type": error_type,
        "message": message,
    }
    for key, value in extra.items():
        if value is not None:
            error[key] = value
    return JSONResponse(
        status_code=status_code,
        content={"error": error},
    )
