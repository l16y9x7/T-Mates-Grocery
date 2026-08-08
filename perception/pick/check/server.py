"""Wrist-camera product check API."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent
if __package__ and __package__.startswith("perception."):
    from ...config import (
        QWEN3_MODEL,
        QWEN3_URL,
        SKU_API_URL,
        hand_camera_snapshot_url,
    )
else:
    PERCEPTION_ROOT = ROOT.parents[1]
    if str(PERCEPTION_ROOT) not in sys.path:
        sys.path.insert(0, str(PERCEPTION_ROOT))
    from config import (
        QWEN3_MODEL,
        QWEN3_URL,
        SKU_API_URL,
        hand_camera_snapshot_url,
    )


PROMPT_PATH = ROOT / "prompt.txt"
CAMERA_TIMEOUT_SECONDS = 5
SKU_TIMEOUT_SECONDS = 5
QWEN_TIMEOUT_SECONDS = 120

app = FastAPI(title="Pick Check", version="1.0.0")
router = APIRouter()


class PickCheckRequest(BaseModel):
    task_type: str
    product_name: str
    hand: str


class PickCheckResponse(BaseModel):
    pick_status: str


def load_prompt(request: PickCheckRequest) -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").format(
        product_name=request.product_name
    )


def fetch_hand_image(hand: str) -> tuple[bytes, str]:
    try:
        camera_url = hand_camera_snapshot_url(hand)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="hand 只能是 left 或 right") from error

    try:
        response = requests.get(camera_url, timeout=CAMERA_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as error:
        raise HTTPException(status_code=400, detail=f"腕部相机读取失败: {error}") from error
    if not response.content:
        raise HTTPException(status_code=400, detail="腕部相机返回空图片")

    media_type = response.headers.get("Content-Type", "image/jpeg").split(";", 1)[0]
    return response.content, media_type


def fetch_reference_image(product_name: str) -> tuple[bytes, str]:
    try:
        product_response = requests.get(
            f"{SKU_API_URL}/sku/search_by_name",
            params={"name": product_name},
            timeout=SKU_TIMEOUT_SECONDS,
        )
        product_response.raise_for_status()
        product = product_response.json()
        image_paths = product["images"]
        image_path = image_paths[0]
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("SKU 参考图路径为空")

        image_response = requests.get(
            f"{SKU_API_URL}/{quote(image_path, safe='/')}",
            timeout=SKU_TIMEOUT_SECONDS,
        )
        image_response.raise_for_status()
    except (
        requests.RequestException,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as error:
        raise HTTPException(status_code=502, detail=f"SKU 参考图读取失败: {error}") from error

    if not image_response.content:
        raise HTTPException(status_code=502, detail="SKU 参考图为空")
    media_type = image_response.headers.get("Content-Type", "image/jpeg").split(
        ";", 1
    )[0]
    return image_response.content, media_type


def image_content(image: bytes, media_type: str) -> dict[str, Any]:
    encoded = base64.b64encode(image).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
    }


def call_qwen(
    prompt: str,
    reference_image: bytes,
    reference_media_type: str,
    hand_image: bytes,
    hand_media_type: str,
) -> PickCheckResponse:
    try:
        response = requests.post(
            QWEN3_URL,
            json={
                "model": QWEN3_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": "标准 SKU 参考图："},
                            image_content(reference_image, reference_media_type),
                            {"type": "text", "text": "腕部相机待校验图："},
                            image_content(hand_image, hand_media_type),
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 32,
            },
            timeout=QWEN_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        content = payload["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
        raise HTTPException(status_code=502, detail=f"Qwen3 校验失败: {error}") from error

    try:
        result = json.loads(str(content).strip())
        pick_status = result["pick_status"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise HTTPException(status_code=502, detail=f"Qwen3 返回格式错误: {content}") from error
    if pick_status not in {"Success", "Fail"}:
        raise HTTPException(status_code=502, detail=f"Qwen3 pick_status 无效: {pick_status}")
    return PickCheckResponse(pick_status=pick_status)


@router.post("/perception/pick/check", response_model=PickCheckResponse)
def check_product(request: PickCheckRequest) -> PickCheckResponse:
    hand_image, hand_media_type = fetch_hand_image(request.hand)
    reference_image, reference_media_type = fetch_reference_image(
        request.product_name
    )
    return call_qwen(
        load_prompt(request),
        reference_image,
        reference_media_type,
        hand_image,
        hand_media_type,
    )


app.include_router(router)
