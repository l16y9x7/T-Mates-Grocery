"""Wrist-camera product placement check API."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import requests
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent
if __package__ and __package__.startswith("perception."):
    from ...config import QWEN3_MODEL, QWEN3_URL
    from ...pick.check.server import (
        fetch_hand_image,
        fetch_reference_image,
        image_content,
    )
else:
    PERCEPTION_ROOT = ROOT.parents[1]
    if str(PERCEPTION_ROOT) not in sys.path:
        sys.path.insert(0, str(PERCEPTION_ROOT))
    from config import QWEN3_MODEL, QWEN3_URL
    from pick.check.server import (
        fetch_hand_image,
        fetch_reference_image,
        image_content,
    )


SORTING_PROMPT_PATH = ROOT / "sorting_prompt.txt"
SHELF_PROMPT_PATH = ROOT / "shelf_prompt.txt"
QWEN_TIMEOUT_SECONDS = 120

app = FastAPI(title="Place Check", version="1.0.0")
router = APIRouter()


class PlaceCheckRequest(BaseModel):
    task_type: str
    product_name: str
    hand: str


class PlaceCheckResponse(BaseModel):
    place_status: str


def load_prompt(request: PlaceCheckRequest) -> str:
    task_type = request.task_type.strip().upper()
    if task_type == "SORTING":
        prompt_path = SORTING_PROMPT_PATH
    elif task_type in {"SHORTAGE", "MISPLACED"}:
        prompt_path = SHELF_PROMPT_PATH
    else:
        raise HTTPException(
            status_code=400,
            detail="task_type 只能是 SORTING、SHORTAGE 或 MISPLACED",
        )
    return prompt_path.read_text(encoding="utf-8").format(
        product_name=request.product_name
    )


def call_qwen(
    prompt: str,
    reference_image: bytes,
    reference_media_type: str,
    hand_image: bytes,
    hand_media_type: str,
) -> PlaceCheckResponse:
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
        place_status = result["place_status"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise HTTPException(status_code=502, detail=f"Qwen3 返回格式错误: {content}") from error
    if place_status not in {"Success", "Fail"}:
        raise HTTPException(status_code=502, detail=f"Qwen3 place_status 无效: {place_status}")
    return PlaceCheckResponse(place_status=place_status)


@router.post("/perception/place/check", response_model=PlaceCheckResponse)
def check_product_placement(request: PlaceCheckRequest) -> PlaceCheckResponse:
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
