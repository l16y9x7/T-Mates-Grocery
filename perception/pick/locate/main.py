import base64
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from prompt_mapping import get_sam3_prompt


ROOT = Path(__file__).resolve().parent
RGB_DIR = ROOT.parents[1] / "test_data" / "2026-08-04"
SAM3_URL = "http://211.137.21.33:25541/api/v1/segment"
QWEN3_URL = "http://211.137.21.33:25542/v1/chat/completions"
QWEN3_MODEL = "Qwen3-VL-4B-Instruct"
SAM3_THRESHOLD = 0.5
SAM3_MASK_THRESHOLD = 0.5
app = FastAPI(title="Sorting Pick Locate")


class LocateRequest(BaseModel):
    product_name: str
    task_type: str


class LocateResponse(BaseModel):
    name: str
    bbox: list[float]
    mask: str


def get_latest_rgb() -> Path:
    """现在从本地取最新 RGB；后续替换成视频流请求。"""
    images = sorted(RGB_DIR.glob("*_rgb.*"))
    if not images:
        raise HTTPException(status_code=404, detail="没有找到 RGB 图片")
    return images[-1]


@app.get("/video/frame")
def get_video_frame() -> FileResponse:
    return FileResponse(get_latest_rgb(), media_type="image/jpeg")


def call_qwen3(name: str, image_path: Path) -> dict:
    """调用 Qwen3-VL，暂时返回未经处理的原始 JSON。"""
    prompt = ""
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

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
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 1024,
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


@app.post("/visual/pick/locate", response_model=LocateResponse)
def locate_product(request: LocateRequest) -> LocateResponse:
    """输入商品名和任务类型，返回一个商品名，bbox和mask。"""
    name = request.product_name.strip()
    task_type = request.task_type.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name 不能为空")
    if not task_type:
        raise HTTPException(status_code=400, detail="Task type不能为空")
    if not task_type in ("SORTING", "SHORTAGE", "MISPLACED"):
        raise HTTPException(status_code=400, detail=f"Task type={task_type}，不支持")

    image_path = get_latest_rgb()
    prompt = get_sam3_prompt(name)
    if not prompt:
        raise HTTPException(status_code=400, detail=f"Product name={name}，不支持")

    try:
        with image_path.open("rb") as image_file:
            response = requests.post(
                SAM3_URL,
                files={"image": (image_path.name, image_file, "image/jpeg")},
                data={
                    "prompt": prompt,
                    "threshold": SAM3_THRESHOLD,
                    "mask_threshold": SAM3_MASK_THRESHOLD,
                },
                timeout=90,
            )
        response.raise_for_status()
        instances = response.json().get("instances", [])
    except (requests.RequestException, ValueError) as error:
        raise HTTPException(status_code=502, detail=f"SAM3 请求失败: {error}") from error

    if not instances:
        raise HTTPException(status_code=404, detail="没有找到目标商品")

    return LocateResponse(name=name, bbox=bbox, mask=mask)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
