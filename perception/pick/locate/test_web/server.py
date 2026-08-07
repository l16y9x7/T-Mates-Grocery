import base64
import binascii
import json
import mimetypes
import os
import re
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
RGB_DIR = ROOT.parents[2] / "test_data" / "2026-08-04"
SKU_CATALOG_PATH = ROOT.parents[2] / "sku" / "products.json"
PROMPT_PAIR_MAPPING_PATH = ROOT.parent / "qwen_sam_prompt_mapping.json"

QWEN3_URL = os.getenv(
    "QWEN3_URL",
    "http://211.137.21.33:25542/v1/chat/completions",
)
QWEN3_MODEL = os.getenv("QWEN3_MODEL", "Qwen3-VL-4B-Instruct")
SAM3_URL = os.getenv(
    "SAM3_URL",
    "http://211.137.21.33:25541/api/v1/segment",
)
LOCATE_DEBUG_URL = os.getenv(
    "LOCATE_DEBUG_URL",
    "http://192.168.130.59:8081/perception/pick/locate/debug",
)
QWEN_SAMPLE_COUNT = 3
QWEN_TEMPERATURE = 0.7

app = FastAPI(title="Qwen3 / SAM3 Prompt Test Web")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PromptRequest(BaseModel):
    image_name: str
    prompt: str


class DirectQwenRequest(BaseModel):
    prompt: str
    image_base64: str
    temperature: float = Field(default=0.5, ge=0, le=2)


class SaveQwenPromptRequest(BaseModel):
    sku_name: str
    prompt: str


class SavePromptPairRequest(BaseModel):
    sku_name: str
    qwen3_prompt: str
    sam3_prompt: str


class LocateDebugProxyRequest(BaseModel):
    name: str
    product_name: str
    hand: str
    qwen3_prompt: str | None = None
    sam3_prompt: str | None = None


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


@app.get("/api/images")
def list_images() -> dict:
    images = sorted(path.name for path in RGB_DIR.glob("*_rgb.*"))
    return {"images": images, "default": images[-1] if images else None}


@app.get("/api/skus")
def list_skus() -> dict:
    prompt_mapping = load_prompt_pair_mapping()
    skus = []
    for sku in load_skus():
        prompt_pair = prompt_mapping.get(sku["name"])
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
    return {"skus": skus}


@app.post("/api/qwen-prompts")
def save_qwen_prompt(request: SaveQwenPromptRequest) -> dict:
    sku_name = request.sku_name.strip()
    prompt = request.prompt.strip()
    if not sku_name:
        raise HTTPException(status_code=400, detail="SKU 不能为空")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt 不能为空")

    valid_names = {sku["name"] for sku in load_skus()}
    if sku_name not in valid_names:
        raise HTTPException(status_code=400, detail=f"商品库中不存在 SKU：{sku_name}")

    mapping = load_prompt_pair_mapping()
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
    write_json_mapping(PROMPT_PAIR_MAPPING_PATH, mapping, "配对 Prompt")
    return {
        "sku_name": sku_name,
        "prompt": prompt,
        "qwen3_prompt": prompt,
        "sam3_prompt": current_pair["sam3_prompt"],
        "overwritten": overwritten,
    }


@app.post("/api/prompt-pairs")
def save_prompt_pair(request: SavePromptPairRequest) -> dict:
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

    mapping = load_prompt_pair_mapping()
    overwritten = sku_name in mapping
    mapping[sku_name] = {
        "qwen3_prompt": qwen3_prompt,
        "sam3_prompt": sam3_prompt,
    }
    write_json_mapping(PROMPT_PAIR_MAPPING_PATH, mapping, "配对 Prompt")
    return {
        "sku_name": sku_name,
        "qwen3_prompt": qwen3_prompt,
        "sam3_prompt": sam3_prompt,
        "overwritten": overwritten,
    }


@app.post("/api/locate-debug")
def run_locate_debug(request: LocateDebugProxyRequest) -> dict:
    payload = {
        "name": request.name.strip(),
        "product_name": request.product_name.strip(),
        "hand": request.hand.strip(),
    }
    if not all(payload.values()):
        raise HTTPException(status_code=400, detail="name、product_name、hand 都不能为空")
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


def load_prompt_pair_mapping() -> dict[str, dict[str, str]]:
    if not PROMPT_PAIR_MAPPING_PATH.exists():
        return {}
    try:
        mapping = json.loads(PROMPT_PAIR_MAPPING_PATH.read_text(encoding="utf-8"))
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
    uvicorn.run(app, host="0.0.0.0", port=8082)
