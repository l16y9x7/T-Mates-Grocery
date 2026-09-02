"""多模态大模型生成 SAM3 实例分割提示词（OpenAI 兼容 /v1/chat/completions）。"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from PIL import Image

_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from config import get_vlm_conf, get_vlm_profile  # noqa: E402

logger = logging.getLogger("vlm_prompt")

VLM_SYSTEM_TEMPLATE = (
    "You are a visual prompting assistant for SAM3 instance segmentation.\n"
    "Rules:\n"
    "1. Translate the Chinese object name in double quotes into standard retail "
    "English commodity name, add shape/package feature description.\n"
    "2. Generate SAM3 prompt following exact format: instance segmentation, translated_name\n"
    "3. Do not add any comments, line breaks, symbols, descriptions. Only one single line result.\n"
    'Task: Look at the image, find all "{chinese_name}".'
)

# 缺货商品识别默认提示词（可在 UI 与 conf.json 覆盖）
DEFAULT_MISSING_PRODUCT_PROMPT = (
    "You are a professional supermarket shelf inspector. Your job is to find out-of-stock goods.\n"
    "Goods placement rule: The same goods are placed vertically.\n"
    "Check if there are empty positions with missing goods on the outermost row "
    "(the row closest to the shelf edge).\n"
    "The missing goods name is the Chinese name of the goods sitting directly behind "
    "that empty outermost position.\n"
    "Do not name goods that are already present on the outermost row.\n"
    "Only output the Chinese name of the missing goods."
)

_MISSING_INSPECT_SUFFIX = (
    "\n\nBefore answering, inspect carefully in Chinese (short):\n"
    "- Which outermost-row positions are truly EMPTY "
    "(no product standing at the shelf edge there)?\n"
    "- For each EMPTY, what Chinese-brand goods sit directly BEHIND it "
    "(same column, deeper on the shelf)?\n"
    "If a brand already stands at the front edge, that column is NOT empty.\n"
    "Write at most 2 short sentences."
)
# 兼容旧常量名
DEFAULT_MISPLACED_PRODUCT_PROMPT = DEFAULT_MISSING_PRODUCT_PROMPT


def _profile_defaults(name: str) -> Dict[str, Any]:
    cfg = get_vlm_profile(name)
    return {
        "provider": str(cfg.get("provider") or "").strip().lower(),
        "api_url": str(cfg.get("api_url") or "").strip(),
        "model": str(cfg.get("model") or "").strip(),
        "timeout_s": float(cfg.get("timeout_s") or 60.0),
        "temperature": float(cfg.get("temperature") or 0.1),
        "max_tokens": int(cfg.get("max_tokens") or 128),
        "api_key": str(cfg.get("api_key") or "").strip(),
    }


_SAM3_VLM = _profile_defaults("sam3_prompt")
_REASON_VLM = _profile_defaults("reason")

# SAM3 提示词：本地 qwen3-vl-4b
DEFAULT_SAM3_VLM_PROVIDER = _SAM3_VLM["provider"]
DEFAULT_SAM3_VLM_API_URL = _SAM3_VLM["api_url"]
DEFAULT_SAM3_VLM_MODEL = _SAM3_VLM["model"]
DEFAULT_SAM3_VLM_TIMEOUT_S = _SAM3_VLM["timeout_s"]
DEFAULT_SAM3_VLM_TEMPERATURE = _SAM3_VLM["temperature"]
DEFAULT_SAM3_VLM_MAX_TOKENS = _SAM3_VLM["max_tokens"]

# 缺货识别 / 放置位移：MiniMax-M3
DEFAULT_REASON_VLM_PROVIDER = _REASON_VLM["provider"]
DEFAULT_REASON_VLM_API_URL = _REASON_VLM["api_url"]
DEFAULT_REASON_VLM_MODEL = _REASON_VLM["model"]
DEFAULT_REASON_VLM_TIMEOUT_S = _REASON_VLM["timeout_s"]
DEFAULT_REASON_VLM_TEMPERATURE = _REASON_VLM["temperature"]
DEFAULT_REASON_VLM_MAX_TOKENS = _REASON_VLM["max_tokens"]

# 兼容旧常量：默认指向 reason（M3）；SAM3 页请用 DEFAULT_SAM3_VLM_*
DEFAULT_VLM_PROVIDER = DEFAULT_REASON_VLM_PROVIDER
DEFAULT_VLM_API_URL = DEFAULT_REASON_VLM_API_URL
DEFAULT_VLM_MODEL = DEFAULT_REASON_VLM_MODEL
DEFAULT_VLM_TIMEOUT_S = DEFAULT_REASON_VLM_TIMEOUT_S
DEFAULT_VLM_TEMPERATURE = DEFAULT_REASON_VLM_TEMPERATURE
DEFAULT_VLM_MAX_TOKENS = DEFAULT_REASON_VLM_MAX_TOKENS


def _resolve_api_key(
    explicit: Optional[str] = None, *, profile: Optional[str] = None
) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    if profile:
        val = str(get_vlm_profile(profile).get("api_key") or "").strip()
        if val:
            return val
    cfg = get_vlm_conf()
    for key in ("api_key", "anthropic_api_key", "minimax_api_key"):
        val = str(cfg.get(key) or "").strip()
        if val:
            return val
    for env in ("ANTHROPIC_API_KEY", "MINIMAX_API_KEY", "VLM_API_KEY"):
        val = (os.environ.get(env) or "").strip()
        if val:
            return val
    return ""


def _is_anthropic_endpoint(api_url: str, provider: str = "") -> bool:
    p = (provider or "").strip().lower()
    if p in {"anthropic", "minimax", "minimax-anthropic"}:
        return True
    u = (api_url or "").lower()
    return "/anthropic" in u or u.rstrip("/").endswith("anthropic")


def _normalize_anthropic_messages_url(api_url: str) -> str:
    """Accept base ``.../anthropic`` or full ``.../v1/messages``."""
    u = (api_url or "").strip().rstrip("/")
    if u.endswith("/v1/messages"):
        return u
    if u.endswith("/messages"):
        return u
    return f"{u}/v1/messages"


def _normalize_openai_chat_url(api_url: str) -> str:
    u = (api_url or "").strip().rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    if u.endswith("/v1"):
        return f"{u}/chat/completions"
    return u


def build_vlm_user_text(chinese_name: str) -> str:
    name = (chinese_name or "").strip()
    if not name:
        raise ValueError("商品中文名不能为空")
    return VLM_SYSTEM_TEMPLATE.format(chinese_name=name)


def _image_to_bytes_and_mime(
    image: Union[Image.Image, Path, str],
) -> Tuple[bytes, str]:
    if isinstance(image, (str, Path)):
        path = Path(image).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"图像不存在: {path}")
        raw = path.read_bytes()
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif suffix == ".webp":
            mime = "image/webp"
        else:
            mime = "image/png"
        return raw, mime

    rgb = image.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def _image_to_data_url(image: Union[Image.Image, Path, str]) -> str:
    raw, mime = _image_to_bytes_and_mime(image)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_openai_message_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"VLM 响应无 choices: {payload}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in (None, "text"):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    raise RuntimeError(f"无法解析 VLM message.content: {message!r}")


def _extract_anthropic_message_text(payload: Dict[str, Any]) -> str:
    """Anthropic / MiniMax Messages API: content blocks → text."""
    if payload.get("type") == "error" or payload.get("error"):
        err = payload.get("error") or payload
        raise RuntimeError(f"VLM Anthropic 错误: {err}")
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        # some gateways wrap like openai
        if payload.get("choices"):
            return _extract_openai_message_text(payload)
        raise RuntimeError(f"无法解析 Anthropic content: {payload}")
    parts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        typ = item.get("type")
        if typ in (None, "text") and item.get("text"):
            parts.append(str(item["text"]))
        # skip thinking / tool_use blocks for final answer
    if not parts:
        raise RuntimeError(f"Anthropic 响应无文本块: {payload}")
    return "\n".join(parts)


def _strip_model_noise(raw: str) -> str:
    text = (raw or "").strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(
        r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.IGNORECASE
    ).strip()
    text = re.sub(
        r"<reasoning_content>[\s\S]*?</reasoning_content>",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    return text


def sanitize_sam3_prompt(raw: str) -> str:
    """清洗模型输出，只保留单行 SAM3 提示词。"""
    text = _strip_model_noise(raw)
    if not text:
        raise RuntimeError("VLM 返回空提示词")

    for line in text.splitlines():
        line = line.strip().strip("`\"'")
        if not line:
            continue
        # 去掉前缀编号 / 标签
        line = re.sub(r"^(?:SAM3\s*)?(?:prompt\s*[:=]\s*)", "", line, flags=re.IGNORECASE)
        if "instance segmentation" in line.lower() or "," in line:
            return line
        return line

    raise RuntimeError(f"无法从 VLM 输出解析提示词: {raw!r}")


def sanitize_product_name(raw: str) -> str:
    """从 VLM 回复中提取商品中文名（取首行有效内容）。"""
    text = _strip_model_noise(raw)
    if not text:
        raise RuntimeError("VLM 未返回商品名")

    for line in text.splitlines():
        line = line.strip().strip("`\"'「」『』")
        if not line:
            continue
        # 兼容两阶段定位格式：左起第2列|怡宝
        if "|" in line:
            line = line.split("|")[-1].strip()
        line = re.sub(
            r"^(?:商品(?:名称|名)?|名字|名称|答[：:]?)\s*[：:\-]?\s*",
            "",
            line,
        )
        line = line.strip().strip("`\"'「」『』。；;")
        if line:
            return line
    raise RuntimeError(f"无法从 VLM 输出解析商品名: {raw!r}")


def chat_completions_vision(
    image: Union[Image.Image, Path, str],
    user_text: str,
    *,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    profile: Optional[str] = None,
) -> str:
    """视觉对话：支持 OpenAI chat/completions 与 Anthropic/MiniMax Messages。

    ``profile``: ``sam3_prompt``（qwen）或 ``reason``（M3）；决定缺省 endpoint。
    """
    prof_name = (profile or "reason").strip() or "reason"
    ep = get_vlm_profile(prof_name)
    url_in = (api_url or ep.get("api_url") or DEFAULT_VLM_API_URL).strip()
    model_name = (model or ep.get("model") or DEFAULT_VLM_MODEL).strip()
    prov = (provider or ep.get("provider") or "").strip()
    timeout = float(
        timeout_s if timeout_s is not None else ep.get("timeout_s") or DEFAULT_VLM_TIMEOUT_S
    )
    temp = float(
        temperature
        if temperature is not None
        else ep.get("temperature")
        if ep.get("temperature") is not None
        else DEFAULT_VLM_TEMPERATURE
    )
    tokens = int(
        max_tokens if max_tokens is not None else ep.get("max_tokens") or DEFAULT_VLM_MAX_TOKENS
    )
    prompt = (user_text or "").strip()
    if not prompt:
        raise ValueError("提示词不能为空")

    key = _resolve_api_key(api_key, profile=prof_name)
    use_anthropic = _is_anthropic_endpoint(url_in, prov)

    # 局域网可不走代理；公网 MiniMax 走系统代理（若有）
    is_local = url_in.startswith("http://127.") or url_in.startswith("http://192.168.")
    proxies = {"http": None, "https": None} if is_local else None

    if use_anthropic:
        url = _normalize_anthropic_messages_url(url_in)
        raw_bytes, mime = _image_to_bytes_and_mime(image)
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        payload: Dict[str, Any] = {
            "model": model_name,
            "max_tokens": tokens,
            "temperature": temp,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        headers = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if key:
            headers["x-api-key"] = key
            headers["Authorization"] = f"Bearer {key}"
        else:
            raise RuntimeError(
                "MiniMax/Anthropic 需要 API Key：设置环境变量 ANTHROPIC_API_KEY "
                "或写入 config/secrets.local.json → vlm.api_key"
            )
        logger.info(
            "VLM anthropic vision: url=%s model=%s prompt[:80]=%r",
            url,
            model_name,
            prompt[:80],
        )
        resp = requests.post(
            url, json=payload, headers=headers, timeout=timeout, proxies=proxies
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"VLM 请求失败 HTTP {resp.status_code}: {resp.text[:500]}"
            )
        try:
            body = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"VLM 响应非 JSON: {resp.text[:500]}") from exc
        return _extract_anthropic_message_text(body)

    # OpenAI-compatible
    url = _normalize_openai_chat_url(url_in)
    data_url = _image_to_data_url(image)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": temp,
        "max_tokens": tokens,
    }
    headers = {"content-type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    logger.info(
        "VLM openai vision: url=%s model=%s prompt[:80]=%r",
        url,
        model_name,
        prompt[:80],
    )
    resp = requests.post(
        url, json=payload, headers=headers, timeout=timeout, proxies=proxies
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"VLM 请求失败 HTTP {resp.status_code}: {resp.text[:500]}"
        )
    try:
        body = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"VLM 响应非 JSON: {resp.text[:500]}") from exc
    return _extract_openai_message_text(body)


def generate_sam3_prompt_from_image(
    image: Union[Image.Image, Path, str],
    chinese_name: str,
    *,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """调用本地 qwen3-vl，根据 RGB 与中文商品名生成 SAM3 提示词。"""
    ep = get_vlm_profile("sam3_prompt")
    raw = chat_completions_vision(
        image,
        build_vlm_user_text(chinese_name),
        api_url=api_url or ep.get("api_url"),
        model=model or ep.get("model"),
        timeout_s=timeout_s if timeout_s is not None else ep.get("timeout_s"),
        temperature=temperature if temperature is not None else ep.get("temperature"),
        max_tokens=max_tokens if max_tokens is not None else ep.get("max_tokens"),
        provider=str(ep.get("provider") or "openai"),
        profile="sam3_prompt",
    )
    prompt = sanitize_sam3_prompt(raw)
    logger.info("VLM generated SAM3 prompt: %r (raw=%r)", prompt, raw)
    return prompt


def identify_missing_product(
    image: Union[Image.Image, Path, str],
    prompt: Optional[str] = None,
    *,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, str]:
    """识别货架边缘排中缺货的商品名（默认 MiniMax-M3）。

    两阶段：先描述「外侧空位 + 空位正后方商品」，再据此输出中文商品名，
    避免把前排已有商品（如薯愿）误报为缺货。

    Returns:
      ``{"product_name": ..., "raw_reply": ...}``
    """
    ep = get_vlm_profile("reason")
    user_text = (prompt or "").strip() or DEFAULT_MISSING_PRODUCT_PROMPT
    tokens = int(
        max_tokens
        if max_tokens is not None
        else max(int(ep.get("max_tokens") or DEFAULT_REASON_VLM_MAX_TOKENS), 256)
    )
    common = dict(
        api_url=api_url or ep.get("api_url"),
        model=model or ep.get("model"),
        timeout_s=timeout_s if timeout_s is not None else ep.get("timeout_s"),
        temperature=temperature if temperature is not None else ep.get("temperature"),
        provider=str(ep.get("provider") or "anthropic"),
        profile="reason",
    )

    note = chat_completions_vision(
        image,
        user_text + _MISSING_INSPECT_SUFFIX,
        max_tokens=tokens,
        **common,
    ).strip()

    extract_prompt = (
        f"Shelf inspection note:\n{note}\n\n"
        f"Now follow the original task:\n{user_text}\n\n"
        "Critical:\n"
        "- Missing goods name = Chinese name of the goods BEHIND the empty outermost slot.\n"
        "- Never output goods that already stand on the outermost row.\n"
        "- If multiple empties are mentioned, prefer the empty whose behind-goods match "
        "neighboring front-row goods of the same brand "
        "(e.g. empty between/beside 可比克 with 可比克 behind → 可比克).\n"
        "Only output one Chinese product name."
    )
    raw = chat_completions_vision(
        image,
        extract_prompt,
        max_tokens=min(128, tokens),
        **common,
    ).strip()
    name = sanitize_product_name(raw)
    # 无效占位回退到 note 里最后一个疑似中文商品片段
    if name.upper() in {"EMPTY", "NONE", "N/A"} or name in {"无", "无缺货", "没有"}:
        name = sanitize_product_name(note)
    logger.info("VLM missing product: name=%r note=%r raw=%r", name, note, raw)
    return {
        "product_name": name,
        "raw_reply": f"{note}\n{raw}".strip(),
    }


# 兼容旧函数名
identify_misplaced_product = identify_missing_product


DEFAULT_PLACE_OFFSET_PROMPT = (
    "You are a supermarket restocking assistant. Infer the EMPTY front slot from SPACE.\n"
    "Task: Among SAME-product instances (colored masks + id labels + xyz_mm),\n"
    "pick which instance to move into the missing OUTERMOST front slot, and by how much.\n"
    "\n"
    "Spatial rules (critical):\n"
    "1) Camera frame mm: +X right, +Y down, +Z into shelf (larger Z = deeper).\n"
    "2) Front row ≈ smaller Z. Empty outermost slot is ON the shelf front, "
    "usually same column X as a deeper instance, with NO product at front_z.\n"
    "3) SELECT the instance sitting BEHIND that gap (similar X, larger Z than front_z_ref).\n"
    "4) Do NOT select an instance that is already at the global front row and then push it "
    "further toward the camera (that flies off the shelf).\n"
    "5) Destination should stay on-shelf: dest Z ≈ front_z_ref of other front products; "
    "dest X ≈ source column X (small right/left only if aligning to the gap).\n"
    "6) forward_mm to the front slot is typically NEGATIVE (toward camera), magnitude ≈ "
    "source_Z - front_z_ref (often 40–150 mm), not hundreds past the front row.\n"
    "\n"
    "Output ONLY one JSON line, no markdown:\n"
    '{{"instance_id": <int>, "right_mm": <number>, "down_mm": <number>, "forward_mm": <number>, '
    '"dest_xyz_mm": [<x>,<y>,<z>], "reason": "<short spatial reason>"}}\n'
)


def _format_instances_catalog(instances: List[Dict[str, Any]]) -> str:
    lines = ["Candidate instances (same product; match colored mask labels in the image):"]
    for inst in instances:
        iid = int(inst.get("instance_id") or 0)
        xyz = inst.get("xyz_mm") or []
        size_m = [float(v) for v in (inst.get("size_3d") or [0, 0, 0])]
        size_mm = [round(v * 1000.0, 1) for v in size_m]
        score = float(inst.get("score") or 0.0)
        bbox = inst.get("bbox")
        xyz_s = (
            f"[{', '.join(f'{float(v):.1f}' for v in xyz[:3])}]"
            if len(xyz) >= 3
            else str(xyz)
        )
        lines.append(
            f"- id={iid}: xyz_mm={xyz_s}, size_mm={size_mm}, "
            f"score={score:.3f}, bbox={bbox}"
        )
    return "\n".join(lines)


def build_place_offset_prompt(
    product_name: str,
    *,
    xyz_mm: List[float],
    size_3d_m: List[float],
    score: float,
    extra_hint: str = "",
    identify_dialogue: str = "",
    instances: Optional[List[Dict[str, Any]]] = None,
    spatial_prior: Optional[Dict[str, Any]] = None,
) -> str:
    name = (product_name or "").strip() or "unknown"
    hint = (extra_hint or "").strip()
    dialogue = (identify_dialogue or "").strip()
    size_m = [float(v) for v in size_3d_m]
    size_mm = [round(v * 1000.0, 1) for v in size_m]
    body = (
        f"{DEFAULT_PLACE_OFFSET_PROMPT}\n"
        f"Product Chinese name: {name}\n"
    )
    if instances:
        body += _format_instances_catalog(instances) + "\n"
        ids = [int(x.get("instance_id") or 0) for x in instances]
        body += f"instance_id MUST be one of: {ids}\n"
    else:
        body += (
            f"Current position xyz_mm [X,Y,Z] from depth camera / GenPose2: "
            f"{[round(float(v), 1) for v in xyz_mm]}\n"
            f"Current size_3d_mm [sx,sy,sz]: {size_mm}\n"
            f"Current size_3d_m [sx,sy,sz]: {[round(v, 4) for v in size_m]}\n"
            f"Detection score: {float(score):.3f}\n"
        )
    if spatial_prior and spatial_prior.get("success"):
        sug = spatial_prior.get("suggested") or {}
        body += (
            "\n--- Geometric spatial prior (from instance xyz clustering) ---\n"
            f"front_z_ref_mm (other products' front row): "
            f"{spatial_prior.get('front_z_ref_mm')}\n"
            f"empty_front_columns: {json.dumps(spatial_prior.get('empty_columns') or [], ensure_ascii=False)}\n"
        )
        if sug:
            body += (
                f"Recommended source instance_id={sug.get('suggested_source_id')} "
                f"→ dest_xyz_mm={sug.get('suggested_dest_xyz_mm')} "
                f"(offset {sug.get('suggested_offset_mm')}).\n"
                "Follow this recommendation unless the inspection dialogue clearly indicates "
                "a different empty column.\n"
            )
        body += "--- End spatial prior ---\n"
    if dialogue:
        if len(dialogue) > 2500:
            dialogue = dialogue[:2500] + "\n...(truncated)"
        body += (
            "\n--- Prior shelf-inspection dialogue (step 1: missing-product identify) ---\n"
            "Use this with xyz geometry to locate the empty outermost slot.\n"
            f"{dialogue}\n"
            "--- End prior dialogue ---\n"
        )
    body += (
        "Decide from SPACE: which gap is empty at the front, which instance is behind it, "
        "and the mm move onto that front slot (dest Z ≈ front_z_ref).\n"
    )
    if hint:
        body += f"Extra hint: {hint}\n"
    return body


def parse_place_offset_mm(raw: str) -> Dict[str, float]:
    """Parse VLM place offset JSON → right/down/forward mm (+ optional instance_id)."""
    decision = parse_place_decision(raw)
    return {
        "right_mm": decision["right_mm"],
        "down_mm": decision["down_mm"],
        "forward_mm": decision["forward_mm"],
    }


def parse_place_decision(raw: str) -> Dict[str, Any]:
    """Parse VLM place decision: instance_id + offset mm + optional reason."""
    text = _strip_model_noise(raw)
    if not text:
        raise RuntimeError("VLM 未返回放置决策")
    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"无法解析放置决策 JSON: {raw!r}")
    try:
        data = json.loads(match.group(0))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"放置决策 JSON 无效: {match.group(0)!r}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"放置决策必须是对象: {data!r}")

    def _num(*keys: str, default: float = 0.0) -> float:
        for k in keys:
            if k in data and data[k] is not None:
                return float(data[k])
        return float(default)

    right = _num("right_mm", "right", "x_mm", "dx_mm")
    down = _num("down_mm", "down", "y_mm", "dy_mm")
    forward = _num("forward_mm", "forward", "z_mm", "dz_mm")
    if "left_mm" in data and "right_mm" not in data and "right" not in data:
        right = -float(data["left_mm"])
    if "up_mm" in data and "down_mm" not in data and "down" not in data:
        down = -float(data["up_mm"])
    if "back_mm" in data and "forward_mm" not in data and "forward" not in data:
        forward = -float(data["back_mm"])

    instance_id: Optional[int] = None
    for k in ("instance_id", "id", "target_id", "source_instance_id"):
        if k in data and data[k] is not None:
            try:
                instance_id = int(data[k])
            except Exception:  # noqa: BLE001
                instance_id = None
            break

    reason = str(data.get("reason") or data.get("why") or "").strip()
    return {
        "instance_id": instance_id,
        "right_mm": right,
        "down_mm": down,
        "forward_mm": forward,
        "reason": reason,
    }


def estimate_place_offset_from_image(
    image: Union[Image.Image, Path, str],
    product_name: str,
    *,
    xyz_mm: List[float],
    size_3d_m: List[float],
    score: float = 1.0,
    extra_hint: str = "",
    identify_dialogue: str = "",
    instances: Optional[List[Dict[str, Any]]] = None,
    spatial_prior: Optional[Dict[str, Any]] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """VLM（默认 MiniMax-M3）：在多实例中选型并估计放到缺货位的平移（mm）。"""
    ep = get_vlm_profile("reason")
    prompt = build_place_offset_prompt(
        product_name,
        xyz_mm=xyz_mm,
        size_3d_m=size_3d_m,
        score=score,
        extra_hint=extra_hint,
        identify_dialogue=identify_dialogue,
        instances=instances,
        spatial_prior=spatial_prior,
    )
    tokens = int(
        max_tokens
        if max_tokens is not None
        else max(int(ep.get("max_tokens") or DEFAULT_REASON_VLM_MAX_TOKENS), 384)
    )
    raw = chat_completions_vision(
        image,
        prompt,
        api_url=api_url or ep.get("api_url"),
        model=model or ep.get("model"),
        timeout_s=timeout_s if timeout_s is not None else ep.get("timeout_s"),
        temperature=(
            temperature
            if temperature is not None
            else float(ep.get("temperature") or 0.2)
        ),
        max_tokens=tokens,
        provider=str(ep.get("provider") or "anthropic"),
        profile="reason",
    )
    decision = parse_place_decision(raw)
    offset = {
        "right_mm": decision["right_mm"],
        "down_mm": decision["down_mm"],
        "forward_mm": decision["forward_mm"],
    }
    logger.info("VLM place decision: %s raw=%r", decision, raw)
    return {
        "offset_mm": offset,
        "instance_id": decision.get("instance_id"),
        "reason": decision.get("reason") or "",
        "decision": decision,
        "raw_reply": (raw or "").strip(),
        "prompt": prompt,
    }
