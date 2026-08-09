"""Compare every *_1/*_2 image pair with Qwen3-VL and save visual results."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import requests


PERCEPTION_ROOT = Path(__file__).resolve().parents[2]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from config import QWEN3_MODEL, QWEN3_URL  # noqa: E402


TARGET_SIZE = (1280, 720)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PAIR_PATTERN = re.compile(r"^(?P<pair>.+)_(?P<side>[12])$")
ALLOWED_CHANGE_TYPES = {
    "moved",
    "swapped",
    "removed",
    "added",
    "replaced",
    "appearance_change",
    "uncertain",
}

SYSTEM_PROMPT = """你是货架前后图像对比器。你会同时看到 BEFORE（变化前）和 AFTER（变化后）两张图。

任务：找出商品数量、位置、排列顺序或商品种类的真实变化。忽略相机小幅位移、透视、裁剪、亮度、阴影和反光变化。特别注意外观相似但颜色、文字或包装不同的商品互换位置。

两张输入图都严格为 1280x720。bbox 必须使用 [x, y, width, height] 像素格式，原点在左上角。移动或互换时同时提供 before_bbox 和 after_bbox；只有单侧出现时另一侧使用 null。bbox 无法可靠判断时使用 null，禁止编造。

只输出一个严格 JSON 对象，不要 Markdown、代码块或说明。格式：
{
  "has_difference": true,
  "summary": "一句话中文总结",
  "changes": [
    {
      "type": "moved|swapped|removed|added|replaced|appearance_change|uncertain",
      "object": "发生变化的商品或区域",
      "before_bbox": [x, y, width, height] 或 null,
      "after_bbox": [x, y, width, height] 或 null,
      "confidence": 0.0到1.0,
      "description": "具体变化"
    }
  ]
}

完全没有商品变化时输出 has_difference=false、changes=[]。"""

USER_PROMPT = "请比较 BEFORE 和 AFTER，完整列出所有真实商品变化。"


def discover_pairs(directory: str | Path) -> list[tuple[str, Path, Path]]:
    """Return complete pair IDs and paths in natural order."""

    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"input directory does not exist: {root}")

    grouped: dict[str, dict[str, Path]] = {}
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = PAIR_PATTERN.fullmatch(path.stem)
        if match:
            grouped.setdefault(match.group("pair"), {})[match.group("side")] = path

    pairs = [
        (pair_id, sides["1"], sides["2"])
        for pair_id, sides in grouped.items()
        if "1" in sides and "2" in sides
    ]
    return sorted(pairs, key=lambda item: _natural_key(item[0]))


def read_720p(path: str | Path) -> np.ndarray:
    """Read a potentially non-ASCII path and resize to 1280x720."""

    source = Path(path)
    try:
        encoded = np.fromfile(source, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"cannot read image: {source}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"invalid image: {source}")
    return cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)


def image_to_data_url(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    if not success:
        raise ValueError("failed to encode image as JPEG")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def build_payload(
    before: np.ndarray,
    after: np.ndarray,
    model: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "BEFORE（变化前）"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(before)},
                    },
                    {"type": "text", "text": "AFTER（变化后）"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(after)},
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 1600,
    }


def request_qwen(
    before: np.ndarray,
    after: np.ndarray,
    *,
    url: str,
    model: str,
    timeout: float,
    api_key: str | None = None,
) -> str:
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.post(
            endpoint,
            json=build_payload(before, after, model),
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"cannot connect to Qwen: {exc}") from exc
    if not response.ok:
        raise RuntimeError(
            f"Qwen returned HTTP {response.status_code}: {response.text[:500]}"
        )
    try:
        value = response.json()
        content = value["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"invalid Qwen response: {response.text[:500]}") from exc
    if not isinstance(content, str):
        raise RuntimeError("Qwen message.content is not a string")
    return content


def parse_qwen_result(content: str) -> dict[str, Any]:
    """Parse and validate Qwen's JSON, accepting an accidental code fence."""

    normalized = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", normalized, re.DOTALL)
    if fenced:
        normalized = fenced.group(1)
    try:
        value = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Qwen output is not valid JSON: {content[:500]}") from exc
    if not isinstance(value, dict):
        raise ValueError("Qwen output must be a JSON object")
    if not isinstance(value.get("has_difference"), bool):
        raise ValueError("has_difference must be boolean")
    if not isinstance(value.get("summary"), str):
        raise ValueError("summary must be a string")
    changes = value.get("changes")
    if not isinstance(changes, list):
        raise ValueError("changes must be an array")

    validated_changes = [
        _validate_change(change, index) for index, change in enumerate(changes)
    ]
    if not value["has_difference"] and validated_changes:
        raise ValueError("changes must be empty when has_difference is false")
    return {
        "has_difference": value["has_difference"],
        "summary": value["summary"].strip(),
        "changes": validated_changes,
    }


def save_pair_results(
    output_dir: str | Path,
    pair_id: str,
    before: np.ndarray,
    after: np.ndarray,
    raw_content: str,
    result: dict[str, Any],
    *,
    model: str,
    before_path: Path,
    after_path: Path,
) -> dict[str, Path]:
    pair_dir = Path(output_dir) / f"pair_{pair_id}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    before_annotated = draw_bboxes(
        before,
        result["changes"],
        "before_bbox",
        "BEFORE",
        result["has_difference"],
    )
    after_annotated = draw_bboxes(
        after,
        result["changes"],
        "after_bbox",
        "AFTER",
        result["has_difference"],
    )
    comparison = np.hstack((before_annotated, after_annotated))

    paths = {
        "before": pair_dir / "01_before.jpg",
        "after": pair_dir / "02_after.jpg",
        "before_bboxes": pair_dir / "03_before_bboxes.jpg",
        "after_bboxes": pair_dir / "04_after_bboxes.jpg",
        "comparison": pair_dir / "05_qwen_comparison.jpg",
        "raw": pair_dir / "qwen_raw.txt",
        "result": pair_dir / "result.json",
    }
    for key, image in (
        ("before", before),
        ("after", after),
        ("before_bboxes", before_annotated),
        ("after_bboxes", after_annotated),
        ("comparison", comparison),
    ):
        write_image(paths[key], image)
    paths["raw"].write_text(raw_content, encoding="utf-8")
    metadata = {
        "pair_id": pair_id,
        "before": str(before_path),
        "after": str(after_path),
        "image_size": [TARGET_SIZE[0], TARGET_SIZE[1]],
        "bbox_format": ["x", "y", "width", "height"],
        "model": model,
        **result,
    }
    paths["result"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {key: path.resolve() for key, path in paths.items()}


def draw_bboxes(
    image: np.ndarray,
    changes: Sequence[dict[str, Any]],
    bbox_key: str,
    title: str,
    has_difference: bool,
) -> np.ndarray:
    output = image.copy()
    color = (0, 0, 255) if has_difference else (0, 180, 0)
    count = sum(change[bbox_key] is not None for change in changes)
    cv2.rectangle(output, (0, 0), (output.shape[1], 44), (20, 20, 20), cv2.FILLED)
    cv2.putText(
        output,
        f"{title} | QWEN CHANGE BBOXES: {count}",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    for index, change in enumerate(changes, start=1):
        bbox = change[bbox_key]
        if bbox is None:
            continue
        x, y, width, height = bbox
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 0, 255), 3)
        cv2.putText(
            output,
            f"#{index} {change['type']} [{x},{y},{width},{height}]",
            (max(4, x), max(68, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def write_image(path: str | Path, image: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(target.suffix or ".jpg", image)
    if not success:
        raise ValueError(f"cannot encode output image: {target}")
    encoded.tofile(target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="directory containing *_1/*_2 pairs")
    parser.add_argument("--output-dir", type=Path, help="default: INPUT/qwen_results")
    parser.add_argument("--pair", action="append", help="only process this pair ID")
    parser.add_argument("--url", default=QWEN3_URL, help="Qwen chat completions URL")
    parser.add_argument("--model", default=QWEN3_MODEL, help="Qwen model name")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per pair")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir or args.input_dir / "qwen_results"
    try:
        pairs = discover_pairs(args.input_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.pair:
        selected = set(args.pair)
        pairs = [pair for pair in pairs if pair[0] in selected]
    if not pairs:
        print("ERROR: no complete *_1/*_2 pairs found", file=sys.stderr)
        return 2

    api_key = os.getenv("QWEN_API_KEY", "").strip() or None
    summary: list[dict[str, Any]] = []
    failures = 0
    for pair_id, before_path, after_path in pairs:
        print(f"[{pair_id}] comparing {before_path.name} -> {after_path.name} ...")
        pair_dir = output_dir / f"pair_{pair_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        try:
            before = read_720p(before_path)
            after = read_720p(after_path)
            raw = request_qwen(
                before,
                after,
                url=args.url,
                model=args.model,
                timeout=args.timeout,
                api_key=api_key,
            )
            (pair_dir / "qwen_raw.txt").write_text(raw, encoding="utf-8")
            result = parse_qwen_result(raw)
            artifacts = save_pair_results(
                output_dir,
                pair_id,
                before,
                after,
                raw,
                result,
                model=args.model,
                before_path=before_path,
                after_path=after_path,
            )
            summary.append({"pair_id": pair_id, "status": "ok", **result})
            print(
                f"[{pair_id}] has_difference={result['has_difference']} "
                f"changes={len(result['changes'])} output={artifacts['result'].parent}"
            )
        except (RuntimeError, ValueError, OSError) as exc:
            failures += 1
            error = {"pair_id": pair_id, "status": "error", "error": str(exc)}
            summary.append(error)
            (pair_dir / "error.json").write_text(
                json.dumps(error, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[{pair_id}] ERROR: {exc}", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"done: pairs={len(pairs)} failures={failures} output={output_dir.resolve()}")
    return 1 if failures else 0


def _validate_change(change: Any, index: int) -> dict[str, Any]:
    if not isinstance(change, dict):
        raise ValueError(f"changes[{index}] must be an object")
    change_type = change.get("type")
    if change_type not in ALLOWED_CHANGE_TYPES:
        raise ValueError(f"changes[{index}].type is invalid: {change_type}")
    object_name = change.get("object")
    description = change.get("description")
    confidence = change.get("confidence")
    if not isinstance(object_name, str) or not object_name.strip():
        raise ValueError(f"changes[{index}].object must be a non-empty string")
    if not isinstance(description, str):
        raise ValueError(f"changes[{index}].description must be a string")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError(f"changes[{index}].confidence must be a number")
    return {
        "type": change_type,
        "object": object_name.strip(),
        "before_bbox": _validate_bbox(change.get("before_bbox"), index, "before_bbox"),
        "after_bbox": _validate_bbox(change.get("after_bbox"), index, "after_bbox"),
        "confidence": min(1.0, max(0.0, float(confidence))),
        "description": description.strip(),
    }


def _validate_bbox(value: Any, index: int, name: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"changes[{index}].{name} must be [x,y,width,height] or null")
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        raise ValueError(f"changes[{index}].{name} values must be numeric")
    x, y, width, height = (round(float(item)) for item in value)
    x = min(TARGET_SIZE[0] - 1, max(0, x))
    y = min(TARGET_SIZE[1] - 1, max(0, y))
    width = min(TARGET_SIZE[0] - x, max(1, width))
    height = min(TARGET_SIZE[1] - y, max(1, height))
    return [x, y, width, height]


def _natural_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


if __name__ == "__main__":
    raise SystemExit(main())
