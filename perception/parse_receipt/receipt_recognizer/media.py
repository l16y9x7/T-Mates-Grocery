"""Local-only image normalization and PDF rendering."""

from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Iterable

from .errors import InputFileError


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SUPPORTED_INPUT_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}


def prepare_input(
    input_path: Path,
    *,
    max_edge: int = 2200,
    jpeg_quality: int = 90,
    pdf_dpi: int = 180,
    max_pdf_pages: int = 1,
) -> list[str]:
    """Return JPEG data URLs without writing normalized images to disk."""
    path = input_path.expanduser().resolve()
    if not path.is_file():
        raise InputFileError(f"输入文件不存在：{path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
        raise InputFileError(f"仅支持 {supported}，实际为：{suffix or '无扩展名'}")

    if suffix == ".pdf":
        rendered = render_pdf_pages(
            path,
            dpi=pdf_dpi,
            max_pages=max_pdf_pages,
        )
        return [
            image_bytes_to_data_url(
                page,
                max_edge=max_edge,
                jpeg_quality=jpeg_quality,
            )
            for page in rendered
        ]

    return [
        image_path_to_data_url(
            path,
            max_edge=max_edge,
            jpeg_quality=jpeg_quality,
        )
    ]


def image_path_to_data_url(
    path: Path,
    *,
    max_edge: int = 2200,
    jpeg_quality: int = 90,
) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise InputFileError("图片必须是 JPEG 或 PNG。")
    return image_bytes_to_data_url(
        path.read_bytes(),
        max_edge=max_edge,
        jpeg_quality=jpeg_quality,
    )


def image_bytes_to_data_url(
    raw: bytes,
    *,
    max_edge: int = 2200,
    jpeg_quality: int = 90,
) -> str:
    if max_edge < 256:
        raise InputFileError("max_edge 不能小于 256。")
    if not 1 <= jpeg_quality <= 95:
        raise InputFileError("jpeg_quality 必须在 1 到 95 之间。")

    suffix = _detect_image_suffix(raw)
    with tempfile.TemporaryDirectory(prefix="receipt-image-bytes-") as temp_dir:
        try:
            return _normalize_image_bytes_with_pillow(
                raw,
                max_edge=max_edge,
                jpeg_quality=jpeg_quality,
            )
        except ImportError:
            pass
        except OSError as exc:
            raise InputFileError(f"图片转换失败：{exc}") from exc

        # Fallback for the original macOS local workflow where Pillow may not
        # be installed yet. Server deployments should install Pillow.
        input_path = Path(temp_dir) / f"input{suffix}"
        input_path.write_bytes(raw)
        return _normalize_image_path(
            input_path,
            max_edge=max_edge,
            jpeg_quality=jpeg_quality,
        )


def _normalize_image_bytes_with_pillow(
    raw: bytes,
    *,
    max_edge: int,
    jpeg_quality: int,
) -> str:
    try:
        from PIL import Image, ImageOps
    except ImportError:
        raise

    with Image.open(BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")

        width, height = image.size
        longest = max(width, height)
        if longest > max_edge:
            scale = max_edge / longest
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )

        output = BytesIO()
        image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _normalize_image_path(
    input_path: Path,
    *,
    max_edge: int,
    jpeg_quality: int,
) -> str:
    executable = shutil.which("sips")
    if executable is None:
        raise InputFileError(
            "找不到 macOS sips；当前零依赖图片预处理仅支持 macOS。"
        )

    with tempfile.TemporaryDirectory(prefix="receipt-image-") as temp_dir:
        output_path = Path(temp_dir) / "normalized.jpg"
        command = [
            executable,
            "--setProperty",
            "format",
            "jpeg",
            "--setProperty",
            "formatOptions",
            str(jpeg_quality),
            "--resampleHeightWidthMax",
            str(max_edge),
            str(input_path),
            "--out",
            str(output_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not output_path.is_file():
            message = result.stderr.strip() or result.stdout.strip()
            raise InputFileError(
                f"图片转换失败：{message or '未知 sips 错误'}"
            )

        normalized = output_path.read_bytes()
        if not normalized.startswith(b"\xff\xd8\xff"):
            raise InputFileError("sips 未生成有效 JPEG。")
        encoded = base64.b64encode(normalized).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


def _detect_image_suffix(raw: bytes) -> str:
    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    raise InputFileError("文件内容不是有效的 JPEG/PNG 图片。")


def render_pdf_pages(
    pdf_path: Path,
    *,
    dpi: int = 180,
    max_pages: int = 4,
) -> list[bytes]:
    if dpi < 72 or dpi > 300:
        raise InputFileError("pdf_dpi 必须在 72 到 300 之间。")
    if max_pages < 1:
        raise InputFileError("max_pdf_pages 必须大于 0。")

    executable = shutil.which("pdftoppm")
    if executable is None:
        raise InputFileError(
            "找不到 pdftoppm；macOS 可运行 brew install poppler。"
        )

    with tempfile.TemporaryDirectory(prefix="receipt-pdf-") as temp_dir:
        prefix = Path(temp_dir) / "page"
        command = [
            executable,
            "-jpeg",
            "-r",
            str(dpi),
            "-f",
            "1",
            "-l",
            str(max_pages),
            str(pdf_path),
            str(prefix),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or "未知 pdftoppm 错误"
            raise InputFileError(f"PDF 渲染失败：{message}")

        pages = sorted(Path(temp_dir).glob("page-*.jpg"))
        if not pages:
            raise InputFileError("PDF 没有可渲染页面。")
        return [page.read_bytes() for page in pages]


def multimodal_content(data_urls: Iterable[str], prompt: str) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for data_url in data_urls:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        )
    return content
