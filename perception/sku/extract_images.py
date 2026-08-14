from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT.parents[2] / "商超场景-“零售服务岗”摆放标准图.docx"
CATALOG_PATH = ROOT / "products.json"
IMAGES_DIR = ROOT / "images_new"


@dataclass(frozen=True)
class EmbeddedImage:
    part_name: str
    blob: bytes
    crop_left: int = 0
    crop_top: int = 0
    crop_right: int = 0
    crop_bottom: int = 0

    @property
    def source_suffix(self) -> str:
        return Path(self.part_name).suffix.lower()


def cell_text(cell) -> str:
    return "".join(
        node.text or "" for node in cell._element.iter(qn("w:t"))
    ).strip()


def normalized_name_key(name: str) -> str:
    return name.strip().replace("’", "").replace("'", "").replace("·", "")


def cell_images(document: Document, cell) -> list[EmbeddedImage]:
    images: list[EmbeddedImage] = []
    for blip in cell._element.iter(qn("a:blip")):
        relation_id = blip.get(qn("r:embed"))
        if not relation_id:
            continue

        image_part = document.part.related_parts[relation_id]
        source_rect = blip.getparent().find(qn("a:srcRect"))
        crop = source_rect.attrib if source_rect is not None else {}
        images.append(
            EmbeddedImage(
                part_name=str(image_part.partname),
                blob=image_part.blob,
                crop_left=int(crop.get("l", 0)),
                crop_top=int(crop.get("t", 0)),
                crop_right=int(crop.get("r", 0)),
                crop_bottom=int(crop.get("b", 0)),
            )
        )
    return images


def extract_name_image_map(document: Document) -> dict[str, list[EmbeddedImage]]:
    occurrences: dict[str, list[EmbeddedImage]] = defaultdict(list)

    # 前两个表格是场景示意图；从第三个表格开始，每个商品图片行的上一行是商品名。
    for table in document.tables[2:]:
        for row_index in range(1, len(table.rows)):
            image_row = table.rows[row_index]
            if not any(cell_images(document, cell) for cell in image_row.cells):
                continue

            name_row = table.rows[row_index - 1]
            for name_cell, image_cell in zip(name_row.cells, image_row.cells):
                name = cell_text(name_cell)
                images = cell_images(document, image_cell)
                if not name and not images:
                    continue
                if not name or len(images) != 1:
                    raise ValueError(
                        f"无法建立商品与图片的一一对应：table row {row_index + 1}"
                    )
                occurrences[name].append(images[0])

    unique_by_name: dict[str, list[EmbeddedImage]] = {}
    for name, images in occurrences.items():
        seen: set[tuple[str, int, int, int, int]] = set()
        unique_images: list[EmbeddedImage] = []
        for image in images:
            key = (
                image.part_name,
                image.crop_left,
                image.crop_top,
                image.crop_right,
                image.crop_bottom,
            )
            if key not in seen:
                seen.add(key)
                unique_images.append(image)
        unique_by_name[name] = unique_images
    return unique_by_name


def save_image(image: EmbeddedImage, destination: Path) -> None:
    with Image.open(io.BytesIO(image.blob)) as source:
        rendered = ImageOps.exif_transpose(source)
        width, height = rendered.size
        left = round(width * image.crop_left / 100_000)
        top = round(height * image.crop_top / 100_000)
        right = width - round(width * image.crop_right / 100_000)
        bottom = height - round(height * image.crop_bottom / 100_000)
        if left >= right or top >= bottom:
            raise ValueError(f"无效裁剪区域：{image.part_name}")
        rendered = rendered.crop((left, top, right, bottom))

        if destination.suffix.lower() == ".png":
            rendered.save(destination, format="PNG", optimize=True)
        else:
            if rendered.mode != "RGB":
                rendered = rendered.convert("RGB")
            rendered.save(
                destination,
                format="JPEG",
                quality=95,
                subsampling=0,
                optimize=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="从标准摆放 DOCX 提取 SKU 图片")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    products = catalog["products"]
    products_by_name = {product["name"]: product for product in products}
    canonical_name_by_key = {
        normalized_name_key(name): name for name in products_by_name
    }
    if len(canonical_name_by_key) != len(products_by_name):
        raise ValueError("商品名在标点归一化后发生冲突")

    document = Document(args.source)
    source_images_by_name = extract_name_image_map(document)
    images_by_name: dict[str, list[EmbeddedImage]] = {}
    unexpected_names: list[str] = []
    for source_name, source_images in source_images_by_name.items():
        canonical_name = canonical_name_by_key.get(normalized_name_key(source_name))
        if canonical_name is None:
            unexpected_names.append(source_name)
            continue
        images_by_name.setdefault(canonical_name, []).extend(source_images)

    missing_names = sorted(set(products_by_name) - set(images_by_name))
    if missing_names or unexpected_names:
        raise ValueError(
            f"商品名不一致；缺少图片={missing_names}；未入库商品={unexpected_names}"
        )

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for name, product in products_by_name.items():
        sku_id = product["sku_id"]
        image_paths: list[str] = []
        for image_index, image in enumerate(images_by_name[name], start=1):
            index_suffix = "" if len(images_by_name[name]) == 1 else f"_{image_index:02d}"
            filename = f"{sku_id}{index_suffix}.jpg"
            destination = IMAGES_DIR / filename
            save_image(image, destination)
            image_paths.append(f"images/{filename}")
            saved_count += 1
        product["images"] = image_paths

    CATALOG_PATH.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已为 {len(products)} 个 SKU 保存 {saved_count} 张参考图片。")


if __name__ == "__main__":
    main()
