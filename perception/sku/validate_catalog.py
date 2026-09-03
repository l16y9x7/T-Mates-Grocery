from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
IMAGES_ROOT = ROOT / "images_new"
SKU_PATTERN = re.compile(r"^SKU_\d{3}$")
LOCATION_PATTERN = re.compile(r"^H[1-3]_L0[1-5]_C\d{2}$")
PRODUCT_FIELDS = {"sku_id", "name", "images", "locations", "inventory"}


def load_json(filename: str) -> dict:
    return json.loads((ROOT / filename).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    catalog = load_json("products.json")
    products = catalog.get("products", [])

    require(catalog.get("schema_version") == "2.0", "schema_version 必须是 2.0")
    require(
        isinstance(catalog.get("catalog_version"), str)
        and bool(catalog["catalog_version"]),
        "catalog_version 不能为空",
    )

    sku_ids = [item.get("sku_id") for item in products]
    names = [item.get("name") for item in products]

    require(all(isinstance(value, str) and value for value in sku_ids), "存在空 SKU ID")
    require(all(isinstance(value, str) and value for value in names), "存在空商品名称")
    require(all(SKU_PATTERN.fullmatch(value) for value in sku_ids), "存在格式错误的 SKU ID")
    require(len(sku_ids) == len(set(sku_ids)), "存在重复 SKU ID")
    require(len(names) == len(set(names)), "存在重复商品名称")

    all_locations: list[str] = []
    image_count = 0
    image_sizes: list[tuple[int, int]] = []
    repeated_product_count = 0
    for product in products:
        require(set(product) == PRODUCT_FIELDS, f"{product.get('sku_id')} 包含非精简字段")
        images = product["images"]
        locations = product["locations"]
        inventory = product["inventory"]
        require(
            isinstance(images, list) and images,
            f"{product['sku_id']} 必须至少有一张图片",
        )
        require(
            all(isinstance(image, str) and image for image in images),
            f"{product['sku_id']} 存在无效图片路径",
        )
        for image in images:
            normalized_image = image.replace("\\", "/")
            image_path = PurePosixPath(normalized_image)
            require(
                normalized_image.startswith("images/")
                and not image_path.is_absolute()
                and ".." not in image_path.parts,
                f"{product['sku_id']} 的图片必须是 images/ 下的相对路径",
            )
            physical_image_path = (
                image_path.with_suffix(".jpg")
                if image_path.suffix.lower() == ".png"
                else image_path
            )
            resolved_image = IMAGES_ROOT.joinpath(*physical_image_path.parts[1:])
            require(
                resolved_image.is_file() and resolved_image.stat().st_size > 0,
                f"{product['sku_id']} 的图片不存在或为空: {image}",
            )
            try:
                with Image.open(resolved_image) as source_image:
                    width, height = source_image.size
                    source_image.verify()
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(
                    f"{product['sku_id']} 的图片无法读取: {image}"
                ) from error
            require(
                width >= 32 and height >= 32,
                f"{product['sku_id']} 的图片分辨率过低: {image}",
            )
            image_count += 1
            image_sizes.append((width, height))
        require(
            isinstance(locations, list) and locations,
            f"{product['sku_id']} 必须至少有一个位置",
        )
        require(len(locations) == len(set(locations)), f"{product['sku_id']} 存在重复位置")
        require(
            all(isinstance(location, str) and LOCATION_PATTERN.fullmatch(location) for location in locations),
            f"{product['sku_id']} 存在格式错误的位置",
        )
        require(isinstance(inventory, list), f"{product['sku_id']} 的 inventory 必须是数组")
        require(len(inventory) == len(set(inventory)), f"{product['sku_id']} 的 inventory 存在重复位置")
        require(
            all(isinstance(location, str) and LOCATION_PATTERN.fullmatch(location) for location in inventory),
            f"{product['sku_id']} 的 inventory 存在格式错误的位置",
        )
        require(
            set(inventory).issubset(set(locations)),
            f"{product['sku_id']} 的 inventory 包含非标准位置",
        )
        all_locations.extend(locations)
        repeated_product_count += len(locations) > 1

    duplicate_locations = [
        location for location, count in Counter(all_locations).items() if count > 1
    ]
    require(not duplicate_locations, f"位置被多个 SKU 占用: {duplicate_locations}")

    row_columns: dict[str, list[int]] = {}
    for location in all_locations:
        row_id, column = location.rsplit("_C", 1)
        row_columns.setdefault(row_id, []).append(int(column))
    for row_id, columns in row_columns.items():
        ordered = sorted(columns)
        require(
            ordered == list(range(1, ordered[-1] + 1)),
            f"{row_id} 的列号不连续: {ordered}",
        )

    print(
        f"校验通过：{len(products)} 个 SKU，{image_count} 张图片，"
        f"{len(all_locations)} 个位置，"
        f"{repeated_product_count} 个 SKU 具有多个标准货位。"
    )
    smallest_image = min(image_sizes, key=lambda size: size[0] * size[1])
    largest_image = max(image_sizes, key=lambda size: size[0] * size[1])
    print(
        f"最小/最大图片：{smallest_image[0]}x{smallest_image[1]} / "
        f"{largest_image[0]}x{largest_image[1]}。"
    )


if __name__ == "__main__":
    main()
