from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
SKU_PATTERN = re.compile(r"^SKU_\d{3}$")
LOCATION_PATTERN = re.compile(r"^H[12]_[FB]_L[1-5]_C\d{2}$")
PRODUCT_FIELDS = {"sku_id", "name", "images", "locations"}


def load_json(filename: str) -> dict:
    return json.loads((ROOT / filename).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    catalog = load_json("products.json")
    products = catalog.get("products", [])

    sku_ids = [item.get("sku_id") for item in products]
    names = [item.get("name") for item in products]

    require(all(isinstance(value, str) and value for value in sku_ids), "存在空 SKU ID")
    require(all(isinstance(value, str) and value for value in names), "存在空商品名称")
    require(all(SKU_PATTERN.fullmatch(value) for value in sku_ids), "存在格式错误的 SKU ID")
    require(len(sku_ids) == len(set(sku_ids)), "存在重复 SKU ID")
    require(len(names) == len(set(names)), "存在重复商品名称")

    all_locations: list[str] = []
    repeated_product_count = 0
    for product in products:
        require(set(product) == PRODUCT_FIELDS, f"{product.get('sku_id')} 包含非精简字段")
        images = product["images"]
        locations = product["locations"]
        require(isinstance(images, list), f"{product['sku_id']} 的 images 必须是数组")
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
        require(
            isinstance(locations, list) and locations,
            f"{product['sku_id']} 必须至少有一个位置",
        )
        require(len(locations) == len(set(locations)), f"{product['sku_id']} 存在重复位置")
        require(
            all(isinstance(location, str) and LOCATION_PATTERN.fullmatch(location) for location in locations),
            f"{product['sku_id']} 存在格式错误的位置",
        )
        all_locations.extend(locations)
        repeated_product_count += len(locations) > 1

    duplicate_locations = [
        location for location, count in Counter(all_locations).items() if count > 1
    ]
    require(not duplicate_locations, f"位置被多个 SKU 占用: {duplicate_locations}")

    print(
        f"校验通过：{len(products)} 个 SKU，{len(all_locations)} 个位置，"
        f"{repeated_product_count} 个 SKU 具有多个标准货位。"
    )


if __name__ == "__main__":
    main()
