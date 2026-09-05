from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CATALOG_VERSION = "retail_perception_sku_2026-09-01_v2"


# SKU 编号保持历史稳定，不因商品减少或货位调整而重新编号。
PRODUCT_SKUS: dict[str, str] = {
    "NFC桔汁": "SKU_001",
    "蒙牛纯牛奶": "SKU_002",
    "纯甄酸奶": "SKU_003",
    "品客薯片烧烤牛排味": "SKU_014",
    "品客原味": "SKU_015",
    "品客酸乳酪洋葱味": "SKU_016",
    "奥利奥香甜不腻": "SKU_017",
    "奥利奥浓醇巧克力味": "SKU_019",
    "奥利奥白桃乌龙味": "SKU_020",
    "呀！土豆番茄酱味": "SKU_026",
    "好友趣韩国泡菜味": "SKU_027",
    "上好佳鲜虾条": "SKU_031",
    "浪味仙": "SKU_032",
    "京东京造毛巾": "SKU_056",
    "可口可乐罐装": "SKU_067",
    "雪碧罐装": "SKU_068",
    "芬达罐装": "SKU_069",
    "百事可乐瓶装": "SKU_070",
    "7喜瓶装": "SKU_071",
    "水溶C100瓶装": "SKU_072",
    "美年达瓶装": "SKU_074",
    "外星人电解质水椰子口味": "SKU_078",
    "外星人电解质水青柠口味": "SKU_079",
    "外星人电解质水白桃口味0糖": "SKU_080",
    "尖叫": "SKU_082",
    "百岁山矿泉水": "SKU_083",
    "脉动观梅止渴饮": "SKU_084",
    "脉动芒果口味": "SKU_085",
    "脉动菠萝口味": "SKU_086",
    "脉动猫薄荷瓶": "SKU_087",
    "舒肤佳香皂芦荟呵护香型": "SKU_091",
    "舒肤佳香皂柠檬清新香型": "SKU_092",
    "汰渍肥皂": "SKU_093",
    "舒克牙膏竹炭薄荷": "SKU_095",
    "舒克牙膏柠檬百香果": "SKU_096",
    "舒克牙膏海盐薄荷": "SKU_097",
    "Dove沐浴泡泡樱花甜香": "SKU_099",
    "Dove沐浴泡泡白桃果香": "SKU_100",
    "Dove沐浴泡泡清甜奶香": "SKU_101",
    "Dove沐浴乳": "SKU_103",
    "拖鞋": "SKU_104",
    "半亩花田洗发水": "SKU_106",
    "心相印厨房纸巾": "SKU_107",
}

# Keep these names in LAYOUT so the physical columns retain their real slot
# IDs, but do not publish them through the orderable SKU catalog.
NON_ORDERABLE_PRODUCTS = {"脉动猫薄荷瓶"}


# 层号从上到下；每个数组元素对应一个物理陈列列，允许相邻列为同一 SKU。
LAYOUT: dict[str, dict[int, list[str]]] = {
    "H1": {
        1: [
            "Dove沐浴乳",
            "Dove沐浴乳",
            "舒肤佳香皂芦荟呵护香型",
            "舒肤佳香皂柠檬清新香型",
            "汰渍肥皂",
        ],
        2: [
            "舒克牙膏竹炭薄荷",
            "舒克牙膏柠檬百香果",
            "舒克牙膏海盐薄荷",
            "半亩花田洗发水",
            "半亩花田洗发水",
        ],
        3: [
            "Dove沐浴泡泡樱花甜香",
            "Dove沐浴泡泡白桃果香",
            "Dove沐浴泡泡清甜奶香",
            "京东京造毛巾",
        ],
        4: ["心相印厨房纸巾", "心相印厨房纸巾", "心相印厨房纸巾"],
        5: ["拖鞋", "拖鞋", "拖鞋"],
    },
    "H2": {
        1: [
            "可口可乐罐装",
            "可口可乐罐装",
            "可口可乐罐装",
            "雪碧罐装",
            "雪碧罐装",
            "芬达罐装",
        ],
        2: [
            "百事可乐瓶装",
            "百事可乐瓶装",
            "7喜瓶装",
            "7喜瓶装",
            "美年达瓶装",
            "美年达瓶装",
        ],
        3: [
            "外星人电解质水椰子口味",
            "外星人电解质水青柠口味",
            "外星人电解质水白桃口味0糖",
            "百岁山矿泉水",
            "百岁山矿泉水",
            "百岁山矿泉水",
        ],
        4: [
            "脉动观梅止渴饮",
            "脉动芒果口味",
            "脉动菠萝口味",
            "脉动猫薄荷瓶",
            "尖叫",
            "尖叫",
        ],
        5: [
            "脉动观梅止渴饮",
            "脉动芒果口味",
            "尖叫",
            "尖叫",
            "水溶C100瓶装",
            "水溶C100瓶装",
        ],
    },
    "H3": {
        1: [
            "NFC桔汁",
            "NFC桔汁",
            "蒙牛纯牛奶",
            "蒙牛纯牛奶",
            "纯甄酸奶",
            "纯甄酸奶",
        ],
        2: [
            "品客薯片烧烤牛排味",
            "品客原味",
            "品客酸乳酪洋葱味",
            "奥利奥香甜不腻",
            "奥利奥浓醇巧克力味",
            "奥利奥白桃乌龙味",
        ],
        3: [
            "呀！土豆番茄酱味",
            "呀！土豆番茄酱味",
            "好友趣韩国泡菜味",
            "好友趣韩国泡菜味",
        ],
        4: ["上好佳鲜虾条", "上好佳鲜虾条", "浪味仙", "浪味仙"],
        5: ["上好佳鲜虾条", "上好佳鲜虾条", "浪味仙", "浪味仙"],
    },
}


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_existing_images() -> dict[str, list[str]]:
    catalog_path = ROOT / "products.json"
    if not catalog_path.exists():
        return {}
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    return {
        product["name"]: list(product.get("images", []))
        for product in catalog.get("products", [])
        if isinstance(product, dict) and isinstance(product.get("name"), str)
    }


def build() -> dict[str, object]:
    layout_names = {
        name
        for levels in LAYOUT.values()
        for row in levels.values()
        for name in row
    }
    if layout_names != set(PRODUCT_SKUS):
        missing = sorted(set(PRODUCT_SKUS) - layout_names)
        unknown = sorted(layout_names - set(PRODUCT_SKUS))
        raise ValueError(f"货架布局与 SKU 清单不一致: missing={missing}, unknown={unknown}")

    locations_by_name: dict[str, list[str]] = {
        name: [] for name in PRODUCT_SKUS
    }
    for shelf_id, levels in LAYOUT.items():
        for level in sorted(levels):
            for column, name in enumerate(levels[level], start=1):
                locations_by_name[name].append(
                    f"{shelf_id}_L{level:02d}_C{column:02d}"
                )

    existing_images = load_existing_images()
    products = []
    for name, sku_id in PRODUCT_SKUS.items():
        if name in NON_ORDERABLE_PRODUCTS:
            continue
        images = existing_images.get(name) or [f"images/{sku_id}.jpg"]
        products.append(
            {
                "sku_id": sku_id,
                "name": name,
                "images": images,
                "locations": locations_by_name[name],
                "inventory": list(locations_by_name[name]),
            }
        )

    return {
        "schema_version": "2.0",
        "catalog_version": CATALOG_VERSION,
        "products": products,
    }


def main() -> None:
    catalog = build()
    write_json(ROOT / "products.json", catalog)
    location_count = sum(
        len(product["locations"]) for product in catalog["products"]
    )
    print(
        f"已生成 {len(catalog['products'])} 个 SKU，"
        f"{location_count} 个物理陈列列。"
    )


if __name__ == "__main__":
    main()
