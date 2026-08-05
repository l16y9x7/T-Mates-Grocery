from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CATALOG_VERSION = "retail_perception_sku_2026-08-04_v1"


# 层号从上到下，列号为面对货架时从左到右。
LAYOUT: dict[str, dict[int, list[str]]] = {
    "H1_F": {
        1: [
            "NFC桔汁",
            "蒙牛纯牛奶",
            "纯甄酸奶",
            "百草味海苔肉松蛋卷",
            "三只松鼠每日坚果",
            "费列罗巧克力",
            "双汇王中王火腿肠",
        ],
        2: [
            "妙芙香芋牛奶味",
            "妙芙绵醇奶油味",
            "妙芙巧克力味",
            "薯愿非油炸清新番茄味",
            "薯愿非油炸韩国泡菜味",
            "薯愿非油炸香烤原味",
            "品客薯片烧烤牛排味",
            "品客原味",
            "品客酸乳酪洋葱味",
            "奥利奥香甜不腻",
            "奥利奥冰淇淋抹茶味",
            "奥利奥浓醇巧克力味",
            "奥利奥白桃乌龙味",
        ],
        3: [
            "Lays乐事薯片墨西哥鸡汁番茄味",
            "Lays乐事薯片经典原味",
            "Lays乐事薯片黄瓜味",
            "Lays乐事薯片意大利香浓红烩味",
            "Lays乐事薯片青柠味",
            "呀！土豆番茄酱味",
            "好友趣韩国泡菜味",
        ],
        4: [
            "卫龙辣条",
            "百醇草莓香草味",
            "百醇巧克力味",
            "上好佳鲜虾条",
            "浪味仙",
        ],
        5: [
            "好丽友派",
            "吐司方方原味",
            "好友趣蜂蜜黄油味",
            "上好佳鲜虾条",
            "浪味仙",
        ],
    },
    "H1_B": {
        1: [
            "可孚棉签",
            "小鹿妈妈牙线",
            "得宝纸巾",
            "草原红太阳烧烤料原味",
            "草原红太阳烧烤料香辣味",
            "草原红太阳烧烤酱香辣",
            "草原红太阳烧烤酱原味",
        ],
        2: [
            "海氏海诺创口贴",
            "德佑湿巾",
            "心相印纸巾",
            "合味道海鲜味",
            "合味道五香牛肉味",
            "农心碗面",
            "健食力低脂香肠",
        ],
        3: [
            "妙洁海绵百洁布",
            "心相印纸巾",
            "康师傅香辣牛肉面",
            "康师傅鲜虾鱼板面",
            "康师傅老坛酸菜牛肉面",
            "好人家火锅底料",
        ],
        4: [
            "纯棉酒店大毛巾",
            "京东京造毛巾",
            "三色藜麦",
            "慢碳十色糙米",
            "高纤七色糙米",
            "山西黄小米",
        ],
        5: [
            "界面医疗医用外科口罩",
            "中盐精制盐",
            "小苏打",
            "镇江香醋",
            "蒸鱼豉油",
            "薄盐生抽",
        ],
    },
    "H2_F": {
        1: [
            "可口可乐罐装",
            "雪碧罐装",
            "芬达罐装",
            "百事可乐瓶装",
            "7喜瓶装",
            "水溶C100瓶装",
        ],
        2: [
            "小青柠汁饮料",
            "美年达瓶装",
            "名仁苏打水饮料",
            "东方树叶茉莉花茶",
            "绿豆冰沙",
        ],
        3: [
            "外星人电解质水椰子口味",
            "外星人电解质水青柠口味",
            "外星人电解质水白桃口味0糖",
            "外星人电解质水西柚口味",
            "尖叫",
            "百岁山矿泉水",
        ],
        4: [
            "脉动观梅止渴饮",
            "脉动芒果口味",
            "脉动菠萝口味",
            "脉动猫薄荷瓶",
            "外星人电解质水青柠口味0糖",
            "外星人电解质水青柠口味",
            "宝矿力水特",
            "百岁山矿泉水",
        ],
        5: [
            "脉动观梅止渴饮",
            "脉动芒果口味",
            "脉动菠萝口味",
            "脉动猫薄荷瓶",
            "外星人电解质水白桃口味0糖",
            "宝矿力水特",
            "百岁山矿泉水",
        ],
    },
    "H2_B": {
        1: [
            "舒肤佳香皂纯白清香型",
            "舒肤佳香皂芦荟呵护香型",
            "舒肤佳香皂柠檬清新香型",
            "汰渍肥皂",
            "杯子",
        ],
        2: [
            "舒克牙膏竹炭薄荷",
            "舒克牙膏柠檬百香果",
            "舒克牙膏海盐薄荷",
            "冷酸灵云感觉牙刷",
            "杯子",
        ],
        3: [
            "Dove沐浴泡泡樱花甜香",
            "Dove沐浴泡泡白桃果香",
            "Dove沐浴泡泡清甜奶香",
            "威露士",
            "Dove沐浴乳",
            "拖鞋",
        ],
        4: [
            "Dove沐浴泡泡樱花甜香袋装",
            "半亩花田洗发水",
            "心相印厨房纸巾",
        ],
        5: [
            "拖鞋",
            "心相印厨房纸巾",
        ],
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
    ordered_names: list[str] = []
    for levels in LAYOUT.values():
        for level in sorted(levels):
            for name in levels[level]:
                if name not in ordered_names:
                    ordered_names.append(name)

    sku_by_name = {
        name: f"SKU_{index:03d}" for index, name in enumerate(ordered_names, start=1)
    }

    locations_by_name: dict[str, list[str]] = {name: [] for name in ordered_names}
    for shelf_face, levels in LAYOUT.items():
        shelf_id, face = shelf_face.split("_")
        for level in sorted(levels):
            for column, name in enumerate(levels[level], start=1):
                locations_by_name[name].append(
                    f"{shelf_id}_{face}_L{level}_C{column:02d}"
                )

    existing_images = load_existing_images()
    products = [
        {
            "sku_id": sku_by_name[name],
            "name": name,
            "images": existing_images.get(name, []),
            "locations": locations_by_name[name],
        }
        for name in ordered_names
    ]

    return {
        "schema_version": "1.0",
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
        f"{location_count} 个标准位置。"
    )


if __name__ == "__main__":
    main()
