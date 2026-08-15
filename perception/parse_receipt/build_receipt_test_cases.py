"""Build deterministic two-product receipt-matching test cases.

The generated inputs intentionally mix two recognition styles:

1. a short product/brand name plus a semantic type or flavor specification;
2. the complete catalog name plus an invented numeric size specification.

Run from any directory with::

    python parse_receipt/build_receipt_test_cases.py
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "sku" / "products.json"
OUTPUT_PATH = Path(__file__).with_name("receipt_test_cases.json")
CASE_COUNT = 150


# The two strings for every entry must concatenate to the canonical catalog name.
# Products without a natural name/type boundary intentionally stay in their complete
# catalog form and receive an invented numeric size instead.
SPLIT_VARIANTS: dict[str, tuple[str, str]] = {
    "SKU_004": ("百草味", "海苔肉松蛋卷"),
    "SKU_008": ("妙芙", "香芋牛奶味"),
    "SKU_009": ("妙芙", "绵醇奶油味"),
    "SKU_010": ("妙芙", "巧克力味"),
    "SKU_011": ("薯愿", "非油炸清新番茄味"),
    "SKU_012": ("薯愿", "非油炸韩国泡菜味"),
    "SKU_013": ("薯愿", "非油炸香烤原味"),
    "SKU_014": ("品客", "薯片烧烤牛排味"),
    "SKU_015": ("品客", "原味"),
    "SKU_016": ("品客", "酸乳酪洋葱味"),
    "SKU_017": ("奥利奥", "香甜不腻"),
    "SKU_018": ("奥利奥", "冰淇淋抹茶味"),
    "SKU_019": ("奥利奥", "浓醇巧克力味"),
    "SKU_020": ("奥利奥", "白桃乌龙味"),
    "SKU_021": ("Lay's乐事薯片", "墨西哥鸡汁番茄味"),
    "SKU_022": ("Lay's乐事薯片", "经典原味"),
    "SKU_023": ("Lay's乐事薯片", "黄瓜味"),
    "SKU_024": ("Lay's乐事薯片", "意大利香浓红烩味"),
    "SKU_025": ("Lay's乐事薯片", "青柠味"),
    "SKU_026": ("呀！土豆", "番茄酱味"),
    "SKU_027": ("好友趣", "韩国泡菜味"),
    "SKU_029": ("百醇", "草莓香草味"),
    "SKU_030": ("百醇", "巧克力味"),
    "SKU_031": ("上好佳", "鲜虾条"),
    "SKU_035": ("好友趣", "蜂蜜黄油味"),
    "SKU_039": ("草原红太阳烧烤料", "原味"),
    "SKU_040": ("草原红太阳烧烤料", "香辣味"),
    "SKU_041": ("草原红太阳烧烤酱", "香辣"),
    "SKU_042": ("草原红太阳烧烤酱", "原味"),
    "SKU_046": ("合味道", "海鲜味"),
    "SKU_047": ("合味道", "五香牛肉味"),
    "SKU_051": ("康师傅", "香辣牛肉面"),
    "SKU_052": ("康师傅", "鲜虾鱼板面"),
    "SKU_053": ("康师傅", "老坛酸菜牛肉面"),
    "SKU_067": ("可口可乐", "罐装"),
    "SKU_068": ("雪碧", "罐装"),
    "SKU_069": ("芬达", "罐装"),
    "SKU_070": ("百事可乐", "瓶装"),
    "SKU_071": ("7喜", "瓶装"),
    "SKU_072": ("水溶C100", "瓶装"),
    "SKU_074": ("美年达", "瓶装"),
    "SKU_076": ("东方树叶", "茉莉花茶"),
    "SKU_078": ("外星人电解质水", "椰子口味"),
    "SKU_079": ("外星人电解质水", "青柠口味"),
    "SKU_080": ("外星人电解质水", "白桃口味0糖"),
    "SKU_081": ("外星人电解质水", "西柚口味"),
    "SKU_084": ("脉动", "观梅止渴饮"),
    "SKU_085": ("脉动", "芒果口味"),
    "SKU_086": ("脉动", "菠萝口味"),
    "SKU_087": ("脉动", "猫薄荷瓶"),
    "SKU_088": ("外星人电解质水", "青柠口味0糖"),
    "SKU_090": ("舒肤佳", "香皂纯白清香型"),
    "SKU_091": ("舒肤佳", "香皂芦荟呵护香型"),
    "SKU_092": ("舒肤佳", "香皂柠檬清新香型"),
    "SKU_095": ("舒克", "牙膏竹炭薄荷"),
    "SKU_096": ("舒克", "牙膏柠檬百香果"),
    "SKU_097": ("舒克", "牙膏海盐薄荷"),
    "SKU_098": ("冷酸灵", "云感觉牙刷"),
    "SKU_099": ("Dove", "沐浴泡泡樱花甜香"),
    "SKU_100": ("Dove", "沐浴泡泡白桃果香"),
    "SKU_101": ("Dove", "沐浴泡泡清甜奶香"),
    "SKU_103": ("Dove", "沐浴乳"),
    "SKU_105": ("Dove", "沐浴泡泡樱花甜香袋装"),
    "SKU_106": ("半亩花田", "洗发水"),
    "SKU_107": ("心相印", "厨房纸巾"),
    "SKU_045": ("心相印", "纸巾")
}


LIQUID_KEYWORDS = (
    "汁",
    "牛奶",
    "酸奶",
    "可乐",
    "雪碧",
    "芬达",
    "7喜",
    "美年达",
    "水",
    "茶",
    "冰沙",
    "尖叫",
    "脉动",
    "宝矿力",
    "醋",
    "豉油",
    "生抽",
    "沐浴",
    "洗发水",
    "威露士",
)
COUNTED_KEYWORDS = (
    "棉签",
    "牙线",
    "纸巾",
    "创口贴",
    "湿巾",
    "百洁布",
    "毛巾",
    "口罩",
    "香皂",
    "肥皂",
    "杯子",
    "牙刷",
    "拖鞋",
)
LIQUID_SIZES = ("250ml", "330毫升", "500ml", "750毫升", "1升")
COUNTED_SIZES = ("1个", "2包", "6片", "10枚", "12抽")
WEIGHT_SIZES = ("55g", "80克", "120g", "250克", "500g")


def load_products() -> list[dict[str, Any]]:
    with CATALOG_PATH.open("r", encoding="utf-8") as file:
        catalog = json.load(file)
    products = catalog.get("products")
    if not isinstance(products, list) or not products:
        raise ValueError(f"商品目录格式错误或为空：{CATALOG_PATH}")
    return products


def invented_size(product_name: str, seed: int) -> str:
    if any(keyword in product_name for keyword in LIQUID_KEYWORDS):
        choices = LIQUID_SIZES
    elif any(keyword in product_name for keyword in COUNTED_KEYWORDS):
        choices = COUNTED_SIZES
    else:
        choices = WEIGHT_SIZES
    return choices[seed % len(choices)]


def build_input_item(
    product: dict[str, Any], occurrence: int, seed: int
) -> tuple[dict[str, str], str]:
    sku_id = product["sku_id"]
    canonical_name = product["name"]
    split = SPLIT_VARIANTS.get(sku_id)
    if occurrence % 2 == 0 and split is not None:
        short_name, semantic_specification = split
        return (
            {"name": short_name, "specification": semantic_specification},
            "split_name_and_type",
        )
    return (
        {
            "name": canonical_name,
            "specification": invented_size(canonical_name, seed),
        },
        "full_name_with_invented_size",
    )


def build_cases(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(products) < 2:
        raise ValueError("至少需要两个不同的商品才能构建测试样例。")

    occurrences: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []
    product_count = len(products)
    for index in range(CASE_COUNT):
        first_index = index % product_count
        cycle = index // product_count
        second_index = (first_index + 53 + cycle * 17) % product_count
        if second_index == first_index:
            second_index = (second_index + 1) % product_count
        selected = (products[first_index], products[second_index])

        inputs: list[dict[str, str]] = []
        styles: list[str] = []
        for item_index, product in enumerate(selected):
            sku_id = product["sku_id"]
            input_item, style = build_input_item(
                product,
                occurrences[sku_id],
                seed=index * 2 + item_index,
            )
            occurrences[sku_id] += 1
            inputs.append(input_item)
            styles.append(style)

        cases.append(
            {
                "case_id": f"receipt_{index + 1:03d}",
                "input": inputs,
                "input_styles": styles,
                "expected": {
                    "sku_ids": [product["sku_id"] for product in selected],
                    "product_names": [product["name"] for product in selected],
                },
            }
        )
    return cases


def validate_cases(
    products: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> Counter[str]:
    product_by_id = {product["sku_id"]: product for product in products}
    if not 100 <= len(cases) <= 200:
        raise AssertionError("测试样例数量必须在 100 到 200 之间。")

    coverage: Counter[str] = Counter()
    seen_pairs: set[frozenset[str]] = set()
    for case in cases:
        inputs = case["input"]
        sku_ids = case["expected"]["sku_ids"]
        product_names = case["expected"]["product_names"]
        if len(inputs) != 2 or len(sku_ids) != 2 or len(product_names) != 2:
            raise AssertionError(f"{case['case_id']} 必须恰好包含两个商品。")
        if sku_ids[0] == sku_ids[1]:
            raise AssertionError(f"{case['case_id']} 的两个商品必须不同。")

        pair = frozenset(sku_ids)
        if pair in seen_pairs:
            raise AssertionError(f"{case['case_id']} 与已有样例的商品组合重复。")
        seen_pairs.add(pair)

        for input_item, sku_id, product_name in zip(
            inputs, sku_ids, product_names, strict=True
        ):
            product = product_by_id[sku_id]
            if product["name"] != product_name:
                raise AssertionError(f"{case['case_id']} 的期望商品信息不一致。")
            if set(input_item) != {"name", "specification"}:
                raise AssertionError(f"{case['case_id']} 输入字段不符合接口格式。")
            if input_item["name"] != product_name:
                if input_item["name"] + input_item["specification"] != product_name:
                    raise AssertionError(f"{case['case_id']} 的拆分名称无法还原商品名。")
            coverage[sku_id] += 1

    if set(coverage) != set(product_by_id):
        missing = sorted(set(product_by_id) - set(coverage))
        raise AssertionError(f"存在未覆盖 SKU：{missing}")
    return coverage


def main() -> None:
    products = load_products()
    catalog_by_id = {product["sku_id"]: product for product in products}
    for sku_id, (name, specification) in SPLIT_VARIANTS.items():
        if sku_id not in catalog_by_id:
            raise ValueError(f"拆分配置包含未知 SKU：{sku_id}")
        if name + specification != catalog_by_id[sku_id]["name"]:
            raise ValueError(f"{sku_id} 的拆分配置无法还原完整商品名。")

    cases = build_cases(products)
    coverage = validate_cases(products, cases)
    document = {
        "schema_version": "1.0",
        "description": (
            "每条样例包含两个不同商品；input 可直接传给收据 SKU 匹配逻辑，"
            "expected 用于校验标准 SKU。"
        ),
        "source_catalog": "sku/products.json",
        "case_count": len(cases),
        "coverage": {
            "catalog_sku_count": len(products),
            "covered_sku_count": len(coverage),
            "min_occurrences_per_sku": min(coverage.values()),
            "max_occurrences_per_sku": max(coverage.values()),
            "split_variant_sku_count": len(SPLIT_VARIANTS),
        },
        "test_cases": cases,
    }
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(document, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"已生成 {len(cases)} 条样例：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
