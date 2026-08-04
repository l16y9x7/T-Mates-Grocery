"""Evaluate recognized receipt JSON against inventory names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .inventory import load_inventory_csv, match_inventory_item


def evaluate_items(
    image: str,
    recognized_items: list[dict[str, Any]],
    inventory_path: Path,
) -> dict[str, Any]:
    """Evaluate whether recognized names exactly match known SKU names.

    The current inventory CSV only contains ``sku_name``.  Therefore this
    module intentionally does not evaluate ``specification`` accuracy.
    """

    inventory = load_inventory_csv(inventory_path)
    rows: list[dict[str, Any]] = []
    matched = 0

    for index, recognized in enumerate(recognized_items, start=1):
        inventory_match = match_inventory_item(recognized, inventory)
        name_inventory_exact = inventory_match.match_status == "matched"
        if name_inventory_exact:
            matched += 1
        rows.append(
            {
                "row_index": index,
                "recognized": recognized,
                "name_inventory_exact": name_inventory_exact,
                "matched_sku_name": inventory_match.matched_sku_name,
                "suggested_sku_names": list(
                    inventory_match.suggested_sku_names
                ),
            }
        )

    total = len(recognized_items)
    return {
        "image": image,
        "summary": {
            "total_rows": total,
            "name_inventory_exact_matches": matched,
            "name_inventory_exact_rate": _rate(matched, total),
        },
        "rows": rows,
    }


def evaluate_items_file(
    items_path: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    image = _image_name_from_items_path(items_path)
    with items_path.open("r", encoding="utf-8") as handle:
        recognized_items = json.load(handle)
    if not isinstance(recognized_items, list):
        raise ValueError(f"{items_path} 顶层必须是数组。")
    return evaluate_items(image, recognized_items, inventory_path)


def summarize_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = sum(
        int(evaluation["summary"]["total_rows"])
        for evaluation in evaluations
    )
    matched = sum(
        int(evaluation["summary"]["name_inventory_exact_matches"])
        for evaluation in evaluations
    )
    return {
        "total_rows": total_rows,
        "name_inventory_exact_matches": matched,
        "name_inventory_exact_rate": _rate(matched, total_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统计小票识别 JSON 的商品名库存命中率。"
    )
    parser.add_argument(
        "items_dir",
        type=Path,
        help="包含 *.items.json 的目录",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/inventory.csv"),
        help="库存 CSV，默认 data/inventory.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选：保存评估 JSON 到指定文件",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    item_paths = sorted(args.items_dir.glob("*.items.json"))
    evaluations = [
        evaluate_items_file(path, args.inventory)
        for path in item_paths
    ]
    result = {
        "summary": summarize_evaluations(evaluations),
        "evaluations": evaluations,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _image_name_from_items_path(path: Path) -> str:
    name = path.name
    if name.endswith(".items.json"):
        return name[: -len(".items.json")]
    return path.stem


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


if __name__ == "__main__":
    raise SystemExit(main())
