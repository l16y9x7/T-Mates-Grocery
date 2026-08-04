import unittest

from receipt_recognizer.inventory import (
    InventorySku,
    item_text_candidates,
    match_inventory_item,
    normalize_product_text,
    source_text_candidates,
    validate_receipt_items,
)


class InventoryMatchingTests(unittest.TestCase):
    def test_normalize_removes_punctuation_and_width_noise(self) -> None:
        self.assertEqual(
            normalize_product_text(" Lay’s 乐事薯片-墨西哥鸡汁番茄味 "),
            "lays乐事薯片墨西哥鸡汁番茄味",
        )

    def test_candidates_use_name_only(self) -> None:
        item = {
            "name": "Lay's乐事薯片墨西哥鸡汁番茄味",
            "specification": "55g",
        }
        self.assertEqual(
            item_text_candidates(item),
            ["Lay's乐事薯片墨西哥鸡汁番茄味"],
        )

    def test_source_text_candidates_are_disabled(self) -> None:
        self.assertEqual(
            source_text_candidates("1. 康师傅香辣牛肉面 1 500g"),
            [],
        )

    def test_exact_match(self) -> None:
        item = {
            "name": "上好佳鲜虾条",
            "specification": "55g",
        }
        match = match_inventory_item(
            item,
            [InventorySku("上好佳鲜虾条", "上好佳鲜虾条")],
        )
        self.assertEqual(match.match_status, "matched")
        self.assertEqual(match.matched_sku_name, "上好佳鲜虾条")

    def test_no_source_text_fallback(self) -> None:
        item = {
            "name": "康师傅",
            "specification": "500g",
            "source_text": "1. 康师傅香辣牛肉面 1 500g",
        }
        match = match_inventory_item(
            item,
            [InventorySku("康师傅香辣牛肉面", "康师傅香辣牛肉面")],
        )
        self.assertEqual(match.match_status, "not_found")
        self.assertIsNone(match.matched_sku_name)

    def test_contains_is_suggestion_only(self) -> None:
        item = {
            "name": "舒肤佳香皂",
            "specification": "55g",
        }
        match = match_inventory_item(
            item,
            [
                InventorySku("舒肤佳香皂纯白清香型", "舒肤佳香皂纯白清香型"),
                InventorySku("舒肤佳香皂柠檬清新香型", "舒肤佳香皂柠檬清新香型"),
            ],
        )
        self.assertEqual(match.match_status, "not_found")
        self.assertEqual(len(match.suggested_sku_names), 2)

    def test_validate_unique_pair_summary(self) -> None:
        items = [
            {"name": "双汇王中王火腿肠", "specification": "55g"},
            {"name": "卫龙大面筋", "specification": "70g"},
        ]
        result = validate_receipt_items(
            items,
            [
                InventorySku("双汇王中王火腿肠", "双汇王中王火腿肠"),
                InventorySku("卫龙大面筋", "卫龙大面筋"),
            ],
        )
        self.assertEqual(result["summary"]["unique_sku_count"], 2)
        self.assertTrue(result["summary"]["is_unique_pair"])


if __name__ == "__main__":
    unittest.main()
