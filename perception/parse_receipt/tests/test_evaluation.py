import tempfile
import unittest
from pathlib import Path

from receipt_recognizer.evaluation import (
    evaluate_items,
    evaluate_items_file,
)


class EvaluationTests(unittest.TestCase):
    def test_evaluate_items_counts_inventory_name_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inventory = Path(tmp) / "inventory.csv"
            inventory.write_text(
                "sku_name\n"
                "Lay's乐事薯片青柠味\n"
                "Lay's乐事薯片黄瓜味\n",
                encoding="utf-8",
            )
            recognized = [
                {
                    "name": "Lay's乐事薯片青柠味",
                    "specification": "55g",
                },
                {
                    "name": "Lay's乐事薯片柠檬味",
                    "specification": "55g",
                },
            ]

            result = evaluate_items(
                "receipt12.jpg",
                recognized,
                inventory,
            )

        self.assertEqual(
            result["summary"]["name_inventory_exact_matches"],
            1,
        )
        self.assertEqual(
            result["summary"]["name_inventory_exact_rate"],
            0.5,
        )
        self.assertEqual(result["rows"][1]["matched_sku_name"], None)

    def test_items_file_uses_stem_as_image_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inventory = root / "inventory.csv"
            inventory.write_text(
                "sku_name\nLay's乐事薯片青柠味\n",
                encoding="utf-8",
            )
            items_path = root / "receipt12.items.json"
            items_path.write_text(
                """[{"name":"Lay's乐事薯片青柠味","specification":"55g"}]""",
                encoding="utf-8",
            )

            result = evaluate_items_file(items_path, inventory)

        self.assertEqual(result["image"], "receipt12")
        self.assertEqual(result["summary"]["name_inventory_exact_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
