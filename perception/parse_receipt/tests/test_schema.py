import json
import unittest

from receipt_recognizer.errors import SchemaValidationError
from receipt_recognizer.schema import parse_receipt_result


def valid_payload() -> dict:
    return {
        "receipt_status": "ok",
        "line_items": [
            {
                "name": "Lay's乐事薯片墨西哥鸡汁番茄味",
                "specification": "55g",
                "needs_review": False,
                "reason": None,
            },
            {
                "name": "呀！土豆番茄酱味",
                "specification": "70g",
                "needs_review": False,
                "reason": None,
            },
        ],
        "review_items": [],
    }


class ReceiptSchemaTests(unittest.TestCase):
    def test_business_output_only_keeps_name_and_specification(self) -> None:
        parsed = parse_receipt_result(
            json.dumps(valid_payload(), ensure_ascii=False)
        )
        self.assertEqual(
            parsed.business_items(),
            [
                {
                    "name": "Lay's乐事薯片墨西哥鸡汁番茄味",
                    "specification": "55g",
                },
                {
                    "name": "呀！土豆番茄酱味",
                    "specification": "70g",
                },
            ],
        )

    def test_preserves_each_recognized_line_item(self) -> None:
        payload = valid_payload()
        payload["line_items"].append(
            {
                "name": "呀！土豆番茄酱味",
                "specification": "70g",
                "needs_review": False,
                "reason": None,
            }
        )
        parsed = parse_receipt_result(
            json.dumps(payload, ensure_ascii=False)
        )
        self.assertEqual(len(parsed.business_items()), 3)

    def test_review_items_are_excluded_from_business_output(self) -> None:
        payload = valid_payload()
        payload["receipt_status"] = "needs_review"
        payload["review_items"] = [
            {
                "name": None,
                "specification": None,
                "needs_review": True,
                "reason": "name_unclear",
            }
        ]
        parsed = parse_receipt_result(json.dumps(payload))
        self.assertEqual(len(parsed.review_items), 1)
        self.assertEqual(len(parsed.business_items()), 2)

    def test_rejects_markdown_wrapped_json(self) -> None:
        raw = "```json\n" + json.dumps(valid_payload()) + "\n```"
        with self.assertRaises(SchemaValidationError):
            parse_receipt_result(raw)

    def test_normalizes_status_from_review_items(self) -> None:
        payload = valid_payload()
        payload["review_items"] = [
            {
                "name": None,
                "specification": None,
                "needs_review": True,
                "reason": "name_unclear",
            }
        ]
        parsed = parse_receipt_result(json.dumps(payload))
        self.assertEqual(parsed.receipt_status, "needs_review")
        self.assertEqual(parsed.reported_receipt_status, "ok")
        self.assertTrue(parsed.diagnostics_dict()["status_normalized"])

    def test_normalizes_needs_review_without_review_items(self) -> None:
        payload = valid_payload()
        payload["receipt_status"] = "needs_review"
        parsed = parse_receipt_result(json.dumps(payload))
        self.assertEqual(parsed.receipt_status, "ok")
        self.assertEqual(
            parsed.reported_receipt_status,
            "needs_review",
        )

    def test_rejects_missing_specification(self) -> None:
        payload = valid_payload()
        del payload["line_items"][0]["specification"]
        with self.assertRaises(SchemaValidationError):
            parse_receipt_result(json.dumps(payload))

    def test_rejects_missing_needs_review(self) -> None:
        payload = valid_payload()
        del payload["line_items"][0]["needs_review"]
        with self.assertRaises(SchemaValidationError):
            parse_receipt_result(json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
