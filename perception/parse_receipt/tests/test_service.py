import json
import unittest

from receipt_recognizer.api import ChatResponse
from receipt_recognizer.config import Settings
from receipt_recognizer.errors import ModelOutputError
from receipt_recognizer.service import ReceiptRecognizer


VALID_RESULT = {
    "receipt_status": "ok",
    "line_items": [
        {
            "name": "Lay's乐事薯片墨西哥鸡汁番茄味",
            "specification": "55g",
            "needs_review": False,
            "reason": None,
        }
    ],
    "review_items": [],
}


class FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls = []

    def create_chat_completion(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        content = self.contents.pop(0)
        return ChatResponse(
            content=content,
            finish_reason="stop",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            },
            raw={},
        )


class ReceiptRecognizerTests(unittest.TestCase):
    def test_valid_output_is_not_retried(self) -> None:
        fake = FakeClient([json.dumps(VALID_RESULT, ensure_ascii=False)])
        service = ReceiptRecognizer(Settings(), client=fake)

        result = service.recognize_data_urls(
            ["data:image/jpeg;base64,AA=="]
        )

        self.assertEqual(
            result.business_items,
            [
                {
                    "name": "Lay's乐事薯片墨西哥鸡汁番茄味",
                    "specification": "55g",
                }
            ],
        )
        self.assertFalse(result.corrected_once)
        self.assertEqual(len(fake.calls), 1)

    def test_temperature_is_forwarded_to_model_request(self) -> None:
        fake = FakeClient([json.dumps(VALID_RESULT, ensure_ascii=False)])
        service = ReceiptRecognizer(Settings(), client=fake)

        result = service.recognize_data_urls(
            ["data:image/jpeg;base64,AA=="],
            temperature=0.5,
        )

        self.assertEqual(fake.calls[0][1]["temperature"], 0.5)
        self.assertEqual(result.diagnostics["temperature"], 0.5)

    def test_invalid_json_is_corrected_once(self) -> None:
        fake = FakeClient(
            [
                "```json\n{}\n```",
                json.dumps(VALID_RESULT, ensure_ascii=False),
            ]
        )
        service = ReceiptRecognizer(Settings(), client=fake)

        result = service.recognize_data_urls(
            ["data:image/jpeg;base64,AA=="]
        )

        self.assertTrue(result.corrected_once)
        self.assertEqual(len(fake.calls), 2)

    def test_validation_error_is_included_in_correction_prompt(self) -> None:
        missing_specification = json.loads(
            json.dumps(VALID_RESULT, ensure_ascii=False)
        )
        del missing_specification["line_items"][0]["specification"]
        fake = FakeClient(
            [
                json.dumps(missing_specification, ensure_ascii=False),
                json.dumps(VALID_RESULT, ensure_ascii=False),
            ]
        )
        service = ReceiptRecognizer(Settings(), client=fake)

        result = service.recognize_data_urls(
            ["data:image/jpeg;base64,AA=="]
        )

        correction_messages = fake.calls[1][0]
        correction_text = correction_messages[1]["content"]
        self.assertTrue(result.corrected_once)
        self.assertIn("specification 是必填字段", correction_text)

    def test_name_and_spec_are_kept_after_correction(self) -> None:
        incomplete = json.loads(json.dumps(VALID_RESULT, ensure_ascii=False))
        del incomplete["line_items"][0]["needs_review"]
        corrected = json.loads(json.dumps(VALID_RESULT, ensure_ascii=False))
        fake = FakeClient(
            [
                json.dumps(incomplete, ensure_ascii=False),
                json.dumps(corrected, ensure_ascii=False),
            ]
        )
        service = ReceiptRecognizer(Settings(), client=fake)

        result = service.recognize_data_urls(
            ["data:image/jpeg;base64,AA=="]
        )

        self.assertTrue(result.corrected_once)
        self.assertEqual(
            result.business_items[0],
            {
                "name": "Lay's乐事薯片墨西哥鸡汁番茄味",
                "specification": "55g",
            },
        )

    def test_second_invalid_output_fails(self) -> None:
        fake = FakeClient(["not-json", "still-not-json"])
        service = ReceiptRecognizer(Settings(), client=fake)

        with self.assertRaises(ModelOutputError):
            service.recognize_data_urls(
                ["data:image/jpeg;base64,AA=="]
            )
        self.assertEqual(len(fake.calls), 2)


if __name__ == "__main__":
    unittest.main()
