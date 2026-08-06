import unittest
from unittest.mock import patch

from receipt_recognizer.ocr import OCRResult, build_parser, extract_ocr_lines


class OCRParsingTests(unittest.TestCase):
    def test_cli_accepts_explicit_gpu_device(self) -> None:
        args = build_parser().parse_args(["receipt.jpg", "--device", "gpu:0"])
        self.assertEqual(args.device, "gpu:0")

    def test_cli_reads_default_device_from_environment(self) -> None:
        with patch.dict("os.environ", {"RECEIPT_OCR_DEVICE": "gpu:0"}):
            args = build_parser().parse_args(["receipt.jpg"])
        self.assertEqual(args.device, "gpu:0")

    def test_cli_rejects_invalid_device(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["receipt.jpg", "--device", "cuda:0"])

    def test_extracts_lines_from_page_wrapped_paddleocr_result(self) -> None:
        raw = [
            [
                [
                    [[0, 0], [10, 0], [10, 10], [0, 10]],
                    ("Lay's乐事薯片墨", 0.98),
                ],
                [
                    [[0, 20], [10, 20], [10, 30], [0, 30]],
                    ("西哥鸡汁番茄味", 0.97),
                ],
            ]
        ]

        lines = extract_ocr_lines(raw)

        self.assertEqual(
            [line.text for line in lines],
            ["Lay's乐事薯片墨", "西哥鸡汁番茄味"],
        )
        self.assertEqual(lines[0].score, 0.98)

    def test_extracts_lines_from_flat_paddleocr_result(self) -> None:
        raw = [
            [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                ("康师傅香辣牛肉", 0.91),
            ],
            [
                [[0, 20], [10, 20], [10, 30], [0, 30]],
                ("面", 0.93),
            ],
        ]

        lines = extract_ocr_lines(raw)

        self.assertEqual(
            [line.text for line in lines],
            ["康师傅香辣牛肉", "面"],
        )

    def test_result_full_text_joins_lines_in_reading_order(self) -> None:
        lines = extract_ocr_lines(
            [[[[0, 0]], ("呀！土豆番茄酱", 0.95)], [[[0, 1]], ("味", 0.96)]]
        )

        result = OCRResult(image="receipt.jpg", ocr_lines=lines)

        self.assertEqual(result.full_text, "呀！土豆番茄酱 味")
        self.assertEqual(result.to_dict()["image"], "receipt.jpg")


if __name__ == "__main__":
    unittest.main()
