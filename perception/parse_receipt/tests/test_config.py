import os
import unittest
from unittest.mock import patch

from receipt_recognizer.config import Settings
from receipt_recognizer.errors import ConfigurationError


class SettingsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()

        self.assertEqual(
            settings.base_url,
            "http://127.0.0.1:8102/v1",
        )
        self.assertEqual(settings.model, "Qwen3-VL-4B-Instruct")
        self.assertIsNone(settings.api_key)
        self.assertEqual(settings.timeout_seconds, 60.0)
        self.assertEqual(settings.sku_base_url, "http://127.0.0.1:25540")
        self.assertEqual(settings.sku_timeout_seconds, 3.0)
        self.assertEqual(settings.sku_edit_distance_max, 3)
        self.assertEqual(settings.sku_fuzzy_limit, 2)

    def test_empty_api_key_is_none(self) -> None:
        with patch.dict(
            os.environ,
            {"QWEN_API_KEY": "   "},
            clear=True,
        ):
            self.assertIsNone(Settings.from_env().api_key)

    def test_invalid_timeout_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"QWEN_TIMEOUT_SECONDS": "soon"},
            clear=True,
        ):
            with self.assertRaises(ConfigurationError):
                Settings.from_env()

    def test_invalid_sku_timeout_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"SKU_TIMEOUT_SECONDS": "later"},
            clear=True,
        ):
            with self.assertRaises(ConfigurationError):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
