"""Tests for the place check API."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import server


class PlaceCheckTest(unittest.TestCase):
    def request(self, task_type: str) -> server.PlaceCheckRequest:
        return server.PlaceCheckRequest(
            task_type=task_type,
            product_name="可口可乐罐装",
            hand="left",
        )

    def test_sorting_prompt_checks_delivery_table(self) -> None:
        prompt = server.load_prompt(self.request("SORTING"))
        self.assertIn("交付台", prompt)
        self.assertNotIn("摆列整齐", prompt)

    def test_shortage_and_misplaced_check_shelf_arrangement(self) -> None:
        for task_type in ("SHORTAGE", "MISPLACED"):
            with self.subTest(task_type=task_type):
                prompt = server.load_prompt(self.request(task_type))
                self.assertIn("货架", prompt)
                self.assertIn("摆列整齐", prompt)

    def test_endpoint_uses_hand_and_reference_images(self) -> None:
        expected = server.PlaceCheckResponse(place_status="Success")
        with (
            patch.object(
                server,
                "fetch_hand_image",
                return_value=(b"hand", "image/jpeg"),
            ) as hand,
            patch.object(
                server,
                "fetch_reference_image",
                return_value=(b"reference", "image/png"),
            ) as reference,
            patch.object(server, "call_qwen", return_value=expected) as qwen,
        ):
            result = server.check_product_placement(self.request("SORTING"))

        self.assertEqual(result, expected)
        hand.assert_called_once_with("left")
        reference.assert_called_once_with("可口可乐罐装")
        self.assertEqual(qwen.call_args.args[1:], (b"reference", "image/png", b"hand", "image/jpeg"))


if __name__ == "__main__":
    unittest.main()
