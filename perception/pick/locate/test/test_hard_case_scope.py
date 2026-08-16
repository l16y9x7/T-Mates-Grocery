from __future__ import annotations

import unittest

from pick.locate.main import hard_case_group_for_product, load_prompt_pair


class HardCaseScopeTest(unittest.TestCase):
    def test_grapefruit_alienergy_l3_left_uses_specific_prompt(self) -> None:
        self.assertIsNone(
            hard_case_group_for_product(
                "外星人电解质水西柚口味",
                "SORTING",
                "L3",
                "left",
            )
        )

    def test_other_l3_left_alienergy_products_keep_hard_case(self) -> None:
        hard_case = hard_case_group_for_product(
            "外星人电解质水椰子口味",
            "SORTING",
            "L3",
            "left",
        )

        self.assertIsNotNone(hard_case)
        self.assertEqual(hard_case[0], "alien_energy")

    def test_shuke_passion_fruit_l2_left_uses_toothpaste_group(self) -> None:
        hard_case = hard_case_group_for_product(
            "舒克牙膏柠檬百香果",
            "SORTING",
            "L2",
            "left",
        )

        self.assertIsNotNone(hard_case)
        assert hard_case is not None
        self.assertEqual(hard_case[0], "shuke_toothpaste")
        self.assertEqual(
            hard_case[1].members,
            (
                "舒克牙膏竹炭薄荷",
                "舒克牙膏柠檬百香果",
                "舒克牙膏海盐薄荷",
            ),
        )

    def test_shuke_passion_fruit_hard_case_uses_group_prompt(self) -> None:
        qwen_prompt, sam_prompt = load_prompt_pair(
            "舒克牙膏柠檬百香果",
            "SORTING",
            hard_case=True,
        )

        self.assertIn("所有舒克牙膏盒", qwen_prompt)
        self.assertIn("不要框右侧的牙刷", qwen_prompt)
        self.assertEqual(sam_prompt, "frontmost toothpaste box")


if __name__ == "__main__":
    unittest.main()
