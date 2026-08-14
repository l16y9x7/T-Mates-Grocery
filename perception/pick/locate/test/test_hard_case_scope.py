from __future__ import annotations

import unittest

from pick.locate.main import hard_case_group_for_product


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


if __name__ == "__main__":
    unittest.main()
