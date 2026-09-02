from __future__ import annotations

import unittest
from unittest.mock import patch

from pick.locate.main import (
    hard_case_group_for_product,
    load_hard_case_view_layout,
    load_prompt_pair,
    validate_slot_hard_case_context,
)


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

    def test_connector_right_alienergy_keeps_hard_case(self) -> None:
        hard_case = hard_case_group_for_product(
            "外星人电解质水椰子口味",
            "SORTING",
            "L3",
            "right",
            slot_id="H2_L03_C01",
            target_id="H12_INSPECT",
        )

        self.assertIsNotNone(hard_case)
        assert hard_case is not None
        self.assertEqual(hard_case[0], "alien_energy")

    def test_scope_file_can_disable_connector_hard_case(self) -> None:
        with patch("pick.locate.main.load_hard_case_scope", return_value=set()):
            self.assertIsNone(
                hard_case_group_for_product(
                    "外星人电解质水椰子口味",
                    "SORTING",
                    "L3",
                    "left",
                    slot_id="H2_L03_C01",
                    target_id="H2_INSPECT",
                )
            )
            self.assertIsNone(
                validate_slot_hard_case_context(
                    {
                        "name": "外星人电解质水椰子口味",
                        "locations": ["H2_L03_C01"],
                    },
                    "SORTING",
                    "L3",
                    "left",
                    "H2_L03_C01",
                    "H2_INSPECT",
                )
            )

    def test_all_configured_slot_routes_are_hard_cases(self) -> None:
        slot_products = {
            "H2_L03_C01": ("外星人电解质水椰子口味", "L3"),
            "H2_L03_C02": ("外星人电解质水青柠口味", "L3"),
            "H2_L03_C03": ("外星人电解质水白桃口味0糖", "L3"),
            "H2_L04_C01": ("脉动观梅止渴饮", "L4"),
            "H2_L04_C02": ("脉动芒果口味", "L4"),
            "H2_L04_C03": ("脉动菠萝口味", "L4"),
            "H2_L05_C01": ("脉动观梅止渴饮", "L5"),
            "H2_L05_C02": ("脉动芒果口味", "L5"),
        }
        slot_groups, views = load_hard_case_view_layout()

        self.assertEqual(set(slot_groups), set(slot_products))
        for slot_id, (product_name, level) in slot_products.items():
            for target_id, hand in (
                ("H2_INSPECT", "left"),
                ("H12_INSPECT", "right"),
            ):
                with self.subTest(slot_id=slot_id, target_id=target_id, hand=hand):
                    group_id = slot_groups[slot_id]
                    self.assertIn(
                        slot_id,
                        views[(target_id, hand, level, group_id)].visible_slot_order,
                    )
                    context = validate_slot_hard_case_context(
                        {"name": product_name, "locations": [slot_id]},
                        "SORTING",
                        level,
                        hand,
                        slot_id,
                        target_id,
                    )
                    self.assertIsNotNone(context)
                    assert context is not None
                    self.assertEqual(context[0], group_id)
                    hard_case = hard_case_group_for_product(
                        product_name,
                        "SORTING",
                        level,
                        hand,
                        slot_id=slot_id,
                        target_id=target_id,
                    )
                    self.assertIsNotNone(hard_case)
                    assert hard_case is not None
                    self.assertEqual(hard_case[0], group_id)

    def test_catnip_maiydong_is_not_hard_case(self) -> None:
        with patch(
            "pick.locate.main.load_hard_case_view_layout",
            side_effect=AssertionError("catnip must not load connector hard-case views"),
        ):
            for target_id, hand in (
                ("H2_INSPECT", "right"),
                ("H23_INSPECT", "left"),
            ):
                with self.subTest(target_id=target_id, hand=hand):
                    self.assertIsNone(
                        hard_case_group_for_product(
                            "脉动猫薄荷瓶",
                            "SORTING",
                            "L4",
                            hand,
                            slot_id="H2_L04_C04",
                            target_id=target_id,
                        )
                    )
                    self.assertIsNone(
                        validate_slot_hard_case_context(
                            {
                                "name": "脉动猫薄荷瓶",
                                "locations": ["H2_L04_C04"],
                            },
                            "SORTING",
                            "L4",
                            hand,
                            "H2_L04_C04",
                            target_id,
                        )
                    )
    def test_shuke_passion_fruit_keeps_legacy_hard_case_with_slot_context(self) -> None:
        hard_case = hard_case_group_for_product(
            "舒克牙膏柠檬百香果",
            "SORTING",
            "L2",
            "left",
            slot_id="H1_L02_C02",
            target_id="H1_INSPECT",
        )

        self.assertIsNotNone(hard_case)
        assert hard_case is not None
        self.assertEqual(hard_case[0], "shuke_toothpaste")

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
