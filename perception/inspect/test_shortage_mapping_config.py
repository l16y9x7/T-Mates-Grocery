from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

try:
    from .sam_shortage_pipeline import load_mapping_config
except ImportError:
    from sam_shortage_pipeline import load_mapping_config


REPO_ROOT = Path(__file__).resolve().parents[2]
NON_ORDERABLE_SLOTS = {"H2_L04_C04": "脉动猫薄荷瓶"}


class ShortageMappingConfigTest(unittest.TestCase):
    def test_slots_names_and_grasp_targets_are_consistent_with_catalog(self) -> None:
        mapping = load_mapping_config()
        catalog_payload = json.loads(
            (REPO_ROOT / "perception" / "sku" / "products.json").read_text(
                encoding="utf-8"
            )
        )
        catalog = {
            slot_id: product["name"]
            for product in catalog_payload["products"]
            for slot_id in product["locations"]
        }
        options_payload = yaml.safe_load(
            (
                REPO_ROOT
                / "agent"
                / "config"
                / "product-hand-options.yaml"
            ).read_text(encoding="utf-8")
        )
        options = options_payload["product_hand_options"]

        self.assertEqual(
            list(mapping),
            [
                "H1_INSPECT",
                "H12_INSPECT",
                "H2_INSPECT",
                "H23_INSPECT",
                "H3_INSPECT",
            ],
        )
        mapped_non_orderable_slots: dict[str, str] = {}
        for target_id, levels in mapping.items():
            mapped_pairs = {
                (slot_id, name)
                for groups in levels.values()
                for group in groups
                for slot_id, name in zip(
                    group["slot_ids"],
                    group["slot_product_names"],
                    strict=True,
                )
            }
            expected_slots = {
                slot_id
                for slot_id, option in options.items()
                if any(
                    grasp["target_id"] == target_id
                    for grasp in (
                        option["grasp_options"]
                        if isinstance(option["grasp_options"], list)
                        else [option["grasp_options"]]
                    )
                )
            }
            physical_only_pairs = {
                (slot, name)
                for slot, name in mapped_pairs
                if NON_ORDERABLE_SLOTS.get(slot) == name
            }
            mapped_non_orderable_slots.update(physical_only_pairs)
            orderable_pairs = mapped_pairs - physical_only_pairs
            self.assertLessEqual(
                {slot for slot, _ in orderable_pairs}, expected_slots
            )
            mapped_orderable_slots = {slot for slot, _ in orderable_pairs}
            self.assertEqual(
                {slot: name for slot, name in orderable_pairs},
                {slot: catalog[slot] for slot in mapped_orderable_slots},
            )
        self.assertEqual(mapped_non_orderable_slots, NON_ORDERABLE_SLOTS)


if __name__ == "__main__":
    unittest.main()
