"""Tests for fixed task0 initial RGB-D scan resolution."""

from __future__ import annotations

import unittest

from initial_scan import (
    InitialScanError,
    load_initial_scan,
    normalize_scan_pose,
    resolve_initial_scan_directory,
    resolve_inspection_target_id,
)
from row_detection import RowDetectionConfig, detect_rows


class InitialScanTest(unittest.TestCase):
    def test_slot_uses_agent_configured_left_right_inspection_target(self) -> None:
        self.assertEqual(
            resolve_inspection_target_id("H1_F_L2_C11"),
            "H1_F_R_INSPECT",
        )
        self.assertEqual(
            resolve_inspection_target_id("H1_B_L2_C01"),
            "H1_B_L_INSPECT",
        )

    def test_pose_can_be_derived_from_product_level(self) -> None:
        self.assertEqual(normalize_scan_pose("", "H1_F_L2_C01"), "SHELF_VIEW_UPPER")
        self.assertEqual(normalize_scan_pose("", "H1_F_L3_C01"), "SHELF_VIEW_LOWER")

    def test_inspection_target_requires_explicit_upper_or_lower(self) -> None:
        with self.assertRaisesRegex(InitialScanError, "pose_type is required"):
            normalize_scan_pose("", "H1_F_L_INSPECT")

    def test_real_task0_rgbd_record_is_valid(self) -> None:
        scan = load_initial_scan("H1_B_L2_C01", "SHELF_VIEW_LOWER")

        self.assertEqual(scan.directory.name, "H1_B_L_INSPECT_LOWER")
        self.assertEqual(scan.rgb.shape, (720, 1280, 3))
        self.assertEqual(scan.depth_mm.shape, (720, 1280))
        self.assertEqual(scan.metadata["depth"]["unit"], "millimeter")

    def test_explicit_target_builds_expected_upper_directory(self) -> None:
        directory, target, pose = resolve_initial_scan_directory(
            "H2_F_R_INSPECT",
            "UPPER",
        )

        self.assertEqual(directory.name, "H2_F_R_INSPECT_UPPER")
        self.assertEqual(target, "H2_F_R_INSPECT")
        self.assertEqual(pose, "SHELF_VIEW_UPPER")

    def test_close_initial_scan_rails_are_consolidated(self) -> None:
        cases = (
            ("H1_F_L_INSPECT", "SHELF_VIEW_UPPER", 2, 2),
            ("H2_B_L_INSPECT", "SHELF_VIEW_UPPER", 2, 2),
            ("H2_F_R_INSPECT", "SHELF_VIEW_LOWER", 3, 3),
        )
        for location_id, pose_type, rail_count, row_count in cases:
            with self.subTest(location_id=location_id, pose_type=pose_type):
                scan = load_initial_scan(location_id, pose_type)
                detection = detect_rows(
                    scan.rgb,
                    RowDetectionConfig(target_size=None, pose_type=pose_type),
                )
                self.assertEqual(len(detection.rails), rail_count)
                self.assertEqual(len(detection.rows), row_count)


if __name__ == "__main__":
    unittest.main()
