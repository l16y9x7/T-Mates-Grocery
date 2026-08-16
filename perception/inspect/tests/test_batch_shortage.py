from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


BATCH_PATH = Path(__file__).resolve().parents[1] / "batch_shortage.py"
SPEC = importlib.util.spec_from_file_location("shortage_batch_test_api", BATCH_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {BATCH_PATH}")
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class ShortageBatchTest(unittest.TestCase):
    def test_discovers_grouped_records_and_resolves_sku_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = (
                root
                / "H1_B_R_INSPECT_UPPER"
                / "record_20260816_010203_123456"
            )
            record.mkdir(parents=True)

            records = batch.discover_records(root)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["inspection_target_id"], "H1_B_R_INSPECT")
        self.assertEqual(records[0]["pose_type"], "SHELF_VIEW_UPPER")
        self.assertEqual(records[0]["location_id"], "H1_B_L1_C04")

    def test_clipped_region_mask_keeps_only_bbox_pixels(self) -> None:
        mask = np.full((20, 30), 255, dtype=np.uint8)

        clipped = batch.clipped_region_mask(mask, [5, 6, 7, 8])

        self.assertEqual(int(np.count_nonzero(clipped)), 56)
        self.assertTrue(np.all(clipped[6:14, 5:12] == 255))

    def test_detection_only_record_saves_bbox_and_region_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_directory = (
                root
                / "H1_B_L_INSPECT_UPPER"
                / "record_20260816_010203_123456"
            )
            record_directory.mkdir(parents=True)
            image = np.full((72, 128, 3), 80, dtype=np.uint8)
            batch.write_image(record_directory / "rgb.jpg", image)
            np.save(
                record_directory / "depth_mm.npy",
                np.full((72, 128), 900, dtype=np.uint16),
            )
            review_mask = np.zeros((72, 128), dtype=np.uint8)
            review_mask[20:40, 30:60] = 255
            finding = SimpleNamespace(
                bbox=[30, 20, 30, 20],
                center=[45, 30],
                sources=["comparison_based"],
                votes=1,
            )
            response = SimpleNamespace(
                findings=[finding],
                image_size=[128, 72],
                bbox_format=["x", "y", "width", "height"],
                algorithms=[
                    SimpleNamespace(
                        name="comparison_based",
                        alignment_success=True,
                    )
                ],
            )
            execution = SimpleNamespace(
                response=response,
                review_image=image,
                review_mask=review_mask,
            )
            initial_scan = SimpleNamespace(
                rgb=image,
                rgb_path=Path("task0/H1_B_L_INSPECT_UPPER/rgb.jpg"),
            )
            entry = {
                "group": "H1_B_L_INSPECT_UPPER",
                "record": record_directory.name,
                "record_directory": record_directory,
                "inspection_target_id": "H1_B_L_INSPECT",
                "location_id": "H1_B_L1_C01",
                "pose_type": "SHELF_VIEW_UPPER",
            }

            with patch.object(
                batch.INSPECT_API,
                "inspect_images_with_artifacts",
                return_value=execution,
            ):
                result = batch.run_record(
                    entry,
                    data_root=root,
                    initial_scan=initial_scan,
                    reviewer=None,
                    detection_only=True,
                    overwrite=True,
                )

            mask_path = root / result["findings"][0]["mask"]
            saved_mask = batch.read_image(mask_path)

        self.assertEqual(result["status"], "detection_only")
        self.assertEqual(result["findings"][0]["bbox"], [30, 20, 30, 20])
        self.assertEqual(result["findings"][0]["mask_pixels"], 600)
        self.assertTrue(mask_path.name.endswith("_mask.png"))
        self.assertEqual(saved_mask.shape[:2], (72, 128))


if __name__ == "__main__":
    unittest.main()
