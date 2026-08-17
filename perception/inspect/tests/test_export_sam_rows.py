from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "export_sam_rows.py"
SPEC = importlib.util.spec_from_file_location("export_sam_rows_test_api", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT_PATH}")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportSamRowsTest(unittest.TestCase):
    def test_export_record_keeps_rgb_depth_alignment_and_original_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / "record"
            output = root / "output"
            record.mkdir()
            image = np.full((60, 100, 3), 90, dtype=np.uint8)
            depth = np.arange(6000, dtype=np.uint16).reshape(60, 100) + 1
            exporter.write_image(record / "rgb.jpg", image)
            np.save(record / "depth_mm.npy", depth, allow_pickle=False)
            rows = [
                SimpleNamespace(index=1, bbox=(0, 0, 100, 20)),
                SimpleNamespace(index=2, bbox=(0, 20, 100, 40)),
            ]
            detection = SimpleNamespace(
                rows=rows,
                draw=lambda: image,
                as_dict=lambda: {"rows": []},
            )

            with patch.object(
                exporter,
                "load_row_detection",
                return_value=(lambda **_: object(), lambda *_: detection),
            ):
                metadata = exporter.export_record(
                    record,
                    output,
                    pose_type="SHELF_VIEW_UPPER",
                    overwrite=False,
                )

            first_depth = np.load(
                output / "row_01_L1" / "depth_mm.npy",
                allow_pickle=False,
            )
            second_depth = np.load(
                output / "row_02_L2" / "depth_mm.npy",
                allow_pickle=False,
            )
            saved = json.loads((output / "rows.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["detected_row_count"], 2)
        self.assertEqual(metadata["rows"][0]["level"], "L1")
        self.assertEqual(metadata["rows"][1]["level"], "L2")
        self.assertEqual(metadata["rows"][1]["crop_origin_xy"], [0, 20])
        self.assertTrue(np.array_equal(first_depth, depth[:20]))
        self.assertTrue(np.array_equal(second_depth, depth[20:]))
        self.assertEqual(saved["rows"][1]["crop_bbox_xywh"], [0, 20, 100, 40])

    def test_dataset_manifest_keeps_processing_after_missing_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output_root = root / "output"
            valid = (
                data_root
                / "H1_F_L_INSPECT_UPPER"
                / "20260816T205750_798810Z_H1_F_L_INSPECT_SHORTAGE_1aa7e2d9"
            )
            invalid = (
                data_root
                / "H1_F_L_INSPECT_UPPER"
                / "20260816T183750_435324Z_H1_F_L_INSPECT_SHORTAGE_69038c8a"
            )
            valid.mkdir(parents=True)
            invalid.mkdir(parents=True)
            image = np.full((30, 40, 3), 90, dtype=np.uint8)
            exporter.write_image(valid / "rgb.jpg", image)
            np.save(valid / "depth_mm.npy", np.full((30, 40), 900, np.uint16))
            np.save(invalid / "depth_mm.npy", np.full((30, 40), 900, np.uint16))
            detection = SimpleNamespace(
                rows=[SimpleNamespace(index=1, bbox=(0, 0, 40, 30))],
                draw=lambda: image,
                as_dict=lambda: {"rows": []},
            )

            with patch.object(
                exporter,
                "load_row_detection",
                return_value=(lambda **_: object(), lambda *_: detection),
            ):
                manifest = exporter.export_dataset(
                    data_root,
                    output_root,
                    overwrite=False,
                )

        self.assertEqual(manifest["completed_records"], 1)
        self.assertEqual(manifest["failed_records"], 1)
        self.assertEqual(manifest["exported_rows"], 1)
        self.assertIn("rgb.jpg", manifest["errors"][0]["error"])


if __name__ == "__main__":
    unittest.main()
