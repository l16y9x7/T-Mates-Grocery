from __future__ import annotations

import argparse
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from pick.locate.test import batch_record_inference as batch


class BatchRecordInferenceTest(unittest.TestCase):
    def test_default_workers_is_four(self) -> None:
        with (
            patch.object(batch, "DEFAULT_WORKERS", 4),
            patch("sys.argv", ["batch_record_inference.py"]),
        ):
            args = batch.parse_args()

        self.assertEqual(args.workers, 4)

    def test_main_runs_four_jobs_concurrently_and_sorts_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping_path = root / "sorting_pick_locate_batch.json"
            mapping_path.write_text("{}", encoding="utf-8")
            record_directory = root / "record_test"
            record_directory.mkdir()
            rgb_path = record_directory / "rgb.jpg"
            depth_path = record_directory / "depth_mm.npy"
            rgb_path.write_bytes(b"rgb")
            depth_path.write_bytes(b"depth")
            product_names = ["product_4", "product_3", "product_2", "product_1"]
            entries = [
                {
                    "record": "record_test",
                    "record_directory": record_directory,
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "hand": "left",
                    "level": "L1",
                    "product_names": product_names,
                }
            ]
            catalog = {
                name: {"sku_id": f"SKU_{index}"}
                for index, name in enumerate(product_names)
            }
            args = argparse.Namespace(
                mapping=mapping_path,
                api_url="http://127.0.0.1:8083",
                timeout=10.0,
                workers=4,
                retries=0,
                max_consecutive_system_errors=3,
                overwrite=True,
                record=None,
                product_name=None,
                dry_run=False,
            )
            barrier = threading.Barrier(4)
            active = 0
            peak_active = 0
            active_lock = threading.Lock()

            def fake_job(**kwargs):
                nonlocal active, peak_active
                with active_lock:
                    active += 1
                    peak_active = max(peak_active, active)
                barrier.wait(timeout=2)
                with active_lock:
                    active -= 1
                return {
                    "record": kwargs["entry"]["record"],
                    "product_name": kwargs["product_name"],
                    "skipped": False,
                    "success": True,
                    "summary": {
                        "record": kwargs["entry"]["record"],
                        "product_name": kwargs["product_name"],
                        "status": "success",
                    },
                    "is_system_error": False,
                    "combined_error": "",
                    "elapsed_seconds": 0.01,
                }

            with (
                patch.object(batch, "parse_args", return_value=args),
                patch.object(
                    batch,
                    "load_and_validate_mapping",
                    return_value=(root, entries, catalog),
                ),
                patch.object(batch, "run_batch_job", side_effect=fake_job),
            ):
                batch.main()

            summary = json.loads(
                (root / "sorting_pick_locate_batch_results.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(peak_active, 4)
        self.assertEqual(
            [item["product_name"] for item in summary["results"]],
            product_names,
        )
        self.assertEqual(summary["completed"], 4)
        self.assertEqual(summary["successes"], 4)


if __name__ == "__main__":
    unittest.main()
