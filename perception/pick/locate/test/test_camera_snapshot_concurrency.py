from __future__ import annotations

import asyncio
import io
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from pick.locate import main as locate_main


class _SnapshotResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def _png_bytes(color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), color).save(buffer, format="PNG")
    return buffer.getvalue()


class CameraSnapshotConcurrencyTest(unittest.TestCase):
    def _fetch_pair(
        self,
        cache_dir: Path,
        cameras: tuple[str, str],
        fake_get,
    ) -> tuple[list[Path | None], list[Path]]:
        original_write_bytes = Path.write_bytes
        write_barrier = threading.Barrier(2)
        observed_temporary_paths: list[Path] = []
        observed_lock = threading.Lock()

        def synchronized_write_bytes(path: Path, data: bytes) -> int:
            if path.parent == cache_dir and path.name.endswith(".tmp"):
                with observed_lock:
                    observed_temporary_paths.append(path)
                write_barrier.wait(timeout=2)
            return original_write_bytes(path, data)

        with (
            patch.object(locate_main, "CAMERA_SNAPSHOT_CACHE_DIR", cache_dir),
            patch.object(locate_main.requests, "get", side_effect=fake_get),
            patch.object(Path, "write_bytes", new=synchronized_write_bytes),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            futures = [
                executor.submit(locate_main.fetch_camera_snapshot, camera)
                for camera in cameras
            ]
            paths = [future.result(timeout=3) for future in futures]

        return paths, observed_temporary_paths

    def test_left_and_right_snapshots_are_isolated_when_fetched_concurrently(
        self,
    ) -> None:
        left_image = _png_bytes("red")
        right_image = _png_bytes("blue")
        responses = {
            locate_main.CAMERA_SNAPSHOT_URLS["left"]: left_image,
            locate_main.CAMERA_SNAPSHOT_URLS["right"]: right_image,
        }
        request_barrier = threading.Barrier(2)

        def fake_get(url: str, **_kwargs) -> _SnapshotResponse:
            request_barrier.wait(timeout=2)
            return _SnapshotResponse(responses[url])

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            paths, temporary_paths = self._fetch_pair(
                cache_dir,
                ("left", "right"),
                fake_get,
            )

            self.assertTrue(all(path is not None for path in paths))
            left_path, right_path = paths
            assert left_path is not None
            assert right_path is not None
            self.assertNotEqual(left_path, right_path)
            self.assertEqual(left_path.read_bytes(), left_image)
            self.assertEqual(right_path.read_bytes(), right_image)
            self.assertEqual(len(temporary_paths), len(set(temporary_paths)))
            self.assertEqual(list(cache_dir.rglob("*.tmp")), [])

    def test_same_camera_snapshots_do_not_overwrite_each_other_when_concurrent(
        self,
    ) -> None:
        images = [_png_bytes("green"), _png_bytes("yellow")]
        request_barrier = threading.Barrier(2)
        response_lock = threading.Lock()
        next_response = 0

        def fake_get(url: str, **_kwargs) -> _SnapshotResponse:
            nonlocal next_response
            self.assertEqual(url, locate_main.CAMERA_SNAPSHOT_URLS["left"])
            with response_lock:
                image = images[next_response]
                next_response += 1
            request_barrier.wait(timeout=2)
            return _SnapshotResponse(image)

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            paths, temporary_paths = self._fetch_pair(
                cache_dir,
                ("left", "left"),
                fake_get,
            )

            self.assertTrue(all(path is not None for path in paths))
            first_path, second_path = paths
            assert first_path is not None
            assert second_path is not None
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(
                {first_path.read_bytes(), second_path.read_bytes()},
                set(images),
            )
            self.assertEqual(len(temporary_paths), len(set(temporary_paths)))
            self.assertEqual(list(cache_dir.rglob("*.tmp")), [])

    def test_identical_monitor_images_use_independent_temporary_files(self) -> None:
        image = _png_bytes("purple")
        write_barrier = threading.Barrier(2)
        observed_temporary_paths: list[Path] = []
        observed_lock = threading.Lock()
        original_write_bytes = Path.write_bytes

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            monitor_dir = root / "monitor"
            source_path.write_bytes(image)

            def synchronized_write_bytes(path: Path, data: bytes) -> int:
                if path.parent == monitor_dir and path.name.endswith(".tmp"):
                    with observed_lock:
                        observed_temporary_paths.append(path)
                    write_barrier.wait(timeout=2)
                return original_write_bytes(path, data)

            with (
                patch.object(locate_main, "MONITOR_IMAGE_DIR", monitor_dir),
                patch.object(Path, "write_bytes", new=synchronized_write_bytes),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                futures = [
                    executor.submit(locate_main.store_monitor_image, source_path)
                    for _ in range(2)
                ]
                stored_paths = [Path(future.result(timeout=3)) for future in futures]

            self.assertEqual(stored_paths[0], stored_paths[1])
            self.assertEqual(stored_paths[0].read_bytes(), image)
            self.assertEqual(
                len(observed_temporary_paths),
                len(set(observed_temporary_paths)),
            )
            self.assertEqual(list(monitor_dir.rglob("*.tmp")), [])

    def test_live_locate_removes_its_private_snapshot_after_inference(self) -> None:
        expected_response = object()
        request = locate_main.LocateRequest(
            task_type="SORTING",
            product_name="test product",
            level="L1",
            hand="left",
        )

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            snapshot_path = cache_dir / "camera_rgb_left_test.png"
            snapshot_path.write_bytes(_png_bytes("orange"))
            with (
                patch.object(locate_main, "CAMERA_SNAPSHOT_CACHE_DIR", cache_dir),
                patch.object(
                    locate_main,
                    "lookup_sku_by_name",
                    return_value={"sku_id": "SKU_TEST", "name": "test product"},
                ),
                patch.object(
                    locate_main,
                    "get_latest_rgb",
                    return_value=snapshot_path,
                ),
                patch.object(
                    locate_main,
                    "locate_product_in_image",
                    return_value=expected_response,
                ),
            ):
                response = locate_main.locate_product_debug(request)

            self.assertIs(response, expected_response)
            self.assertFalse(snapshot_path.exists())

    def test_video_frame_removes_snapshot_only_after_response_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            snapshot_path = cache_dir / "camera_rgb_left_video.png"
            snapshot_path.write_bytes(_png_bytes("white"))
            with (
                patch.object(locate_main, "CAMERA_SNAPSHOT_CACHE_DIR", cache_dir),
                patch.object(
                    locate_main,
                    "get_latest_rgb",
                    return_value=snapshot_path,
                ),
            ):
                response = locate_main.get_video_frame()
                self.assertTrue(snapshot_path.exists())
                assert response.background is not None
                asyncio.run(response.background())

            self.assertFalse(snapshot_path.exists())


if __name__ == "__main__":
    unittest.main()
