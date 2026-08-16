from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PERCEPTION_ROOT = Path(__file__).resolve().parents[2]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from camera_capture import CameraCaptureError, capture_head_rgbd


class _Response:
    def __init__(self, content: bytes, headers: dict[str, str] | None = None) -> None:
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        return None


class _Session:
    def __init__(self, rgb: bytes, depth: bytes, depth_headers: dict[str, str]) -> None:
        self.rgb = rgb
        self.depth = depth
        self.depth_headers = depth_headers
        self.calls: list[str] = []

    def get(self, url: str, *, timeout: float) -> _Response:
        self.calls.append(url)
        if "type=depth" in url:
            return _Response(self.depth, self.depth_headers)
        return _Response(self.rgb)


class CameraCaptureTest(unittest.TestCase):
    def setUp(self) -> None:
        image = np.full((24, 32, 3), (20, 80, 160), dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise RuntimeError("failed to encode test RGB")
        self.rgb = encoded.tobytes()
        self.depth = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)

    def test_capture_saves_validated_rgb_depth_and_metadata(self) -> None:
        headers = {
            "X-Image-Width": "32",
            "X-Image-Height": "24",
            "X-Image-Encoding": "16UC1",
            "X-Image-Step": "64",
            "X-Image-Is-Bigendian": "0",
        }
        session = _Session(self.rgb, self.depth.astype("<u2").tobytes(), headers)

        with tempfile.TemporaryDirectory() as directory:
            result = capture_head_rgbd(directory, session=session)

            self.assertEqual(result.rgb.shape, (24, 32, 3))
            np.testing.assert_array_equal(result.depth_mm, self.depth)
            self.assertTrue(result.rgb_path.is_file())
            self.assertTrue(result.depth_path.is_file())
            self.assertTrue((Path(directory) / "meta.json").is_file())
            np.testing.assert_array_equal(
                np.load(result.depth_path, allow_pickle=False),
                self.depth,
            )
        self.assertEqual(len(session.calls), 2)

    def test_capture_rejects_rgb_depth_size_mismatch(self) -> None:
        headers = {
            "X-Image-Width": "16",
            "X-Image-Height": "12",
            "X-Image-Encoding": "16UC1",
            "X-Image-Step": "32",
            "X-Image-Is-Bigendian": "0",
        }
        depth = np.zeros((12, 16), dtype="<u2").tobytes()
        session = _Session(self.rgb, depth, headers)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CameraCaptureError, "size mismatch"):
                capture_head_rgbd(directory, session=session)


if __name__ == "__main__":
    unittest.main()
