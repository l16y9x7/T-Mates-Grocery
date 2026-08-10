from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import cv2
import numpy as np


QWEN_REVIEW_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_ROOT = QWEN_REVIEW_ROOT.parent
if str(COMPARISON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARISON_ROOT))

from qwen_review import (  # noqa: E402
    QwenReviewError,
    QwenReviewer,
    normalize_reference_image,
)


class FakeResponse:
    def __init__(
        self,
        *,
        payload: Any | None = None,
        content: bytes = b"",
        content_type: str = "application/json",
        status_code: int = 200,
    ) -> None:
        self._payload = payload
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(payload, ensure_ascii=False) if payload is not None else ""

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self, qwen_content: str) -> None:
        self.qwen_content = qwen_content
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        image = np.full((80, 60, 3), 120, dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise RuntimeError("cannot create reference image")
        self.image = encoded.tobytes()

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        if url.endswith("/sku/get_candidate_SKU"):
            return FakeResponse(
                payload=[
                    [
                        {"sku_id": "SKU_A", "name": "绿色奥利奥"},
                        {"sku_id": "SKU_B", "name": "棕色奥利奥"},
                    ]
                ]
            )
        if url.endswith("/sku/get_image"):
            name = kwargs["params"]["name"]
            suffix = "a" if name == "绿色奥利奥" else "b"
            return FakeResponse(payload=[f"images/{suffix}.jpg"])
        if url.endswith("/images/a.jpg") or url.endswith("/images/b.jpg"):
            return FakeResponse(content=self.image, content_type="image/jpeg")
        if url.endswith("/chat/completions"):
            return FakeResponse(
                payload={
                    "choices": [
                        {"message": {"content": self.qwen_content}}
                    ]
                }
            )
        raise AssertionError(f"unexpected request: {method} {url}")


class QwenReviewerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = np.full((720, 1280, 3), 40, dtype=np.uint8)
        self.current = self.baseline.copy()
        cv2.rectangle(self.baseline, (300, 300), (400, 520), (0, 180, 0), -1)

    def test_shortage_fetches_candidates_images_and_reviews_region(self) -> None:
        session = FakeSession(
            json.dumps(
                {
                    "shortage_product_name": "绿色奥利奥",
                    "confidence": 0.94,
                },
                ensure_ascii=False,
            )
        )
        reviewer = QwenReviewer(
            sku_base_url="http://sku",
            qwen_url="http://qwen/v1",
            session=session,
        )

        result = reviewer.review(
            task_type="SHORTAGE",
            location_id="H1_F_L2_C03",
            pose_type="SHELF_VIEW_UPPER",
            current=self.current,
            bboxes=[[300, 300, 101, 221]],
        )

        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].shortage_product_name, "绿色奥利奥")
        candidate_call = session.calls[0]
        self.assertEqual(candidate_call[0], "GET")
        self.assertEqual(
            candidate_call[2]["json"],
            {
                "location_id": "H1_F_L2_C03",
                "pose_type": "SHELF_VIEW_UPPER",
            },
        )
        image_lookups = [
            call for call in session.calls if call[1].endswith("/sku/get_image")
        ]
        self.assertEqual(len(image_lookups), 2)
        qwen_payload = session.calls[-1][2]["json"]
        serialized = json.dumps(qwen_payload, ensure_ascii=False)
        self.assertNotIn("同列后排仍可见", serialized)
        self.assertIn("候选商品：绿色奥利奥、棕色奥利奥", serialized)
        self.assertIn("CANDIDATE 1: 绿色奥利奥;", serialized)
        self.assertNotIn("sku_id=SKU_A", serialized)
        self.assertNotIn("location_id=H1_F_L2_C03", serialized)
        user_content = qwen_payload["messages"][1]["content"]
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertIn("候选商品：", user_content[2]["text"])
        self.assertIn("绿色奥利奥", serialized)
        self.assertIn("棕色奥利奥", serialized)
        self.assertNotIn("BEFORE", serialized)
        self.assertNotIn("AFTER", serialized)
        self.assertNotIn("差分算法", serialized)
        system_prompt = qwen_payload["messages"][0]["content"]
        self.assertIn("shortage_product_name", system_prompt)
        self.assertNotIn("misplaced_product_name", system_prompt)

    def test_oversized_candidate_image_is_downscaled(self) -> None:
        image = np.full((1536, 2048, 3), (50, 120, 220), dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(success)

        normalized, media_type = normalize_reference_image(
            encoded.tobytes(),
            "image/jpeg",
        )
        decoded = cv2.imdecode(np.frombuffer(normalized, dtype=np.uint8), cv2.IMREAD_COLOR)

        self.assertEqual(media_type, "image/jpeg")
        self.assertIsNotNone(decoded)
        self.assertEqual(max(decoded.shape[:2]), 1024)

    def test_misplaced_rejects_product_outside_candidates(self) -> None:
        session = FakeSession(
            json.dumps(
                {
                    "misplaced_product_name": "不存在商品",
                    "gt_product_name": "绿色奥利奥",
                    "confidence": 0.8,
                },
                ensure_ascii=False,
            )
        )
        reviewer = QwenReviewer(
            sku_base_url="http://sku",
            qwen_url="http://qwen/v1",
            session=session,
        )

        with self.assertRaises(QwenReviewError) as context:
            reviewer.review(
                task_type="MISPLACED",
                location_id="H1_F_L2_C03",
                pose_type="",
                current=self.current,
                bboxes=[[300, 300, 101, 221]],
            )

        self.assertEqual(context.exception.stage, "qwen_output_validation")
        qwen_payload = session.calls[-1][2]["json"]
        system_prompt = qwen_payload["messages"][0]["content"]
        self.assertIn("misplaced_product_name", system_prompt)
        self.assertIn("gt_product_name", system_prompt)
        self.assertNotIn("shortage_product_name", system_prompt)

    def test_no_bbox_skips_all_upstream_requests(self) -> None:
        session = FakeSession(
            '{"shortage_product_name":"UNKNOWN","confidence":0}'
        )
        reviewer = QwenReviewer(
            sku_base_url="http://sku",
            qwen_url="http://qwen/v1",
            session=session,
        )

        result = reviewer.review(
            task_type="SHORTAGE",
            location_id="H1_F_L2_C03",
            pose_type="",
            current=self.current,
            bboxes=[],
        )

        self.assertEqual(result.findings, ())
        self.assertEqual(session.calls, [])

    def test_each_bbox_uses_a_separate_qwen_request(self) -> None:
        session = FakeSession(
            json.dumps(
                {
                    "shortage_product_name": "绿色奥利奥",
                    "confidence": 0.7,
                },
                ensure_ascii=False,
            )
        )
        reviewer = QwenReviewer(
            sku_base_url="http://sku",
            qwen_url="http://qwen/v1",
            session=session,
        )

        result = reviewer.review(
            task_type="SHORTAGE",
            location_id="H1_F_L2_C03",
            pose_type="",
            current=self.current,
            bboxes=[[100, 200, 80, 160], [500, 220, 90, 170]],
        )

        qwen_calls = [
            call for call in session.calls if call[1].endswith("/chat/completions")
        ]
        self.assertEqual(len(qwen_calls), 2)
        self.assertEqual([item.region_index for item in result.findings], [1, 2])
        for call in qwen_calls:
            content = call[2]["json"]["messages"][1]["content"]
            image_items = [item for item in content if item["type"] == "image_url"]
            # One expanded bbox crop plus one standard image for each candidate.
            self.assertEqual(len(image_items), 3)

    def test_debug_directory_saves_prompt_crop_and_results(self) -> None:
        session = FakeSession(
            json.dumps(
                {
                    "shortage_product_name": "绿色奥利奥",
                    "confidence": 0.88,
                },
                ensure_ascii=False,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            reviewer = QwenReviewer(
                sku_base_url="http://sku",
                qwen_url="http://qwen/v1",
                session=session,
                debug_root=directory,
            )

            result = reviewer.review(
                task_type="SHORTAGE",
                location_id="H1_F_L2_C03",
                pose_type="",
                current=self.current,
                bboxes=[[300, 300, 101, 221]],
            )

            self.assertIsNotNone(result.debug_directory)
            debug = result.debug_directory
            assert debug is not None
            expected = {
                debug / "request.json",
                debug / "candidates.json",
                debug / "result.json",
                debug / "region_01" / "bbox_expanded.jpg",
                debug / "region_01" / "prompt.txt",
                debug / "region_01" / "qwen_raw.txt",
                debug / "region_01" / "parsed_result.json",
            }
            self.assertTrue(all(path.is_file() for path in expected))
            prompt = (debug / "region_01" / "prompt.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("=== SYSTEM ===", prompt)
            self.assertIn("[IMAGE 1]", prompt)
            self.assertNotIn("data:image", prompt)
            crop = cv2.imdecode(
                np.fromfile(debug / "region_01" / "bbox_expanded.jpg", dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            self.assertEqual(crop.shape[:2], (421, 161))


if __name__ == "__main__":
    unittest.main()
