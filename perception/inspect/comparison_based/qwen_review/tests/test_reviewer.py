from __future__ import annotations

import base64
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
    CandidateProduct,
    QwenReviewError,
    QwenReviewer,
    ReviewRowConstraint,
    build_candidate_contact_sheets,
    normalize_reference_image,
)
from qwen_review.visual_retrieval import RetrievalMatch


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
    def __init__(
        self,
        qwen_content: str | list[str],
        *,
        candidate_rows: list[list[dict[str, str]]] | None = None,
    ) -> None:
        self.qwen_contents = (
            list(qwen_content) if isinstance(qwen_content, list) else [qwen_content]
        )
        self.qwen_call_count = 0
        self.candidate_rows = candidate_rows or [
            [
                {"sku_id": "SKU_A", "name": "绿色奥利奥"},
                {"sku_id": "SKU_B", "name": "棕色奥利奥"},
            ]
        ]
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        image = np.full((80, 60, 3), 120, dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise RuntimeError("cannot create reference image")
        self.image = encoded.tobytes()

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        if url.endswith("/sku/get_candidate_SKU") or url.endswith(
            "/sku/get_inspection_candidate_SKU"
        ):
            return FakeResponse(payload=self.candidate_rows)
        if url.endswith("/sku/get_image"):
            name = kwargs["params"]["name"]
            suffix = "a" if name == "绿色奥利奥" else "b"
            return FakeResponse(payload=[f"images/{suffix}.jpg"])
        if url.endswith("/sku/search_by_SKU"):
            sku = kwargs["params"]["sku"]
            products = {
                "SKU_GLOBAL": {
                    "sku_id": "SKU_GLOBAL",
                    "name": "全库商品",
                    "images": ["images/global.jpg"],
                    "locations": ["H2_B_L4_C03"],
                }
            }
            return FakeResponse(payload=products[sku])
        if (
            url.endswith("/images/a.jpg")
            or url.endswith("/images/b.jpg")
            or url.endswith("/images/global.jpg")
        ):
            return FakeResponse(content=self.image, content_type="image/jpeg")
        if url.endswith("/chat/completions"):
            content_index = min(
                self.qwen_call_count,
                len(self.qwen_contents) - 1,
            )
            content = self.qwen_contents[content_index]
            self.qwen_call_count += 1
            return FakeResponse(
                payload={
                    "choices": [
                        {"message": {"content": content}}
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
                    "product_name": "绿色奥利奥",
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
            baseline=self.baseline,
            bboxes=[[300, 300, 101, 221]],
        )

        self.assertEqual(len(result.findings), 1)
        self.assertEqual(len(result.prompts), 1)
        self.assertIn("=== SYSTEM ===", result.prompts[0])
        self.assertIn("1: 绿色奥利奥", result.prompts[0])
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
        self.assertIn("1: 绿色奥利奥", serialized)
        self.assertIn("2: 棕色奥利奥", serialized)
        self.assertIn("候选（与下方标准图拼图上方数字一致）", serialized)
        self.assertNotIn("所有输出商品名必须从以上名称中逐字选择。", serialized)
        self.assertNotIn("sku_id=SKU_A", serialized)
        self.assertNotIn("location_id=H1_F_L2_C03", serialized)
        user_content = qwen_payload["messages"][1]["content"]
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertEqual(
            user_content[0]["text"],
            "请只审核下面这一张货架局部图："
            "只能从以下候选商品中选择最像的一个商品。",
        )
        region_bytes = base64.b64decode(
            user_content[1]["image_url"]["url"].split(",", 1)[1]
        )
        region_image = cv2.imdecode(
            np.frombuffer(region_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertGreater(int(region_image[110, 40, 1]), 120)
        self.assertLess(int(region_image[110, 40, 0]), 40)
        self.assertIn("候选（与下方标准图拼图上方数字一致）", user_content[2]["text"])
        self.assertIn("绿色奥利奥", serialized)
        self.assertIn("棕色奥利奥", serialized)
        self.assertNotIn("BEFORE", serialized)
        self.assertNotIn("AFTER", serialized)
        self.assertNotIn("差分算法", serialized)
        system_prompt = qwen_payload["messages"][0]["content"]
        self.assertIn('"product_name"', system_prompt)
        self.assertNotIn("shortage_product_name", system_prompt)
        self.assertNotIn("misplaced_product_name", system_prompt)
        image_items = [item for item in user_content if item["type"] == "image_url"]
        self.assertEqual(len(image_items), 2)
        sheet_bytes = base64.b64decode(
            image_items[1]["image_url"]["url"].split(",", 1)[1]
        )
        sheet_image = cv2.imdecode(
            np.frombuffer(sheet_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(sheet_image)
        self.assertGreater(sheet_image.shape[1], sheet_image.shape[0])

    def test_inspection_target_uses_view_candidate_endpoint(self) -> None:
        session = FakeSession(
            json.dumps(
                {
                    "product_name": "绿色奥利奥",
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

        reviewer.review(
            task_type="SHORTAGE",
            location_id="h1_f_l_inspect",
            pose_type="SHELF_VIEW_UPPER",
            current=self.current,
            baseline=self.baseline,
            bboxes=[[300, 300, 101, 221]],
        )

        candidate_call = session.calls[0]
        self.assertTrue(
            candidate_call[1].endswith("/sku/get_inspection_candidate_SKU")
        )
        self.assertEqual(
            candidate_call[2]["json"],
            {
                "location_id": "H1_F_L_INSPECT",
                "pose_type": "SHELF_VIEW_UPPER",
            },
        )

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

    def test_candidate_contact_sheets_are_numbered_and_bounded(self) -> None:
        candidates = [
            CandidateProduct(
                sku_id=f"SKU_{index}",
                name=f"商品{index}",
                row_numbers=(1,),
                image=FakeSession("{}").image,
                media_type="image/jpeg",
            )
            for index in range(1, 22)
        ]

        sheets = build_candidate_contact_sheets(candidates)

        self.assertEqual(len(sheets), 2)
        self.assertEqual(
            (sheets[0].first_candidate_number, sheets[0].last_candidate_number),
            (1, 20),
        )
        self.assertEqual(
            (sheets[1].first_candidate_number, sheets[1].last_candidate_number),
            (21, 21),
        )
        first = cv2.imdecode(
            np.frombuffer(sheets[0].image, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(first)
        self.assertLessEqual(first.shape[1], 1024)
        self.assertLessEqual(first.shape[0], 1024)

    def test_shortage_row_constraint_fetches_only_expected_row_candidates(self) -> None:
        session = FakeSession(
            json.dumps(
                {
                    "shortage_product_name": "棕色奥利奥",
                    "confidence": 0.93,
                },
                ensure_ascii=False,
            ),
            candidate_rows=[
                [{"sku_id": "SKU_A", "name": "绿色奥利奥"}],
                [{"sku_id": "SKU_R", "name": "红色奥利奥"}],
                [
                    {"sku_id": "SKU_B", "name": "棕色奥利奥"},
                    {"sku_id": "SKU_C", "name": "蓝色奥利奥"},
                ],
            ],
        )
        reviewer = QwenReviewer(
            sku_base_url="http://sku",
            qwen_url="http://qwen/v1",
            session=session,
        )

        result = reviewer.review(
            task_type="SHORTAGE",
            location_id="H1_F_L2_C03",
            pose_type="SHELF_VIEW_LOWER",
            current=self.current,
            bboxes=[[300, 500, 101, 120]],
            row_constraints=[
                ReviewRowConstraint(
                    row_index=3,
                    row_bbox=(0, 480, 1280, 180),
                    overlap_ratio=1.0,
                    detected_row_index=4,
                )
            ],
        )

        self.assertEqual(result.findings[0].shortage_product_name, "棕色奥利奥")
        image_lookups = [
            call for call in session.calls if call[1].endswith("/sku/get_image")
        ]
        self.assertEqual(
            [call[2]["params"]["name"] for call in image_lookups],
            ["棕色奥利奥", "蓝色奥利奥"],
        )
        serialized = json.dumps(session.calls[-1][2]["json"], ensure_ascii=False)
        self.assertNotIn("画面检测第 4 行", serialized)
        self.assertIn("棕色奥利奥", serialized)
        self.assertIn("蓝色奥利奥", serialized)
        self.assertNotIn("绿色奥利奥", serialized)
        self.assertNotIn("红色奥利奥", serialized)
        region_url = session.calls[-1][2]["json"]["messages"][1]["content"][1][
            "image_url"
        ]["url"]
        region_bytes = base64.b64decode(region_url.split(",", 1)[1])
        region_image = cv2.imdecode(
            np.frombuffer(region_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(region_image.shape[:2], (204, 161))

    def test_misplaced_row_constraint_limits_only_expected_product(self) -> None:
        candidate_rows = [
            [
                {"sku_id": "SKU_A", "name": "绿色奥利奥"},
                {"sku_id": "SKU_C", "name": "蓝色奥利奥"},
            ],
            [{"sku_id": "SKU_B", "name": "棕色奥利奥"}],
        ]
        constraint = ReviewRowConstraint(
            row_index=1,
            row_bbox=(0, 0, 1280, 360),
            overlap_ratio=1.0,
        )
        session = FakeSession(
            [
                json.dumps(
                    {
                        "misplaced_product_name": "棕色奥利奥",
                        "confidence": 0.92,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "gt_product_name": "绿色奥利奥",
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                ),
            ],
            candidate_rows=candidate_rows,
        )
        reviewer = QwenReviewer(
            sku_base_url="http://sku",
            qwen_url="http://qwen/v1",
            session=session,
        )

        result = reviewer.review(
            task_type="MISPLACED",
            location_id="H1_F_L1_C03",
            pose_type="SHELF_VIEW_UPPER",
            current=self.current,
            baseline=self.baseline,
            bboxes=[[300, 180, 101, 120]],
            row_constraints=[constraint],
        )

        self.assertEqual(result.findings[0].misplaced_product_name, "棕色奥利奥")
        self.assertEqual(result.findings[0].gt_product_name, "绿色奥利奥")
        self.assertEqual(result.findings[0].confidence, 0.9)
        qwen_calls = [
            call for call in session.calls if call[1].endswith("/chat/completions")
        ]
        self.assertEqual(len(qwen_calls), 2)
        misplaced_serialized = json.dumps(
            qwen_calls[0][2]["json"],
            ensure_ascii=False,
        )
        expected_serialized = json.dumps(
            qwen_calls[1][2]["json"],
            ensure_ascii=False,
        )
        self.assertIn("候选来自全量商品标准库", misplaced_serialized)
        self.assertIn(
            "任务:识别局部图红色 bbox 中当前实际放置的商品。",
            misplaced_serialized,
        )
        self.assertNotIn("location_id=", misplaced_serialized)
        self.assertIn("SKU 1: 绿色奥利奥", misplaced_serialized)
        self.assertIn("SKU 2: 蓝色奥利奥", misplaced_serialized)
        self.assertIn("SKU 3: 棕色奥利奥", misplaced_serialized)
        self.assertIn("绿色奥利奥", misplaced_serialized)
        self.assertIn("棕色奥利奥", misplaced_serialized)
        self.assertIn("货架标准放置图", expected_serialized)
        self.assertIn("红色 bbox 中的物体是什么", expected_serialized)
        self.assertIn("标准放置组合图", expected_serialized)
        self.assertIn("这一层从左到右SKU标准放置编号如下", expected_serialized)
        self.assertIn("输出商品名必须从以上名称中逐字选择", expected_serialized)
        self.assertNotIn("原本应该放置", expected_serialized)
        self.assertNotIn("对比图", expected_serialized)
        self.assertIn("上半部分是完整标准放置行", expected_serialized)
        self.assertIn("下半部分是该红色 bbox 内物体的原图抠图", expected_serialized)
        self.assertIn("标准放置组合图", expected_serialized)
        self.assertIn("任务：根据候选SKU，识别红色 bbox 中的物体是什么", expected_serialized)
        self.assertNotIn("候选 SKU 1..N 按标准货位列号从左到右排列", expected_serialized)
        self.assertNotIn("任务=MISPLACED 第二阶段", expected_serialized)
        self.assertIn("绿色奥利奥", expected_serialized)
        self.assertIn("蓝色奥利奥", expected_serialized)
        self.assertNotIn("棕色奥利奥", expected_serialized)
        self.assertLess(
            expected_serialized.index("SKU 1: 绿色奥利奥"),
            expected_serialized.index("SKU 2: 蓝色奥利奥"),
        )
        misplaced_content = qwen_calls[0][2]["json"]["messages"][1]["content"]
        self.assertEqual(
            len([item for item in misplaced_content if item["type"] == "image_url"]),
            2,
        )
        misplaced_url = next(
            item["image_url"]["url"]
            for item in misplaced_content
            if item["type"] == "image_url"
        )
        misplaced_image = cv2.imdecode(
            np.frombuffer(
                base64.b64decode(misplaced_url.split(",", 1)[1]),
                dtype=np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(misplaced_image.shape[:2], (324, 201))
        misplaced_red_pixels = (
            (misplaced_image[:, :, 2] > 180)
            & (misplaced_image[:, :, 1] < 100)
            & (misplaced_image[:, :, 0] < 100)
        )
        self.assertFalse(misplaced_red_pixels[2, 50])
        self.assertTrue(misplaced_red_pixels[12, 50])
        self.assertTrue(misplaced_red_pixels[312, 50])
        self.assertFalse(misplaced_red_pixels[321, 50])
        self.assertTrue(misplaced_red_pixels[160, 151])
        self.assertFalse(misplaced_red_pixels[160, 25])
        self.assertFalse(misplaced_red_pixels[160, 175])
        expected_content = qwen_calls[1][2]["json"]["messages"][1]["content"]
        self.assertEqual(
            len([item for item in expected_content if item["type"] == "image_url"]),
            2,
        )
        row_url = next(
            item["image_url"]["url"]
            for item in expected_content
            if item["type"] == "image_url"
        )
        row_image = cv2.imdecode(
            np.frombuffer(base64.b64decode(row_url.split(",", 1)[1]), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(row_image.shape[:2], (656, 1280))
        red_pixels = (
            (row_image[:, :, 2] > 180)
            & (row_image[:, :, 1] < 100)
            & (row_image[:, :, 0] < 100)
        )
        self.assertGreater(int(red_pixels.sum()), 200)
        self.assertFalse(red_pixels[2, 350])
        self.assertTrue(red_pixels[12, 350])
        self.assertTrue(red_pixels[312, 350])
        self.assertFalse(red_pixels[500, 640])
        self.assertGreater(int(row_image[270, 350, 1]), 100)

    def test_misplaced_uses_full_catalog_retrieval_for_first_stage_only(self) -> None:
        candidate_rows = [
            [
                {"sku_id": "SKU_A", "name": "绿色奥利奥"},
                {"sku_id": "SKU_C", "name": "蓝色奥利奥"},
            ],
            [{"sku_id": "SKU_B", "name": "棕色奥利奥"}],
        ]
        constraint = ReviewRowConstraint(
            row_index=1,
            row_bbox=(0, 0, 1280, 360),
            overlap_ratio=1.0,
        )

        class FakeRetriever:
            def retrieve(self, image: np.ndarray) -> list[RetrievalMatch]:
                self.image = image
                return [RetrievalMatch("SKU_GLOBAL", "全库商品", 0.93)]

        retriever = FakeRetriever()
        session = FakeSession(
            [
                json.dumps(
                    {"misplaced_product_name": "全库商品", "confidence": 0.9},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {"gt_product_name": "绿色奥利奥", "confidence": 0.8},
                    ensure_ascii=False,
                ),
            ]
        )
        reviewer = QwenReviewer(
            sku_base_url="http://sku",
            qwen_url="http://qwen/v1",
            session=session,
            visual_retriever=retriever,
        )

        result = reviewer.review(
            task_type="MISPLACED",
            location_id="H1_F_L1_C03",
            pose_type="SHELF_VIEW_UPPER",
            current=self.current,
            baseline=self.baseline,
            bboxes=[[300, 180, 101, 120]],
            row_constraints=[
                ReviewRowConstraint(
                    row_index=1,
                    row_bbox=(0, 0, 1280, 360),
                    overlap_ratio=1.0,
                )
            ],
        )

        self.assertEqual(result.findings[0].misplaced_product_name, "全库商品")
        self.assertEqual(result.findings[0].gt_product_name, "绿色奥利奥")
        qwen_calls = [
            call for call in session.calls if call[1].endswith("/chat/completions")
        ]
        misplaced_payload = json.dumps(qwen_calls[0][2]["json"], ensure_ascii=False)
        expected_payload = json.dumps(qwen_calls[1][2]["json"], ensure_ascii=False)
        self.assertIn("全库商品", misplaced_payload)
        self.assertNotIn("绿色奥利奥", misplaced_payload)
        self.assertIn("绿色奥利奥", expected_payload)
        self.assertNotIn("全库商品", expected_payload)

        invalid_session = FakeSession(
            [
                json.dumps(
                    {
                        "misplaced_product_name": "棕色奥利奥",
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "gt_product_name": "棕色奥利奥",
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                ),
            ],
            candidate_rows=candidate_rows,
        )
        invalid_reviewer = QwenReviewer(
            sku_base_url="http://sku",
            qwen_url="http://qwen/v1",
            session=invalid_session,
        )
        with self.assertRaises(QwenReviewError):
            invalid_reviewer.review(
                task_type="MISPLACED",
                location_id="H1_F_L1_C03",
                pose_type="SHELF_VIEW_UPPER",
                current=self.current,
                baseline=self.baseline,
                bboxes=[[300, 180, 101, 120]],
                row_constraints=[constraint],
            )

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
        self.assertNotIn("gt_product_name", system_prompt)
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
            # One expanded bbox crop plus one numbered candidate contact sheet.
            self.assertEqual(len(image_items), 2)

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
                baseline=self.current.copy(),
                debug_current_depth_mm=np.full(
                    self.current.shape[:2],
                    1100,
                    dtype=np.uint16,
                ),
                debug_baseline_depth_mm=np.full(
                    self.current.shape[:2],
                    900,
                    dtype=np.uint16,
                ),
                bboxes=[[300, 300, 101, 221]],
            )

            self.assertIsNotNone(result.debug_directory)
            debug = result.debug_directory
            assert debug is not None
            expected = {
                debug / "request.json",
                debug / "candidates.json",
                debug / "result.json",
                debug / "current_rgb.jpg",
                debug / "current_depth_mm.npy",
                debug / "baseline_rgb.jpg",
                debug / "baseline_depth_mm.npy",
                debug / "rgbd.json",
                debug / "region_01" / "bbox_expanded.jpg",
                debug / "region_01" / "prompt.txt",
                debug / "region_01" / "qwen_image_01.jpg",
                debug / "region_01" / "qwen_image_02.jpg",
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
            np.testing.assert_array_equal(
                np.load(debug / "current_depth_mm.npy", allow_pickle=False),
                np.full(self.current.shape[:2], 1100, dtype=np.uint16),
            )
            np.testing.assert_array_equal(
                np.load(debug / "baseline_depth_mm.npy", allow_pickle=False),
                np.full(self.current.shape[:2], 900, dtype=np.uint16),
            )


if __name__ == "__main__":
    unittest.main()
