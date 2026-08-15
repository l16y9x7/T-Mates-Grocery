from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


QWEN_REVIEW_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_ROOT = QWEN_REVIEW_ROOT.parent
if str(COMPARISON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARISON_ROOT))

from qwen_review.visual_retrieval import VisualSkuRetriever  # noqa: E402


class DeterministicRetriever(VisualSkuRetriever):
    """Tiny color feature extractor used to test ranking/cache without torch."""

    def _embed_pil_images(self, images):  # type: ignore[no-untyped-def]
        features = []
        for image in images:
            rgb = np.asarray(image, dtype=np.float32)
            features.append(rgb.mean(axis=(0, 1)))
        values = np.asarray(features, dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1e-12)


class VisualRetrievalTest(unittest.TestCase):
    def test_full_catalog_ranks_by_feature_and_writes_reusable_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            images = root / "images_new"
            model.mkdir()
            images.mkdir()
            red = np.zeros((40, 30, 3), dtype=np.uint8)
            red[:, :, 2] = 255
            green = np.zeros((40, 30, 3), dtype=np.uint8)
            green[:, :, 1] = 255
            cv2.imwrite(str(images / "SKU_RED.jpg"), red)
            cv2.imwrite(str(images / "SKU_GREEN.jpg"), green)
            catalog = root / "products.json"
            catalog.write_text(
                json.dumps(
                    {
                        "products": [
                            {
                                "sku_id": "SKU_RED",
                                "name": "红色商品",
                                "images": ["images/SKU_RED.jpg"],
                                "locations": ["H1_F_L1_C01"],
                            },
                            {
                                "sku_id": "SKU_GREEN",
                                "name": "绿色商品",
                                "images": ["images/SKU_GREEN.jpg"],
                                "locations": ["H2_B_L5_C01"],
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            index = root / "cache" / "index.npz"
            retriever = DeterministicRetriever(
                model_path=model,
                catalog_path=catalog,
                images_root=images,
                index_path=index,
                top_k=2,
            )

            matches = retriever.retrieve(red)

            self.assertEqual([match.sku_id for match in matches], ["SKU_RED", "SKU_GREEN"])
            self.assertGreater(matches[0].score, matches[1].score)
            self.assertTrue(index.is_file())
            self.assertEqual(retriever.prepare(), (2, 3))


if __name__ == "__main__":
    unittest.main()
