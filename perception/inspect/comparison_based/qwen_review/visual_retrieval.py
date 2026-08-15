"""Offline full-catalog visual retrieval for misplaced products.

The model is intentionally loaded lazily: normal shortage inspection and unit tests do
not need torch/transformers.  A configured MISPLACED deployment builds (or loads) one
normalized feature vector per SKU reference image and retrieves a small candidate set
for Qwen to review.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from PIL import Image


PERCEPTION_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG_PATH = PERCEPTION_ROOT / "sku" / "products.json"
DEFAULT_IMAGES_ROOT = PERCEPTION_ROOT / "sku" / "images_new"
DEFAULT_INDEX_PATH = PERCEPTION_ROOT / "sku" / ".feature_cache" / "misplaced_siglip.npz"
DEFAULT_TOP_K = 10


class VisualRetrievalError(RuntimeError):
    """Raised when an explicitly enabled retriever cannot produce candidates."""


@dataclass(frozen=True)
class CatalogImage:
    sku_id: str
    name: str
    path: Path


@dataclass(frozen=True)
class RetrievalMatch:
    sku_id: str
    name: str
    score: float


class VisualSkuRetriever:
    """Retrieve SKU candidates with a locally stored Hugging Face vision model."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        catalog_path: str | Path = DEFAULT_CATALOG_PATH,
        images_root: str | Path = DEFAULT_IMAGES_ROOT,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        top_k: int = DEFAULT_TOP_K,
        batch_size: int = 32,
        device: str = "auto",
        model_loader: Callable[[Path, str], tuple[Any, Any]] | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.catalog_path = Path(catalog_path).expanduser().resolve()
        self.images_root = Path(images_root).expanduser().resolve()
        self.index_path = Path(index_path).expanduser().resolve()
        self.top_k = top_k
        self.batch_size = batch_size
        self.requested_device = device
        self.model_loader = model_loader or _load_transformers_model
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self._catalog: tuple[CatalogImage, ...] | None = None
        self._embeddings: np.ndarray | None = None

        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @classmethod
    def from_environment(cls) -> VisualSkuRetriever | None:
        """Create the retriever only when a local model path is configured."""

        model_path = os.getenv("INSPECT_SKU_RETRIEVAL_MODEL_PATH", "").strip()
        if not model_path:
            return None
        return _cached_environment_retriever(
            model_path,
            os.getenv("INSPECT_SKU_CATALOG_PATH", str(DEFAULT_CATALOG_PATH)),
            os.getenv("INSPECT_SKU_IMAGES_ROOT", str(DEFAULT_IMAGES_ROOT)),
            os.getenv("INSPECT_SKU_RETRIEVAL_INDEX_PATH", str(DEFAULT_INDEX_PATH)),
            int(os.getenv("INSPECT_SKU_RETRIEVAL_TOP_K", str(DEFAULT_TOP_K))),
            int(os.getenv("INSPECT_SKU_RETRIEVAL_BATCH_SIZE", "32")),
            os.getenv("INSPECT_SKU_RETRIEVAL_DEVICE", "auto").strip(),
        )

    def prepare(self) -> tuple[int, int]:
        """Build/load the reference index and return (SKU count, feature size)."""

        catalog, embeddings = self._ensure_index()
        return len(catalog), int(embeddings.shape[1])

    def retrieve(self, image_bgr: np.ndarray) -> list[RetrievalMatch]:
        if image_bgr.size == 0:
            raise VisualRetrievalError("放错商品裁图为空")
        catalog, embeddings = self._ensure_index()
        query = self._embed_pil_images([_bgr_to_pil(image_bgr)])[0]
        scores = embeddings @ query
        limit = min(self.top_k, len(catalog))
        ranked_indices = np.argsort(-scores, kind="stable")[:limit]
        return [
            RetrievalMatch(
                sku_id=catalog[index].sku_id,
                name=catalog[index].name,
                score=float(scores[index]),
            )
            for index in ranked_indices
        ]

    def _ensure_index(self) -> tuple[tuple[CatalogImage, ...], np.ndarray]:
        if self._catalog is not None and self._embeddings is not None:
            return self._catalog, self._embeddings
        catalog = self._load_catalog()
        fingerprint = self._catalog_fingerprint(catalog)
        embeddings = self._load_cached_index(catalog, fingerprint)
        if embeddings is None:
            embeddings = self._build_index(catalog)
            self._write_index(catalog, fingerprint, embeddings)
        self._catalog = catalog
        self._embeddings = embeddings
        return catalog, embeddings

    def _load_catalog(self) -> tuple[CatalogImage, ...]:
        if not self.model_path.is_dir():
            raise VisualRetrievalError(f"本地特征模型目录不存在: {self.model_path}")
        try:
            payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise VisualRetrievalError(f"无法读取 SKU 商品库: {self.catalog_path}") from error
        products = payload.get("products")
        if not isinstance(products, list) or not products:
            raise VisualRetrievalError("SKU 商品库 products 为空或格式错误")

        catalog: list[CatalogImage] = []
        for product in products:
            if not isinstance(product, dict):
                raise VisualRetrievalError("SKU 商品库包含无效商品")
            sku_id = product.get("sku_id")
            name = product.get("name")
            images = product.get("images")
            if not isinstance(sku_id, str) or not isinstance(name, str):
                raise VisualRetrievalError("SKU 商品缺少 sku_id/name")
            if not isinstance(images, list) or not images or not isinstance(images[0], str):
                raise VisualRetrievalError(f"商品 {name} 没有标准图片")
            filename = Path(images[0]).name
            path = self.images_root / filename
            if not path.is_file() and path.suffix.lower() == ".png":
                path = path.with_suffix(".jpg")
            if not path.is_file():
                raise VisualRetrievalError(f"商品 {name} 的标准图片不存在: {path}")
            catalog.append(CatalogImage(sku_id=sku_id, name=name, path=path))
        return tuple(catalog)

    def _catalog_fingerprint(self, catalog: Sequence[CatalogImage]) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.model_path).encode())
        for item in catalog:
            stat = item.path.stat()
            digest.update(
                f"{item.sku_id}\0{item.name}\0{item.path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode(
                    "utf-8"
                )
            )
        return digest.hexdigest()

    def _load_cached_index(
        self,
        catalog: Sequence[CatalogImage],
        fingerprint: str,
    ) -> np.ndarray | None:
        if not self.index_path.is_file():
            return None
        try:
            with np.load(self.index_path, allow_pickle=False) as payload:
                saved_fingerprint = str(payload["fingerprint"].item())
                saved_skus = payload["sku_ids"].astype(str).tolist()
                embeddings = payload["embeddings"].astype(np.float32)
        except (OSError, ValueError, KeyError):
            return None
        expected_skus = [item.sku_id for item in catalog]
        if saved_fingerprint != fingerprint or saved_skus != expected_skus:
            return None
        if embeddings.ndim != 2 or embeddings.shape[0] != len(catalog):
            return None
        return _normalize_rows(embeddings)

    def _build_index(self, catalog: Sequence[CatalogImage]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for offset in range(0, len(catalog), self.batch_size):
            images: list[Image.Image] = []
            for item in catalog[offset : offset + self.batch_size]:
                try:
                    with Image.open(item.path) as source:
                        images.append(source.convert("RGB"))
                except OSError as error:
                    raise VisualRetrievalError(f"无法读取标准图片: {item.path}") from error
            batches.append(self._embed_pil_images(images))
        return np.concatenate(batches, axis=0)

    def _write_index(
        self,
        catalog: Sequence[CatalogImage],
        fingerprint: str,
        embeddings: np.ndarray,
    ) -> None:
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.index_path.with_suffix(self.index_path.suffix + ".tmp.npz")
            np.savez_compressed(
                temporary,
                fingerprint=np.asarray(fingerprint),
                sku_ids=np.asarray([item.sku_id for item in catalog]),
                names=np.asarray([item.name for item in catalog]),
                embeddings=embeddings.astype(np.float32),
            )
            temporary.replace(self.index_path)
        except OSError as error:
            raise VisualRetrievalError(f"无法写入特征索引: {self.index_path}") from error

    def _embed_pil_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        processor, model, device = self._ensure_model()
        try:
            import torch

            inputs = processor(images=list(images), return_tensors="pt")
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                if hasattr(model, "get_image_features"):
                    features = model.get_image_features(**inputs)
                    # transformers 4.x returns a Tensor here, while 5.x may
                    # return BaseModelOutputWithPooling.
                    if not hasattr(features, "detach"):
                        pooled = getattr(features, "pooler_output", None)
                        if pooled is None:
                            pooled = features.last_hidden_state[:, 0]
                        features = pooled
                else:
                    output = model(**inputs)
                    features = getattr(output, "pooler_output", None)
                    if features is None:
                        features = output.last_hidden_state[:, 0]
            array = features.detach().float().cpu().numpy()
        except Exception as error:  # model backends expose several exception types
            raise VisualRetrievalError(f"视觉特征推理失败: {error}") from error
        return _normalize_rows(array)

    def _ensure_model(self) -> tuple[Any, Any, str]:
        if self._processor is not None and self._model is not None and self._device:
            return self._processor, self._model, self._device
        device = _resolve_device(self.requested_device)
        try:
            processor, model = self.model_loader(self.model_path, device)
        except Exception as error:
            raise VisualRetrievalError(f"无法加载本地视觉特征模型: {error}") from error
        self._processor = processor
        self._model = model
        self._device = device
        return processor, model, device


def _load_transformers_model(model_path: Path, device: str) -> tuple[Any, Any]:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise VisualRetrievalError("缺少 transformers/torch 依赖") from error
    # Retrieval uses only the vision tower. AutoProcessor would also initialize
    # SigLIP's text tokenizer and unnecessarily require sentencepiece/protobuf.
    processor = AutoImageProcessor.from_pretrained(
        model_path, local_files_only=True
    )
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    model.eval()
    model.to(device)
    return processor, model


@lru_cache(maxsize=4)
def _cached_environment_retriever(
    model_path: str,
    catalog_path: str,
    images_root: str,
    index_path: str,
    top_k: int,
    batch_size: int,
    device: str,
) -> VisualSkuRetriever:
    # Reusing this instance keeps the model resident on GPU/MPS across requests.
    return VisualSkuRetriever(
        model_path=model_path,
        catalog_path=catalog_path,
        images_root=images_root,
        index_path=index_path,
        top_k=top_k,
        batch_size=batch_size,
        device=device,
    )


def _resolve_device(requested: str) -> str:
    try:
        import torch
    except ImportError as error:
        raise VisualRetrievalError("缺少 torch 依赖") from error
    normalized = requested.strip().lower()
    if normalized != "auto":
        return normalized
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _bgr_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim != 3 or image.shape[2] != 3:
        raise VisualRetrievalError("商品查询图必须是 BGR 三通道图像")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)
