"""Download the visual model once and optionally build the SKU feature index."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PERCEPTION_ROOT = ROOT.parent
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))
INSPECT_ROOT = PERCEPTION_ROOT / "inspect"
if str(INSPECT_ROOT) not in sys.path:
    sys.path.insert(0, str(INSPECT_ROOT))

DEFAULT_MODEL_ID = "google/siglip-base-patch16-224"
DEFAULT_MODEL_DIR = ROOT / "models" / "siglip-base-patch16-224"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="只下载模型，不生成当前商品目录的特征索引",
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cuda", "mps", "cpu")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "缺少 huggingface_hub，请先安装 perception/requirements.txt"
        ) from error

    model_dir = args.model_dir.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model_id,
        local_dir=model_dir,
    )
    print(f"模型已下载到: {model_dir}")
    if args.download_only:
        return

    os.environ["INSPECT_SKU_RETRIEVAL_MODEL_PATH"] = str(model_dir)
    os.environ["INSPECT_SKU_RETRIEVAL_DEVICE"] = args.device
    from comparison_based.qwen_review.visual_retrieval import VisualSkuRetriever

    retriever = VisualSkuRetriever.from_environment()
    if retriever is None:
        raise SystemExit("特征检索配置未生效")
    sku_count, feature_size = retriever.prepare()
    print(f"特征索引已生成: {sku_count} SKU, {feature_size} 维")
    print(f"运行前设置: INSPECT_SKU_RETRIEVAL_MODEL_PATH={model_dir}")


if __name__ == "__main__":
    main()
