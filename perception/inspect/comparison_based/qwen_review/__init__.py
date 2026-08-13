"""Qwen semantic review for comparison-based shelf findings."""

from .reviewer import (
    CandidateProduct,
    DEFAULT_DEBUG_ROOT,
    PoseType,
    QwenReviewError,
    QwenReviewResult,
    QwenReviewer,
    ReviewedFinding,
    normalize_reference_image,
)

__all__ = [
    "CandidateProduct",
    "DEFAULT_DEBUG_ROOT",
    "PoseType",
    "QwenReviewError",
    "QwenReviewResult",
    "QwenReviewer",
    "ReviewedFinding",
    "normalize_reference_image",
]
