"""Qwen semantic review for comparison-based shelf findings."""

from .reviewer import (
    CandidateContactSheet,
    CandidateProduct,
    DEFAULT_DEBUG_ROOT,
    PoseType,
    QwenReviewError,
    QwenReviewResult,
    QwenReviewer,
    ReviewRowConstraint,
    ReviewedFinding,
    build_candidate_contact_sheets,
    normalize_reference_image,
)

__all__ = [
    "CandidateContactSheet",
    "CandidateProduct",
    "DEFAULT_DEBUG_ROOT",
    "PoseType",
    "QwenReviewError",
    "QwenReviewResult",
    "QwenReviewer",
    "ReviewRowConstraint",
    "ReviewedFinding",
    "build_candidate_contact_sheets",
    "normalize_reference_image",
]
