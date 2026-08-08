"""Training-free shortage detection based on before/after image comparison."""

from .detector import (
    AlignmentInfo,
    ComparisonConfig,
    ComparisonResult,
    ShortageDetector,
    ShortageRegion,
    detect_shortage,
)

__all__ = [
    "AlignmentInfo",
    "ComparisonConfig",
    "ComparisonResult",
    "ShortageDetector",
    "ShortageRegion",
    "detect_shortage",
]

