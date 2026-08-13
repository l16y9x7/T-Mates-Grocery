"""Detect horizontal shelf rails and the product rows above them."""

from .detector import (
    RowDetectionConfig,
    RowDetectionResult,
    ShelfRail,
    ShelfRow,
    ShelfRowMatch,
    detect_rows,
    read_image,
    write_image,
)

__all__ = [
    "RowDetectionConfig",
    "RowDetectionResult",
    "ShelfRail",
    "ShelfRow",
    "ShelfRowMatch",
    "detect_rows",
    "read_image",
    "write_image",
]
