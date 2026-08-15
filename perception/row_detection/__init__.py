"""Shared shelf-rail and product-row detection for perception services."""

from .detector import (
    RowDetectionConfig,
    RowDetectionResult,
    PoseType,
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
    "PoseType",
    "ShelfRail",
    "ShelfRow",
    "ShelfRowMatch",
    "detect_rows",
    "read_image",
    "write_image",
]
