"""Robust group-relative depth outlier selection for shortage detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DepthOutlierSelection:
    indices: tuple[int, ...]
    median_mm: float | None
    mad_mm: float | None
    cutoff_mm: float
    robust_cutoff_mm: float | None
    gap_cutoff_mm: float | None
    hard_threshold_mm: float


def select_positive_depth_outliers(
    depth_deltas_mm: Sequence[float | None],
    *,
    absolute_threshold_mm: float = 40.0,
    hard_threshold_mm: float = 100.0,
    max_outliers: int = 2,
    mad_multiplier: float = 3.0,
    minimum_robust_margin_mm: float = 12.0,
    minimum_cluster_gap_mm: float = 25.0,
) -> DepthOutlierSelection:
    """Select at most ``max_outliers`` unusually large positive depth deltas.

    The robust median/MAD cutoff rejects common camera/view drift.  A large
    adjacent gap may also isolate a tail of one or two shortage candidates,
    including short groups where median/MAD alone is easily contaminated.
    """

    indexed = [
        (index, float(value))
        for index, value in enumerate(depth_deltas_mm)
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if not indexed:
        return DepthOutlierSelection(
            indices=(),
            median_mm=None,
            mad_mm=None,
            cutoff_mm=float(absolute_threshold_mm),
            robust_cutoff_mm=None,
            gap_cutoff_mm=None,
            hard_threshold_mm=float(hard_threshold_mm),
        )

    values = np.asarray([value for _, value in indexed], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if len(indexed) == 1:
        # A single value has no group-relative evidence.  Only the independent
        # hard threshold may classify it as a shortage.
        robust_cutoff = float(hard_threshold_mm)
    else:
        robust_cutoff = max(
            float(absolute_threshold_mm),
            median
            + max(
                float(minimum_robust_margin_mm),
                float(mad_multiplier) * mad,
            ),
        )

    ordered = sorted(indexed, key=lambda item: item[1])
    best_gap: tuple[float, float] | None = None
    for split_index in range(1, len(ordered)):
        upper_count = len(ordered) - split_index
        if upper_count > max(1, int(max_outliers)):
            continue
        lower_value = ordered[split_index - 1][1]
        upper_value = ordered[split_index][1]
        gap = upper_value - lower_value
        if (
            gap >= float(minimum_cluster_gap_mm)
            and upper_value > float(absolute_threshold_mm)
            and (best_gap is None or gap > best_gap[0])
        ):
            best_gap = (gap, (lower_value + upper_value) / 2.0)

    gap_cutoff = best_gap[1] if best_gap is not None else None
    cutoff = (
        min(robust_cutoff, gap_cutoff)
        if gap_cutoff is not None
        else robust_cutoff
    )
    candidates = [
        (index, value)
        for index, value in indexed
        if (
            value >= float(hard_threshold_mm)
            or (
                value > float(absolute_threshold_mm)
                and value > cutoff
            )
        )
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    selected = tuple(
        sorted(index for index, _ in candidates[: max(1, int(max_outliers))])
    )
    return DepthOutlierSelection(
        indices=selected,
        median_mm=median,
        mad_mm=mad,
        cutoff_mm=float(cutoff),
        robust_cutoff_mm=float(robust_cutoff),
        gap_cutoff_mm=(float(gap_cutoff) if gap_cutoff is not None else None),
        hard_threshold_mm=float(hard_threshold_mm),
    )
