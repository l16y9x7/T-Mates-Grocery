"""Order-preserving shortage slot matching in source-image coordinates.

SAM candidates from two captures can have the same count while covering very
different horizontal spans. Treating those candidates as an ordinal one-to-one
match associates products from neighbouring groups. This module keeps the
normal common-translation path, but falls back to an original-image edge anchor
when the left and right endpoints cannot represent one camera translation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlotMatchResult:
    matches: dict[int, int]
    normalized_shift: float
    strategy: str
    baseline_span: float
    current_span: float
    left_endpoint_shift: float | None
    right_endpoint_shift: float | None


def _edge_alignment(
    baseline_u: list[float],
    current_u: list[float],
    *,
    shift: float,
    normalized_pitch: float,
    anchor: str,
) -> tuple[dict[int, int], float]:
    """Match from one source-image edge while allowing extras and missing slots."""

    gate = max(0.55 * normalized_pitch, 0.035)
    matches: dict[int, int] = {}
    residual = 0.0
    if anchor == "left":
        baseline_index = 0
        current_index = 0
        while baseline_index < len(baseline_u) and current_index < len(current_u):
            predicted = baseline_u[baseline_index] + shift
            distance = current_u[current_index] - predicted
            if abs(distance) <= gate:
                matches[baseline_index] = current_index
                residual += abs(distance)
                baseline_index += 1
                current_index += 1
            elif current_u[current_index] < predicted:
                current_index += 1
            else:
                baseline_index += 1
    else:
        baseline_index = len(baseline_u) - 1
        current_index = len(current_u) - 1
        while baseline_index >= 0 and current_index >= 0:
            predicted = baseline_u[baseline_index] + shift
            distance = current_u[current_index] - predicted
            if abs(distance) <= gate:
                matches[baseline_index] = current_index
                residual += abs(distance)
                baseline_index -= 1
                current_index -= 1
            elif current_u[current_index] > predicted:
                current_index -= 1
            else:
                baseline_index -= 1
    return matches, residual


def _monotonic_alignment(
    baseline_u: list[float], current_u: list[float], shift: float
) -> tuple[dict[int, int], float]:
    baseline_is_shorter = len(baseline_u) <= len(current_u)
    short_values = baseline_u if baseline_is_shorter else current_u
    long_values = current_u if baseline_is_shorter else baseline_u
    short_count, long_count = len(short_values), len(long_values)
    costs = np.full((short_count + 1, long_count + 1), np.inf, dtype=np.float64)
    take = np.zeros((short_count + 1, long_count + 1), dtype=np.uint8)
    costs[0, :] = 0.0
    for short_index in range(1, short_count + 1):
        for long_index in range(1, long_count + 1):
            skip_cost = costs[short_index, long_index - 1]
            if baseline_is_shorter:
                base_value = short_values[short_index - 1]
                current_value = long_values[long_index - 1]
            else:
                base_value = long_values[long_index - 1]
                current_value = short_values[short_index - 1]
            match_cost = costs[short_index - 1, long_index - 1] + abs(
                base_value + shift - current_value
            )
            if np.isfinite(match_cost) and match_cost <= skip_cost:
                costs[short_index, long_index] = match_cost
                take[short_index, long_index] = 1
            else:
                costs[short_index, long_index] = skip_cost
    matches: dict[int, int] = {}
    short_index, long_index = short_count, long_count
    while short_index > 0 and long_index > 0:
        if take[short_index, long_index]:
            if baseline_is_shorter:
                matches[short_index - 1] = long_index - 1
            else:
                matches[long_index - 1] = short_index - 1
            short_index -= 1
            long_index -= 1
        else:
            long_index -= 1
    return matches, float(costs[short_count, long_count])


def _fixed_source_alignment(
    baseline_u: list[float],
    current_u: list[float],
    *,
    normalized_pitch: float,
) -> dict[int, int]:
    """Match in the unchanged source frame, allowing gaps on either side."""

    gate = max(0.55 * normalized_pitch, 0.035)
    baseline_count, current_count = len(baseline_u), len(current_u)
    # Each cell ranks paths by matched count first and residual second.
    matched = np.zeros((baseline_count + 1, current_count + 1), dtype=np.int32)
    residual = np.zeros((baseline_count + 1, current_count + 1), dtype=np.float64)
    action = np.zeros((baseline_count + 1, current_count + 1), dtype=np.uint8)
    action[1:, 0] = 1  # skip baseline slot
    action[0, 1:] = 2  # skip current candidate

    def better(
        candidate_count: int,
        candidate_residual: float,
        best_count: int,
        best_residual: float,
    ) -> bool:
        return candidate_count > best_count or (
            candidate_count == best_count and candidate_residual < best_residual
        )

    for baseline_index in range(1, baseline_count + 1):
        for current_index in range(1, current_count + 1):
            best_count = int(matched[baseline_index - 1, current_index])
            best_residual = float(residual[baseline_index - 1, current_index])
            best_action = 1
            skip_current_count = int(matched[baseline_index, current_index - 1])
            skip_current_residual = float(residual[baseline_index, current_index - 1])
            if better(
                skip_current_count,
                skip_current_residual,
                best_count,
                best_residual,
            ):
                best_count = skip_current_count
                best_residual = skip_current_residual
                best_action = 2
            distance = abs(
                baseline_u[baseline_index - 1] - current_u[current_index - 1]
            )
            if distance <= gate:
                match_count = int(matched[baseline_index - 1, current_index - 1]) + 1
                match_residual = (
                    float(residual[baseline_index - 1, current_index - 1]) + distance
                )
                if better(match_count, match_residual, best_count, best_residual):
                    best_count = match_count
                    best_residual = match_residual
                    best_action = 3
            matched[baseline_index, current_index] = best_count
            residual[baseline_index, current_index] = best_residual
            action[baseline_index, current_index] = best_action

    matches: dict[int, int] = {}
    baseline_index, current_index = baseline_count, current_count
    while baseline_index > 0 or current_index > 0:
        selected_action = action[baseline_index, current_index]
        if selected_action == 3:
            matches[baseline_index - 1] = current_index - 1
            baseline_index -= 1
            current_index -= 1
        elif selected_action == 2 and current_index > 0:
            current_index -= 1
        elif baseline_index > 0:
            baseline_index -= 1
        else:
            break
    return matches


def match_normalized_slots(
    baseline_u: list[float],
    current_u: list[float],
    *,
    normalized_pitch: float,
) -> SlotMatchResult:
    """Match sorted centers expressed as normalized full-image x coordinates."""

    baseline_span = baseline_u[-1] - baseline_u[0] if len(baseline_u) >= 2 else 0.0
    current_span = current_u[-1] - current_u[0] if len(current_u) >= 2 else 0.0
    left_shift = current_u[0] - baseline_u[0] if baseline_u and current_u else None
    right_shift = current_u[-1] - baseline_u[-1] if baseline_u and current_u else None
    if not baseline_u or not current_u:
        return SlotMatchResult(
            {}, 0.0, "empty_sequence", baseline_span, current_span, left_shift, right_shift
        )

    # A genuine view translation moves both endpoints by roughly the same
    # amount. A disagreement close to a product pitch indicates that SAM has
    # pulled candidates from outside this configured product group.
    endpoint_disagreement = abs(float(left_shift) - float(right_shift))
    distribution_threshold = max(0.65 * normalized_pitch, 0.05)
    distribution_mismatch = (
        len(baseline_u) >= 2
        and len(current_u) >= 2
        and endpoint_disagreement > distribution_threshold
    )
    midpoint_shift = (float(left_shift) + float(right_shift)) / 2.0
    no_systematic_shift = (
        float(left_shift) * float(right_shift) <= 0.0
        or abs(midpoint_shift) <= max(0.30 * normalized_pitch, 0.025)
    )
    if distribution_mismatch and no_systematic_shift:
        return SlotMatchResult(
            _fixed_source_alignment(
                baseline_u,
                current_u,
                normalized_pitch=normalized_pitch,
            ),
            0.0,
            "source_image_endpoint_bounds",
            baseline_span,
            current_span,
            left_shift,
            right_shift,
        )

    if distribution_mismatch:
        candidates: list[tuple[tuple[int, float, float], dict[int, int], float, str]] = []
        for anchor, shift in (("left", float(left_shift)), ("right", float(right_shift))):
            matches, residual = _edge_alignment(
                baseline_u,
                current_u,
                shift=shift,
                normalized_pitch=normalized_pitch,
                anchor=anchor,
            )
            rank = (-len(matches), residual, abs(shift))
            candidates.append((rank, matches, shift, f"source_edge_anchor_{anchor}"))
        _, matches, shift, strategy = min(candidates, key=lambda item: item[0])
        return SlotMatchResult(
            matches, shift, strategy, baseline_span, current_span, left_shift, right_shift
        )

    if len(baseline_u) == len(current_u):
        shift = float(
            np.median(
                np.asarray(current_u, dtype=np.float32)
                - np.asarray(baseline_u, dtype=np.float32)
            )
        )
        return SlotMatchResult(
            {index: index for index in range(len(baseline_u))},
            shift,
            "ordinal_left_to_right",
            baseline_span,
            current_span,
            left_shift,
            right_shift,
        )

    shift_candidates = {0.0}
    shift_candidates.update(
        round(current_value - baseline_value, 6)
        for baseline_value in baseline_u
        for current_value in current_u
    )
    best_matches: dict[int, int] = {}
    best_shift = 0.0
    best_rank: tuple[float, float] | None = None
    for shift in shift_candidates:
        matches, total_distance = _monotonic_alignment(baseline_u, current_u, shift)
        rank = (total_distance, abs(shift))
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_matches = matches
            best_shift = shift
    if best_matches:
        best_shift = float(
            np.median(
                np.asarray(
                    [
                        current_u[current_index] - baseline_u[baseline_index]
                        for baseline_index, current_index in best_matches.items()
                    ],
                    dtype=np.float32,
                )
            )
        )
    return SlotMatchResult(
        best_matches,
        best_shift,
        "monotonic_sequence_alignment",
        baseline_span,
        current_span,
        left_shift,
        right_shift,
    )
