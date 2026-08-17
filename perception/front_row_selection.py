"""Select all front-row SAM instances using local RGB-aligned depth evidence."""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _adaptive_inner_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    x1, y1, x2, y2 = _bbox_from_mask(mask)
    short_side = min(x2 - x1, y2 - y1)
    if short_side < 12:
        return mask.copy(), 1
    kernel_size = int(round(short_side * 0.03))
    kernel_size = max(3, min(9, kernel_size | 1))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    inner = cv2.erode(mask.astype(np.uint8) * 255, kernel) > 0
    original_pixels = int(np.count_nonzero(mask))
    if np.count_nonzero(inner) < max(20, round(original_pixels * 0.30)):
        return mask.copy(), 1
    return inner, kernel_size


def _stable_near_depth(values: np.ndarray, *, minimum_pixels: int) -> dict[str, Any]:
    z = np.asarray(values, dtype=np.float32)
    z = z[np.isfinite(z) & (z > 0)]
    result: dict[str, Any] = {
        "reliable": False,
        "depth_mm": None,
        "mad_mm": None,
        "support_pixels": 0,
        "support_ratio": 0.0,
        "valid_pixels": int(z.size),
    }
    if z.size < minimum_pixels:
        return result

    low, high = np.percentile(z, (1, 99))
    trimmed = z[(z >= low) & (z <= high)]
    if trimmed.size < minimum_pixels:
        return result

    bin_width = 20.0
    first_bin = float(np.floor(trimmed.min() / bin_width) * bin_width)
    last_bin = float(np.ceil(trimmed.max() / bin_width) * bin_width + bin_width)
    if last_bin - first_bin > 8000:
        low, high = np.percentile(trimmed, (5, 95))
        trimmed = trimmed[(trimmed >= low) & (trimmed <= high)]
        first_bin = float(np.floor(trimmed.min() / bin_width) * bin_width)
        last_bin = float(np.ceil(trimmed.max() / bin_width) * bin_width + bin_width)
    edges = np.arange(first_bin, last_bin + bin_width, bin_width, dtype=np.float32)
    if edges.size < 2:
        return result
    counts, _ = np.histogram(trimmed, bins=edges)
    smooth = np.convolve(
        np.pad(counts, (1, 1)),
        np.array([1, 2, 1], dtype=np.int32),
        mode="valid",
    )
    peak_floor = max(
        minimum_pixels,
        round(trimmed.size * 0.04),
        round(float(smooth.max()) * 0.18),
    )
    candidates = np.flatnonzero(smooth >= peak_floor)
    if candidates.size == 0:
        return result

    # Prefer the closest stable return. This avoids transparent packages being
    # represented by the much denser shelf/background return behind them.
    peak_index = int(candidates[0])
    center = float((edges[peak_index] + edges[peak_index + 1]) * 0.5)
    cluster = trimmed[np.abs(trimmed - center) <= 40.0]
    if cluster.size < minimum_pixels:
        return result
    median = float(np.median(cluster))
    mad = float(np.median(np.abs(cluster - median)))
    radius = max(25.0, 3.0 * 1.4826 * mad)
    refined = trimmed[np.abs(trimmed - median) <= radius]
    if refined.size < minimum_pixels:
        return result
    median = float(np.median(refined))
    mad = float(np.median(np.abs(refined - median)))
    support_ratio = float(refined.size / max(1, z.size))
    result.update(
        {
            "reliable": support_ratio >= 0.035,
            "depth_mm": round(median, 2),
            "mad_mm": round(mad, 2),
            "support_pixels": int(refined.size),
            "support_ratio": round(support_ratio, 6),
        }
    )
    return result


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return intersection / max(1, union)


def _overlap_ratio(first: tuple[int, int], second: tuple[int, int]) -> float:
    overlap = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    return overlap / max(1, min(first[1] - first[0], second[1] - second[0]))


def _apply_expected_count(
    instances: list[dict[str, Any]],
    *,
    scores: Sequence[float | None],
    expected_count: int,
    image_shape: tuple[int, int],
    horizontal_roi: tuple[int, int] | None,
    prefer_global_depth_layer: bool,
    prefer_regular_columns: bool,
    prefer_vertical_position_anomaly: bool,
    max_same_prompt_depth_spread_mm: float | None,
    enforce_expected_count: bool,
) -> dict[str, Any]:
    """Select front columns while preferring graph-front, shelf-aligned masks."""

    image_height, image_width = image_shape
    roi_left, roi_right = horizontal_roi or (0, image_width)
    candidates: list[int] = []
    for index, instance in enumerate(instances):
        if instance["duplicate_of"] is not None or not instance["depth_estimate"]["reliable"]:
            continue
        x1, _, x2, _ = instance["bbox_xyxy"]
        center_x = (x1 + x2) / 2.0
        if not roi_left <= center_x < roi_right:
            instance["selected"] = False
            instance["selection_reason"] = "outside_view_shelf_roi"
            continue
        candidates.append(index)
    if not candidates:
        return {
            "expected": expected_count,
            "selected": 0,
            "satisfied": False,
            "available_candidates": 0,
            "horizontal_roi": [roi_left, roi_right],
        }

    candidate_centers = np.asarray(
        [
            (
                instances[index]["bbox_xyxy"][0]
                + instances[index]["bbox_xyxy"][2]
            )
            / 2.0
            for index in candidates
        ],
        dtype=np.float32,
    )
    image_slot_width = image_width / max(1, expected_count)
    if expected_count > 1 and candidate_centers.size > 1:
        observed_slot_width = float(
            (candidate_centers.max() - candidate_centers.min())
            / (expected_count - 1)
        )
        # A prompt group may occupy only a small part of the shelf row.  Using
        # the full image width then forces legitimate neighboring products too
        # far apart and can make a detached SAM fragment win instead.
        nominal_slot_width = min(image_slot_width, max(1.0, observed_slot_width))
    else:
        nominal_slot_width = image_slot_width
    candidate_widths = np.asarray(
        [
            instances[index]["bbox_xyxy"][2]
            - instances[index]["bbox_xyxy"][0]
            for index in candidates
        ],
        dtype=np.float32,
    )
    # A single class prompt often returns both complete front objects and thin
    # visible fragments of the same product behind them.  Estimate the normal
    # class width from the upper-middle of plausible masks; very large merged
    # masks are excluded using the expected column spacing.
    plausible_widths = candidate_widths[
        candidate_widths <= max(12.0, nominal_slot_width * 1.25)
    ]
    if plausible_widths.size == 0:
        plausible_widths = candidate_widths
    # SAM commonly returns more thin rear fragments than complete foreground
    # objects.  The 65th percentile can therefore still describe a fragment
    # (for example 52 px versus 80--98 px complete cans).  Use the upper
    # quartile as the class-width reference; genuinely narrower products are
    # handled by the soft geometry score and the deliberately loose 0.55 hard
    # completeness floor below.
    typical_width = float(np.percentile(plausible_widths, 75))

    global_depth_layer: dict[str, Any] = {
        "requested": prefer_global_depth_layer,
        "enabled": False,
        "reason": "not_requested",
    }
    selection_candidates = candidates
    if prefer_global_depth_layer:
        complete_graph_front = []
        for index in candidates:
            instance = instances[index]
            x1, _, x2, _ = instance["bbox_xyxy"]
            width_ratio = (x2 - x1) / max(1.0, typical_width)
            if not instance["incoming"] and width_ratio >= 0.65:
                complete_graph_front.append(index)
        depth_order = sorted(
            complete_graph_front,
            key=lambda index: float(instances[index]["depth_estimate"]["depth_mm"]),
        )
        global_depth_layer.update(
            {
                "reason": "insufficient_complete_candidates",
                "complete_graph_front_candidates": len(depth_order),
            }
        )
        if len(depth_order) > expected_count:
            depths = np.asarray(
                [
                    float(instances[index]["depth_estimate"]["depth_mm"])
                    for index in depth_order
                ],
                dtype=np.float32,
            )
            split_position = expected_count
            front_depth_max = float(depths[split_position - 1])
            back_depth_min = float(depths[split_position])
            split_gap = back_depth_min - front_depth_max
            other_gaps = np.delete(np.diff(depths), split_position - 1)
            normal_gap = float(np.median(other_gaps)) if other_gaps.size else 0.0
            split_mads = np.asarray(
                [
                    float(
                        instances[index]["depth_estimate"]["mad_mm"] or 0.0
                    )
                    for index in depth_order[: split_position + 1]
                ],
                dtype=np.float32,
            )
            normal_mad = float(np.median(split_mads)) if split_mads.size else 0.0
            split_threshold = max(
                3.0,
                3.0 * 1.4826 * normal_mad,
                2.5 * normal_gap,
                0.015 * float(np.median(depths[:split_position])),
            )
            cutoff = (front_depth_max + back_depth_min) / 2.0
            near_candidates = [
                index
                for index in candidates
                if float(instances[index]["depth_estimate"]["depth_mm"]) <= cutoff
            ]
            global_depth_layer.update(
                {
                    "reason": "gap_not_significant",
                    "front_depth_max_mm": round(front_depth_max, 2),
                    "back_depth_min_mm": round(back_depth_min, 2),
                    "split_gap_mm": round(split_gap, 2),
                    "split_threshold_mm": round(split_threshold, 2),
                    "cutoff_mm": round(cutoff, 2),
                    "near_candidates": len(near_candidates),
                }
            )
            if split_gap > split_threshold and len(near_candidates) >= expected_count:
                selection_candidates = near_candidates
                global_depth_layer["enabled"] = True
                global_depth_layer["reason"] = "significant_n_to_n_plus_1_gap"

    # For regularly arranged liquids, a complete front-row instance should
    # terminate on the same shelf/rail line as its neighbours.  SAM frequently
    # returns caps, labels and visible rear fragments whose depth is plausible,
    # but whose lower edge is far above that line.  Estimate the line from the
    # lower envelope of complete-looking masks, then reject those fragments
    # before the exact-count/column-spacing optimizer sees them.  If an expected
    # column has no aligned mask, returning fewer than expected is intentional:
    # that column is a shortage, not an invitation to promote a rear fragment.
    bottom_line_prior: dict[str, Any] = {
        "requested": prefer_regular_columns,
        "enabled": False,
        "reason": "not_requested",
    }
    bottom_line_rejected: set[int] = set()
    geometry_rejected: set[int] = set()
    typical_height = 0.0
    if prefer_regular_columns:
        complete_width_candidates = []
        for index in selection_candidates:
            x1, y1, x2, y2 = instances[index]["bbox_xyxy"]
            if (x2 - x1) >= max(8.0, 0.55 * typical_width):
                complete_width_candidates.append(index)
        bottom_line_prior["reason"] = "insufficient_complete_masks"
        if len(complete_width_candidates) >= 2:
            bottoms = np.asarray(
                [instances[index]["bbox_xyxy"][3] for index in complete_width_candidates],
                dtype=np.float32,
            )
            # The rear row can contain more SAM instances than the true front
            # row.  A 70th-percentile lower edge then lands between the two and
            # fits the numerically dominant rear layer.  The physical front row
            # is the stable bottom-most cluster, so seed the fit from the 90th
            # percentile while retaining a tolerance for mild rail slope.
            lower_envelope = float(np.percentile(bottoms, 90))
            fit_tolerance = max(6.0, min(18.0, image_height * 0.08))
            fit_indices = [
                index
                for index in complete_width_candidates
                if instances[index]["bbox_xyxy"][3]
                >= lower_envelope - fit_tolerance
            ]
            if len(fit_indices) >= 2:
                fit_x = np.asarray(
                    [
                        (
                            instances[index]["bbox_xyxy"][0]
                            + instances[index]["bbox_xyxy"][2]
                        )
                        / 2.0
                        for index in fit_indices
                    ],
                    dtype=np.float32,
                )
                fit_y = np.asarray(
                    [instances[index]["bbox_xyxy"][3] for index in fit_indices],
                    dtype=np.float32,
                )
                if len(fit_indices) >= 3 and float(np.ptp(fit_x)) > 1.0:
                    slope, intercept = np.polyfit(fit_x, fit_y, 1)
                    # Row crops are already rail-aligned.  A large fitted slope
                    # almost always means that an outlier influenced the fit.
                    slope = float(np.clip(slope, -0.06, 0.06))
                    intercept = float(np.median(fit_y - slope * fit_x))
                else:
                    slope = 0.0
                    intercept = float(np.median(fit_y))

                aligned_candidates: list[int] = []
                aligned_heights: list[float] = []
                for index in selection_candidates:
                    x1, y1, x2, y2 = instances[index]["bbox_xyxy"]
                    center_x = (x1 + x2) / 2.0
                    predicted_bottom = slope * center_x + intercept
                    residual = abs(float(y2) - predicted_bottom)
                    if residual <= fit_tolerance:
                        aligned_candidates.append(index)
                        aligned_heights.append(float(y2 - y1))
                    else:
                        bottom_line_rejected.add(index)

                typical_height = (
                    float(np.median(np.asarray(aligned_heights, dtype=np.float32)))
                    if aligned_heights
                    else 0.0
                )
                geometry_candidates: list[int] = []
                for index in aligned_candidates:
                    x1, y1, x2, y2 = instances[index]["bbox_xyxy"]
                    width_ratio = (x2 - x1) / max(1.0, typical_width)
                    height_ratio = (y2 - y1) / max(1.0, typical_height)
                    if width_ratio < 0.55 or height_ratio < 0.58:
                        geometry_rejected.add(index)
                    else:
                        geometry_candidates.append(index)

                selection_candidates = geometry_candidates
                bottom_line_prior.update(
                    {
                        "enabled": True,
                        "reason": "front_bottom_line_fitted",
                        "slope": round(slope, 6),
                        "intercept_px": round(intercept, 2),
                        "tolerance_px": round(fit_tolerance, 2),
                        "fit_instances": [index + 1 for index in fit_indices],
                        "aligned_instances": [
                            index + 1 for index in aligned_candidates
                        ],
                        "bottom_rejected_instances": [
                            index + 1 for index in sorted(bottom_line_rejected)
                        ],
                        "geometry_rejected_instances": [
                            index + 1 for index in sorted(geometry_rejected)
                        ],
                        "typical_height_px": round(typical_height, 2),
                    }
                )

    selection_pool = set(selection_candidates)
    same_prompt_depth_rejected: set[int] = set()
    same_prompt_depth_band: dict[str, Any] = {
        "requested": max_same_prompt_depth_spread_mm is not None,
        "enabled": False,
        "reason": "not_requested",
        "max_spread_mm": max_same_prompt_depth_spread_mm,
    }

    def effective_incoming(index: int) -> list[int]:
        """Occluders that survived the same bottom/geometry filtering."""

        return [
            incoming
            for incoming in instances[index]["incoming"]
            if incoming - 1 in selection_pool
        ]

    def quality(index: int) -> float:
        instance = instances[index]
        x1, y1, x2, y2 = instance["bbox_xyxy"]
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        fill = min(1.0, instance["mask_pixels"] / bbox_area)
        height_score = min(1.0, (y2 - y1) / max(1.0, image_height * 0.45))
        width_completeness = min(1.0, (x2 - x1) / max(1.0, typical_width))
        geometry_penalty = 0.0
        if prefer_regular_columns and typical_height > 0:
            width_ratio = (x2 - x1) / max(1.0, typical_width)
            height_ratio = (y2 - y1) / max(1.0, typical_height)
            # Within a configured liquid group the projected masks should be
            # comparable to their neighbours.  Use a smooth log-ratio penalty
            # instead of a brittle hard size threshold so perspective-induced
            # size changes remain possible while adjacent-shelf products lose.
            geometry_penalty = (
                2.5 * abs(float(np.log(max(0.05, width_ratio))))
                + 1.2 * abs(float(np.log(max(0.05, height_ratio))))
            )
        sam_score = float(scores[index] or 0.0)
        support = float(instance["depth_estimate"]["support_ratio"])
        incoming = effective_incoming(index)
        graph_score = 2.5 if not incoming else -0.35 * len(incoming)
        if horizontal_roi is not None:
            # The caller already removed candidates whose centers fall outside
            # the view-specific shelf ROI.  Penalizing the full-image edge as
            # well can suppress a legitimate first product that straddles the
            # ROI boundary and promote a rear fragment beside it.
            edge_penalty = 0.0
        else:
            edge_clearance = min(x1, image_width - x2)
            touches_frame = x1 <= 2 or x2 >= image_width - 2
            incomplete_at_edge = width_completeness < 0.65
            if edge_clearance <= image_width * 0.02 and (
                touches_frame or incomplete_at_edge
            ):
                edge_penalty = 2.5
            elif (
                edge_clearance <= image_width * 0.05
                and incomplete_at_edge
            ):
                edge_penalty = 0.8
            else:
                edge_penalty = 0.0
        return (
            graph_score
            + sam_score
            + 0.55 * support
            + 0.45 * fill
            + 0.50 * height_score
            + 0.75 * width_completeness
            - geometry_penalty
            - edge_penalty
        )

    if max_same_prompt_depth_spread_mm is not None and selection_candidates:
        depth_limit = float(max_same_prompt_depth_spread_mm)
        candidate_mads = np.asarray(
            [
                float(instances[index]["depth_estimate"].get("mad_mm") or 0.0)
                for index in selection_candidates
            ],
            dtype=np.float32,
        )
        # The configured spread describes the physical layer.  Allow a small,
        # data-driven margin for RGB/depth edge noise so two genuine foreground
        # masks at (for example) 719 mm and 751 mm are not split solely because
        # their robust depth estimates straddle the nominal 30 mm boundary.
        depth_noise_allowance = min(
            15.0,
            float(np.median(candidate_mads)) if candidate_mads.size else 0.0,
        )
        effective_depth_limit = depth_limit + depth_noise_allowance
        depth_order = sorted(
            selection_candidates,
            key=lambda index: float(instances[index]["depth_estimate"]["depth_mm"]),
        )
        best_window: list[int] = []
        best_rank: tuple[float, int, float, float] | None = None
        for left_position, left_index in enumerate(depth_order):
            left_depth = float(instances[left_index]["depth_estimate"]["depth_mm"])
            window: list[int] = []
            for index in depth_order[left_position:]:
                depth_value = float(instances[index]["depth_estimate"]["depth_mm"])
                if depth_value - left_depth > effective_depth_limit:
                    break
                window.append(index)
            if not window:
                continue
            ranked_quality = sorted(
                (quality(index) for index in window), reverse=True
            )[:expected_count]
            window_depths = [
                float(instances[index]["depth_estimate"]["depth_mm"])
                for index in window
            ]
            spread = max(window_depths) - min(window_depths)
            median_depth = float(
                np.median(np.asarray(window_depths, dtype=np.float32))
            )
            # Depth is the primary front/back signal.  Do not prefer a farther
            # cluster merely because it can provide ``expected_count`` masks:
            # expected_count is the number of physical slots, while the current
            # image may legitimately contain fewer masks because a slot is
            # empty.  Quality and support only break ties within the near layer.
            rank = (
                -median_depth,
                min(len(window), expected_count),
                float(sum(ranked_quality)),
                -spread,
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_window = window

        if best_window:
            previous_candidates = set(selection_candidates)
            selection_candidates = best_window
            selection_pool = set(selection_candidates)
            same_prompt_depth_rejected = previous_candidates - selection_pool
            selected_depths = [
                float(instances[index]["depth_estimate"]["depth_mm"])
                for index in selection_candidates
            ]
            same_prompt_depth_band.update(
                {
                    "enabled": True,
                    "reason": "nearest_same_prompt_depth_band",
                    "depth_noise_allowance_mm": round(depth_noise_allowance, 2),
                    "effective_max_spread_mm": round(effective_depth_limit, 2),
                    "depth_min_mm": round(min(selected_depths), 2),
                    "depth_max_mm": round(max(selected_depths), 2),
                    "depth_spread_mm": round(
                        max(selected_depths) - min(selected_depths), 2
                    ),
                    "band_instances": [
                        index + 1 for index in sorted(selection_candidates)
                    ],
                    "rejected_instances": [
                        index + 1 for index in sorted(same_prompt_depth_rejected)
                    ],
                }
            )

    ordered = sorted(
        selection_candidates,
        key=lambda index: (
            (instances[index]["bbox_xyxy"][0] + instances[index]["bbox_xyxy"][2]) / 2,
            -quality(index),
        ),
    )
    centers = {
        index: (
            instances[index]["bbox_xyxy"][0] + instances[index]["bbox_xyxy"][2]
        )
        / 2.0
        for index in ordered
    }

    def solve(
        minimum_gap: float,
        *,
        target_pitch: float | None = None,
    ) -> list[int]:
        count = min(expected_count, len(ordered))
        # dp[(amount, end_position)] = (score, selected candidate indices)
        dp: dict[tuple[int, int], tuple[float, list[int]]] = {}
        for position, index in enumerate(ordered):
            dp[(1, position)] = (quality(index), [index])
        for amount in range(2, count + 1):
            for position, index in enumerate(ordered):
                best: tuple[float, list[int]] | None = None
                for previous_position in range(position):
                    previous = dp.get((amount - 1, previous_position))
                    previous_index = ordered[previous_position]
                    if previous is None or centers[index] - centers[previous_index] < minimum_gap:
                        continue
                    spacing_penalty = 0.0
                    if target_pitch is not None and target_pitch > 0:
                        actual_gap = centers[index] - centers[previous_index]
                        relative_error = abs(actual_gap - target_pitch) / target_pitch
                        spacing_penalty = 1.75 * min(2.0, relative_error**2)
                    candidate = (
                        previous[0] + quality(index) - spacing_penalty,
                        previous[1] + [index],
                    )
                    if best is None or candidate[0] > best[0]:
                        best = candidate
                if best is not None:
                    dp[(amount, position)] = best
        choices = [value for (amount, _), value in dp.items() if amount == count]
        return max(choices, key=lambda value: value[0])[1] if choices else []

    selected: list[int] = []
    for gap_ratio in (0.28, 0.18, 0.08, 0.0):
        selected = solve(nominal_slot_width * gap_ratio)
        if len(selected) == min(expected_count, len(ordered)):
            break
    regular_column_prior: dict[str, Any] = {
        "requested": prefer_regular_columns,
        "enabled": False,
        "reason": "not_requested",
    }
    if prefer_regular_columns:
        regular_column_prior["reason"] = "insufficient_selected_columns"
        if len(selected) >= 3:
            selected_centers = sorted(centers[index] for index in selected)
            selected_gaps = np.diff(np.asarray(selected_centers, dtype=np.float32))
            target_pitch = float(np.median(selected_gaps))
            regular_column_prior.update(
                {
                    "reason": "regularized",
                    "enabled": True,
                    "target_pitch_px": round(target_pitch, 2),
                    "initial_gaps_px": [round(float(gap), 2) for gap in selected_gaps],
                }
            )
            regularized = solve(
                nominal_slot_width * 0.08,
                target_pitch=target_pitch,
            )
            if len(regularized) == min(expected_count, len(ordered)):
                selected = regularized
            # A sequence of N masks that contains a roughly 2× pitch jump has
            # actually occupied N+1 shelf slots.  This is the common failure
            # mode where an adjacent/rear object is used to hide a real empty
            # column.  Trim the weaker edge until the selected run fits within
            # the configured slot count; the unfilled slot remains a shortage.
            def occupied_slots(indices: list[int]) -> tuple[int, list[int]]:
                ordered_indices = sorted(indices, key=lambda item: centers[item])
                missing_after: list[int] = []
                slot_total = 1 if ordered_indices else 0
                for position, (left, right) in enumerate(
                    zip(ordered_indices, ordered_indices[1:]), start=1
                ):
                    gap = centers[right] - centers[left]
                    jump = max(1, int(round(gap / max(1.0, target_pitch))))
                    if gap < target_pitch * 1.55:
                        jump = 1
                    slot_total += jump
                    if jump > 1:
                        missing_after.extend([position] * (jump - 1))
                return slot_total, missing_after

            slot_span, missing_after = occupied_slots(selected)
            while len(selected) > 1 and slot_span > expected_count:
                left_removed = selected[1:]
                right_removed = selected[:-1]
                left_span, _ = occupied_slots(left_removed)
                right_span, _ = occupied_slots(right_removed)
                left_rank = (
                    left_span <= expected_count,
                    -abs(expected_count - left_span),
                    sum(quality(index) for index in left_removed),
                )
                right_rank = (
                    right_span <= expected_count,
                    -abs(expected_count - right_span),
                    sum(quality(index) for index in right_removed),
                )
                selected = left_removed if left_rank >= right_rank else right_removed
                slot_span, missing_after = occupied_slots(selected)
            final_centers = sorted(centers[index] for index in selected)
            final_gaps = np.diff(np.asarray(final_centers, dtype=np.float32))
            missing_columns = max(0, expected_count - len(selected))
            regular_column_prior.update(
                {
                    "final_gaps_px": [round(float(gap), 2) for gap in final_gaps],
                    "missing_column_count": missing_columns,
                    "missing_after_selected_positions": missing_after,
                    "occupied_slot_span": slot_span,
                }
            )
        else:
            regular_column_prior["missing_column_count"] = max(
                0, expected_count - len(selected)
            )
    forced_promoted: set[int] = set()
    hard_count_constraint: dict[str, Any] = {
        "requested": enforce_expected_count,
        "available_candidates": len(candidates),
        "promoted_instances": [],
        "candidate_shortfall": 0,
    }
    if enforce_expected_count and len(selected) < expected_count:
        remaining = sorted(
            (index for index in candidates if index not in selected),
            key=lambda index: quality(index),
            reverse=True,
        )
        needed = expected_count - len(selected)
        forced_promoted = set(remaining[:needed])
        selected.extend(remaining[:needed])
        selection_pool.update(forced_promoted)
        for index in forced_promoted:
            x1, _, x2, _ = instances[index]["bbox_xyxy"]
            centers[index] = (x1 + x2) / 2.0
        hard_count_constraint["promoted_instances"] = [
            index + 1 for index in sorted(forced_promoted)
        ]
        hard_count_constraint["candidate_shortfall"] = max(
            0, expected_count - len(selected)
        )

    chosen = set(selected)
    depth_layer_pool = set(selection_candidates)
    for index in candidates:
        instance = instances[index]
        if index in chosen:
            instance["selected"] = True
            instance["selection_reason"] = (
                "forced_expected_count"
                if index in forced_promoted
                else "front_layer_expected_count"
            )
        else:
            instance["selected"] = False
            if index in bottom_line_rejected:
                instance["selection_reason"] = "off_front_bottom_line"
            elif index in geometry_rejected:
                instance["selection_reason"] = "incomplete_mask_geometry"
            elif index in same_prompt_depth_rejected:
                instance["selection_reason"] = "outside_same_prompt_depth_band"
            elif not effective_incoming(index):
                instance["selection_reason"] = (
                    "global_back_depth_layer"
                    if global_depth_layer["enabled"] and index not in depth_layer_pool
                    else "exceeds_expected_front_count"
                )

    suspected_missing_regions: list[dict[str, Any]] = []
    selected_left_to_right = sorted(selected, key=lambda index: centers[index])
    if same_prompt_depth_band.get("enabled"):
        band_depth_min = float(same_prompt_depth_band["depth_min_mm"])
        depth_limit = float(same_prompt_depth_band["max_spread_mm"])
        for index in sorted(same_prompt_depth_rejected):
            instance_depth = float(instances[index]["depth_estimate"]["depth_mm"])
            if (
                instance_depth <= band_depth_min + depth_limit
                or effective_incoming(index)
            ):
                continue
            suspected_missing_regions.append(
                {
                    "strategy": "same_prompt_depth_outlier",
                    "bbox_xyxy": [
                        round(float(value), 2) for value in instances[index]["bbox_xyxy"]
                    ],
                    "instance_index": index + 1,
                    "depth_mm": round(instance_depth, 2),
                    "front_band_min_mm": round(band_depth_min, 2),
                    "excess_depth_mm": round(instance_depth - band_depth_min, 2),
                    "threshold_mm": round(depth_limit, 2),
                }
            )
    if prefer_regular_columns and regular_column_prior.get("enabled"):
        target_pitch = float(regular_column_prior["target_pitch_px"])
        typical_height = float(bottom_line_prior.get("typical_height_px") or 0.0)
        slope = float(bottom_line_prior.get("slope") or 0.0)
        intercept = float(bottom_line_prior.get("intercept_px") or image_height)

        def append_regular_gap(
            center_x: float,
            *,
            strategy: str,
            details: dict[str, Any],
        ) -> None:
            if any(
                abs(
                    center_x
                    - (
                        float(region["bbox_xyxy"][0])
                        + float(region["bbox_xyxy"][2])
                    )
                    / 2.0
                )
                < target_pitch * 0.45
                for region in suspected_missing_regions
                if isinstance(region.get("bbox_xyxy"), list)
            ):
                return
            bottom_y = slope * center_x + intercept
            half_width = max(5.0, typical_width * 0.50)
            box_height = max(10.0, typical_height)
            bbox = [
                max(float(roi_left), center_x - half_width),
                max(0.0, bottom_y - box_height),
                min(float(roi_right), center_x + half_width),
                min(float(image_height), bottom_y),
            ]
            suspected_missing_regions.append(
                {
                    "strategy": strategy,
                    "bbox_xyxy": [round(value, 2) for value in bbox],
                    "center_x_px": round(center_x, 2),
                    "expected_pitch_px": round(target_pitch, 2),
                    **details,
                }
            )

        for left_index, right_index in zip(
            selected_left_to_right, selected_left_to_right[1:]
        ):
            left_center = centers[left_index]
            right_center = centers[right_index]
            gap = right_center - left_center
            if gap < target_pitch * 1.55:
                continue
            slot_jump = max(2, int(round(gap / max(1.0, target_pitch))))
            for offset in range(1, slot_jump):
                center_x = left_center + target_pitch * offset
                if center_x >= right_center - target_pitch * 0.45:
                    break
                append_regular_gap(
                    center_x,
                    strategy="regular_column_gap",
                    details={
                        "left_instance": left_index + 1,
                        "right_instance": right_index + 1,
                        "observed_gap_px": round(gap, 2),
                    },
                )

        # Internal gaps are unambiguous.  When all detected columns are
        # continuous but fewer than configured, cautiously extrapolate the two
        # ends.  Mark an edge only when one side has clearly more shelf room and
        # no complete bottom-aligned SAM mask already occupies the projected
        # slot.  Symmetric/ambiguous edge shortages are intentionally left
        # unlabelled.
        remaining_missing = max(
            0,
            expected_count
            - len(selected_left_to_right)
            - len(suspected_missing_regions),
        )
        if remaining_missing and selected_left_to_right:
            aligned_centers = [centers[index] for index in selection_candidates]
            current_left = centers[selected_left_to_right[0]]
            current_right = centers[selected_left_to_right[-1]]
            half_width = max(5.0, typical_width * 0.50)
            for _ in range(remaining_missing):
                left_center = current_left - target_pitch
                right_center = current_right + target_pitch

                def edge_is_free(center_x: float) -> bool:
                    if not (
                        center_x - half_width >= roi_left
                        and center_x + half_width <= roi_right
                    ):
                        return False
                    return all(
                        abs(center_x - existing_center) > target_pitch * 0.45
                        for existing_center in aligned_centers
                    )

                left_free = edge_is_free(left_center)
                right_free = edge_is_free(right_center)
                left_margin = current_left - roi_left
                right_margin = roi_right - current_right
                if left_free and not right_free:
                    side = "left"
                elif right_free and not left_free:
                    side = "right"
                elif left_free and right_free:
                    if abs(left_margin - right_margin) < target_pitch * 0.45:
                        break
                    side = "left" if left_margin > right_margin else "right"
                else:
                    break
                center_x = left_center if side == "left" else right_center
                append_regular_gap(
                    center_x,
                    strategy="regular_column_edge_gap",
                    details={
                        "edge": side,
                        "left_margin_px": round(left_margin, 2),
                        "right_margin_px": round(right_margin, 2),
                    },
                )
                aligned_centers.append(center_x)
                if side == "left":
                    current_left = center_x
                else:
                    current_right = center_x
    elif prefer_vertical_position_anomaly and len(selected_left_to_right) >= 3:
        selected_bottom_y = np.asarray(
            [
                float(instances[index]["bbox_xyxy"][3])
                for index in selected_left_to_right
            ],
            dtype=np.float32,
        )
        selected_heights = np.asarray(
            [
                float(
                    instances[index]["bbox_xyxy"][3]
                    - instances[index]["bbox_xyxy"][1]
                )
                for index in selected_left_to_right
            ],
            dtype=np.float32,
        )
        y_threshold = max(16.0, 0.18 * float(np.median(selected_heights)))
        for position in range(1, len(selected_left_to_right) - 1):
            left_y = float(selected_bottom_y[position - 1])
            current_y = float(selected_bottom_y[position])
            right_y = float(selected_bottom_y[position + 1])
            delta_left = current_y - left_y
            delta_right = current_y - right_y
            # Perspective produces a gradual monotonic Y trend.  A rear/upper
            # object is different: its bottom is an outlier in the same
            # direction relative to both immediate neighbours.
            same_direction = delta_left * delta_right > 0
            if not (
                same_direction
                and abs(delta_left) > y_threshold
                and abs(delta_right) > y_threshold
            ):
                continue
            index = selected_left_to_right[position]
            suspected_missing_regions.append(
                {
                    "strategy": "bilateral_y_anomaly",
                    "bbox_xyxy": [
                        round(float(value), 2) for value in instances[index]["bbox_xyxy"]
                    ],
                    "instance_index": index + 1,
                    "bottom_y_px": round(current_y, 2),
                    "left_bottom_y_px": round(left_y, 2),
                    "right_bottom_y_px": round(right_y, 2),
                    "left_delta_px": round(delta_left, 2),
                    "right_delta_px": round(delta_right, 2),
                    "threshold_px": round(y_threshold, 2),
                }
            )
    elif len(selected_left_to_right) >= 3:
        # For non-regular products, compensate for shelf perspective with a
        # robust depth-vs-x line.  Only positive residuals are meaningful here:
        # a substantially farther mask indicates that the front product may be
        # absent and a rear product/background is visible in its place.
        selected_x = np.asarray(
            [centers[index] for index in selected_left_to_right], dtype=np.float32
        )
        selected_depth = np.asarray(
            [
                float(instances[index]["depth_estimate"]["depth_mm"])
                for index in selected_left_to_right
            ],
            dtype=np.float32,
        )
        pair_slopes = []
        for left_position in range(len(selected_left_to_right)):
            for right_position in range(left_position + 1, len(selected_left_to_right)):
                delta_x = float(selected_x[right_position] - selected_x[left_position])
                if abs(delta_x) < max(8.0, typical_width * 0.35):
                    continue
                pair_slopes.append(
                    float(
                        (selected_depth[right_position] - selected_depth[left_position])
                        / delta_x
                    )
                )
        depth_slope = float(np.median(pair_slopes)) if pair_slopes else 0.0
        depth_intercept = float(np.median(selected_depth - depth_slope * selected_x))
        expected_depth = depth_slope * selected_x + depth_intercept
        depth_residual = selected_depth - expected_depth
        residual_center = float(np.median(depth_residual))
        residual_mad = float(np.median(np.abs(depth_residual - residual_center)))
        depth_threshold = max(
            60.0,
            3.5 * 1.4826 * residual_mad,
            0.035 * float(np.median(selected_depth)),
        )
        for position, index in enumerate(selected_left_to_right):
            residual = float(depth_residual[position] - residual_center)
            if residual <= depth_threshold:
                continue
            suspected_missing_regions.append(
                {
                    "strategy": "positive_depth_anomaly",
                    "bbox_xyxy": [
                        round(float(value), 2) for value in instances[index]["bbox_xyxy"]
                    ],
                    "instance_index": index + 1,
                    "depth_mm": round(float(selected_depth[position]), 2),
                    "expected_depth_mm": round(float(expected_depth[position]), 2),
                    "positive_residual_mm": round(residual, 2),
                    "threshold_mm": round(depth_threshold, 2),
                }
            )
    return {
        "expected": expected_count,
        "selected": len(selected),
        "satisfied": len(selected) == expected_count,
        "available_candidates": len(candidates),
        "selection_pool_candidates": len(selection_candidates),
        "horizontal_roi": [roi_left, roi_right],
        "typical_instance_width_px": round(typical_width, 2),
        "nominal_slot_width_px": round(nominal_slot_width, 2),
        "global_depth_layer": global_depth_layer,
        "bottom_line_prior": bottom_line_prior,
        "regular_column_prior": regular_column_prior,
        "same_prompt_depth_band": same_prompt_depth_band,
        "hard_count_constraint": hard_count_constraint,
        "missing_detection_strategy": (
            "regular_column_gap"
            if prefer_regular_columns
            else (
                "bilateral_y_anomaly"
                if prefer_vertical_position_anomaly
                else "positive_depth_anomaly"
            )
        ),
        "suspected_missing_regions": suspected_missing_regions,
    }


def select_front_row_instances(
    masks: Sequence[np.ndarray],
    depth_mm: np.ndarray,
    *,
    scores: Sequence[float | None] | None = None,
    expected_front_count: int | None = None,
    horizontal_roi: tuple[int, int] | None = None,
    prefer_global_depth_layer: bool = False,
    prefer_regular_columns: bool = False,
    prefer_vertical_position_anomaly: bool = False,
    max_same_prompt_depth_spread_mm: float | None = None,
    enforce_expected_count: bool = False,
) -> dict[str, Any]:
    """Return front instances, duplicate suppression and pairwise depth edges."""

    depth = np.asarray(depth_mm)
    normalized_masks = [np.asarray(mask) > 0 for mask in masks]
    if any(mask.shape != depth.shape for mask in normalized_masks):
        raise ValueError("mask and depth shapes must match")
    if expected_front_count is not None and expected_front_count <= 0:
        raise ValueError("expected_front_count must be positive")
    if (
        max_same_prompt_depth_spread_mm is not None
        and max_same_prompt_depth_spread_mm <= 0
    ):
        raise ValueError("max_same_prompt_depth_spread_mm must be positive")
    normalized_scores = list(scores or [None] * len(normalized_masks))
    if len(normalized_scores) != len(normalized_masks):
        raise ValueError("scores and masks must have the same length")
    instances: list[dict[str, Any]] = []
    inner_masks: list[np.ndarray] = []
    for index, mask in enumerate(normalized_masks):
        inner, erode_kernel = _adaptive_inner_mask(mask)
        valid_values = depth[inner & np.isfinite(depth) & (depth > 0)]
        estimate = _stable_near_depth(valid_values, minimum_pixels=20)
        bbox = _bbox_from_mask(mask)
        instances.append(
            {
                "instance_index": index + 1,
                "bbox_xyxy": list(bbox),
                "mask_pixels": int(np.count_nonzero(mask)),
                "inner_mask_pixels": int(np.count_nonzero(inner)),
                "erode_kernel_px": erode_kernel,
                "depth_estimate": estimate,
                "duplicate_of": None,
                "incoming": [],
                "outgoing": [],
                "selected": False,
                "selection_reason": "",
            }
        )
        inner_masks.append(inner)

    # SAM can return near-identical masks for the same object. Keep the higher
    # confidence result before building the occlusion graph.
    for first_index in range(len(instances)):
        if instances[first_index]["duplicate_of"] is not None:
            continue
        for second_index in range(first_index + 1, len(instances)):
            if instances[second_index]["duplicate_of"] is not None:
                continue
            if _mask_iou(normalized_masks[first_index], normalized_masks[second_index]) < 0.72:
                continue
            first_score = normalized_scores[first_index]
            second_score = normalized_scores[second_index]
            keep_first = float(first_score or 0.0) >= float(second_score or 0.0)
            duplicate_index = second_index if keep_first else first_index
            kept_index = first_index if keep_first else second_index
            instances[duplicate_index]["duplicate_of"] = kept_index + 1
            if not keep_first:
                break

    edges: list[dict[str, Any]] = []
    active = [
        index
        for index, instance in enumerate(instances)
        if instance["duplicate_of"] is None
        and instance["depth_estimate"]["reliable"]
    ]
    for position, first_index in enumerate(active):
        first_bbox = instances[first_index]["bbox_xyxy"]
        for second_index in active[position + 1 :]:
            second_bbox = instances[second_index]["bbox_xyxy"]
            horizontal_overlap = _overlap_ratio(
                (first_bbox[0], first_bbox[2]),
                (second_bbox[0], second_bbox[2]),
            )
            vertical_overlap = _overlap_ratio(
                (first_bbox[1], first_bbox[3]),
                (second_bbox[1], second_bbox[3]),
            )
            if horizontal_overlap < 0.15 or vertical_overlap < 0.25:
                continue

            short_side = min(
                first_bbox[2] - first_bbox[0],
                second_bbox[2] - second_bbox[0],
                first_bbox[3] - first_bbox[1],
                second_bbox[3] - second_bbox[1],
            )
            contact_radius = max(3, min(11, round(short_side * 0.06)))
            kernel_size = contact_radius * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            )
            # Never sample the intersection itself: overlapping SAM masks would
            # otherwise read the same foreground pixels on both sides and hide
            # the real front/back depth difference.
            first_near = (
                normalized_masks[first_index]
                & ~normalized_masks[second_index]
                & (cv2.dilate(normalized_masks[second_index].astype(np.uint8), kernel) > 0)
            )
            second_near = (
                normalized_masks[second_index]
                & ~normalized_masks[first_index]
                & (cv2.dilate(normalized_masks[first_index].astype(np.uint8), kernel) > 0)
            )
            contact_pixels = min(
                int(np.count_nonzero(first_near)),
                int(np.count_nonzero(second_near)),
            )
            minimum_contact = max(
                8,
                round(
                    min(
                        instances[first_index]["mask_pixels"],
                        instances[second_index]["mask_pixels"],
                    )
                    * 0.002
                ),
            )
            use_boundary = contact_pixels >= minimum_contact
            if use_boundary:
                first_local = _stable_near_depth(
                    depth[first_near & np.isfinite(depth) & (depth > 0)],
                    minimum_pixels=10,
                )
                second_local = _stable_near_depth(
                    depth[second_near & np.isfinite(depth) & (depth > 0)],
                    minimum_pixels=10,
                )
            else:
                first_local = {"reliable": False}
                second_local = {"reliable": False}
            comparison_source = "boundary"
            if not first_local["reliable"] or not second_local["reliable"]:
                # Strongly overlapping projected boxes are likely front/back
                # instances even when SAM leaves a small gap between masks.
                if horizontal_overlap < 0.30 or vertical_overlap < 0.35:
                    continue
                first_local = instances[first_index]["depth_estimate"]
                second_local = instances[second_index]["depth_estimate"]
                comparison_source = "overlapping_instance"
            first_depth = float(first_local["depth_mm"])
            second_depth = float(second_local["depth_mm"])
            first_mad = float(first_local["mad_mm"] or 0.0)
            second_mad = float(second_local["mad_mm"] or 0.0)
            threshold = max(
                30.0,
                3.0 * 1.4826 * max(first_mad, second_mad),
                0.012 * min(first_depth, second_depth),
            )
            difference = abs(first_depth - second_depth)
            if comparison_source == "boundary":
                # Boundary pixels may still be noisy for transparent packages.
                # Compare their direction and strength with the complete
                # instance estimates before accepting a marginal boundary edge.
                first_global = instances[first_index]["depth_estimate"]
                second_global = instances[second_index]["depth_estimate"]
                global_first_depth = float(first_global["depth_mm"])
                global_second_depth = float(second_global["depth_mm"])
                global_threshold = max(
                    30.0,
                    3.0
                    * 1.4826
                    * max(
                        float(first_global["mad_mm"] or 0.0),
                        float(second_global["mad_mm"] or 0.0),
                    ),
                    0.012 * min(global_first_depth, global_second_depth),
                )
                global_difference = abs(global_first_depth - global_second_depth)
                local_strength = difference / max(1.0, threshold)
                global_strength = global_difference / max(1.0, global_threshold)
                directions_conflict = (
                    (second_depth - first_depth)
                    * (global_second_depth - global_first_depth)
                    < 0
                )
                use_global = global_strength > 1.0 and (
                    local_strength <= 1.0
                    or global_strength >= local_strength * 1.25
                )
                if directions_conflict and local_strength > 1.0 and not use_global:
                    # Two similarly strong but contradictory estimates are not
                    # sufficient evidence for either occlusion direction.
                    continue
                if (
                    not directions_conflict
                    and local_strength < 1.35
                    and global_strength <= 1.0
                ):
                    # A marginal boundary-only difference is easily produced
                    # by transparent bottles or a thin SAM fragment.  Do not
                    # let it hide a complete instance unless the full-instance
                    # depth agrees or the local separation is clearly strong.
                    continue
                if use_global:
                    first_depth = global_first_depth
                    second_depth = global_second_depth
                    threshold = global_threshold
                    difference = global_difference
                    comparison_source = (
                        "instance_conflict_override"
                        if directions_conflict
                        else "instance_fallback"
                    )
            if difference <= threshold:
                continue
            front_index, back_index = (
                (first_index, second_index)
                if first_depth < second_depth
                else (second_index, first_index)
            )
            edge = {
                "front": front_index + 1,
                "back": back_index + 1,
                "depth_delta_mm": round(difference, 2),
                "threshold_mm": round(threshold, 2),
                "contact_pixels": contact_pixels,
                "horizontal_overlap": round(horizontal_overlap, 4),
                "vertical_overlap": round(vertical_overlap, 4),
                "comparison_source": comparison_source,
                "confidence": round(min(1.0, difference / max(1.0, threshold * 2.0)), 4),
            }
            edges.append(edge)
            instances[front_index]["outgoing"].append(back_index + 1)
            instances[back_index]["incoming"].append(front_index + 1)

    front_indices: list[int] = []
    uncertain_indices: list[int] = []
    for instance in instances:
        index = int(instance["instance_index"])
        if instance["duplicate_of"] is not None:
            instance["selection_reason"] = f"duplicate_of_{instance['duplicate_of']}"
        elif not instance["depth_estimate"]["reliable"]:
            instance["selection_reason"] = "unreliable_depth"
            uncertain_indices.append(index)
        elif instance["incoming"]:
            instance["selection_reason"] = "occluded_by_" + "_".join(
                str(value) for value in instance["incoming"]
            )
        else:
            instance["selected"] = True
            instance["selection_reason"] = "front_layer"
            front_indices.append(index)

    count_constraint = None
    if expected_front_count is not None:
        count_constraint = _apply_expected_count(
            instances,
            scores=normalized_scores,
            expected_count=expected_front_count,
            image_shape=depth.shape,
            horizontal_roi=horizontal_roi,
            prefer_global_depth_layer=prefer_global_depth_layer,
            prefer_regular_columns=prefer_regular_columns,
            prefer_vertical_position_anomaly=prefer_vertical_position_anomaly,
            max_same_prompt_depth_spread_mm=max_same_prompt_depth_spread_mm,
            enforce_expected_count=enforce_expected_count,
        )
        front_indices = [
            int(instance["instance_index"])
            for instance in instances
            if instance["selected"]
        ]

    combined = np.zeros(depth.shape, dtype=np.uint8)
    for index in front_indices:
        combined[normalized_masks[index - 1]] = 255
    return {
        "instances": instances,
        "edges": edges,
        "front_indices": front_indices,
        "uncertain_indices": uncertain_indices,
        "count_constraint": count_constraint,
        "combined_front_mask": combined,
        "inner_masks": inner_masks,
    }
