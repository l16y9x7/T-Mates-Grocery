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
) -> dict[str, Any]:
    """Select exactly N well-spaced columns while preferring graph-front masks."""

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

    def quality(index: int) -> float:
        instance = instances[index]
        x1, y1, x2, y2 = instance["bbox_xyxy"]
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        fill = min(1.0, instance["mask_pixels"] / bbox_area)
        height_score = min(1.0, (y2 - y1) / max(1.0, image_height * 0.45))
        sam_score = float(scores[index] or 0.0)
        support = float(instance["depth_estimate"]["support_ratio"])
        graph_score = 2.5 if not instance["incoming"] else -0.35 * len(instance["incoming"])
        edge_clearance = min(x1, image_width - x2)
        if edge_clearance <= image_width * 0.02:
            edge_penalty = 2.5
        elif edge_clearance <= image_width * 0.05:
            edge_penalty = 0.8
        else:
            edge_penalty = 0.0
        return (
            graph_score
            + sam_score
            + 0.55 * support
            + 0.45 * fill
            + 0.50 * height_score
            - edge_penalty
        )

    ordered = sorted(
        candidates,
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

    nominal_slot_width = image_width / max(1, expected_count)

    def solve(minimum_gap: float) -> list[int]:
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
                    candidate = (
                        previous[0] + quality(index),
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
    chosen = set(selected)
    for index in candidates:
        instance = instances[index]
        if index in chosen:
            instance["selected"] = True
            instance["selection_reason"] = "front_layer_expected_count"
        else:
            instance["selected"] = False
            if not instance["incoming"]:
                instance["selection_reason"] = "exceeds_expected_front_count"
    return {
        "expected": expected_count,
        "selected": len(selected),
        "satisfied": len(selected) == expected_count,
        "available_candidates": len(candidates),
        "horizontal_roi": [roi_left, roi_right],
    }


def select_front_row_instances(
    masks: Sequence[np.ndarray],
    depth_mm: np.ndarray,
    *,
    scores: Sequence[float | None] | None = None,
    expected_front_count: int | None = None,
    horizontal_roi: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Return front instances, duplicate suppression and pairwise depth edges."""

    depth = np.asarray(depth_mm)
    normalized_masks = [np.asarray(mask) > 0 for mask in masks]
    if any(mask.shape != depth.shape for mask in normalized_masks):
        raise ValueError("mask and depth shapes must match")
    if expected_front_count is not None and expected_front_count <= 0:
        raise ValueError("expected_front_count must be positive")
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
