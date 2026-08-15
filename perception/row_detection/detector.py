"""Shared detection of red shelf rails and product-row regions.

The detector is intentionally training-free.  It isolates red pixels in HSV,
joins horizontal fragments, scores every image row by horizontal support, and
merges adjacent responses into shelf rails.  Each accepted rail closes the
product row immediately above it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np


ImageInput = str | Path | np.ndarray
BBox = tuple[int, int, int, int]
PoseType = Literal["", "SHELF_VIEW_UPPER", "SHELF_VIEW_LOWER"]


@dataclass(frozen=True)
class RowDetectionConfig:
    """Tunable parameters for 1280x720 shelf images."""

    target_size: tuple[int, int] | None = (1280, 720)
    red_hue_low_max: int = 16
    red_hue_high_min: int = 164
    min_saturation: int = 48
    min_value: int = 38
    horizontal_close_ratio: float = 0.065
    horizontal_open_ratio: float = 0.022
    morphology_height: int = 5
    min_row_red_ratio: float = 0.18
    min_horizontal_run_ratio: float = 0.28
    min_rail_span_ratio: float = 0.34
    max_rail_height_ratio: float = 0.12
    merge_y_gap_ratio: float = 0.010
    min_product_row_height_ratio: float = 0.075
    include_trailing_row: bool = False
    pose_type: PoseType = ""
    enable_hough_fallback: bool = True
    hough_min_line_ratio: float = 0.125
    hough_max_line_gap_ratio: float = 0.125
    hough_max_slope: float = 0.20
    hough_min_red_ratio: float = 0.35
    hough_cluster_gap_ratio: float = 0.06

    def __post_init__(self) -> None:
        if self.target_size is not None and (
            len(self.target_size) != 2
            or self.target_size[0] <= 0
            or self.target_size[1] <= 0
        ):
            raise ValueError("target_size must be a positive (width, height) pair")
        if not 0 <= self.red_hue_low_max < self.red_hue_high_min <= 180:
            raise ValueError("invalid red hue ranges")
        for name in ("min_saturation", "min_value"):
            value = getattr(self, name)
            if not 0 <= value <= 255:
                raise ValueError(f"{name} must be in [0, 255]")
        ratio_names = (
            "horizontal_close_ratio",
            "horizontal_open_ratio",
            "min_row_red_ratio",
            "min_horizontal_run_ratio",
            "min_rail_span_ratio",
            "max_rail_height_ratio",
            "merge_y_gap_ratio",
            "min_product_row_height_ratio",
            "hough_min_line_ratio",
            "hough_max_line_gap_ratio",
            "hough_max_slope",
            "hough_min_red_ratio",
            "hough_cluster_gap_ratio",
        )
        for name in ratio_names:
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.morphology_height <= 0:
            raise ValueError("morphology_height must be positive")
        if self.pose_type not in {"", "SHELF_VIEW_UPPER", "SHELF_VIEW_LOWER"}:
            raise ValueError("pose_type is invalid")


@dataclass(frozen=True)
class ShelfRail:
    """One detected horizontal shelf rail."""

    bbox: BBox
    y_center: int
    span_ratio: float
    red_support_ratio: float
    line: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class ShelfRow:
    """Product area above a shelf rail, indexed from top to bottom."""

    index: int
    bbox: BBox
    lower_rail_index: int | None


@dataclass(frozen=True)
class ShelfRowMatch:
    """Reliable vertical assignment of one bbox to a detected shelf row."""

    row_index: int
    row_bbox: BBox
    overlap_ratio: float
    detected_row_index: int | None = None


@dataclass
class RowDetectionResult:
    image_size: tuple[int, int]
    pose_type: PoseType
    rails: list[ShelfRail]
    rows: list[ShelfRow]
    red_mask: np.ndarray = field(repr=False)
    horizontal_mask: np.ndarray = field(repr=False)
    resized_image: np.ndarray = field(repr=False)

    def row_for_bbox(self, bbox: Sequence[int | float]) -> ShelfRow | None:
        """Return the product row with the largest vertical overlap."""

        if len(bbox) != 4:
            raise ValueError("bbox must be [x, y, width, height]")
        _, y, _, height = (float(value) for value in bbox)
        if height <= 0:
            return None
        bottom = y + height
        best_row: ShelfRow | None = None
        best_overlap = 0.0
        for row in self.rows:
            _, row_y, _, row_height = row.bbox
            overlap = max(0.0, min(bottom, row_y + row_height) - max(y, row_y))
            if overlap > best_overlap:
                best_overlap = overlap
                best_row = row
        return best_row

    def match_bboxes(
        self,
        bboxes: Sequence[Sequence[int | float]],
        *,
        expected_row_count: int | None = None,
        min_overlap_ratio: float = 0.6,
    ) -> list[ShelfRowMatch | None]:
        """Assign bboxes to rows only when the row layout and overlap are reliable."""

        if expected_row_count is not None and expected_row_count <= 0:
            raise ValueError("expected_row_count must be positive")
        if not 0 <= min_overlap_ratio <= 1:
            raise ValueError("min_overlap_ratio must be in [0, 1]")
        if expected_row_count is not None and len(self.rows) != expected_row_count:
            return [None] * len(bboxes)

        matches: list[ShelfRowMatch | None] = []
        for bbox in bboxes:
            if len(bbox) != 4:
                raise ValueError("bbox must be [x, y, width, height]")
            _, y, _, height = (float(value) for value in bbox)
            row = self.row_for_bbox(bbox)
            if row is None or height <= 0:
                matches.append(None)
                continue
            _, row_y, _, row_height = row.bbox
            overlap = max(
                0.0,
                min(y + height, row_y + row_height) - max(y, row_y),
            )
            overlap_ratio = overlap / height
            if overlap_ratio < min_overlap_ratio:
                matches.append(None)
                continue
            matches.append(
                ShelfRowMatch(
                    row_index=row.index,
                    row_bbox=row.bbox,
                    overlap_ratio=round(overlap_ratio, 4),
                    detected_row_index=row.index,
                )
            )
        return matches

    def match_bboxes_to_row_window(
        self,
        bboxes: Sequence[Sequence[int | float]],
        *,
        row_count: int,
        anchor: Literal["top", "bottom"],
        min_overlap_ratio: float = 0.6,
    ) -> list[ShelfRowMatch | None]:
        """Map bboxes into a top/bottom row window and renumber it from one.

        Cameras can include one adjacent shelf row outside the requested pose.
        SKU candidate rows are pose-relative, so an upper view uses the top N
        detected rows while a lower view uses the bottom N detected rows.  More
        than one extra row is treated as an unreliable layout and falls back.
        """

        if row_count <= 0:
            raise ValueError("row_count must be positive")
        if anchor not in {"top", "bottom"}:
            raise ValueError("anchor must be top or bottom")
        if not 0 <= min_overlap_ratio <= 1:
            raise ValueError("min_overlap_ratio must be in [0, 1]")
        if len(self.rows) < row_count or len(self.rows) > row_count + 1:
            return [None] * len(bboxes)

        selected_rows = (
            self.rows[:row_count]
            if anchor == "top"
            else self.rows[-row_count:]
        )
        sku_index_by_detected_index = {
            row.index: sku_index
            for sku_index, row in enumerate(selected_rows, start=1)
        }
        matches: list[ShelfRowMatch | None] = []
        for bbox in bboxes:
            if len(bbox) != 4:
                raise ValueError("bbox must be [x, y, width, height]")
            _, y, _, height = (float(value) for value in bbox)
            if height <= 0:
                matches.append(None)
                continue
            bottom = y + height
            best_row: ShelfRow | None = None
            best_overlap = 0.0
            for row in selected_rows:
                _, row_y, _, row_height = row.bbox
                overlap = max(
                    0.0,
                    min(bottom, row_y + row_height) - max(y, row_y),
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_row = row
            overlap_ratio = best_overlap / height
            if best_row is None or overlap_ratio < min_overlap_ratio:
                matches.append(None)
                continue
            matches.append(
                ShelfRowMatch(
                    row_index=sku_index_by_detected_index[best_row.index],
                    row_bbox=best_row.bbox,
                    overlap_ratio=round(overlap_ratio, 4),
                    detected_row_index=best_row.index,
                )
            )
        return matches

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_size": list(self.image_size),
            "pose_type": self.pose_type,
            "rails": [asdict(rail) for rail in self.rails],
            "rows": [asdict(row) for row in self.rows],
        }

    def draw(self) -> np.ndarray:
        """Draw rails and alternating product-row overlays."""

        canvas = self.resized_image.copy()
        overlay = canvas.copy()
        colors = ((44, 130, 255), (62, 190, 92))
        for row in self.rows:
            x, y, width, height = row.bbox
            color = colors[(row.index - 1) % len(colors)]
            cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), color, -1)
        canvas = cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0.0)

        for row in self.rows:
            x, y, _, _ = row.bbox
            cv2.putText(
                canvas,
                f"ROW {row.index}",
                (x + 12, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        for index, rail in enumerate(self.rails, start=1):
            x, y, width, height = rail.bbox
            if rail.line is not None:
                cv2.line(
                    canvas,
                    rail.line[:2],
                    rail.line[2:],
                    (0, 0, 255),
                    5,
                    cv2.LINE_AA,
                )
            else:
                cv2.rectangle(
                    canvas,
                    (x, y),
                    (x + width - 1, y + height - 1),
                    (0, 0, 255),
                    3,
                )
            cv2.putText(
                canvas,
                f"RAIL {index}",
                (max(4, x), max(24, y - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return canvas

    def save_debug(self, output_dir: str | Path) -> dict[str, Path]:
        """Save masks, annotation, and machine-readable detection details."""

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "resized": directory / "01_resized.jpg",
            "red_mask": directory / "02_red_mask.png",
            "horizontal_mask": directory / "03_horizontal_mask.png",
            "annotated": directory / "04_rows_and_rails.jpg",
            "result": directory / "result.json",
        }
        write_image(artifacts["resized"], self.resized_image)
        write_image(artifacts["red_mask"], self.red_mask)
        write_image(artifacts["horizontal_mask"], self.horizontal_mask)
        write_image(artifacts["annotated"], self.draw())
        artifacts["result"].write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return artifacts


def read_image(image: ImageInput) -> np.ndarray:
    """Read an image, including Windows paths containing non-ASCII text."""

    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("image array is empty")
        return image.copy()
    path = Path(image)
    encoded = np.fromfile(path, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"cannot read image: {path}")
    return decoded


def write_image(path: str | Path, image: np.ndarray) -> Path:
    """Write an image to a path that may contain non-ASCII text."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    extension = output.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"cannot encode image as {extension}")
    encoded.tofile(output)
    return output


def _odd_kernel(value: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(value)))
    return size if size % 2 == 1 else size + 1


def _longest_true_run(row: np.ndarray) -> int:
    padded = np.pad(row.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    if edges.size < 2:
        return 0
    return int(np.max(edges[1::2] - edges[::2]))


def _contiguous_bands(active: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(active.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2])]


def _merge_bands(
    bands: list[tuple[int, int]], max_gap: int
) -> list[tuple[int, int]]:
    if not bands:
        return []
    merged = [bands[0]]
    for start, end in bands[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= max_gap:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def _find_rails(
    red_mask: np.ndarray,
    horizontal_mask: np.ndarray,
    config: RowDetectionConfig,
) -> list[ShelfRail]:
    height, width = red_mask.shape
    red_counts = np.count_nonzero(red_mask, axis=1)
    # Use the unclosed mask for scoring.  The closed mask is excellent for
    # visualization, but it can join several red packages into one wide block.
    longest_runs = np.array(
        [_longest_true_run(row > 0) for row in red_mask], dtype=np.int32
    )
    active = (red_counts >= width * config.min_row_red_ratio) & (
        longest_runs >= width * config.min_horizontal_run_ratio
    )

    max_gap = max(2, int(round(height * config.merge_y_gap_ratio)))
    max_height = max(6, int(round(height * config.max_rail_height_ratio)))
    bands = _merge_bands(_contiguous_bands(active), max_gap)
    rails: list[ShelfRail] = []
    for start, end in bands:
        # Include nearby raw red response when computing the visible bbox.
        y0 = max(0, start - 2)
        y1 = min(height, end + 2)
        ys, xs = np.where(red_mask[y0:y1] > 0)
        if xs.size == 0:
            continue
        top = y0 + int(ys.min())
        bottom = y0 + int(ys.max()) + 1
        rail_height = bottom - top
        left = int(xs.min())
        right = int(xs.max()) + 1
        span_ratio = (right - left) / width
        if rail_height > max_height or span_ratio < config.min_rail_span_ratio:
            continue
        red_support = float(np.max(red_counts[start:end]) / width)
        rails.append(
            ShelfRail(
                bbox=(left, top, right - left, rail_height),
                y_center=(top + bottom) // 2,
                span_ratio=round(span_ratio, 4),
                red_support_ratio=round(red_support, 4),
                line=(left, (top + bottom) // 2, right - 1, (top + bottom) // 2),
            )
        )
    return rails


def _red_ratio_along_line(
    red_mask: np.ndarray,
    line: tuple[int, int, int, int],
    thickness: int,
) -> float:
    sample_mask = np.zeros_like(red_mask)
    x1, y1, x2, y2 = line
    cv2.line(sample_mask, (x1, y1), (x2, y2), 255, thickness)
    selected = sample_mask > 0
    sample_count = int(np.count_nonzero(selected))
    if sample_count == 0:
        return 0.0
    return float(np.count_nonzero(red_mask[selected]) / sample_count)


def _find_sloped_rails(
    image: np.ndarray,
    red_mask: np.ndarray,
    config: RowDetectionConfig,
) -> list[ShelfRail]:
    """Hough fallback for shelf rails with noticeable perspective slope."""

    height, width = red_mask.shape
    # A broader mask is safe here because geometry has already constrained the
    # candidate to a long near-horizontal line.  It recovers pink rails whose
    # saturation falls after glare or JPEG compression.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    loose_red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 20, 25), (28, 255, 255)),
        cv2.inRange(hsv, (152, 20, 25), (180, 255, 255)),
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray_edges = cv2.Canny(gray, 40, 130)
    color_edges = cv2.Canny(loose_red_mask, 50, 150)
    edges = cv2.bitwise_or(gray_edges, color_edges)
    min_length = max(80, int(round(width * config.hough_min_line_ratio)))
    max_gap = max(20, int(round(width * config.hough_max_line_gap_ratio)))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360,
        threshold=45,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )
    if lines is None:
        return []

    center_x = width / 2.0
    line_thickness = max(9, int(round(height * 0.03)))
    candidates: list[tuple[float, tuple[int, int, int, int], float]] = []
    for raw_line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = (int(value) for value in raw_line)
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        delta_x = x2 - x1
        delta_y = y2 - y1
        if delta_x < min_length or abs(delta_y) / max(delta_x, 1) > config.hough_max_slope:
            continue
        line = (x1, y1, x2, y2)
        red_ratio = _red_ratio_along_line(loose_red_mask, line, line_thickness)
        if red_ratio < config.hough_min_red_ratio:
            continue
        y_at_center = y1 + delta_y * (center_x - x1) / delta_x
        candidates.append((y_at_center, line, red_ratio))

    if not candidates:
        return []
    candidates.sort(key=lambda candidate: candidate[0])
    cluster_gap = max(16, int(round(height * config.hough_cluster_gap_ratio)))
    clusters: list[list[tuple[float, tuple[int, int, int, int], float]]] = []
    for candidate in candidates:
        if not clusters or candidate[0] - np.median(
            [item[0] for item in clusters[-1]]
        ) > cluster_gap:
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)

    rails: list[ShelfRail] = []
    padding = line_thickness // 2
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        endpoints = [coordinate for _, line, _ in cluster for coordinate in (line[:2], line[2:])]
        xs = [point[0] for point in endpoints]
        ys = [point[1] for point in endpoints]
        left = max(0, min(xs) - padding)
        right = min(width, max(xs) + padding + 1)
        top = max(0, min(ys) - padding)
        bottom = min(height, max(ys) + padding + 1)
        span_ratio = (right - left) / width
        if span_ratio < config.min_rail_span_ratio:
            continue
        y_center = int(round(np.median([candidate[0] for candidate in cluster])))
        if y_center < height * 0.055:
            continue
        slopes = []
        intercepts = []
        for _, line, _ in cluster:
            x1, y1, x2, y2 = line
            slope = (y2 - y1) / (x2 - x1)
            slopes.append(slope)
            intercepts.append(y1 - slope * x1)
        slope = float(np.median(slopes))
        intercept = float(np.median(intercepts))
        line_y1 = int(round(np.clip(slope * left + intercept, 0, height - 1)))
        line_y2 = int(round(np.clip(slope * (right - 1) + intercept, 0, height - 1)))
        rails.append(
            ShelfRail(
                bbox=(left, top, right - left, bottom - top),
                y_center=y_center,
                span_ratio=round(span_ratio, 4),
                red_support_ratio=round(max(candidate[2] for candidate in cluster), 4),
                line=(left, line_y1, right - 1, line_y2),
            )
        )
    return rails


def _merge_rail_sources(
    horizontal: Sequence[ShelfRail],
    sloped: Sequence[ShelfRail],
    height: int,
) -> list[ShelfRail]:
    """Prefer color-projection rails and add non-duplicate Hough rails."""

    merged = list(horizontal)
    duplicate_gap = max(18, int(round(height * 0.075)))
    for rail in sloped:
        if any(abs(rail.y_center - existing.y_center) <= duplicate_gap for existing in merged):
            continue
        merged.append(rail)
    return sorted(merged, key=lambda rail: rail.y_center)


def _build_rows(
    rails: Sequence[ShelfRail],
    width: int,
    height: int,
    config: RowDetectionConfig,
) -> list[ShelfRow]:
    min_height = max(20, int(round(height * config.min_product_row_height_ratio)))
    rail_rows: list[ShelfRow] = []
    top = 0
    for rail_index, rail in enumerate(rails):
        # Center lines are a more stable boundary than axis-aligned rail boxes
        # when perspective turns a long shelf edge into a tall diagonal bbox.
        rail_y = rail.y_center
        if rail_y - top >= min_height:
            rail_rows.append(
                ShelfRow(
                    index=len(rail_rows) + 1,
                    bbox=(0, top, width, rail_y - top),
                    lower_rail_index=rail_index,
                )
            )
        top = rail_y
    trailing_row: ShelfRow | None = None
    if height - top >= min_height:
        trailing_row = ShelfRow(
            index=len(rail_rows) + 1,
            bbox=(0, top, width, height - top),
            lower_rail_index=None,
        )

    if config.pose_type == "SHELF_VIEW_UPPER":
        # Upper images always correspond to the first two product rows.  Only
        # use the image bottom when the second lower rail itself is missing.
        candidates = list(rail_rows)
        if len(candidates) < 2 and trailing_row is not None:
            candidates.append(trailing_row)
        selected = candidates[:2]
    elif config.pose_type == "SHELF_VIEW_LOWER":
        # A lower image normally contains a partial row above the target three.
        # Four detected rails therefore close all three target rows.  With only
        # three rails, the bottom target row has no visible lower rail and the
        # image boundary closes it instead.
        if len(rail_rows) >= 4:
            selected = rail_rows[-3:]
        else:
            candidates = list(rail_rows)
            if trailing_row is not None:
                candidates.append(trailing_row)
            selected = candidates[-3:]
    else:
        candidates = list(rail_rows)
        if config.include_trailing_row and trailing_row is not None:
            candidates.append(trailing_row)
        selected = candidates
    return [
        ShelfRow(
            index=index,
            bbox=row.bbox,
            lower_rail_index=row.lower_rail_index,
        )
        for index, row in enumerate(selected, start=1)
    ]


def detect_rows(
    image: ImageInput,
    config: RowDetectionConfig | None = None,
) -> RowDetectionResult:
    """Detect shelf rails and product rows in one image."""

    settings = config or RowDetectionConfig()
    source = read_image(image)
    if settings.target_size is not None:
        source = cv2.resize(source, settings.target_size, interpolation=cv2.INTER_LINEAR)
    height, width = source.shape[:2]

    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    lower_red = cv2.inRange(
        hsv,
        (0, settings.min_saturation, settings.min_value),
        (settings.red_hue_low_max, 255, 255),
    )
    upper_red = cv2.inRange(
        hsv,
        (settings.red_hue_high_min, settings.min_saturation, settings.min_value),
        (180, 255, 255),
    )
    red_mask = cv2.bitwise_or(lower_red, upper_red)
    red_mask = cv2.medianBlur(red_mask, 3)

    close_width = _odd_kernel(width * settings.horizontal_close_ratio)
    open_width = _odd_kernel(width * settings.horizontal_open_ratio)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_width, settings.morphology_height)
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (open_width, 3)
    )
    horizontal_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, close_kernel)
    horizontal_mask = cv2.morphologyEx(
        horizontal_mask, cv2.MORPH_OPEN, open_kernel
    )

    rails = _find_rails(red_mask, horizontal_mask, settings)
    if settings.enable_hough_fallback and len(rails) < 2:
        sloped_rails = _find_sloped_rails(source, red_mask, settings)
        rails = _merge_rail_sources(rails, sloped_rails, height)
    rows = _build_rows(rails, width, height, settings)
    return RowDetectionResult(
        image_size=(width, height),
        pose_type=settings.pose_type,
        rails=rails,
        rows=rows,
        red_mask=red_mask,
        horizontal_mask=horizontal_mask,
        resized_image=source,
    )
