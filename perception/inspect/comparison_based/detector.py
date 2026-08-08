"""Detect shelf shortages by comparing a full-shelf image with a later image.

The implementation follows the classic OpenCV pipeline supplied with this
project: resize/align, grayscale + CLAHE, absolute difference, OTSU
thresholding, morphology, and contour-area filtering.  Feature-based image
registration is included because small camera pose changes otherwise dominate
the pixel difference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


ImageInput = str | Path | np.ndarray


@dataclass(frozen=True)
class ComparisonConfig:
    """Parameters for the comparison pipeline.

    ``reference_item_area`` is the pixel area of one product in the baseline
    image.  When provided, only regions whose contour area is at least
    ``shortage_area_ratio * reference_item_area`` are returned.  When it is
    omitted, ``min_contour_area_ratio`` supplies a conservative image-relative
    lower bound.
    """

    enable_registration: bool = True
    registration_max_dimension: int = 1200
    orb_features: int = 5000
    match_ratio: float = 0.75
    ransac_reprojection_threshold: float = 4.0
    min_registration_matches: int = 20
    min_registration_inliers: int = 12

    clahe_clip_limit: float = 3.0
    clahe_grid_size: tuple[int, int] = (8, 8)
    blur_kernel_size: int = 5
    min_diff_threshold: int = 18
    otsu_threshold_scale: float = 1.5
    open_kernel_size: int = 5
    close_kernel_size: int = 17
    morphology_iterations: int = 1

    reference_item_area: float | None = None
    shortage_area_ratio: float = 0.8
    min_contour_area_ratio: float = 0.0015
    max_contour_area_ratio: float = 0.25
    ignore_border_ratio: float = 0.015
    overlap_edge_clearance_ratio: float = 0.02
    merge_gap_ratio: float = 0.006
    min_area_relative_to_largest: float = 0.2

    def __post_init__(self) -> None:
        odd_positive = {
            "blur_kernel_size": self.blur_kernel_size,
            "open_kernel_size": self.open_kernel_size,
            "close_kernel_size": self.close_kernel_size,
        }
        for name, value in odd_positive.items():
            if value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if self.reference_item_area is not None and self.reference_item_area <= 0:
            raise ValueError("reference_item_area must be positive")
        if not 0 < self.match_ratio < 1:
            raise ValueError("match_ratio must be between 0 and 1")
        if self.otsu_threshold_scale <= 0:
            raise ValueError("otsu_threshold_scale must be positive")
        if not 0 < self.shortage_area_ratio <= 1:
            raise ValueError("shortage_area_ratio must be between 0 and 1")
        if not 0 <= self.ignore_border_ratio < 0.5:
            raise ValueError("ignore_border_ratio must be in [0, 0.5)")
        if not 0 <= self.overlap_edge_clearance_ratio < 0.5:
            raise ValueError("overlap_edge_clearance_ratio must be in [0, 0.5)")
        if not 0 <= self.min_area_relative_to_largest <= 1:
            raise ValueError("min_area_relative_to_largest must be in [0, 1]")
        if not 0 < self.min_contour_area_ratio < self.max_contour_area_ratio <= 1:
            raise ValueError("contour area ratios must satisfy 0 < min < max <= 1")


@dataclass(frozen=True)
class AlignmentInfo:
    attempted: bool
    success: bool
    matches: int = 0
    inliers: int = 0
    reason: str = ""
    homography: list[list[float]] | None = None


@dataclass(frozen=True)
class ShortageRegion:
    """One detected shortage region in baseline-image pixel coordinates."""

    bbox: tuple[int, int, int, int]
    contour_area: float
    changed_pixels: int
    area_ratio_to_reference: float | None
    center: tuple[int, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonResult:
    image_size: tuple[int, int]
    threshold: float
    minimum_shortage_area: float
    alignment: AlignmentInfo
    shortages: list[ShortageRegion]
    aligned_current: np.ndarray = field(repr=False)
    difference: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)

    @property
    def has_shortage(self) -> bool:
        return bool(self.shortages)

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_shortage": self.has_shortage,
            "image_size": self.image_size,
            "threshold": self.threshold,
            "minimum_shortage_area": self.minimum_shortage_area,
            "alignment": asdict(self.alignment),
            "shortages": [region.as_dict() for region in self.shortages],
        }

    def draw(self, baseline: ImageInput) -> np.ndarray:
        """Return the baseline image annotated with shortage boxes."""

        output = _read_image(baseline).copy()
        target_width, target_height = self.image_size
        if output.shape[1] != target_width or output.shape[0] != target_height:
            output = cv2.resize(
                output,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        for index, region in enumerate(self.shortages, start=1):
            x, y, width, height = region.bbox
            cv2.rectangle(output, (x, y), (x + width, y + height), (0, 0, 255), 4)
            label = f"shortage {index}: {region.contour_area:.0f}px"
            cv2.putText(
                output,
                label,
                (x, max(28, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return output


class ShortageDetector:
    """Reusable detector configured for a particular camera/shelf setup."""

    def __init__(self, config: ComparisonConfig | None = None) -> None:
        self.config = config or ComparisonConfig()

    def detect(self, baseline: ImageInput, current: ImageInput) -> ComparisonResult:
        baseline_image = _read_image(baseline)
        current_image = _read_image(current)

        height, width = baseline_image.shape[:2]
        if current_image.shape[:2] != (height, width):
            current_image = cv2.resize(
                current_image,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        aligned, valid_mask, alignment = self._align(baseline_image, current_image)
        base_gray = self._preprocess(baseline_image)
        current_gray = self._preprocess(aligned)
        difference = cv2.absdiff(current_gray, base_gray)

        # Exclude non-overlapping warp borders from OTSU's histogram.  Large
        # black triangles can otherwise raise the threshold and hide the item.
        valid_differences = difference[valid_mask != 0]
        otsu_threshold, _ = cv2.threshold(
            valid_differences,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        # A modest multiplier suppresses residual shelf edges after projective
        # registration while retaining OTSU's scene-adaptive behavior.
        threshold = max(
            float(otsu_threshold) * self.config.otsu_threshold_scale,
            float(self.config.min_diff_threshold),
        )
        _, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_and(mask, valid_mask)
        mask = self._clean_mask(mask)
        regions, minimum_area = self._extract_regions(mask, valid_mask)

        return ComparisonResult(
            image_size=(width, height),
            threshold=threshold,
            minimum_shortage_area=minimum_area,
            alignment=alignment,
            shortages=regions,
            aligned_current=aligned,
            difference=difference,
            mask=mask,
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )
        gray = clahe.apply(gray)
        if self.config.blur_kernel_size > 1:
            gray = cv2.GaussianBlur(
                gray,
                (self.config.blur_kernel_size, self.config.blur_kernel_size),
                0,
            )
        return gray

    def _align(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, AlignmentInfo]:
        height, width = baseline.shape[:2]
        valid_mask = np.full((height, width), 255, dtype=np.uint8)
        if not self.config.enable_registration:
            return (
                current,
                self._remove_border(valid_mask),
                AlignmentInfo(False, False, reason="disabled"),
            )

        scale = min(
            1.0,
            self.config.registration_max_dimension / max(height, width),
        )
        small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        base_small = cv2.resize(baseline, small_size, interpolation=cv2.INTER_AREA)
        current_small = cv2.resize(current, small_size, interpolation=cv2.INTER_AREA)
        base_gray = cv2.cvtColor(base_small, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=self.config.orb_features)
        keypoints_base, descriptors_base = orb.detectAndCompute(base_gray, None)
        keypoints_current, descriptors_current = orb.detectAndCompute(current_gray, None)
        if descriptors_base is None or descriptors_current is None:
            info = AlignmentInfo(True, False, reason="not enough image features")
            return current, self._remove_border(valid_mask), info

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(descriptors_current, descriptors_base, k=2)
        matches = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < self.config.match_ratio * second.distance
        ]
        if len(matches) < self.config.min_registration_matches:
            info = AlignmentInfo(
                True,
                False,
                matches=len(matches),
                reason="not enough reliable feature matches",
            )
            return current, self._remove_border(valid_mask), info

        source = np.float32(
            [keypoints_current[match.queryIdx].pt for match in matches]
        ).reshape(-1, 1, 2)
        target = np.float32(
            [keypoints_base[match.trainIdx].pt for match in matches]
        ).reshape(-1, 1, 2)
        homography_small, inlier_mask = cv2.findHomography(
            source,
            target,
            cv2.RANSAC,
            self.config.ransac_reprojection_threshold,
        )
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if homography_small is None or inliers < self.config.min_registration_inliers:
            info = AlignmentInfo(
                True,
                False,
                matches=len(matches),
                inliers=inliers,
                reason="homography estimation was unreliable",
            )
            return current, self._remove_border(valid_mask), info

        scale_to_small = np.array(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        homography = np.linalg.inv(scale_to_small) @ homography_small @ scale_to_small
        aligned = cv2.warpPerspective(
            current,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid_mask = cv2.warpPerspective(
            valid_mask,
            homography,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid_mask = self._remove_border(valid_mask)
        info = AlignmentInfo(
            attempted=True,
            success=True,
            matches=len(matches),
            inliers=inliers,
            homography=homography.tolist(),
        )
        return aligned, valid_mask, info

    def _remove_border(self, valid_mask: np.ndarray) -> np.ndarray:
        height, width = valid_mask.shape
        border = round(min(height, width) * self.config.ignore_border_ratio)
        if border:
            valid_mask[:border, :] = 0
            valid_mask[-border:, :] = 0
            valid_mask[:, :border] = 0
            valid_mask[:, -border:] = 0
        return valid_mask

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config.open_kernel_size, self.config.open_kernel_size),
        )
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config.close_kernel_size, self.config.close_kernel_size),
        )
        cleaned = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=self.config.morphology_iterations,
        )
        return cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=self.config.morphology_iterations,
        )

    def _extract_regions(
        self,
        mask: np.ndarray,
        valid_mask: np.ndarray,
    ) -> tuple[list[ShortageRegion], float]:
        height, width = mask.shape
        image_area = float(height * width)
        minimum_area = (
            self.config.shortage_area_ratio * self.config.reference_item_area
            if self.config.reference_item_area is not None
            else self.config.min_contour_area_ratio * image_area
        )
        maximum_area = self.config.max_contour_area_ratio * image_area
        edge_clearance = (
            min(height, width) * self.config.overlap_edge_clearance_ratio
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        distance_to_overlap_edge = cv2.distanceTransform(
            valid_mask,
            cv2.DIST_L2,
            cv2.DIST_MASK_3,
        )
        boxes = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not minimum_area <= area <= maximum_area:
                continue
            points = contour.reshape(-1, 2)
            # A region clipped by the homography overlap is almost always an
            # entering person, floor, or black warp border rather than stock.
            if (
                np.min(distance_to_overlap_edge[points[:, 1], points[:, 0]])
                <= edge_clearance
            ):
                continue
            boxes.append(cv2.boundingRect(contour))
        boxes = _merge_boxes(boxes, round(min(height, width) * self.config.merge_gap_ratio))

        regions: list[ShortageRegion] = []
        for x, y, box_width, box_height in boxes:
            roi = mask[y : y + box_height, x : x + box_width]
            changed_pixels = int(cv2.countNonZero(roi))
            roi_contours, _ = cv2.findContours(
                roi,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            area = float(sum(cv2.contourArea(contour) for contour in roi_contours))
            if area < minimum_area or area > maximum_area:
                continue
            ratio = (
                area / self.config.reference_item_area
                if self.config.reference_item_area is not None
                else None
            )
            regions.append(
                ShortageRegion(
                    bbox=(x, y, box_width, box_height),
                    contour_area=area,
                    changed_pixels=changed_pixels,
                    area_ratio_to_reference=ratio,
                    center=(x + box_width // 2, y + box_height // 2),
                )
            )
        regions.sort(key=lambda region: region.contour_area, reverse=True)
        if regions and self.config.reference_item_area is None:
            relative_minimum = (
                regions[0].contour_area * self.config.min_area_relative_to_largest
            )
            regions = [
                region
                for region in regions
                if region.contour_area >= relative_minimum
            ]
        return regions, minimum_area


def detect_shortage(
    baseline: ImageInput,
    current: ImageInput,
    config: ComparisonConfig | None = None,
) -> ComparisonResult:
    """Convenience wrapper for one before/after comparison."""

    return ShortageDetector(config).detect(baseline, current)


def _read_image(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image array must have shape (height, width, 3)")
        if image.dtype != np.uint8:
            raise ValueError("image array must use uint8 pixels")
        return image

    path = Path(image)
    # np.fromfile + imdecode supports non-ASCII paths on Windows.
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"cannot read image: {path}") from exc
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"unsupported or invalid image: {path}")
    return decoded


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write an OpenCV image, including to a non-ASCII Windows path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    extension = target.suffix or ".jpg"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"unsupported output image extension: {extension}")
    encoded.tofile(target)


def _merge_boxes(
    boxes: Sequence[tuple[int, int, int, int]],
    gap: int,
) -> list[tuple[int, int, int, int]]:
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        result: list[tuple[int, int, int, int]] = []
        while merged:
            current = merged.pop()
            index = next(
                (
                    i
                    for i, other in enumerate(merged)
                    if _boxes_touch(current, other, gap)
                ),
                None,
            )
            if index is None:
                result.append(current)
                continue
            other = merged.pop(index)
            merged.append(_union_box(current, other))
            changed = True
        merged = result
    return merged


def _boxes_touch(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return not (
        ax + aw + gap < bx
        or bx + bw + gap < ax
        or ay + ah + gap < by
        or by + bh + gap < ay
    )


def _union_box(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = min(ax, bx), min(ay, by)
    right, bottom = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    return left, top, right - left, bottom - top
