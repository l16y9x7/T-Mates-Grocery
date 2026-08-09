"""Detect shelf shortages by comparing a full-shelf image with a later image.

The implementation follows the classic OpenCV pipeline supplied with this
project: resize/align, grayscale + CLAHE, absolute difference, OTSU
thresholding, morphology, and contour-area filtering.  Feature-based image
registration is included because small camera pose changes otherwise dominate
the pixel difference.
"""

from __future__ import annotations

import json
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

    target_size: tuple[int, int] | None = (1280, 720)
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
    otsu_threshold_scale: float = 1.6
    color_difference_weight: float = 2.0
    difference_mode: str = "hybrid"
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
    min_area_relative_to_largest: float = 0.3
    max_bbox_aspect_ratio: float = 5.0
    min_chroma_dominance_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.target_size is not None and (
            len(self.target_size) != 2
            or self.target_size[0] <= 0
            or self.target_size[1] <= 0
        ):
            raise ValueError("target_size must be a positive (width, height) pair")
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
        if self.color_difference_weight < 0:
            raise ValueError("color_difference_weight cannot be negative")
        if self.difference_mode not in {"hybrid", "chroma"}:
            raise ValueError("difference_mode must be 'hybrid' or 'chroma'")
        if not 0 < self.shortage_area_ratio <= 1:
            raise ValueError("shortage_area_ratio must be between 0 and 1")
        if not 0 <= self.ignore_border_ratio < 0.5:
            raise ValueError("ignore_border_ratio must be in [0, 0.5)")
        if not 0 <= self.overlap_edge_clearance_ratio < 0.5:
            raise ValueError("overlap_edge_clearance_ratio must be in [0, 0.5)")
        if not 0 <= self.min_area_relative_to_largest <= 1:
            raise ValueError("min_area_relative_to_largest must be in [0, 1]")
        if self.max_bbox_aspect_ratio < 1:
            raise ValueError("max_bbox_aspect_ratio must be at least 1")
        if not 0 <= self.min_chroma_dominance_ratio <= 1:
            raise ValueError("min_chroma_dominance_ratio must be in [0, 1]")
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
    mean_luminance_difference: float
    mean_chroma_difference: float
    chroma_dominance_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonResult:
    image_size: tuple[int, int]
    difference_mode: str
    threshold: float
    minimum_shortage_area: float
    alignment: AlignmentInfo
    shortages: list[ShortageRegion]
    aligned_current: np.ndarray = field(repr=False)
    luminance_difference: np.ndarray = field(repr=False)
    chroma_difference: np.ndarray = field(repr=False)
    difference: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)

    @property
    def has_shortage(self) -> bool:
        return bool(self.shortages)

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_shortage": self.has_shortage,
            "image_size": self.image_size,
            "difference_mode": self.difference_mode,
            "threshold": self.threshold,
            "minimum_shortage_area": self.minimum_shortage_area,
            "alignment": asdict(self.alignment),
            "shortages": [region.as_dict() for region in self.shortages],
        }

    def draw(self, baseline: ImageInput) -> np.ndarray:
        """Return the baseline image annotated with shortage boxes."""

        output = _resize_to(_read_image(baseline), self.image_size)
        return _draw_regions(output, self.shortages, "BASELINE")

    def save_debug(
        self,
        directory: str | Path,
        baseline: ImageInput,
    ) -> dict[str, Path]:
        """Save intermediate images, bbox visualizations, and result metadata.

        All bbox coordinates use the baseline image coordinate system.  The
        returned mapping contains the absolute path of every saved artifact.
        """

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)

        baseline_image = _resize_to(_read_image(baseline), self.image_size)
        baseline_bboxes = _draw_regions(
            baseline_image,
            self.shortages,
            "BASELINE",
        )
        current_bboxes = _draw_regions(
            self.aligned_current,
            self.shortages,
            "ALIGNED CURRENT",
        )
        difference_heatmap = cv2.applyColorMap(
            self.difference,
            cv2.COLORMAP_TURBO,
        )
        difference_bboxes = _draw_regions(
            difference_heatmap,
            self.shortages,
            "DIFFERENCE HEATMAP",
        )
        comparison_bboxes = np.hstack((baseline_bboxes, current_bboxes))

        names_and_images = {
            "baseline": ("01_baseline.jpg", baseline_image),
            "aligned_current": ("02_aligned_current.jpg", self.aligned_current),
            "luminance_difference": (
                "03a_luminance_difference.png",
                self.luminance_difference,
            ),
            "chroma_difference": (
                "03b_chroma_difference.png",
                self.chroma_difference,
            ),
            "difference": ("03_difference.png", self.difference),
            "difference_heatmap": (
                "04_difference_heatmap.jpg",
                difference_heatmap,
            ),
            "binary_mask": ("05_binary_mask.png", self.mask),
            "baseline_bboxes": ("06_baseline_bboxes.jpg", baseline_bboxes),
            "current_bboxes": ("07_current_bboxes.jpg", current_bboxes),
            "difference_bboxes": ("08_difference_bboxes.jpg", difference_bboxes),
            "comparison_bboxes": ("09_comparison_bboxes.jpg", comparison_bboxes),
        }
        artifacts: dict[str, Path] = {}
        for key, (name, image) in names_and_images.items():
            path = (target / name).resolve()
            write_image(path, image)
            artifacts[key] = path

        metadata_path = (target / "result.json").resolve()
        metadata = self.as_dict()
        metadata["bbox_format"] = ["x", "y", "width", "height"]
        metadata["artifacts"] = {
            key: path.name for key, path in artifacts.items()
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts["metadata"] = metadata_path
        return artifacts


class ShortageDetector:
    """Reusable detector configured for a particular camera/shelf setup."""

    def __init__(self, config: ComparisonConfig | None = None) -> None:
        self.config = config or ComparisonConfig()

    def detect(self, baseline: ImageInput, current: ImageInput) -> ComparisonResult:
        baseline_image = _read_image(baseline)
        current_image = _read_image(current)

        if self.config.target_size is not None:
            target_width, target_height = self.config.target_size
            baseline_image = cv2.resize(
                baseline_image,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            current_image = cv2.resize(
                current_image,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )

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
        luminance_difference = cv2.absdiff(current_gray, base_gray)
        chroma_difference = self._chroma_difference(baseline_image, aligned)
        difference = (
            chroma_difference
            if self.config.difference_mode == "chroma"
            else cv2.max(luminance_difference, chroma_difference)
        )

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
        if self.config.difference_mode == "chroma":
            # Low-saturation pastel swaps have a naturally lower chroma OTSU
            # value, so avoid over-scaling those scenes. More colorful scenes
            # keep the stricter factor to suppress unchanged package edges.
            threshold_scale = (
                self.config.otsu_threshold_scale
                if otsu_threshold >= 30
                else 1.2
            )
            minimum_threshold = 12.0
        else:
            threshold_scale = self.config.otsu_threshold_scale
            minimum_threshold = float(self.config.min_diff_threshold)
        threshold = max(float(otsu_threshold) * threshold_scale, minimum_threshold)
        _, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_and(mask, valid_mask)
        mask = self._clean_mask(mask)
        regions, minimum_area = self._extract_regions(
            mask,
            valid_mask,
            luminance_difference,
            chroma_difference,
        )

        return ComparisonResult(
            image_size=(width, height),
            difference_mode=self.config.difference_mode,
            threshold=threshold,
            minimum_shortage_area=minimum_area,
            alignment=alignment,
            shortages=regions,
            aligned_current=aligned,
            luminance_difference=luminance_difference,
            chroma_difference=chroma_difference,
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

    def _chroma_difference(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> np.ndarray:
        if self.config.color_difference_weight == 0:
            return np.zeros(baseline.shape[:2], dtype=np.uint8)
        kernel_size = self.config.blur_kernel_size
        if kernel_size > 1:
            baseline = cv2.GaussianBlur(
                baseline,
                (kernel_size, kernel_size),
                0,
            )
            current = cv2.GaussianBlur(
                current,
                (kernel_size, kernel_size),
                0,
            )
        baseline_lab = cv2.cvtColor(baseline, cv2.COLOR_BGR2LAB)
        current_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB)
        delta_a = cv2.absdiff(baseline_lab[:, :, 1], current_lab[:, :, 1])
        delta_b = cv2.absdiff(baseline_lab[:, :, 2], current_lab[:, :, 2])
        magnitude = cv2.magnitude(
            delta_a.astype(np.float32),
            delta_b.astype(np.float32),
        )
        return np.clip(
            magnitude * self.config.color_difference_weight,
            0,
            255,
        ).astype(np.uint8)

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
        luminance_difference: np.ndarray,
        chroma_difference: np.ndarray,
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
            aspect_ratio = max(
                box_width / box_height,
                box_height / box_width,
            )
            if aspect_ratio > self.config.max_bbox_aspect_ratio:
                continue
            roi = mask[y : y + box_height, x : x + box_width]
            changed_pixels = int(cv2.countNonZero(roi))
            changed = roi != 0
            luminance_values = luminance_difference[
                y : y + box_height,
                x : x + box_width,
            ][changed]
            chroma_values = chroma_difference[
                y : y + box_height,
                x : x + box_width,
            ][changed]
            chroma_dominance = float(
                np.mean(chroma_values > luminance_values)
            )
            if chroma_dominance < self.config.min_chroma_dominance_ratio:
                continue
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
                    mean_luminance_difference=float(np.mean(luminance_values)),
                    mean_chroma_difference=float(np.mean(chroma_values)),
                    chroma_dominance_ratio=chroma_dominance,
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


def _resize_to(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if image.shape[1] == width and image.shape[0] == height:
        return image.copy()
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _draw_regions(
    image: np.ndarray,
    regions: Sequence[ShortageRegion],
    panel_name: str,
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    thickness = max(2, round(min(height, width) / 400))
    status_color = (0, 0, 255) if regions else (0, 180, 0)
    status = f"CHANGE REGIONS: {len(regions)}" if regions else "NO CHANGE REGION"

    cv2.rectangle(output, (0, 0), (width, 44), (20, 20, 20), cv2.FILLED)
    cv2.putText(
        output,
        f"{panel_name} | {status}",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )

    for index, region in enumerate(regions, start=1):
        x, y, box_width, box_height = region.bbox
        cv2.rectangle(
            output,
            (x, y),
            (x + box_width, y + box_height),
            (0, 0, 255),
            thickness,
        )
        label = f"#{index} x={x} y={y} w={box_width} h={box_height}"
        label_y = min(height - 8, max(68, y - 9))
        cv2.putText(
            output,
            label,
            (max(4, min(x, width - 260)), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return output


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
