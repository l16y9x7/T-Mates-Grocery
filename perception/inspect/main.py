"""Unified shelf-inspection API and algorithm fusion entry point.

The public endpoint currently runs the training-free comparison algorithm.  The
pipeline keeps localization details internally; the HTTP response exposes the
stable product-oriented contract that a later Qwen recognition stage will fill.
"""

from __future__ import annotations

import base64
import binascii
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

if __package__:
    from .comparison_based import ComparisonConfig, detect_shortage
    from .comparison_based.qwen_review import (
        DEFAULT_DEBUG_ROOT,
        PoseType,
        QwenReviewError,
        QwenReviewResult,
        QwenReviewer,
        ReviewedFinding,
    )
else:
    INSPECT_ROOT = Path(__file__).resolve().parent
    if str(INSPECT_ROOT) not in sys.path:
        sys.path.insert(0, str(INSPECT_ROOT))
    from comparison_based import ComparisonConfig, detect_shortage
    from comparison_based.qwen_review import (
        DEFAULT_DEBUG_ROOT,
        PoseType,
        QwenReviewError,
        QwenReviewResult,
        QwenReviewer,
        ReviewedFinding,
    )


TaskType = Literal["SHORTAGE", "MISPLACED"]
MAX_IMAGE_BYTES = 20 * 1024 * 1024
FUSION_IOU_THRESHOLD = 0.4

app = FastAPI(
    title="Shelf Inspection",
    version="1.0.0",
    description="Run and fuse shelf anomaly detectors on a baseline/current pair.",
)
router = APIRouter()


class InspectRequest(BaseModel):
    """Two shelf images and the inspection task to run."""

    task_type: TaskType
    location_id: str = Field(min_length=1)
    pose_type: PoseType
    baseline_image_base64: str = Field(min_length=1)
    current_image_base64: str = Field(min_length=1)
    reference_item_area: float | None = Field(default=None, gt=0)

    @field_validator("location_id")
    @classmethod
    def normalize_location_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("location_id must not be blank")
        return normalized


class ShortageProductFinding(BaseModel):
    shortage_product_name: str


class MisplacedProductFinding(BaseModel):
    misplaced_product_name: str
    gt_product_name: str


InspectApiResponse = (
    list[ShortageProductFinding] | list[MisplacedProductFinding]
)


class Finding(BaseModel):
    """One fused change region in the standardized 1280x720 coordinates."""

    bbox: list[int]
    center: list[int]
    sources: list[str]
    votes: int


class AlgorithmFinding(BaseModel):
    bbox: list[int]
    center: list[int]
    contour_area: float
    changed_pixels: int
    chroma_dominance_ratio: float


class AlgorithmResult(BaseModel):
    name: str
    success: bool
    elapsed_ms: float
    findings: list[AlgorithmFinding] = Field(default_factory=list)
    error: str | None = None
    difference_mode: str | None = None
    threshold: float | None = None
    alignment_success: bool | None = None


class InspectResponse(BaseModel):
    location_id: str
    pose_type: PoseType
    task_type: TaskType
    has_anomaly: bool
    image_size: list[int]
    bbox_format: list[str]
    findings: list[Finding]
    algorithms: list[AlgorithmResult]


@dataclass(frozen=True)
class InspectionContext:
    location_id: str
    pose_type: PoseType
    task_type: TaskType
    baseline: np.ndarray
    current: np.ndarray
    reference_item_area: float | None = None


class InspectionAlgorithm(Protocol):
    """Interface implemented by every detector participating in fusion."""

    name: str

    def run(self, context: InspectionContext) -> AlgorithmResult:
        ...


class ComparisonBasedAlgorithm:
    name = "comparison_based"

    def run(self, context: InspectionContext) -> AlgorithmResult:
        started_at = time.perf_counter()
        misplaced = context.task_type == "MISPLACED"
        config = ComparisonConfig(
            target_size=(1280, 720),
            reference_item_area=context.reference_item_area,
            difference_mode="chroma" if misplaced else "hybrid",
            min_chroma_dominance_ratio=0.35 if misplaced else 0.0,
        )
        result = detect_shortage(context.baseline, context.current, config)
        findings = [
            AlgorithmFinding(
                bbox=list(region.bbox),
                center=list(region.center),
                contour_area=region.contour_area,
                changed_pixels=region.changed_pixels,
                chroma_dominance_ratio=region.chroma_dominance_ratio,
            )
            for region in result.shortages
        ]
        return AlgorithmResult(
            name=self.name,
            success=True,
            elapsed_ms=_elapsed_ms(started_at),
            findings=findings,
            difference_mode=result.difference_mode,
            threshold=result.threshold,
            alignment_success=result.alignment.success,
        )


class InspectionPipeline:
    """Run independent algorithms and fuse spatially overlapping findings."""

    def __init__(self, algorithms: Sequence[InspectionAlgorithm] | None = None) -> None:
        self.algorithms = list(algorithms or [ComparisonBasedAlgorithm()])
        if not self.algorithms:
            raise ValueError("at least one inspection algorithm is required")

    def inspect(self, context: InspectionContext) -> InspectResponse:
        algorithm_results: list[AlgorithmResult] = []
        for algorithm in self.algorithms:
            started_at = time.perf_counter()
            try:
                algorithm_results.append(algorithm.run(context))
            except Exception as error:  # Keep future multi-algorithm fusion available.
                algorithm_results.append(
                    AlgorithmResult(
                        name=algorithm.name,
                        success=False,
                        elapsed_ms=_elapsed_ms(started_at),
                        error=f"{type(error).__name__}: {error}",
                    )
                )

        successful = [result for result in algorithm_results if result.success]
        if not successful:
            errors = "; ".join(
                f"{result.name}: {result.error}" for result in algorithm_results
            )
            raise RuntimeError(f"all inspection algorithms failed: {errors}")

        findings = fuse_findings(successful)
        return InspectResponse(
            location_id=context.location_id,
            pose_type=context.pose_type,
            task_type=context.task_type,
            has_anomaly=bool(findings),
            image_size=[1280, 720],
            bbox_format=["x", "y", "width", "height"],
            findings=findings,
            algorithms=algorithm_results,
        )


def inspect_images(
    task_type: TaskType,
    baseline: np.ndarray,
    current: np.ndarray,
    *,
    location_id: str,
    pose_type: PoseType,
    reference_item_area: float | None = None,
    pipeline: InspectionPipeline | None = None,
) -> InspectResponse:
    """Python entry point used by the HTTP route and offline callers."""

    if task_type not in {"SHORTAGE", "MISPLACED"}:
        raise ValueError("task_type must be SHORTAGE or MISPLACED")
    location_id = location_id.strip()
    if not location_id:
        raise ValueError("location_id must not be blank")
    if pose_type not in {"", "SHELF_VIEW_UPPER", "SHELF_VIEW_LOWER"}:
        raise ValueError("pose_type is invalid")
    if reference_item_area is not None and reference_item_area <= 0:
        raise ValueError("reference_item_area must be positive")
    _validate_image(baseline, "baseline")
    _validate_image(current, "current")
    context = InspectionContext(
        location_id=location_id,
        pose_type=pose_type,
        task_type=task_type,
        baseline=baseline,
        current=current,
        reference_item_area=reference_item_area,
    )
    return (pipeline or InspectionPipeline()).inspect(context)


@router.post("/perception/inspect", response_model=InspectApiResponse)
def inspect_shelf(request: InspectRequest) -> InspectApiResponse:
    """Compare a full-shelf reference image with the current shelf image."""

    baseline = decode_image(request.baseline_image_base64, "baseline_image_base64")
    current = decode_image(request.current_image_base64, "current_image_base64")
    try:
        result = inspect_images(
            request.task_type,
            baseline,
            current,
            location_id=request.location_id,
            pose_type=request.pose_type,
            reference_item_area=request.reference_item_area,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    if not result.findings:
        return []

    try:
        review = QwenReviewer(debug_root=DEFAULT_DEBUG_ROOT).review(
            task_type=request.task_type,
            location_id=request.location_id,
            pose_type=request.pose_type,
            current=current,
            bboxes=[finding.bbox for finding in result.findings],
        )
    except QwenReviewError as error:
        raise HTTPException(
            status_code=502,
            detail={
                "type": "qwen_review_error",
                "stage": error.stage,
                "message": str(error),
            },
        ) from error
    return build_product_findings(request.task_type, review.findings)


def build_product_findings(
    task_type: TaskType,
    reviewed_findings: Sequence[ReviewedFinding],
) -> InspectApiResponse:
    """Convert validated Qwen findings into the public response contract."""

    if task_type == "SHORTAGE":
        return [
            ShortageProductFinding(
                shortage_product_name=finding.shortage_product_name or ""
            )
            for finding in reviewed_findings
            if finding.shortage_product_name
        ]
    return [
        MisplacedProductFinding(
            misplaced_product_name=finding.misplaced_product_name or "",
            gt_product_name=finding.gt_product_name or "",
        )
        for finding in reviewed_findings
        if finding.misplaced_product_name and finding.gt_product_name
    ]


def decode_image(value: str, field_name: str) -> np.ndarray:
    """Decode plain base64 or a data URL into an OpenCV BGR image."""

    encoded = value.strip()
    data_url = re.fullmatch(
        r"data:image/[A-Za-z0-9.+-]+;base64,(.+)",
        encoded,
        flags=re.DOTALL,
    )
    if data_url:
        encoded = data_url.group(1)
    encoded = re.sub(r"\s+", "", encoded)

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is not valid base64 image data",
        ) from error
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} exceeds the {MAX_IMAGE_BYTES}-byte limit",
        )

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is not a decodable JPEG/PNG image",
        )
    return image


def fuse_findings(results: Sequence[AlgorithmResult]) -> list[Finding]:
    """Cluster overlapping bboxes and retain their contributing algorithms."""

    clusters: list[list[tuple[str, list[int]]]] = []
    for result in results:
        for finding in result.findings:
            match = next(
                (
                    cluster
                    for cluster in clusters
                    if max(
                        _bbox_iou(finding.bbox, member_bbox)
                        for _, member_bbox in cluster
                    )
                    >= FUSION_IOU_THRESHOLD
                ),
                None,
            )
            if match is None:
                clusters.append([(result.name, finding.bbox)])
            else:
                match.append((result.name, finding.bbox))

    fused: list[Finding] = []
    for cluster in clusters:
        boxes = [bbox for _, bbox in cluster]
        bbox = [round(sum(box[index] for box in boxes) / len(boxes)) for index in range(4)]
        sources = sorted({source for source, _ in cluster})
        fused.append(
            Finding(
                bbox=bbox,
                center=[bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2],
                sources=sources,
                votes=len(sources),
            )
        )
    fused.sort(key=lambda finding: (finding.bbox[1], finding.bbox[0]))
    return fused


def _bbox_iou(first: Sequence[int], second: Sequence[int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection_width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _validate_image(image: np.ndarray, name: str) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be an uint8 BGR image")


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 1)


app.include_router(router)
