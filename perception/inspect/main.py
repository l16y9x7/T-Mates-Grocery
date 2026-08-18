"""Unified shelf-inspection API and algorithm fusion entry point.

The formal SHORTAGE endpoint uses the production baseline/current SAM3
front-row slot comparison.  The legacy comparison/Qwen path remains available
for MISPLACED and caller-supplied offline diagnostics.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, Sequence

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

if __package__ and __package__.startswith("perception."):
    from ..camera_capture import (
        CameraCaptureError,
        capture_head_rgbd,
        inspection_temporary_directory,
    )
    from ..initial_scan import InitialScanError, load_initial_scan
    from .sam_shortage_pipeline import (
        SamShortageError,
        ShortageAnalysis,
        analysis_as_dict,
        analyze_shortage,
        save_shelf_preprocessing_artifacts,
    )
    from .comparison_based import ComparisonConfig, detect_shortage
    from .comparison_based.qwen_review import (
        DEFAULT_DEBUG_ROOT,
        PoseType,
        QwenReviewError,
        QwenReviewResult,
        QwenReviewer,
        ReviewRowConstraint,
        ReviewedFinding,
    )
    from ..row_detection import RowDetectionConfig, RowDetectionResult, detect_rows
else:
    INSPECT_ROOT = Path(__file__).resolve().parent
    PERCEPTION_ROOT = INSPECT_ROOT.parent
    if str(PERCEPTION_ROOT) not in sys.path:
        sys.path.insert(0, str(PERCEPTION_ROOT))
    if str(INSPECT_ROOT) not in sys.path:
        sys.path.insert(0, str(INSPECT_ROOT))
    from camera_capture import (
        CameraCaptureError,
        capture_head_rgbd,
        inspection_temporary_directory,
    )
    from comparison_based import ComparisonConfig, detect_shortage
    from comparison_based.qwen_review import (
        DEFAULT_DEBUG_ROOT,
        PoseType,
        QwenReviewError,
        QwenReviewResult,
        QwenReviewer,
        ReviewRowConstraint,
        ReviewedFinding,
    )
    from initial_scan import InitialScanError, load_initial_scan
    from row_detection import RowDetectionConfig, RowDetectionResult, detect_rows
    from sam_shortage_pipeline import (
        SamShortageError,
        ShortageAnalysis,
        analysis_as_dict,
        analyze_shortage,
        save_shelf_preprocessing_artifacts,
    )


TaskType = Literal["SHORTAGE", "MISPLACED"]
MAX_IMAGE_BYTES = 20 * 1024 * 1024
FUSION_IOU_THRESHOLD = 0.4
MIN_ROW_OVERLAP_RATIO = 0.6
EXPECTED_ROW_COUNTS: dict[PoseType, int] = {
    "": 1,
    "SHELF_VIEW_UPPER": 2,
    "SHELF_VIEW_LOWER": 3,
}
logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Shelf Inspection",
    version="1.0.0",
    description="Run and fuse shelf anomaly detectors on a baseline/current pair.",
)
router = APIRouter()


class InspectRequest(BaseModel):
    """A shelf inspection task whose RGB-D inputs are resolved at runtime."""

    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    location_id: str = Field(min_length=1)
    pose_type: PoseType
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


class InspectApiResponse(BaseModel):
    findings: list[ShortageProductFinding | MisplacedProductFinding] = Field(
        default_factory=list
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


@dataclass(frozen=True)
class AlgorithmExecution:
    """One public algorithm result plus optional internal review artifacts."""

    result: AlgorithmResult
    review_image: np.ndarray | None = None
    review_mask: np.ndarray | None = None
    review_homography: np.ndarray | None = None


@dataclass(frozen=True)
class InspectionExecution:
    """Inspection response and the image sharing its bbox coordinate system."""

    response: InspectResponse
    review_image: np.ndarray
    review_mask: np.ndarray
    review_homography: np.ndarray | None = None


class InspectionAlgorithm(Protocol):
    """Interface implemented by every detector participating in fusion."""

    name: str

    def run(self, context: InspectionContext) -> AlgorithmExecution | AlgorithmResult:
        ...


class ComparisonBasedAlgorithm:
    name = "comparison_based"

    def run(self, context: InspectionContext) -> AlgorithmExecution:
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
        return AlgorithmExecution(
            result=AlgorithmResult(
                name=self.name,
                success=True,
                elapsed_ms=_elapsed_ms(started_at),
                findings=findings,
                difference_mode=result.difference_mode,
                threshold=result.threshold,
                alignment_success=result.alignment.success,
            ),
            review_image=result.aligned_current,
            review_mask=result.mask,
            review_homography=(
                np.asarray(result.alignment.homography, dtype=np.float64)
                if result.alignment.homography is not None
                else None
            ),
        )


class InspectionPipeline:
    """Run independent algorithms and fuse spatially overlapping findings."""

    def __init__(self, algorithms: Sequence[InspectionAlgorithm] | None = None) -> None:
        self.algorithms = list(algorithms or [ComparisonBasedAlgorithm()])
        if not self.algorithms:
            raise ValueError("at least one inspection algorithm is required")

    def inspect(self, context: InspectionContext) -> InspectResponse:
        return self.inspect_with_artifacts(context).response

    def inspect_with_artifacts(self, context: InspectionContext) -> InspectionExecution:
        """Run detectors and retain the aligned image used by their bboxes."""

        executions: list[AlgorithmExecution] = []
        algorithm_results: list[AlgorithmResult] = []
        for algorithm in self.algorithms:
            started_at = time.perf_counter()
            try:
                execution = algorithm.run(context)
                if isinstance(execution, AlgorithmResult):
                    execution = AlgorithmExecution(result=execution)
            except Exception as error:  # Keep future multi-algorithm fusion available.
                execution = AlgorithmExecution(
                    result=AlgorithmResult(
                        name=algorithm.name,
                        success=False,
                        elapsed_ms=_elapsed_ms(started_at),
                        error=f"{type(error).__name__}: {error}",
                    )
                )
            executions.append(execution)
            algorithm_results.append(execution.result)

        successful = [result for result in algorithm_results if result.success]
        if not successful:
            errors = "; ".join(
                f"{result.name}: {result.error}" for result in algorithm_results
            )
            raise RuntimeError(f"all inspection algorithms failed: {errors}")

        findings = fuse_findings(successful)
        review_execution = next(
            (
                execution
                for execution in executions
                if execution.result.success and execution.review_image is not None
            ),
            None,
        )
        review_image = (
            review_execution.review_image
            if review_execution is not None
            else cv2.resize(
                context.current,
                (1280, 720),
                interpolation=cv2.INTER_LINEAR,
            )
        )
        assert review_image is not None
        review_mask = (
            review_execution.review_mask
            if review_execution is not None
            and review_execution.review_mask is not None
            and review_execution.review_mask.shape == review_image.shape[:2]
            else np.zeros(review_image.shape[:2], dtype=np.uint8)
        )
        return InspectionExecution(
            response=InspectResponse(
                location_id=context.location_id,
                pose_type=context.pose_type,
                task_type=context.task_type,
                has_anomaly=bool(findings),
                image_size=[1280, 720],
                bbox_format=["x", "y", "width", "height"],
                findings=findings,
                algorithms=algorithm_results,
            ),
            review_image=review_image,
            review_mask=review_mask,
            review_homography=(
                review_execution.review_homography
                if review_execution is not None
                else None
            ),
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

    return inspect_images_with_artifacts(
        task_type,
        baseline,
        current,
        location_id=location_id,
        pose_type=pose_type,
        reference_item_area=reference_item_area,
        pipeline=pipeline,
    ).response


def inspect_images_with_artifacts(
    task_type: TaskType,
    baseline: np.ndarray,
    current: np.ndarray,
    *,
    location_id: str,
    pose_type: PoseType,
    reference_item_area: float | None = None,
    pipeline: InspectionPipeline | None = None,
) -> InspectionExecution:
    """Run inspection and retain the aligned current image for Qwen review."""

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
    return (pipeline or InspectionPipeline()).inspect_with_artifacts(context)


def build_row_constraints(
    findings: Sequence[Finding],
    row_detection: RowDetectionResult,
    pose_type: PoseType,
) -> list[ReviewRowConstraint | None]:
    """Map findings to reliable top-to-bottom shelf rows, otherwise fall back."""

    bboxes = [finding.bbox for finding in findings]
    expected_row_count = EXPECTED_ROW_COUNTS[pose_type]
    if pose_type in {"SHELF_VIEW_UPPER", "SHELF_VIEW_LOWER"}:
        matches = row_detection.match_bboxes_to_row_window(
            bboxes,
            row_count=expected_row_count,
            anchor="top" if pose_type == "SHELF_VIEW_UPPER" else "bottom",
            min_overlap_ratio=MIN_ROW_OVERLAP_RATIO,
        )
    else:
        matches = row_detection.match_bboxes(
            bboxes,
            expected_row_count=expected_row_count,
            min_overlap_ratio=MIN_ROW_OVERLAP_RATIO,
        )
    return [
        ReviewRowConstraint(
            row_index=match.row_index,
            row_bbox=match.row_bbox,
            overlap_ratio=match.overlap_ratio,
            detected_row_index=match.detected_row_index,
        )
        if match is not None
        else None
        for match in matches
    ]


def review_inspection_execution(
    execution: InspectionExecution,
    *,
    task_type: TaskType,
    location_id: str,
    pose_type: PoseType,
    baseline: np.ndarray,
    current_source: np.ndarray | None = None,
    baseline_depth_mm: np.ndarray | None = None,
    current_depth_mm: np.ndarray | None = None,
    reviewer: QwenReviewer | None = None,
) -> QwenReviewResult:
    """Apply the same row constraints and Qwen review used by the HTTP route."""

    result = execution.response
    if not result.findings:
        return QwenReviewResult((), "", ())
    try:
        row_detection = detect_rows(
            baseline,
            RowDetectionConfig(pose_type=pose_type),
        )
        row_constraints = build_row_constraints(
            result.findings,
            row_detection,
            pose_type,
        )
    except (ValueError, cv2.error) as error:
        logger.warning("Shelf row detection failed; using all SKU candidates: %s", error)
        row_constraints = [None] * len(result.findings)
    return (reviewer or QwenReviewer(debug_root=DEFAULT_DEBUG_ROOT)).review(
        task_type=task_type,
        location_id=location_id,
        pose_type=pose_type,
        current=execution.review_image,
        baseline=baseline,
        debug_current_rgb=current_source,
        debug_current_depth_mm=current_depth_mm,
        debug_baseline_rgb=baseline,
        debug_baseline_depth_mm=baseline_depth_mm,
        bboxes=[finding.bbox for finding in result.findings],
        row_constraints=row_constraints,
    )


def apply_shortage_depth_filter(
    execution: InspectionExecution,
    *,
    baseline: np.ndarray,
    baseline_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
) -> InspectionExecution:
    """Apply the shared shelf-range and aligned-depth filter used by batch runs."""

    if __package__ and __package__.startswith("perception."):
        from . import batch_shortage as shortage_depth
    else:
        import batch_shortage as shortage_depth

    filtered, _, summary, _ = shortage_depth.filter_execution_with_depth(
        execution,
        baseline_depth_mm,
        current_depth_mm,
        baseline,
    )
    logger.info(
        "Shortage shelf/depth filter: input=%s kept=%s shelf_rejected=%s "
        "closed_depth_recovered=%s",
        summary.get("input_findings"),
        summary.get("kept_findings"),
        summary.get("shelf_range_rejected_findings", 0),
        summary.get("closed_depth_recovery_findings", 0),
    )
    return filtered


@router.post("/perception/inspect", response_model=InspectApiResponse)
def inspect_shelf(request: InspectRequest) -> InspectApiResponse:
    """Load task0 reference RGB-D and compare it with a live head-camera capture."""

    try:
        baseline = load_initial_scan(request.location_id, request.pose_type)
    except InitialScanError as error:
        raise HTTPException(
            status_code=500,
            detail=f"读取 task0 初始 RGB-D 失败: {error}",
        ) from error

    try:
        with inspection_temporary_directory() as temporary_directory:
            current = capture_head_rgbd(temporary_directory)
            logger.info(
                "Inspection RGB-D captured: directory=%s valid_depth_pixels=%s",
                current.directory,
                int(np.count_nonzero(current.depth_mm > 0)),
            )
            if (
                request.task_type == "SHORTAGE"
                and hasattr(baseline, "inspection_target_id")
                and hasattr(baseline, "pose_type")
            ):
                return inspect_shortage_sam_images(
                    location_id=getattr(
                        baseline,
                        "inspection_target_id",
                        request.location_id,
                    ),
                    pose_type=getattr(baseline, "pose_type", request.pose_type),
                    baseline=baseline.rgb,
                    current=current.rgb,
                    baseline_depth_mm=baseline.depth_mm,
                    current_depth_mm=current.depth_mm,
                )
            return inspect_supplied_images(
                task_type=request.task_type,
                location_id=request.location_id,
                pose_type=request.pose_type,
                baseline=baseline.rgb,
                current=current.rgb,
                baseline_depth_mm=baseline.depth_mm,
                current_depth_mm=current.depth_mm,
                reference_item_area=request.reference_item_area,
            )
    except CameraCaptureError as error:
        raise HTTPException(
            status_code=502,
            detail=f"获取当前 head camera RGB-D 失败: {error}",
        ) from error
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"创建巡检临时目录失败: {error}",
        ) from error


def inspect_shortage_sam_images(
    *,
    location_id: str,
    pose_type: PoseType,
    baseline: np.ndarray,
    current: np.ndarray,
    baseline_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
) -> InspectApiResponse:
    """Run the production SAM3 slot comparison on in-memory RGB-D images."""

    try:
        analysis = analyze_shortage(
            location_id=location_id,
            pose_type=pose_type,
            baseline_rgb=baseline,
            baseline_depth_mm=baseline_depth_mm,
            current_rgb=current,
            current_depth_mm=current_depth_mm,
        )
    except SamShortageError as error:
        detail = str(error)
        status_code = 502 if detail.startswith("SAM3 ") else 500
        raise HTTPException(
            status_code=status_code,
            detail={
                "type": "sam_shortage_failed",
                "message": detail,
            },
        ) from error
    response = InspectApiResponse(
        findings=[
            ShortageProductFinding(shortage_product_name=name)
            for name in dict.fromkeys(analysis.missing_product_names)
            if name.strip()
        ]
    )
    try:
        save_sam_shortage_debug_artifacts(
            location_id=location_id,
            pose_type=pose_type,
            baseline=baseline,
            current=current,
            baseline_depth_mm=baseline_depth_mm,
            current_depth_mm=current_depth_mm,
            response=response,
            analysis=analysis_as_dict(analysis),
            shortage_analysis=analysis,
        )
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"保存 SAM3 shortage 巡检日志失败: {error}",
        ) from error
    return response


def save_sam_shortage_debug_artifacts(
    *,
    location_id: str,
    pose_type: PoseType,
    baseline: np.ndarray,
    current: np.ndarray,
    baseline_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    response: InspectApiResponse,
    analysis: dict[str, object],
    shortage_analysis: ShortageAnalysis,
) -> Path:
    """Persist the exact formal SHORTAGE RGB-D inputs and slot result."""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    safe_location = re.sub(r"[^A-Za-z0-9_.-]+", "_", location_id.strip())
    directory = DEFAULT_DEBUG_ROOT / (
        f"{timestamp}_{safe_location}_SHORTAGE_{uuid.uuid4().hex[:8]}"
    )
    directory.mkdir(parents=True, exist_ok=False)

    def write_image(name: str, image: np.ndarray) -> None:
        success, encoded = cv2.imencode(".jpg", np.asarray(image))
        if not success:
            raise OSError(f"无法编码巡检日志图片: {name}")
        (directory / name).write_bytes(encoded.tobytes())

    (directory / "request.json").write_text(
        json.dumps(
            {
                "task_type": "SHORTAGE",
                "location_id": location_id,
                "pose_type": pose_type,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_image("baseline_rgb.jpg", baseline)
    write_image("current_rgb.jpg", current)
    np.save(directory / "baseline_depth_mm.npy", baseline_depth_mm, allow_pickle=False)
    np.save(directory / "current_depth_mm.npy", current_depth_mm, allow_pickle=False)
    shelf_artifacts = save_shelf_preprocessing_artifacts(
        directory,
        shortage_analysis,
    )
    saved_result = response.model_dump(mode="json")
    saved_result["sam_shortage_analysis"] = analysis
    saved_result["artifacts"] = {
        "baseline_rgb": "baseline_rgb.jpg",
        "baseline_depth_mm": "baseline_depth_mm.npy",
        "current_rgb": "current_rgb.jpg",
        "current_depth_mm": "current_depth_mm.npy",
        "shelf_preprocessing": shelf_artifacts,
    }
    (directory / "result.json").write_text(
        json.dumps(saved_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return directory


def inspect_supplied_images(
    *,
    task_type: TaskType,
    location_id: str,
    pose_type: PoseType,
    baseline: np.ndarray,
    current: np.ndarray,
    baseline_depth_mm: np.ndarray | None = None,
    current_depth_mm: np.ndarray | None = None,
    reference_item_area: float | None = None,
) -> InspectApiResponse:
    """Run the public result pipeline on caller-supplied images for offline tests."""

    try:
        execution = inspect_images_with_artifacts(
            task_type,
            baseline,
            current,
            location_id=location_id,
            pose_type=pose_type,
            reference_item_area=reference_item_area,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    if (
        task_type == "SHORTAGE"
        and baseline_depth_mm is not None
        and current_depth_mm is not None
    ):
        try:
            execution = apply_shortage_depth_filter(
                execution,
                baseline=baseline,
                baseline_depth_mm=baseline_depth_mm,
                current_depth_mm=current_depth_mm,
            )
        except (RuntimeError, ValueError, cv2.error) as error:
            raise HTTPException(
                status_code=500,
                detail=f"货架范围/深度缺货筛选失败: {error}",
            ) from error
    if not execution.response.findings:
        return InspectApiResponse()

    try:
        review = review_inspection_execution(
            execution,
            task_type=task_type,
            location_id=location_id,
            pose_type=pose_type,
            baseline=baseline,
            current_source=current,
            baseline_depth_mm=baseline_depth_mm,
            current_depth_mm=current_depth_mm,
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
    return build_product_findings(task_type, review.findings)


def build_product_findings(
    task_type: TaskType,
    reviewed_findings: Sequence[ReviewedFinding],
) -> InspectApiResponse:
    """Convert validated Qwen findings into the deduplicated public contract."""

    if task_type == "SHORTAGE":
        seen_names: set[str] = set()
        findings: list[ShortageProductFinding] = []
        for finding in reviewed_findings:
            name = (finding.shortage_product_name or "").strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            findings.append(ShortageProductFinding(shortage_product_name=name))
        return InspectApiResponse(
            findings=findings
        )

    seen_pairs: set[tuple[str, str]] = set()
    findings: list[MisplacedProductFinding] = []
    for finding in reviewed_findings:
        misplaced_name = (finding.misplaced_product_name or "").strip()
        gt_name = (finding.gt_product_name or "").strip()
        key = (misplaced_name, gt_name)
        if not misplaced_name or not gt_name or key in seen_pairs:
            continue
        seen_pairs.add(key)
        findings.append(
            MisplacedProductFinding(
                misplaced_product_name=misplaced_name,
                gt_product_name=gt_name,
            )
        )
    return InspectApiResponse(
        findings=findings
    )


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
