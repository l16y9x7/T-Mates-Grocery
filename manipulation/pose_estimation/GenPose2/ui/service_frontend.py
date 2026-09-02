"""Independent Gradio frontend that calls SAM3 and GenPose2 HTTP services."""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np
import requests
from PIL import Image, ImageDraw

from ui.common import filepath_from_upload, load_depth_mm
from ui.service_client import (
    ServiceCallError,
    ServiceResult,
    call_pose_service,
    call_sam3_box,
    decode_coco_rle,
    normalize_box,
    resolve_camera,
    select_best_detection,
    write_camera_json,
    write_mask_png,
)
from ui.service_visualization import (
    export_pose_scene,
    render_box,
    render_depth_colormap,
    render_mask,
    render_mask_overlay,
    render_pose_overlay,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "service_outputs" / "frontend_runs"
DEFAULT_SAM3_URL = os.environ.get(
    "SAM3_URL", "http://127.0.0.1:18003/infer"
)
DEFAULT_POSE_URL = "http://127.0.0.1:8084/manipulation/pick_pose"

Sam3Caller = Callable[..., ServiceResult]
PoseCaller = Callable[..., ServiceResult]


@dataclass(frozen=True)
class BoxState:
    """Two-click box interaction state."""

    first: Optional[Tuple[int, int]] = None
    box: Optional[List[int]] = None
    status: str = "empty"


@dataclass
class PipelineState:
    """Artifacts and responses accumulated across independent stages."""

    run_dir: Optional[str] = None
    rgb_path: Optional[str] = None
    depth_path: Optional[str] = None
    camera_path: Optional[str] = None
    mask_path: Optional[str] = None
    sam3_payload: Optional[Dict[str, Any]] = None
    pose_payload: Optional[Dict[str, Any]] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


def advance_box_click(
    current: Optional[BoxState],
    click: Tuple[int, int],
    image_size: Tuple[int, int],
) -> BoxState:
    """Use one click as a corner and the next click to complete a box."""

    x = int(round(click[0]))
    y = int(round(click[1]))
    width, height = image_size
    point = (
        min(max(x, 0), width - 1),
        min(max(y, 0), height - 1),
    )
    if current is None or current.status in ("empty", "complete"):
        return BoxState(first=point, status="awaiting_second")
    if current.first is None:
        return BoxState(first=point, status="awaiting_second")
    return BoxState(
        first=current.first,
        box=normalize_box([*current.first, *point], image_size),
        status="complete",
    )


def clear_box() -> BoxState:
    """Return a fresh two-click state."""

    return BoxState()


def _new_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(output_root) / f"{timestamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(payload: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def _error_message(stage: str, error: Exception) -> str:
    if isinstance(error, ServiceCallError):
        detail = json.dumps(error.payload, ensure_ascii=False)
        return f"{stage}失败（HTTP {error.status_code}）：{detail}"
    return f"{stage}失败：{error}"


def _refresh_total(state: PipelineState) -> None:
    state.timings_ms["total_ms"] = sum(
        value
        for key, value in state.timings_ms.items()
        if key != "total_ms"
    )


def run_sam3_stage(
    rgb_path: Path,
    box: Sequence[int],
    sam3_url: str,
    timeout_s: float,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    sam3_caller: Sam3Caller = call_sam3_box,
) -> PipelineState:
    """Create a run and execute only the remote SAM3 box stage."""

    state = PipelineState()
    preparation_started = time.perf_counter()
    try:
        source = Path(rgb_path)
        if not source.is_file():
            raise FileNotFoundError(f"RGB image not found: {source}")
        run_dir = _new_run_dir(Path(output_root))
        rgb = Image.open(source).convert("RGB")
        effective_box = normalize_box(box, rgb.size)
        saved_rgb = run_dir / "rgb.png"
        rgb.save(saved_rgb)
        box_path = run_dir / "box.png"
        render_box(rgb, effective_box).save(box_path)
        state.run_dir = str(run_dir)
        state.rgb_path = str(saved_rgb)
        state.artifacts["box"] = str(box_path)
        state.timings_ms["input_prepare_ms"] = (
            time.perf_counter() - preparation_started
        ) * 1000.0

        result = sam3_caller(
            sam3_url,
            saved_rgb,
            effective_box,
            timeout_s=float(timeout_s),
        )
        state.timings_ms["sam3_http_ms"] = result.elapsed_ms
        state.sam3_payload = result.payload
        raw_sam3_path = _write_json(result.payload, run_dir / "sam3_response.json")
        state.artifacts["sam3_response"] = str(raw_sam3_path)

        decode_started = time.perf_counter()
        selected = select_best_detection(result.payload)
        mask = decode_coco_rle(selected["segmentation"])
        if mask.shape != (rgb.height, rgb.width):
            raise ValueError(
                f"SAM3 mask shape {mask.shape} does not match RGB "
                f"{(rgb.height, rgb.width)}"
            )
        mask_path = write_mask_png(mask, run_dir / "mask.png")
        mask_preview_path = run_dir / "mask_preview.png"
        render_mask(mask).save(mask_preview_path)
        overlay_path = run_dir / "mask_overlay.png"
        render_mask_overlay(rgb, mask).save(overlay_path)
        state.mask_path = str(mask_path)
        state.artifacts.update(
            {
                "mask": str(mask_preview_path),
                "mask_overlay": str(overlay_path),
            }
        )
        state.timings_ms["mask_decode_ms"] = (
            time.perf_counter() - decode_started
        ) * 1000.0
        state.error = None
    except (ServiceCallError, OSError, ValueError, TypeError, RuntimeError) as error:
        if isinstance(error, ServiceCallError):
            state.timings_ms["sam3_http_ms"] = error.elapsed_ms
        state.error = _error_message("SAM3", error)
    _refresh_total(state)
    return state


def run_pose_stage(
    previous: PipelineState,
    depth_path: Path,
    camera_path: Optional[Path],
    *,
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
    depth_scale: Optional[float],
    pose_url: str,
    timeout_s: float,
    pose_caller: PoseCaller = call_pose_service,
) -> PipelineState:
    """Execute the remote pose stage and retain prior SAM3 artifacts on failure."""

    state = copy.deepcopy(previous)
    state.error = None
    preparation_started = time.perf_counter()
    try:
        if not state.run_dir or not state.rgb_path or not state.mask_path:
            raise ValueError("请先成功运行 SAM3，生成当前掩码")
        run_dir = Path(state.run_dir)
        source_depth = Path(depth_path)
        if not source_depth.is_file():
            raise FileNotFoundError(f"depth not found: {source_depth}")
        camera = resolve_camera(
            Path(camera_path) if camera_path is not None else None,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            depth_scale=depth_scale,
        )
        saved_depth = run_dir / f"depth{source_depth.suffix.lower() or '.png'}"
        if source_depth.resolve() != saved_depth.resolve():
            shutil.copy2(source_depth, saved_depth)
        effective_camera = write_camera_json(camera, run_dir / "camera.json")
        state.depth_path = str(saved_depth)
        state.camera_path = str(effective_camera)
        state.timings_ms["pose_input_prepare_ms"] = (
            time.perf_counter() - preparation_started
        ) * 1000.0

        result = pose_caller(
            pose_url,
            Path(state.rgb_path),
            saved_depth,
            effective_camera,
            Path(state.mask_path),
            timeout_s=float(timeout_s),
        )
        state.timings_ms["pose_http_ms"] = result.elapsed_ms
        state.pose_payload = result.payload
        raw_pose_path = _write_json(result.payload, run_dir / "pose_response.json")
        state.artifacts["pose_response"] = str(raw_pose_path)

        rgb = Image.open(state.rgb_path).convert("RGB")
        depth_mm = load_depth_mm(saved_depth)
        render_started = time.perf_counter()
        depth_preview = run_dir / "depth_colormap.png"
        render_depth_colormap(depth_mm).save(depth_preview)
        pose_preview = run_dir / "pose_overlay.png"
        render_pose_overlay(rgb, result.payload, camera.intrinsics).save(pose_preview)
        state.artifacts.update(
            {
                "depth_colormap": str(depth_preview),
                "pose_overlay": str(pose_preview),
            }
        )
        state.timings_ms["visualization_2d_ms"] = (
            time.perf_counter() - render_started
        ) * 1000.0

        pointcloud_started = time.perf_counter()
        scene = export_pose_scene(
            np.asarray(rgb),
            depth_mm,
            camera.intrinsics,
            result.payload,
            run_dir,
        )
        state.artifacts["scene_glb"] = str(scene.glb_path)
        state.artifacts["scene_ply"] = str(scene.ply_path)
        state.timings_ms["pointcloud_ms"] = (
            time.perf_counter() - pointcloud_started
        ) * 1000.0
        state.error = None
    except (ServiceCallError, OSError, ValueError, TypeError, RuntimeError) as error:
        if isinstance(error, ServiceCallError):
            state.timings_ms["pose_http_ms"] = error.elapsed_ms
        state.error = _error_message("位姿估计", error)
    _refresh_total(state)
    return state


def run_full_pipeline(
    rgb_path: Path,
    depth_path: Path,
    camera_path: Optional[Path],
    box: Sequence[int],
    *,
    sam3_url: str,
    pose_url: str,
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
    depth_scale: Optional[float],
    sam3_timeout_s: float,
    pose_timeout_s: float,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    sam3_caller: Sam3Caller = call_sam3_box,
    pose_caller: PoseCaller = call_pose_service,
) -> PipelineState:
    """Execute SAM3 then pose, stopping after the first failed stage."""

    state = run_sam3_stage(
        rgb_path,
        box,
        sam3_url,
        sam3_timeout_s,
        output_root,
        sam3_caller=sam3_caller,
    )
    if state.error is not None:
        return state
    return run_pose_stage(
        state,
        depth_path,
        camera_path,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        depth_scale=depth_scale,
        pose_url=pose_url,
        timeout_s=pose_timeout_s,
        pose_caller=pose_caller,
    )


def _box_instruction(state: BoxState) -> str:
    if state.status == "awaiting_second" and state.first is not None:
        return f"**已选第一角 {state.first}。请点击矩形对角。**"
    if state.status == "complete" and state.box is not None:
        return f"**矩形已完成：{state.box}。可运行推理；再次点击会开始新框。**"
    return "**请在图像上点击矩形第一个角。**"


def _first_click_preview(image: Image.Image, point: Tuple[int, int]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    x, y = point
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 190, 0))
    return output


def _health_url(service_url: str) -> str:
    parsed = urlsplit(service_url.strip())
    path = "/manipulation/health" if "/manipulation/" in parsed.path else "/health"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def check_service_health(service_url: str, timeout_s: float = 3.0) -> str:
    """Check one backend without starting or restarting it."""

    target = _health_url(service_url)
    started = time.perf_counter()
    try:
        response = requests.get(target, timeout=float(timeout_s))
        elapsed = (time.perf_counter() - started) * 1000.0
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text[:300]
        if response.ok:
            return f"✅ `{target}` 可用，{elapsed:.1f} ms\n\n```json\n{json.dumps(body, ensure_ascii=False, indent=2)}\n```"
        return f"❌ `{target}` 返回 HTTP {response.status_code}：`{body}`"
    except requests.RequestException as error:
        return f"❌ `{target}` 不可用：`{error}`。前端仍保持运行，不会自动启动后端。"


def _timing_rows(state: PipelineState) -> List[List[Any]]:
    order = (
        "input_prepare_ms",
        "sam3_http_ms",
        "mask_decode_ms",
        "pose_input_prepare_ms",
        "pose_http_ms",
        "visualization_2d_ms",
        "pointcloud_ms",
        "total_ms",
    )
    labels = {
        "input_prepare_ms": "输入准备",
        "sam3_http_ms": "SAM3 HTTP",
        "mask_decode_ms": "掩码解码",
        "pose_input_prepare_ms": "位姿输入准备",
        "pose_http_ms": "GenPose2 HTTP",
        "visualization_2d_ms": "2D 可视化",
        "pointcloud_ms": "点云/GLB",
        "total_ms": "总耗时",
    }
    return [
        [labels[key], round(state.timings_ms[key], 3)]
        for key in order
        if key in state.timings_ms
    ]


def _downloads(state: PipelineState) -> List[str]:
    preferred = (
        "mask_overlay",
        "depth_colormap",
        "pose_overlay",
        "scene_glb",
        "scene_ply",
        "sam3_response",
        "pose_response",
    )
    return [state.artifacts[key] for key in preferred if key in state.artifacts]


def _state_outputs(state: PipelineState) -> Tuple[Any, ...]:
    status = f"❌ {state.error}" if state.error else "✅ 当前已完成的流程输出如下。"
    return (
        state,
        state.artifacts.get("mask"),
        state.artifacts.get("mask_overlay"),
        state.artifacts.get("depth_colormap"),
        state.artifacts.get("pose_overlay"),
        state.artifacts.get("scene_glb"),
        state.sam3_payload,
        state.pose_payload,
        _timing_rows(state),
        _downloads(state),
        status,
    )


def _empty_state_outputs() -> Tuple[Any, ...]:
    return _state_outputs(PipelineState())


def _ui_load_rgb(rgb_value: Any) -> Tuple[Any, ...]:
    path = filepath_from_upload(rgb_value)
    if path is None:
        return (None, clear_box(), _box_instruction(clear_box()), *_empty_state_outputs())
    image = Image.open(path).convert("RGB")
    state = clear_box()
    return (image, state, _box_instruction(state), *_empty_state_outputs())


def _ui_box_click(
    current: Optional[BoxState],
    rgb_value: Any,
    event: gr.SelectData,
) -> Tuple[Any, ...]:
    path = filepath_from_upload(rgb_value)
    if path is None:
        raise gr.Error("请先上传 RGB 图像")
    image = Image.open(path).convert("RGB")
    index = event.index
    if not isinstance(index, (list, tuple)) or len(index) < 2:
        raise gr.Error("没有获得有效图像坐标，请重新点击")
    next_state = advance_box_click(
        current,
        (int(index[0]), int(index[1])),
        image.size,
    )
    if next_state.box is not None:
        preview = render_box(image, next_state.box)
    elif next_state.first is not None:
        preview = _first_click_preview(image, next_state.first)
    else:
        preview = image
    return (
        next_state,
        preview,
        _box_instruction(next_state),
        *_empty_state_outputs(),
    )


def _ui_clear(rgb_value: Any) -> Tuple[Any, ...]:
    path = filepath_from_upload(rgb_value)
    image = Image.open(path).convert("RGB") if path is not None else None
    state = clear_box()
    return (state, image, _box_instruction(state), *_empty_state_outputs())


def _require_path(value: Any, label: str) -> Path:
    path = filepath_from_upload(value)
    if path is None:
        raise ValueError(f"请提供{label}")
    return path


def _optional_float(value: Any) -> Optional[float]:
    """Parse a blank Gradio textbox as an omitted manual camera value."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return float(value)


def _ui_run_sam3(
    rgb_value: Any,
    box_state: Optional[BoxState],
    sam3_url: str,
    sam3_timeout: float,
) -> Tuple[Any, ...]:
    try:
        rgb_path = _require_path(rgb_value, " RGB 图像")
        if box_state is None or box_state.box is None:
            raise ValueError("请先用两次点击完成矩形框")
        state = run_sam3_stage(
            rgb_path,
            box_state.box,
            sam3_url,
            sam3_timeout,
        )
    except (OSError, ValueError, TypeError) as error:
        state = PipelineState(error=str(error))
    return _state_outputs(state)


def _ui_run_pose(
    pipeline: Optional[PipelineState],
    depth_value: Any,
    camera_value: Any,
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
    depth_scale: Optional[float],
    pose_url: str,
    pose_timeout: float,
) -> Tuple[Any, ...]:
    try:
        depth_path = _require_path(depth_value, "深度文件")
        camera_path = filepath_from_upload(camera_value)
        state = run_pose_stage(
            pipeline or PipelineState(),
            depth_path,
            camera_path,
            fx=_optional_float(fx),
            fy=_optional_float(fy),
            cx=_optional_float(cx),
            cy=_optional_float(cy),
            depth_scale=_optional_float(depth_scale),
            pose_url=pose_url,
            timeout_s=pose_timeout,
        )
    except (OSError, ValueError, TypeError) as error:
        state = copy.deepcopy(pipeline or PipelineState())
        state.error = str(error)
    return _state_outputs(state)


def _ui_run_full(
    rgb_value: Any,
    depth_value: Any,
    camera_value: Any,
    box_state: Optional[BoxState],
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
    depth_scale: Optional[float],
    sam3_url: str,
    pose_url: str,
    sam3_timeout: float,
    pose_timeout: float,
) -> Tuple[Any, ...]:
    try:
        rgb_path = _require_path(rgb_value, " RGB 图像")
        depth_path = _require_path(depth_value, "深度文件")
        if box_state is None or box_state.box is None:
            raise ValueError("请先用两次点击完成矩形框")
        state = run_full_pipeline(
            rgb_path,
            depth_path,
            filepath_from_upload(camera_value),
            box_state.box,
            sam3_url=sam3_url,
            pose_url=pose_url,
            fx=_optional_float(fx),
            fy=_optional_float(fy),
            cx=_optional_float(cx),
            cy=_optional_float(cy),
            depth_scale=_optional_float(depth_scale),
            sam3_timeout_s=sam3_timeout,
            pose_timeout_s=pose_timeout,
        )
    except (OSError, ValueError, TypeError) as error:
        state = PipelineState(error=str(error))
    return _state_outputs(state)


def build_service_frontend() -> gr.Blocks:
    """Build the model-free service test application."""

    with gr.Blocks(title="RGB-D 位姿服务测试台") as app:
        gr.Markdown(
            "# RGB-D 位姿服务测试台\n"
            "独立调用现有 SAM3 与 GenPose2 HTTP 服务；本页面不加载模型，"
            "也不会自动启动、停止或重启后端。"
        )
        box_state = gr.State(clear_box())
        pipeline_state = gr.State(PipelineState())

        with gr.Row():
            sam3_url = gr.Textbox(label="SAM3 URL", value=DEFAULT_SAM3_URL, scale=4)
            sam3_timeout = gr.Number(label="SAM3 超时（秒）", value=120, minimum=1, scale=1)
            sam3_health = gr.Button("检查 SAM3", scale=1)
        with gr.Row():
            pose_url = gr.Textbox(label="位姿 URL（可改为 place_pose）", value=DEFAULT_POSE_URL, scale=4)
            pose_timeout = gr.Number(label="位姿超时（秒）", value=300, minimum=1, scale=1)
            pose_health = gr.Button("检查位姿服务", scale=1)
        health_status = gr.Markdown("尚未检查后端。")
        sam3_health.click(check_service_health, inputs=sam3_url, outputs=health_status)
        pose_health.click(check_service_health, inputs=pose_url, outputs=health_status)

        with gr.Row():
            with gr.Column(scale=1):
                rgb_input = gr.Image(label="RGB", type="filepath")
                depth_input = gr.File(
                    label="Depth（PNG/TIFF/NPY）",
                    file_types=[".png", ".tif", ".tiff", ".npy"],
                )
                camera_input = gr.File(label="camera.json（手工 K 完整填写时可不传）", file_types=[".json"])
            with gr.Column(scale=2):
                box_canvas = gr.Image(label="两次点击画矩形框", type="pil", interactive=True)
                box_instruction = gr.Markdown(_box_instruction(clear_box()))
                clear_button = gr.Button("清空矩形与结果")

        gr.Markdown("手工参数只要填写任意一项，就必须完整填写五项，并覆盖上传的 camera.json。")
        with gr.Row():
            fx_input = gr.Textbox(label="fx", value="", placeholder="例如 600.0")
            fy_input = gr.Textbox(label="fy", value="", placeholder="例如 600.0")
            cx_input = gr.Textbox(label="cx", value="", placeholder="例如 320.0")
            cy_input = gr.Textbox(label="cy", value="", placeholder="例如 240.0")
            scale_input = gr.Textbox(
                label="depth_scale", value="", placeholder="例如 0.001"
            )

        with gr.Row():
            sam3_button = gr.Button("仅运行 SAM3", variant="secondary")
            pose_button = gr.Button("使用当前掩码运行位姿", variant="secondary")
            full_button = gr.Button("运行完整流程", variant="primary")

        status = gr.Markdown("等待输入。")
        with gr.Tabs():
            with gr.Tab("2D 结果"):
                with gr.Row():
                    mask_view = gr.Image(label="SAM3 掩码")
                    mask_overlay = gr.Image(label="RGB + 掩码")
                with gr.Row():
                    depth_view = gr.Image(label="深度伪彩色")
                    pose_view = gr.Image(label="位姿轴 + corners_mm 立方体")
            with gr.Tab("3D 点云"):
                model_view = gr.Model3D(label="点云 + 位姿轴 + 立方体", height=560)
            with gr.Tab("原始响应"):
                with gr.Row():
                    sam3_json = gr.JSON(label="SAM3 JSON")
                    pose_json = gr.JSON(label="位姿 JSON")
            with gr.Tab("耗时与下载"):
                timing_table = gr.Dataframe(
                    headers=["流程", "耗时 (ms)"],
                    datatype=["str", "number"],
                    interactive=False,
                    label="每次推理流程耗时",
                )
                downloads = gr.File(label="下载本次推理产物", file_count="multiple")

        stage_outputs = [
            pipeline_state,
            mask_view,
            mask_overlay,
            depth_view,
            pose_view,
            model_view,
            sam3_json,
            pose_json,
            timing_table,
            downloads,
            status,
        ]
        reset_outputs = [box_canvas, box_state, box_instruction, *stage_outputs]
        rgb_input.change(_ui_load_rgb, inputs=rgb_input, outputs=reset_outputs)
        box_canvas.select(
            _ui_box_click,
            inputs=[box_state, rgb_input],
            outputs=[box_state, box_canvas, box_instruction, *stage_outputs],
        )
        clear_button.click(
            _ui_clear,
            inputs=rgb_input,
            outputs=[box_state, box_canvas, box_instruction, *stage_outputs],
        )
        sam3_button.click(
            _ui_run_sam3,
            inputs=[rgb_input, box_state, sam3_url, sam3_timeout],
            outputs=stage_outputs,
        )
        pose_button.click(
            _ui_run_pose,
            inputs=[
                pipeline_state,
                depth_input,
                camera_input,
                fx_input,
                fy_input,
                cx_input,
                cy_input,
                scale_input,
                pose_url,
                pose_timeout,
            ],
            outputs=stage_outputs,
        )
        full_button.click(
            _ui_run_full,
            inputs=[
                rgb_input,
                depth_input,
                camera_input,
                box_state,
                fx_input,
                fy_input,
                cx_input,
                cy_input,
                scale_input,
                sam3_url,
                pose_url,
                sam3_timeout,
                pose_timeout,
            ],
            outputs=stage_outputs,
        )

    return app
