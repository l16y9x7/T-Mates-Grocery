"""Gradio Tab 1: SAM3 分割 + 实例分色点云 GLB 预览."""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import get_sam3_conf  # noqa: E402
from ui.common import (  # noqa: E402
    clean_instance_id_map,
    clean_workspace_mask,
    depth_to_colormap,
    detection_summary,
    filepath_from_upload,
    load_camera_json_full,
    load_depth_mm,
    render_mask_bbox_previews,
    resolve_rgb_depth_alignment,
    shift_rgb_xy,
    vis_colors_rgb,
)
from ui.pointcloud import (  # noqa: E402
    build_instance_id_map,
    depth_instance_to_pointcloud,
    export_scene_files,
)

from scripts.sam3_seg import (  # noqa: E402
    DEFAULT_SAM3_API_URL,
    DEFAULT_SAM3_MASK_THRESHOLD,
    DEFAULT_SAM3_PROMPT,
    DEFAULT_SAM3_THRESHOLD,
    DEFAULT_SAM3_TIMEOUT_S,
    _decode_detection_mask,
    get_instance_bool_masks,
    run_sam3_segmentation,
)
from scripts import sam3_seg  # noqa: E402

OUTPUT_ROOT = ROOT_DIR / "output" / "ui_runs"
Sam3TabOut = Tuple[
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[str],
    Optional[str],
    Optional[str],
    str,
]


def _error(message: str, sensor_vis: Optional[Image.Image] = None) -> Sam3TabOut:
    err = json.dumps({"success": False, "message": message}, ensure_ascii=False, indent=2)
    return sensor_vis, None, None, None, None, None, err


def run_sam3_tab(
    rgb_img: Optional[Image.Image],
    depth_file: Any,
    camera_file: Any,
    prompt: str,
    api_url: str,
    threshold: float,
    mask_threshold: float,
    timeout_s: float,
    max_instances: int,
    rgb_shift_x: float = -45.0,
    rgb_shift_y: float = 0.0,
    enable_depth_align: bool = True,
    enable_workspace_outlier: bool = True,
    max_depth_mm: float = 2500,
    min_depth_mm: float = 50,
    depth_percentile_hi: float = 99.0,
    enable_instance_outlier: bool = True,
    sor_nb_neighbors: float = 20,
    sor_std_ratio: float = 1.5,
    z_mad_ratio: float = 2.5,
) -> Sam3TabOut:
    try:
        if rgb_img is None:
            return _error("请上传 RGB 图像")
        depth_path = filepath_from_upload(depth_file)
        camera_path = filepath_from_upload(camera_file)
        if depth_path is None or not depth_path.is_file():
            return _error("请上传深度文件（.npy / .png）")
        if camera_path is None or not camera_path.is_file():
            return _error("请上传 camera.json")

        prompt_text = (prompt or "").strip() or DEFAULT_SAM3_PROMPT
        depth_mm = load_depth_mm(depth_path)
        intrinsic, factor_depth, camera_meta = load_camera_json_full(camera_path)
        rgb = rgb_img.convert("RGB")
        if rgb.size != (depth_mm.shape[1], depth_mm.shape[0]):
            return _error(
                f"RGB 尺寸 {rgb.size} 与深度 {depth_mm.shape[1]}x{depth_mm.shape[0]} 不一致"
            )

        color_np = np.asarray(rgb, dtype=np.uint8)
        depth_mm, _color_for_cloud, align_meta = resolve_rgb_depth_alignment(
            depth_mm,
            color_np,
            camera_meta,
            ui_rgb_shift_x=float(rgb_shift_x or 0),
            ui_rgb_shift_y=float(rgb_shift_y or 0),
            enable_depth_align=bool(enable_depth_align),
            auto_if_zero=False,
        )
        rgb_shift_meta = dict(align_meta.get("rgb_shift") or {})
        depth_align_meta = dict(align_meta.get("depth_align") or {})

        workspace_stats: Dict[str, Any] = {"enabled": False}
        if bool(enable_workspace_outlier):
            ws_mask, workspace_stats = clean_workspace_mask(
                depth_mm,
                intrinsic,
                factor_depth=factor_depth,
                max_depth_mm=float(max_depth_mm or 0),
                min_depth_mm=float(min_depth_mm or 0),
                depth_percentile_hi=float(depth_percentile_hi or 0),
                remove_statistical_outliers=True,
                sor_nb_neighbors=int(sor_nb_neighbors or 20),
                sor_std_ratio=float(sor_std_ratio or 1.5),
            )
            workspace_stats["enabled"] = True
            depth_mm = depth_mm.copy()
            depth_mm[~ws_mask] = 0

        sensor_vis = depth_to_colormap(depth_mm)
        run_dir = OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S") / f"sam3_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = run_dir / "rgb.png"
        rgb.save(rgb_path)

        sam3_seg.DEFAULT_SAM3_API_URL = (api_url or DEFAULT_SAM3_API_URL).strip()
        sam3_seg.DEFAULT_SAM3_TIMEOUT_S = float(timeout_s or DEFAULT_SAM3_TIMEOUT_S)

        t0 = time.perf_counter()
        result = run_sam3_segmentation(
            rgb_path,
            run_dir / "sam3_results",
            prompt=prompt_text,
            threshold=float(threshold),
            mask_threshold=float(mask_threshold),
            max_instances=int(max_instances) if max_instances else 0,
        )
        elapsed = time.perf_counter() - t0

        dets = result.instance_dets or []
        if not dets:
            return _error("SAM3 未返回实例", sensor_vis)

        mask_vis, bbox_vis = render_mask_bbox_previews(
            rgb, dets, _decode_detection_mask, prompt=prompt_text
        )
        masks = get_instance_bool_masks(dets, rgb.size)
        id_map = build_instance_id_map(masks, depth_mm.shape[0], depth_mm.shape[1])

        # Depth 已对齐到 RGB 时，mask 与 depth 同网格，无需再平移 id_map；
        # 仅在关闭对齐、走历史上色偏移时，把 id_map 与 depth 对齐到同一约定。
        if not bool(enable_depth_align):
            sx = int(rgb_shift_meta.get("dx") or 0)
            sy = int(rgb_shift_meta.get("dy") or 0)
            if sx != 0 or sy != 0:
                id_map = shift_rgb_xy(id_map, sx, sy, nearest=True)

        # 全局剔除后的无效深度上不再保留实例标签
        id_map = id_map.copy()
        id_map[depth_mm <= 0] = 0

        instance_outlier_stats: Dict[str, Any] = {"enabled": False}
        if bool(enable_instance_outlier):
            id_map, _removed, instance_outlier_stats = clean_instance_id_map(
                depth_mm,
                id_map,
                intrinsic,
                factor_depth=factor_depth,
                std_ratio=float(sor_std_ratio or 1.5),
                z_mad_ratio=float(z_mad_ratio or 2.5),
                sor_nb_neighbors=int(sor_nb_neighbors or 20),
            )
            if not np.any(id_map > 0):
                return _error("离群点剔除后无有效实例点，请放宽参数", sensor_vis)

        points, colors, pc_stats = depth_instance_to_pointcloud(
            depth_mm,
            id_map,
            intrinsic,
            vis_colors_rgb(),
            factor_depth=factor_depth,
        )
        files = export_scene_files(points, colors, run_dir, stem="sam3_instances")
        glb = str(files["glb_path"])
        ply = str(files["ply_path"])

        payload = {
            "success": True,
            "elapsed_s": round(elapsed, 3),
            "num_instances": int(len(np.unique(id_map[id_map > 0]))),
            "detections": detection_summary(dets),
            "pointcloud": {
                **pc_stats,
                "frame": "camera",
                "preview_frame": "glb_y_up",
                "unit": "m",
            },
            "workspace_outlier_filter": workspace_stats,
            "instance_outlier_filter": instance_outlier_stats,
            "rgb_shift": rgb_shift_meta,
            "depth_align": depth_align_meta,
            "scene_glb": glb,
            "scene_ply": ply,
            "sam3_api": sam3_seg.DEFAULT_SAM3_API_URL,
            "prompt": prompt_text,
            "threshold": float(threshold),
            "mask_threshold": float(mask_threshold),
            "run_dir": str(run_dir),
        }
        return (
            sensor_vis,
            mask_vis,
            bbox_vis,
            glb,
            glb,
            ply,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc))


def build_sam3_tab() -> None:
    cfg = get_sam3_conf()
    with gr.Tab("SAM3 分割"):
        gr.Markdown(
            "验证 **SAM3 文本分割** + **传感器深度点云**（实例分色）。"
            "快速预览拆成 **mask** / **bbox** 两张图；3D 用浏览器 GLB 预览。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                rgb = gr.Image(type="pil", label="上传 RGB", height=280, interactive=True)
                with gr.Row():
                    depth = gr.File(
                        label="深度（.npy / uint16 .png）",
                        type="filepath",
                        file_types=[".npy", ".png", ".tif", ".tiff"],
                    )
                    camera = gr.File(
                        label="相机 camera.json",
                        type="filepath",
                        file_types=[".json"],
                    )
                enable_depth_align = gr.Checkbox(
                    label="对齐 Depth→RGB（推荐：修正像素/3D 偏差；默认开启）",
                    value=True,
                )
                with gr.Row():
                    rgb_shift_x = gr.Number(
                        label="RGB↔Depth 偏移 dx（默认 -45→Depth 右移 45）",
                        value=-45,
                        precision=0,
                    )
                    rgb_shift_y = gr.Number(
                        label="RGB↔Depth 偏移 dy（+下）",
                        value=0,
                        precision=0,
                    )
                prompt = gr.Textbox(
                    label="实例分割提示词",
                    value=str(cfg.get("default_prompt") or DEFAULT_SAM3_PROMPT),
                    lines=3,
                )
                with gr.Accordion("SAM3 推理参数", open=True):
                    api = gr.Textbox(
                        label="SAM3 API URL",
                        value=str(cfg.get("api_url") or DEFAULT_SAM3_API_URL),
                    )
                    thr = gr.Slider(
                        label="Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=float(cfg.get("threshold", DEFAULT_SAM3_THRESHOLD)),
                    )
                    mthr = gr.Slider(
                        label="Mask Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=float(cfg.get("mask_threshold", DEFAULT_SAM3_MASK_THRESHOLD)),
                    )
                    timeout = gr.Number(
                        label="超时（秒）",
                        value=float(cfg.get("timeout_s", DEFAULT_SAM3_TIMEOUT_S)),
                        precision=0,
                    )
                    max_inst = gr.Number(label="max_instances（0=全部）", value=5, precision=0)
                with gr.Accordion("点云离群点剔除", open=True):
                    enable_workspace = gr.Checkbox(
                        label="① 全局：整幅深度点云离群点剔除（深度范围 + SOR）",
                        value=True,
                    )
                    with gr.Row():
                        max_depth = gr.Number(label="max_depth_mm（0=不限制）", value=2500, precision=0)
                        min_depth = gr.Number(label="min_depth_mm", value=50, precision=0)
                        depth_pct = gr.Number(label="depth_percentile_hi（0=关）", value=99.0)
                    enable_instance = gr.Checkbox(
                        label="② 按实例：各 SAM3 实例独立剔除（median/MAD + SOR）",
                        value=True,
                    )
                    sor_k = gr.Number(label="SOR 邻域点数 nb_neighbors", value=20, precision=0)
                    sor_std = gr.Number(
                        label="SOR / 空间 std_ratio（越小越狠，常用 1.0~2.0）",
                        value=1.5,
                    )
                    z_mad = gr.Number(
                        label="实例深度 z_mad_ratio（MAD 倍数）",
                        value=2.5,
                    )
                btn = gr.Button("开始分割", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("#### 快速预览")
                out_depth = gr.Image(type="pil", label="传感器深度（伪彩色）", height=200)
                out_mask = gr.Image(type="pil", label="SAM3 实例 mask", height=200)
                out_bbox = gr.Image(type="pil", label="SAM3 实例 bbox", height=200)

        gr.Markdown("### 3D 点云")
        gr.Markdown(
            "> **灰色**=背景；**彩色**=各 SAM3 实例。"
            "**数值坐标系 `frame=camera`（X右 Y下 Z前）**；"
            "**3D 预览 `preview_frame=glb_y_up`（仅翻 Y）**。"
        )
        with gr.Row():
            with gr.Column(scale=4):
                out_glb = gr.Model3D(label="浏览器预览（GLB）", height=520)
            with gr.Column(scale=1):
                gr.Markdown("**下载**")
                out_glb_dl = gr.File(label="GLB", interactive=False)
                out_ply_dl = gr.File(label="PLY", interactive=False)
        out_json = gr.Code(label="分割 / 点云详情", language="json", lines=16)

        btn.click(
            fn=run_sam3_tab,
            inputs=[
                rgb,
                depth,
                camera,
                prompt,
                api,
                thr,
                mthr,
                timeout,
                max_inst,
                rgb_shift_x,
                rgb_shift_y,
                enable_depth_align,
                enable_workspace,
                max_depth,
                min_depth,
                depth_pct,
                enable_instance,
                sor_k,
                sor_std,
                z_mad,
            ],
            outputs=[out_depth, out_mask, out_bbox, out_glb, out_glb_dl, out_ply_dl, out_json],
        )

        gr.Markdown(
            f"""
            **说明**
            - 实例分割：SAM3 `POST /infer`（`image_base64`），默认 API `{cfg.get('api_url') or DEFAULT_SAM3_API_URL}`
            - 点云：先 **全局** 深度/SOR 剔除，再按 **实例** 做 MAD+SOR，最后导出 GLB
            - **Depth→RGB 对齐**默认开启；`dx=-45` 时 Depth 右移 45 对齐到 RGB（也可用 `camera.json` 的 `depth_to_rgb_shift` / `rgb_shift`）
            - 配置见 `config/conf.json`
            """
        )
