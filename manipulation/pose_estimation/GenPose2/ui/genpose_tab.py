"""Gradio Tab 2: SAM3 分割 + GenPose2 6D 位姿估计."""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (  # noqa: E402
    get_genpose2_conf,
    get_sam3_conf,
    get_vlm_profile,
    resolve_repo_path,
)
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
)
from ui.genpose_runner import (  # noqa: E402
    build_grasp_display_payload,
    build_poses_payload,
    camera_json_to_meta,
    export_pose_scene,
    load_mask_u8,
    run_genpose2_from_mask,
)
from ui.pointcloud import GRASP_CLOUD_MAX_POINTS  # noqa: E402

from scripts.sam3_seg import (  # noqa: E402
    DEFAULT_SAM3_API_URL,
    DEFAULT_SAM3_MASK_THRESHOLD,
    DEFAULT_SAM3_PROMPT,
    DEFAULT_SAM3_THRESHOLD,
    DEFAULT_SAM3_TIMEOUT_S,
    _decode_detection_mask,
    run_sam3_segmentation,
)
from scripts import sam3_seg  # noqa: E402
from scripts.vlm_prompt import (  # noqa: E402
    DEFAULT_SAM3_VLM_API_URL,
    DEFAULT_SAM3_VLM_MODEL,
    DEFAULT_SAM3_VLM_TIMEOUT_S,
    generate_sam3_prompt_from_image,
)

OUTPUT_ROOT = ROOT_DIR / "output" / "ui_runs"
logger = logging.getLogger("genpose_tab")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# sensor, mask, bbox, pose_overlay,
# glb, glb_dl, ply_dl,
# grasp_json, poses_json, detail_json
GenPoseTabOut = Tuple[
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[str],
    Optional[str],
    Optional[str],
    str,
    str,
    str,
]


def _empty_poses(message: str = "尚未计算出位姿") -> str:
    return json.dumps(
        {
            "success": False,
            "message": message,
            "xyzrxryrz": None,
            "frame": "camera",
            "preview_frame": "glb_y_up",
            "unit": {"xyz": "mm", "rx_ry_rz": "rad", "size_3d": "m"},
            "poses": [],
            "num_poses": 0,
        },
        ensure_ascii=False,
        indent=2,
    )


def _empty_grasp_json(message: str = "尚未计算出抓取位姿") -> str:
    return json.dumps(
        {
            "success": False,
            "message": message,
            "xyzrxryrz": None,
            "cube": None,
            "unit": {
                "xyz": "mm",
                "rx_ry_rz": "deg",
                "size_3d": "m",
                "size_3d_mm": "mm",
                "cube_corners_mm": "mm",
                "euler": "ZYX → [rx,ry,rz]=[X,Y,Z]",
            },
            "instances": [],
            "num_poses": 0,
        },
        ensure_ascii=False,
        indent=2,
    )


def _error(message: str, sensor_vis: Optional[Image.Image] = None) -> GenPoseTabOut:
    logger.error("genpose_tab failed: %s", message)
    err = json.dumps({"success": False, "message": message}, ensure_ascii=False, indent=2)
    return (
        sensor_vis,
        None,
        None,
        None,
        None,
        None,
        None,
        _empty_grasp_json(message),
        _empty_poses(message),
        err,
    )


def generate_prompt_ui(
    rgb_img: Optional[Image.Image],
    chinese_name: str,
    vlm_api_url: str,
    vlm_model: str,
    vlm_timeout_s: float,
) -> Tuple[str, str]:
    """UI：根据 RGB + 商品中文名调用 VLM，回填实例分割提示词。"""
    try:
        if rgb_img is None:
            return "", "请先上传 RGB 图像"
        name = (chinese_name or "").strip()
        if not name:
            return "", "请填写商品中文名"
        prompt = generate_sam3_prompt_from_image(
            rgb_img.convert("RGB"),
            name,
            api_url=(vlm_api_url or DEFAULT_SAM3_VLM_API_URL).strip(),
            model=(vlm_model or DEFAULT_SAM3_VLM_MODEL).strip(),
            timeout_s=float(vlm_timeout_s or DEFAULT_SAM3_VLM_TIMEOUT_S),
        )
        return prompt, f"已生成提示词：{prompt}"
    except Exception as exc:  # noqa: BLE001
        logger.error("VLM generate prompt failed:\n%s", traceback.format_exc())
        return "", f"生成失败：{exc}"


def run_sam3_genpose_tab(
    rgb_img: Optional[Image.Image],
    depth_file: Any,
    camera_file: Any,
    prompt: str,
    chinese_name: str,
    use_vlm_prompt: bool,
    vlm_api_url: str,
    vlm_model: str,
    vlm_timeout_s: float,
    api_url: str,
    threshold: float,
    mask_threshold: float,
    timeout_s: float,
    max_instances: int,
    score_ckpt: str,
    energy_ckpt: str,
    scale_ckpt: str,
    max_points: float,
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
) -> GenPoseTabOut:
    try:
        if rgb_img is None:
            return _error("请上传 RGB 图像")
        depth_path = filepath_from_upload(depth_file)
        camera_path = filepath_from_upload(camera_file)
        if depth_path is None or not depth_path.is_file():
            return _error("请上传深度文件（.npy / .png / .exr）")
        if camera_path is None or not camera_path.is_file():
            return _error("请上传 camera.json")

        for name, ckpt in (
            ("score", score_ckpt),
            ("energy", energy_ckpt),
            ("scale", scale_ckpt),
        ):
            path = resolve_repo_path((ckpt or "").strip())
            if not path.is_file():
                return _error(f"{name} checkpoint 不存在: {path}")

        rgb = rgb_img.convert("RGB")
        vlm_meta: Dict[str, Any] = {"enabled": False}
        if bool(use_vlm_prompt):
            name = (chinese_name or "").strip()
            if not name:
                return _error("已勾选「大模型生成提示词」，请填写商品中文名")
            t_vlm = time.perf_counter()
            prompt_text = generate_sam3_prompt_from_image(
                rgb,
                name,
                api_url=(vlm_api_url or DEFAULT_SAM3_VLM_API_URL).strip(),
                model=(vlm_model or DEFAULT_SAM3_VLM_MODEL).strip(),
                timeout_s=float(vlm_timeout_s or DEFAULT_SAM3_VLM_TIMEOUT_S),
            )
            vlm_meta = {
                "enabled": True,
                "chinese_name": name,
                "prompt": prompt_text,
                "api_url": (vlm_api_url or DEFAULT_SAM3_VLM_API_URL).strip(),
                "model": (vlm_model or DEFAULT_SAM3_VLM_MODEL).strip(),
                "elapsed_s": round(time.perf_counter() - t_vlm, 3),
            }
        else:
            prompt_text = (prompt or "").strip() or DEFAULT_SAM3_PROMPT

        depth_mm = load_depth_mm(depth_path)
        intrinsic, factor_depth, camera_meta = load_camera_json_full(camera_path)
        if rgb.size != (depth_mm.shape[1], depth_mm.shape[0]):
            return _error(
                f"RGB 尺寸 {rgb.size} 与深度 {depth_mm.shape[1]}x{depth_mm.shape[0]} 不一致"
            )

        color_np = np.asarray(rgb, dtype=np.uint8)
        depth_mm, color_for_cloud, align_meta = resolve_rgb_depth_alignment(
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
        run_dir = OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S") / f"pose_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = run_dir / "rgb.png"
        rgb.save(rgb_path)

        sam3_seg.DEFAULT_SAM3_API_URL = (api_url or DEFAULT_SAM3_API_URL).strip()
        sam3_seg.DEFAULT_SAM3_TIMEOUT_S = float(timeout_s or DEFAULT_SAM3_TIMEOUT_S)

        t0 = time.perf_counter()
        sam3_result = run_sam3_segmentation(
            rgb_path,
            run_dir / "sam3_results",
            prompt=prompt_text,
            threshold=float(threshold),
            mask_threshold=float(mask_threshold),
            max_instances=int(max_instances) if max_instances else 0,
        )
        sam3_s = time.perf_counter() - t0

        dets = sam3_result.instance_dets or []
        if not dets:
            return _error("SAM3 未返回实例", sensor_vis)

        mask_vis, bbox_vis = render_mask_bbox_previews(
            rgb, dets, _decode_detection_mask, prompt=prompt_text
        )
        mask_u8 = load_mask_u8(sam3_result.mask_exr)
        if mask_u8.shape[:2] != (depth_mm.shape[0], depth_mm.shape[1]):
            mask_u8 = np.array(
                Image.fromarray(mask_u8).resize(
                    (depth_mm.shape[1], depth_mm.shape[0]), Image.NEAREST
                )
            )
        # 全局无效深度上的实例像素清掉
        mask_u8 = mask_u8.copy()
        mask_u8[depth_mm <= 0] = 0

        instance_outlier_stats: Dict[str, Any] = {"enabled": False}
        if bool(enable_instance_outlier):
            mask_u8, _removed, instance_outlier_stats = clean_instance_id_map(
                depth_mm,
                mask_u8,
                intrinsic,
                factor_depth=factor_depth,
                std_ratio=float(sor_std_ratio or 1.5),
                z_mad_ratio=float(z_mad_ratio or 2.5),
                sor_nb_neighbors=int(sor_nb_neighbors or 20),
            )
            if not np.any(mask_u8 > 0):
                return _error("离群点剔除后无有效实例点，请放宽参数", sensor_vis)
            cleaned_mask_path = run_dir / "mask_cleaned.png"
            Image.fromarray(mask_u8.astype(np.uint8)).save(cleaned_mask_path)
            instance_outlier_stats["mask_cleaned"] = str(cleaned_mask_path)

        h, w = color_np.shape[:2]
        meta = camera_json_to_meta(camera_meta, width=w, height=h)
        depth_scale = float(meta.get("depth_scale", camera_meta.get("depth_scale", 0.001)))
        if depth_path.suffix.lower() == ".png" and depth_scale >= 0.1:
            depth_scale = 0.001
        # 与清洗后的 depth_mm 对齐（米）
        depth_m = depth_mm.astype(np.float32) / float(factor_depth)
        if depth_m.shape[:2] != (h, w):
            return _error(
                f"深度尺寸 {depth_m.shape[:2]} 与 RGB {(h, w)} 不一致", sensor_vis
            )

        t1 = time.perf_counter()
        poses_np, lengths_np, pose_overlay = run_genpose2_from_mask(
            color_rgb=color_np,
            depth_m=depth_m,
            mask_u8=mask_u8,
            meta=meta,
            score_ckpt=score_ckpt,
            energy_ckpt=energy_ckpt,
            scale_ckpt=scale_ckpt,
        )
        pose_s = time.perf_counter() - t1
        elapsed = time.perf_counter() - t0

        pose_overlay_path = run_dir / "vis_pem.png"
        pose_overlay.save(pose_overlay_path)

        instance_scores = list(sam3_result.instance_scores or [])
        instance_bboxes = [d.get("bbox") for d in dets]
        poses_payload = build_poses_payload(
            poses_np,
            lengths_np,
            instance_scores=instance_scores,
            instance_bboxes=instance_bboxes,
        )
        grasp_payload = build_grasp_display_payload(
            poses_np,
            lengths_np,
            instance_scores=instance_scores,
        )
        files = export_pose_scene(
            depth_mm=depth_mm,
            color_rgb=color_for_cloud,
            intrinsic=intrinsic,
            factor_depth=factor_depth,
            poses_np=poses_np,
            lengths_np=lengths_np,
            run_dir=run_dir,
            stem="scene_sam3_genpose2",
            max_points=int(max_points) if max_points else GRASP_CLOUD_MAX_POINTS,
        )
        poses_payload["scene_glb"] = files["glb"]
        poses_payload["scene_ply"] = files["ply"]
        (run_dir / "poses.json").write_text(
            json.dumps(poses_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "grasp_pose.json").write_text(
            json.dumps(grasp_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Also write detection_pem.json (HTTP 服务风格)
        detections_pem: List[Dict[str, Any]] = []
        for i, p in enumerate(poses_payload["poses"]):
            detections_pem.append(
                {
                    "scene_id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "instance_id": p["instance_id"],
                    "score": p["score"],
                    "t": p["xyz_mm"],
                    "R": p["rotation_matrix"],
                    "size_3d": p["size_3d"],
                    "bbox": p.get("bbox"),
                }
            )
        (run_dir / "detection_pem.json").write_text(
            json.dumps(detections_pem, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 供缺货放置等后续阶段复用
        np.save(run_dir / "depth_mm.npy", depth_mm)
        np.save(run_dir / "mask_u8.npy", mask_u8)
        (run_dir / "intrinsic.json").write_text(
            json.dumps(
                {
                    "intrinsic": np.asarray(intrinsic, dtype=float).reshape(3, 3).tolist(),
                    "factor_depth": float(factor_depth),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        detail = {
            "success": True,
            "pipeline": "SAM3 → GenPose2",
            "elapsed_s": round(elapsed, 3),
            "timing": {
                "sam3_s": round(sam3_s, 3),
                "pose_s": round(pose_s, 3),
            },
            "num_instances": int(sam3_result.num_instances),
            "num_poses": int(poses_np.shape[0]),
            "detections": detection_summary(dets),
            "workspace_outlier_filter": workspace_stats,
            "instance_outlier_filter": instance_outlier_stats,
            "poses": {
                "num": int(poses_np.shape[0]),
                "glb": files["glb"],
                "json": str(run_dir / "poses.json"),
                "grasp_json": str(run_dir / "grasp_pose.json"),
                "detection_pem": str(run_dir / "detection_pem.json"),
            },
            "pose_overlay": str(pose_overlay_path),
            "rgb_shift": rgb_shift_meta,
            "depth_align": depth_align_meta,
            "sam3_api": sam3_seg.DEFAULT_SAM3_API_URL,
            "prompt": prompt_text,
            "vlm_prompt": vlm_meta,
            "score_ckpt": str(resolve_repo_path(score_ckpt)),
            "energy_ckpt": str(resolve_repo_path(energy_ckpt)),
            "scale_ckpt": str(resolve_repo_path(scale_ckpt)),
            "run_dir": str(run_dir),
        }

        return (
            sensor_vis,
            mask_vis,
            bbox_vis,
            pose_overlay,
            files["glb"],
            files["glb"],
            files["ply"],
            json.dumps(grasp_payload, ensure_ascii=False, indent=2),
            json.dumps(poses_payload, ensure_ascii=False, indent=2),
            json.dumps(detail, ensure_ascii=False, indent=2),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("genpose_tab exception:\n%s", traceback.format_exc())
        return _error(str(exc))


def build_sam3_genpose_tab() -> None:
    sam3_cfg = get_sam3_conf()
    gp_cfg = get_genpose2_conf()
    vlm_cfg = get_vlm_profile("sam3_prompt")
    with gr.Tab("SAM3 + GenPose2"):
        gr.Markdown(
            "流水线：**SAM3 文本分割 → GenPose2 6D 位姿估计**。"
            "输出位姿叠加图、抓取位姿（`xyzrxryrz` mm/° + 目标正方体）、`poses.json`、点云 GLB（RGB 坐标轴）。"
            "实例分割提示词可由多模态大模型根据 RGB + 商品中文名自动生成。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                rgb = gr.Image(type="pil", label="上传 RGB", height=280, interactive=True)
                with gr.Row():
                    depth = gr.File(
                        label="深度（.npy / uint16 .png / .exr）",
                        type="filepath",
                        file_types=[".npy", ".png", ".tif", ".tiff", ".exr"],
                    )
                    camera = gr.File(
                        label="相机 camera.json",
                        type="filepath",
                        file_types=[".json"],
                    )
                enable_depth_align = gr.Checkbox(
                    label="对齐 Depth→RGB（推荐：修正像素/3D 偏差；开启后 dx/dy 用于几何对齐）",
                    value=True,
                )
                with gr.Row():
                    rgb_shift_x = gr.Number(
                        label="RGB↔Depth 偏移 dx（历史上色约定：+右；默认 -45→Depth 右移 45）",
                        value=-45,
                        precision=0,
                    )
                    rgb_shift_y = gr.Number(
                        label="RGB↔Depth 偏移 dy（+下）",
                        value=0,
                        precision=0,
                    )
                chinese_name = gr.Textbox(
                    label="商品中文名（供大模型生成提示词）",
                    placeholder="例如：白色塑料托盘",
                    lines=1,
                )
                vlm_api = gr.Textbox(
                    label="SAM3 提示词 VLM URL（qwen3-vl / chat/completions）",
                    value=str(vlm_cfg.get("api_url") or DEFAULT_SAM3_VLM_API_URL),
                    lines=1,
                )
                with gr.Row():
                    vlm_model = gr.Textbox(
                        label="SAM3 提示词模型（默认 qwen3-vl-4b）",
                        value=str(vlm_cfg.get("model") or DEFAULT_SAM3_VLM_MODEL),
                        scale=3,
                    )
                    vlm_timeout = gr.Number(
                        label="VLM 超时（秒）",
                        value=float(vlm_cfg.get("timeout_s", DEFAULT_SAM3_VLM_TIMEOUT_S)),
                        precision=0,
                        scale=1,
                    )
                with gr.Row():
                    use_vlm_prompt = gr.Checkbox(
                        label="运行前由大模型生成提示词",
                        value=False,
                    )
                    btn_gen_prompt = gr.Button("生成提示词", variant="secondary")
                prompt = gr.Textbox(
                    label="实例分割提示词",
                    value=str(sam3_cfg.get("default_prompt") or DEFAULT_SAM3_PROMPT),
                    lines=3,
                )
                vlm_status = gr.Textbox(label="提示词生成状态", interactive=False, lines=1)
                with gr.Accordion("SAM3 推理参数", open=False):
                    api = gr.Textbox(
                        label="SAM3 API URL",
                        value=str(sam3_cfg.get("api_url") or DEFAULT_SAM3_API_URL),
                    )
                    thr = gr.Slider(
                        label="Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=float(sam3_cfg.get("threshold", DEFAULT_SAM3_THRESHOLD)),
                    )
                    mthr = gr.Slider(
                        label="Mask Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=float(
                            sam3_cfg.get("mask_threshold", DEFAULT_SAM3_MASK_THRESHOLD)
                        ),
                    )
                    timeout = gr.Number(
                        label="超时（秒）",
                        value=float(sam3_cfg.get("timeout_s", DEFAULT_SAM3_TIMEOUT_S)),
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
                        label="② 按实例：各 SAM3 实例独立剔除后再送 GenPose2（median/MAD + SOR）",
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
                with gr.Accordion("GenPose2 参数", open=True):
                    score = gr.Textbox(
                        label="score_ckpt",
                        value=str(gp_cfg.get("score_ckpt") or "results/ckpts/ScoreNet/scorenet.pth"),
                    )
                    energy = gr.Textbox(
                        label="energy_ckpt",
                        value=str(
                            gp_cfg.get("energy_ckpt") or "results/ckpts/EnergyNet/energynet.pth"
                        ),
                    )
                    scale = gr.Textbox(
                        label="scale_ckpt",
                        value=str(gp_cfg.get("scale_ckpt") or "results/ckpts/ScaleNet/scalenet.pth"),
                    )
                    max_pts = gr.Number(
                        label="点云最大点数（降采样）",
                        value=int(GRASP_CLOUD_MAX_POINTS),
                        precision=0,
                    )
                btn = gr.Button("运行 SAM3 + GenPose2", variant="primary")
                out_grasp = gr.Code(
                    label="抓取位姿 · xyzrxryrz（mm / °）+ 目标正方体",
                    language="json",
                    lines=14,
                    value=_empty_grasp_json(),
                )

            with gr.Column(scale=1):
                gr.Markdown("#### 快速预览")
                out_depth = gr.Image(type="pil", label="传感器深度（伪彩色）", height=200)
                out_mask = gr.Image(type="pil", label="SAM3 实例 mask", height=200)
                out_bbox = gr.Image(type="pil", label="SAM3 实例 bbox", height=200)
                out_pose_img = gr.Image(
                    type="pil",
                    label="位姿叠加（坐标轴 + 3D 尺寸框）",
                    height=280,
                )

        gr.Markdown("### 点云 + 位姿坐标轴")
        gr.Markdown(
            "> **全幅 RGB 彩色点云** + **每实例 RGB 坐标轴**（X红/Y绿/Z蓝）。"
            "数值坐标系 `frame=camera`；3D 预览翻 Y 与 RGB 同向。"
            "左侧 **抓取位姿** 框：`xyzrxryrz` 为 mm/°（对齐 Gen6D 摘取点格式），"
            "并含目标空间正方体 `size_3d` / `corners_mm`。"
        )
        with gr.Row():
            with gr.Column(scale=4):
                out_glb = gr.Model3D(label="位姿 GLB", height=480, display_mode="solid")
            with gr.Column(scale=1):
                gr.Markdown("**下载**")
                out_glb_dl = gr.File(label="GLB", interactive=False)
                out_ply_dl = gr.File(label="PLY", interactive=False)
        out_poses = gr.Code(
            label="poses.json",
            language="json",
            lines=12,
            value=_empty_poses(),
        )
        out_json = gr.Code(label="详情", language="json", lines=14)

        btn_gen_prompt.click(
            fn=generate_prompt_ui,
            inputs=[rgb, chinese_name, vlm_api, vlm_model, vlm_timeout],
            outputs=[prompt, vlm_status],
        )

        btn.click(
            fn=run_sam3_genpose_tab,
            inputs=[
                rgb,
                depth,
                camera,
                prompt,
                chinese_name,
                use_vlm_prompt,
                vlm_api,
                vlm_model,
                vlm_timeout,
                api,
                thr,
                mthr,
                timeout,
                max_inst,
                score,
                energy,
                scale,
                max_pts,
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
            outputs=[
                out_depth,
                out_mask,
                out_bbox,
                out_pose_img,
                out_glb,
                out_glb_dl,
                out_ply_dl,
                out_grasp,
                out_poses,
                out_json,
            ],
        )

        gr.Markdown(
            """
            **说明**
            - **提示词**：可手写；或填「商品中文名」后点「生成提示词」/勾选「运行前由大模型生成」
            - **VLM**：界面可改 API URL / 模型名；`POST /v1/chat/completions`（传 RGB + 指令），默认见 `config/conf.json` → `vlm`
            - **SAM3**：外部 HTTP `POST /infer`（`image_base64`），生成实例 mask
            - **Depth→RGB 对齐**：默认开启。将 Depth warp 到 RGB 网格后再做 GenPose2 / 2D 叠加，消除横向偏差；`dx=-45` 表示历史 RGB 上色偏移，对齐时 Depth 使用 `+45`
            - **camera.json**：可用 `depth_to_rgb_shift:[dx,dy]`（Depth 平移），或沿用 `rgb_shift:[dx,dy]`（按上色约定取反）
            - **离群点剔除**：先 **全局**（深度范围+SOR），再 **按实例**（MAD+SOR），清洗后再送 GenPose2
            - **GenPose2**：ScoreNet → EnergyNet → ScaleNet，估计 6D 位姿与 3D 尺寸（无需 CAD）
            - **抓取位姿框**：`xyzrxryrz = [x,y,z,rx,ry,rz]`（mm / °，ZYX），含目标正方体 `size_3d` / `size_3d_mm` / `corners_mm`
            - **poses.json**：`xyz_mm` + ZYX 欧拉角（rad）+ `size_3d`（米），相机系
            - **运行产物**：`output/ui_runs/<时间戳>/pose_*/`（含 `grasp_pose.json`）
            - 首次运行会加载三网权重，耗时较长；之后复用缓存
            """
        )
