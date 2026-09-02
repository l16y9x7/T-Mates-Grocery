"""Gradio Tab: 缺货商品位姿估计。

① 手填或 VLM 识别缺货商品中文名
② VLM 生成 SAM3 提示词
③ SAM3 分割 → GenPose2 6D 位姿（复用 SAM3+GenPose2 流水线）
④ VLM 估计放置位移 → 目的 6D + 2D/GLB 可视化
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional, Tuple

import gradio as gr
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (  # noqa: E402
    get_genpose2_conf,
    get_sam3_conf,
    get_vlm_conf,
    get_vlm_profile,
)
from ui.genpose_tab import (  # noqa: E402
    _empty_grasp_json,
    _empty_poses,
    generate_prompt_ui,
    run_sam3_genpose_tab,
)
from ui.place_missing import run_place_missing_stage  # noqa: E402
from ui.pointcloud import GRASP_CLOUD_MAX_POINTS  # noqa: E402
from scripts.sam3_seg import (  # noqa: E402
    DEFAULT_SAM3_API_URL,
    DEFAULT_SAM3_MASK_THRESHOLD,
    DEFAULT_SAM3_PROMPT,
    DEFAULT_SAM3_THRESHOLD,
    DEFAULT_SAM3_TIMEOUT_S,
)
from scripts.vlm_prompt import (  # noqa: E402
    DEFAULT_MISSING_PRODUCT_PROMPT,
    DEFAULT_REASON_VLM_API_URL,
    DEFAULT_REASON_VLM_MODEL,
    DEFAULT_REASON_VLM_TIMEOUT_S,
    DEFAULT_SAM3_VLM_API_URL,
    DEFAULT_SAM3_VLM_MODEL,
    DEFAULT_SAM3_VLM_TIMEOUT_S,
    identify_missing_product,
)

logger = logging.getLogger("misplaced_tab")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

IdentifyOut = Tuple[str, str, str]
# product_name, identify_raw, sam3_prompt, vlm_status + GenPoseTabOut (10:
#   depth/mask/bbox/pose_img/glb/glb_dl/ply/grasp/poses/json)
# + place_overlay, place_glb, place_glb_dl, place_ply_dl, place_6d_json, place_summary
PipelineOut = Tuple[
    str,
    str,
    str,
    str,
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
    Optional[Image.Image],
    Optional[str],
    Optional[str],
    Optional[str],
    str,
    str,
]


def _axis_move_zh(value_mm: float, pos_label: str, neg_label: str) -> str:
    v = float(value_mm or 0.0)
    if abs(v) < 0.5:
        return f"几乎不{pos_label}/不{neg_label}（{v:.1f} mm）"
    if v > 0:
        return f"向{pos_label} {v:.1f} mm"
    return f"向{neg_label} {abs(v):.1f} mm"


def format_place_summary(place: dict, product_name: str = "") -> str:
    """人可读：选用哪个实例、如何移动到缺货位。"""
    if not place or not place.get("success", True):
        msg = (place or {}).get("message") or "尚未估计摆放"
        return f"摆放摘要：{msg}"

    src = place.get("source_pose") or {}
    dest = place.get("destination_pose") or {}
    off = place.get("place_offset_mm") or {}
    name = product_name or place.get("product_name") or "商品"
    inst = src.get("instance_id", dest.get("source_instance_id", "?"))
    score = float(src.get("score") or 0.0)
    src_xyz = src.get("xyz_mm") or []
    dest_xyz = dest.get("xyz_mm") or place.get("xyz_mm") or []

    right = float(off.get("right_mm") or 0.0)
    down = float(off.get("down_mm") or 0.0)
    forward = float(off.get("forward_mm") or 0.0)

    selected_by = place.get("selected_by") or dest.get("selected_by") or "vlm"
    reason = (place.get("vlm_reason") or dest.get("vlm_reason") or "").strip()
    spatial_note = (
        place.get("spatial_note")
        or dest.get("spatial_note")
        or ((place.get("spatial_resolve") or {}).get("spatial_note"))
        or ""
    ).strip()
    prior = place.get("spatial_prior") or {}
    cands = place.get("candidate_instances") or []
    lines = [
        f"【选用目标】商品「{name}」· 实例 #{inst}（空间推断+模型，{selected_by}，score={score:.3f}）",
    ]
    if prior.get("front_z_ref_mm") is not None:
        lines.append(
            f"【空间先验】前排 front_z_ref≈{prior.get('front_z_ref_mm')} mm；"
            f"空列建议源 id={((prior.get('suggested') or {}).get('suggested_source_id'))}"
        )
    if cands:
        ids = ", ".join(str(c.get("instance_id")) for c in cands)
        lines.append(f"【候选实例】共 {len(cands)} 个：id={ids}")
    lines += [
        (
            f"【当前位置】xyz_mm = "
            f"[{', '.join(f'{float(v):.1f}' for v in src_xyz[:3])}]"
            if len(src_xyz) >= 3
            else "【当前位置】未知"
        ),
        "【如何移动】相机系（右 / 下 / 前=深入货架；负前=靠近相机的前排空位）",
        f"  · {_axis_move_zh(right, '右', '左')}",
        f"  · {_axis_move_zh(down, '下', '上')}",
        f"  · {_axis_move_zh(forward, '前(深入货架)', '后(靠近相机/前排)')}",
        f"【位移数值】right_mm={right:.1f}, down_mm={down:.1f}, forward_mm={forward:.1f}",
    ]
    if reason:
        lines.append(f"【选型理由】{reason}")
    if spatial_note:
        lines.append(f"【空间校正】{spatial_note}")
    if len(dest_xyz) >= 3:
        lines.append(
            "【目的位置】xyz_mm = "
            f"[{', '.join(f'{float(v):.1f}' for v in dest_xyz[:3])}]"
        )
    raw = ((place.get("vlm") or {}).get("raw_reply") or "").strip()
    if raw:
        lines.append(f"【VLM 原始输出】{raw}")
    return "\n".join(lines)


def _empty_place_summary(message: str = "") -> str:
    return format_place_summary({"success": False, "message": message or "尚未估计摆放"})


def _default_missing_prompt() -> str:
    cfg = get_vlm_conf()
    return str(
        cfg.get("missing_prompt")
        or cfg.get("misplaced_prompt")
        or DEFAULT_MISSING_PRODUCT_PROMPT
    )


def run_identify_missing(
    rgb_img: Optional[Image.Image],
    identify_prompt: str,
    vlm_api_url: str,
    vlm_model: str,
    vlm_timeout_s: float,
) -> IdentifyOut:
    """用大模型识别缺货商品名，回填可编辑的商品名文本框。"""
    try:
        if rgb_img is None:
            return "", "", json.dumps(
                {"success": False, "message": "请上传货架 RGB 图像"},
                ensure_ascii=False,
                indent=2,
            )
        t0 = time.perf_counter()
        result = identify_missing_product(
            rgb_img.convert("RGB"),
            identify_prompt,
            api_url=(vlm_api_url or DEFAULT_REASON_VLM_API_URL).strip(),
            model=(vlm_model or DEFAULT_REASON_VLM_MODEL).strip(),
            timeout_s=float(vlm_timeout_s or DEFAULT_REASON_VLM_TIMEOUT_S),
        )
        elapsed = time.perf_counter() - t0
        detail = {
            "success": True,
            "step": 1,
            "step_name": "识别缺货商品名",
            "product_name": result["product_name"],
            "raw_reply": result["raw_reply"],
            "prompt": (identify_prompt or "").strip() or DEFAULT_MISSING_PRODUCT_PROMPT,
            "vlm_api": (vlm_api_url or DEFAULT_REASON_VLM_API_URL).strip(),
            "vlm_model": (vlm_model or DEFAULT_REASON_VLM_MODEL).strip(),
            "vlm_profile": "reason",
            "elapsed_s": round(elapsed, 3),
            "next": "确认/修改商品名后，可生成 SAM3 提示词或直接运行完整流水线",
        }
        return (
            result["product_name"],
            result["raw_reply"],
            json.dumps(detail, ensure_ascii=False, indent=2),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("identify missing product failed:\n%s", traceback.format_exc())
        return "", "", json.dumps(
            {"success": False, "message": str(exc)},
            ensure_ascii=False,
            indent=2,
        )


def _empty_place_6d(message: str = "") -> str:
    return json.dumps(
        {
            "success": False,
            "message": message or "尚未估计目的位姿",
            "destination_pose": None,
            "xyzrxryrz": None,
            "place_offset_mm": None,
        },
        ensure_ascii=False,
        indent=2,
    )


def _pipeline_error(
    message: str,
    product_name: str = "",
    sam3_prompt: str = "",
    identify_raw: str = "",
) -> PipelineOut:
    err = json.dumps({"success": False, "message": message}, ensure_ascii=False, indent=2)
    return (
        product_name,
        identify_raw,
        sam3_prompt,
        message,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        _empty_grasp_json(message),
        _empty_poses(message),
        err,
        None,
        None,
        None,
        None,
        _empty_place_6d(message),
        _empty_place_summary(message),
    )


def run_missing_pose_pipeline(
    rgb_img: Optional[Image.Image],
    depth_file: Any,
    camera_file: Any,
    product_name: str,
    auto_identify: bool,
    identify_prompt: str,
    identify_raw_in: str,
    sam3_prompt: str,
    use_vlm_sam3_prompt: bool,
    sam3_vlm_api_url: str,
    sam3_vlm_model: str,
    sam3_vlm_timeout_s: float,
    reason_vlm_api_url: str,
    reason_vlm_model: str,
    reason_vlm_timeout_s: float,
    api_url: str,
    threshold: float,
    mask_threshold: float,
    timeout_s: float,
    max_instances: int,
    score_ckpt: str,
    energy_ckpt: str,
    scale_ckpt: str,
    max_points: float,
    rgb_shift_x: float,
    rgb_shift_y: float,
    enable_depth_align: bool,
    enable_workspace_outlier: bool,
    max_depth_mm: float,
    min_depth_mm: float,
    depth_percentile_hi: float,
    enable_instance_outlier: bool,
    sor_nb_neighbors: float,
    sor_std_ratio: float,
    z_mad_ratio: float,
) -> PipelineOut:
    """完整流水线：商品名 → SAM3 → GenPose2 → VLM 放置位移 → 目的 6D。"""
    name = (product_name or "").strip()
    identify_dialogue = (identify_raw_in or "").strip()
    identify_meta: dict = {
        "enabled": False,
        "raw_reply": identify_dialogue,
        "from_ui": bool(identify_dialogue),
    }
    reason_api = (reason_vlm_api_url or DEFAULT_REASON_VLM_API_URL).strip()
    reason_model = (reason_vlm_model or DEFAULT_REASON_VLM_MODEL).strip()
    reason_timeout = float(reason_vlm_timeout_s or DEFAULT_REASON_VLM_TIMEOUT_S)
    try:
        if rgb_img is None:
            return _pipeline_error("请上传货架 RGB 图像")

        # ① 缺货识别对话：自动识别商品名，或在放置前补跑以提供空位上下文
        need_identify_for_name = bool(auto_identify) and not name
        need_identify_for_place = not identify_dialogue
        if need_identify_for_name or (bool(auto_identify) and need_identify_for_place):
            t0 = time.perf_counter()
            result = identify_missing_product(
                rgb_img.convert("RGB"),
                identify_prompt,
                api_url=reason_api,
                model=reason_model,
                timeout_s=reason_timeout,
            )
            if need_identify_for_name or not name:
                name = result["product_name"]
            identify_dialogue = (result.get("raw_reply") or "").strip()
            identify_meta = {
                "enabled": True,
                "product_name": name,
                "raw_reply": identify_dialogue,
                "vlm_profile": "reason",
                "vlm_model": reason_model,
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "from_ui": False,
            }
        if not name:
            return _pipeline_error(
                "请手填缺货商品名，或勾选「运行前自动识别缺货商品名」",
                product_name=product_name or "",
                sam3_prompt=sam3_prompt or "",
            )

        # ②③ 复用 SAM3+GenPose2：SAM3 提示词用 qwen3-vl-4b
        outs = run_sam3_genpose_tab(
            rgb_img,
            depth_file,
            camera_file,
            sam3_prompt,
            name,
            bool(use_vlm_sam3_prompt),
            sam3_vlm_api_url,
            sam3_vlm_model,
            sam3_vlm_timeout_s,
            api_url,
            threshold,
            mask_threshold,
            timeout_s,
            max_instances,
            score_ckpt,
            energy_ckpt,
            scale_ckpt,
            max_points,
            rgb_shift_x,
            rgb_shift_y,
            enable_depth_align,
            enable_workspace_outlier,
            max_depth_mm,
            min_depth_mm,
            depth_percentile_hi,
            enable_instance_outlier,
            sor_nb_neighbors,
            sor_std_ratio,
            z_mad_ratio,
        )

        used_prompt = (sam3_prompt or "").strip()
        vlm_status = "已运行完整流水线"
        place_overlay = None
        place_glb = None
        place_glb_dl = None
        place_ply_dl = None
        place_6d_json = _empty_place_6d()
        place_summary = _empty_place_summary()
        detail_obj: dict = {}
        try:
            detail_obj = json.loads(outs[-1])
            if identify_meta.get("enabled"):
                detail_obj["missing_identify"] = identify_meta
            detail_obj["product_name"] = name
            detail_obj["pipeline"] = (
                "缺货识别 → SAM3 → GenPose2 → VLM放置位移 → 目的6D"
            )
            if detail_obj.get("vlm_prompt", {}).get("prompt"):
                used_prompt = str(detail_obj["vlm_prompt"]["prompt"])
            elif detail_obj.get("prompt"):
                used_prompt = str(detail_obj["prompt"])
            if detail_obj.get("success"):
                vlm_status = (
                    f"缺货商品「{name}」→ SAM3「{used_prompt}」→ GenPose2 完成"
                )
            else:
                vlm_status = str(detail_obj.get("message") or "流水线失败")
        except Exception:  # noqa: BLE001
            detail_obj = {}

        # ④ 置信度最高实例 → VLM 放置位移（附带①识别对话）→ 目的位姿可视化
        if detail_obj.get("success") and detail_obj.get("run_dir"):
            try:
                # 若仍无识别对话，补跑一步①，仅作放置上下文（不覆盖已有商品名）
                if not identify_dialogue:
                    t0 = time.perf_counter()
                    result = identify_missing_product(
                        rgb_img.convert("RGB"),
                        identify_prompt,
                        api_url=reason_api,
                        model=reason_model,
                        timeout_s=reason_timeout,
                    )
                    identify_dialogue = (result.get("raw_reply") or "").strip()
                    identify_meta = {
                        "enabled": True,
                        "product_name": name,
                        "raw_reply": identify_dialogue,
                        "vlm_profile": "reason",
                        "vlm_model": reason_model,
                        "elapsed_s": round(time.perf_counter() - t0, 3),
                        "for_place_context_only": True,
                    }
                    detail_obj["missing_identify"] = identify_meta

                poses_payload = json.loads(outs[-2])
                place = run_place_missing_stage(
                    run_dir=Path(str(detail_obj["run_dir"])),
                    product_name=name,
                    rgb=rgb_img.convert("RGB"),
                    poses_payload=poses_payload,
                    vlm_api_url=reason_api,
                    vlm_model=reason_model,
                    vlm_timeout_s=reason_timeout,
                    max_points=int(max_points or GRASP_CLOUD_MAX_POINTS),
                    identify_dialogue=identify_dialogue,
                )
                place_summary = format_place_summary(place, name)
                detail_obj["place_destination"] = {
                    "xyz_mm": place.get("xyz_mm"),
                    "xyzrxryrz": place.get("xyzrxryrz"),
                    "place_offset_mm": place.get("place_offset_mm"),
                    "destination_pose": place.get("destination_pose"),
                    "source_pose": place.get("source_pose"),
                    "identify_dialogue": identify_dialogue,
                    "overlay": place.get("overlay"),
                    "scene_glb": place.get("scene_glb"),
                    "vlm": place.get("vlm"),
                    "summary": place_summary,
                    "json": str(Path(detail_obj["run_dir"]) / "place_destination.json"),
                }
                place_overlay = Image.open(place["overlay"]).convert("RGB")
                place_glb = place.get("scene_glb")
                place_glb_dl = place.get("scene_glb")
                place_ply_dl = place.get("scene_ply")
                place_6d_json = json.dumps(
                    {
                        "success": True,
                        "product_name": name,
                        "frame": "camera",
                        "axis_hint": "X right, Y down, Z forward",
                        "unit": {
                            "xyz": "mm",
                            "rx_ry_rz": "rad",
                            "offset": "mm",
                        },
                        "selected_instance_id": (
                            (place.get("source_pose") or {}).get("instance_id")
                        ),
                        "selected_score": (place.get("source_pose") or {}).get("score"),
                        "how_to_move": {
                            "right_mm": (place.get("place_offset_mm") or {}).get(
                                "right_mm"
                            ),
                            "down_mm": (place.get("place_offset_mm") or {}).get(
                                "down_mm"
                            ),
                            "forward_mm": (place.get("place_offset_mm") or {}).get(
                                "forward_mm"
                            ),
                            "zh": place_summary,
                        },
                        "source_pose": place.get("source_pose"),
                        "destination_pose": place.get("destination_pose"),
                        "place_offset_mm": place.get("place_offset_mm"),
                        "identify_dialogue": identify_dialogue,
                        "xyz_mm": place.get("xyz_mm"),
                        "xyzrxryrz": place.get("xyzrxryrz"),
                        "vlm_raw_reply": (place.get("vlm") or {}).get("raw_reply"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                src = place.get("source_pose") or {}
                off = place.get("place_offset_mm") or {}
                vlm_status = (
                    f"选用实例#{src.get('instance_id', '?')} "
                    f"(score={float(src.get('score') or 0):.3f}) → "
                    f"{_axis_move_zh(float(off.get('right_mm') or 0), '右', '左')}；"
                    f"{_axis_move_zh(float(off.get('down_mm') or 0), '下', '上')}；"
                    f"{_axis_move_zh(float(off.get('forward_mm') or 0), '前', '后')}"
                )
            except Exception as place_exc:  # noqa: BLE001
                logger.error("place missing stage failed:\n%s", traceback.format_exc())
                detail_obj["place_destination"] = {
                    "success": False,
                    "message": str(place_exc),
                }
                place_6d_json = _empty_place_6d(str(place_exc))
                place_summary = _empty_place_summary(str(place_exc))
                vlm_status = f"GenPose2 完成，但放置估计失败: {place_exc}"

        if identify_meta.get("enabled") or identify_dialogue:
            detail_obj["missing_identify"] = {
                **identify_meta,
                "raw_reply": identify_dialogue or identify_meta.get("raw_reply"),
                "used_for_place": bool(identify_dialogue),
            }
        outs = (*outs[:-1], json.dumps(detail_obj, ensure_ascii=False, indent=2))
        return (
            name,
            identify_dialogue,
            used_prompt,
            vlm_status,
            *outs,
            place_overlay,
            place_glb,
            place_glb_dl,
            place_ply_dl,
            place_6d_json,
            place_summary,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("missing pose pipeline failed:\n%s", traceback.format_exc())
        return _pipeline_error(
            str(exc),
            product_name=name or (product_name or ""),
            sam3_prompt=sam3_prompt or "",
        )


def build_misplaced_tab() -> None:
    sam3_cfg = get_sam3_conf()
    gp_cfg = get_genpose2_conf()
    sam3_vlm_cfg = get_vlm_profile("sam3_prompt")
    reason_vlm_cfg = get_vlm_profile("reason")
    with gr.Tab("缺货商品位姿估计"):
        gr.Markdown(
            "流水线：**① 缺货识别（MiniMax-M3）→ ② SAM3 提示词（qwen3-vl-4b）→ "
            "③ SAM3 → ④ GenPose2 → ⑤ 放置位移（MiniMax-M3）→ ⑥ 目的 6D**。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                rgb = gr.Image(type="pil", label="上传货架 RGB", height=260, interactive=True)
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
                    label="对齐 Depth→RGB（推荐：修正像素/3D 偏差；默认开启）",
                    value=True,
                )
                with gr.Row():
                    rgb_shift_x = gr.Number(
                        label="RGB↔Depth 偏移 dx（默认 -45→Depth 右移 45 对齐到 RGB）",
                        value=-45,
                        precision=0,
                    )
                    rgb_shift_y = gr.Number(
                        label="RGB↔Depth 偏移 dy（+下）",
                        value=0,
                        precision=0,
                    )

                gr.Markdown("#### ① 缺货商品名")
                product_name = gr.Textbox(
                    label="缺货商品名字（可手填，也可由大模型识别后回填）",
                    placeholder="例如：可比克 / 怡宝",
                    lines=1,
                )
                identify_prompt = gr.Textbox(
                    label="缺货识别提示词",
                    value=_default_missing_prompt(),
                    lines=4,
                )
                with gr.Row():
                    auto_identify = gr.Checkbox(
                        label="运行前若未填商品名则自动识别",
                        value=True,
                    )
                    btn_identify = gr.Button("仅识别缺货商品名", variant="secondary")
                identify_raw = gr.Textbox(
                    label="识别模型原始回复",
                    interactive=False,
                    lines=3,
                )

                gr.Markdown(
                    "#### ② 缺货 / 放置 VLM（MiniMax-M3）\n"
                    "API Key：`ANTHROPIC_API_KEY` 或 `config/secrets.local.json`"
                )
                reason_vlm_api = gr.Textbox(
                    label="Reason VLM API（Anthropic / MiniMax）",
                    value=str(
                        reason_vlm_cfg.get("api_url") or DEFAULT_REASON_VLM_API_URL
                    ),
                    lines=1,
                )
                with gr.Row():
                    reason_vlm_model = gr.Textbox(
                        label="Reason 模型",
                        value=str(
                            reason_vlm_cfg.get("model") or DEFAULT_REASON_VLM_MODEL
                        ),
                        scale=3,
                    )
                    reason_vlm_timeout = gr.Number(
                        label="超时（秒）",
                        value=float(
                            reason_vlm_cfg.get(
                                "timeout_s", DEFAULT_REASON_VLM_TIMEOUT_S
                            )
                        ),
                        precision=0,
                        scale=1,
                    )

                gr.Markdown("#### ③ SAM3 提示词 VLM（本地 qwen3-vl-4b）")
                sam3_vlm_api = gr.Textbox(
                    label="SAM3 提示词 VLM API（OpenAI chat/completions）",
                    value=str(sam3_vlm_cfg.get("api_url") or DEFAULT_SAM3_VLM_API_URL),
                    lines=1,
                )
                with gr.Row():
                    sam3_vlm_model = gr.Textbox(
                        label="SAM3 提示词模型",
                        value=str(
                            sam3_vlm_cfg.get("model") or DEFAULT_SAM3_VLM_MODEL
                        ),
                        scale=3,
                    )
                    sam3_vlm_timeout = gr.Number(
                        label="超时（秒）",
                        value=float(
                            sam3_vlm_cfg.get("timeout_s", DEFAULT_SAM3_VLM_TIMEOUT_S)
                        ),
                        precision=0,
                        scale=1,
                    )
                with gr.Row():
                    use_vlm_sam3_prompt = gr.Checkbox(
                        label="运行前由 qwen 根据商品名生成 SAM3 提示词",
                        value=True,
                    )
                    btn_gen_prompt = gr.Button("生成 SAM3 提示词", variant="secondary")
                sam3_prompt = gr.Textbox(
                    label="实例分割提示词（可手写 / 由上一步生成）",
                    value=str(sam3_cfg.get("default_prompt") or DEFAULT_SAM3_PROMPT),
                    lines=2,
                )
                vlm_status = gr.Textbox(label="状态", interactive=False, lines=2)

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
                    max_inst = gr.Number(
                        label="max_instances（0=全部）", value=5, precision=0
                    )
                with gr.Accordion("点云离群点剔除", open=False):
                    enable_workspace = gr.Checkbox(
                        label="① 全局：整幅深度点云离群点剔除（深度范围 + SOR）",
                        value=True,
                    )
                    with gr.Row():
                        max_depth = gr.Number(
                            label="max_depth_mm（0=不限制）", value=2500, precision=0
                        )
                        min_depth = gr.Number(label="min_depth_mm", value=50, precision=0)
                        depth_pct = gr.Number(label="depth_percentile_hi（0=关）", value=99.0)
                    enable_instance = gr.Checkbox(
                        label="② 按实例：各 SAM3 实例独立剔除后再送 GenPose2",
                        value=True,
                    )
                    sor_k = gr.Number(label="SOR 邻域点数 nb_neighbors", value=20, precision=0)
                    sor_std = gr.Number(label="SOR / 空间 std_ratio", value=1.5)
                    z_mad = gr.Number(label="实例深度 z_mad_ratio（MAD 倍数）", value=2.5)
                with gr.Accordion("GenPose2 参数", open=False):
                    score = gr.Textbox(
                        label="score_ckpt",
                        value=str(
                            gp_cfg.get("score_ckpt") or "results/ckpts/ScoreNet/scorenet.pth"
                        ),
                    )
                    energy = gr.Textbox(
                        label="energy_ckpt",
                        value=str(
                            gp_cfg.get("energy_ckpt")
                            or "results/ckpts/EnergyNet/energynet.pth"
                        ),
                    )
                    scale = gr.Textbox(
                        label="scale_ckpt",
                        value=str(
                            gp_cfg.get("scale_ckpt") or "results/ckpts/ScaleNet/scalenet.pth"
                        ),
                    )
                    max_pts = gr.Number(
                        label="点云最大点数（降采样）",
                        value=int(GRASP_CLOUD_MAX_POINTS),
                        precision=0,
                    )

                btn_run = gr.Button("运行缺货位姿估计（全流程）", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("#### 快速预览")
                out_depth = gr.Image(type="pil", label="传感器深度（伪彩色）", height=180)
                out_mask = gr.Image(type="pil", label="SAM3 实例 mask", height=180)
                out_bbox = gr.Image(type="pil", label="SAM3 实例 bbox", height=180)
                out_pose_img = gr.Image(
                    type="pil",
                    label="位姿叠加（坐标轴 + 3D 尺寸框）",
                    height=260,
                )
                out_place_img = gr.Image(
                    type="pil",
                    label="目的位姿叠加（品红=放置目标）",
                    height=260,
                )
                out_place_summary = gr.Textbox(
                    label="缺货摆放：选用目标 & 如何移动",
                    value=_empty_place_summary(),
                    lines=10,
                    interactive=False,
                )

        gr.Markdown("### 当前实例点云 + 位姿")
        with gr.Row():
            with gr.Column(scale=4):
                out_glb = gr.Model3D(label="当前位姿 GLB", height=440, display_mode="solid")
            with gr.Column(scale=1):
                gr.Markdown("**下载**")
                out_glb_dl = gr.File(label="GLB", interactive=False)
                out_ply_dl = gr.File(label="PLY", interactive=False)

        gr.Markdown("### 目的放置点云（品红实例）")
        with gr.Row():
            with gr.Column(scale=4):
                out_place_glb = gr.Model3D(
                    label="目的位姿 GLB（品红=移到缺货位的实例）",
                    height=440,
                    display_mode="solid",
                )
            with gr.Column(scale=1):
                gr.Markdown("**下载**")
                out_place_glb_dl = gr.File(label="目的 GLB", interactive=False)
                out_place_ply_dl = gr.File(label="目的 PLY", interactive=False)

        out_grasp = gr.Code(
            label="抓取位姿 grasp_pose.json（xyzrxryrz mm/° + 目标正方体）",
            language="json",
            lines=10,
            value=_empty_grasp_json(),
        )
        out_poses = gr.Code(
            label="poses.json（当前检测）",
            language="json",
            lines=10,
            value=_empty_poses(),
        )
        out_place_6d = gr.Code(
            label="目的 6D（destination xyzrxryrz）",
            language="json",
            lines=12,
            value=_empty_place_6d(),
        )
        out_json = gr.Code(label="详情", language="json", lines=12)

        btn_identify.click(
            fn=run_identify_missing,
            inputs=[
                rgb,
                identify_prompt,
                reason_vlm_api,
                reason_vlm_model,
                reason_vlm_timeout,
            ],
            outputs=[product_name, identify_raw, out_json],
        )
        btn_gen_prompt.click(
            fn=generate_prompt_ui,
            inputs=[
                rgb,
                product_name,
                sam3_vlm_api,
                sam3_vlm_model,
                sam3_vlm_timeout,
            ],
            outputs=[sam3_prompt, vlm_status],
        )
        btn_run.click(
            fn=run_missing_pose_pipeline,
            inputs=[
                rgb,
                depth,
                camera,
                product_name,
                auto_identify,
                identify_prompt,
                identify_raw,
                sam3_prompt,
                use_vlm_sam3_prompt,
                sam3_vlm_api,
                sam3_vlm_model,
                sam3_vlm_timeout,
                reason_vlm_api,
                reason_vlm_model,
                reason_vlm_timeout,
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
                product_name,
                identify_raw,
                sam3_prompt,
                vlm_status,
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
                out_place_img,
                out_place_glb,
                out_place_glb_dl,
                out_place_ply_dl,
                out_place_6d,
                out_place_summary,
            ],
        )

        gr.Markdown(
            """
            **说明**
            - **缺货识别 / 放置位移**：MiniMax-M3（`vlm.reason`）
            - **SAM3 提示词**：本地 qwen3-vl-4b（`vlm.sam3_prompt`）
            - **全流程**：缺货名(M3) → SAM3 提示词(qwen) → SAM3 → GenPose2 → 放置位移(M3) → 目的 6D
            - **放置位移**会附带第①步「识别模型原始回复」作为空位上下文
            - **目的可视化**：2D 品红框/轴；另一份 GLB 中品红点云为平移后的实例
            - **Depth→RGB 对齐**默认开启；产物在 `output/ui_runs/`（含 `place_destination.json`）
            """
        )
