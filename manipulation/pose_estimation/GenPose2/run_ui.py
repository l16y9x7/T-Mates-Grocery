"""Gradio UI entry: SAM3 分割 / SAM3+GenPose2 / 缺货商品位姿估计。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _parse_ui_args(argv: list[str] | None = None):
    """先解析 UI 参数，再清空 sys.argv，避免 GenPose2 ``get_config()`` 吃掉 --host/--port。"""
    parser = argparse.ArgumentParser(description="GenPose2 Gradio UI")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--skip-preload",
        action="store_true",
        help="跳过启动时预加载 GenPose2（不推荐；worker 线程首次加载 CUDA 可能假死）",
    )
    args = parser.parse_args(argv)
    # networks/pts_encoder/pointnet2.py 在 import 时会调用 get_config() 解析全局 argv
    sys.argv = [sys.argv[0] if sys.argv else "run_ui.py"]
    return args


def build_app():
    import gradio as gr
    from ui.genpose_tab import build_sam3_genpose_tab
    from ui.misplaced_tab import build_misplaced_tab
    from ui.sam3_tab import build_sam3_tab

    with gr.Blocks(title="GenPose2") as demo:
        gr.Markdown("# GenPose2")
        gr.Markdown(
            "页签1：**SAM3 分割**；"
            "页签2：**SAM3 + GenPose2**；"
            "页签3：**缺货商品位姿估计**（手填/识别商品名 → SAM3 → GenPose2）。"
            "外部依赖见 `config/conf.json`。"
        )
        with gr.Tabs():
            build_sam3_tab()
            build_sam3_genpose_tab()
            build_misplaced_tab()
    return demo


def main():
    args = _parse_ui_args()

    # 主线程预加载：避免 Gradio worker 里首次 init CUDA 卡住
    if not args.skip_preload:
        print("[run_ui] preloading GenPose2 on main thread ...", flush=True)
        from ui.genpose_runner import preload_genpose2

        preload_genpose2()
        print("[run_ui] preload done", flush=True)

    demo = build_app()
    print(f"[run_ui] launching Gradio on http://{args.host}:{args.port}/ ...", flush=True)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(ROOT_DIR / "output")],
        ssr_mode=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
