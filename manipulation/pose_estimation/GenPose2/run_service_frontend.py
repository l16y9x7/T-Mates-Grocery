#!/usr/bin/env python3
"""Launch the independent, model-free pose service test frontend."""

from __future__ import annotations

import argparse
import logging


def parse_args() -> argparse.Namespace:
    """Parse bind settings without importing Gradio or model packages."""

    parser = argparse.ArgumentParser(
        description="Service-based RGB-D pose inference test frontend"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=18086, help="Bind port")
    return parser.parse_args()


def main() -> None:
    """Build and launch the Gradio application."""

    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from ui.service_frontend import build_service_frontend

    app = build_service_frontend()
    app.queue(default_concurrency_limit=1, max_size=16).launch(
        server_name=args.host,
        server_port=args.port,
        ssr_mode=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
