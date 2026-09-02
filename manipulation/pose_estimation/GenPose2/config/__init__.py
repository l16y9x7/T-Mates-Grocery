"""Load project external dependency config from ``config/conf.json``."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path(__file__).resolve().parent
CONF_PATH = CONFIG_DIR / "conf.json"
SECRETS_PATH = CONFIG_DIR / "secrets.local.json"
ROOT_DIR = CONFIG_DIR.parent


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=1)
def load_conf(path: str | None = None) -> Dict[str, Any]:
    conf_path = Path(path).expanduser().resolve() if path else CONF_PATH
    with conf_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {conf_path}")
    # Optional local secrets (API keys); not committed
    if SECRETS_PATH.is_file() and path is None:
        try:
            secrets = json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
            if isinstance(secrets, dict):
                data = _deep_merge(data, secrets)
        except Exception:  # noqa: BLE001
            pass
    return data


def get_sam3_conf() -> Dict[str, Any]:
    return dict(load_conf().get("sam3") or {})


def get_vlm_conf() -> Dict[str, Any]:
    """Raw ``vlm`` section (may contain nested ``sam3_prompt`` / ``reason``)."""
    cfg = dict(load_conf().get("vlm") or {})
    # Top-level key still honored (secrets.local.json → vlm.api_key)
    for env_key in ("ANTHROPIC_API_KEY", "MINIMAX_API_KEY", "VLM_API_KEY"):
        val = (os.environ.get(env_key) or "").strip()
        if val:
            cfg["api_key"] = val
            break
    return cfg


def get_vlm_profile(name: str = "reason") -> Dict[str, Any]:
    """Return one VLM endpoint profile.

    - ``sam3_prompt``: local qwen3-vl for SAM3 text prompts
    - ``reason``: MiniMax-M3 for missing-product / place-offset reasoning
    """
    cfg = get_vlm_conf()
    key = (name or "reason").strip().lower()
    aliases = {
        "sam3": "sam3_prompt",
        "sam3_prompt": "sam3_prompt",
        "prompt": "sam3_prompt",
        "qwen": "sam3_prompt",
        "reason": "reason",
        "missing": "reason",
        "place": "reason",
        "m3": "reason",
        "minimax": "reason",
        "advanced": "reason",
    }
    profile_key = aliases.get(key, key)
    nested = cfg.get(profile_key)
    if isinstance(nested, dict) and nested:
        out = dict(nested)
    elif profile_key == "reason" and cfg.get("api_url"):
        # Backward compat: flat vlm.{api_url,model,...}
        out = {
            k: cfg[k]
            for k in (
                "provider",
                "api_url",
                "model",
                "timeout_s",
                "temperature",
                "max_tokens",
                "api_key",
            )
            if k in cfg
        }
    else:
        out = {}

    # Inherit top-level api_key (from secrets / env)
    if not out.get("api_key") and cfg.get("api_key"):
        out["api_key"] = cfg["api_key"]

    if profile_key == "reason":
        base = (os.environ.get("ANTHROPIC_BASE_URL") or "").strip()
        if base:
            out["api_url"] = base
        model = (os.environ.get("VLM_REASON_MODEL") or os.environ.get("VLM_MODEL") or "").strip()
        if model:
            out["model"] = model
        out.setdefault("provider", "anthropic")
        out.setdefault("api_url", "https://api.minimaxi.com/anthropic")
        out.setdefault("model", "MiniMax-M3")
        out.setdefault("timeout_s", 120.0)
        out.setdefault("temperature", 0.2)
        out.setdefault("max_tokens", 1024)
    else:
        url = (os.environ.get("VLM_SAM3_API_URL") or "").strip()
        if url:
            out["api_url"] = url
        model = (os.environ.get("VLM_SAM3_MODEL") or "").strip()
        if model:
            out["model"] = model
        out.setdefault("provider", "openai")
        out.setdefault(
            "api_url", "http://192.168.130.88:8000/v1/chat/completions"
        )
        out.setdefault("model", "qwen3-vl-4b")
        out.setdefault("timeout_s", 60.0)
        out.setdefault("temperature", 0.1)
        out.setdefault("max_tokens", 128)

    return out


def get_genpose2_conf() -> Dict[str, Any]:
    return dict(load_conf().get("genpose2") or {})


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    else:
        path = path.resolve()
    return path
