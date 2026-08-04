"""Environment-backed client configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import ConfigurationError


DEFAULT_BASE_URL = "http://127.0.0.1:8102/v1"
DEFAULT_MODEL = "Qwen3-VL-4B-Instruct"
DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class Settings:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "Settings":
        raw_timeout = os.getenv(
            "QWEN_TIMEOUT_SECONDS",
            str(DEFAULT_TIMEOUT_SECONDS),
        )
        try:
            timeout_seconds = float(raw_timeout)
        except ValueError as exc:
            raise ConfigurationError(
                "QWEN_TIMEOUT_SECONDS 必须是数字。"
            ) from exc

        if timeout_seconds <= 0:
            raise ConfigurationError(
                "QWEN_TIMEOUT_SECONDS 必须大于 0。"
            )

        base_url = os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ConfigurationError(
                "QWEN_BASE_URL 必须以 http:// 或 https:// 开头。"
            )

        model = os.getenv("QWEN_MODEL", DEFAULT_MODEL).strip()
        if not model:
            raise ConfigurationError("QWEN_MODEL 不能为空。")

        api_key = os.getenv("QWEN_API_KEY")
        if api_key is not None:
            api_key = api_key.strip() or None

        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )

