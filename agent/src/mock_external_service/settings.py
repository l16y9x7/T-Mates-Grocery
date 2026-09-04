"""Settings for the standalone external API mock."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MockExternalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = Field(default=8109, ge=1, le=65535)
    access_token: str | None = None
    callback_url: str | None = None
    callback_access_token: str | None = None
    stage_delay_seconds: float = Field(default=1, ge=0)
    request_timeout_seconds: float = Field(default=5, gt=0)
    max_retries: int = Field(default=5, ge=0)
    retry_backoff_seconds: float = Field(default=1, ge=0)
    inspection_points: list[str] = Field(
        default_factory=lambda: [
            "H1_INSPECT",
            "H12_INSPECT",
            "H2_INSPECT",
            "H23_INSPECT",
            "H3_INSPECT",
        ],
        min_length=1,
    )

    @field_validator("callback_url")
    @classmethod
    def trim_callback_url(cls, value: str | None) -> str | None:
        return value.strip() if value else None

    @classmethod
    def load(cls, path: str | Path) -> "MockExternalSettings":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        if not isinstance(document, dict):
            raise RuntimeError("mock runtime config must be a YAML object")
        section = document.get("mock_external", document)
        if not isinstance(section, dict):
            raise RuntimeError("mock_external config must be a YAML object")
        return cls.model_validate(section)

