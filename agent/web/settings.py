"""Load and validate the web console YAML configuration."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


WEB_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("WEB_CONFIG_FILE", WEB_ROOT / "config.yaml")).expanduser().resolve()


class ServerSettings(BaseModel):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)


class ServiceSettings(BaseModel):
    pick_url: str
    task1_url: str
    locate_url: str
    perception_url: str
    navigation_url: str
    pose_url: str

    @field_validator("*")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("服务地址必须以 http:// 或 https:// 开头")
        return value


class PathSettings(BaseModel):
    log_dir: str = Field(min_length=1)
    locate_image_roots: list[str] = Field(min_length=1)


class WebSettings(BaseModel):
    server: ServerSettings
    services: ServiceSettings
    request_timeout_seconds: float = Field(gt=0)
    paths: PathSettings


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = CONFIG_PATH.parent / path
    return path.resolve()


def load_settings() -> WebSettings:
    try:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Web 配置文件不存在: {CONFIG_PATH}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Web 配置文件读取失败: {CONFIG_PATH}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"Web 配置文件内容必须是 YAML 对象: {CONFIG_PATH}")
    return WebSettings.model_validate(raw)


SETTINGS = load_settings()
LOG_ROOT = _resolve_path(SETTINGS.paths.log_dir)
LOCATE_IMAGE_ROOTS = tuple(_resolve_path(item) for item in SETTINGS.paths.locate_image_roots)


if __name__ == "__main__":
    print(f"{SETTINGS.server.host}\t{SETTINGS.server.port}")
