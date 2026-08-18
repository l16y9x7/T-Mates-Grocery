"""Shared loader for the production runtime configuration."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RobotPorts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    navigation: int = Field(ge=1, le=65535)
    pose: int = Field(ge=1, le=65535)
    camera: int = Field(ge=1, le=65535)


class RobotSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ip: IPv4Address
    ports: RobotPorts

    @property
    def navigation_url(self) -> str:
        return f"http://{self.ip}:{self.ports.navigation}"

    @property
    def pose_url(self) -> str:
        return f"http://{self.ip}:{self.ports.pose}"

    @property
    def camera_url(self) -> str:
        return f"http://{self.ip}:{self.ports.camera}"


class LocalServices(BaseModel):
    model_config = ConfigDict(extra="forbid")

    perception: str
    pose_estimation: str
    pick_place: str
    sku: str

    @field_validator("*")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("service URL must start with http:// or https://")
        return value


class ServerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)


class RuntimeServers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: ServerSettings
    pick_place: ServerSettings


@dataclass(frozen=True)
class RuntimeDocument:
    path: Path
    raw: dict[str, Any]
    robot: RobotSettings
    local_services: LocalServices
    servers: RuntimeServers

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise RuntimeError(f"runtime config section must be a YAML object: {name}")
        return deepcopy(value)

    def task_section(
        self, name: str, settings_type: type[BaseModel]
    ) -> dict[str, Any]:
        """Merge task-wide shared values into one task's supported fields."""

        tasks = self.section("tasks")
        shared = tasks.get("shared", {})
        if not isinstance(shared, dict):
            raise RuntimeError("runtime task section must be a YAML object: shared")
        task = tasks.get(name)
        if not isinstance(task, dict):
            raise RuntimeError(f"runtime task section must be a YAML object: {name}")

        values = _values_supported_by_model(shared, settings_type)
        return _merge_mappings(values, task)

    def resolve(self, value: str | Path) -> str:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.path.parent / candidate
        return str(candidate.resolve())


def validate_runtime_raw(path: Path, raw: object) -> RuntimeDocument:
    if not isinstance(raw, dict):
        raise RuntimeError(f"runtime config must be a YAML object: {path}")
    try:
        robot = RobotSettings.model_validate(raw.get("robot"))
        local_services = LocalServices.model_validate(raw.get("local_services"))
        servers = RuntimeServers.model_validate(raw.get("servers"))
    except ValueError as exc:
        raise RuntimeError(f"invalid runtime config {path}: {exc}") from exc
    for section_name in ("tasks", "pick_place", "web"):
        if not isinstance(raw.get(section_name), dict):
            raise RuntimeError(
                f"runtime config section must be a YAML object: {section_name}"
            )
    return RuntimeDocument(
        path=path,
        raw=deepcopy(raw),
        robot=robot,
        local_services=local_services,
        servers=servers,
    )


def load_runtime_document(path: str | Path) -> RuntimeDocument:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"runtime config not found: {config_path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read runtime config {config_path}: {exc}") from exc
    # A process-level override lets launch scripts target a robot without
    # rewriting the shared production configuration on disk.
    robot_ip = os.environ.get("ROBOT_IP", "").strip()
    if robot_ip:
        if not isinstance(raw, dict):
            raise RuntimeError(f"runtime config must be a YAML object: {config_path}")
        raw = deepcopy(raw)
        robot = raw.setdefault("robot", {})
        if not isinstance(robot, dict):
            raise RuntimeError(f"runtime config robot section must be a YAML object: {config_path}")
        robot["ip"] = robot_ip
    return validate_runtime_raw(config_path, raw)


def _values_supported_by_model(
    values: dict[str, Any], model_type: type[BaseModel]
) -> dict[str, Any]:
    """Select shared values understood by a strict task settings model."""

    selected: dict[str, Any] = {}
    for name, field in model_type.model_fields.items():
        if name not in values:
            continue
        value = values[name]
        nested_type = field.annotation
        if (
            isinstance(value, dict)
            and isinstance(nested_type, type)
            and issubclass(nested_type, BaseModel)
        ):
            value = _values_supported_by_model(value, nested_type)
        selected[name] = deepcopy(value)
    return selected


def _merge_mappings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for name, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(name), dict):
            merged[name] = _merge_mappings(merged[name], value)
        else:
            merged[name] = deepcopy(value)
    return merged
