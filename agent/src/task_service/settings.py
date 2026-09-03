"""Configuration for the unified task API and web console."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from runtime_config import RobotSettings, ServerSettings, load_runtime_document
from task0_service.models import Task0Settings
from task1_service.models import Task1Settings
from task2_service.models import Task2Settings
from task3_service.models import Task3Settings


class TaskConfigSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task0: Task0Settings
    task1: Task1Settings
    task2: Task2Settings
    task3: Task3Settings


class WebServiceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pick_url: str
    place_url: str
    locate_url: str
    perception_url: str
    navigation_url: str
    pose_url: str

    @field_validator("*")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("service URL must start with http:// or https://")
        return value


class WebPathSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_dir: Path
    locate_image_roots: list[Path] = Field(min_length=1)


class WebSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    services: WebServiceSettings
    request_timeout_seconds: float = Field(gt=0)
    paths: WebPathSettings


class ExternalServiceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    access_token: str | None = None
    callback_url: str | None = None
    callback_access_token: str | None = None
    callback_allowed_hosts: list[str] = Field(default_factory=list)
    request_timeout_seconds: float = Field(gt=0, default=5)
    heartbeat_seconds: float = Field(gt=0, default=5)
    max_retries: int = Field(ge=0, default=3)
    retry_backoff_seconds: float = Field(ge=0, default=1)


class TaskServiceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    config_path: Path
    robot: RobotSettings
    server: ServerSettings
    tasks: TaskConfigSettings
    web: WebSettings
    external: ExternalServiceSettings = Field(default_factory=ExternalServiceSettings)
    pick_place_status_url: str

    @classmethod
    def load(cls, path: str | Path) -> "TaskServiceSettings":
        document = load_runtime_document(path)

        task0 = document.task_section("task0", Task0Settings)
        task0["services"] = {
            "navigation": document.robot.navigation_url,
            "pose": document.robot.pose_url,
            "camera": document.robot.camera_url,
        }
        for field_name in ("output_dir", "log_dir"):
            if field_name in task0:
                task0[field_name] = document.resolve(str(task0[field_name]))

        task1 = document.task_section("task1", Task1Settings)
        task1["services"] = {
            "navigation": document.robot.navigation_url,
            "pose": document.robot.pose_url,
            "pick_place": document.local_services.pick_place,
            "sku": document.local_services.sku,
        }
        if "log_dir" in task1:
            task1["log_dir"] = document.resolve(str(task1["log_dir"]))

        task2 = document.task_section("task2", Task2Settings)
        task2["services"] = {
            "navigation": document.robot.navigation_url,
            "perception": document.local_services.perception,
            "pose": document.robot.pose_url,
            "pick_place": document.local_services.pick_place,
            "camera": document.robot.camera_url,
        }
        for field_name in ("baseline_dir", "log_dir"):
            if field_name in task2:
                task2[field_name] = document.resolve(str(task2[field_name]))

        task3 = document.task_section("task3", Task3Settings)
        task3["services"] = {
            "navigation": document.robot.navigation_url,
            "perception": document.local_services.perception,
            "pose": document.robot.pose_url,
            "pick_place": document.local_services.pick_place,
            "sku": document.local_services.sku,
            "camera": document.robot.camera_url,
        }
        for field_name in ("baseline_dir", "log_dir"):
            if field_name in task3:
                task3[field_name] = document.resolve(str(task3[field_name]))

        web = document.section("web")
        paths = web.get("paths")
        if not isinstance(paths, dict):
            raise RuntimeError("runtime web.paths must be a YAML object")
        paths = dict(paths)
        paths["log_dir"] = document.resolve(str(paths["log_dir"]))
        roots = paths.get("locate_image_roots")
        if not isinstance(roots, list):
            raise RuntimeError("runtime web.paths.locate_image_roots must be a list")
        paths["locate_image_roots"] = [document.resolve(str(item)) for item in roots]
        web["paths"] = paths
        web["services"] = {
            "pick_url": f"{document.local_services.pick_place}/pick",
            "place_url": f"{document.local_services.pick_place}/place",
            "locate_url": f"{document.local_services.perception}/perception/pick/locate",
            "perception_url": document.local_services.perception,
            "navigation_url": document.robot.navigation_url,
            "pose_url": document.robot.pose_url,
        }

        external_raw = document.raw.get("external", {})
        if not isinstance(external_raw, dict):
            raise RuntimeError("runtime external section must be a YAML object")

        return cls(
            config_path=document.path,
            robot=document.robot,
            server=document.servers.tasks,
            tasks=TaskConfigSettings(
                task0=Task0Settings.from_mapping(task0),
                task1=Task1Settings.from_mapping(task1, document.path.parent),
                task2=Task2Settings.from_mapping(task2, document.path.parent),
                task3=Task3Settings.from_mapping(task3, document.path.parent),
            ),
            web=WebSettings.model_validate(web),
            external=ExternalServiceSettings.model_validate(external_raw),
            pick_place_status_url=f"{document.local_services.pick_place}/status",
        )
