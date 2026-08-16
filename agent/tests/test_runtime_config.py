from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from pick_place_service.models import PickPlaceSettings
from task_service.settings import TaskServiceSettings


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
RUNTIME_CONFIG = CONFIG_DIR / "runtime.production.yaml"


def _runtime_copy(tmp_path: Path, robot_ip: str) -> Path:
    raw = yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    raw["robot"]["ip"] = robot_ip
    config_path = tmp_path / "runtime.production.yaml"
    config_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(CONFIG_DIR / "product-hand-options.yaml", tmp_path)
    return config_path


def test_one_robot_ip_drives_every_robot_service_url(tmp_path: Path) -> None:
    robot_ip = "10.21.32.43"
    config_path = _runtime_copy(tmp_path, robot_ip)
    tasks = TaskServiceSettings.load(config_path)
    pick_place = PickPlaceSettings.load(config_path)

    assert str(tasks.robot.ip) == robot_ip
    assert tasks.tasks.task0.services.navigation == f"http://{robot_ip}:8081"
    assert tasks.tasks.task0.services.pose == f"http://{robot_ip}:8084"
    assert tasks.tasks.task0.services.camera == f"http://{robot_ip}:8085"
    for task in (tasks.tasks.task1, tasks.tasks.task2, tasks.tasks.task3):
        assert task.services.navigation == f"http://{robot_ip}:8081"
        assert task.services.pose == f"http://{robot_ip}:8084"
        assert task.start_target_id == "start"
    assert tasks.tasks.task2.services.camera == f"http://{robot_ip}:8085"
    assert tasks.tasks.task3.services.camera == f"http://{robot_ip}:8085"
    assert tasks.web.services.navigation_url == f"http://{robot_ip}:8081"
    assert tasks.web.services.pose_url == f"http://{robot_ip}:8084"
    assert pick_place.manipulation_url == f"http://{robot_ip}:8084"
    assert pick_place.camera_url == f"http://{robot_ip}:8085"

    assert tasks.tasks.task1.services.perception == "http://127.0.0.1:8083"
    assert tasks.tasks.task1.services.pick_place == "http://127.0.0.1:8086"
    assert tasks.tasks.task1.services.sku == "http://127.0.0.1:25540"
    assert pick_place.pose_estimation_url == "http://127.0.0.1:8084"


def test_production_runtime_uses_one_yaml_and_external_product_map() -> None:
    settings = TaskServiceSettings.load(RUNTIME_CONFIG)
    pick_place = PickPlaceSettings.load(RUNTIME_CONFIG)

    assert settings.server.port == 8108
    assert pick_place.pick_cameras == {
        "left": "left_wrist",
        "right": "right_wrist",
    }
    assert len(settings.tasks.task1.product_hand_options) == 122
    assert Path(pick_place.calibration_files["head"]) == CONFIG_DIR / "camera/head.json"


def test_production_runtime_shares_same_named_task_settings() -> None:
    raw = yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    shared = raw["tasks"]["shared"]
    settings = TaskServiceSettings.load(RUNTIME_CONFIG).tasks

    assert settings.task0.inspection_points == settings.task2.inspection_points
    assert [point.target_id for point in settings.task3.inspection_points] == shared[
        "inspection_points"
    ]
    assert settings.task0.camera == settings.task2.camera == settings.task3.camera
    assert (
        settings.task1.task_boundary
        == settings.task2.task_boundary
        == settings.task3.task_boundary
    )
    assert (
        settings.task0.start_target_id
        == settings.task1.start_target_id
        == settings.task2.start_target_id
        == settings.task3.start_target_id
    )
    assert settings.task1.log_dir == settings.task2.log_dir == settings.task3.log_dir
    assert settings.task0.timeouts.navigation_seconds == settings.task3.timeouts.navigation_seconds
    for task_name in ("task0", "task1", "task2", "task3", "test1"):
        task_section = raw["tasks"][task_name]
        assert "log_dir" not in task_section
        assert "inspection_points" not in task_section
