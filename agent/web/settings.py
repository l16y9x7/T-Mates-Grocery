"""Default unified-service settings used when importing the web module in tests."""

from __future__ import annotations

import os
from pathlib import Path

from task_service.settings import TaskServiceSettings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(
    os.environ.get(
        "RUNTIME_CONFIG_FILE", PROJECT_ROOT / "config" / "runtime.production.yaml"
    )
).expanduser().resolve()

UNIFIED_SETTINGS = TaskServiceSettings.load(CONFIG_PATH)
SETTINGS = UNIFIED_SETTINGS.web
LOG_ROOT = SETTINGS.paths.log_dir
LOCATE_IMAGE_ROOTS = tuple(SETTINGS.paths.locate_image_roots)
