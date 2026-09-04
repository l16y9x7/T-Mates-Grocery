"""Versioned publication and resolution for complete Task0 shelf scans."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


CURRENT_POINTER_NAME = "current.json"
MANIFEST_NAME = "manifest.json"
RUNS_DIRECTORY_NAME = "runs"
STORAGE_SCHEMA_VERSION = "1.0"


class Task0StorageError(ValueError):
    """Raised when the published Task0 scan pointer is invalid."""


def resolve_current_scan_root(
    output_dir: str | Path,
    *,
    allow_legacy: bool = True,
) -> Path:
    """Resolve the immutable scan selected by ``current.json``.

    A flat legacy Task0 directory remains readable when no pointer has been
    published yet, so existing captured data does not need a migration step.
    """

    root = Path(output_dir)
    pointer_path = root / CURRENT_POINTER_NAME
    if not pointer_path.is_file():
        if allow_legacy:
            return root
        raise Task0StorageError(f"Task0 current pointer is missing: {pointer_path}")

    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Task0StorageError(
            f"Task0 current pointer is not valid JSON: {pointer_path}"
        ) from exc
    if not isinstance(pointer, dict):
        raise Task0StorageError("Task0 current pointer must contain an object")
    if pointer.get("schema_version") != STORAGE_SCHEMA_VERSION:
        raise Task0StorageError("Task0 current pointer has an unsupported schema")

    scan_id = pointer.get("scan_id")
    run_directory = pointer.get("run_directory")
    if not isinstance(scan_id, str) or not scan_id:
        raise Task0StorageError("Task0 current pointer is missing scan_id")
    expected_directory = f"{RUNS_DIRECTORY_NAME}/{scan_id}"
    if run_directory != expected_directory:
        raise Task0StorageError(
            "Task0 current pointer run_directory does not match scan_id"
        )

    scan_root = (root / Path(run_directory)).resolve()
    runs_root = (root / RUNS_DIRECTORY_NAME).resolve()
    if scan_root.parent != runs_root or not scan_root.is_dir():
        raise Task0StorageError(
            f"Task0 current scan directory is missing or unsafe: {scan_root}"
        )

    manifest_path = scan_root / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Task0StorageError(
            f"Task0 current scan manifest is invalid: {manifest_path}"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != STORAGE_SCHEMA_VERSION
        or manifest.get("scan_id") != scan_id
        or manifest.get("complete") is not True
    ):
        raise Task0StorageError(
            f"Task0 current scan manifest is incomplete or inconsistent: {manifest_path}"
        )
    return scan_root


def publish_current_pointer(root: Path, payload: dict[str, Any]) -> Path:
    """Atomically replace the pointer selecting the complete current scan."""

    pointer_path = root / CURRENT_POINTER_NAME
    temporary_path = root / f".{CURRENT_POINTER_NAME}.{uuid4().hex}.tmp"
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, pointer_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return pointer_path
