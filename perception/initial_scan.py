"""Resolve and load fixed initial shelf RGB-D scans produced by task0."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PERCEPTION_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PERCEPTION_ROOT.parent
DEFAULT_INITIAL_SCAN_ROOT = REPOSITORY_ROOT / "agent" / "output" / "task0"
DEFAULT_SLOT_MAPPING_PATH = (
    REPOSITORY_ROOT / "agent" / "config" / "product-hand-options.yaml"
)
INITIAL_SCAN_ROOT_ENVIRONMENT = "INITIAL_SCAN_ROOT"
SLOT_MAPPING_ENVIRONMENT = "PRODUCT_HAND_OPTIONS_PATH"

SLOT_PATTERN = re.compile(r"^H[12]_[FB]_L(?P<level>[1-5])_C\d{2}$")
INSPECTION_TARGET_PATTERN = re.compile(r"^H[12]_[FB]_[LR]_INSPECT$")
MAPPING_LINE_PATTERN = re.compile(
    r"^\s*(?P<slot>H[12]_[FB]_L[1-5]_C\d{2})\s*:.*?"
    r"\btarget_id\s*:\s*(?P<target>H[12]_[FB]_[LR]_INSPECT)\s*[},]"
)


class InitialScanError(ValueError):
    """Raised when a fixed task0 scan cannot be resolved or validated."""


@dataclass(frozen=True)
class InitialScan:
    inspection_target_id: str
    pose_type: str
    directory: Path
    rgb_path: Path
    depth_path: Path
    rgb: np.ndarray
    depth_mm: np.ndarray
    metadata: dict[str, Any]


def initial_scan_root() -> Path:
    configured = os.getenv(INITIAL_SCAN_ROOT_ENVIRONMENT, "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_INITIAL_SCAN_ROOT


def slot_mapping_path() -> Path:
    configured = os.getenv(SLOT_MAPPING_ENVIRONMENT, "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_SLOT_MAPPING_PATH


def normalize_scan_pose(pose_type: str, location_id: str) -> str:
    """Return SHELF_VIEW_UPPER/LOWER, deriving it from a slot when omitted."""

    normalized = pose_type.strip().upper()
    aliases = {
        "UPPER": "SHELF_VIEW_UPPER",
        "LOWER": "SHELF_VIEW_LOWER",
        "SHELF_VIEW_UPPER": "SHELF_VIEW_UPPER",
        "SHELF_VIEW_LOWER": "SHELF_VIEW_LOWER",
    }
    if normalized:
        try:
            return aliases[normalized]
        except KeyError as error:
            raise InitialScanError(f"unsupported inspection pose: {pose_type}") from error

    slot_match = SLOT_PATTERN.fullmatch(location_id.strip().upper())
    if slot_match is None:
        raise InitialScanError(
            "pose_type is required when location_id is an inspection target"
        )
    level = int(slot_match.group("level"))
    return "SHELF_VIEW_UPPER" if level <= 2 else "SHELF_VIEW_LOWER"


def load_slot_target_mapping(path: str | Path | None = None) -> dict[str, str]:
    """Read only the slot -> target_id fields from agent's simple YAML mapping."""

    source = Path(path) if path is not None else slot_mapping_path()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise InitialScanError(f"cannot read slot mapping: {source}: {error}") from error
    mapping: dict[str, str] = {}
    for line in lines:
        match = MAPPING_LINE_PATTERN.match(line)
        if match is not None:
            mapping[match.group("slot")] = match.group("target")
    if not mapping:
        raise InitialScanError(f"slot mapping contains no target_id entries: {source}")
    return mapping


def resolve_inspection_target_id(
    location_id: str,
    *,
    mapping_path: str | Path | None = None,
) -> str:
    """Resolve a product slot or accept an explicit inspection navigation target."""

    normalized = location_id.strip().upper()
    if INSPECTION_TARGET_PATTERN.fullmatch(normalized):
        return normalized
    if SLOT_PATTERN.fullmatch(normalized) is None:
        raise InitialScanError(
            "location_id must be a product slot or inspection target, got "
            f"{location_id!r}"
        )
    mapping = load_slot_target_mapping(mapping_path)
    try:
        return mapping[normalized]
    except KeyError as error:
        raise InitialScanError(f"no inspection target configured for {normalized}") from error


def resolve_initial_scan_directory(
    location_id: str,
    pose_type: str,
    *,
    root: str | Path | None = None,
    mapping_path: str | Path | None = None,
) -> tuple[Path, str, str]:
    """Resolve task0/<target>_UPPER|LOWER without accepting arbitrary paths."""

    target_id = resolve_inspection_target_id(location_id, mapping_path=mapping_path)
    normalized_pose = normalize_scan_pose(pose_type, location_id)
    pose_suffix = "UPPER" if normalized_pose == "SHELF_VIEW_UPPER" else "LOWER"
    scan_root = Path(root) if root is not None else initial_scan_root()
    directory = scan_root / f"{target_id}_{pose_suffix}"
    return directory, target_id, normalized_pose


def _safe_child(directory: Path, filename: object, field_name: str) -> Path:
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise InitialScanError(f"meta.json {field_name} must be a plain filename")
    return directory / filename


def _read_rgb(path: Path) -> np.ndarray:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as error:
        raise InitialScanError(f"cannot read initial RGB: {path}: {error}") from error
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise InitialScanError(f"initial RGB is not a valid image: {path}")
    return image


def _read_depth(path: Path) -> np.ndarray:
    try:
        depth = np.load(path, allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise InitialScanError(f"cannot read initial depth: {path}: {error}") from error
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        raise InitialScanError(f"initial depth must be a two-dimensional NPY: {path}")
    if not np.issubdtype(depth.dtype, np.number):
        raise InitialScanError(f"initial depth must be numeric: {path}")
    depth_mm = depth.astype(np.float32)
    if not np.isfinite(depth_mm).all() or np.any(depth_mm < 0):
        raise InitialScanError(f"initial depth contains invalid values: {path}")
    return depth_mm


def load_initial_scan(
    location_id: str,
    pose_type: str,
    *,
    root: str | Path | None = None,
    mapping_path: str | Path | None = None,
) -> InitialScan:
    """Load and validate one fixed task0 RGB-D record."""

    directory, target_id, normalized_pose = resolve_initial_scan_directory(
        location_id,
        pose_type,
        root=root,
        mapping_path=mapping_path,
    )
    metadata_path = directory / "meta.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InitialScanError(
            f"cannot read initial scan metadata: {metadata_path}: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise InitialScanError(f"initial scan metadata must be an object: {metadata_path}")
    rgb_metadata = metadata.get("rgb")
    depth_metadata = metadata.get("depth")
    if not isinstance(rgb_metadata, dict) or not isinstance(depth_metadata, dict):
        raise InitialScanError(f"meta.json is missing rgb/depth objects: {metadata_path}")
    if depth_metadata.get("aligned_to") != "rgb":
        raise InitialScanError(f"initial depth is not aligned to RGB: {metadata_path}")
    if depth_metadata.get("unit") != "millimeter":
        raise InitialScanError(f"initial depth unit must be millimeter: {metadata_path}")

    rgb_path = _safe_child(directory, rgb_metadata.get("file"), "rgb.file")
    depth_path = _safe_child(directory, depth_metadata.get("file"), "depth.file")
    rgb = _read_rgb(rgb_path)
    depth_mm = _read_depth(depth_path)
    if depth_mm.shape != rgb.shape[:2]:
        raise InitialScanError(
            "initial RGB/depth size mismatch: "
            f"rgb={rgb.shape[:2]}, depth={depth_mm.shape}"
        )
    metadata_size = (metadata.get("height"), metadata.get("width"))
    if metadata_size != rgb.shape[:2]:
        raise InitialScanError(
            f"meta.json size {metadata_size} does not match RGB {rgb.shape[:2]}"
        )
    return InitialScan(
        inspection_target_id=target_id,
        pose_type=normalized_pose,
        directory=directory.resolve(),
        rgb_path=rgb_path.resolve(),
        depth_path=depth_path.resolve(),
        rgb=rgb,
        depth_mm=depth_mm,
        metadata=metadata,
    )
