from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ui.service_visualization import (
    euler_zyx_matrix,
    export_pose_scene,
    project_camera_points,
    render_box,
    render_depth_colormap,
    render_mask,
    render_mask_overlay,
    render_pose_overlay,
)


def _pose_response() -> dict[str, object]:
    return {
        "pose": [0, 0, 1000, 0, 0, 0],
        "corners_mm": [
            [-100, -100, 900],
            [100, -100, 900],
            [100, 100, 900],
            [-100, 100, 900],
            [-100, -100, 1100],
            [100, -100, 1100],
            [100, 100, 1100],
            [-100, 100, 1100],
        ],
        "frame": "camera",
        "pose_unit": "mm_rad",
        "rotation_order": "zyx",
    }


def test_project_camera_points() -> None:
    intrinsics = np.array(
        [[100, 0, 50], [0, 100, 40], [0, 0, 1]], dtype=float
    )
    points = np.array([[0, 0, 1], [0.1, 0.2, 1]], dtype=float)
    assert project_camera_points(points, intrinsics).tolist() == [
        [50.0, 40.0],
        [60.0, 60.0],
    ]


def test_project_camera_points_marks_nonpositive_depth_invalid() -> None:
    intrinsics = np.eye(3, dtype=float)
    pixels = project_camera_points(np.array([[1, 2, 0], [1, 2, -1]]), intrinsics)
    assert np.isnan(pixels).all()


def test_euler_zyx_identity() -> None:
    assert np.allclose(euler_zyx_matrix(0, 0, 0), np.eye(3))


def test_renderers_preserve_image_shape() -> None:
    image = Image.new("RGB", (100, 80), color=(20, 30, 40))
    mask = np.zeros((80, 100), dtype=bool)
    mask[20:60, 30:70] = True
    depth = np.full((80, 100), 1000, dtype=np.uint16)
    intrinsics = np.array(
        [[100, 0, 50], [0, 100, 40], [0, 0, 1]], dtype=float
    )

    outputs = [
        render_box(image, [30, 20, 70, 60]),
        render_mask(mask),
        render_mask_overlay(image, mask),
        render_depth_colormap(depth),
        render_pose_overlay(image, _pose_response(), intrinsics),
    ]
    assert all(output.size == (100, 80) for output in outputs)


def test_export_pose_scene_writes_glb_and_ply(tmp_path: Path) -> None:
    rgb = np.full((4, 4, 3), [120, 80, 40], dtype=np.uint8)
    depth = np.full((4, 4), 1000, dtype=np.uint16)
    depth[0, 0] = 0
    intrinsics = np.array(
        [[100, 0, 1.5], [0, 100, 1.5], [0, 0, 1]], dtype=float
    )

    artifacts = export_pose_scene(
        rgb,
        depth,
        intrinsics,
        _pose_response(),
        tmp_path,
        max_points=100,
    )

    assert artifacts.num_points == 15
    assert artifacts.glb_path.is_file() and artifacts.glb_path.stat().st_size > 0
    assert artifacts.ply_path.is_file() and artifacts.ply_path.stat().st_size > 0
