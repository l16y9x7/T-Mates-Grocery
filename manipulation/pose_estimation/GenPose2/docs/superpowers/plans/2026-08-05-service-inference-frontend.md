# Service-based Pose Inference Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build an independent Gradio frontend on port 18086 that uses a true SAM3 box prompt, calls the existing GenPose2 HTTP pose endpoint, and visualizes the mask, pose, point cloud, raw responses, and per-stage timing without loading either model.

**Architecture:** Extend the existing SAM3 HTTP request with an optional box while preserving text and point prompts. Add a model-free frontend to the GenPose2 repository with separate HTTP-client, visualization, orchestration, and entrypoint modules. Use pure-function unit tests before GPU integration tests.

**Tech Stack:** Python 3.10, Gradio 6.21, requests, Pillow, NumPy, OpenCV, pycocotools, trimesh/GLB, pytest, existing SAM3 and GenPose2 HTTP services.

---

## File Map

SAM3 repository (/home/ubuntu/stephen/01-code/sam3):

- Modify run_server.py: parse optional box, dispatch box inference, preserve old behavior.
- Modify scripts/infer.py: run a true positive geometric box prompt with the cached processor.
- Create test/test_box_prompt_api.py: box validation and dispatch regression tests.

GenPose2 repository (/home/ubuntu/stephen/01-code/T-Mates-Grocery/manipulation/pose_estimation/GenPose2):

- Create ui/service_client.py: camera resolution, RLE decoding, HTTP clients, response validation.
- Create ui/service_visualization.py: 2D rendering, projection, point-cloud scene export.
- Create ui/service_frontend.py: click state, staged/full orchestration, Gradio layout.
- Create run_service_frontend.py: model-free entrypoint.
- Create tests/test_service_client.py, tests/test_service_visualization.py, tests/test_service_frontend.py.
- Modify README.md with startup and process-boundary documentation.

### Task 1: SAM3 Box Contract and Geometric Prompt

**Files:**
- Modify: /home/ubuntu/stephen/01-code/sam3/run_server.py
- Modify: /home/ubuntu/stephen/01-code/sam3/scripts/infer.py
- Create: /home/ubuntu/stephen/01-code/sam3/test/test_box_prompt_api.py

- [ ] **Step 1: Write failing request-validation tests**

~~~python
import pytest
import run_server

def test_coerce_box_accepts_four_numeric_values() -> None:
    assert run_server._coerce_box([120, 80, 420, 360]) == [
        120.0, 80.0, 420.0, 360.0
    ]

@pytest.mark.parametrize("value", [[], [1, 2, 3], [1, 2, 3, 4, 5], "1,2,3,4"])
def test_coerce_box_rejects_invalid_shape(value: object) -> None:
    with pytest.raises(ValueError, match="box must be"):
        run_server._coerce_box(value)
~~~

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

~~~bash
/home/ubuntu/miniconda3/envs/sam3/bin/python -m pytest test/test_box_prompt_api.py -v
~~~

Expected: FAIL because run_server._coerce_box does not exist.

- [ ] **Step 3: Add minimal server parsing**

~~~python
def _coerce_box(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("box must be [x1, y1, x2, y2]")
    box = [float(item) for item in value]
    if not all(math.isfinite(item) for item in box):
        raise ValueError("box coordinates must be finite")
    return box
~~~

Parse box from the request, pass it to run_sam3_segmentation, and echo the effective box in the response.

- [ ] **Step 4: Write failing image-bound box tests**

~~~python
from scripts import infer

def test_normalize_box_orders_and_clamps() -> None:
    assert infer._normalize_box_xyxy([40, 30, -5, 10], 32, 24) == [
        0.0, 10.0, 31.0, 23.0
    ]

def test_normalize_box_rejects_zero_area() -> None:
    with pytest.raises(ValueError, match="positive area"):
        infer._normalize_box_xyxy([10, 10, 10, 20], 32, 24)
~~~

- [ ] **Step 5: Run and confirm RED because the normalization helper is absent**

- [ ] **Step 6: Implement box normalization and the geometric branch**

~~~python
def _normalize_box_xyxy(
    box: List[float], image_width: int, image_height: int
) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(x1, 0.0), float(image_width - 1))
    x2 = min(max(x2, 0.0), float(image_width - 1))
    y1 = min(max(y1, 0.0), float(image_height - 1))
    y2 = min(max(y2, 0.0), float(image_height - 1))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive area inside the image")
    return [x1, y1, x2, y2]
~~~

Extend run_sam3_segmentation with box: Optional[List[float]] = None. For a box request, set the image once, reset prompts, convert xyxy to normalized cxcywh, and call:

~~~python
state = processor.add_geometric_prompt(
    state=state,
    box=normalized_box_cxcywh,
    label=True,
)
scores = state["scores"].detach().float().cpu().reshape(-1)
selected_index = int(torch.argmax(scores).item())
selected_mask = _extract_geometric_mask(
    state["masks"], selected_index, image_width, image_height
)
raw_masks = [selected_mask]
raw_scores = [float(scores[selected_index].item())]
raw_boxes = [_bbox_from_mask(selected_mask)]
~~~

Use the proven mask extraction shape handling from sam3_box_tools/infer_box_json.py. Continue through the existing detection JSON and overlay writer, producing one detection.

- [ ] **Step 7: Verify SAM3 tests GREEN**

~~~bash
/home/ubuntu/miniconda3/envs/sam3/bin/python -m pytest \
  test/test_box_prompt_api.py test/test_io_utils.py -v
~~~

- [ ] **Step 8: Review and commit only the three intended paths**

~~~bash
git diff --check
git diff -- run_server.py scripts/infer.py test/test_box_prompt_api.py
git add run_server.py scripts/infer.py test/test_box_prompt_api.py
git commit -m "feat: support SAM3 box prompts over HTTP"
~~~

Do not stage existing untracked outputs/, run.log.pid, or test_inputs/.

### Task 2: HTTP Client and Camera Contract

**Files:**
- Create: ui/service_client.py
- Create: tests/test_service_client.py

- [ ] **Step 1: Write failing pure-function tests**

~~~python
def test_normalize_box_orders_and_clamps() -> None:
    assert normalize_box([40, 30, -5, 10], (32, 24)) == [0, 10, 31, 23]

def test_manual_camera_overrides_uploaded_camera(tmp_path: Path) -> None:
    uploaded = tmp_path / "camera.json"
    uploaded.write_text(json.dumps({
        "cam_K": [1, 0, 2, 0, 1, 2, 0, 0, 1],
        "depth_scale": 1.0,
    }))
    camera = resolve_camera(
        uploaded, fx=600, fy=601, cx=320, cy=240, depth_scale=0.001
    )
    assert camera.camera_json["cam_K"] == [
        600.0, 0.0, 320.0, 0.0, 601.0, 240.0, 0.0, 0.0, 1.0
    ]

def test_validate_pose_response_requires_eight_corners() -> None:
    with pytest.raises(ValueError, match="corners_mm"):
        validate_pose_response({"pose": [0] * 6, "corners_mm": []})
~~~

- [ ] **Step 2: Run and confirm RED because ui.service_client is missing**

- [ ] **Step 3: Implement immutable input/result containers and validation**

~~~python
@dataclass(frozen=True)
class CameraSpec:
    camera_json: Dict[str, Any]
    intrinsics: np.ndarray
    depth_scale: float

@dataclass(frozen=True)
class ServiceResult:
    payload: Dict[str, Any]
    elapsed_ms: float

class ServiceCallError(RuntimeError):
    def __init__(self, status_code: int, payload: Any) -> None:
        super().__init__(f"service HTTP {status_code}: {payload}")
        self.status_code = status_code
        self.payload = payload

def normalize_box(
    box: Sequence[float], image_size: Tuple[int, int]
) -> List[int]:
    width, height = image_size
    x1, y1, x2, y2 = [float(value) for value in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    result = [
        int(round(min(max(x1, 0), width - 1))),
        int(round(min(max(y1, 0), height - 1))),
        int(round(min(max(x2, 0), width - 1))),
        int(round(min(max(y2, 0), height - 1))),
    ]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("box must have positive area")
    return result

def resolve_camera(
    camera_path: Optional[Path],
    *,
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
    depth_scale: Optional[float],
) -> CameraSpec:
    manual = [fx, fy, cx, cy, depth_scale]
    if any(value is not None for value in manual):
        if not all(value is not None and math.isfinite(float(value)) for value in manual):
            raise ValueError("manual camera parameters must provide all five finite values")
        if float(fx) <= 0 or float(fy) <= 0 or float(depth_scale) <= 0:
            raise ValueError("fx, fy, and depth_scale must be positive")
        values = [float(fx), 0.0, float(cx), 0.0, float(fy), float(cy), 0.0, 0.0, 1.0]
        payload = {"cam_K": values, "depth_scale": float(depth_scale)}
    else:
        if camera_path is None:
            raise ValueError("upload camera.json or provide all manual camera parameters")
        payload = json.loads(camera_path.read_text(encoding="utf-8"))
        values = [float(value) for value in payload["cam_K"]]
    return CameraSpec(
        camera_json=payload,
        intrinsics=np.asarray(values, dtype=np.float64).reshape(3, 3),
        depth_scale=float(payload["depth_scale"]),
    )

def decode_coco_rle(rle: Dict[str, Any]) -> np.ndarray:
    encoded = dict(rle)
    if isinstance(encoded.get("counts"), str):
        encoded["counts"] = encoded["counts"].encode("ascii")
    return np.asarray(cocomask.decode(encoded), dtype=bool)

def validate_pose_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    pose = np.asarray(payload.get("pose"), dtype=float)
    corners = np.asarray(payload.get("corners_mm"), dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("pose must contain six finite values")
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError("corners_mm must contain eight finite xyz points")
    if payload.get("frame") != "camera":
        raise ValueError("pose frame must be camera")
    return payload
~~~

Implement write_mask_png as an 8-bit single-channel PNG writer. Reject partial manual camera input rather than silently mixing sources. Extend resolve_camera with the already-supported nested camera.intrinsics form while preserving the same CameraSpec output.

- [ ] **Step 4: Add failing request-shape tests using a recording requests.Session**

Assert the SAM3 JSON contains box and the pose multipart keys are exactly rgb, depth, camera, and mask.

- [ ] **Step 5: Implement call_sam3_box and call_pose_service**

~~~python
def call_sam3_box(
    url: str,
    rgb_path: Path,
    box: Sequence[int],
    *,
    timeout_s: float,
    session: Optional[requests.Session] = None,
) -> ServiceResult:
    client = session or requests.Session()
    started = time.perf_counter()
    try:
        payload = {
            "image_base64": base64.b64encode(rgb_path.read_bytes()).decode("ascii"),
            "box": [int(value) for value in box],
            "save_vis": False,
            "return_vis_base64": False,
        }
        response = client.post(url, json=payload, timeout=float(timeout_s))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        body = response.json()
        if not response.ok or body.get("ok") is False:
            raise ServiceCallError(response.status_code, body)
        return ServiceResult(payload=body, elapsed_ms=elapsed_ms)
    finally:
        if session is None:
            client.close()

def call_pose_service(
    url: str,
    rgb_path: Path,
    depth_path: Path,
    camera_path: Path,
    mask_path: Path,
    *,
    timeout_s: float,
    session: Optional[requests.Session] = None,
) -> ServiceResult:
    client = session or requests.Session()
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            files = {
                "rgb": stack.enter_context(rgb_path.open("rb")),
                "depth": stack.enter_context(depth_path.open("rb")),
                "camera": stack.enter_context(camera_path.open("rb")),
                "mask": stack.enter_context(mask_path.open("rb")),
            }
            response = client.post(url, files=files, timeout=float(timeout_s))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        body = response.json()
        if not response.ok:
            raise ServiceCallError(response.status_code, body)
        return ServiceResult(
            payload=validate_pose_response(body),
            elapsed_ms=elapsed_ms,
        )
    finally:
        if session is None:
            client.close()
~~~

Both functions measure wall time, parse JSON, preserve backend error bodies in ServiceCallError, and close multipart file handles with ExitStack.

- [ ] **Step 6: Run tests GREEN and commit**

~~~bash
/home/ubuntu/miniconda3/envs/genpose2/bin/python -m pytest \
  tests/test_service_client.py -v
git add ui/service_client.py tests/test_service_client.py
git commit -m "feat: add pose service HTTP client"
~~~

### Task 3: 2D and 3D Visualization

**Files:**
- Create: ui/service_visualization.py
- Create: tests/test_service_visualization.py
- Reuse: ui/pointcloud.py

- [ ] **Step 1: Write failing projection tests**

~~~python
def test_project_camera_points() -> None:
    k = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]], dtype=float)
    points = np.array([[0, 0, 1], [0.1, 0.2, 1]], dtype=float)
    assert project_camera_points(points, k).tolist() == [
        [50.0, 40.0], [60.0, 60.0]
    ]

def test_euler_zyx_identity() -> None:
    assert np.allclose(euler_zyx_matrix(0, 0, 0), np.eye(3))
~~~

- [ ] **Step 2: Run and confirm RED**

- [ ] **Step 3: Implement deterministic rendering functions**

~~~python
CUBOID_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

def euler_zyx_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    rx_matrix = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_matrix = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_matrix = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz_matrix @ ry_matrix @ rx_matrix

def project_camera_points(
    points_m: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    points = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
    pixels = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    valid = points[:, 2] > 1e-9
    projected = (np.asarray(intrinsics) @ points[valid].T).T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels

def render_box(image: Image.Image, box: Sequence[int]) -> Image.Image:
    output = image.convert("RGB").copy()
    ImageDraw.Draw(output).rectangle(tuple(box), outline=(255, 190, 0), width=3)
    return output

def render_mask(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8))

def render_mask_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    foreground = np.asarray(mask, dtype=bool)
    rgb[foreground] = (
        0.5 * rgb[foreground] + 0.5 * np.array([0, 255, 80])
    ).astype(np.uint8)
    return Image.fromarray(rgb)

def render_pose_overlay(
    image: Image.Image,
    response: Dict[str, Any],
    intrinsics: np.ndarray,
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    corners_m = np.asarray(response["corners_mm"], dtype=float) / 1000.0
    corner_pixels = project_camera_points(corners_m, intrinsics)
    for first, second in CUBOID_EDGES:
        p0, p1 = corner_pixels[first], corner_pixels[second]
        if np.isfinite(p0).all() and np.isfinite(p1).all():
            draw.line([tuple(p0), tuple(p1)], fill=(255, 220, 0), width=3)
    pose = np.asarray(response["pose"], dtype=float)
    origin = pose[:3] / 1000.0
    rotation = euler_zyx_matrix(*pose[3:])
    endpoints = np.vstack([origin, origin + (rotation * 0.08).T])
    pixels = project_camera_points(endpoints, intrinsics)
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 96, 255))):
        if np.isfinite(pixels[[0, index + 1]]).all():
            draw.line(
                [tuple(pixels[0]), tuple(pixels[index + 1])],
                fill=color,
                width=4,
            )
    return output
~~~

Draw red/green/blue pose axes and all twelve cuboid edges. Skip points with non-positive camera Z.

- [ ] **Step 4: Add a synthetic point-cloud export test**

Use a 4x4 RGB-D input with one invalid depth pixel. Assert non-empty GLB and PLY outputs and no CUDA import.

- [ ] **Step 5: Implement export_pose_scene**

~~~python
@dataclass(frozen=True)
class SceneArtifacts:
    glb_path: Path
    ply_path: Path
    num_points: int

def create_cuboid_mesh(
    corners_glb: np.ndarray,
    edges: Sequence[Tuple[int, int]],
) -> trimesh.Trimesh:
    parts = []
    for first, second in edges:
        p0, p1 = corners_glb[first], corners_glb[second]
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            continue
        cylinder = trimesh.creation.cylinder(
            radius=0.0015, height=length, sections=10
        )
        transform = trimesh.geometry.align_vectors(
            [0.0, 0.0, 1.0], direction / length
        )
        cylinder.apply_transform(transform)
        cylinder.apply_translation((p0 + p1) * 0.5)
        cylinder.visual.face_colors = [255, 220, 0, 255]
        parts.append(cylinder)
    if not parts:
        raise ValueError("cuboid contains no valid edges")
    return trimesh.util.concatenate(parts)

def export_pose_scene(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: np.ndarray,
    pose_response: Dict[str, Any],
    output_dir: Path,
    *,
    max_points: int = 80_000,
) -> SceneArtifacts:
    points_glb, colors, stats = depth_rgb_to_pointcloud(
        depth_mm,
        rgb,
        intrinsics,
        factor_depth=1000.0,
        max_points=max_points,
    )
    pose = np.asarray(pose_response["pose"], dtype=float)
    origin_glb, rotation_glb = camera_pose_m_to_glb(
        pose[:3] / 1000.0,
        euler_zyx_matrix(*pose[3:]),
    )
    axes = create_pose_axes_mesh(origin_glb, rotation_glb)
    corners_glb = camera_points_m_to_glb(
        np.asarray(pose_response["corners_mm"], dtype=float) / 1000.0
    )
    cuboid = create_cuboid_mesh(corners_glb, CUBOID_EDGES)
    files = export_scene_files(
        points_glb,
        colors,
        output_dir,
        "pose_scene",
        extra_geometries=[axes, cuboid],
    )
    return SceneArtifacts(
        glb_path=files["glb_path"],
        ply_path=files["ply_path"],
        num_points=int(stats["total"]),
    )
~~~

Reuse depth_rgb_to_pointcloud, camera_points_m_to_glb, create_pose_axes_mesh, points_to_colored_mesh, export_scene_glb, and export_pointcloud_ply. Build the cuboid from corners_mm as thin cylinders after camera-to-GLB conversion.

- [ ] **Step 6: Run tests GREEN and commit**

~~~bash
/home/ubuntu/miniconda3/envs/genpose2/bin/python -m pytest \
  tests/test_service_visualization.py -v
git add ui/service_visualization.py tests/test_service_visualization.py
git commit -m "feat: visualize service pose and point cloud"
~~~

### Task 4: Staged Pipeline and Gradio UI

**Files:**
- Create: ui/service_frontend.py
- Create: tests/test_service_frontend.py

- [ ] **Step 1: Write failing click-state tests**

~~~python
def test_two_clicks_complete_box() -> None:
    first = advance_box_click(None, (100, 80), (640, 480))
    assert first.status == "awaiting_second"
    second = advance_box_click(first, (400, 300), (640, 480))
    assert second.box == [100, 80, 400, 300]
    assert second.status == "complete"

def test_third_click_starts_new_box() -> None:
    complete = BoxState(first=(1, 2), box=[1, 2, 10, 20], status="complete")
    state = advance_box_click(complete, (5, 6), (32, 24))
    assert state.first == (5, 6)
    assert state.box is None
~~~

- [ ] **Step 2: Run and confirm RED**

- [ ] **Step 3: Implement click and pipeline state**

~~~python
@dataclass(frozen=True)
class BoxState:
    first: Optional[Tuple[int, int]] = None
    box: Optional[List[int]] = None
    status: str = "empty"

@dataclass
class PipelineState:
    run_dir: Optional[str] = None
    mask_path: Optional[str] = None
    sam3_payload: Optional[Dict[str, Any]] = None
    pose_payload: Optional[Dict[str, Any]] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)

def advance_box_click(
    current: Optional[BoxState],
    click: Tuple[int, int],
    image_size: Tuple[int, int],
) -> BoxState:
    if current is None or current.status in ("empty", "complete"):
        return BoxState(first=click, box=None, status="awaiting_second")
    if current.first is None:
        return BoxState(first=click, box=None, status="awaiting_second")
    return BoxState(
        first=current.first,
        box=normalize_box([*current.first, *click], image_size),
        status="complete",
    )

def clear_box() -> BoxState:
    return BoxState()
~~~

Implement run_sam3_stage by creating one run directory, saving RGB and generated camera JSON, calling call_sam3_box, decoding exactly one detection mask, writing mask.png, and storing input_prepare_ms, sam3_http_ms, and mask_decode_ms. Implement run_pose_stage by calling call_pose_service with the stored mask, rendering 2D/3D artifacts, and adding pose_http_ms, visualization_2d_ms, pointcloud_ms, and total_ms. Implement run_full_pipeline as run_sam3_stage followed by run_pose_stage using the returned PipelineState. Every stage catches ServiceCallError, records the failed-stage elapsed time, and returns the previous successful state plus a visible error string.

- [ ] **Step 4: Add and pass failure-preservation tests**

Inject callables. Make SAM3 succeed and pose raise ServiceCallError. Assert the mask, SAM3 response, and SAM3 timing remain.

- [ ] **Step 5: Build the Gradio layout**

Create editable URLs and health checks; RGB select events; depth/camera/manual K inputs; clear/SAM3/pose/full buttons; box/mask/depth/pose images; Model3D; response JSON; timing dataframe; and file downloads. Do not import a model runner, torch, or CUDA.

- [ ] **Step 6: Run tests GREEN and commit**

~~~bash
/home/ubuntu/miniconda3/envs/genpose2/bin/python -m pytest \
  tests/test_service_frontend.py -v
git add ui/service_frontend.py tests/test_service_frontend.py
git commit -m "feat: add staged service inference UI"
~~~

### Task 5: Model-free Entrypoint and Documentation

**Files:**
- Create: run_service_frontend.py
- Modify: README.md
- Modify: tests/test_service_frontend.py

- [ ] **Step 1: Add a failing import smoke test**

~~~python
def test_entrypoint_import_does_not_import_torch() -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; import run_service_frontend; assert 'torch' not in sys.modules",
    ]
    subprocess.run(command, cwd=ROOT_DIR, check=True)
~~~

- [ ] **Step 2: Run and confirm RED because the entrypoint is missing**

- [ ] **Step 3: Implement run_service_frontend.py**

~~~python
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Service-based pose test frontend"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18086)
    args = parser.parse_args()
    from ui.service_frontend import build_service_frontend
    app = build_service_frontend()
    app.queue().launch(
        server_name=args.host,
        server_port=args.port,
        ssr_mode=False,
        show_error=True,
    )
~~~

- [ ] **Step 4: Document startup and independent services**

Document:

~~~bash
/home/ubuntu/miniconda3/envs/genpose2/bin/python run_service_frontend.py \
  --host 0.0.0.0 --port 18086
~~~

State that the frontend never starts or loads SAM3/GenPose2.

- [ ] **Step 5: Run all CPU tests and commit**

~~~bash
/home/ubuntu/miniconda3/envs/genpose2/bin/python -m pytest \
  tests/test_service_client.py \
  tests/test_service_visualization.py \
  tests/test_service_frontend.py -v
git add run_service_frontend.py README.md tests/test_service_frontend.py
git commit -m "feat: add model-free service frontend entrypoint"
~~~

### Task 6: Real Service Integration

- [ ] **Step 1: Restart the existing SAM3 service on 18003, never a duplicate**

Stop only its verified exact process/session, relaunch the same service, and verify one listener and one SAM3 process.

- [ ] **Step 2: Start the latest GenPose2 service on 8084 only if absent**

Use explicit latest-project checkpoint/output environment paths. Verify /manipulation/health returns READY.

- [ ] **Step 3: Run a real box request through SAM3**

Verify HTTP 200, one detection, non-empty RLE mask, and the input box echoed.

- [ ] **Step 4: Run that mask through the pose service**

Verify exact response keys: pose, corners_mm, frame, pose_unit, rotation_order.

- [ ] **Step 5: Run the complete frontend pipeline function**

Verify non-empty mask, overlays, GLB, PLY, responses, and timing JSON. Confirm the frontend did not create a GPU compute process.

### Task 7: Deploy and Browser Verification

- [ ] **Step 1: Launch in independent tmux session pose_service_frontend_18086**

Use a nohup wrapper and bind 0.0.0.0:18086.

- [ ] **Step 2: Verify runtime evidence**

Check the session, exact process, listener, HTTP 200, logs, and GPU process list.

- [ ] **Step 3: Inspect the UI in a browser**

Upload real RGB-D, perform the two clicks, run the full pipeline, and inspect every output and instruction transition.

- [ ] **Step 4: Verify service-offline behavior**

Use intentionally unused SAM3 and pose URLs. Confirm the page remains up and preserves completed earlier stages.

- [ ] **Step 5: Run fresh final verification**

~~~bash
/home/ubuntu/miniconda3/envs/sam3/bin/python -m pytest \
  test/test_box_prompt_api.py test/test_io_utils.py -v
/home/ubuntu/miniconda3/envs/genpose2/bin/python -m pytest \
  tests/test_service_client.py \
  tests/test_service_visualization.py \
  tests/test_service_frontend.py -v
git status --short
git log --oneline -8
~~~

Review committed diffs and confirm the pre-existing uncommitted http_server.py and SAM3 untracked runtime artifacts were preserved and never staged.
