# Service-based Pose Inference Test Frontend Design

Date: 2026-08-05
Status: approved for implementation

## 1. Purpose

Build an independent Gradio test frontend for the existing SAM3 and GenPose2 HTTP services. The frontend must not import or load either model. It accepts RGB-D data and camera intrinsics, lets the operator define one target with two clicks, obtains a mask from SAM3 using a true box prompt, calls the existing pose endpoint, and visualizes all intermediate and final results.

## 2. Goals

- Run as an independent process on `0.0.0.0:18086`.
- Call the existing SAM3 service, default `http://127.0.0.1:18003/infer`.
- Call the existing GenPose2 service, default `http://127.0.0.1:8084/manipulation/pick_pose`.
- Keep both service URLs editable. A user can replace the pose URL with `place_pose` without changing UI modes.
- Accept RGB, aligned depth, and either `camera.json` or manually entered `fx`, `fy`, `cx`, `cy`, and `depth_scale`.
- Define a target box by clicking the top-left and bottom-right corners on the RGB image.
- Extend the existing SAM3 `/infer` contract with an optional true geometric box prompt.
- Display masks, overlays, pose projection, an interactive point cloud, pose axes, the oriented cuboid, raw responses, downloads, and per-stage timing.
- Remain usable when either backend is offline and preserve successful earlier-stage results.

## 3. Non-goals

- Do not load SAM3 or GenPose2 in the frontend process.
- Do not automatically start, restart, or supervise backend services.
- Do not add separate pick/place workflows.
- Do not change the documented service ports.
- Do not implement multi-box or multi-instance pose inference in this version.
- Do not add text/VLM prompting to the primary UI flow.

## 4. Process and Port Boundaries

| Process | Default address | GPU model ownership |
|---|---|---|
| SAM3 service | `127.0.0.1:18003` | One SAM3 model instance |
| GenPose2 service | `127.0.0.1:8084` | One GenPose2 model instance |
| Test frontend | `0.0.0.0:18086` | None |

The frontend performs CPU-side decoding, HTTP multipart construction, 2D rendering, point-cloud generation, and GLB/PLY export. It must not initialize CUDA.

## 5. Architecture

```text
Browser :18086
  -> upload RGB/depth and provide camera parameters
  -> click top-left and bottom-right target corners
  -> SAM3 HTTP :18003/infer (RGB + box)
  <- one RLE mask, score, bbox, service metadata
  -> GenPose2 HTTP :8084/manipulation/pick_pose
     (RGB + depth + generated camera.json + generated mask)
  <- pose[6], corners_mm[8][3], frame and units
  -> CPU visualization and artifact export
```

## 6. SAM3 API Extension

The existing request fields remain backward compatible. Add one optional field:

```json
{
  "image_base64": "...",
  "box": [120, 80, 420, 360]
}
```

Rules:

- `box` uses pixel `xyxy` coordinates: `[x1, y1, x2, y2]`.
- Coordinates are ordered, clamped to the image, and must have positive area.
- When `box` is present, SAM3 uses its existing positive geometric box-prompt path.
- Box prompting returns one selected complete target instance.
- Existing text and point-prompt requests continue to behave unchanged.
- The service continues to use its existing model cache and global inference lock.
- Response detections retain the existing RLE segmentation, score, and `xywh` output bbox representation.

## 7. Frontend Interaction

### 7.1 Input and service controls

- Editable SAM3 and pose endpoint URLs.
- A service-health check button with independent status for each backend.
- RGB upload.
- Depth upload supporting the formats already handled by the project (`.png`, `.exr`, and compatible existing formats).
- Optional `camera.json` upload.
- Manual `fx`, `fy`, `cx`, `cy`, and `depth_scale` fields.
- Manual values override uploaded camera values when all five manual values are valid.

### 7.2 Two-click target box

The RGB image uses native Gradio selection events:

1. First click stores the top-left corner.
2. Second click stores the opposite corner, normalizes coordinate order, and renders the box.
3. Further clicks start a new selection.
4. A clear button resets the box.

The visible prompt changes through:

- `请先点击目标框左上角`
- `已选择左上角 (x1,y1)，请点击右下角`
- `目标框已完成：[x1,y1,x2,y2]`

### 7.3 Execution controls

- `SAM3 生成掩码`: runs only segmentation and stores the generated mask state.
- `调用位姿服务`: uses the latest valid mask and calls the configured pose endpoint.
- `一键完整推理`: runs both stages sequentially.

## 8. Outputs

- RGB with the input box.
- Binary SAM3 mask.
- RGB plus transparent mask overlay.
- Depth colormap.
- RGB with projected pose axes and `corners_mm` cuboid edges.
- Interactive RGB point cloud.
- Pose axes and oriented cuboid in the point-cloud scene.
- Raw SAM3 JSON response.
- Raw pose JSON response.
- Stage-status and timing table.
- Downloadable mask, response JSON, GLB, and PLY artifacts.

Artifacts are stored under:

```text
output/service_frontend_runs/<timestamp_request-id>/
```

## 9. Camera and Geometry Conventions

- Intrinsics form `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`.
- `depth_scale` converts stored depth values to meters for point-cloud generation.
- Pose translation and `corners_mm` are in camera-frame millimetres.
- Euler angles are `[rx, ry, rz]` in radians with ZYX composition.
- 2D projection uses the provided camera intrinsics and ignores points with non-positive camera Z.
- The point-cloud preview reuses the existing camera-to-GLB convention in `ui/pointcloud.py`.

## 10. Timing

Every execution records wall-clock time for:

- input preparation
- SAM3 HTTP request
- mask decoding and normalization
- pose HTTP request
- 2D visualization
- point-cloud and GLB/PLY generation
- end-to-end total

When a service supplies internal timing, the UI displays it separately from frontend-observed HTTP latency.

## 11. Error Handling

- Invalid or missing RGB/depth/camera/box inputs stop before network calls.
- A zero-area or out-of-bounds box produces an actionable validation message.
- Backend connection failures are reported as service offline without terminating the frontend.
- Non-2xx JSON error bodies are displayed without hiding the backend `error_code`.
- If SAM3 succeeds and pose inference fails, mask outputs and SAM3 timing remain visible.
- If visualization fails after pose inference, raw backend responses remain visible.
- HTTP requests use explicit configurable timeouts.

## 12. File Structure

GenPose2 repository additions:

```text
run_service_frontend.py
ui/service_frontend.py
ui/service_client.py
ui/service_visualization.py
tests/test_service_client.py
tests/test_service_visualization.py
```

SAM3 repository changes are local to its request parsing and box-prompt execution path. Existing public text/point behavior remains covered by regression tests.

## 13. Test Strategy

Test-driven unit coverage includes:

- click-state transitions and box coordinate normalization
- coordinate clamping and zero-area rejection
- uploaded camera parsing and manual-intrinsics precedence
- SAM3 box request construction and RLE decoding
- pose multipart construction and response validation
- pose/cuboid 2D projection
- depth-to-point-cloud generation and downsampling
- timing population on success and failure
- preservation of earlier-stage state after later-stage failure
- SAM3 backward compatibility for text and point prompts

Real integration verification on the 4090-1 server includes:

1. Existing SAM3 service with a true box request.
2. Existing GenPose2 service with the generated single-instance mask.
3. Full frontend workflow with real RGB-D and intrinsics.
4. Visual inspection of mask, pose overlay, point cloud, axes, and cuboid.
5. Independent service-offline checks for SAM3 and GenPose2.

## 14. Deployment

The frontend runs in its own `tmux` session named `pose_service_frontend_18086`. It binds to `0.0.0.0:18086`. Backend processes are not managed by the frontend. The frontend startup command must not preload any model or import a model runner.
