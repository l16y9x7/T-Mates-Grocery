const form = document.querySelector("#pickForm");
const submitButton = document.querySelector("#submitButton");
const errorMessage = document.querySelector("#errorMessage");
const timeline = document.querySelector("#timeline");
const resultCard = document.querySelector("#resultCard");
const resultStatus = document.querySelector("#resultStatus");
const resultBody = document.querySelector("#resultBody");
const operationKey = document.querySelector("#operationKey");
const placeForm = document.querySelector("#placeForm");
const placeSubmitButton = document.querySelector("#placeSubmitButton");
const placeErrorMessage = document.querySelector("#placeErrorMessage");
const placeTimeline = document.querySelector("#placeTimeline");
const placeResultCard = document.querySelector("#placeResultCard");
const placeResultStatus = document.querySelector("#placeResultStatus");
const placeResultBody = document.querySelector("#placeResultBody");
const placeOperationKey = document.querySelector("#placeOperationKey");
const operationModeButtons = document.querySelectorAll("[data-operation-mode]");
const operationViews = {
  pick: document.querySelector("#pickWorkspace"),
  place: document.querySelector("#placeWorkspace"),
};
const connectionText = document.querySelector("#connectionText");
const pulse = document.querySelector("#pulse");
const poseForm = document.querySelector("#poseForm");
const poseTypePreset = document.querySelector("#poseTypePreset");
const poseTypeCustom = document.querySelector("#poseTypeCustom");
const shelfLevelPreset = document.querySelector("#shelfLevelPreset");
const shelfLevelCustom = document.querySelector("#shelfLevelCustom");
const shelfLevelLabel = document.querySelector("#shelfLevelLabel");
const navigationForm = document.querySelector("#navigationForm");
const navigationTargetPreset = document.querySelector("#navigationTargetPreset");
const navigationTargetsLoading = document.querySelector("#navigationTargetsLoading");
const taskNavigationTargets = document.querySelector("#taskNavigationTargets");
const inspectionNavigationTargets = document.querySelector("#inspectionNavigationTargets");
const targetId = document.querySelector("#targetId");
const gripperForm = document.querySelector("#gripperForm");
const gripperSubmitButton = document.querySelector("#gripperSubmitButton");
const healthButton = document.querySelector("#healthButton");
const robotStatus = document.querySelector("#robotStatus");
const robotOutput = document.querySelector("#robotOutput");
const taskForm = document.querySelector("#taskForm");
const taskSubmitButton = document.querySelector("#taskSubmitButton");
const taskTerminateButton = document.querySelector("#taskTerminateButton");
const taskErrorMessage = document.querySelector("#taskErrorMessage");
const taskTimeline = document.querySelector("#taskTimeline");
const taskLiveStatus = document.querySelector("#taskLiveStatus");
const taskTitle = document.querySelector("#taskTitle");
const taskDescription = document.querySelector("#taskDescription");
const taskProgress = document.querySelector("#taskProgress");
const taskProgressText = document.querySelector("#taskProgressText");
const taskProgressTrack = document.querySelector("#taskProgressTrack");
const taskProgressBar = document.querySelector("#taskProgressBar");
const taskCaptures = document.querySelector("#taskCaptures");
const taskCaptureCount = document.querySelector("#taskCaptureCount");
const taskCaptureList = document.querySelector("#taskCaptureList");
const taskDetails = document.querySelector("#taskDetails");
const taskDetailsTitle = document.querySelector("#taskDetailsTitle");
const taskDetailsCount = document.querySelector("#taskDetailsCount");
const taskDetailsList = document.querySelector("#taskDetailsList");
const taskErrorDetails = document.querySelector("#taskErrorDetails");
const taskErrorBody = document.querySelector("#taskErrorBody");
const taskResultCard = document.querySelector("#taskResultCard");
const taskResultStatus = document.querySelector("#taskResultStatus");
const taskResultBody = document.querySelector("#taskResultBody");
const taskOperationKey = document.querySelector("#taskOperationKey");
const task1MockOrder = document.querySelector("#task1MockOrder");
const mockOrderStatus = document.querySelector("#mockOrderStatus");
const mockOrderProducts = document.querySelector("#mockOrderProducts");
const mockOrderEmpty = document.querySelector("#mockOrderEmpty");
const mockOrderId = document.querySelector("#mockOrderId");
const mockOrderCatalogSize = document.querySelector("#mockOrderCatalogSize");
const mockOrderRefreshButton = document.querySelector("#mockOrderRefreshButton");
const taskInterfaceMetrics = document.querySelector("#taskInterfaceMetrics");
const taskInterfaceMetricsCount = document.querySelector("#taskInterfaceMetricsCount");
const taskInterfaceMetricsBody = document.querySelector("#taskInterfaceMetricsBody");
const taskInterfaceMetricsEmpty = document.querySelector("#taskInterfaceMetricsEmpty");
const pickVisual = document.querySelector("#pickVisual");
const pickVisualCanvas = document.querySelector("#pickVisualCanvas");
const pickVisualStatus = document.querySelector("#pickVisualStatus");
const pickPoseStatus = document.querySelector("#pickPoseStatus");
const pickPoseValues = document.querySelector("#pickPoseValues");
const pickPoseMeta = document.querySelector("#pickPoseMeta");
const placeVisual = document.querySelector("#placeVisual");
const placeVisualCanvas = document.querySelector("#placeVisualCanvas");
const placeVisualStatus = document.querySelector("#placeVisualStatus");
const placePoseStatus = document.querySelector("#placePoseStatus");
const placePoseValues = document.querySelector("#placePoseValues");
const placePoseMeta = document.querySelector("#placePoseMeta");
const taskVisual = document.querySelector("#taskVisual");
const taskVisualCanvas = document.querySelector("#taskVisualCanvas");
const taskVisualStatus = document.querySelector("#taskVisualStatus");
const taskPoseStatus = document.querySelector("#taskPoseStatus");
const taskPoseValues = document.querySelector("#taskPoseValues");
const taskPoseMeta = document.querySelector("#taskPoseMeta");
const taskElapsedTime = document.querySelector("#taskElapsedTime");
const pickElapsedTime = document.querySelector("#pickElapsedTime");
const placeElapsedTime = document.querySelector("#placeElapsedTime");
const robotIpForm = document.querySelector("#robotIpForm");
const robotIpInput = document.querySelector("#robotIpInput");
const robotIpSubmitButton = document.querySelector("#robotIpSubmitButton");
const currentRobotIp = document.querySelector("#currentRobotIp");
const robotIpRuntimeStatus = document.querySelector("#robotIpRuntimeStatus");
const robotIpMessage = document.querySelector("#robotIpMessage");
const forceRestartDialog = document.querySelector("#forceRestartDialog");
const activeOperationsSummary = document.querySelector("#activeOperationsSummary");
let eventSource = null;
let placeEventSource = null;
let taskEventSource = null;
let currentTaskRunId = null;
let currentMockOrder = null;
let taskBusy = false;
let mockOrderLoading = false;
const taskInterfaceCallValues = new Map();
const visualPollers = { pick: null, place: null, task: null };
const visualRefreshers = { pick: null, place: null, task: null };

function createElapsedTimer(element) {
  let startedAt = null;
  let intervalId = null;

  function render() {
    const elapsedMilliseconds = startedAt === null ? 0 : performance.now() - startedAt;
    const totalSeconds = Math.max(0, Math.floor(elapsedMilliseconds / 1000));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    element.textContent = [hours, minutes, seconds]
      .map((value) => String(value).padStart(2, "0"))
      .join(":");
  }

  return {
    start() {
      if (intervalId !== null) window.clearInterval(intervalId);
      startedAt = performance.now();
      render();
      intervalId = window.setInterval(render, 250);
    },
    stop() {
      render();
      if (intervalId !== null) window.clearInterval(intervalId);
      intervalId = null;
    },
    reset() {
      if (intervalId !== null) window.clearInterval(intervalId);
      intervalId = null;
      startedAt = null;
      render();
    },
  };
}

const elapsedTimers = {
  task: createElapsedTimer(taskElapsedTime),
  pick: createElapsedTimer(pickElapsedTime),
  place: createElapsedTimer(placeElapsedTime),
};

function setOperationMode(mode) {
  if (!operationViews[mode]) return;
  Object.entries(operationViews).forEach(([viewMode, view]) => {
    view.hidden = viewMode !== mode;
  });
  operationModeButtons.forEach((button) => {
    const active = button.dataset.operationMode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

operationModeButtons.forEach((button) => {
  button.addEventListener("click", () => setOperationMode(button.dataset.operationMode));
});

const TASK_INFO = {
  0: { title: "采集准备", description: "采集货架巡检点的头部 RGB-D 基准数据" },
  1: { title: "订单分拣", description: "从模拟点单系统获取两件随机商品，并完成货架抓取和交付台放置" },
  2: { title: "缺货补货", description: "巡检缺货商品并从补货台抓取后放回货架" },
  3: { title: "乱放纠正", description: "识别错放商品并交换两件商品的货架位置" },
};

const CUSTOM_VALUE = "__custom__";
const POSES_REQUIRING_SHELF_LEVEL = new Set([
  "SHELF_INSPECT",
  "SHELF_PICK_READY",
  "SHELF_PLACE_READY",
]);
const POSES_WITHOUT_SHELF_LEVEL = new Set([
  "RECEIPT_VIEW",
  "SHELF_VIEW_UPPER",
  "SHELF_VIEW_LOWER",
  "REPLENISHMENT_TABLE_PICK_READY",
  "DELIVERY_TABLE_PLACE_READY",
  "START_POSITION",
]);

function selectedPoseType() {
  return poseTypePreset.value === CUSTOM_VALUE
    ? poseTypeCustom.value.trim()
    : poseTypePreset.value;
}

function selectedShelfLevel() {
  return shelfLevelPreset.value === CUSTOM_VALUE
    ? shelfLevelCustom.value.trim()
    : shelfLevelPreset.value;
}

function updateShelfLevelCustomField() {
  const usesCustomLevel = !shelfLevelPreset.disabled && shelfLevelPreset.value === CUSTOM_VALUE;
  shelfLevelCustom.hidden = !usesCustomLevel;
  shelfLevelCustom.required = usesCustomLevel;
}

function updateShelfLevelField() {
  const poseType = selectedPoseType();
  const requiresLevel = POSES_REQUIRING_SHELF_LEVEL.has(poseType);
  const omitsLevel = POSES_WITHOUT_SHELF_LEVEL.has(poseType);
  shelfLevelPreset.disabled = omitsLevel;
  shelfLevelPreset.required = requiresLevel;
  shelfLevelLabel.textContent = requiresLevel
    ? "货架层级（必填）"
    : omitsLevel
      ? "货架层级（当前位姿不传）"
      : "货架层级（自定义位姿可选）";
  updateShelfLevelCustomField();
}

function updatePoseTypeCustomField() {
  const usesCustomPose = poseTypePreset.value === CUSTOM_VALUE;
  poseTypeCustom.hidden = !usesCustomPose;
  poseTypeCustom.required = usesCustomPose;
  updateShelfLevelField();
}

function selectedNavigationTarget() {
  return navigationTargetPreset.value === CUSTOM_VALUE
    ? targetId.value.trim()
    : navigationTargetPreset.value;
}

function updateNavigationTargetCustomField() {
  const usesCustomTarget = navigationTargetPreset.value === CUSTOM_VALUE;
  targetId.hidden = !usesCustomTarget;
  targetId.required = usesCustomTarget;
}

function applyNavigationTargets(navigationTargets) {
  const configuredGroups = [
    [taskNavigationTargets, navigationTargets?.task_points],
    [inspectionNavigationTargets, navigationTargets?.inspection_points],
  ];
  const entries = configuredGroups.flatMap(([, values]) =>
    Array.isArray(values)
      ? values.filter((entry) => typeof entry?.target_id === "string" && entry.target_id.trim())
      : [],
  );
  if (!entries.length) return;

  const previouslyLoaded = navigationTargetPreset.dataset.runtimeTargetsLoaded === "true";
  const previousValue = previouslyLoaded ? navigationTargetPreset.value : null;
  const seenTargets = new Set();
  configuredGroups.forEach(([group, values]) => {
    group.replaceChildren();
    if (!Array.isArray(values)) return;
    values.forEach((entry) => {
      const targetIdValue = typeof entry?.target_id === "string" ? entry.target_id.trim() : "";
      if (!targetIdValue || seenTargets.has(targetIdValue)) return;
      seenTargets.add(targetIdValue);
      const option = document.createElement("option");
      option.value = targetIdValue;
      const label = typeof entry.label === "string" ? entry.label.trim() : "";
      option.textContent = label ? `${targetIdValue} · ${label}` : targetIdValue;
      group.append(option);
    });
  });
  navigationTargetsLoading?.remove();
  navigationTargetPreset.dataset.runtimeTargetsLoaded = "true";
  navigationTargetPreset.value =
    previousValue && (seenTargets.has(previousValue) || previousValue === CUSTOM_VALUE)
      ? previousValue
      : entries[0].target_id.trim();
  updateNavigationTargetCustomField();
}

poseTypePreset.addEventListener("change", updatePoseTypeCustomField);
poseTypeCustom.addEventListener("input", updateShelfLevelField);
shelfLevelPreset.addEventListener("change", updateShelfLevelCustomField);
updatePoseTypeCustomField();
updateShelfLevelField();
navigationTargetPreset.addEventListener("change", updateNavigationTargetCustomField);
updateNavigationTargetCustomField();

function setError(message = "") {
  errorMessage.textContent = message;
  errorMessage.hidden = !message;
}

function setBusy(busy) {
  submitButton.disabled = busy;
  submitButton.querySelector("span:last-child").textContent = busy ? "执行中" : "开始拣取";
  connectionText.textContent = busy ? "实时连接中" : "接口待命";
  pulse.classList.toggle("ready", !busy);
}

function resetView() {
  elapsedTimers.pick.reset();
  stopVisualPolling("pick");
  timeline.hidden = false;
  timeline.replaceChildren();
  resultCard.hidden = true;
  pickVisual.hidden = true;
  resetVisualPanel(pickVisualCanvas, pickVisualStatus, pickPoseStatus, pickPoseValues, pickPoseMeta);
  operationKey.textContent = "-";
  setError();
}

function setPlaceError(message = "") {
  placeErrorMessage.textContent = message;
  placeErrorMessage.hidden = !message;
}

function setPlaceBusy(busy) {
  placeSubmitButton.disabled = busy;
  placeSubmitButton.querySelector("span:last-child").textContent = busy ? "执行中" : "开始放置";
  connectionText.textContent = busy ? "放置流程连接中" : "接口待命";
  pulse.classList.toggle("ready", !busy);
}

function resetPlaceView() {
  elapsedTimers.place.reset();
  stopVisualPolling("place");
  placeTimeline.hidden = false;
  placeTimeline.replaceChildren();
  placeResultCard.hidden = true;
  placeVisual.hidden = true;
  resetVisualPanel(placeVisualCanvas, placeVisualStatus, placePoseStatus, placePoseValues, placePoseMeta);
  placeOperationKey.textContent = "-";
  setPlaceError();
}

function resetVisualPanel(canvas, status, poseStatus, poseValues, poseMeta) {
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  status.textContent = "等待图像";
  poseStatus.textContent = "等待位姿估计";
  poseValues.replaceChildren(Object.assign(document.createElement("span"), { className: "pose-empty", textContent: "位姿估计完成后显示" }));
  poseMeta.textContent = "";
}

function formatPoseValue(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "-";
}

function drawMaskOverlay(context, mask, width, height) {
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskContext = maskCanvas.getContext("2d", { willReadFrequently: true });
  maskContext.drawImage(mask, 0, 0, width, height);
  const pixels = maskContext.getImageData(0, 0, width, height);
  const overlay = maskContext.createImageData(width, height);
  for (let index = 0; index < pixels.data.length; index += 4) {
    const value = pixels.data[index];
    if (value > 20) {
      overlay.data[index] = 217;
      overlay.data[index + 1] = 242;
      overlay.data[index + 2] = 107;
      overlay.data[index + 3] = Math.min(115, value);
    }
  }
  maskContext.putImageData(overlay, 0, 0);
  context.drawImage(maskCanvas, 0, 0, width, height);
}

function drawPoseAxes(context, axes, canvas, image) {
  if (!axes || !Array.isArray(axes.origin) || axes.origin.length !== 2) return;
  const sourceWidth = image.naturalWidth || canvas.width;
  const sourceHeight = image.naturalHeight || canvas.height;
  const scaleX = canvas.width / sourceWidth;
  const scaleY = canvas.height / sourceHeight;
  const point = (value) => [Number(value[0]) * scaleX, Number(value[1]) * scaleY];
  const origin = point(axes.origin);
  if (!origin.every(Number.isFinite)) return;

  const colors = { x: "#ef4444", y: "#22c55e", z: "#3b82f6" };
  context.save();
  context.lineCap = "round";
  context.lineWidth = 4;
  Object.entries(colors).forEach(([axis, color]) => {
    if (!Array.isArray(axes[axis]) || axes[axis].length !== 2) return;
    const endpoint = point(axes[axis]);
    if (!endpoint.every(Number.isFinite)) return;
    context.beginPath();
    context.moveTo(...origin);
    context.lineTo(...endpoint);
    context.strokeStyle = color;
    context.stroke();
    context.beginPath();
    context.arc(endpoint[0], endpoint[1], 4, 0, Math.PI * 2);
    context.fillStyle = color;
    context.fill();
    context.font = "700 14px ui-monospace, monospace";
    context.fillText(axis.toUpperCase(), endpoint[0] + 7, endpoint[1] - 7);
  });
  context.beginPath();
  context.arc(origin[0], origin[1], 4, 0, Math.PI * 2);
  context.fillStyle = "#ffffff";
  context.fill();
  context.restore();
}

function renderPose(visual, poseStatus, poseValues, poseMeta) {
  const pose = Array.isArray(visual.pose) && visual.pose.length === 6 ? visual.pose : null;
  if (!pose) {
    poseStatus.textContent = "等待位姿估计";
    return;
  }
  poseStatus.textContent = "已完成";
  poseValues.replaceChildren();
  ["X", "Y", "Z", "RX", "RY", "RZ"].forEach((label, index) => {
    const cell = document.createElement("div");
    cell.className = "pose-cell";
    const name = document.createElement("span");
    name.textContent = label;
    const value = document.createElement("strong");
    value.textContent = formatPoseValue(pose[index]);
    cell.append(name, value);
    poseValues.append(cell);
  });
  const meta = [visual.frame, visual.pose_unit, visual.rotation_order].filter(Boolean);
  poseMeta.textContent = meta.join(" · ");
}

function drawVisual(visual, canvas, status, poseStatus, poseValues, poseMeta, imageCache) {
  if (!visual || !visual.available) return;
  const context = canvas.getContext("2d");
  const imageData = visual.image_data;
  const revision = String(visual.visual_revision || "legacy");
  const revisionChanged = imageCache.revision !== revision;
  imageCache.revision = revision;
  if (imageData && (imageData !== imageCache.imageData || revisionChanged)) {
    imageCache.imageData = imageData;
    const image = new Image();
    imageCache.image = image;
    image.onload = () => {
      if (imageCache.revision === revision && imageCache.image === image) {
        drawVisual(visual, canvas, status, poseStatus, poseValues, poseMeta, imageCache);
      }
    };
    image.src = imageData;
  }
  if (imageCache.image?.complete) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(imageCache.image, 0, 0, canvas.width, canvas.height);
    const maskDataList = Array.isArray(visual.mask_data_list) && visual.mask_data_list.length
      ? visual.mask_data_list
      : (visual.mask_data ? [visual.mask_data] : []);
    const maskKey = JSON.stringify(maskDataList);
    if (maskKey !== imageCache.maskKey || revisionChanged) {
      imageCache.maskKey = maskKey;
      imageCache.masks = maskDataList.map((maskData) => {
        const mask = new Image();
        mask.onload = () => {
          if (imageCache.revision === revision && imageCache.masks?.includes(mask)) {
            drawVisual(visual, canvas, status, poseStatus, poseValues, poseMeta, imageCache);
          }
        };
        mask.src = maskData;
        return mask;
      });
    }
    (imageCache.masks || []).forEach((mask) => {
      if (mask.complete && mask.naturalWidth > 0) {
        context.save();
        drawMaskOverlay(context, mask, canvas.width, canvas.height);
        context.restore();
      }
    });
    const bboxes = Array.isArray(visual.bboxes) && visual.bboxes.length
      ? visual.bboxes
      : (Array.isArray(visual.bbox) ? [visual.bbox] : []);
    bboxes.forEach((bbox, index) => {
      if (!Array.isArray(bbox) || bbox.length !== 4) return;
      const coordinateWidth = Number(visual.bbox_coordinate_width)
        || imageCache.image.naturalWidth
        || canvas.width;
      const coordinateHeight = Number(visual.bbox_coordinate_height)
        || imageCache.image.naturalHeight
        || canvas.height;
      const scaleX = canvas.width / coordinateWidth;
      const scaleY = canvas.height / coordinateHeight;
      const [x1, y1, x2, y2] = bbox.map(Number);
      if (![x1, y1, x2, y2].every(Number.isFinite)) return;
      context.save();
      context.strokeStyle = "#d9f26b";
      context.lineWidth = 3;
      context.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
      context.fillStyle = "#d9f26b";
      context.font = "600 13px ui-monospace, monospace";
      const label = bboxes.length > 1 ? `BBOX ${index + 1}` : "BBOX";
      context.fillText(label, x1 * scaleX + 6, Math.max(18, y1 * scaleY - 7));
      context.restore();
    });
    drawPoseAxes(context, visual.pose_axes, canvas, imageCache.image);
    const layers = ["RGB"];
    if (maskDataList.length) layers.push(maskDataList.length > 1 ? `${maskDataList.length} MASKS` : "MASK");
    if (bboxes.length) layers.push(bboxes.length > 1 ? `${bboxes.length} BBOXES` : "BBOX");
    if (visual.pose_axes) layers.push("6D AXES");
    status.textContent = layers.join(" + ");
  }
  renderPose(visual, poseStatus, poseValues, poseMeta);
}

function beginVisualPolling(taskId, endpoint, panel, canvas, status, poseStatus, poseValues, poseMeta, kind = "pick") {
  const imageCache = {};
  const poll = async () => {
    try {
      const response = await fetch(`${endpoint}/${taskId}/visual`, { cache: "no-store" });
      if (!response.ok) return;
      const visual = await response.json();
      if (visual.available) panel.hidden = false;
      drawVisual(visual, canvas, status, poseStatus, poseValues, poseMeta, imageCache);
    } catch (_) { /* SSE remains the source of flow status if polling is unavailable. */ }
  };
  poll();
  visualPollers[kind] = window.setInterval(poll, 700);
  visualRefreshers[kind] = poll;
}

async function stopVisualPolling(kind = "pick", refresh = false) {
  const timer = visualPollers[kind];
  if (timer) window.clearInterval(timer);
  const finalRefresh = visualRefreshers[kind];
  if (refresh && finalRefresh) await finalRefresh();
  visualPollers[kind] = null;
  visualRefreshers[kind] = null;
}

function eventLabel(event) {
  const labels = { started: "开始", succeeded: "完成", failed: "失败", cancelled: "已取消" };
  return labels[event.status] || event.status || "更新";
}

function formatJson(value) {
  if (value === undefined || value === null) return "-";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function appendInterfacePayload(item, label, payload) {
  const section = document.createElement("section");
  section.className = "interface-payload";
  const heading = document.createElement("div");
  heading.className = "interface-payload-heading";
  heading.textContent = label;
  const content = document.createElement("pre");
  content.textContent = formatJson(payload);
  section.append(heading, content);
  item.append(section);
}

function addFlowEvent(event, target = timeline) {
  const isInterfaceCall = event.event === "接口调用" && ("request" in event || "response" in event);
  const item = document.createElement("article");
  item.className = `timeline-item ${event.status || "info"}${isInterfaceCall ? " interface-call" : ""}`;
  const title = document.createElement("div");
  title.className = "timeline-title";
  const name = document.createElement("strong");
  name.textContent = isInterfaceCall
    ? `外部接口 · ${event.interface || "未知接口"}`
    : event.event || "流程";
  const state = document.createElement("span");
  state.textContent = eventLabel(event);
  title.append(name, state);
  const meta = document.createElement("time");
  meta.textContent = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : "刚刚";
  item.append(title, meta);
  if (isInterfaceCall) {
    const method = event.method || event.request?.method;
    const url = event.url || event.request?.url;
    const duration = Number.isFinite(Number(event.duration_ms))
      ? `本次 ${formatDuration(event.duration_ms)}`
      : "";
    if (method || url || duration) {
      const route = document.createElement("div");
      route.className = "interface-route";
      route.textContent = [method, url, duration].filter(Boolean).join("  ·  ");
      item.append(route);
    }
    appendInterfacePayload(item, "完整入参 · REQUEST", event.request);
    appendInterfacePayload(item, "完整出参 · RESPONSE", event.response);
  } else {
    const detail = document.createElement("pre");
    const details = { ...event };
    delete details.event;
    delete details.status;
    delete details.timestamp;
    detail.textContent = Object.keys(details).length ? JSON.stringify(details, null, 2) : "";
    item.append(detail);
  }
  target.append(item);
  item.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function taskStatusText(event) {
  const status = eventLabel(event);
  return `${event.event || "流程"} · ${status}`;
}

function formatTaskError(body, fallback = "任务执行失败") {
  if (!body) return fallback;
  if (typeof body === "string") return body;
  const lines = [];
  if (body.error_code) lines.push(`错误码: ${body.error_code}`);
  if (body.failed_step) lines.push(`失败步骤: ${body.failed_step}`);
  if (body.failed_interface) lines.push(`失败接口: ${body.failed_interface}`);
  if (body.message) lines.push(`错误消息: ${body.message}`);
  if (body.url) lines.push(`请求地址: ${body.url}`);
  if (body.current_action_status) lines.push(`动作状态: ${body.current_action_status}`);
  if (body.current_action_id) lines.push(`动作编号: ${body.current_action_id}`);
  if (!lines.length) return JSON.stringify(body, null, 2);
  const details = Object.entries(body)
    .filter(([key]) => ![
      "error_code",
      "failed_step",
      "failed_interface",
      "message",
      "url",
      "current_action_status",
      "current_action_id",
    ].includes(key));
  if (details.length) lines.push(`上下文: ${JSON.stringify(Object.fromEntries(details), null, 2)}`);
  return lines.join("\n");
}

function selectedTaskId() {
  return taskForm.querySelector('input[name="task_id"]:checked')?.value || "0";
}

function task1OrderPayload() {
  if (!currentMockOrder) throw new Error("请先生成模拟订单");
  return {
    order_source: "mock_random",
    order_id: currentMockOrder.order_id,
    product_names: [...currentMockOrder.product_names],
  };
}

function setMockOrderStatus(message, state = "") {
  mockOrderStatus.textContent = message;
  mockOrderStatus.className = state;
}

function normalizeMockOrder(value) {
  const orderIdValue = typeof value?.order_id === "string" ? value.order_id.trim() : "";
  const source = typeof value?.source === "string" ? value.source.trim() : "";
  const products = Array.isArray(value?.product_names)
    ? value.product_names
      .filter((name) => typeof name === "string" && name.trim())
      .map((name) => name.trim())
    : [];
  const catalogSize = Number(value?.catalog_size);
  if (!orderIdValue || source !== "mock_random" || products.length !== 2) {
    throw new Error("模拟点单接口返回格式不正确");
  }
  if (new Set(products).size !== products.length) {
    throw new Error("模拟点单接口返回了重复商品");
  }
  return {
    order_id: orderIdValue,
    source,
    catalog_size: Number.isFinite(catalogSize) && catalogSize > 0 ? catalogSize : null,
    product_names: products,
  };
}

function renderMockOrder(order) {
  mockOrderProducts.replaceChildren();
  order.product_names.forEach((name) => {
    const item = document.createElement("li");
    item.textContent = name;
    mockOrderProducts.append(item);
  });
  mockOrderProducts.hidden = false;
  mockOrderEmpty.hidden = true;
  mockOrderId.textContent = order.order_id;
  mockOrderCatalogSize.textContent = order.catalog_size === null ? "-" : `${order.catalog_size} 个 SKU`;
  setMockOrderStatus("已生成 · 2 件", "success");
}

function showMockOrderError(message) {
  currentMockOrder = null;
  mockOrderProducts.hidden = true;
  mockOrderProducts.replaceChildren();
  mockOrderEmpty.hidden = false;
  mockOrderEmpty.textContent = message;
  mockOrderId.textContent = "-";
  mockOrderCatalogSize.textContent = "-";
  setMockOrderStatus("生成失败", "failure");
}

function syncTaskControls() {
  const taskId = selectedTaskId();
  taskSubmitButton.disabled = taskBusy || (taskId === "1" && (mockOrderLoading || !currentMockOrder));
  taskTerminateButton.hidden = !taskBusy || !currentTaskRunId;
  taskTerminateButton.disabled = !taskBusy || !currentTaskRunId;
  taskTerminateButton.querySelector("span:last-child").textContent = "终止任务";
  taskForm.querySelectorAll('input[name="task_id"]').forEach((input) => { input.disabled = taskBusy; });
  mockOrderRefreshButton.disabled = taskBusy || mockOrderLoading;
  taskSubmitButton.querySelector("span:last-child").textContent = taskBusy
    ? "执行中"
    : `开始 Task ${taskId}`;
}

async function requestMockOrder() {
  if (mockOrderLoading || taskBusy) return currentMockOrder;
  mockOrderLoading = true;
  mockOrderProducts.hidden = true;
  mockOrderEmpty.hidden = false;
  mockOrderEmpty.textContent = "正在从模拟点单系统获取订单...";
  setMockOrderStatus("生成中");
  syncTaskControls();
  try {
    const response = await fetch("/api/task1/mock-order", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const message = body?.detail || body?.message;
      throw new Error(typeof message === "string" ? message : `模拟点单失败（HTTP ${response.status}）`);
    }
    const order = normalizeMockOrder(body);
    currentMockOrder = order;
    renderMockOrder(order);
    if (selectedTaskId() === "1") setTaskError();
    return order;
  } catch (error) {
    const message = error.message || "无法生成模拟订单";
    showMockOrderError(message);
    if (selectedTaskId() === "1") setTaskError(message);
    throw error;
  } finally {
    mockOrderLoading = false;
    syncTaskControls();
  }
}

function updateTaskSelection() {
  const taskId = selectedTaskId();
  const info = TASK_INFO[taskId];
  taskTitle.textContent = info.title;
  taskDescription.textContent = info.description;
  task1MockOrder.hidden = taskId !== "1";
  taskInterfaceMetrics.hidden = taskId !== "1";
  taskProgress.hidden = taskId !== "0";
  syncTaskControls();
  if (taskId === "1" && !currentMockOrder && !mockOrderLoading) {
    requestMockOrder().catch(() => { /* The inline error keeps retry available. */ });
  }
}

function setTaskProgress(completed = 0, total = 0) {
  const safeCompleted = Math.max(0, Number(completed) || 0);
  const safeTotal = Math.max(0, Number(total) || 0);
  const boundedCompleted = safeTotal ? Math.min(safeCompleted, safeTotal) : safeCompleted;
  taskProgressText.textContent = `${boundedCompleted} / ${safeTotal || "-"}`;
  taskProgressTrack.setAttribute("aria-valuemax", String(safeTotal));
  taskProgressTrack.setAttribute("aria-valuenow", String(boundedCompleted));
  taskProgressBar.style.width = safeTotal
    ? `${Math.round((boundedCompleted / safeTotal) * 100)}%`
    : "0";
}

function appendTaskCapture(capture) {
  if (!capture?.directory) return;
  const exists = [...taskCaptureList.children]
    .some((item) => item.dataset.directory === capture.directory);
  if (exists) return;
  const item = document.createElement("li");
  item.dataset.directory = capture.directory;
  const title = document.createElement("strong");
  const pose = String(capture.pose_type || "").replace("SHELF_VIEW_", "") || "RGB-D";
  title.textContent = `${capture.target_id || "未知点位"} · ${pose}`;
  const directory = document.createElement("span");
  directory.textContent = capture.directory;
  item.append(title, directory);
  taskCaptureList.append(item);
  taskCaptureCount.textContent = `${taskCaptureList.children.length} 组`;
  taskCaptures.hidden = false;
}

function appendTaskDetail(titleText, detailText, key) {
  if ([...taskDetailsList.children].some((item) => item.dataset.key === key)) return;
  const item = document.createElement("li");
  item.dataset.key = key;
  const title = document.createElement("strong");
  title.textContent = titleText;
  const detail = document.createElement("span");
  detail.textContent = detailText;
  item.append(title, detail);
  taskDetailsList.append(item);
  taskDetailsCount.textContent = `${taskDetailsList.children.length} 项`;
  taskDetails.hidden = false;
}

function metricNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : fallback;
}

function formatDuration(milliseconds) {
  if (milliseconds === null || milliseconds === undefined || milliseconds === "") return "-";
  const value = Number(milliseconds);
  if (!Number.isFinite(value) || value < 0) return "-";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(1)} ms`;
}

function normalizeInterfaceCall(value) {
  // Only completed HTTP-attempt events have both a unique call id and the
  // elapsed time for that attempt. Aggregate result metrics must not be
  // converted into fake per-call rows.
  const callId = typeof value?.call_id === "string" ? value.call_id.trim() : "";
  const interfaceName = typeof value?.interface === "string" ? value.interface.trim() : "";
  if (!callId || !interfaceName || value?.duration_ms === undefined || value?.duration_ms === null) return null;
  const method = typeof value?.method === "string" ? value.method.trim().toUpperCase() : "";
  const rawStatusCode = value?.status_code ?? value?.response?.status_code;
  const parsedStatusCode = Number(rawStatusCode);
  const statusCode = Number.isInteger(parsedStatusCode) && parsedStatusCode >= 100
    ? parsedStatusCode
    : null;
  const explicitStatus = typeof value?.status === "string" ? value.status.trim() : "";
  const succeeded = explicitStatus
    ? explicitStatus === "succeeded"
    : statusCode !== null && statusCode >= 200 && statusCode < 300;
  const error = typeof value?.error === "string"
    ? value.error
    : typeof value?.response?.error === "string"
      ? value.response.error
      : "";
  return {
    call_id: callId,
    interface: interfaceName,
    method,
    url: typeof value?.url === "string" ? value.url : "",
    attempt: Math.max(1, Math.trunc(metricNumber(value?.attempt, 1))),
    status_code: statusCode,
    succeeded,
    error,
    duration_ms: metricNumber(value.duration_ms),
  };
}

function renderInterfaceMetricTable() {
  taskInterfaceMetricsBody.replaceChildren();
  [...taskInterfaceCallValues.values()]
    .sort((left, right) => left.sequence - right.sequence)
    .forEach((call) => {
      const row = document.createElement("tr");
      const values = [
        `#${call.sequence}`,
        [call.method, call.interface].filter(Boolean).join(" "),
        String(call.attempt),
        call.succeeded ? "成功" : "失败",
        call.status_code === null ? "-" : String(call.status_code),
        formatDuration(call.duration_ms),
      ];
      values.forEach((value, index) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        if (index === 1 && call.url) cell.title = call.url;
        if (index === 3 && call.error) cell.title = call.error;
        row.append(cell);
      });
      taskInterfaceMetricsBody.append(row);
    });
  const count = taskInterfaceCallValues.size;
  taskInterfaceMetricsCount.textContent = `${count} 次调用`;
  taskInterfaceMetricsEmpty.hidden = count > 0;
  taskInterfaceMetricsBody.closest(".interface-metrics-table-wrap").hidden = count === 0;
}

function applyInterfaceCall(value) {
  const call = normalizeInterfaceCall(value);
  if (!call) return;
  const previous = taskInterfaceCallValues.get(call.call_id);
  call.sequence = previous?.sequence ?? taskInterfaceCallValues.size + 1;
  taskInterfaceCallValues.set(call.call_id, call);
  renderInterfaceMetricTable();
}

function showMockOrderEvent(event) {
  const source = event.source || event.order_source;
  const isMockOrderEvent = source === "mock_random"
    || event.event === "模拟订单"
    || event.event === "模拟点单";
  if (!isMockOrderEvent || !Array.isArray(event.product_names)) return;
  const productNames = event.product_names
    .filter((name) => typeof name === "string" && name.trim())
    .map((name) => name.trim());
  if (productNames.length !== 2) return;
  const candidate = {
    order_id: event.order_id || currentMockOrder?.order_id,
    source: "mock_random",
    catalog_size: event.catalog_size ?? currentMockOrder?.catalog_size,
    product_names: productNames,
  };
  try {
    currentMockOrder = normalizeMockOrder(candidate);
    renderMockOrder(currentMockOrder);
  } catch (_) { /* Keep the order selected before task start. */ }
  taskDetailsTitle.textContent = "模拟订单";
  productNames.forEach((name, index) => appendTaskDetail(
    name,
    `第 ${index + 1} 件 · ${event.order_id || currentMockOrder?.order_id || "未知订单"}`,
    `mock-order:${event.order_id || "current"}:${index}:${name}`,
  ));
}

function updateTaskLiveStatus(event, taskId) {
  const strong = taskLiveStatus?.querySelector("strong");
  if (!strong) return;
  strong.textContent = taskStatusText(event);
  taskLiveStatus.className = `live-status ${event.status || "info"}`;
  if (taskId === "0" && event.event === "operation" && event.status === "started") {
    setTaskProgress(0, event.total_captures);
  }
  if (taskId === "0" && event.event === "RGB-D采集" && event.status === "succeeded") {
    setTaskProgress(event.capture_number, event.total_captures);
    appendTaskCapture(event);
  }
  if (taskId === "1") showMockOrderEvent(event);
  if (taskId === "2" && event.event === "缺货记录" && event.status === "succeeded") {
    taskDetailsTitle.textContent = "缺货记录";
    appendTaskDetail(
      event.product_name || "未知商品",
      `${event.target_id || "未知点位"} · ${event.pose_type || "未知姿态"}`,
      `${event.product_name}:${event.target_id}:${event.pose_type}`,
    );
  }
  if (taskId === "3" && event.event === "乱放识别" && event.status === "succeeded") {
    const findings = Array.isArray(event.findings) ? event.findings : [];
    taskDetailsTitle.textContent = "乱放识别";
    findings.forEach((finding) => appendTaskDetail(
      finding.misplaced_product_name || "未知商品",
      `应为 ${finding.gt_product_name || "未知商品"}`,
      `${finding.misplaced_product_name}:${finding.gt_product_name}`,
    ));
  }
  // Internal actions and HTTP attempts may fail and then recover. Only the
  // operation terminal event means the whole task has stopped.
  if (event.status === "failed" && event.event === "operation") {
    const body = {
      error_code: event.error_code,
      failed_step: event.step,
      failed_interface: event.failed_interface,
      message: event.message,
      url: event.url,
    };
    taskErrorDetails.hidden = false;
    taskErrorBody.textContent = formatTaskError(body);
  }
}

function setTaskError(message = "") {
  taskErrorMessage.textContent = message;
  taskErrorMessage.hidden = !message;
}

function resetTaskView() {
  elapsedTimers.task.reset();
  stopVisualPolling("task");
  taskTimeline.hidden = false;
  taskTimeline.replaceChildren();
  taskCaptures.hidden = true;
  taskCaptureList.replaceChildren();
  taskCaptureCount.textContent = "0 组";
  taskInterfaceCallValues.clear();
  renderInterfaceMetricTable();
  taskDetails.hidden = true;
  taskDetailsList.replaceChildren();
  taskDetailsCount.textContent = "-";
  taskResultCard.hidden = true;
  taskErrorDetails.hidden = true;
  taskErrorBody.textContent = "-";
  taskLiveStatus.className = "live-status";
  taskLiveStatus.querySelector("strong").textContent = "等待启动";
  taskVisual.hidden = true;
  resetVisualPanel(taskVisualCanvas, taskVisualStatus, taskPoseStatus, taskPoseValues, taskPoseMeta);
  taskOperationKey.textContent = "-";
  setTaskProgress();
  setTaskError();
}

function setTaskBusy(busy) {
  taskBusy = busy;
  syncTaskControls();
}

function setTaskTerminating(terminating) {
  taskTerminateButton.disabled = terminating;
  taskTerminateButton.querySelector("span:last-child").textContent = terminating
    ? "终止中"
    : "终止任务";
  if (terminating) {
    taskLiveStatus.className = "live-status failed";
    taskLiveStatus.querySelector("strong").textContent = "正在终止任务";
  }
}

function renderTaskSpecificResult(taskId, body) {
  if (taskId === "0") {
    const captures = Array.isArray(body?.captures) ? body.captures : [];
    taskCaptureList.replaceChildren();
    captures.forEach(appendTaskCapture);
    setTaskProgress(captures.length, captures.length);
    return;
  }
  if (taskId === "2") {
    taskDetailsTitle.textContent = "补货商品";
  } else if (taskId === "3") {
    taskDetailsTitle.textContent = "货位交换";
  } else {
    return;
  }
  taskDetailsList.replaceChildren();
  const items = Array.isArray(body?.target_items) ? body.target_items : [];
  items.forEach((item, index) => {
    const hand = item.hand || "UNKNOWN";
    const route = taskId === "3"
      ? `${item.source_slot_id} → ${item.destination_slot_id} · ${hand}`
      : `${item.inspection_target_id} · ${item.inspection_pose_type} · ${hand}`;
    appendTaskDetail(item.product_name || `商品 ${index + 1}`, route, `${index}:${route}`);
  });
}

function showTaskResult(taskId, result) {
  const body = result.body ?? result;
  const terminated = body?.error_code === "TASK_TERMINATED";
  if (taskId === "1" && body && typeof body === "object") {
    if (Array.isArray(body.product_names)) {
      showMockOrderEvent({
        ...currentMockOrder,
        ...(body.order && typeof body.order === "object" ? body.order : {}),
        ...body,
        source: "mock_random",
      });
    }
  }
  taskResultCard.hidden = false;
  taskResultStatus.textContent = terminated
    ? "任务已终止"
    : result.ok
    ? `HTTP ${result.status_code} · 请求完成`
    : `HTTP ${result.status_code || 502} · 请求失败`;
  taskResultStatus.className = result.ok ? "success" : "failure";
  taskResultBody.textContent = typeof body === "string"
    ? body
    : JSON.stringify(body, null, 2);
  if (terminated) {
    setTaskError();
    taskErrorDetails.hidden = false;
    taskErrorBody.textContent = body.message || "任务已由用户终止";
    taskLiveStatus.className = "live-status failed";
    taskLiveStatus.querySelector("strong").textContent = "任务已终止";
  } else if (result.ok) {
    setTaskError();
    taskErrorDetails.hidden = true;
    taskErrorBody.textContent = "-";
    renderTaskSpecificResult(taskId, body);
    taskLiveStatus.className = "live-status succeeded";
    taskLiveStatus.querySelector("strong").textContent = "任务完成 · 成功";
  } else {
    const error = formatTaskError(body);
    setTaskError(error);
    taskErrorDetails.hidden = false;
    taskErrorBody.textContent = error;
    taskLiveStatus.className = "live-status failed";
    taskLiveStatus.querySelector("strong").textContent = "任务失败 · 已停止";
  }
}

function showResult(result) {
  const body = result.body ?? result;
  resultCard.hidden = false;
  resultStatus.textContent = result.ok ? `HTTP ${result.status_code} · 请求完成` : `HTTP ${result.status_code || 502} · 请求失败`;
  resultStatus.className = result.ok ? "success" : "failure";
  resultBody.textContent = typeof body === "string" ? body : JSON.stringify(body, null, 2);
  connectionText.textContent = result.ok ? "任务完成" : "任务失败";
  pulse.classList.toggle("ready", result.ok);
  if (!result.ok) setError(typeof body === "string" ? body : body?.message || "取放请求失败");
}

function showPlaceResult(result) {
  const body = result.body ?? result;
  placeResultCard.hidden = false;
  placeResultStatus.textContent = result.ok
    ? `HTTP ${result.status_code} · 请求完成`
    : `HTTP ${result.status_code || 502} · 请求失败`;
  placeResultStatus.className = result.ok ? "success" : "failure";
  placeResultBody.textContent = typeof body === "string" ? body : JSON.stringify(body, null, 2);
  connectionText.textContent = result.ok ? "放置完成" : "放置失败";
  pulse.classList.toggle("ready", result.ok);
  if (!result.ok) setPlaceError(typeof body === "string" ? body : body?.message || "放置请求失败");
}

async function startPick(payload) {
  const response = await fetch("/api/pick/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || body.message || "无法启动取放任务");
  return body;
}

async function startPlace(payload) {
  const response = await fetch("/api/place/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || body.message || "无法启动放置任务");
  return body;
}

async function startUnifiedTask(taskId, payload = {}) {
  const response = await fetch(`/api/tasks/${taskId}/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || body.message || `无法启动 Task ${taskId}`);
  return body;
}

async function terminateUnifiedTask(runId) {
  const response = await fetch(`/api/task-runs/${runId}/terminate`, {
    method: "POST",
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || body.message || `终止任务失败（HTTP ${response.status}）`);
  }
  return body;
}

function robotKey(prefix) {
  return `web:${prefix}:${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
}

async function callRobot(path, payload, prefix) {
  robotStatus.textContent = "请求发送中";
  robotOutput.textContent = JSON.stringify({ request: payload }, null, 2);
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Idempotency-Key": robotKey(prefix),
    },
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({ message: "服务返回了非 JSON 数据" }));
  robotOutput.textContent = JSON.stringify(body, null, 2);
  robotStatus.textContent = response.ok && body.ok !== false ? `执行完成 · HTTP ${response.status}` : `执行失败 · HTTP ${response.status}`;
  robotStatus.className = response.ok && body.ok !== false ? "robot-status success" : "robot-status failure";
  if (!response.ok || body.ok === false) throw new Error(body.body?.message || body.detail || "机器人接口调用失败");
  return body;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (eventSource) eventSource.close();
  resetView();
  setBusy(true);
  const data = new FormData(form);
  const payload = {
    task_type: data.get("task_type"),
    product_name: data.get("product_name"),
    hand: data.get("hand"),
  };
  try {
    const task = await startPick(payload);
    elapsedTimers.pick.start();
    operationKey.textContent = task.operation_key;
    beginVisualPolling(task.task_id, "/api/pick", pickVisual, pickVisualCanvas, pickVisualStatus, pickPoseStatus, pickPoseValues, pickPoseMeta);
    eventSource = new EventSource(task.events_url);
    eventSource.addEventListener("flow", (message) => {
      try { addFlowEvent(JSON.parse(message.data)); } catch (_) { /* ignore malformed event */ }
    });
    eventSource.addEventListener("result", async (message) => {
      elapsedTimers.pick.stop();
      try { showResult(JSON.parse(message.data)); } catch (_) { setError("任务结果格式无效"); }
      setBusy(false);
      await stopVisualPolling("pick", true);
      eventSource.close();
      eventSource = null;
    });
    eventSource.onerror = () => {
      if (eventSource?.readyState === EventSource.CLOSED) return;
      setError("实时日志连接中断，请查看服务器日志");
      connectionText.textContent = "日志连接中断";
      pulse.classList.remove("ready");
    };
  } catch (error) {
    setBusy(false);
    setError(error.message || "无法启动取放任务");
    connectionText.textContent = "启动失败";
    pulse.classList.remove("ready");
  }
});

placeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (placeEventSource) placeEventSource.close();
  resetPlaceView();
  setPlaceBusy(true);
  const data = new FormData(placeForm);
  const payload = {
    task_type: data.get("task_type"),
    product_name: data.get("product_name"),
    hand: data.get("hand"),
  };
  try {
    const task = await startPlace(payload);
    elapsedTimers.place.start();
    placeOperationKey.textContent = task.operation_key;
    beginVisualPolling(
      task.task_id,
      "/api/place",
      placeVisual,
      placeVisualCanvas,
      placeVisualStatus,
      placePoseStatus,
      placePoseValues,
      placePoseMeta,
      "place",
    );
    placeEventSource = new EventSource(task.events_url);
    placeEventSource.addEventListener("flow", (message) => {
      try { addFlowEvent(JSON.parse(message.data), placeTimeline); } catch (_) { /* ignore malformed event */ }
    });
    placeEventSource.addEventListener("result", async (message) => {
      elapsedTimers.place.stop();
      try { showPlaceResult(JSON.parse(message.data)); } catch (_) { setPlaceError("放置任务结果格式无效"); }
      setPlaceBusy(false);
      await stopVisualPolling("place", true);
      placeEventSource.close();
      placeEventSource = null;
    });
    placeEventSource.onerror = () => {
      if (placeEventSource?.readyState === EventSource.CLOSED) return;
      setPlaceError("放置实时日志连接中断，请查看服务器日志");
      connectionText.textContent = "放置日志连接中断";
      pulse.classList.remove("ready");
    };
  } catch (error) {
    setPlaceBusy(false);
    setPlaceError(error.message || "无法启动放置任务");
    connectionText.textContent = "放置启动失败";
    pulse.classList.remove("ready");
  }
});

mockOrderRefreshButton.addEventListener("click", () => {
  requestMockOrder().catch(() => { /* The inline error keeps retry available. */ });
});

taskForm.addEventListener("change", (event) => {
  if (event.target.matches('input[name="task_id"]')) {
    resetTaskView();
    updateTaskSelection();
  }
});

taskForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (taskEventSource) taskEventSource.close();
  const taskId = selectedTaskId();
  currentTaskRunId = null;
  resetTaskView();
  setTaskBusy(true);
  try {
    const payload = taskId === "1" ? task1OrderPayload() : {};
    const task = await startUnifiedTask(taskId, payload);
    currentTaskRunId = task.run_id;
    setTaskBusy(true);
    elapsedTimers.task.start();
    taskOperationKey.textContent = task.operation_key;
    beginVisualPolling(
      task.run_id,
      "/api/task-runs",
      taskVisual,
      taskVisualCanvas,
      taskVisualStatus,
      taskPoseStatus,
      taskPoseValues,
      taskPoseMeta,
      "task",
    );
    taskEventSource = new EventSource(task.events_url);
    taskEventSource.addEventListener("flow", (message) => {
      try {
        const flowEvent = JSON.parse(message.data);
        if (taskId === "1" && flowEvent.event === "接口调用") {
          applyInterfaceCall(flowEvent);
        }
        addFlowEvent(flowEvent, taskTimeline);
        updateTaskLiveStatus(flowEvent, taskId);
      } catch (_) { /* ignore malformed event */ }
    });
    taskEventSource.addEventListener("result", async (message) => {
      elapsedTimers.task.stop();
      try {
        showTaskResult(taskId, JSON.parse(message.data));
      } catch (_) {
        setTaskError(`Task ${taskId} 结果格式无效`);
      }
      setTaskBusy(false);
      currentTaskRunId = null;
      await stopVisualPolling("task", true);
      taskEventSource.close();
      taskEventSource = null;
    });
    taskEventSource.onerror = () => {
      if (taskEventSource?.readyState === EventSource.CLOSED) return;
      setTaskError(`Task ${taskId} 实时日志连接中断，请查看服务器日志`);
    };
  } catch (error) {
    currentTaskRunId = null;
    setTaskBusy(false);
    setTaskError(error.message || `无法启动 Task ${taskId}`);
  }
});

taskTerminateButton.addEventListener("click", async () => {
  const runId = currentTaskRunId;
  if (!runId) return;
  let terminationAccepted = false;
  setTaskTerminating(true);
  setTaskError();
  try {
    await terminateUnifiedTask(runId);
    terminationAccepted = true;
  } catch (error) {
    setTaskError(error.message || "终止任务失败");
  } finally {
    if (!terminationAccepted && currentTaskRunId === runId) setTaskTerminating(false);
  }
});

poseForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const poseType = selectedPoseType();
  const shelfLevel = selectedShelfLevel();
  if (!poseType) return;
  const payload = { pose_type: poseType };
  if (!POSES_WITHOUT_SHELF_LEVEL.has(poseType) && shelfLevel) payload.shelf_level = shelfLevel;
  try {
    await callRobot("/api/robot/prepare", payload, "pose");
  } catch (error) {
    robotStatus.textContent = "位姿准备失败";
    robotStatus.className = "robot-status failure";
    setError(error.message || "位姿准备失败");
  }
});

navigationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const target = selectedNavigationTarget();
  if (!target) return;
  try {
    await callRobot("/api/robot/navigate", { target_id: target }, "navigation");
  } catch (error) {
    robotStatus.textContent = "导航失败";
    robotStatus.className = "robot-status failure";
    setError(error.message || "导航失败");
  }
});

gripperForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const hand = new FormData(gripperForm).get("gripper_hand");
  if (hand !== "LEFT" && hand !== "RIGHT") return;
  gripperSubmitButton.disabled = true;
  gripperSubmitButton.querySelector("span:last-child").textContent = "执行中";
  try {
    await callRobot("/api/robot/gripper/open", { hand }, `gripper-open-${hand.toLowerCase()}`);
    robotStatus.textContent = `${hand === "LEFT" ? "左手" : "右手"}夹爪已松开`;
    robotStatus.className = "robot-status success";
  } catch (error) {
    robotStatus.textContent = "夹爪松手失败";
    robotStatus.className = "robot-status failure";
    setError(error.message || "夹爪松手失败");
  } finally {
    gripperSubmitButton.disabled = false;
    gripperSubmitButton.querySelector("span:last-child").textContent = "执行松手";
  }
});

healthButton.addEventListener("click", async () => {
  healthButton.disabled = true;
  robotStatus.textContent = "健康检查中";
  try {
    const response = await fetch("/api/robot/health");
    const body = await response.json();
    robotOutput.textContent = JSON.stringify(body, null, 2);
    const healthy = body.pose?.ok && body.navigation?.ok;
    robotStatus.textContent = healthy ? "导航 / 位姿服务正常" : "至少一个机器人服务异常";
    robotStatus.className = healthy ? "robot-status success" : "robot-status failure";
  } catch (error) {
    robotStatus.textContent = "健康检查失败";
    robotStatus.className = "robot-status failure";
    robotOutput.textContent = error.message || "健康检查失败";
  } finally {
    healthButton.disabled = false;
  }
});

function setRobotIpMessage(message = "", state = "") {
  robotIpMessage.textContent = message;
  robotIpMessage.className = `system-message ${state}`.trim();
}

function applyRuntimeConfig(config, updateInput = true) {
  applyNavigationTargets(config.navigation_targets);
  currentRobotIp.textContent = config.robot_ip || "-";
  if (updateInput || !robotIpInput.value) robotIpInput.value = config.robot_ip || "";
  robotIpRuntimeStatus.textContent = config.restart_supported
    ? "8086 / 8108 受管运行"
    : config.restart_unavailable_reason || "网页重启不可用";
  robotIpSubmitButton.disabled = !config.restart_supported;
  document.querySelector("#footerUrl").textContent = `${config.tasks_url} · ${config.pick_url} · ${config.place_url}`;
}

async function loadRuntimeConfig(updateInput = true) {
  const response = await fetch("/api/config", { cache: "no-store" });
  if (!response.ok) throw new Error(`配置接口返回 HTTP ${response.status}`);
  const config = await response.json();
  applyRuntimeConfig(config, updateInput);
  return config;
}

function responseMessage(body, fallback) {
  if (typeof body?.message === "string") return body.message;
  if (Array.isArray(body?.detail)) {
    return body.detail.map((item) => item.msg || JSON.stringify(item)).join("；");
  }
  return fallback;
}

function closeRunningConnections() {
  [eventSource, placeEventSource, taskEventSource].forEach((source) => source?.close());
  eventSource = null;
  placeEventSource = null;
  taskEventSource = null;
  Object.values(elapsedTimers).forEach((timer) => timer.stop());
}

async function waitForRuntime(robotIp) {
  const deadline = Date.now() + 90000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
    try {
      const config = await loadRuntimeConfig(false);
      if (config.robot_ip === robotIp && config.restart_supported) {
        robotIpInput.value = robotIp;
        setBusy(false);
        setPlaceBusy(false);
        setTaskBusy(false);
        connectionText.textContent = "接口待命";
        pulse.classList.add("ready");
        setRobotIpMessage(`机器人 IP 已更新为 ${robotIp}，两个服务已重启`, "success");
        return;
      }
    } catch (_) {
      robotIpRuntimeStatus.textContent = "服务重启中";
    }
  }
  robotIpSubmitButton.disabled = false;
  setRobotIpMessage("等待服务重启超时，请检查 log/process 中的重启日志", "failure");
}

async function updateRobotIp(forceRestart = false) {
  const robotIp = robotIpInput.value.trim();
  robotIpSubmitButton.disabled = true;
  setRobotIpMessage(forceRestart ? "正在强制应用配置" : "正在检查运行状态");
  try {
    const response = await fetch("/api/system/robot-ip", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ robot_ip: robotIp, force_restart: forceRestart }),
    });
    const body = await response.json().catch(() => ({}));
    if (response.status === 409 && body.requires_force) {
      activeOperationsSummary.textContent = JSON.stringify(body.active, null, 2);
      setRobotIpMessage("检测到正在执行的任务，等待强制重启确认", "failure");
      robotIpSubmitButton.disabled = false;
      forceRestartDialog.showModal();
      return;
    }
    if (!response.ok) {
      throw new Error(responseMessage(body, `配置更新失败（HTTP ${response.status}）`));
    }
    closeRunningConnections();
    currentRobotIp.textContent = robotIp;
    robotIpRuntimeStatus.textContent = "服务重启中";
    setRobotIpMessage("配置已保存，正在重启 8086 和 8108");
    await waitForRuntime(robotIp);
  } catch (error) {
    robotIpSubmitButton.disabled = false;
    setRobotIpMessage(error.message || "机器人 IP 更新失败", "failure");
  }
}

robotIpForm.addEventListener("submit", (event) => {
  event.preventDefault();
  updateRobotIp(false);
});

forceRestartDialog.addEventListener("close", () => {
  if (forceRestartDialog.returnValue === "force") updateRobotIp(true);
});

loadRuntimeConfig()
  .catch((error) => {
    document.querySelector("#footerUrl").textContent = "配置不可用";
    robotIpRuntimeStatus.textContent = "配置不可用";
    robotIpSubmitButton.disabled = true;
    setRobotIpMessage(error.message || "无法读取运行配置", "failure");
  });

updateTaskSelection();
