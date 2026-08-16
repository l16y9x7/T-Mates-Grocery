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
const targetId = document.querySelector("#targetId");
const gripperForm = document.querySelector("#gripperForm");
const gripperSubmitButton = document.querySelector("#gripperSubmitButton");
const healthButton = document.querySelector("#healthButton");
const robotStatus = document.querySelector("#robotStatus");
const robotOutput = document.querySelector("#robotOutput");
const parseReceiptButton = document.querySelector("#parseReceiptButton");
const receiptStatus = document.querySelector("#receiptStatus");
const receiptProducts = document.querySelector("#receiptProducts");
const receiptEmpty = document.querySelector("#receiptEmpty");
const receiptOutput = document.querySelector("#receiptOutput");
const taskForm = document.querySelector("#taskForm");
const taskSubmitButton = document.querySelector("#taskSubmitButton");
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
  1: { title: "订单分拣", description: "识别小票商品并完成货架抓取和交付台放置" },
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
  if (imageData && imageData !== imageCache.imageData) {
    imageCache.imageData = imageData;
    imageCache.image = new Image();
    imageCache.image.onload = () => drawVisual(visual, canvas, status, poseStatus, poseValues, poseMeta, imageCache);
    imageCache.image.src = imageData;
  }
  if (imageCache.image?.complete) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(imageCache.image, 0, 0, canvas.width, canvas.height);
    if (visual.mask_data && visual.mask_data !== imageCache.maskData) {
      imageCache.maskData = visual.mask_data;
      imageCache.mask = new Image();
      imageCache.mask.onload = () => drawVisual(visual, canvas, status, poseStatus, poseValues, poseMeta, imageCache);
      imageCache.mask.src = visual.mask_data;
    }
    if (imageCache.mask?.complete) {
      context.save();
      drawMaskOverlay(context, imageCache.mask, canvas.width, canvas.height);
      context.restore();
    }
    if (Array.isArray(visual.bbox) && visual.bbox.length === 4) {
      const space = Number(visual.bbox_coordinate_space) || 1000;
      const scaleX = canvas.width / space;
      const scaleY = canvas.height / space;
      const [x1, y1, x2, y2] = visual.bbox.map(Number);
      context.save();
      context.strokeStyle = "#d9f26b";
      context.lineWidth = 3;
      context.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
      context.fillStyle = "#d9f26b";
      context.font = "600 13px ui-monospace, monospace";
      context.fillText("BBOX", x1 * scaleX + 6, Math.max(18, y1 * scaleY - 7));
      context.restore();
    }
    status.textContent = visual.mask_data ? "RGB + MASK + BBOX" : visual.image_data ? "RGB + BBOX" : "定位结果";
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
  const labels = { started: "开始", succeeded: "完成", failed: "失败" };
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
    const method = event.request?.method;
    const url = event.request?.url;
    if (method || url) {
      const route = document.createElement("div");
      route.className = "interface-route";
      route.textContent = [method, url].filter(Boolean).join("  ");
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
  return new FormData(taskForm).get("task_id") || "0";
}

function updateTaskSelection() {
  const taskId = selectedTaskId();
  const info = TASK_INFO[taskId];
  taskTitle.textContent = info.title;
  taskDescription.textContent = info.description;
  taskSubmitButton.querySelector("span:last-child").textContent = `开始 Task ${taskId}`;
  taskProgress.hidden = taskId !== "0";
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
  if (event.status === "failed") {
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
  taskSubmitButton.disabled = busy;
  taskForm.querySelectorAll('input[name="task_id"]').forEach((input) => { input.disabled = busy; });
  taskSubmitButton.querySelector("span:last-child").textContent = busy
    ? "执行中"
    : `开始 Task ${selectedTaskId()}`;
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
  taskResultCard.hidden = false;
  taskResultStatus.textContent = result.ok
    ? `HTTP ${result.status_code} · 请求完成`
    : `HTTP ${result.status_code || 502} · 请求失败`;
  taskResultStatus.className = result.ok ? "success" : "failure";
  taskResultBody.textContent = typeof body === "string"
    ? body
    : JSON.stringify(body, null, 2);
  if (result.ok) {
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

async function startUnifiedTask(taskId) {
  const response = await fetch(`/api/tasks/${taskId}/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || body.message || `无法启动 Task ${taskId}`);
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

function setReceiptStatus(message, state = "") {
  receiptStatus.textContent = message;
  receiptStatus.className = state;
}

function renderReceiptProducts(productNames) {
  receiptProducts.replaceChildren();
  if (!productNames.length) {
    receiptProducts.hidden = true;
    receiptEmpty.hidden = false;
    receiptEmpty.textContent = "接口未识别到商品名称";
    return;
  }
  productNames.forEach((name) => {
    const item = document.createElement("li");
    item.textContent = name;
    receiptProducts.append(item);
  });
  receiptProducts.hidden = false;
  receiptEmpty.hidden = true;
}

parseReceiptButton.addEventListener("click", async () => {
  parseReceiptButton.disabled = true;
  parseReceiptButton.querySelector("span:last-child").textContent = "识别中";
  setReceiptStatus("请求发送中");
  receiptProducts.hidden = true;
  receiptEmpty.hidden = false;
  receiptEmpty.textContent = "正在读取小票...";
  try {
    const response = await fetch("/api/perception/parse", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": robotKey("perception-parse"),
      },
      body: "{}",
    });
    const result = await response.json().catch(() => ({ body: { message: "服务返回了非 JSON 数据" } }));
    const body = result.body ?? result;
    receiptOutput.textContent = JSON.stringify(body, null, 2);
    const productNames = body && Array.isArray(body.product_names)
      ? body.product_names.filter((name) => typeof name === "string" && name.trim())
      : null;
    if (!response.ok || result.ok === false) {
      throw new Error(body?.message || `小票识别失败（HTTP ${response.status}）`);
    }
    if (productNames === null) {
      throw new Error("小票识别响应缺少 product_names");
    }
    renderReceiptProducts(productNames);
    setReceiptStatus(`识别完成 · ${productNames.length} 项`, "success");
  } catch (error) {
    receiptProducts.hidden = true;
    receiptEmpty.hidden = false;
    receiptEmpty.textContent = error.message || "小票识别失败";
    receiptOutput.textContent = JSON.stringify({ message: error.message || "小票识别失败" }, null, 2);
    setReceiptStatus("识别失败", "failure");
  } finally {
    parseReceiptButton.disabled = false;
    parseReceiptButton.querySelector("span:last-child").textContent = "识别小票";
  }
});

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
  resetTaskView();
  setTaskBusy(true);
  try {
    const task = await startUnifiedTask(taskId);
    elapsedTimers.task.start();
    taskOperationKey.textContent = task.operation_key;
    if (taskId === "1") {
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
    }
    taskEventSource = new EventSource(task.events_url);
    taskEventSource.addEventListener("flow", (message) => {
      try {
        const flowEvent = JSON.parse(message.data);
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
      await stopVisualPolling("task", true);
      taskEventSource.close();
      taskEventSource = null;
    });
    taskEventSource.onerror = () => {
      if (taskEventSource?.readyState === EventSource.CLOSED) return;
      setTaskError(`Task ${taskId} 实时日志连接中断，请查看服务器日志`);
    };
  } catch (error) {
    setTaskBusy(false);
    setTaskError(error.message || `无法启动 Task ${taskId}`);
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
