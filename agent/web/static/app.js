const form = document.querySelector("#pickForm");
const submitButton = document.querySelector("#submitButton");
const errorMessage = document.querySelector("#errorMessage");
const emptyState = document.querySelector("#emptyState");
const timeline = document.querySelector("#timeline");
const resultCard = document.querySelector("#resultCard");
const resultStatus = document.querySelector("#resultStatus");
const resultBody = document.querySelector("#resultBody");
const operationKey = document.querySelector("#operationKey");
const connectionText = document.querySelector("#connectionText");
const pulse = document.querySelector("#pulse");
const poseForm = document.querySelector("#poseForm");
const poseTypePreset = document.querySelector("#poseTypePreset");
const poseTypeCustom = document.querySelector("#poseTypeCustom");
const shelfLevelPreset = document.querySelector("#shelfLevelPreset");
const shelfLevelCustom = document.querySelector("#shelfLevelCustom");
const shelfLevelLabel = document.querySelector("#shelfLevelLabel");
const navigationForm = document.querySelector("#navigationForm");
const healthButton = document.querySelector("#healthButton");
const robotStatus = document.querySelector("#robotStatus");
const robotOutput = document.querySelector("#robotOutput");
const parseReceiptButton = document.querySelector("#parseReceiptButton");
const receiptStatus = document.querySelector("#receiptStatus");
const receiptProducts = document.querySelector("#receiptProducts");
const receiptEmpty = document.querySelector("#receiptEmpty");
const receiptOutput = document.querySelector("#receiptOutput");
const task1Form = document.querySelector("#task1Form");
const task1SubmitButton = document.querySelector("#task1SubmitButton");
const task1PickCount = document.querySelector("#task1PickCount");
const task1ErrorMessage = document.querySelector("#task1ErrorMessage");
const task1EmptyState = document.querySelector("#task1EmptyState");
const task1Timeline = document.querySelector("#task1Timeline");
const task1ResultCard = document.querySelector("#task1ResultCard");
const task1ResultStatus = document.querySelector("#task1ResultStatus");
const task1ResultBody = document.querySelector("#task1ResultBody");
const task1OperationKey = document.querySelector("#task1OperationKey");
const pickVisual = document.querySelector("#pickVisual");
const pickVisualCanvas = document.querySelector("#pickVisualCanvas");
const pickVisualStatus = document.querySelector("#pickVisualStatus");
const pickPoseStatus = document.querySelector("#pickPoseStatus");
const pickPoseValues = document.querySelector("#pickPoseValues");
const pickPoseMeta = document.querySelector("#pickPoseMeta");
const task1Visual = document.querySelector("#task1Visual");
const task1VisualCanvas = document.querySelector("#task1VisualCanvas");
const task1VisualStatus = document.querySelector("#task1VisualStatus");
const task1PoseStatus = document.querySelector("#task1PoseStatus");
const task1PoseValues = document.querySelector("#task1PoseValues");
const task1PoseMeta = document.querySelector("#task1PoseMeta");
let eventSource = null;
let task1EventSource = null;
let visualPoller = null;
let task1VisualPoller = null;
let refreshVisual = null;
let refreshTask1Visual = null;

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

poseTypePreset.addEventListener("change", updatePoseTypeCustomField);
poseTypeCustom.addEventListener("input", updateShelfLevelField);
shelfLevelPreset.addEventListener("change", updateShelfLevelCustomField);
updatePoseTypeCustomField();
updateShelfLevelField();

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
  stopVisualPolling();
  emptyState.hidden = true;
  timeline.hidden = false;
  timeline.replaceChildren();
  resultCard.hidden = true;
  pickVisual.hidden = true;
  resetVisualPanel(pickVisualCanvas, pickVisualStatus, pickPoseStatus, pickPoseValues, pickPoseMeta);
  operationKey.textContent = "-";
  setError();
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

function beginVisualPolling(taskId, endpoint, panel, canvas, status, poseStatus, poseValues, poseMeta, isTask1 = false) {
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
  const timer = window.setInterval(poll, 700);
  if (isTask1) {
    task1VisualPoller = timer;
    refreshTask1Visual = poll;
  } else {
    visualPoller = timer;
    refreshVisual = poll;
  }
}

async function stopVisualPolling(isTask1 = false, refresh = false) {
  const timer = isTask1 ? task1VisualPoller : visualPoller;
  if (timer) window.clearInterval(timer);
  const finalRefresh = isTask1 ? refreshTask1Visual : refreshVisual;
  if (refresh && finalRefresh) await finalRefresh();
  if (isTask1) {
    task1VisualPoller = null;
    refreshTask1Visual = null;
  } else {
    visualPoller = null;
    refreshVisual = null;
  }
}

function eventLabel(event) {
  const labels = { started: "开始", succeeded: "完成", failed: "失败" };
  return labels[event.status] || event.status || "更新";
}

function addFlowEvent(event, target = timeline) {
  const item = document.createElement("article");
  item.className = `timeline-item ${event.status || "info"}`;
  const title = document.createElement("div");
  title.className = "timeline-title";
  const name = document.createElement("strong");
  name.textContent = event.event || "流程";
  const state = document.createElement("span");
  state.textContent = eventLabel(event);
  title.append(name, state);
  const meta = document.createElement("time");
  meta.textContent = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : "刚刚";
  const detail = document.createElement("pre");
  const details = { ...event };
  delete details.event;
  delete details.status;
  delete details.timestamp;
  detail.textContent = Object.keys(details).length ? JSON.stringify(details, null, 2) : "";
  item.append(title, meta, detail);
  target.append(item);
  item.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function setTask1Error(message = "") {
  task1ErrorMessage.textContent = message;
  task1ErrorMessage.hidden = !message;
}

function resetTask1View() {
  stopVisualPolling(true);
  task1EmptyState.hidden = true;
  task1Timeline.hidden = false;
  task1Timeline.replaceChildren();
  task1ResultCard.hidden = true;
  task1Visual.hidden = true;
  resetVisualPanel(task1VisualCanvas, task1VisualStatus, task1PoseStatus, task1PoseValues, task1PoseMeta);
  task1OperationKey.textContent = "-";
  setTask1Error();
}

function setTask1Busy(busy) {
  task1SubmitButton.disabled = busy;
  task1PickCount.disabled = busy;
  task1SubmitButton.querySelector("span:last-child").textContent = busy ? "执行中" : "开始任务一";
}

function showTask1Result(result) {
  const body = result.body ?? result;
  task1ResultCard.hidden = false;
  task1ResultStatus.textContent = result.ok
    ? `HTTP ${result.status_code} · 请求完成`
    : `HTTP ${result.status_code || 502} · 请求失败`;
  task1ResultStatus.className = result.ok ? "success" : "failure";
  task1ResultBody.textContent = typeof body === "string" ? body : JSON.stringify(body, null, 2);
  if (!result.ok) setTask1Error(typeof body === "string" ? body : body?.message || "任务一执行失败");
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

async function startTask1(payload) {
  const response = await fetch("/api/task1/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.detail || body.message || "无法启动任务一");
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
    operationKey.textContent = task.operation_key;
    beginVisualPolling(task.task_id, "/api/pick", pickVisual, pickVisualCanvas, pickVisualStatus, pickPoseStatus, pickPoseValues, pickPoseMeta);
    eventSource = new EventSource(task.events_url);
    eventSource.addEventListener("flow", (message) => {
      try { addFlowEvent(JSON.parse(message.data)); } catch (_) { /* ignore malformed event */ }
    });
    eventSource.addEventListener("result", async (message) => {
      try { showResult(JSON.parse(message.data)); } catch (_) { setError("任务结果格式无效"); }
      setBusy(false);
      await stopVisualPolling(false, true);
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
    emptyState.hidden = false;
    timeline.hidden = true;
    setError(error.message || "无法启动取放任务");
    connectionText.textContent = "启动失败";
    pulse.classList.remove("ready");
  }
});

task1Form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (task1EventSource) task1EventSource.close();
  resetTask1View();
  setTask1Busy(true);
  try {
    const task = await startTask1({ pick_count: Number(task1PickCount.value) });
    task1OperationKey.textContent = task.operation_key;
    beginVisualPolling(task.task_id, "/api/task1", task1Visual, task1VisualCanvas, task1VisualStatus, task1PoseStatus, task1PoseValues, task1PoseMeta, true);
    task1EventSource = new EventSource(task.events_url);
    task1EventSource.addEventListener("flow", (message) => {
      try { addFlowEvent(JSON.parse(message.data), task1Timeline); } catch (_) { /* ignore malformed event */ }
    });
    task1EventSource.addEventListener("result", async (message) => {
      try { showTask1Result(JSON.parse(message.data)); } catch (_) { setTask1Error("任务一结果格式无效"); }
      setTask1Busy(false);
      await stopVisualPolling(true, true);
      task1EventSource.close();
      task1EventSource = null;
    });
    task1EventSource.onerror = () => {
      if (task1EventSource?.readyState === EventSource.CLOSED) return;
      setTask1Error("任务一实时日志连接中断，请查看服务器日志");
    };
  } catch (error) {
    setTask1Busy(false);
    task1EmptyState.hidden = false;
    task1Timeline.hidden = true;
    setTask1Error(error.message || "无法启动任务一");
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
  const targetId = document.querySelector("#targetId").value.trim();
  if (!targetId) return;
  try {
    await callRobot("/api/robot/navigate", { target_id: targetId }, "navigation");
  } catch (error) {
    robotStatus.textContent = "导航失败";
    robotStatus.className = "robot-status failure";
    setError(error.message || "导航失败");
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

fetch("/api/config")
  .then((response) => response.json())
  .then((config) => { document.querySelector("#footerUrl").textContent = config.pick_url; })
  .catch(() => { document.querySelector("#footerUrl").textContent = "配置不可用"; });
