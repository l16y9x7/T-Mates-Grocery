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
let eventSource = null;
let task1EventSource = null;

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
  emptyState.hidden = true;
  timeline.hidden = false;
  timeline.replaceChildren();
  resultCard.hidden = true;
  operationKey.textContent = "-";
  setError();
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
  task1EmptyState.hidden = true;
  task1Timeline.hidden = false;
  task1Timeline.replaceChildren();
  task1ResultCard.hidden = true;
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
    eventSource = new EventSource(task.events_url);
    eventSource.addEventListener("flow", (message) => {
      try { addFlowEvent(JSON.parse(message.data)); } catch (_) { /* ignore malformed event */ }
    });
    eventSource.addEventListener("result", (message) => {
      try { showResult(JSON.parse(message.data)); } catch (_) { setError("任务结果格式无效"); }
      setBusy(false);
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
    task1EventSource = new EventSource(task.events_url);
    task1EventSource.addEventListener("flow", (message) => {
      try { addFlowEvent(JSON.parse(message.data), task1Timeline); } catch (_) { /* ignore malformed event */ }
    });
    task1EventSource.addEventListener("result", (message) => {
      try { showTask1Result(JSON.parse(message.data)); } catch (_) { setTask1Error("任务一结果格式无效"); }
      setTask1Busy(false);
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
