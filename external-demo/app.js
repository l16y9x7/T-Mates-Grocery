const state = {
  task: "0",
  runId: "",
  pollTimer: null,
  eventTimer: null,
  callbackUrl: "",
  events: [],
  apiCalls: 0,
  catalog: [
    { sku_id: "SKU_001", name: "NFC桔汁" },
    { sku_id: "SKU_002", name: "蒙牛纯牛奶" },
    { sku_id: "SKU_003", name: "纯甄酸奶" },
    { sku_id: "SKU_014", name: "品客薯片烧烤牛排味" },
    { sku_id: "SKU_015", name: "品客原味" },
    { sku_id: "SKU_016", name: "品客酸乳酪洋葱味" },
    { sku_id: "SKU_017", name: "奥利奥香甜不腻" },
    { sku_id: "SKU_019", name: "奥利奥浓醇巧克力味" }
  ]
};

const $ = (id) => document.getElementById(id);
const taskInfo = {
  "0": { name: "理货", endpoint: "/api/external/v1/tasks/0/runs", description: "建立货架基准，记录巡检区域状态" },
  "1": { name: "取货", endpoint: "/api/external/v1/task1/orders", description: "提交两件商品订单并触发取货" },
  "2": { name: "补货", endpoint: "/api/external/v1/tasks/2/runs", description: "依据理货数据检查货架并完成补货" }
};

function nowId(prefix) {
  const stamp = new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 14);
  return `${prefix}-${stamp}-${Math.random().toString(16).slice(2, 8)}`;
}

function setDefaultIds() {
  $("externalTaskId").value = nowId(state.task === "1" ? "ORD" : state.task === "0" ? "PREP" : "REPLENISH");
  $("idempotencyKey").value = nowId("idem");
}

function setMessage(message, error = false) {
  $("formMessage").textContent = message || "";
  $("formMessage").style.color = error ? "var(--red)" : "var(--lime)";
}

function addApiCall(method, path, status, payload) {
  state.apiCalls += 1;
  $("apiCount").textContent = state.apiCalls;
  const empty = $("apiLog").querySelector(".empty-state");
  if (empty) empty.remove();
  const item = document.createElement("article");
  item.className = `api-entry ${status >= 200 && status < 300 ? "ok" : "error"}`;
  item.innerHTML = `<header><strong>${method} ${path}</strong><span>HTTP ${status || "ERR"}</span></header><p>${new Date().toLocaleTimeString("zh-CN")}</p><pre></pre>`;
  item.querySelector("pre").textContent = JSON.stringify(payload, null, 2);
  $("apiLog").prepend(item);
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  const token = $("accessToken")?.value.trim() || localStorage.getItem("external-access-token");
  if (token) headers.Authorization = `Bearer ${token}`;
  if (options.body) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...options, headers, cache: "no-store" });
  let payload;
  try { payload = await response.json(); } catch { payload = { message: await response.text() }; }
  addApiCall(options.method || "GET", path, response.status, payload);
  if (!response.ok) {
    const error = new Error(payload.message || payload.error_code || `HTTP ${response.status}`);
    error.payload = payload;
    error.status = response.status;
    throw error;
  }
  return payload;
}

function updateTaskForm() {
  const info = taskInfo[state.task];
  document.querySelectorAll(".task-tab").forEach((button) => {
    const active = button.dataset.task === state.task;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active);
  });
  $("orderFields").hidden = state.task !== "1";
  $("requestEndpoint").textContent = info.endpoint;
  $("submitLabel").textContent = `发送${info.name}请求`;
  $("externalTaskId").value = nowId(state.task === "1" ? "ORD" : state.task === "0" ? "PREP" : "REPLENISH");
  $("idempotencyKey").value = nowId("idem");
  setMessage("");
}

function fillCatalog() {
  [$("productOne"), $("productTwo")].forEach((select, index) => {
    select.innerHTML = state.catalog.map((product, productIndex) => `<option value="${productIndex}">${product.name} (${product.sku_id})</option>`).join("");
    select.value = String(index);
  });
  $("catalogSize").textContent = `商品池 ${state.catalog.length} 个 SKU`;
  updateOrderSummary();
}

function updateOrderSummary() {
  const one = state.catalog[Number($("productOne").value)];
  let two = state.catalog[Number($("productTwo").value)];
  if (one?.sku_id === two?.sku_id) {
    $("productTwo").value = String((Number($("productTwo").value) + 1) % state.catalog.length);
    two = state.catalog[Number($("productTwo").value)];
  }
  $("orderId").textContent = `订单号 ${$("externalTaskId").value || "-"}`;
  $("catalogSize").textContent = `${one?.name || "-"} · ${two?.name || "-"}`;
}

function statusClass(status) { return String(status || "unknown").toLowerCase(); }

function renderHealth(payload) {
  const status = payload.status || "ERROR";
  $("healthBadge").className = `badge ${statusClass(status)}`;
  $("healthBadge").textContent = status;
  $("healthMessage").textContent = status === "READY" ? "系统可以接收新的外部任务" : status === "BUSY" ? "当前有任务执行中，请等待完成" : "任务服务未就绪，请检查依赖";
  $("healthTasks").innerHTML = [0, 1, 2].map((number) => {
    const ready = payload[`ready_for_task${number}`];
    return `<div class="health-task ${ready ? "ready" : ""}"><div><span>Task ${number}</span><strong>${taskInfo[String(number)].name}</strong></div><em>${ready ? "READY" : "BLOCKED"}</em></div>`;
  }).join("");
  const dependencies = Array.isArray(payload.dependencies) ? payload.dependencies : [];
  $("dependencies").innerHTML = dependencies.length
    ? dependencies.map((dependency) => `<span class="dependency"><i class="${dependency.status === "READY" ? "" : "bad"}"></i>${dependency.name} · ${dependency.status}</span>`).join("")
    : '<span class="dependency muted-dependency">当前服务未返回依赖明细</span>';
  $("connectionDot").className = `status-dot ${status === "READY" || status === "BUSY" ? "ready" : "failed"}`;
  $("connectionText").textContent = status === "READY" ? "任务服务在线" : status === "BUSY" ? "任务服务忙碌" : "任务服务异常";
}

async function refreshHealth() {
  $("healthButton").disabled = true;
  try { renderHealth(await api("/api/external/v1/health")); }
  catch (error) { $("healthBadge").className = "badge failed"; $("healthBadge").textContent = "ERROR"; $("healthMessage").textContent = error.message; $("connectionDot").className = "status-dot failed"; $("connectionText").textContent = "无法连接任务服务"; }
  finally { $("healthButton").disabled = false; }
}

function renderStatus(payload, source = "状态查询") {
  if (!payload) return;
  const status = payload.status || "UNKNOWN";
  const step = payload.current_step || {};
  const notice = payload.user_notice || {};
  const progress = Math.max(0, Math.min(100, Number(step.progress_percent || 0)));
  $("runStatus").className = `badge ${statusClass(status)}`;
  $("runStatus").textContent = status;
  $("runId").textContent = payload.task_run_id || state.runId || "-";
  $("progressLabel").textContent = step.label || "任务进度";
  $("progressValue").textContent = `${progress}%`;
  $("progressBar").style.width = `${progress}%`;
  $("displayTitle").textContent = payload.display_title || "等待状态反馈";
  $("displayMessage").textContent = payload.display_message || "-";
  $("currentStep").innerHTML = `<span>当前步骤 · ${step.code || "UNKNOWN"}</span><strong>${step.label || "等待启动"}</strong>`;
  $("notice").className = `notice ${(notice.level || "INFO").toLowerCase()}`;
  $("notice").textContent = notice.message || "-";
  $("sequence").textContent = payload.sequence ?? "-";
  $("eventType").textContent = payload.event_type || "-";
  $("feedbackMode").textContent = source;
}

function appendEvent(payload) {
  if (!payload || state.events.some((event) => event.event_id === payload.event_id)) return;
  state.events.push(payload);
  const empty = $("timeline").querySelector(".empty-state");
  if (empty) empty.remove();
  const item = document.createElement("article");
  item.className = `timeline-item ${statusClass(payload.status)}`;
  const step = payload.current_step || {};
  item.innerHTML = `<header><strong>${payload.display_title || payload.event_type || "状态更新"}</strong><span>#${payload.sequence ?? "-"}</span></header><time>${payload.occurred_at ? new Date(payload.occurred_at).toLocaleTimeString("zh-CN") : "刚刚"} · ${payload.event_type || "STATUS"}</time><p>${payload.display_message || step.label || "-"}</p><pre></pre>`;
  item.querySelector("pre").textContent = JSON.stringify({ status: payload.status, current_step: payload.current_step, summary: payload.summary }, null, 2);
  $("timeline").prepend(item);
}

async function pollStatus() {
  if (!state.runId) return;
  try {
    const payload = await api(`/api/external/v1/tasks/${encodeURIComponent(state.runId)}/status`);
    renderStatus(payload);
    appendEvent(payload);
    if (["SUCCEEDED", "PARTIAL_SUCCESS", "FAILED"].includes(payload.status)) stopPolling();
  } catch (error) { setMessage(`状态查询失败：${error.message}`, true); }
}

async function pollCallbackEvents() {
  if (!state.runId) return;
  try {
    const result = await fetch(`/api/events?task_run_id=${encodeURIComponent(state.runId)}`, { cache: "no-store" }).then((response) => response.json());
    result.events.forEach((event) => { renderStatus(event, "状态回调"); appendEvent(event); });
  } catch { /* Status polling remains the fallback. */ }
}

function stopPolling() {
  clearInterval(state.pollTimer); clearInterval(state.eventTimer); state.pollTimer = null; state.eventTimer = null;
  $("submitButton").disabled = false;
}

function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(pollStatus, 1200);
  state.eventTimer = setInterval(pollCallbackEvents, 900);
  pollStatus(); pollCallbackEvents();
}

async function submitTask(event) {
  event.preventDefault();
  const info = taskInfo[state.task];
  const externalTaskId = $("externalTaskId").value.trim();
  const requestBody = { external_task_id: externalTaskId };
  if (state.task === "1") {
    requestBody.external_order_id = externalTaskId;
    requestBody.items = [Number($("productOne").value), Number($("productTwo").value)].map((index) => ({ sku_id: state.catalog[index].sku_id, quantity: 1 }));
  }
  if ($("callbackEnabled").checked) requestBody.status_callback_url = state.callbackUrl;
  $("submitButton").disabled = true; setMessage("正在发送受理请求…");
  try {
    const payload = await api(info.endpoint, { method: "POST", headers: { "Idempotency-Key": $("idempotencyKey").value.trim(), "X-Request-Id": nowId("request") }, body: JSON.stringify(requestBody) });
    state.runId = payload.task_run_id; state.events = [];
    $("timeline").innerHTML = ""; renderStatus({ ...payload, display_title: "已提交任务", display_message: "任务已受理，等待机器人开始执行", current_step: { label: "已受理", progress_percent: 0 }, user_notice: { level: "INFO", message: "任务进入执行队列" } });
    setMessage(payload.duplicate ? "幂等请求已返回原任务" : "任务已受理，正在接收状态反馈");
    appendEvent({ ...payload, event_id: `accepted-${payload.task_run_id}`, event_type: "TASK_ACCEPTED", display_title: "任务已受理", display_message: "外部系统收到 202 Accepted", current_step: { label: "已受理", progress_percent: 0 }, user_notice: { level: "INFO", message: "任务进入执行队列" }, sequence: 0, status: payload.status });
    startPolling();
  } catch (error) { setMessage(`请求失败：${error.message}`, true); $("submitButton").disabled = false; }
}

async function boot() {
  const config = await fetch("/api/config").then((response) => response.json());
  state.callbackUrl = config.callback_url;
  $("serviceUrl").textContent = config.robot_task_url;
  $("callbackUrl").textContent = `callback: ${config.callback_url}`;
  $("accessToken").value = localStorage.getItem("external-access-token") || "";
  $("accessToken").addEventListener("input", () => localStorage.setItem("external-access-token", $("accessToken").value.trim()));
  fillCatalog(); setDefaultIds(); await refreshHealth();
}

document.querySelectorAll(".task-tab").forEach((button) => button.addEventListener("click", () => { state.task = button.dataset.task; updateTaskForm(); }));
$("taskForm").addEventListener("submit", submitTask);
$("healthButton").addEventListener("click", refreshHealth);
$("clearTimelineButton").addEventListener("click", () => { state.events = []; $("timeline").innerHTML = '<div class="empty-state">时间线已清空。</div>'; });
$("randomOrderButton").addEventListener("click", () => { const second = Math.floor(Math.random() * state.catalog.length); $("productOne").value = String(Math.floor(Math.random() * state.catalog.length)); $("productTwo").value = String((second + (Number($("productOne").value) === second ? 1 : 0)) % state.catalog.length); updateOrderSummary(); });
$("productOne").addEventListener("change", updateOrderSummary); $("productTwo").addEventListener("change", updateOrderSummary); $("externalTaskId").addEventListener("input", updateOrderSummary);
boot().catch((error) => { $("connectionText").textContent = `演示网关异常：${error.message}`; $("connectionDot").className = "status-dot failed"; });
