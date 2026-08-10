const taskSelect = document.querySelector("#taskSelect");
const pairSelect = document.querySelector("#pairSelect");
const regionSelect = document.querySelector("#regionSelect");
const promptInput = document.querySelector("#promptInput");
const runButton = document.querySelector("#runButton");
const savePromptButton = document.querySelector("#savePromptButton");

let samples = [];
let currentSample = null;
let currentRegion = null;

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    const detail = typeof payload.detail === "string"
      ? payload.detail
      : JSON.stringify(payload.detail || payload);
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return payload;
}

function status(selector, message, kind = "") {
  const element = document.querySelector(selector);
  element.textContent = message;
  element.className = `status ${kind}`.trim();
}

function option(value, label) {
  const element = document.createElement("option");
  element.value = value;
  element.textContent = label;
  return element;
}

function selectedSamples() {
  return samples.filter((sample) => sample.task_type === taskSelect.value);
}

function populatePairs() {
  pairSelect.replaceChildren();
  selectedSamples().forEach((sample) => {
    pairSelect.append(option(
      `${sample.dataset}:${sample.pair_number}`,
      `pair_${sample.pair_number} · ${sample.location_id} · ${sample.pose_type}`,
    ));
  });
  selectPair();
}

function selectPair() {
  const [dataset, pairNumber] = pairSelect.value.split(":");
  currentSample = samples.find(
    (sample) => sample.dataset === dataset && sample.pair_number === Number(pairNumber),
  ) || null;
  regionSelect.replaceChildren();
  (currentSample?.regions || []).forEach((region) => {
    regionSelect.append(option(
      String(region.region_index),
      `region_${String(region.region_index).padStart(2, "0")} · bbox [${region.bbox.join(", ")}]`,
    ));
  });
  selectRegion();
}

function selectRegion() {
  currentRegion = currentSample?.regions.find(
    (region) => region.region_index === Number(regionSelect.value),
  ) || null;
  promptInput.value = currentRegion?.prompt || "";
  document.querySelector("#promptSource").textContent = currentRegion
    ? `${currentRegion.prompt_source} · region_${String(currentRegion.region_index).padStart(2, "0")}`
    : "无 region";
  renderResult(currentRegion?.last_result || null, Boolean(currentRegion?.result_stale));
  renderImages();
  status("#saveStatus", "");
  status("#runStatus", currentRegion ? "样例已载入" : "当前 pair 没有 bbox", currentRegion ? "success" : "error");
  runButton.disabled = !currentRegion;
  savePromptButton.disabled = !currentRegion;
}

function renderResult(result, stale = false) {
  const parsed = document.querySelector("#parsedResult");
  const raw = document.querySelector("#rawOutput");
  const meta = document.querySelector("#resultMeta");
  if (!result) {
    parsed.textContent = stale ? "输入图片已更新，请重新运行" : "等待推理";
    raw.textContent = stale ? "旧结果已保留在磁盘，但不再作为当前结果展示。" : "等待推理";
    meta.textContent = stale ? "旧结果已过期" : "尚未运行";
    return;
  }
  parsed.textContent = result.parsed_result === null
    ? `解析失败：${result.parse_error || "未知错误"}`
    : JSON.stringify(result.parsed_result, null, 2);
  raw.textContent = result.raw_output || "（空输出）";
  const createdAt = result.created_at ? new Date(result.created_at).toLocaleString() : "";
  meta.textContent = `${result.elapsed_ms ?? "?"} ms · temp ${result.temperature} · ${createdAt}`;
}

function imageFigure(url, title, description, primary = false) {
  const figure = document.createElement("figure");
  if (primary) figure.classList.add("primary-input");
  const caption = document.createElement("figcaption");
  const titleElement = document.createElement("strong");
  titleElement.textContent = title;
  const detail = document.createElement("small");
  detail.textContent = description;
  caption.append(titleElement, detail);
  const image = document.createElement("img");
  image.src = url;
  image.alt = title;
  image.loading = primary ? "eager" : "lazy";
  figure.append(caption, image);
  return figure;
}

function renderImages() {
  const container = document.querySelector("#modelInputs");
  container.replaceChildren();
  if (!currentSample || !currentRegion) {
    document.querySelector("#inputCount").textContent = "0 张";
    return;
  }
  container.append(imageFigure(
    currentRegion.expanded_image_url,
    "IMAGE 1 · bbox 扩展图",
    `bbox [${currentRegion.bbox.join(", ")}]`,
    true,
  ));
  currentSample.candidate_images.forEach((candidate) => {
    container.append(imageFigure(
      candidate.url,
      `IMAGE ${candidate.prompt_image_number} · ${candidate.name}`,
      `${candidate.sku_id} · 可见行 ${candidate.visible_row_numbers.join(", ")}`,
    ));
  });
  document.querySelector("#inputCount").textContent = `${currentSample.candidate_images.length + 1} 张`;
  document.querySelector("#baselineImage").src = currentSample.baseline_url;
  document.querySelector("#currentImage").src = currentSample.current_url;
}

async function runInfer() {
  if (!currentSample || !currentRegion) return;
  runButton.disabled = true;
  status("#runStatus", "Qwen 推理中…", "");
  const started = performance.now();
  try {
    const result = await api("/api/qwen-infer/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: currentSample.dataset,
        pair_number: currentSample.pair_number,
        region_index: currentRegion.region_index,
        prompt: promptInput.value,
        temperature: Number(document.querySelector("#temperature").value),
      }),
    });
    currentRegion.last_result = result;
    currentRegion.result_stale = false;
    renderResult(result, false);
    status("#runStatus", `完成，页面耗时 ${Math.round(performance.now() - started)} ms`, "success");
  } catch (error) {
    status("#runStatus", error.message, "error");
  } finally {
    runButton.disabled = false;
  }
}

async function savePrompt() {
  if (!currentSample || !currentRegion) return;
  savePromptButton.disabled = true;
  status("#saveStatus", "保存中…");
  try {
    await api("/api/qwen-infer/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: currentSample.dataset,
        pair_number: currentSample.pair_number,
        region_index: currentRegion.region_index,
        prompt: promptInput.value,
      }),
    });
    currentRegion.prompt = promptInput.value;
    currentRegion.prompt_source = "override";
    document.querySelector("#promptSource").textContent = `override · region_${String(currentRegion.region_index).padStart(2, "0")}`;
    status("#saveStatus", "已保存为该样例的 Prompt override", "success");
  } catch (error) {
    status("#saveStatus", error.message, "error");
  } finally {
    savePromptButton.disabled = false;
  }
}

async function initialize() {
  try {
    const payload = await api("/api/qwen-infer/samples");
    samples = payload.samples || [];
    if (!samples.length) throw new Error("没有找到 qwen_prompt_samples，请先运行样例生成脚本");
    populatePairs();
  } catch (error) {
    status("#runStatus", error.message, "error");
    runButton.disabled = true;
    savePromptButton.disabled = true;
  }
}

taskSelect.addEventListener("change", populatePairs);
pairSelect.addEventListener("change", selectPair);
regionSelect.addEventListener("change", selectRegion);
runButton.addEventListener("click", runInfer);
savePromptButton.addEventListener("click", savePrompt);
initialize();
