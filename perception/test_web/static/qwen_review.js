const taskSelect = document.querySelector("#taskSelect");
const pairSelect = document.querySelector("#pairSelect");
const regionSelect = document.querySelector("#regionSelect");
const promptStageSelect = document.querySelector("#promptStageSelect");
const promptInput = document.querySelector("#promptInput");
const runButton = document.querySelector("#runButton");
const runFullButton = document.querySelector("#runFullButton");
const savePromptButton = document.querySelector("#savePromptButton");
const initialScanSelect = document.querySelector("#initialScanSelect");

let samples = [];
let initialScans = [];
let currentSample = null;
let currentRegion = null;
let currentPromptStage = null;

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let payload;
  try {
    payload = await response.json();
  } catch (_error) {
    payload = {};
  }
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

function renderInitialScan() {
  const sample = initialScans.find((item) => item.scan_name === initialScanSelect.value);
  if (!sample) return;
  document.querySelector("#initialScanPose").textContent = sample.pose_type;
  document.querySelector("#initialScanRailCount").textContent = String(sample.rail_count);
  document.querySelector("#initialScanRowCount").textContent = String(sample.row_count);
  document.querySelector("#initialScanImageSize").textContent =
    Array.isArray(sample.image_size) ? sample.image_size.join(" × ") : "—";
  document.querySelector("#initialScanSourceImage").src = sample.source_url;
  document.querySelector("#initialScanOverlayImage").src = sample.overlay_url;
  document.querySelector("#initialScanDetails").textContent = JSON.stringify({
    inspection_target_id: sample.inspection_target_id,
    pose_type: sample.pose_type,
    rails: sample.rails,
    rows: sample.rows,
  }, null, 2);
  status(
    "#initialScanStatus",
    `${sample.scan_name} · ${sample.rail_count} rails · ${sample.row_count} rows`,
    "success",
  );
}

async function initializeInitialScans() {
  try {
    const payload = await api("/api/qwen-review/initial-scans");
    initialScans = payload.samples || [];
    initialScanSelect.replaceChildren();
    initialScans.forEach((sample) => {
      initialScanSelect.append(option(
        sample.scan_name,
        `${sample.inspection_target_id} · ${sample.pose_type.replace("SHELF_VIEW_", "")}`,
      ));
    });
    if (!initialScans.length) throw new Error("没有找到 task0 初始扫描");
    renderInitialScan();
  } catch (error) {
    status("#initialScanStatus", error.message, "error");
    initialScanSelect.disabled = true;
  }
}

function selectedSamples() {
  return samples.filter((sample) => sample.task_type === taskSelect.value);
}

function promptStages(region) {
  if (Array.isArray(region?.prompt_stages) && region.prompt_stages.length) {
    return region.prompt_stages;
  }
  return region ? [{
    stage: "legacy",
    label: "当前区域 Prompt",
    prompt: region.prompt,
    prompt_source: region.prompt_source,
    prompt_warning: region.prompt_warning,
    result_stale: region.result_stale,
    input_image_url: region.expanded_image_url,
    candidate_images: region.candidate_images,
    candidate_sheets: region.candidate_sheets,
    candidate_count_before: region.candidate_count_before,
    candidate_count_after: region.candidate_count_after,
    last_result: region.last_result,
  }] : [];
}

function stageCandidates(sample, stage) {
  if (Array.isArray(stage?.candidate_images)) return stage.candidate_images;
  return Array.isArray(sample?.candidate_images) ? sample.candidate_images : [];
}

function stageCandidateSheets(stage) {
  return Array.isArray(stage?.candidate_sheets) ? stage.candidate_sheets : [];
}

function firstNumber(...values) {
  return values.find((value) => typeof value === "number" && Number.isFinite(value));
}

function formatBBox(bbox) {
  return Array.isArray(bbox) && bbox.length === 4 ? `[${bbox.join(", ")}]` : "—";
}

function formatOverlap(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const percent = value <= 1 ? value * 100 : value;
  return `${percent.toFixed(1)}%`;
}

function formatMilliseconds(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${Math.round(value)} ms`
    : "—";
}

function rowMappingLabel(constraint) {
  const skuRowIndex = constraint?.row_index;
  const detectedRowIndex = constraint?.detected_row_index;
  if (!Number.isInteger(skuRowIndex)) return "未分配行";
  if (Number.isInteger(detectedRowIndex) && detectedRowIndex !== skuRowIndex) {
    return `检测第 ${detectedRowIndex} 行 → SKU 第 ${skuRowIndex} 行`;
  }
  return `第 ${skuRowIndex} 行`;
}

function populatePairs() {
  pairSelect.replaceChildren();
  const taskSamples = selectedSamples();
  taskSamples.forEach((sample) => {
    pairSelect.append(option(
      `${sample.dataset}:${sample.pair_number}`,
      `pair_${sample.pair_number} · ${sample.location_id || "未知位置"} · ${sample.pose_type || "未知视角"}`,
    ));
  });
  const preferredSample = taskSamples.find((sample) =>
    (sample.regions || []).some((region) => Number.isInteger(region.row_constraint?.row_index))
  ) || taskSamples[0];
  if (preferredSample) {
    pairSelect.value = `${preferredSample.dataset}:${preferredSample.pair_number}`;
  }
  selectPair();
}

function selectPair() {
  const [dataset, pairNumber] = pairSelect.value.split(":");
  currentSample = samples.find(
    (sample) => sample.dataset === dataset && sample.pair_number === Number(pairNumber),
  ) || null;
  renderFullInspectResult(currentSample?.full_inspect_result || null);
  runFullButton.disabled = !currentSample;
  status(
    "#fullRunStatus",
    currentSample?.full_inspect_result ? "已加载本次完整链路结果" : "完整链路尚未运行",
    currentSample?.full_inspect_result ? "success" : "",
  );

  regionSelect.replaceChildren();
  (currentSample?.regions || []).forEach((region) => {
    const rowIndex = region.row_constraint?.row_index;
    const rowLabel = Number.isInteger(rowIndex)
      ? ` · ${rowMappingLabel(region.row_constraint)}`
      : " · 未分配行";
    regionSelect.append(option(
      String(region.region_index),
      `region_${String(region.region_index).padStart(2, "0")} · bbox ${formatBBox(region.bbox)}${rowLabel}`,
    ));
  });
  selectRegion();
}

function selectRegion() {
  currentRegion = currentSample?.regions.find(
    (region) => region.region_index === Number(regionSelect.value),
  ) || null;
  promptStageSelect.replaceChildren();
  promptStages(currentRegion).forEach((stage) => {
    promptStageSelect.append(option(stage.stage, stage.label || stage.stage));
  });
  promptStageSelect.disabled = promptStageSelect.options.length <= 1;
  selectPromptStage();
}

function selectPromptStage() {
  currentPromptStage = promptStages(currentRegion).find(
    (stage) => stage.stage === promptStageSelect.value,
  ) || promptStages(currentRegion)[0] || null;
  if (currentPromptStage && promptStageSelect.value !== currentPromptStage.stage) {
    promptStageSelect.value = currentPromptStage.stage;
  }
  promptInput.value = currentPromptStage?.prompt || "";
  document.querySelector("#promptSource").textContent = currentRegion && currentPromptStage
    ? `${currentPromptStage.prompt_source || "generated"} · region_${String(currentRegion.region_index).padStart(2, "0")} · ${currentPromptStage.stage}`
    : "无审核阶段";
  renderResult(
    currentPromptStage?.last_result || null,
    Boolean(currentPromptStage?.result_stale),
  );
  renderConstraint();
  renderImages();
  status("#saveStatus", currentPromptStage?.prompt_warning || "");
  status(
    "#runStatus",
    currentPromptStage ? "审核阶段已加载" : "当前 pair 没有异常区域",
    currentPromptStage ? "success" : "error",
  );
  runButton.disabled = !currentPromptStage;
  savePromptButton.disabled = !currentPromptStage;
}

function renderConstraint() {
  const constraint = currentRegion?.row_constraint || null;
  const candidates = stageCandidates(currentSample, currentPromptStage);
  const filter = currentRegion?.candidate_filter || {};
  const afterCount = firstNumber(
    currentPromptStage?.candidate_count_after,
    currentRegion?.candidate_count_after,
    filter.after_count,
    filter.filtered_count,
    candidates.length,
  );
  const beforeCount = firstNumber(
    currentPromptStage?.candidate_count_before,
    currentRegion?.candidate_count_before,
    filter.before_count,
    filter.total_count,
    currentSample?.candidate_images?.length,
    afterCount,
  );

  const badge = document.querySelector("#rowStatusBadge");
  const hasConstraint = Number.isInteger(constraint?.row_index);
  badge.textContent = hasConstraint ? rowMappingLabel(constraint) : "未使用行约束";
  badge.className = `badge ${hasConstraint ? "active" : "neutral"}`;
  document.querySelector("#rowIndexValue").textContent = hasConstraint
    ? rowMappingLabel(constraint)
    : "—";
  document.querySelector("#rowBBoxValue").textContent = formatBBox(constraint?.row_bbox);
  document.querySelector("#rowOverlapValue").textContent = formatOverlap(constraint?.overlap_ratio);
  document.querySelector("#candidateFilterValue").textContent = beforeCount === undefined
    ? "—"
    : `${beforeCount} → ${afterCount ?? 0} 个`;

  const hint = document.querySelector("#constraintHint");
  if (!currentRegion) {
    hint.textContent = "请选择一个异常区域。";
  } else if (!hasConstraint) {
    hint.textContent = "该异常框没有可靠地落入预期货架行，候选列表采用未约束回退。";
  } else if (currentSample?.task_type === "SHORTAGE") {
    hint.textContent = `SHORTAGE 只发送 SKU 接口第 ${constraint.row_index} 行的候选，降低跨行误识别。`;
  } else if (currentPromptStage?.stage === "misplaced_product") {
    hint.textContent = "MISPLACED 第一阶段识别当前放错商品，使用全量标准库视觉检索得到的 Top-K 候选。";
  } else {
    hint.textContent = `MISPLACED 第二阶段显示当前/Reference 整行对比图，只从 SKU 第 ${constraint.row_index} 行候选判断缺失商品。`;
  }
}

function renderResult(result, stale = false) {
  const parsed = document.querySelector("#parsedResult");
  const raw = document.querySelector("#rawOutput");
  const meta = document.querySelector("#resultMeta");
  if (!result) {
    parsed.textContent = stale ? "输入图片已更新，请重新运行" : "等待推理";
    raw.textContent = stale
      ? "旧结果仍保留在磁盘，但不再作为当前结果展示。"
      : "等待推理";
    meta.textContent = stale ? "旧结果已过期" : "尚未运行";
    document.querySelector("#endToEndTiming").textContent = "—";
    document.querySelector("#backendTiming").textContent = "—";
    document.querySelector("#qwenTiming").textContent = "—";
    return;
  }
  parsed.textContent = result.parsed_result === null
    ? `解析失败：${result.parse_error || "未知错误"}`
    : JSON.stringify(result.parsed_result, null, 2);
  raw.textContent = result.raw_output || "（空输出）";
  const createdAt = result.created_at ? new Date(result.created_at).toLocaleString() : "";
  meta.textContent = `temp ${result.temperature ?? "?"} · ${createdAt}`;
  document.querySelector("#endToEndTiming").textContent = formatMilliseconds(
    result.review_round_trip_ms,
  );
  document.querySelector("#backendTiming").textContent = formatMilliseconds(
    firstNumber(result.backend_elapsed_ms, result.timings?.backend_processing_ms),
  );
  document.querySelector("#qwenTiming").textContent = formatMilliseconds(
    firstNumber(result.qwen_elapsed_ms, result.timings?.qwen_request_ms, result.elapsed_ms),
  );
}

function normalizeFullInspectPayload(result) {
  const payload = result?.result;
  if (Array.isArray(payload)) return { findings: payload };
  if (payload && Array.isArray(payload.findings)) {
    return { findings: payload.findings };
  }
  return { findings: [] };
}

function renderFullInspectResult(result) {
  const output = document.querySelector("#fullInspectResult");
  const meta = document.querySelector("#fullChainMeta");
  if (!result) {
    document.querySelector("#fullEndToEndTiming").textContent = "—";
    document.querySelector("#fullBackendTiming").textContent = "—";
    document.querySelector("#fullInspectTiming").textContent = "—";
    document.querySelector("#fullFindingCount").textContent = "—";
    output.textContent = "等待完整巡检";
    meta.textContent = "尚未运行";
    return;
  }
  document.querySelector("#fullEndToEndTiming").textContent = formatMilliseconds(
    result.full_round_trip_ms,
  );
  document.querySelector("#fullBackendTiming").textContent = formatMilliseconds(
    firstNumber(result.backend_elapsed_ms, result.timings?.backend_processing_ms),
  );
  document.querySelector("#fullInspectTiming").textContent = formatMilliseconds(
    firstNumber(result.inspect_elapsed_ms, result.timings?.inspect_pipeline_ms),
  );
  const payload = normalizeFullInspectPayload(result);
  document.querySelector("#fullFindingCount").textContent = Number.isInteger(result.finding_count)
    ? String(result.finding_count)
    : String(payload.findings.length);
  output.textContent = JSON.stringify(payload, null, 2);
  meta.textContent = result.created_at
    ? new Date(result.created_at).toLocaleString()
    : "已完成";
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

function setSourceImage(imageId, figureId, url) {
  const image = document.querySelector(`#${imageId}`);
  const figure = document.querySelector(`#${figureId}`);
  if (url) {
    image.src = url;
    figure.classList.remove("unavailable");
  } else {
    image.removeAttribute("src");
    figure.classList.add("unavailable");
  }
}

function renderImages() {
  const container = document.querySelector("#modelInputs");
  container.replaceChildren();
  if (!currentSample || !currentRegion) {
    document.querySelector("#inputCount").textContent = "0 张";
    setSourceImage("baselineImage", "baselineFigure", currentSample?.baseline_url);
    setSourceImage("currentImage", "currentFigure", currentSample?.current_url);
    setSourceImage(
      "alignedCurrentImage",
      "alignedCurrentFigure",
      currentSample?.aligned_current_url || currentSample?.aligned_url,
    );
    setSourceImage(
      "rowOverlayImage",
      "rowOverlayFigure",
      currentSample?.row_overlay_url || currentSample?.rows_overlay_url || currentSample?.row_detection_overlay_url,
    );
    return;
  }

  const isExpectedStage = currentPromptStage?.stage === "expected_product";
  const isShortageStage = currentSample.task_type === "SHORTAGE";
  container.append(imageFigure(
    currentPromptStage?.input_image_url || currentRegion.expanded_image_url,
    isExpectedStage
      ? "IMAGE 1 · 当前/Reference 整行对比图（红框为同一异常位置）"
      : isShortageStage
        ? "IMAGE 1 · 缺货前 reference 局部图"
        : "IMAGE 1 · 当前异常商品局部图",
    isShortageStage
      ? `reference 图对应 bbox ${formatBBox(currentRegion.bbox)}`
      : `异常 bbox ${formatBBox(currentRegion.bbox)}`,
    true,
  ));

  const candidates = stageCandidates(currentSample, currentPromptStage);
  const candidateSheets = stageCandidateSheets(currentPromptStage);
  if (candidateSheets.length) {
    candidateSheets.forEach((sheet, index) => {
      const first = sheet.first_candidate_number ?? "?";
      const last = sheet.last_candidate_number ?? "?";
      const promptNumber = sheet.prompt_image_number ?? index + 2;
      const figure = imageFigure(
        sheet.url,
        `IMAGE ${promptNumber} · SKU 标准图拼图`,
        `候选编号 ${first}–${last} · ${sheet.candidate_count ?? "?"} 个 SKU`,
      );
      figure.classList.add("candidate-sheet-input");
      container.append(figure);
    });
  } else {
    candidates.forEach((candidate, index) => {
      const promptNumber = candidate.prompt_image_number ?? index + 2;
      const visibleRows = Array.isArray(candidate.visible_row_numbers)
        ? candidate.visible_row_numbers.join(", ") || "未知"
        : "未知";
      container.append(imageFigure(
        candidate.url,
        `IMAGE ${promptNumber} · ${candidate.name || "未命名 SKU"}`,
        `${candidate.sku_id || "无 SKU ID"} · 可见行 ${visibleRows}`,
      ));
    });
  }
  const referenceImageCount = candidateSheets.length || candidates.length;
  document.querySelector("#inputCount").textContent = `${referenceImageCount + 1} 张`;
  document.querySelector("#inputScopeNote").textContent = isExpectedStage
    ? "第二阶段：IMAGE 1 上半部分是当前异常行，下半部分是摆放正确时的 Reference 行；后续发送按标准货位从左到右排列的目标行 SKU 拼图。"
    : currentSample.task_type === "MISPLACED"
      ? "第一阶段：IMAGE 1 是当前放错商品局部图；后续发送全量标准库视觉检索 Top-K 的带编号 SKU 拼图，不使用目标行限制。"
      : "IMAGE 1 从缺货前 reference（每组 _1 图）按异常 bbox 裁出，框内保留原商品；后续只发送目标 SKU 行候选。";

  setSourceImage("baselineImage", "baselineFigure", currentSample.baseline_url);
  setSourceImage("currentImage", "currentFigure", currentSample.current_url);
  setSourceImage(
    "alignedCurrentImage",
    "alignedCurrentFigure",
    currentSample.aligned_current_url || currentSample.aligned_url,
  );
  setSourceImage(
    "rowOverlayImage",
    "rowOverlayFigure",
    currentSample.row_overlay_url || currentSample.rows_overlay_url || currentSample.row_detection_overlay_url,
  );
}

async function runInfer() {
  if (!currentSample || !currentRegion || !currentPromptStage) return;
  const stage = currentPromptStage;
  const prompt = promptInput.value;
  runButton.disabled = true;
  promptStageSelect.disabled = true;
  status("#runStatus", "Qwen 推理中…");
  const started = performance.now();
  try {
    const result = await api("/api/qwen-review/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: currentSample.dataset,
        pair_number: currentSample.pair_number,
        region_index: currentRegion.region_index,
        stage: stage.stage,
        prompt,
        temperature: Number(document.querySelector("#temperature").value),
      }),
    });
    result.review_round_trip_ms = Math.round((performance.now() - started) * 10) / 10;
    stage.last_result = result;
    stage.result_stale = false;
    if (currentPromptStage === stage) renderResult(result, false);
    status(
      "#runStatus",
      `完成，本页审核端到端 ${Math.round(result.review_round_trip_ms)} ms`,
      "success",
    );
  } catch (error) {
    status(
      "#runStatus",
      `${error.message} · ${Math.round(performance.now() - started)} ms`,
      "error",
    );
  } finally {
    runButton.disabled = false;
    promptStageSelect.disabled = promptStageSelect.options.length <= 1;
  }
}

async function runFullInspect() {
  if (!currentSample) return;
  runFullButton.disabled = true;
  status("#fullRunStatus", "完整巡检链路运行中…");
  const started = performance.now();
  try {
    const result = await api("/api/qwen-review/run-full", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: currentSample.dataset,
        pair_number: currentSample.pair_number,
      }),
    });
    result.full_round_trip_ms = Math.round((performance.now() - started) * 10) / 10;
    currentSample.full_inspect_result = result;
    renderFullInspectResult(result);
    status(
      "#fullRunStatus",
      `完成，完整链路端到端 ${Math.round(result.full_round_trip_ms)} ms`,
      "success",
    );
  } catch (error) {
    status(
      "#fullRunStatus",
      `${error.message} · ${Math.round(performance.now() - started)} ms`,
      "error",
    );
  } finally {
    runFullButton.disabled = false;
  }
}

async function savePrompt() {
  if (!currentSample || !currentRegion || !currentPromptStage) return;
  const stage = currentPromptStage;
  const prompt = promptInput.value;
  savePromptButton.disabled = true;
  promptStageSelect.disabled = true;
  status("#saveStatus", "保存中…");
  try {
    await api("/api/qwen-review/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: currentSample.dataset,
        pair_number: currentSample.pair_number,
        region_index: currentRegion.region_index,
        stage: stage.stage,
        prompt,
      }),
    });
    stage.prompt = prompt;
    stage.prompt_source = "override";
    stage.prompt_warning = null;
    document.querySelector("#promptSource").textContent = `override · region_${String(currentRegion.region_index).padStart(2, "0")} · ${stage.stage}`;
    status("#saveStatus", "已保存当前审核阶段的 Prompt override", "success");
  } catch (error) {
    status("#saveStatus", error.message, "error");
  } finally {
    savePromptButton.disabled = false;
    promptStageSelect.disabled = promptStageSelect.options.length <= 1;
  }
}

async function initialize() {
  try {
    const payload = await api("/api/qwen-review/samples");
    samples = payload.samples || [];
    if (!samples.length) {
      throw new Error("没有找到 qwen_prompt_samples，请先运行样例生成脚本。");
    }
    populatePairs();
  } catch (error) {
    status("#runStatus", error.message, "error");
    runButton.disabled = true;
    runFullButton.disabled = true;
    savePromptButton.disabled = true;
  }
}

taskSelect.addEventListener("change", populatePairs);
initialScanSelect.addEventListener("change", renderInitialScan);
pairSelect.addEventListener("change", selectPair);
regionSelect.addEventListener("change", selectRegion);
promptStageSelect.addEventListener("change", selectPromptStage);
runButton.addEventListener("click", runInfer);
runFullButton.addEventListener("click", runFullInspect);
savePromptButton.addEventListener("click", savePrompt);
initialize();
initializeInitialScans();
