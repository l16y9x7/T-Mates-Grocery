const initialScanSelect = document.querySelector("#initialScanSelect");
const shortageBatchGroupSelect = document.querySelector("#shortageBatchGroupSelect");
const shortageBatchRecordSelect = document.querySelector("#shortageBatchRecordSelect");

let initialScans = [];
let shortageBatchSamples = [];

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

function formatBBox(bbox) {
  return Array.isArray(bbox) ? `[${bbox.join(", ")}]` : "—";
}

function formatMilliseconds(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${Math.round(value)} ms`
    : "—";
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

function selectedShortageBatchSample() {
  return shortageBatchSamples.find((sample) =>
    sample.group === shortageBatchGroupSelect.value
    && sample.record === shortageBatchRecordSelect.value
  ) || null;
}

function renderShortageBatchFindings(sample) {
  const container = document.querySelector("#shortageBatchFindingGrid");
  container.replaceChildren();
  const findings = Array.isArray(sample?.findings) ? sample.findings : [];
  if (!findings.length) {
    const empty = document.createElement("div");
    empty.className = "shortage-empty";
    empty.textContent = sample?.status === "no_anomaly"
      ? "该 record 未检测到缺货区域"
      : "该 record 暂无可展示 finding";
    container.append(empty);
    return;
  }
  findings.forEach((finding) => {
    const card = document.createElement("article");
    card.className = "shortage-finding";
    const header = document.createElement("header");
    const title = document.createElement("h3");
    title.textContent = `REGION ${finding.region_index} · ${finding.product_name || "未识别商品"}`;
    const details = document.createElement("p");
    const confidence = typeof finding.confidence === "number"
      ? ` · confidence ${finding.confidence.toFixed(3)}`
      : "";
    details.textContent = `bbox ${formatBBox(finding.bbox)} · mask ${finding.mask_pixels ?? 0} px${confidence}`;
    header.append(title, details);
    card.append(header);
    if (finding.mask_url) {
      const image = document.createElement("img");
      image.src = finding.mask_url;
      image.alt = `Region ${finding.region_index} shortage mask`;
      image.loading = "lazy";
      card.append(image);
    }
    container.append(card);
  });
}

function renderReferenceMaskResult(container, result) {
  container.replaceChildren();
  if (!result) {
    const empty = document.createElement("div");
    empty.className = "shortage-empty";
    empty.textContent = "尚未生成 reference mask";
    container.append(empty);
    return;
  }

  const gallery = document.createElement("div");
  gallery.className = "reference-mask-gallery";
  [
    {
      title: "Task0 原图 + Reference Mask",
      description: "绿色为最终商品 mask；黄色为缺货 bbox；青色为 SAM crop；红色为选中实例。",
      url: result.reference_overlay_data_url,
    },
    {
      title: "Reference Mask（二值图）",
      description: `${result.mask_pixels ?? 0} pixels · 与 Task0 RGB / depth 完全同尺寸`,
      url: result.reference_mask_data_url,
    },
  ].forEach((item) => {
    const figure = document.createElement("figure");
    const caption = document.createElement("figcaption");
    const title = document.createElement("strong");
    title.textContent = item.title;
    const description = document.createElement("small");
    description.textContent = item.description;
    caption.append(title, description);
    const image = document.createElement("img");
    image.src = item.url;
    image.alt = item.title;
    figure.append(caption, image);
    gallery.append(figure);
  });

  const details = document.createElement("pre");
  details.className = "reference-mask-details";
  details.textContent = JSON.stringify({
    product_name: result.product_name,
    reference_image_size: result.reference_image_size,
    reference_bbox_xywh: result.reference_bbox,
    row_bbox_xywh: result.row_bbox,
    sam3_prompt: result.sam3_prompt,
    crop_box_xyxy: result.crop_box,
    selected_bbox_xyxy: result.selected_bbox,
    selected_score: result.selected_score,
    candidate_count: result.candidate_count,
    mask_pixels: result.mask_pixels,
    elapsed_ms: result.elapsed_ms,
  }, null, 2);
  container.append(gallery, details);
}

async function generateReferenceMask(sample, finding, controls) {
  controls.button.disabled = true;
  controls.status.textContent = "正在读取 Task0 原图并运行 SAM3…";
  controls.status.className = "reference-mask-status";
  try {
    const result = await api("/api/qwen-review/shortage-batch/reference-mask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        group: sample.group,
        record: sample.record,
        region_index: finding.region_index,
      }),
    });
    finding.reference_mask_result = result;
    renderReferenceMaskResult(controls.result, result);
    controls.status.textContent = `生成完成 · ${result.mask_pixels} pixels · ${formatMilliseconds(result.elapsed_ms)}`;
    controls.status.className = "reference-mask-status success";
  } catch (error) {
    controls.status.textContent = error.message;
    controls.status.className = "reference-mask-status error";
  } finally {
    controls.button.disabled = false;
  }
}

function renderShortageReferenceMasks(sample) {
  const container = document.querySelector("#shortageReferenceMaskList");
  container.replaceChildren();
  const findings = Array.isArray(sample?.findings) ? sample.findings : [];
  if (!findings.length) {
    const empty = document.createElement("div");
    empty.className = "shortage-empty";
    empty.textContent = "该 record 没有可用于生成 reference mask 的缺货项";
    container.append(empty);
    return;
  }
  findings.forEach((finding) => {
    const card = document.createElement("article");
    card.className = "reference-mask-card";
    const header = document.createElement("header");
    const title = document.createElement("h4");
    title.textContent = `REGION ${finding.region_index} · ${finding.product_name || "未识别商品"}`;
    const subtitle = document.createElement("p");
    subtitle.textContent = `reference bbox ${formatBBox(finding.bbox)}`;
    header.append(title, subtitle);

    const actions = document.createElement("div");
    actions.className = "reference-mask-actions";
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "生成 Reference Mask";
    button.disabled = !finding.product_name;
    const maskStatus = document.createElement("span");
    maskStatus.className = "reference-mask-status";
    maskStatus.textContent = finding.product_name
      ? "使用上方该项已有的完整输入生成，不重新执行 shortage 对比。"
      : "该项尚未识别商品名称，无法选择 SAM3 Prompt。";
    actions.append(button, maskStatus);

    const result = document.createElement("div");
    result.className = "reference-mask-result";
    renderReferenceMaskResult(result, finding.reference_mask_result);
    const controls = {button, status: maskStatus, result};
    button.addEventListener("click", () => generateReferenceMask(sample, finding, controls));
    card.append(header, actions, result);
    container.append(card);
  });
}

function renderQwenImages(finding) {
  const gallery = document.createElement("div");
  gallery.className = "shortage-qwen-image-grid";
  const images = Array.isArray(finding.qwen_images) ? finding.qwen_images : [];
  if (!images.length) {
    const empty = document.createElement("div");
    empty.className = "shortage-empty";
    empty.textContent = "没有找到本次 Qwen 请求对应的调试图片";
    gallery.append(empty);
    return gallery;
  }
  images.forEach((input) => {
    const figure = document.createElement("figure");
    if (input.image_index === 1) figure.classList.add("primary-qwen-input");
    const caption = document.createElement("figcaption");
    const title = document.createElement("strong");
    title.textContent = `IMAGE ${input.image_index} · ${input.label || "未命名输入"}`;
    const description = document.createElement("small");
    description.textContent = input.description || input.kind || "";
    caption.append(title, description);
    const image = document.createElement("img");
    image.src = input.url;
    image.alt = title.textContent;
    image.loading = "lazy";
    figure.append(caption, image);
    gallery.append(figure);
  });
  return gallery;
}

function originalQwenResult(finding) {
  if (!finding.qwen_original_raw_output) return null;
  return {
    result_source: "批测原始返回",
    parsed_result: finding.qwen_original_parsed_result,
    raw_output: finding.qwen_original_raw_output,
    parse_error: null,
    temperature: null,
    qwen_elapsed_ms: null,
    backend_elapsed_ms: null,
    created_at: null,
  };
}

function renderQwenRetryResult(container, result) {
  container.replaceChildren();
  if (!result) {
    container.classList.add("empty");
    container.textContent = "暂无模型返回";
    return;
  }
  container.classList.remove("empty");
  const heading = document.createElement("div");
  heading.className = "shortage-retry-result-heading";
  const title = document.createElement("h5");
  title.textContent = result.result_source || "本次重试返回";
  const meta = document.createElement("span");
  const timing = typeof result.qwen_elapsed_ms === "number"
    ? `Qwen ${formatMilliseconds(result.qwen_elapsed_ms)}`
    : "历史批测结果";
  const temperature = typeof result.temperature === "number"
    ? ` · temperature ${result.temperature}`
    : "";
  meta.textContent = `${timing}${temperature}`;
  heading.append(title, meta);

  const parsedTitle = document.createElement("h6");
  parsedTitle.textContent = "JSON 解析结果";
  const parsed = document.createElement("pre");
  parsed.className = "shortage-retry-parsed";
  parsed.textContent = result.parsed_result === null || result.parsed_result === undefined
    ? `解析失败：${result.parse_error || "没有可用结果"}`
    : JSON.stringify(result.parsed_result, null, 2);

  const rawTitle = document.createElement("h6");
  rawTitle.textContent = "Qwen 模型原始返回";
  const raw = document.createElement("pre");
  raw.className = "shortage-retry-raw";
  raw.textContent = result.raw_output || "（空输出）";
  container.append(heading, parsedTitle, parsed, rawTitle, raw);
}

async function retryShortageQwen(sample, finding, controls) {
  const prompt = controls.prompt.value.trim();
  if (!prompt) {
    controls.status.textContent = "Prompt 不能为空";
    controls.status.className = "shortage-retry-status error";
    return;
  }
  controls.button.disabled = true;
  controls.status.textContent = "正在请求 Qwen…";
  controls.status.className = "shortage-retry-status";
  const started = performance.now();
  try {
    const result = await api("/api/qwen-review/shortage-batch/retry", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        group: sample.group,
        record: sample.record,
        region_index: finding.region_index,
        prompt,
        temperature: Number(controls.temperature.value),
      }),
    });
    result.result_source = "本次重试返回";
    result.browser_round_trip_ms = Math.round((performance.now() - started) * 10) / 10;
    finding.last_qwen_retry = result;
    renderQwenRetryResult(controls.result, result);
    controls.status.textContent = `完成 · 页面往返 ${formatMilliseconds(result.browser_round_trip_ms)}`;
    controls.status.className = "shortage-retry-status success";
  } catch (error) {
    controls.status.textContent = error.message;
    controls.status.className = "shortage-retry-status error";
  } finally {
    controls.button.disabled = false;
  }
}

function renderShortageBatchPrompts(sample) {
  const container = document.querySelector("#shortageBatchPromptList");
  container.replaceChildren();
  const findings = Array.isArray(sample?.findings) ? sample.findings : [];
  if (!findings.length) {
    const empty = document.createElement("div");
    empty.className = "shortage-empty";
    empty.textContent = "该 record 没有调用 Qwen，无可展示输入";
    container.append(empty);
    return;
  }
  findings.forEach((finding) => {
    const card = document.createElement("article");
    card.className = "shortage-prompt";
    const title = document.createElement("h4");
    title.textContent = `REGION ${finding.region_index} · ${finding.product_name || "未识别商品"}`;
    const promptTitle = document.createElement("h5");
    promptTitle.textContent = "可编辑 SYSTEM / USER Prompt";
    const prompt = document.createElement("textarea");
    prompt.className = "shortage-prompt-editor";
    prompt.spellcheck = false;
    prompt.value = finding.last_qwen_retry?.prompt_used
      || finding.qwen_prompt
      || "";

    const actions = document.createElement("div");
    actions.className = "shortage-retry-actions";
    const temperatureLabel = document.createElement("label");
    const temperatureText = document.createElement("span");
    temperatureText.textContent = "temperature";
    const temperature = document.createElement("input");
    temperature.type = "number";
    temperature.min = "0";
    temperature.max = "2";
    temperature.step = "0.1";
    temperature.value = String(finding.last_qwen_retry?.temperature ?? 0);
    temperatureLabel.append(temperatureText, temperature);
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "使用修改后的 Prompt 重试 Qwen";
    const retryStatus = document.createElement("span");
    retryStatus.className = "shortage-retry-status";
    retryStatus.textContent = "修改 Prompt 后点击重试；bbox 和图片输入保持不变。";
    actions.append(temperatureLabel, button, retryStatus);

    const retryResult = document.createElement("section");
    retryResult.className = "shortage-retry-result";
    const displayedResult = finding.last_qwen_retry
      ? {...finding.last_qwen_retry, result_source: "最近一次重试返回"}
      : originalQwenResult(finding);
    renderQwenRetryResult(retryResult, displayedResult);
    const controls = {
      prompt,
      temperature,
      button,
      status: retryStatus,
      result: retryResult,
    };
    button.addEventListener("click", () => retryShortageQwen(sample, finding, controls));
    card.append(
      title,
      renderQwenImages(finding),
      promptTitle,
      prompt,
      actions,
      retryResult,
    );
    container.append(card);
  });
}

function renderShortageBatchRecord() {
  const sample = selectedShortageBatchSample();
  if (!sample) return;
  const findings = Array.isArray(sample.findings) ? sample.findings : [];
  const productNames = findings.map((finding) => finding.product_name).filter(Boolean);
  document.querySelector("#shortageBatchRecordStatus").textContent = sample.status || "—";
  document.querySelector("#shortageBatchFindingCount").textContent = String(findings.length);
  document.querySelector("#shortageBatchProducts").textContent = productNames.join("、") || "未识别";
  document.querySelector("#shortageBatchElapsed").textContent = formatMilliseconds(sample.elapsed_ms);
  setSourceImage("shortageBatchBaselineImage", "shortageBatchBaselineFigure", sample.baseline_rgb_url);
  setSourceImage("shortageBatchSourceImage", "shortageBatchSourceFigure", sample.source_rgb_url);
  setSourceImage("shortageBatchOverlayImage", "shortageBatchOverlayFigure", sample.overlay_url);
  setSourceImage("shortageBatchMaskImage", "shortageBatchMaskFigure", sample.combined_mask_url);
  setSourceImage("shortageBatchRowImage", "shortageBatchRowFigure", sample.row_detection_url);
  const rowDetection = sample.row_detection || {};
  const rails = Array.isArray(rowDetection.rails) ? rowDetection.rails.length : 0;
  const rows = Array.isArray(rowDetection.rows) ? rowDetection.rows.length : 0;
  document.querySelector("#shortageBatchRowSummary").textContent = sample.row_detection_url
    ? `${rails} rails · ${rows} rows · 当前对齐图`
    : (sample.row_detection_error || "暂无 row_detection 结果");
  if (initialScans.some((scan) => scan.scan_name === sample.group)) {
    initialScanSelect.value = sample.group;
    renderInitialScan();
  }
  renderShortageBatchFindings(sample);
  renderShortageReferenceMasks(sample);
  renderShortageBatchPrompts(sample);
  const errorMessage = sample.recognition_error?.message || sample.error;
  const statusKind = ["success", "partial", "no_anomaly"].includes(sample.status)
    ? "success"
    : ["error", "recognition_error"].includes(sample.status) ? "error" : "";
  status(
    "#shortageBatchStatus",
    `${sample.group}/${sample.record} · ${sample.status}${errorMessage ? ` · ${errorMessage}` : ""}`,
    statusKind,
  );
}

function populateShortageBatchRecords() {
  shortageBatchRecordSelect.replaceChildren();
  const groupSamples = shortageBatchSamples.filter(
    (sample) => sample.group === shortageBatchGroupSelect.value,
  );
  groupSamples.forEach((sample) => {
    const names = (sample.findings || [])
      .map((finding) => finding.product_name)
      .filter(Boolean)
      .join("、");
    shortageBatchRecordSelect.append(option(
      sample.record,
      `${sample.record} · ${sample.status}${names ? ` · ${names}` : ""}`,
    ));
  });
  renderShortageBatchRecord();
}

function populateShortageBatchGroups() {
  shortageBatchGroupSelect.replaceChildren();
  [...new Set(shortageBatchSamples.map((sample) => sample.group))].forEach((group) => {
    const count = shortageBatchSamples.filter((sample) => sample.group === group).length;
    shortageBatchGroupSelect.append(option(group, `${group} · ${count} records`));
  });
  populateShortageBatchRecords();
}

async function initializeShortageBatch() {
  try {
    const payload = await api("/api/qwen-review/shortage-batch");
    shortageBatchSamples = payload.samples || [];
    if (!shortageBatchSamples.length) {
      status("#shortageBatchStatus", "尚无批测结果，请先运行 inspect/batch_shortage.py");
      shortageBatchGroupSelect.disabled = true;
      shortageBatchRecordSelect.disabled = true;
      return;
    }
    populateShortageBatchGroups();
  } catch (error) {
    status("#shortageBatchStatus", error.message, "error");
    shortageBatchGroupSelect.disabled = true;
    shortageBatchRecordSelect.disabled = true;
  }
}

initialScanSelect.addEventListener("change", renderInitialScan);
shortageBatchGroupSelect.addEventListener("change", populateShortageBatchRecords);
shortageBatchRecordSelect.addEventListener("change", renderShortageBatchRecord);
initializeInitialScans();
initializeShortageBatch();
