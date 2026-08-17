const imageSelect = document.querySelector("#imageSelect");
const requestTaskType =
  document.querySelector("#requestTaskType") || document.querySelector("#requestName");
const requestHand = document.querySelector("#requestHand");
const requestLevel = document.querySelector("#requestLevel");
const qwenPrompt = document.querySelector("#qwenPrompt");
const qwenSku = document.querySelector("#qwenSku");
const samPrompt = document.querySelector("#samPrompt");
const qwenReferencePanel = document.querySelector("#qwenReferencePanel");
const qwenReferenceImage = document.querySelector("#qwenReferenceImage");
const qwenReferenceEmpty = document.querySelector("#qwenReferenceEmpty");
const qwenReferenceStatus = document.querySelector("#qwenReferenceStatus");
const rawQwenCanvas = document.querySelector("#rawQwenCanvas");
const qwenCanvas = document.querySelector("#qwenCanvas");
const rawSamCanvas = document.querySelector("#rawSamCanvas");
const samCanvas = document.querySelector("#samCanvas");
const finalLocateBboxCanvas = document.querySelector("#finalLocateBboxCanvas");
const finalLocateCropCanvas = document.querySelector("#finalLocateCropCanvas");
const cropDetectionSelect = document.querySelector("#cropDetectionSelect");
const locateImageInput = document.querySelector("#locateImageInput");
const locateDepthInput = document.querySelector("#locateDepthInput");
const locateDepthByteOrder = document.querySelector("#locateDepthByteOrder");
const clearLocateImage = document.querySelector("#clearLocateImage");
const batchResultFileSelect = document.querySelector("#batchResultFileSelect");
const batchRecordSelect = document.querySelector("#batchRecordSelect");
const batchProductSelect = document.querySelector("#batchProductSelect");
const batchPrevious = document.querySelector("#batchPrevious");
const batchNext = document.querySelector("#batchNext");
const batchRefresh = document.querySelector("#batchRefresh");

const colors = ["#2dd4bf", "#f59e0b", "#60a5fa", "#f472b6", "#a78bfa", "#fb7185"];
const qwenSampleColors = ["#a78bfa", "#2dd4bf", "#f59e0b"];
const cropPaddingRatio = 0.1;
let cropChoices = [];
let currentCrop = null;
const promptPairsBySku = new Map();
let loadedPromptSku = "";
let latestImageBase64 = "";
let latestImageMediaType = "image/jpeg";
let offlineImageName = "";
let offlineImageBase64 = "";
let offlineDepthName = "";
let offlineDepthBase64 = "";
let batchRecords = [];
let batchRerunPollTimer = null;

function resetQwenReferencePreview() {
  const shortage = requestTaskType?.value === "SHORTAGE";
  qwenReferencePanel.hidden = !shortage;
  qwenReferenceImage.hidden = true;
  qwenReferenceImage.removeAttribute("src");
  qwenReferenceEmpty.hidden = false;
  qwenReferenceEmpty.textContent = "运行后显示对应 product_name 的 SKU 图片";
  qwenReferenceStatus.textContent = shortage ? "等待运行" : "仅 SHORTAGE 使用";
}

function renderQwenReferencePreview(result) {
  if (requestTaskType?.value !== "SHORTAGE") {
    resetQwenReferencePreview();
    return;
  }
  qwenReferencePanel.hidden = false;
  qwenReferenceImage.hidden = true;
  qwenReferenceEmpty.hidden = false;
  qwenReferenceEmpty.textContent = "正在加载 SKU 样例图…";
  qwenReferenceStatus.textContent = result.qwen_reference_image_used
    ? `本次 Qwen 已附带 · ${result.qwen_reference_image_name || result.sku_id}`
    : "样例图预览（运行中的8083尚未返回双图启用标记）";
  qwenReferenceImage.onload = () => {
    qwenReferenceImage.hidden = false;
    qwenReferenceEmpty.hidden = true;
  };
  qwenReferenceImage.onerror = () => {
    qwenReferenceImage.hidden = true;
    qwenReferenceEmpty.hidden = false;
    qwenReferenceEmpty.textContent = "SKU 样例图读取失败";
  };
  qwenReferenceImage.src =
    `/api/qwen-review/shortage-batch/sku-image/${encodeURIComponent(result.sku_id)}`;
}

function selectedBatchResultFile() {
  return batchResultFileSelect.value || "";
}

function sortingBatchApiUrl(path) {
  const resultFile = selectedBatchResultFile();
  if (!resultFile) return path;
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}result_file=${encodeURIComponent(resultFile)}`;
}

function setBatchImage(imageSelector, emptySelector, url, emptyText) {
  const image = document.querySelector(imageSelector);
  const empty = document.querySelector(emptySelector);
  image.removeAttribute("src");
  empty.textContent = emptyText;
  empty.hidden = Boolean(url);
  image.hidden = !url;
  if (!url) return;
  image.onload = () => {
    image.hidden = false;
    empty.hidden = true;
  };
  image.onerror = () => {
    image.hidden = true;
    empty.textContent = "图片加载失败";
    empty.hidden = false;
  };
  image.src = url;
}

function currentBatchRecord() {
  return batchRecords[Number(batchRecordSelect.value) || 0] || null;
}

async function loadBatchResultFiles() {
  const previousResultFile = selectedBatchResultFile();
  const data = await api("/api/sorting-batch-results/files", { cache: "no-store" });
  const files = data.files || [];
  batchResultFileSelect.replaceChildren();
  files.forEach((file) => {
    const option = document.createElement("option");
    option.value = file.id;
    option.textContent =
      `${file.label} · ${file.completed || 0}/${file.total_detections || 0}` +
      `${file.failures ? ` · ${file.failures} 失败` : ""}`;
    batchResultFileSelect.append(option);
  });
  const preferred = files.some((file) => file.id === previousResultFile)
    ? previousResultFile
    : data.default;
  if (preferred) batchResultFileSelect.value = preferred;
  batchResultFileSelect.disabled = files.length === 0;
  if (!files.length) {
    throw new Error("没有找到 sorting_pick_locate_batch_results.json");
  }
  await loadBatchResults();
}

function renderBatchProduct() {
  const record = currentBatchRecord();
  const item = record?.items?.[Number(batchProductSelect.value) || 0] || null;
  if (!record || !item) return;
  setBatchImage("#batchRgbImage", "#batchRgbEmpty", record.rgb_url, "没有 RGB 图片");
  setBatchImage(
    "#batchDepthImage",
    "#batchDepthEmpty",
    record.depth_url,
    "没有深度数据",
  );
  setBatchImage(
    "#batchResultImage",
    "#batchResultEmpty",
    item.result_url,
    "没有检测结果图",
  );
  document.querySelector("#batchResultLabel").textContent = `${item.product_name}.jpg`;
  const status = document.querySelector("#batchResultStatus");
  status.className = `batch-result-status ${item.status === "success" ? "success" : "error"}`;
  status.textContent = item.status === "success" ? "检测成功" : "检测失败";
  document.querySelector("#batchResultDetail").textContent = JSON.stringify(
    {
      record: record.record,
      product_name: item.product_name,
      sku_id: item.sku_id,
      level: item.level,
      hand: item.hand,
      selected_depth_mm: item.selected_depth_mm,
      selected_bbox_pixel: item.selected_bbox_pixel,
      elapsed_seconds: item.elapsed_seconds,
      error: item.error,
      prompt_mapping_file: item.prompt_mapping_file,
      prompt_matches_current_mapping: item.prompt_matches_current_mapping,
      qwen3_prompt_used: item.qwen3_prompt_used,
      sam3_prompt_used: item.sam3_prompt_used,
    },
    null,
    2,
  );
}

function renderBatchRecord(preferredProductName = "") {
  const recordIndex = Number(batchRecordSelect.value) || 0;
  const record = batchRecords[recordIndex];
  batchProductSelect.replaceChildren();
  (record?.items || []).forEach((item, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${item.status === "success" ? "✓" : "✕"} ${item.product_name}`;
    option.selected = item.product_name === preferredProductName;
    batchProductSelect.append(option);
  });
  batchProductSelect.disabled = !record?.items?.length;
  batchPrevious.disabled = recordIndex <= 0;
  batchNext.disabled = recordIndex >= batchRecords.length - 1;
  renderBatchProduct();
}

async function loadBatchResults() {
  const selectedRecordName = currentBatchRecord()?.record;
  const selectedProductName =
    currentBatchRecord()?.items?.[Number(batchProductSelect.value) || 0]?.product_name || "";
  batchRefresh.disabled = true;
  document.querySelector("#batchSummary").textContent = "正在读取批测结果…";
  try {
    const data = await api(sortingBatchApiUrl("/api/sorting-batch-results"), {
      cache: "no-store",
    });
    batchRecords = data.records || [];
    document.querySelector("#batchDatasetLabel").textContent =
      `SORTING BATCH / ${data.dataset || selectedBatchResultFile()}`;
    batchRecordSelect.replaceChildren();
    batchRecords.forEach((record, index) => {
      const failures = record.items.filter((item) => item.status !== "success").length;
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `${record.record} · ${record.items.length} 项${failures ? ` · ${failures} 失败` : ""}`;
      option.selected = record.record === selectedRecordName;
      batchRecordSelect.append(option);
    });
    if (selectedRecordName && !batchRecords.some((record) => record.record === selectedRecordName)) {
      batchRecordSelect.value = "0";
    }
    batchRecordSelect.disabled = batchRecords.length === 0;
    const summary = data.summary || {};
    document.querySelector("#batchSummary").textContent =
      `${summary.total_records || 0} 个 record · ${summary.completed || 0}/${summary.total_detections || 0} 项完成 · ` +
      `${summary.successes || 0} 成功 · ${summary.failures || 0} 失败`;
    renderBatchRecord(selectedProductName);
  } catch (error) {
    batchRecords = [];
    batchRecordSelect.disabled = true;
    batchProductSelect.disabled = true;
    document.querySelector("#batchSummary").textContent = error.message;
    document.querySelector("#batchResultStatus").textContent = "批测结果加载失败";
  } finally {
    batchRefresh.disabled = false;
  }
}

function batchProgressText(status) {
  const progress = status.progress || {};
  const completed = Number(progress.completed || 0);
  const total = Number(progress.total_detections || 0);
  const counts = total ? `${completed}/${total}` : `${completed}`;
  const target = status.target;
  const label = target ? `${target.record} / ${target.product_name}` : "当前项";
  return `正在覆盖重跑 ${label} · ${counts}`;
}

function stopBatchRerunPolling() {
  if (batchRerunPollTimer !== null) {
    clearTimeout(batchRerunPollTimer);
    batchRerunPollTimer = null;
  }
}

async function pollBatchRerunStatus() {
  stopBatchRerunPolling();
  try {
    const status = await api(sortingBatchApiUrl("/api/sorting-batch-results/rerun"));
    if (status.running) {
      batchRefresh.disabled = true;
      batchRefresh.textContent = "批测运行中…";
      document.querySelector("#batchSummary").textContent = batchProgressText(status);
      batchRerunPollTimer = setTimeout(pollBatchRerunStatus, 2000);
      return;
    }

    batchRefresh.disabled = false;
    batchRefresh.textContent = "重跑当前项（--overwrite）";
    if (status.state === "succeeded") {
      await loadBatchResults();
    } else if (status.state === "failed") {
      const detail = status.last_log_line || `进程退出码 ${status.exit_code}`;
      document.querySelector("#batchSummary").textContent = `批测失败：${detail}`;
    }
  } catch (error) {
    batchRefresh.disabled = false;
    batchRefresh.textContent = "重跑当前项（--overwrite）";
    document.querySelector("#batchSummary").textContent = `读取批测进度失败：${error.message}`;
  }
}

async function rerunBatchResultsWithOverwrite() {
  const record = currentBatchRecord();
  const item = record?.items?.[Number(batchProductSelect.value) || 0];
  if (!record || !item) {
    document.querySelector("#batchSummary").textContent = "请先选择要重跑的 record 和商品";
    return;
  }
  stopBatchRerunPolling();
  batchRefresh.disabled = true;
  batchRefresh.textContent = "正在启动…";
  document.querySelector("#batchSummary").textContent =
    `正在覆盖重跑 ${record.record} / ${item.product_name}…`;
  try {
    const status = await api("/api/sorting-batch-results/rerun", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        record: record.record,
        product_name: item.product_name,
        result_file: selectedBatchResultFile(),
      }),
    });
    if (!status.running && status.state === "failed") {
      throw new Error(status.last_log_line || `进程退出码 ${status.exit_code}`);
    }
    await pollBatchRerunStatus();
  } catch (error) {
    batchRefresh.disabled = false;
    batchRefresh.textContent = "重跑当前项（--overwrite）";
    document.querySelector("#batchSummary").textContent = `启动批测失败：${error.message}`;
  }
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("离线文件读取失败"));
    reader.readAsDataURL(file);
  });
}

async function selectOfflineDepth() {
  const file = locateDepthInput.files?.[0];
  if (!file) return;
  if (!/\.(npy|raw|bin|png|tif|tiff)$/i.test(file.name)) {
    locateDepthInput.value = "";
    setStatus("#locateDepthStatus", "只支持 NPY、16 位 PNG/TIFF 或 RAW/BIN", "error");
    return;
  }
  try {
    offlineDepthName = file.name;
    offlineDepthBase64 = await readFileAsDataUrl(file);
    clearLocateImage.disabled = false;
    setStatus(
      "#locateDepthStatus",
      `深度数据：${file.name} · ${(file.size / 1024).toFixed(1)} KB`,
      offlineImageBase64 ? "success" : "running",
    );
  } catch (error) {
    offlineDepthName = "";
    offlineDepthBase64 = "";
    setStatus("#locateDepthStatus", error.message, "error");
  }
}

async function selectOfflineImage() {
  const file = locateImageInput.files?.[0];
  if (!file) return;
  if (!/^image\/(jpeg|png)$/.test(file.type)) {
    locateImageInput.value = "";
    setStatus("#locateImageStatus", "只支持 JPG/PNG", "error");
    return;
  }
  try {
    offlineImageName = file.name;
    offlineImageBase64 = await readFileAsDataUrl(file);
    clearLocateImage.disabled = false;
    setStatus(
      "#locateImageStatus",
      `离线图片：${file.name} · ${(file.size / 1024).toFixed(1)} KB`,
      "success",
    );
  } catch (error) {
    offlineImageName = "";
    offlineImageBase64 = "";
    setStatus("#locateImageStatus", error.message, "error");
  }
}

function clearOfflineImage() {
  offlineImageName = "";
  offlineImageBase64 = "";
  offlineDepthName = "";
  offlineDepthBase64 = "";
  locateImageInput.value = "";
  locateDepthInput.value = "";
  clearLocateImage.disabled = true;
  setStatus("#locateImageStatus", "当前使用腕部相机", "");
  setStatus("#locateDepthStatus", "未上传深度数据，将使用无深度回退", "");
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || `HTTP ${response.status}`);
  }
  return data;
}

async function loadImages() {
  const data = await api("/api/images");
  imageSelect.replaceChildren();
  data.images.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    option.selected = name === data.default;
    imageSelect.append(option);
  });
  if (data.default) {
    await drawBase(qwenCanvas);
    resetCropPanel();
  }
}

async function loadSkus() {
  const taskType = requestTaskType?.value || "SORTING";
  const params = new URLSearchParams({
    task_type: taskType,
    level: requestLevel.value,
    hand: requestHand.value,
  });
  const data = await api(`/api/skus?${params}`);
  const options = document.querySelector("#qwenSkuOptions");
  options.replaceChildren();
  promptPairsBySku.clear();
  loadedPromptSku = "";
  data.skus.forEach(({ name, qwen3_prompt: qwen3Prompt, sam3_prompt: sam3Prompt }) => {
    const option = document.createElement("option");
    option.value = name;
    options.append(option);
    if (typeof qwen3Prompt === "string" && typeof sam3Prompt === "string") {
      promptPairsBySku.set(name, {
        qwen3_prompt: qwen3Prompt,
        sam3_prompt: sam3Prompt,
      });
    }
  });
}

function loadPromptPairForSelectedSku() {
  const skuName = qwenSku.value.trim();
  const promptPair = promptPairsBySku.get(skuName);
  if (!promptPair) {
    loadedPromptSku = skuName;
    qwenPrompt.value = "";
    samPrompt.value = "";
    const message = skuName ? "该商品尚未配置 Prompt，可直接编辑" : "请输入目标商品名称";
    setStatus("#savePromptStatus", message, "");
    setStatus("#saveSamPromptStatus", message, "");
    return;
  }
  if (loadedPromptSku === skuName) {
    return;
  }
  loadedPromptSku = skuName;
  qwenPrompt.value = promptPair.qwen3_prompt;
  samPrompt.value = promptPair.sam3_prompt;
  setStatus("#savePromptStatus", "已加载配对 Prompt", "success");
  setStatus("#saveSamPromptStatus", "已加载配对 Prompt", "success");
}

function imageUrl() {
  return `/api/image/${encodeURIComponent(imageSelect.value)}`;
}

async function drawBase(canvas) {
  if (!latestImageBase64) {
    throw new Error("Debug 接口没有返回原图");
  }
  const image = new Image();
  image.src = `data:${latestImageMediaType};base64,${latestImageBase64}`;
  await image.decode();
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  canvas.getContext("2d").drawImage(image, 0, 0);
  return image;
}

function resetCropPanel(message = "先运行 Qwen3 获取 crop") {
  cropChoices = [];
  currentCrop = null;
  cropDetectionSelect.replaceChildren();
  const option = document.createElement("option");
  option.value = "";
  option.textContent = message;
  cropDetectionSelect.append(option);
  cropDetectionSelect.disabled = true;
  document.querySelector("#runSam").disabled = true;
  samCanvas.getContext("2d").clearRect(0, 0, samCanvas.width, samCanvas.height);
  const empty = document.querySelector("#samEmpty");
  empty.textContent = message;
  empty.hidden = false;
  document.querySelector("#samResult").textContent = "[]";
  setStatus("#samStatus", "等待 Qwen3", "");
}

function populateCropChoices(samples) {
  cropChoices = [];
  cropDetectionSelect.replaceChildren();
  samples.forEach((sample) => {
    (sample.detections || []).forEach((detection, detectionIndex) => {
      const choiceIndex = cropChoices.length;
      cropChoices.push({
        sampleIndex: sample.sample_index,
        detectionIndex,
        detection,
      });
      const option = document.createElement("option");
      option.value = String(choiceIndex);
      option.textContent = `S${sample.sample_index} · ${detection.name} #${detectionIndex + 1}`;
      cropDetectionSelect.append(option);
    });
  });

  const hasChoices = cropChoices.length > 0;
  cropDetectionSelect.disabled = !hasChoices;
  document.querySelector("#runSam").disabled = !hasChoices;
  if (!hasChoices) {
    resetCropPanel("Qwen3 没有返回可裁剪的 bbox");
  }
}

function bboxToPixels(bbox, image) {
  const normalized = document.querySelector("#qwenCoordinateMode").value === "normalized";
  const converted = [...bbox];
  if (normalized) {
    converted[0] = (converted[0] / 1000) * image.naturalWidth;
    converted[1] = (converted[1] / 1000) * image.naturalHeight;
    converted[2] = (converted[2] / 1000) * image.naturalWidth;
    converted[3] = (converted[3] / 1000) * image.naturalHeight;
  }
  const x1 = Math.min(converted[0], converted[2]);
  const y1 = Math.min(converted[1], converted[3]);
  const x2 = Math.max(converted[0], converted[2]);
  const y2 = Math.max(converted[1], converted[3]);
  const paddingX = (x2 - x1) * cropPaddingRatio;
  const paddingY = (y2 - y1) * cropPaddingRatio;
  return [
    Math.max(0, Math.floor(x1 - paddingX)),
    Math.max(0, Math.floor(y1 - paddingY)),
    Math.min(image.naturalWidth, Math.ceil(x2 + paddingX)),
    Math.min(image.naturalHeight, Math.ceil(y2 + paddingY)),
  ];
}

async function drawSelectedCrop() {
  const choice = cropChoices[Number(cropDetectionSelect.value)];
  if (!choice) {
    throw new Error("请先选择一个 Qwen 检测结果");
  }

  const image = new Image();
  image.src = `${imageUrl()}?t=${Date.now()}`;
  await image.decode();
  const cropBox = bboxToPixels(choice.detection.bbox, image);
  const [x1, y1, x2, y2] = cropBox;
  const width = x2 - x1;
  const height = y2 - y1;
  if (width < 2 || height < 2) {
    throw new Error("Qwen bbox 无法生成有效 crop");
  }

  samCanvas.width = width;
  samCanvas.height = height;
  samCanvas.getContext("2d").drawImage(
    image,
    x1,
    y1,
    width,
    height,
    0,
    0,
    width,
    height,
  );
  currentCrop = { ...choice, cropBox };
  document.querySelector("#samEmpty").hidden = true;
  document.querySelector("#samResult").textContent = "[]";
  setStatus("#samStatus", `crop ${width} × ${height}`, "success");
  return currentCrop;
}

function setStatus(id, text, kind = "") {
  const element = document.querySelector(id);
  element.textContent = text;
  element.className = `status ${kind}`.trim();
}

function drawBox(ctx, bbox, label, color, lineWidth = null) {
  const [x1, y1, x2, y2] = bbox;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth || Math.max(3, ctx.canvas.width / 420);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.font = `700 ${Math.max(16, ctx.canvas.width / 55)}px sans-serif`;
  const padding = 7;
  const width = ctx.measureText(label).width + padding * 2;
  const height = Math.max(26, ctx.canvas.width / 35);
  const labelY = Math.max(0, y1 - height);
  ctx.fillStyle = color;
  ctx.fillRect(x1, labelY, width, height);
  ctx.fillStyle = "#071017";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x1 + padding, labelY + height / 2);
  ctx.restore();
}

function boxIou(boxA, boxB) {
  const intersectionWidth = Math.max(0, Math.min(boxA[2], boxB[2]) - Math.max(boxA[0], boxB[0]));
  const intersectionHeight = Math.max(0, Math.min(boxA[3], boxB[3]) - Math.max(boxA[1], boxB[1]));
  const intersection = intersectionWidth * intersectionHeight;
  const areaA = Math.max(0, boxA[2] - boxA[0]) * Math.max(0, boxA[3] - boxA[1]);
  const areaB = Math.max(0, boxB[2] - boxB[0]) * Math.max(0, boxB[3] - boxB[1]);
  const union = areaA + areaB - intersection;
  return union > 0 ? intersection / union : 0;
}

function matchSamplePair(sampleA, sampleB) {
  const detectionsA = sampleA.detections || [];
  const detectionsB = sampleB.detections || [];
  const candidates = [];

  detectionsA.forEach((detectionA, indexA) => {
    detectionsB.forEach((detectionB, indexB) => {
      if (detectionA.name === detectionB.name) {
        candidates.push({
          indexA,
          indexB,
          iou: boxIou(detectionA.bbox, detectionB.bbox),
        });
      }
    });
  });

  candidates.sort((left, right) => right.iou - left.iou);
  const usedA = new Set();
  const usedB = new Set();
  const matches = [];
  candidates.forEach((candidate) => {
    if (usedA.has(candidate.indexA) || usedB.has(candidate.indexB)) {
      return;
    }
    usedA.add(candidate.indexA);
    usedB.add(candidate.indexB);
    const detectionA = detectionsA[candidate.indexA];
    const detectionB = detectionsB[candidate.indexB];
    matches.push({
      name: detectionA.name,
      target_a: candidate.indexA + 1,
      target_b: candidate.indexB + 1,
      bbox_a: detectionA.bbox,
      bbox_b: detectionB.bbox,
      iou: Number(candidate.iou.toFixed(4)),
    });
  });

  const mean = matches.length
    ? matches.reduce((sum, match) => sum + match.iou, 0) / matches.length
    : null;
  return {
    samples: `${sampleA.sample_index} ↔ ${sampleB.sample_index}`,
    mean_iou: mean === null ? null : Number(mean.toFixed(4)),
    matches,
    unmatched: {
      [`sample_${sampleA.sample_index}`]: detectionsA.length - usedA.size,
      [`sample_${sampleB.sample_index}`]: detectionsB.length - usedB.size,
    },
  };
}

function calculateIouReport(samples) {
  const pairs = [];
  for (let first = 0; first < samples.length; first += 1) {
    for (let second = first + 1; second < samples.length; second += 1) {
      pairs.push(matchSamplePair(samples[first], samples[second]));
    }
  }
  const allMatches = pairs.flatMap((pair) => pair.matches);
  const overallMean = allMatches.length
    ? allMatches.reduce((sum, match) => sum + match.iou, 0) / allMatches.length
    : null;
  return {
    matching: "同名目标按最大 IoU 一对一匹配",
    overall_mean_iou: overallMean === null ? null : Number(overallMean.toFixed(4)),
    pairs,
  };
}

function formatRawOutputs(samples) {
  return samples
    .map((sample) => {
      const output = sample.raw_output || `[错误] ${sample.error || "没有模型输出"}`;
      return `===== 第 ${sample.sample_index} 次 =====\n${output}`;
    })
    .join("\n\n");
}

async function saveQwenPrompt() {
  const button = document.querySelector("#saveQwenPrompt");
  button.disabled = true;
  setStatus("#savePromptStatus", "保存中…", "running");
  try {
    const result = await api("/api/qwen-prompts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task_type: requestTaskType.value,
        sku_name: qwenSku.value,
        prompt: qwenPrompt.value,
        level: requestLevel.value,
        hand: requestHand.value,
      }),
    });
    promptPairsBySku.set(result.sku_name, {
      qwen3_prompt: result.qwen3_prompt,
      sam3_prompt: result.sam3_prompt,
    });
    loadedPromptSku = result.sku_name;
    setStatus(
      "#savePromptStatus",
      result.overwritten ? "已覆盖原 Prompt" : "已保存",
      "success",
    );
  } catch (error) {
    setStatus("#savePromptStatus", error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function syncSamPairedSku() {
  const skuName = qwenSku.value.trim();
  document.querySelector("#samPairedSku").textContent = skuName || "请在左侧输入商品名称";
}

async function saveSamPromptPair() {
  const button = document.querySelector("#saveSamPrompt");
  button.disabled = true;
  setStatus("#saveSamPromptStatus", "保存中…", "running");
  try {
    const result = await api("/api/prompt-pairs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task_type: requestTaskType.value,
        sku_name: qwenSku.value,
        qwen3_prompt: qwenPrompt.value,
        sam3_prompt: samPrompt.value,
        level: requestLevel.value,
        hand: requestHand.value,
      }),
    });
    promptPairsBySku.set(result.sku_name, {
      qwen3_prompt: result.qwen3_prompt,
      sam3_prompt: result.sam3_prompt,
    });
    loadedPromptSku = result.sku_name;
    setStatus(
      "#saveSamPromptStatus",
      result.overwritten ? "已覆盖该 SKU 的配对 Prompt" : "配对 Prompt 已保存",
      "success",
    );
  } catch (error) {
    setStatus("#saveSamPromptStatus", error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function runQwen() {
  const button = document.querySelector("#runQwen");
  const rawResult = document.querySelector("#qwenRawResult");
  button.disabled = true;
  rawResult.textContent = "等待 Locate Debug 返回…";
  setStatus("#qwenStatus", "运行中…", "running");
  setStatus("#samStatus", "运行中…", "running");
  resetQwenReferencePreview();
  try {
    if (!requestTaskType) {
      throw new Error("页面版本不一致，请按 Ctrl+F5 强制刷新");
    }
    const requestPayload = {
      task_type: requestTaskType.value,
      product_name: qwenSku.value,
      level: requestLevel.value,
      hand: requestHand.value,
      qwen3_prompt: qwenPrompt.value,
      sam3_prompt: samPrompt.value,
    };
    if (offlineImageBase64) {
      requestPayload.image_name = offlineImageName;
      requestPayload.image_base64 = offlineImageBase64;
    }
    if (offlineDepthBase64) {
      if (!offlineImageBase64) {
        throw new Error("上传离线深度数据时必须同时上传对应 RGB 图片");
      }
      requestPayload.depth_image_name = offlineDepthName;
      requestPayload.depth_image_base64 = offlineDepthBase64;
      requestPayload.depth_is_bigendian = locateDepthByteOrder.value === "big";
    }
    const result = await api("/api/locate-debug", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload),
    });
    latestImageBase64 = result.image_base64;
    latestImageMediaType = result.image_media_type || "image/jpeg";
    const rawQwenBboxes = result.raw_qwen_bboxes || [];
    const qwenBboxes = result.qwen_bboxes || [];
    const rawSamInstances = result.raw_sam_instances || [];
    const instances = result.instances || [];
    const selectedInstance = result.selected_instance || null;
    const selectedInstanceIndex = Number(result.selected_instance_index || 0);
    renderQwenReferencePreview(result);
    const summarizeInstance = (instance) =>
      instance
        ? {
            bbox: instance.bbox,
            score: instance.score,
            depth_mm: instance.depth_mm,
            mapped_product_name: instance.mapped_product_name,
            is_selected: instance.is_selected,
            mask: `<base64 ${instance.mask.length} chars>`,
          }
        : null;

    await drawBase(rawQwenCanvas);
    const rawQwenContext = rawQwenCanvas.getContext("2d");
    rawQwenBboxes.forEach((record, index) => {
      const sampleIndex = Number(record.sample_index || 1);
      drawBox(
        rawQwenContext,
        record.bbox_original,
        `S${sampleIndex} #${index + 1}`,
        qwenSampleColors[(sampleIndex - 1) % qwenSampleColors.length],
      );
    });
    document.querySelector("#rawQwenEmpty").hidden = true;

    await drawBase(qwenCanvas);
    const qwenContext = qwenCanvas.getContext("2d");
    qwenBboxes.forEach((record, index) => {
      drawBox(
        qwenContext,
        record.bbox_original,
        `Qwen #${index + 1}`,
        qwenSampleColors[index % qwenSampleColors.length],
      );
    });
    document.querySelector("#qwenEmpty").hidden = true;
    document.querySelector("#qwenResult").textContent = JSON.stringify(qwenBboxes, null, 2);
    document.querySelector("#qwenIouResult").textContent = JSON.stringify(
      {
        sku_id: result.sku_id,
        product_name: result.product_name,
        image_name: result.image_name,
        image_path: result.image_path,
        image_size: result.image_size,
        qwen3_prompt_used: result.qwen3_prompt_used,
        sam3_prompt_used: result.sam3_prompt_used,
        qwen_reference_image_used: Boolean(result.qwen_reference_image_used),
        qwen_reference_image_name: result.qwen_reference_image_name || null,
        qwen_reference_image_media_type: result.qwen_reference_image_media_type || null,
        hard_case: result.hard_case || null,
      },
      null,
      2,
    );
    rawResult.textContent = JSON.stringify(
      {
        ...result,
        image_base64: `<base64 ${result.image_base64.length} chars>`,
        raw_sam_instances: rawSamInstances.map(summarizeInstance),
        instances: instances.map(summarizeInstance),
        selected_instance: summarizeInstance(selectedInstance),
      },
      null,
      2,
    );

    await drawBase(rawSamCanvas);
    for (let index = 0; index < rawSamInstances.length; index += 1) {
      await drawMask(
        rawSamCanvas,
        rawSamInstances[index].mask,
        colors[index % colors.length],
      );
    }
    const rawSamContext = rawSamCanvas.getContext("2d");
    rawSamInstances.forEach((instance, index) => {
      const score = Number(instance.score || 0).toFixed(3);
      drawBox(
        rawSamContext,
        instance.bbox,
        `#${index + 1} ${score}`,
        colors[index % colors.length],
      );
    });
    document.querySelector("#rawSamEmpty").hidden = true;

    await drawBase(samCanvas);
    for (let index = 0; index < instances.length; index += 1) {
      await drawMask(samCanvas, instances[index].mask, colors[index % colors.length]);
    }
    const samContext = samCanvas.getContext("2d");
    instances.forEach((instance, index) => {
      const score = Number(instance.score || 0).toFixed(3);
      const mappedName = instance.mapped_product_name || "";
      const groupLabel = instance.hard_case_group_index
        ? `G${instance.hard_case_group_index} `
        : "";
      const selectedLabel = instance.is_selected ? "目标 " : "";
      drawBox(
        samContext,
        instance.bbox,
        `${selectedLabel}${groupLabel}${mappedName || `#${index + 1}`} ${score}`,
        instance.is_selected ? "#22c55e" : colors[index % colors.length],
      );
    });
    if (selectedInstance) {
      drawBox(
        samContext,
        selectedInstance.bbox,
        `PICK #${selectedInstanceIndex}`,
        "#ef4444",
        Math.max(6, samCanvas.width / 180),
      );
    }
    document.querySelector("#samEmpty").hidden = true;
    document.querySelector("#samResult").textContent = JSON.stringify(
      {
        selected_instance_index: selectedInstanceIndex || null,
        selected_instance: selectedInstance
          ? {
              bbox: selectedInstance.bbox,
              score: selectedInstance.score,
              depth_mm: selectedInstance.depth_mm,
              mapped_product_name: selectedInstance.mapped_product_name,
              is_selected: selectedInstance.is_selected,
            }
          : null,
        candidates: instances.map(
          ({ bbox, score, depth_mm, hard_case_group_index, mapped_product_name, is_selected }) => ({
            bbox,
            score,
            depth_mm,
            hard_case_group_index,
            mapped_product_name,
            is_selected,
          }),
        ),
      },
      null,
      2,
    );
    const image = await drawBase(finalLocateBboxCanvas);
    if (selectedInstance) {
      const bboxContext = finalLocateBboxCanvas.getContext("2d");
      drawBox(bboxContext, selectedInstance.bbox, `最终目标 ${result.product_name}`, "#22c55e");
      document.querySelector("#finalLocateBboxEmpty").hidden = true;

      const [x1, y1, x2, y2] = selectedInstance.bbox;
      const paddingX = Math.max(8, (x2 - x1) * 0.08);
      const paddingY = Math.max(8, (y2 - y1) * 0.08);
      const cropX = Math.max(0, Math.floor(x1 - paddingX));
      const cropY = Math.max(0, Math.floor(y1 - paddingY));
      const cropRight = Math.min(image.naturalWidth, Math.ceil(x2 + paddingX));
      const cropBottom = Math.min(image.naturalHeight, Math.ceil(y2 + paddingY));
      finalLocateCropCanvas.width = Math.max(1, cropRight - cropX);
      finalLocateCropCanvas.height = Math.max(1, cropBottom - cropY);
      const cropContext = finalLocateCropCanvas.getContext("2d");
      cropContext.drawImage(
        image, cropX, cropY, cropRight - cropX, cropBottom - cropY,
        0, 0, cropRight - cropX, cropBottom - cropY,
      );
      drawBox(
        cropContext,
        [x1 - cropX, y1 - cropY, x2 - cropX, y2 - cropY],
        result.product_name,
        "#22c55e",
      );
      document.querySelector("#finalLocateCropEmpty").hidden = true;
    }
    if (result.error) {
      const errorMessage = `HTTP ${result.error_status_code || 500}: ${result.error}`;
      setStatus("#qwenStatus", errorMessage, "error");
      setStatus("#samStatus", "推理失败，已显示 Debug 接口返回的原图", "error");
    } else {
      setStatus("#qwenStatus", `${qwenBboxes.length} 个共识 bbox`, "success");
      setStatus(
        "#samStatus",
        result.hard_case
          ? `${instances.length} 个第一排实例 · ${result.hard_case.group_id} · 最终 PICK #${selectedInstanceIndex}`
          : selectedInstance
            ? `${instances.length} 个候选 · 最终 PICK #${selectedInstanceIndex}`
            : `${instances.length} 个候选 · 未返回最终 PICK`,
        selectedInstance ? "success" : "error",
      );
    }
  } catch (error) {
    rawResult.textContent = error.message;
    setStatus("#qwenStatus", error.message, "error");
    setStatus("#samStatus", error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function maskImage(base64) {
  const image = new Image();
  image.src = `data:image/png;base64,${base64}`;
  await image.decode();
  return image;
}

async function drawMask(canvas, base64, color) {
  const mask = await maskImage(base64);
  const layer = document.createElement("canvas");
  layer.width = canvas.width;
  layer.height = canvas.height;
  const layerContext = layer.getContext("2d", { willReadFrequently: true });
  layerContext.drawImage(mask, 0, 0, layer.width, layer.height);
  const pixels = layerContext.getImageData(0, 0, layer.width, layer.height);
  const rgb = color.match(/[a-f\d]{2}/gi).map((value) => parseInt(value, 16));
  for (let index = 0; index < pixels.data.length; index += 4) {
    const foreground = pixels.data[index];
    pixels.data[index] = rgb[0];
    pixels.data[index + 1] = rgb[1];
    pixels.data[index + 2] = rgb[2];
    pixels.data[index + 3] = foreground > 127 ? 92 : 0;
  }
  layerContext.putImageData(pixels, 0, 0);
  canvas.getContext("2d").drawImage(layer, 0, 0);
}

async function runSam() {
  const button = document.querySelector("#runSam");
  button.disabled = true;
  try {
    const crop = await drawSelectedCrop();
    setStatus("#samStatus", "运行中…", "running");
    const imageBase64 = samCanvas.toDataURL("image/jpeg", 0.95).split(",")[1];
    const result = await api("/api/sam3-crop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: samPrompt.value,
        image_base64: imageBase64,
        crop_box_original: crop.cropBox,
      }),
    });
    const instances = result.instances || [];
    for (let index = 0; index < instances.length; index += 1) {
      const instance = instances[index];
      const color = colors[index % colors.length];
      await drawMask(samCanvas, instance.mask_png_base64, color);
    }
    const context = samCanvas.getContext("2d");
    instances.forEach((instance, index) => {
      const score = Number(instance.score || 0).toFixed(3);
      drawBox(context, instance.bbox_xyxy, `#${instance.instance_id}  ${score}`, colors[index % colors.length]);
    });
    document.querySelector("#samEmpty").hidden = true;
    document.querySelector("#samResult").textContent = JSON.stringify(
      {
        qwen_detection: {
          sample_index: crop.sampleIndex,
          target_index: crop.detectionIndex + 1,
          name: crop.detection.name,
          bbox: crop.detection.bbox,
        },
        crop_box_original: result.crop_box_original,
        instances: instances.map(
          ({ instance_id, score, bbox_xyxy, bbox_original_xyxy }) => ({
            instance_id,
            score,
            bbox_crop_xyxy: bbox_xyxy,
            bbox_original_xyxy,
          }),
        ),
      },
      null,
      2,
    );
    setStatus("#samStatus", `${instances.length} 个 crop 实例`, "success");
  } catch (error) {
    setStatus("#samStatus", error.message, "error");
  } finally {
    button.disabled = cropChoices.length === 0;
  }
}

document.querySelector("#runQwen").addEventListener("click", runQwen);
document.querySelector("#saveQwenPrompt").addEventListener("click", saveQwenPrompt);
document.querySelector("#saveSamPrompt").addEventListener("click", saveSamPromptPair);
document.querySelector("#runSam").addEventListener("click", runSam);
locateImageInput.addEventListener("change", selectOfflineImage);
locateDepthInput.addEventListener("change", selectOfflineDepth);
clearLocateImage.addEventListener("click", clearOfflineImage);
requestTaskType?.addEventListener("change", () => {
  resetQwenReferencePreview();
  loadSkus()
    .then(() => loadPromptPairForSelectedSku())
    .catch((error) => {
      setStatus("#savePromptStatus", error.message, "error");
      setStatus("#saveSamPromptStatus", error.message, "error");
    });
});
[requestLevel, requestHand].forEach((control) => {
  control.addEventListener("change", () => {
    loadSkus()
      .then(() => loadPromptPairForSelectedSku())
      .catch((error) => setStatus("#savePromptStatus", error.message, "error"));
  });
});
qwenSku.addEventListener("input", () => {
  syncSamPairedSku();
  setStatus("#savePromptStatus", "未保存");
  setStatus("#saveSamPromptStatus", "未保存");
  loadPromptPairForSelectedSku();
});
qwenPrompt.addEventListener("input", () => {
  setStatus("#savePromptStatus", "未保存");
  setStatus("#saveSamPromptStatus", "未保存");
});
samPrompt.addEventListener("input", () => setStatus("#saveSamPromptStatus", "未保存"));
cropDetectionSelect.addEventListener("change", () => {
  drawSelectedCrop().catch((error) => setStatus("#samStatus", error.message, "error"));
});
batchRecordSelect.addEventListener("change", () => renderBatchRecord());
batchProductSelect.addEventListener("change", renderBatchProduct);
batchResultFileSelect.addEventListener("change", () => {
  stopBatchRerunPolling();
  batchRecords = [];
  batchRecordSelect.replaceChildren();
  batchProductSelect.replaceChildren();
  loadBatchResults().finally(pollBatchRerunStatus);
});
batchPrevious.addEventListener("click", () => {
  batchRecordSelect.value = String(Math.max(0, Number(batchRecordSelect.value) - 1));
  renderBatchRecord();
});
batchNext.addEventListener("click", () => {
  batchRecordSelect.value = String(
    Math.min(batchRecords.length - 1, Number(batchRecordSelect.value) + 1),
  );
  renderBatchRecord();
});
batchRefresh.addEventListener("click", rerunBatchResultsWithOverwrite);
document.querySelector("#qwenCoordinateMode").addEventListener("change", () => {
  if (cropChoices.length) {
    drawSelectedCrop().catch((error) => setStatus("#samStatus", error.message, "error"));
  }
});
loadSkus().catch((error) => {
  setStatus("#qwenStatus", error.message, "error");
  setStatus("#savePromptStatus", error.message, "error");
  setStatus("#saveSamPromptStatus", error.message, "error");
  setStatus("#samStatus", error.message, "error");
});
loadBatchResultFiles().finally(pollBatchRerunStatus);

syncSamPairedSku();
resetQwenReferencePreview();
