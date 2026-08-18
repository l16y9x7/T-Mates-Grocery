const groupSelect = document.querySelector("#groupSelect");
const recordSelect = document.querySelector("#recordSelect");
const rowSelect = document.querySelector("#rowSelect");
const skuSelect = document.querySelector("#skuSelect");
const promptInput = document.querySelector("#promptInput");
const expectedCountInput = document.querySelector("#expectedCount");
const runPromptButton = document.querySelector("#runPrompt");
const runAllButton = document.querySelector("#runAll");

let records = [];
let mapping = [];
let frontCompareRequestKey = "";

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let payload = {};
  try { payload = await response.json(); } catch (_error) { /* no-op */ }
  if (!response.ok) {
    const detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail || payload);
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return payload;
}

function setStatus(id, message, kind = "") {
  const element = document.querySelector(id);
  element.textContent = message;
  element.className = `status ${kind}`.trim();
}

function addOption(select, value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  select.append(option);
}

function selectedRecord() {
  return records.find((item) => item.group === groupSelect.value && item.record === recordSelect.value) || null;
}

function selectedRow() {
  const record = selectedRecord();
  return record?.rows?.find((row) => String(row.row_index) === rowSelect.value) || null;
}

function setImage(id, url) {
  const image = document.querySelector(id);
  const figure = image.closest("figure");
  if (url) {
    image.src = url;
    figure?.classList.remove("unavailable");
  } else {
    image.removeAttribute("src");
    figure?.classList.add("unavailable");
  }
}

function populateRecords() {
  const filtered = records.filter((item) => item.group === groupSelect.value);
  const previous = recordSelect.value;
  recordSelect.replaceChildren();
  filtered.forEach((item) => addOption(recordSelect, item.record, item.record));
  if (filtered.some((item) => item.record === previous)) recordSelect.value = previous;
  populateRows();
}

function populateRows() {
  const record = selectedRecord();
  rowSelect.replaceChildren();
  (record?.rows || []).forEach((row) => addOption(
    rowSelect,
    String(row.row_index),
    `ROW ${row.row_index} · ${row.level || "UNKNOWN"}`,
  ));
  renderSelection();
}

function rowPromptOptions(row) {
  const wanted = new Set(row?.candidate_skus || []);
  const filtered = mapping.filter((item) => wanted.has(item.sku_name));
  return filtered.length ? filtered : mapping;
}

function selectedConfiguredGroup() {
  const row = selectedRow();
  if (!skuSelect.value.startsWith("config:")) return null;
  const index = Number(skuSelect.value.split(":", 2)[1]);
  return (row?.prompt_groups || []).find((item) => item.group_index === index) || null;
}

function syncSelectedPrompt() {
  const configured = selectedConfiguredGroup();
  if (configured) {
    promptInput.value = configured.sam3_prompt;
    expectedCountInput.value = configured.expected_front_count;
    return;
  }
  const mapped = mapping.find((item) => item.sku_name === skuSelect.value);
  if (mapped) promptInput.value = mapped.prompt;
  expectedCountInput.value = "";
}

function populatePrompts(row) {
  const configuredGroups = row?.prompt_groups || [];
  const options = rowPromptOptions(row);
  const previous = skuSelect.value;
  skuSelect.replaceChildren();
  addOption(skuSelect, "", "自定义 Prompt");
  if (configuredGroups.length) {
    configuredGroups.forEach((item) => addOption(
      skuSelect,
      `config:${item.group_index}`,
      `组 ${item.group_index} · ${item.expected_front_count} 列 · ${item.sam3_prompt}`,
    ));
    if (configuredGroups.some((item) => `config:${item.group_index}` === previous)) {
      skuSelect.value = previous;
    } else {
      skuSelect.value = `config:${configuredGroups[0].group_index}`;
    }
  } else {
    options.forEach((item) => addOption(skuSelect, item.sku_name, `${item.sku_name} · ${item.prompt}`));
    if (options.some((item) => item.sku_name === previous)) skuSelect.value = previous;
  }
  syncSelectedPrompt();
  runAllButton.disabled = !configuredGroups.length && !(row?.candidate_skus || []).length;
}

function renderFrontCompare(payload) {
  const container = document.querySelector("#frontCompareResults");
  container.replaceChildren();
  const promptGroups = payload.prompt_groups || [];
  if (!promptGroups.length) {
    const empty = document.createElement("div");
    empty.className = "flow-compare-empty";
    empty.textContent = "该层没有可展示的 Prompt 对比结果";
    container.append(empty);
    return;
  }
  promptGroups.forEach((item) => {
    const card = document.createElement("article");
    card.className = "flow-compare-card";
    const header = document.createElement("header");
    const heading = document.createElement("h3");
    const summary = document.createElement("span");
    const missingCount = (item.missing_slots || []).length;
    const systematicShift = item.systematic_depth_shift;
    const systematicShiftText = systematicShift?.detected
      ? ` · 共同偏移 ${systematicShift.median_delta_mm >= 0 ? "+" : ""}${systematicShift.median_delta_mm}mm（已抑制）`
      : "";
    const missingProductText = (item.missing_product_names || []).length
      ? ` · 商品：${item.missing_product_names.join("、")}`
      : "";
    const detectionFailureText = item.current_detection_failed
      ? " · Current SAM3 未检出（未判缺失）"
      : "";
    const matchingStrategyText = item.slot_matching_strategy === "ordinal_left_to_right"
      ? " · 左右顺序匹配"
      : " · 单调序列匹配";
    heading.textContent = `GROUP ${item.group_index} · ${item.prompt}`;
    const resolvedSlotCount = item.resolved_slot_count ?? (item.slots || []).length;
    summary.textContent = `槽位 ${resolvedSlotCount}/${item.expected_front_count} · baseline前排mask ${item.baseline_front_count} · current近层mask ${item.current_front_count}${matchingStrategyText} · Δ>${item.depth_delta_threshold_mm}mm 判缺失 · ${missingCount ? `缺失 ${missingCount}` : "无缺失"}${missingProductText}${systematicShiftText}${detectionFailureText}`;
    summary.className = (missingCount || item.current_detection_failed) ? "status error" : "status success";
    header.append(heading, summary);
    card.append(header);

    const images = document.createElement("div");
    images.className = "flow-compare-images";
    [
      [item.artifact_urls?.baseline_front, "Baseline 新流程", "基准前排 mask"],
      [item.artifact_urls?.current_front, "Current 新流程", "当前前排 mask"],
      [item.artifact_urls?.comparison, "槽位对比", "青色=已占用槽位，紫色虚线=缺失"],
      [item.artifact_urls?.place_references, "Place 参照 bbox", "紫色=目标槽位，绿色=返回的邻居 bbox"],
    ].forEach(([url, title, description]) => {
      if (!url) return;
      const figure = document.createElement("figure");
      const caption = document.createElement("figcaption");
      const strong = document.createElement("strong");
      const small = document.createElement("small");
      const image = document.createElement("img");
      strong.textContent = title;
      small.textContent = description;
      caption.append(strong, small);
      if (url) image.src = url;
      image.alt = title;
      figure.append(caption, image);
      images.append(figure);
    });
    card.append(images);

    const table = document.createElement("table");
    table.className = "slot-table";
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    ["槽位", "商品身份", "状态", "Baseline depth", "Current depth", "Δ depth", "实例匹配"].forEach((label) => {
      const th = document.createElement("th");
      th.textContent = label;
      headRow.append(th);
    });
    thead.append(headRow);
    table.append(thead);
    const tbody = document.createElement("tbody");
    (item.slots || []).forEach((slot) => {
      const row = document.createElement("tr");
      row.className = slot.status.startsWith("missing_") ? "missing" : "occupied";
      const statusLabels = {
        occupied: "已占用",
        occupied_systematic_shift: "已占用（共同偏移）",
        missing_unmatched: "当前无匹配 mask",
        missing_depth_delta: "深度后移",
        baseline_incomplete: "基准不完整",
        current_detection_failed: "Current SAM3 未检出（未知）",
      };
      const baselineInstance = slot.baseline_instance_index == null
        ? "—"
        : `#${slot.baseline_instance_index}`;
      const currentInstance = slot.current_instance_index == null
        ? "—"
        : `#${slot.current_instance_index}`;
      const values = [
        `SLOT ${slot.slot_index}`,
        slot.product_name || "未配置",
        statusLabels[slot.status] || slot.status,
        slot.baseline_depth_mm == null ? "—" : `${slot.baseline_depth_mm} mm`,
        slot.current_depth_mm == null ? "—" : `${slot.current_depth_mm} mm`,
        slot.depth_delta_mm == null ? "—" : `${slot.depth_delta_mm} mm`,
        `${baselineInstance} → ${currentInstance}`,
      ];
      values.forEach((value) => {
        const td = document.createElement("td");
        td.textContent = value;
        row.append(td);
      });
      tbody.append(row);
    });
    table.append(tbody);
    card.append(table);

    const placeTests = item.place_reference_tests || [];
    if (placeTests.length) {
      const placeTable = document.createElement("table");
      placeTable.className = "slot-table place-reference-table";
      const placeHead = document.createElement("thead");
      const placeHeadRow = document.createElement("tr");
      ["Place 目标", "商品", "direction", "返回 bbox", "状态"].forEach((label) => {
        const th = document.createElement("th");
        th.textContent = label;
        placeHeadRow.append(th);
      });
      placeHead.append(placeHeadRow);
      placeTable.append(placeHead);
      const placeBody = document.createElement("tbody");
      placeTests.forEach((test) => {
        const row = document.createElement("tr");
        row.className = test.status === "success" ? "occupied" : "missing";
        const bboxes = (test.references || []).map((ref) =>
          `[${(ref.bbox_original_xyxy || ref.bbox_xyxy || []).map((value) => Math.round(value)).join(", ")}]`
        ).join(" · ");
        [
          `SLOT ${test.slot_index}`,
          test.product_name || "未配置",
          test.direction || "—",
          bboxes || "—",
          test.status === "success" ? "已选参照" : "不足两个参照",
        ].forEach((value) => {
          const td = document.createElement("td");
          td.textContent = value;
          row.append(td);
        });
        placeBody.append(row);
      });
      placeTable.append(placeBody);
      card.append(placeTable);
    }
    container.append(card);
  });
}

async function loadFrontCompare(record, row) {
  const container = document.querySelector("#frontCompareResults");
  const requestKey = `${record.group}/${record.record}/${row.level || ""}`;
  frontCompareRequestKey = requestKey;
  container.replaceChildren();
  if (!record.front_compare_available) {
    const empty = document.createElement("div");
    empty.className = "flow-compare-empty";
    empty.textContent = "尚未生成。运行 python inspect/batch_front_row_compare.py --workers 4 --overwrite 后刷新页面。";
    container.append(empty);
    setStatus("#frontCompareStatus", "尚未生成批量结果");
    return;
  }
  if (!row.level) {
    setStatus("#frontCompareStatus", "当前行没有物理层编号", "error");
    return;
  }
  setStatus("#frontCompareStatus", "正在读取缓存结果……");
  try {
    const payload = await api(
      `/api/sam-row-compare/result/${encodeURIComponent(record.group)}/${encodeURIComponent(record.record)}/${encodeURIComponent(row.level)}`,
    );
    if (frontCompareRequestKey !== requestKey) return;
    renderFrontCompare(payload);
    const missingCount = (payload.prompt_groups || []).reduce(
      (total, item) => total + (item.missing_slots || []).length,
      0,
    );
    const missingProductNames = (payload.findings || []).map(
      (item) => item.shortage_product_name,
    );
    setStatus(
      "#frontCompareStatus",
      missingCount
        ? `完成 · 疑似缺失 ${missingCount}${missingProductNames.length ? ` · ${missingProductNames.join("、")}` : ""}`
        : "完成 · 无缺失",
      missingCount ? "error" : "success",
    );
  } catch (error) {
    if (frontCompareRequestKey !== requestKey) return;
    const empty = document.createElement("div");
    empty.className = "flow-compare-empty";
    empty.textContent = error.message;
    container.append(empty);
    setStatus("#frontCompareStatus", error.message, "error");
  }
}

function renderSelection() {
  const record = selectedRecord();
  const row = selectedRow();
  if (!record || !row) return;
  setImage("#baselineImage", record.baseline_url);
  setImage("#currentImage", record.current_url);
  setImage("#rowDetectionImage", record.row_detection_url);
  setImage("#rowRgbImage", row.rgb_url);
  setImage("#rowDepthImage", row.depth_preview_url);
  const baselineShelf = row.shelf_inputs.baseline;
  const currentShelf = row.shelf_inputs.current;
  setImage("#baselineShelfFiltered", baselineShelf.shelf_filtered_url);
  setImage("#currentShelfFiltered", currentShelf.shelf_filtered_url);
  setImage("#baselineShelfMask", baselineShelf.shelf_mask_url);
  setImage("#currentShelfMask", currentShelf.shelf_mask_url);
  setImage("#baselineRetainedMask", baselineShelf.retained_mask_url);
  setImage("#currentRetainedMask", currentShelf.retained_mask_url);
  const baselineWidth = baselineShelf.selected_component?.width_ratio;
  const currentWidth = currentShelf.selected_component?.width_ratio;
  const shelfSummary = (item, width) => item.fallback_to_full_image
    ? "完整图回退"
    : `覆盖 ${(100 * width).toFixed(1)}%`;
  setStatus(
    "#shelfInputStatus",
    `Baseline ${shelfSummary(baselineShelf, baselineWidth)} · Current ${shelfSummary(currentShelf, currentWidth)}`,
    (baselineShelf.fallback_to_full_image || currentShelf.fallback_to_full_image) ? "error" : "success",
  );
  document.querySelector("#rowRgbTitle").textContent = `ROW ${row.row_index} · ${row.level || "UNKNOWN"}`;
  document.querySelector("#rowBBox").textContent = `原图 bbox [${(row.crop_bbox_xywh || []).join(", ")}]`;
  document.querySelector("#rowDepthSummary").textContent =
    `有效 ${(100 * (row.valid_depth_ratio || 0)).toFixed(1)}% · ${row.valid_depth_pixels || 0} px`;
  populatePrompts(row);
  document.querySelector("#results").replaceChildren();
  setStatus("#loadStatus", `${record.group} · ${record.record} · ROW ${row.row_index}`, "success");
  loadFrontCompare(record, row);
}

function metric(label, value, suffix = "") {
  const wrapper = document.createElement("div");
  const key = document.createElement("span");
  const content = document.createElement("strong");
  key.textContent = label;
  content.textContent = value === null || value === undefined ? "—" : `${value}${suffix}`;
  wrapper.append(key, content);
  return wrapper;
}

function renderResult(result) {
  const card = document.createElement("article");
  card.className = "result-card";
  const header = document.createElement("header");
  const title = document.createElement("h2");
  const groupLabel = result.config_group_index ? `GROUP ${result.config_group_index}` : (result.sku_name || "CUSTOM");
  title.textContent = `${groupLabel} · ${result.prompt}`;
  const timing = document.createElement("span");
  const countText = result.count_constraint
    ? ` · expected ${result.count_constraint.expected} ${result.count_constraint.satisfied ? "✓" : "不足"}`
    : "";
  const depthLayer = result.count_constraint?.global_depth_layer;
  const depthLayerText = depthLayer?.requested
    ? depthLayer.enabled
      ? ` · 深度近层 ≤ ${depthLayer.front_depth_max_mm}mm（断层 ${depthLayer.split_gap_mm}mm）`
      : ` · 深度分层回退（${depthLayer.reason}）`
    : "";
  const regularColumns = result.count_constraint?.regular_column_prior;
  const regularColumnsText = regularColumns?.enabled
    ? ` · 等距列 ${regularColumns.target_pitch_px}px${regularColumns.missing_column_count ? ` · 缺失 ${regularColumns.missing_column_count} 列` : ""}`
    : "";
  const bottomLine = result.count_constraint?.bottom_line_prior;
  const bottomLineText = bottomLine?.enabled
    ? ` · 底线 ±${bottomLine.tolerance_px}px`
    : "";
  const suspectedMissingCount = (result.suspected_missing_regions || []).length;
  const suspectedMissingText = suspectedMissingCount
    ? ` · 疑似缺失 ${suspectedMissingCount}`
    : "";
  const samePromptDepth = result.count_constraint?.same_prompt_depth_band;
  const samePromptDepthText = samePromptDepth?.requested
    ? ` · 同Prompt深度≤${samePromptDepth.max_spread_mm}mm`
    : "";
  const samRetryText = (result.sam3_attempts || []).length > 1
    ? (result.sam3_detection_status === "detected"
      ? " · SAM3 低阈值重试成功"
      : " · SAM3 重试后仍未检出")
    : "";
  timing.textContent = `${result.front_instance_indices.length}/${result.instances.length} front${countText}${depthLayerText}${bottomLineText}${regularColumnsText}${samePromptDepthText}${suspectedMissingText}${samRetryText} · ${Math.round(result.elapsed_ms)} ms`;
  header.append(title, timing);
  card.append(header);

  const gallery = document.createElement("div");
  gallery.className = "result-gallery";
  const summary = document.createElement("div");
  summary.className = "result-summary";
  [
    [result.overlay_data_url, "全部实例：绿=前排，红=后排，黄=深度不可靠，紫色虚线=疑似缺失"],
    [result.front_overlay_data_url, "最终 front-row 实例；紫色虚线=疑似缺失"],
    [result.front_mask_data_url, "最终 front-row mask"],
  ].forEach(([url, label]) => {
    const figure = document.createElement("figure");
    const caption = document.createElement("figcaption");
    const image = document.createElement("img");
    caption.textContent = label;
    image.src = url;
    image.alt = label;
    figure.append(caption, image);
    summary.append(figure);
  });
  if ((result.occlusion_edges || []).length) {
    const graph = document.createElement("div");
    graph.className = "edge-list";
    graph.textContent = result.occlusion_edges.map((edge) =>
      `#${edge.front} → #${edge.back} · Δ${edge.depth_delta_mm}mm · ${edge.comparison_source}`
    ).join("\n");
    summary.append(graph);
  }
  gallery.append(summary);
  const instances = document.createElement("div");
  instances.className = "instances";
  if (!result.instances.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "SAM3 没有返回实例";
    instances.append(empty);
  }
  result.instances.forEach((item) => {
    const instance = document.createElement("article");
    instance.className = `instance ${item.front_status || "uncertain"}`;
    const heading = document.createElement("h3");
    const score = typeof item.score === "number" ? item.score.toFixed(3) : "—";
    const statusLabels = { front: "前排保留", back: "后排排除", uncertain: "深度不可靠", duplicate: "重复 mask" };
    heading.textContent = `#${item.instance_index} · ${statusLabels[item.front_status] || item.front_status} · score ${score}`;
    const reason = document.createElement("p");
    reason.className = "selection-reason";
    reason.textContent = item.selection_reason;
    const images = document.createElement("div");
    images.className = "instance-images";
    [item.mask_data_url, item.inner_mask_data_url, item.masked_depth_data_url].forEach((url, index) => {
      const image = document.createElement("img");
      image.src = url;
      image.alt = ["SAM mask", "Eroded inner mask", "Masked depth"][index];
      images.append(image);
    });
    const stats = document.createElement("div");
    stats.className = "depth-stats";
    stats.append(
      metric("稳定近层", item.stable_depth_mm, " mm"),
      metric("MAD", item.depth_mad_mm, " mm"),
      metric("深度簇支持", `${(100 * item.depth_cluster_support_ratio).toFixed(1)}%`),
      metric("腐蚀核", item.erode_kernel_px, " px"),
      metric("有效深度", `${(100 * item.valid_depth_ratio).toFixed(1)}%`),
      metric("mask", item.mask_pixels, " px"),
    );
    instance.append(heading, reason, images, stats);
    instances.append(instance);
  });
  gallery.append(instances);
  card.append(gallery);
  document.querySelector("#results").prepend(card);
}

async function runOne(skuName, prompt, expectedFrontCount = null, configGroupIndex = null) {
  const record = selectedRecord();
  const row = selectedRow();
  if (!record || !row) throw new Error("请先选择记录和货架层");
  return api("/api/sam-row-debug/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      group: record.group,
      record: record.record,
      row_index: row.row_index,
      sku_name: skuName,
      prompt,
      expected_front_count: expectedFrontCount,
      config_group_index: configGroupIndex,
    }),
  });
}

async function runCurrentPrompt() {
  const prompt = promptInput.value.trim();
  const configured = selectedConfiguredGroup();
  const expectedCount = expectedCountInput.value ? Number(expectedCountInput.value) : null;
  if (!prompt) {
    setStatus("#runStatus", "SAM3 Prompt 不能为空", "error");
    return;
  }
  runPromptButton.disabled = true;
  setStatus("#runStatus", "SAM3 正在运行……");
  try {
    const result = await runOne(
      configured ? "" : skuSelect.value,
      prompt,
      expectedCount,
      configured?.group_index || null,
    );
    renderResult(result);
    setStatus("#runStatus", `完成：${result.instances.length} 个实例`, "success");
  } catch (error) {
    setStatus("#runStatus", error.message, "error");
  } finally {
    runPromptButton.disabled = false;
  }
}

async function runAllMappedPrompts() {
  const row = selectedRow();
  const configuredGroups = row?.prompt_groups || [];
  if (configuredGroups.length) {
    runAllButton.disabled = true;
    runPromptButton.disabled = true;
    document.querySelector("#results").replaceChildren();
    try {
      for (let index = 0; index < configuredGroups.length; index += 1) {
        const item = configuredGroups[index];
        setStatus("#runStatus", `正在运行配置组 ${index + 1}/${configuredGroups.length} · ${item.sam3_prompt}`);
        const result = await runOne("", item.sam3_prompt, item.expected_front_count, item.group_index);
        renderResult(result);
      }
      setStatus("#runStatus", `全部完成：${configuredGroups.length} 个独立配置组`, "success");
    } catch (error) {
      setStatus("#runStatus", error.message, "error");
    } finally {
      runPromptButton.disabled = false;
      runAllButton.disabled = false;
    }
    return;
  }
  const wanted = new Set(row?.candidate_skus || []);
  const unique = [];
  const seen = new Set();
  mapping.filter((item) => wanted.has(item.sku_name)).forEach((item) => {
    if (!seen.has(item.prompt)) {
      seen.add(item.prompt);
      unique.push(item);
    }
  });
  if (!unique.length) {
    setStatus("#runStatus", "本层没有候选 SKU 映射 Prompt", "error");
    return;
  }
  runAllButton.disabled = true;
  runPromptButton.disabled = true;
  document.querySelector("#results").replaceChildren();
  try {
    for (let index = 0; index < unique.length; index += 1) {
      const item = unique[index];
      setStatus("#runStatus", `正在运行 ${index + 1}/${unique.length} · ${item.prompt}`);
      const result = await runOne(item.sku_name, item.prompt);
      renderResult(result);
    }
    setStatus("#runStatus", `全部完成：${unique.length} 个唯一 Prompt`, "success");
  } catch (error) {
    setStatus("#runStatus", error.message, "error");
  } finally {
    runPromptButton.disabled = false;
    runAllButton.disabled = false;
  }
}

async function initialize() {
  try {
    const payload = await api("/api/sam-row-debug/records");
    records = payload.records || [];
    mapping = payload.prompt_mapping || [];
    const groups = [...new Set(records.map((item) => item.group))];
    groupSelect.replaceChildren();
    groups.forEach((group) => addOption(groupSelect, group, group));
    if (!records.length) throw new Error("没有可用的实测 RGB-D record");
    populateRecords();
    const suffix = (payload.errors || []).length ? ` · 跳过 ${payload.errors.length} 条缺文件记录` : "";
    setStatus("#loadStatus", `已载入 ${records.length} 条记录${suffix}`, "success");
  } catch (error) {
    setStatus("#loadStatus", error.message, "error");
  }
}

groupSelect.addEventListener("change", populateRecords);
recordSelect.addEventListener("change", populateRows);
rowSelect.addEventListener("change", renderSelection);
skuSelect.addEventListener("change", () => {
  syncSelectedPrompt();
});
runPromptButton.addEventListener("click", runCurrentPrompt);
runAllButton.addEventListener("click", runAllMappedPrompts);
initialize();
