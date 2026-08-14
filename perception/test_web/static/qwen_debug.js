const promptInput = document.querySelector("#prompt");
const samPromptInput = document.querySelector("#samPrompt");
const pasteZone = document.querySelector("#pasteZone");
const fileInput = document.querySelector("#fileInput");
const depthFileInput = document.querySelector("#depthFileInput");
const depthByteOrder = document.querySelector("#depthByteOrder");
const coordinateMode = document.querySelector("#coordinateMode");
const detectionSelect = document.querySelector("#detectionSelect");
const qwenCanvas = document.querySelector("#qwenCanvas");
const samCanvas = document.querySelector("#samCanvas");
const fullLocateCanvas = document.querySelector("#fullLocateCanvas");
const finalLocateBboxCanvas = document.querySelector("#finalLocateBboxCanvas");
const finalLocateCropCanvas = document.querySelector("#finalLocateCropCanvas");
const colors = ["#2dd4bf", "#fb7185", "#60a5fa", "#fbbf24", "#c084fc"];
const cropPaddingRatio = 0.1;
const targetImageWidth = 1280;
const targetImageHeight = 720;

let imageDataUrl = "";
let originalImageDataUrl = "";
let originalImageName = "offline_test.jpg";
let depthImageDataUrl = "";
let depthImageName = "";
let sourceImage = null;
let latestDetections = [];
let currentCropBox = null;

async function api(url, options) {
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

function setStatus(id, text, kind = "") {
  const element = document.querySelector(id);
  element.textContent = text;
  element.className = `status ${kind}`.trim();
}

function resetSam(message = "先运行 Qwen3 获取 crop") {
  currentCropBox = null;
  detectionSelect.replaceChildren();
  const option = document.createElement("option");
  option.value = "";
  option.textContent = message;
  detectionSelect.append(option);
  detectionSelect.disabled = true;
  document.querySelector("#runSam").disabled = true;
  document.querySelector("#samEmpty").hidden = false;
  document.querySelector("#samEmpty").textContent = message;
  document.querySelector("#samResult").textContent = "[]";
  samCanvas.width = 0;
  samCanvas.height = 0;
  setStatus("#samStatus", "等待 Qwen3", "");
}

async function setImage(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const originalImage = new Image();
      originalImage.src = reader.result;
      await originalImage.decode();
      originalImageDataUrl = String(reader.result || "");
      const suppliedName = String(file.name || "").trim();
      originalImageName = /\.(jpe?g|png)$/i.test(suppliedName)
        ? suppliedName
        : `offline_test.${file.type === "image/png" ? "png" : "jpg"}`;
      const originalWidth = originalImage.naturalWidth;
      const originalHeight = originalImage.naturalHeight;

      const resizeCanvas = document.createElement("canvas");
      resizeCanvas.width = targetImageWidth;
      resizeCanvas.height = targetImageHeight;
      resizeCanvas.getContext("2d").drawImage(
        originalImage,
        0,
        0,
        targetImageWidth,
        targetImageHeight,
      );
      imageDataUrl = resizeCanvas.toDataURL("image/jpeg", 0.95);
      sourceImage = new Image();
      sourceImage.src = imageDataUrl;
      await sourceImage.decode();
      latestDetections = [];
      document.querySelector("#imageStatus").textContent = `${originalWidth} × ${originalHeight} → ${targetImageWidth} × ${targetImageHeight}`;
      document.querySelector("#qwenEmpty").hidden = true;
      document.querySelector("#detections").textContent = "[]";
      drawQwenDetections();
      resetSam();
      document.querySelector("#fullLocateEmpty").hidden = false;
      document.querySelector("#fullLocateResult").textContent = "{}";
      setStatus("#fullLocateStatus", "图片已就绪，请选择商品", "success");
    } catch (error) {
      document.querySelector("#imageStatus").textContent = `图片读取失败：${error.message}`;
    }
  };
  reader.onerror = () => {
    document.querySelector("#imageStatus").textContent = "图片读取失败";
  };
  reader.readAsDataURL(file);
}

function setDepthFile(file) {
  if (!file) return;
  if (!/\.(npy|raw|bin|png|tif|tiff)$/i.test(file.name || "")) {
    depthFileInput.value = "";
    document.querySelector("#depthStatus").textContent = "只支持 NPY、16 位 PNG/TIFF 或 RAW/BIN";
    depthImageDataUrl = "";
    depthImageName = "";
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    depthImageDataUrl = String(reader.result || "");
    depthImageName = file.name;
    document.querySelector("#depthStatus").textContent =
      `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
  };
  reader.onerror = () => {
    depthImageDataUrl = "";
    depthImageName = "";
    document.querySelector("#depthStatus").textContent = "深度数据读取失败";
  };
  reader.readAsDataURL(file);
}

function detectionBoxPixels(detection, padded = false) {
  let [x1, y1, x2, y2] = detection.bbox.map(Number);
  if (coordinateMode.value === "normalized") {
    x1 = x1 / 1000 * sourceImage.naturalWidth;
    x2 = x2 / 1000 * sourceImage.naturalWidth;
    y1 = y1 / 1000 * sourceImage.naturalHeight;
    y2 = y2 / 1000 * sourceImage.naturalHeight;
  }
  [x1, x2] = [Math.min(x1, x2), Math.max(x1, x2)];
  [y1, y2] = [Math.min(y1, y2), Math.max(y1, y2)];
  if (padded) {
    const paddingX = (x2 - x1) * cropPaddingRatio;
    const paddingY = (y2 - y1) * cropPaddingRatio;
    x1 -= paddingX;
    x2 += paddingX;
    y1 -= paddingY;
    y2 += paddingY;
  }
  return [
    Math.max(0, Math.floor(x1)),
    Math.max(0, Math.floor(y1)),
    Math.min(sourceImage.naturalWidth, Math.ceil(x2)),
    Math.min(sourceImage.naturalHeight, Math.ceil(y2)),
  ];
}

function drawBox(context, bbox, label, color) {
  const [x1, y1, x2, y2] = bbox;
  context.save();
  context.strokeStyle = color;
  context.lineWidth = Math.max(3, context.canvas.width / 420);
  context.strokeRect(x1, y1, x2 - x1, y2 - y1);
  context.font = `700 ${Math.max(15, context.canvas.width / 55)}px sans-serif`;
  const padding = 6;
  const labelWidth = context.measureText(label).width + padding * 2;
  const labelHeight = Math.max(24, context.canvas.width / 38);
  const labelY = Math.max(0, y1 - labelHeight);
  context.fillStyle = color;
  context.fillRect(x1, labelY, labelWidth, labelHeight);
  context.fillStyle = "#071017";
  context.textBaseline = "middle";
  context.fillText(label, x1 + padding, labelY + labelHeight / 2);
  context.restore();
}

function drawQwenDetections() {
  if (!sourceImage) return;
  qwenCanvas.width = sourceImage.naturalWidth;
  qwenCanvas.height = sourceImage.naturalHeight;
  const context = qwenCanvas.getContext("2d");
  context.drawImage(sourceImage, 0, 0);
  latestDetections.forEach((detection, index) => {
    drawBox(context, detectionBoxPixels(detection), `#${index + 1} ${detection.name}`, "#f59e0b");
  });
}

function populateDetectionSelect() {
  detectionSelect.replaceChildren();
  latestDetections.forEach((detection, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `#${index + 1} ${detection.name} [${detection.bbox.join(", ")}]`;
    detectionSelect.append(option);
  });
  const hasDetections = latestDetections.length > 0;
  detectionSelect.disabled = !hasDetections;
  document.querySelector("#runSam").disabled = !hasDetections;
  if (!hasDetections) resetSam("Qwen3 没有返回可裁剪的 bbox");
}

function drawSelectedCrop() {
  const detection = latestDetections[Number(detectionSelect.value)];
  if (!sourceImage || !detection) throw new Error("请先选择一个 Qwen3 bbox");
  currentCropBox = detectionBoxPixels(detection, true);
  const [x1, y1, x2, y2] = currentCropBox;
  const width = x2 - x1;
  const height = y2 - y1;
  if (width < 2 || height < 2) throw new Error("Qwen3 bbox 无法生成有效 crop");
  samCanvas.width = width;
  samCanvas.height = height;
  samCanvas.getContext("2d").drawImage(sourceImage, x1, y1, width, height, 0, 0, width, height);
  document.querySelector("#samEmpty").hidden = true;
  document.querySelector("#samResult").textContent = JSON.stringify({ crop_box_original: currentCropBox }, null, 2);
  setStatus("#samStatus", `crop ${width} × ${height}`, "success");
  return { detection, cropBox: currentCropBox };
}

function maskImage(base64) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = base64.startsWith("data:") ? base64 : `data:image/png;base64,${base64}`;
  });
}

async function drawMaskOnCanvas(canvas, base64, color) {
  if (!base64) return;
  const mask = await maskImage(base64);
  const layer = document.createElement("canvas");
  layer.width = canvas.width;
  layer.height = canvas.height;
  const context = layer.getContext("2d", { willReadFrequently: true });
  context.drawImage(mask, 0, 0, layer.width, layer.height);
  const pixels = context.getImageData(0, 0, layer.width, layer.height);
  const rgb = color.match(/[a-f\d]{2}/gi).map((value) => parseInt(value, 16));
  for (let index = 0; index < pixels.data.length; index += 4) {
    const foreground = pixels.data[index];
    pixels.data[index] = rgb[0];
    pixels.data[index + 1] = rgb[1];
    pixels.data[index + 2] = rgb[2];
    pixels.data[index + 3] = foreground > 127 ? 92 : 0;
  }
  context.putImageData(pixels, 0, 0);
  canvas.getContext("2d").drawImage(layer, 0, 0);
}

async function drawMask(base64, color) {
  await drawMaskOnCanvas(samCanvas, base64, color);
}

async function loadLocateSkus() {
  const result = await api("/api/skus?task_type=SORTING");
  const options = document.querySelector("#locateSkuOptions");
  options.replaceChildren();
  (result.skus || []).forEach(({ name }) => {
    const option = document.createElement("option");
    option.value = name;
    options.append(option);
  });
}

async function runFullLocate() {
  if (!originalImageDataUrl) {
    return setStatus("#fullLocateStatus", "请先粘贴或上传图片", "error");
  }
  const productName = document.querySelector("#locateSku").value.trim();
  if (!productName) {
    return setStatus("#fullLocateStatus", "请输入目标商品名称", "error");
  }
  const button = document.querySelector("#runFullLocate");
  button.disabled = true;
  setStatus("#fullLocateStatus", "运行完整 Locate Debug…", "running");
  try {
    const result = await api("/api/locate-debug", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task_type: "SORTING",
        product_name: productName,
        level: document.querySelector("#locateLevel").value,
        hand: document.querySelector("#locateHand").value,
        image_name: originalImageName,
        image_base64: originalImageDataUrl,
        ...(depthImageDataUrl
          ? {
              depth_image_name: depthImageName,
              depth_image_base64: depthImageDataUrl,
              depth_is_bigendian: depthByteOrder.value === "big",
            }
          : {}),
      }),
    });
    const image = new Image();
    image.src = `data:${result.image_media_type || "image/jpeg"};base64,${result.image_base64}`;
    await image.decode();
    fullLocateCanvas.width = image.naturalWidth;
    fullLocateCanvas.height = image.naturalHeight;
    fullLocateCanvas.getContext("2d").drawImage(image, 0, 0);
    const instances = result.instances || [];
    for (let index = 0; index < instances.length; index += 1) {
      await drawMaskOnCanvas(
        fullLocateCanvas,
        instances[index].mask,
        instances[index].is_selected ? "#22c55e" : colors[index % colors.length],
      );
    }
    const context = fullLocateCanvas.getContext("2d");
    instances.forEach((instance, index) => {
      const mapped = instance.mapped_product_name || `#${index + 1}`;
      const group = instance.hard_case_group_index ? `G${instance.hard_case_group_index} ` : "";
      const selected = instance.is_selected ? "目标 " : "";
      drawBox(
        context,
        instance.bbox,
        `${selected}${group}${mapped}`,
        instance.is_selected ? "#22c55e" : colors[index % colors.length],
      );
    });
    document.querySelector("#fullLocateEmpty").hidden = true;
    const imageCenterX = image.naturalWidth / 2;
    const imageCenterY = image.naturalHeight / 2;
    const selectedInstance = result.selected_instance || (
      result.hard_case
        ? instances.find((instance) => instance.is_selected)
        : instances.reduce((best, instance) => {
          if (!best) return instance;
          const distance = (candidate) => {
            const centerX = (candidate.bbox[0] + candidate.bbox[2]) / 2;
            const centerY = (candidate.bbox[1] + candidate.bbox[3]) / 2;
            return (centerX - imageCenterX) ** 2 + (centerY - imageCenterY) ** 2;
          };
          return distance(instance) < distance(best) ? instance : best;
        }, null)
    );
    if (selectedInstance) {
      finalLocateBboxCanvas.width = image.naturalWidth;
      finalLocateBboxCanvas.height = image.naturalHeight;
      const bboxContext = finalLocateBboxCanvas.getContext("2d");
      bboxContext.drawImage(image, 0, 0);
      drawBox(bboxContext, selectedInstance.bbox, `最终目标 ${productName}`, "#22c55e");
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
        image,
        cropX,
        cropY,
        cropRight - cropX,
        cropBottom - cropY,
        0,
        0,
        cropRight - cropX,
        cropBottom - cropY,
      );
      drawBox(
        cropContext,
        [x1 - cropX, y1 - cropY, x2 - cropX, y2 - cropY],
        productName,
        "#22c55e",
      );
      document.querySelector("#finalLocateCropEmpty").hidden = true;
    }
    document.querySelector("#fullLocateResult").textContent = JSON.stringify(
      {
        ...result,
        image_base64: `<base64 ${result.image_base64.length} chars>`,
        raw_sam_instances: (result.raw_sam_instances || []).map((instance) => ({
          ...instance,
          mask: `<base64 ${instance.mask.length} chars>`,
        })),
        instances: instances.map((instance) => ({
          ...instance,
          mask: `<base64 ${instance.mask.length} chars>`,
        })),
      },
      null,
      2,
    );
    setStatus(
      "#fullLocateStatus",
      result.error
        ? `HTTP ${result.error_status_code || 500}: ${result.error}`
        : `完成：${instances.length} 个最终实例`,
      result.error ? "error" : "success",
    );
  } catch (error) {
    setStatus("#fullLocateStatus", error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function runQwen() {
  if (!imageDataUrl) return setStatus("#qwenStatus", "请先粘贴图片", "error");
  if (!promptInput.value.trim()) return setStatus("#qwenStatus", "请输入 Qwen3 Prompt", "error");
  const button = document.querySelector("#runQwen");
  button.disabled = true;
  setStatus("#qwenStatus", "请求中…", "running");
  try {
    const result = await api("/api/qwen-direct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: promptInput.value,
        image_base64: imageDataUrl,
        temperature: Number(document.querySelector("#temperature").value),
      }),
    });
    latestDetections = result.detections || [];
    drawQwenDetections();
    populateDetectionSelect();
    document.querySelector("#promptUsed").textContent = result.prompt_used;
    document.querySelector("#detections").textContent = JSON.stringify(latestDetections, null, 2);
    document.querySelector("#rawOutput").textContent = result.parse_error
      ? `${result.raw_output}\n\n[解析错误] ${result.parse_error}`
      : result.raw_output;
    setStatus("#qwenStatus", `完成：${latestDetections.length} 个 bbox`, "success");
    if (latestDetections.length) drawSelectedCrop();
  } catch (error) {
    setStatus("#qwenStatus", error.message, "error");
    resetSam("Qwen3 请求失败");
  } finally {
    button.disabled = false;
  }
}

async function runSam() {
  if (!samPromptInput.value.trim()) return setStatus("#samStatus", "请输入 SAM3 Prompt", "error");
  const button = document.querySelector("#runSam");
  button.disabled = true;
  try {
    const crop = drawSelectedCrop();
    const cropBase64 = samCanvas.toDataURL("image/jpeg", 0.95).split(",")[1];
    setStatus("#samStatus", "请求中…", "running");
    const result = await api("/api/sam3-crop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: samPromptInput.value,
        image_base64: cropBase64,
        crop_box_original: crop.cropBox,
      }),
    });
    const instances = result.instances || [];
    for (let index = 0; index < instances.length; index += 1) {
      await drawMask(instances[index].mask_png_base64, colors[index % colors.length]);
    }
    const context = samCanvas.getContext("2d");
    instances.forEach((instance, index) => {
      const score = Number(instance.score || 0).toFixed(3);
      drawBox(context, instance.bbox_xyxy, `#${instance.instance_id} ${score}`, colors[index % colors.length]);
    });
    document.querySelector("#samResult").textContent = JSON.stringify({
      qwen_detection: crop.detection,
      crop_box_original: result.crop_box_original,
      instances: instances.map(({ instance_id, score, bbox_xyxy, bbox_original_xyxy }) => ({
        instance_id,
        score,
        bbox_crop_xyxy: bbox_xyxy,
        bbox_original_xyxy,
      })),
    }, null, 2);
    setStatus("#samStatus", `完成：${instances.length} 个实例`, "success");
  } catch (error) {
    setStatus("#samStatus", error.message, "error");
  } finally {
    button.disabled = latestDetections.length === 0;
  }
}

function handlePaste(event) {
  if (event.defaultPrevented) return;
  const item = [...event.clipboardData.items].find((entry) => entry.type.startsWith("image/"));
  if (item) {
    event.preventDefault();
    pasteZone.textContent = "点击这里后按 Ctrl+V 粘贴图片";
    setImage(item.getAsFile());
  }
}

pasteZone.addEventListener("paste", handlePaste);
document.addEventListener("paste", handlePaste);
fileInput.addEventListener("change", () => setImage(fileInput.files[0]));
depthFileInput.addEventListener("change", () => setDepthFile(depthFileInput.files[0]));
pasteZone.addEventListener("click", () => {
  pasteZone.focus();
  const selection = window.getSelection();
  selection.removeAllRanges();
});
coordinateMode.addEventListener("change", () => {
  drawQwenDetections();
  if (latestDetections.length) drawSelectedCrop();
});
detectionSelect.addEventListener("change", drawSelectedCrop);
document.querySelector("#runQwen").addEventListener("click", runQwen);
document.querySelector("#runSam").addEventListener("click", runSam);
document.querySelector("#runFullLocate").addEventListener("click", runFullLocate);
loadLocateSkus().catch((error) => setStatus("#fullLocateStatus", error.message, "error"));
