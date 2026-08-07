const promptInput = document.querySelector("#prompt");
const pasteZone = document.querySelector("#pasteZone");
const fileInput = document.querySelector("#fileInput");
const canvas = document.querySelector("#canvas");
const context = canvas.getContext("2d");
let imageDataUrl = "";
let sourceImage = null;
let latestDetections = [];

function setImage(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const reader = new FileReader();
  reader.onload = async () => {
    imageDataUrl = reader.result;
    sourceImage = new Image();
    sourceImage.src = imageDataUrl;
    await sourceImage.decode();
    document.querySelector("#imageStatus").textContent = `${sourceImage.naturalWidth} × ${sourceImage.naturalHeight}`;
    document.querySelector("#empty").hidden = true;
    drawDetections();
  };
  reader.readAsDataURL(file);
}

function drawDetections() {
  if (!sourceImage) return;
  canvas.width = sourceImage.naturalWidth;
  canvas.height = sourceImage.naturalHeight;
  context.drawImage(sourceImage, 0, 0);
  const normalized = document.querySelector("#coordinateMode").value === "normalized";
  latestDetections.forEach((detection, index) => {
    let [x1, y1, x2, y2] = detection.bbox.map(Number);
    if (normalized) {
      x1 = x1 / 1000 * canvas.width;
      x2 = x2 / 1000 * canvas.width;
      y1 = y1 / 1000 * canvas.height;
      y2 = y2 / 1000 * canvas.height;
    }
    context.strokeStyle = "#f59e0b";
    context.lineWidth = Math.max(3, canvas.width / 400);
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `#${index + 1} ${detection.name}`;
    context.font = `700 ${Math.max(16, canvas.width / 55)}px sans-serif`;
    const width = context.measureText(label).width + 12;
    const height = Math.max(24, canvas.width / 38);
    context.fillStyle = "#f59e0b";
    context.fillRect(x1, Math.max(0, y1 - height), width, height);
    context.fillStyle = "#111827";
    context.fillText(label, x1 + 6, Math.max(18, y1 - 6));
  });
}

document.addEventListener("paste", (event) => {
  const item = [...event.clipboardData.items].find((entry) => entry.type.startsWith("image/"));
  if (item) {
    event.preventDefault();
    setImage(item.getAsFile());
  }
});
fileInput.addEventListener("change", () => setImage(fileInput.files[0]));
pasteZone.addEventListener("click", () => pasteZone.focus());
document.querySelector("#coordinateMode").addEventListener("change", drawDetections);

document.querySelector("#run").addEventListener("click", async () => {
  const status = document.querySelector("#requestStatus");
  if (!imageDataUrl) return void (status.textContent = "请先粘贴图片");
  if (!promptInput.value.trim()) return void (status.textContent = "请输入 Prompt");
  status.textContent = "请求中…";
  try {
    const response = await fetch("/api/qwen-direct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: promptInput.value,
        image_base64: imageDataUrl,
        temperature: Number(document.querySelector("#temperature").value),
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
    latestDetections = result.detections || [];
    drawDetections();
    document.querySelector("#promptUsed").textContent = result.prompt_used;
    document.querySelector("#detections").textContent = JSON.stringify(latestDetections, null, 2);
    document.querySelector("#rawOutput").textContent = result.parse_error
      ? `${result.raw_output}\n\n[解析错误] ${result.parse_error}`
      : result.raw_output;
    status.textContent = `完成：${latestDetections.length} 个 bbox`;
  } catch (error) {
    status.textContent = error.message;
  }
});
