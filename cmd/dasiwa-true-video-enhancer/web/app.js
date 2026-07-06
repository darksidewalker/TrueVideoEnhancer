const $ = (id) => document.getElementById(id);
let jobEvents = null;
let runtimeModels = [];
let sourceProbe = null;
let currentJob = null;
let livePreviewTimer = null;
const maxVisibleLogLines = 500;
// Persisted model selections: { upscaler: "", interpolation: "" } where "" means Auto.
const modelSelectionStorageVersion = "auto-dropdown-v1";
function migrateModelSelectionStorage() {
  if (localStorage.getItem("model_selection_storage_version") === modelSelectionStorageVersion) return;
  // Older builds wrote auto-selected model IDs into selected_models. Start the
  // new dropdown UI in Auto mode so those stale automatic picks do not become
  // manual overrides.
  localStorage.removeItem("selected_models");
  localStorage.setItem("model_selection_storage_version", modelSelectionStorageVersion);
}
function getSelectedModels() {
  migrateModelSelectionStorage();
  try { return JSON.parse(localStorage.getItem("selected_models") || "{}"); } catch(e) { return {}; }
}
function setSelectedModel(category, id) {
  const sel = getSelectedModels();
  sel[category] = id;
  localStorage.setItem("selected_models", JSON.stringify(sel));
  applySelectionToDropdown(category, id || "");
}
function parseModelScale(model) {
  const text = `${model.id || ""} ${model.name || ""} ${model.file || ""}`.toLowerCase();
  const match = text.match(/(?:^|[^0-9])([24])x(?:[^0-9]|$)/);
  return match ? Number(match[1]) : 0;
}

function selectedModelDestination(category) {
  const selectID = category === "upscaler" ? "upscaleModel" : "rifeModel";
  const id = $(selectID)?.value || "";
  if (!id) return "";
  const model = runtimeModels.find(function(r){ return r.id === id; });
  return model ? model.destination : id;
}

function bestUpscalerModel() {
  if (!runtimeModels.length || !$("contentType") || !$("scale")) return null;
  const contentType = $("contentType").value || "mixed";
  const requestedScale = Number($("overrideUpscaleScale").value || $("scale").value || 1);
  if (requestedScale < 2) return null;

  const upscalers = runtimeModels.filter(function(m) { return m.category === "upscaler"; });
  const sameContent = upscalers.filter(function(m) { return m.subcategory === contentType; });
  let candidates = sameContent.filter(function(m) { return parseModelScale(m) === requestedScale; });

  // Mixed content currently has no 2x model. Prefer the realism 2x model for
  // general/live-action video at 2x; otherwise fall back to the closest model
  // in the selected content bucket, then any exact-scale upscaler.
  if (!candidates.length && contentType === "mixed" && requestedScale === 2) {
    candidates = upscalers.filter(function(m) { return m.subcategory === "realism" && parseModelScale(m) === 2; });
  }
  if (!candidates.length && sameContent.length) {
    candidates = sameContent.slice().sort(function(a, b) {
      return Math.abs(parseModelScale(a) - requestedScale) - Math.abs(parseModelScale(b) - requestedScale);
    });
  }
  if (!candidates.length) {
    candidates = upscalers.filter(function(m) { return parseModelScale(m) === requestedScale; });
  }

  const selected = candidates[0] || null;
  return selected;
}

function bestInterpolationModel() {
  if (!runtimeModels.length || !$("contentType")) return null;
  const interpolation = runtimeModels.filter(function(m) { return m.category === "interpolation" && !isHeavyRifeModel(m); });
  if (!interpolation.length) return null;
  return interpolation[0];
}

function isHeavyRifeModel(model) {
  return `${model?.id || ""} ${model?.name || ""} ${model?.file || ""} ${model?.destination || ""}`.toLowerCase().includes("heavy");
}

function resolvedAutoModel(category) {
  return category === "upscaler" ? bestUpscalerModel() : bestInterpolationModel();
}

function resolvedModelDestination(category) {
  return selectedModelDestination(category) || (resolvedAutoModel(category)?.destination || "");
}

function applySelectionToDropdown(category, id) {
  const select = $(category === "upscaler" ? "upscaleModel" : "rifeModel");
  if (select) select.value = id || "";
}

function selectBestUpscalerModel() {
  updateModelAutoLabels();
  return bestUpscalerModel();
}

function applySelectionsToFields() {
  populateAdvancedModelDropdowns();
  var sel = getSelectedModels();
  applySelectionToDropdown("upscaler", sel.upscaler || "");
  applySelectionToDropdown("interpolation", sel.interpolation || "");
  selectBestUpscalerModel();
}

function populateAdvancedModelDropdowns() {
  populateModelDropdown("upscaleModel", "upscaler", "Auto (content/scale/preset)");
  populateModelDropdown("rifeModel", "interpolation", "Auto (content/preset)");
  updateModelAutoLabels();
}

function populateModelDropdown(selectID, category, autoLabel) {
  const select = $(selectID);
  if (!select) return;
  const current = select.value || "";
  select.innerHTML = "";
  
  // Add "No upscale" option first (only for upscaler)
  if (category === "upscaler") {
    const noUpscale = document.createElement("option");
    noUpscale.value = "__no_upscale__";
    noUpscale.textContent = "No upscale";
    select.appendChild(noUpscale);
  }
  
  const auto = document.createElement("option");
  auto.value = "";
  auto.textContent = autoLabel;
  select.appendChild(auto);
  runtimeModels.filter(function(m) { return m.category === category && !(category === "interpolation" && isHeavyRifeModel(m)); }).forEach(function(model) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = `${model.name}${model.present ? "" : " (missing)"}`;
    select.appendChild(option);
  });
  select.value = Array.from(select.options).some(function(option) { return option.value === current; }) ? current : "";
}

function updateModelAutoLabels() {
  const upscaler = bestUpscalerModel();
  const interpolation = bestInterpolationModel();
  const upAuto = $("upscaleModel")?.querySelector('option[value=""]');
  const rifeAuto = $("rifeModel")?.querySelector('option[value=""]');
  if (upAuto) upAuto.textContent = upscaler ? `Auto: ${upscaler.name}` : "Auto (no upscaling)";
  if (rifeAuto) rifeAuto.textContent = needsInterpolation()
    ? (interpolation ? `Auto: ${interpolation.name}` : "Auto (default interpolation)")
    : "Auto: no RIFE (target FPS <= source)";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    cache: "no-store",
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const details = Array.isArray(data.logs) && data.logs.length ? `\n${data.logs.join("\n")}` : "";
    throw new Error(data.error || `${response.statusText}${details}`);
  }
  return data;
}

async function loadSourceProbe(path) {
  const inputPath = (path || $("input")?.value || "").trim();
  if (!inputPath) {
    sourceProbe = null;
    return null;
  }
  try {
    const probe = await api(`/api/probe?path=${encodeURIComponent(inputPath)}`);
    sourceProbe = probe.error ? null : probe;
    return sourceProbe;
  } catch (_) {
    sourceProbe = null;
    return null;
  }
}

function parseRate(rate) {
  if (!rate) return 0;
  const text = String(rate);
  if (text.includes("/")) {
    const parts = text.split("/").map(Number);
    if (parts[0] > 0 && parts[1] > 0) return parts[0] / parts[1];
  }
  return Number(text) || 0;
}

function needsInterpolation() {
  const target = Number($("targetFps")?.value || 0);
  const source = parseRate(sourceProbe?.r_frame_rate);
  return target > 0 && (!source || target > source + 1e-6);
}

async function refreshRuntime() {
  try {
    const status = await api("/api/runtime/status");
    $("statusDot").classList.add("online");
    $("runtimeSummary").textContent = `${status.python?.available ? 'Python ✓' : 'Python ✗'} · ${status.nvidia_smi?.available ? 'GPU ✓' : 'GPU ✗'}`;
  } catch (error) {
    $("statusDot").classList.remove("online");
    $("runtimeSummary").textContent = error.message;
  }
}

async function runBackendCheck() {
  const resultsBox = $("checkResults");
  resultsBox.innerHTML = '<div class="hint" style="text-align:center">Running diagnostics…</div>';
  $("checkDialog").showModal();
  try {
    const result = await api("/api/runtime/check", { method: "POST", body: JSON.stringify({}) });
    renderCheckResults(result);
  } catch (error) {
    resultsBox.innerHTML = `<p class="warning">Backend check failed: ${error.message}</p>`;
  }
}

function renderCheckResults(data) {
  const box = $("checkResults");
  box.innerHTML = "";
  if (!data.items || !data.items.length) {
    box.textContent = "No checks returned.";
    return;
  }

  for (const item of data.items) {
    const row = document.createElement("div");
    row.className = "check-item";
    const iconClass = item.pass ? "ok" : "fail";
    const iconSymbol = item.pass ? "✓" : "✗";
    const text = item.detail || item.error;
    row.innerHTML = `<span class="check-icon ${iconClass}">${iconSymbol}</span><span class="check-name">${item.name}</span>${text ? `<span class="check-detail">${text}</span>` : ""}`;
    box.appendChild(row);
  }

  // Summary bar
  const summaryEl = document.createElement("div");
  const passText = `${data.passed_count || 0}/${data.total_count || data.items.length} checks passed`;
  summaryEl.className = `check-summary ${data.status === "ok" ? "ok" : "fail"}`;
  summaryEl.textContent = passText;
  box.appendChild(summaryEl);

  // Update status dot based on result
  if (data.status === "ok") {
    $("statusDot").classList.add("online");
  } else {
    $("statusDot").classList.remove("online");
  }
}

async function installRuntime() {
  $("installRuntime").disabled = true;
  log("Connecting to installation stream…\n");
  let logsCollected = [];
  try {
    await new Promise((resolve, reject) => {
      const evtSource = new EventSource("/api/runtime/install/stream");

      evtSource.addEventListener("plan", (message) => {
        const data = JSON.parse(message.data);
        log(`Install plan:\n  Root: ${data.plan.root_dir}\n  Python: ${data.plan.python}\n  uv: ${data.plan.uv}\n  Requirements: ${data.plan.requirements}`);
      });

      evtSource.addEventListener("log", (message) => {
        const data = JSON.parse(message.data);
        logsCollected.push(data.line);
        // Append to existing log instead of replacing
        $("log").textContent += "\n" + data.line;
      });

      evtSource.addEventListener("done", () => {
        log(`\nRuntime installed successfully.\n${logsCollected.join("\n")}`);
        evtSource.close();
        resolve();
      });

      evtSource.addEventListener("error", (message) => {
        const data = JSON.parse(message.data);
        log(`\nInstallation error: ${data.error}\nLogs:\n${logsCollected.join("\n")}`);
        evtSource.close();
        reject(new Error(data.error));
      });

      // Fallback: if the connection errors (e.g. network), stop waiting after 10 min
      setTimeout(() => {
        evtSource.close();
        log("Installation timed out or stream disconnected.");
        resolve();
      }, 600 * 1000);
    });

    await refreshRuntime();
  } catch (error) {
    // Error already logged inside the promise above, but handle unexpected ones.
    if (!$("log").textContent.includes(error.message)) {
      log(`Installation failed: ${error.message}`);
    }
  } finally {
    $("installRuntime").disabled = false;
  }
}

async function loadRuntimeModels() {
  if (runtimeModels.length) return;
  try {
    const result = await api("/api/runtime/models");
    runtimeModels = result.models || [];
  } catch (error) {
    log(`Model list could not be loaded: ${error.message}`);
    runtimeModels = [];
  }
}

async function refreshRuntimeModels() {
  try {
    const result = await api("/api/runtime/models");
    runtimeModels = result.models || [];
  } catch (e) {}
  renderModelList();
  applySelectionsToFields();
  $("modelsDialog").showModal();
}

function renderModelList() {
  var box = $("modelList");
  box.innerHTML = "";

  if (!runtimeModels.length) {
    box.textContent = "No model list available.";
    return;
  }

  // Group by category, preserving order: upscaler first, then interpolation
  var categories = ["upscaler", "interpolation"];
  var catLabels = { upscaler: "Upscaler (Super-Resolution)", interpolation: "Interpolation (Frame Generation)" };

  for (var ci of categories) {
    var group = runtimeModels.filter(function(m){ return m.category === ci });
    if (!group.length) continue;

    // Category header
    var hdr = document.createElement("div");
    hdr.className = "model-category-header";
    hdr.textContent = catLabels[ci] || ci;
    box.appendChild(hdr);

    // Upscaler models: split into subcategories (anime, mixed, realism)
    if (ci === "upscaler") {
      var subCategories = ["anime", "mixed", "realism"];
      var subLabels = { anime: "Anime", mixed: "Mixed Content", realism: "Realism" };

      for (var si of subCategories) {
        var subgroup = group.filter(function(m){ return m.subcategory === si });
        if (!subgroup.length) continue;

        // Subcategory header
        var subHdr = document.createElement("div");
        subHdr.className = "model-subcategory-header";
        subHdr.textContent = subLabels[si] || si;
        box.appendChild(subHdr);

        for (var mi of subgroup) {
          renderModelRow(box, mi);
        }
      }
    } else {
      // Interpolation: no subcategories, render flat
      for (var mi2 of group) {
        renderModelRow(box, mi2);
      }
    }
  }
}

function renderModelRow(box, model) {
  var row = document.createElement("div");
  var selected = getSelectedModels()[model.category] === model.id;
  row.className = "model-row" + (!model.present ? " model-missing" : "") + (selected ? " model-selected" : "");

  // Status icon: checkmark or download arrow
  var statusSpan = document.createElement("span");
  if (model.present) {
    statusSpan.className = "check-icon ok";
    statusSpan.textContent = "\u2713";
  } else {
    statusSpan.className = "model-dl-arrow";
    statusSpan.innerHTML = "<svg width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'><path d='M12 5v14M5 12l7 7 7-7'/></svg>";
  }

  // Name + file info
  var nameSpan = document.createElement("span");
  nameSpan.className = "model-name";
  nameSpan.textContent = model.name;

  var smallEl = document.createElement("small");
  smallEl.textContent = model.file;

  row.appendChild(statusSpan);
  row.appendChild(nameSpan);
  row.appendChild(smallEl);

  if (!model.present) {
    var dlBtn = document.createElement("button");
    dlBtn.type = "button";
    dlBtn.textContent = "Download";
    (function(mdl){ dlBtn.addEventListener("click", function(e){ e.stopPropagation(); downloadSingleModel(mdl); }); })(model);
    row.appendChild(dlBtn);
  }

  row.addEventListener("click", function() {
    setSelectedModel(model.category, model.id);
    if (model.category === "upscaler" && model.subcategory && $("contentType")) {
      $("contentType").value = model.subcategory;
      const scale = parseModelScale(model);
      if (scale && $("scale")) $("scale").value = String(scale);
    }
    renderModelList();
  });

  box.appendChild(row);
}

async function downloadSingleModel(model) {
  $("modelsDialog").close();
  $("downloadLog").textContent = `Downloading ${model.name}…\n`;
  $("downloadDialog").showModal();

  const encodedParams = encodeURIComponent(JSON.stringify({ models: [model.id] }));
  try {
    await new Promise((resolve, reject) => {
      const evtSource = new EventSource(`/api/models/download/stream?models=${encodedParams}`);

      evtSource.addEventListener("log", (message) => {
        const data = JSON.parse(message.data);
        $("downloadLog").textContent += data.line + "\n";
      });

      evtSource.addEventListener("done", () => {
        $("downloadLog").textContent += `\nDownload complete.\n`;
        evtSource.close();
        $("downloadDialog").close();
        resolve();
      });

      evtSource.addEventListener("error", (message) => {
        const data = JSON.parse(message.data);
        $("downloadLog").textContent += `\nError: ${data.error}\n`;
        evtSource.close();
        reject(new Error(data.error));
      });

      setTimeout(() => {
        evtSource.close();
        $("downloadLog").textContent += "Download timed out or stream disconnected.\n";
        resolve();
      }, 600 * 1000);
    });
  } catch (error) {
    if (!$("downloadLog").textContent.includes(error.message)) {
      $("downloadLog").textContent += `Failed: ${error.message}\n`;
    }
  }

  // Refresh model list so checkmarks update, then reopen models dialog
  await refreshRuntimeModels();
}

function collectJob() {
  return {
    input: $("input").value.trim(),
    output: buildOutputPath($("input").value),
    output_container: $("outputContainer").value,
    backend: $("backend").value,
    preset: document.querySelector('input[name="preset"]:checked').value,
    content_type: $("contentType").value,
    target_fps: Number($("targetFps").value || 0),
    scale: Number($("scale").value || 1),
    upscale_model: resolvedModelDestination("upscaler"),
    rife_model: resolvedModelDestination("interpolation"),
    crf: $("crf").value.trim(),
    video_encoder_preset: $("videoEncoderPreset").value,
    video_pixel_format: $("videoPixelFormat").value,
    audio_encoder_preset: $("audioEncoderPreset").value,
    subtitle_encoder_preset: $("subtitleEncoderPreset").value,
    audio_bitrate: $("audioBitrate").value.trim(),
    tile_size: Number($("tileSize").value || 0),
    tensorrt_dynamic_shapes: $("tensorrtDynamicShapes").checked,
    tensorrt_opt_profile: Number($("tensorrtOptProfile").value || 0),
    scene_detect_method: $("sceneDetectMethod").value,
    scene_detect_threshold: Number($("sceneDetectThreshold").value || 0),
    custom_encoder: $("customEncoder").value.trim(),
    override_upscale_scale: Number($("overrideUpscaleScale").value || 0),
    hdr_mode: $("hdrMode").checked,
    uhd_mode: $("uhdMode").checked,
    slomo_mode: $("sloMoMode") ? $("sloMoMode").checked : false,
    ensemble: $("ensemble").checked,
    dynamic_optical_flow: $("dynamicOpticalFlow").checked,
    benchmark: $("benchmarkMode") ? $("benchmarkMode").checked : false,
    start_time: Number($("startTime").value || 0),
    end_time: Number($("endTime").value || 0),
    device: $("deviceSelect") ? $("deviceSelect").value.trim() : "",
    pytorch_gpu_id: Number($("pytorchGpuId").value || 0),
    ncnn_gpu_id: Number($("ncnnGpuId").value || 0),
    dry_run: $("dryRun").checked,
  };
}

async function loadSupportedOptions() {
  try {
    const options = await api("/api/options");
    setSelectOptions("outputContainer", options.output_containers, "mp4");
    setSelectOptions("videoEncoderPreset", options.video_encoder_presets, "auto");
    setSelectOptions("videoPixelFormat", options.video_pixel_formats, "yuv420p");
    setSelectOptions("audioEncoderPreset", options.audio_encoder_presets, "copy_audio");
    setSelectOptions("subtitleEncoderPreset", options.subtitle_encoder_presets, "copy_subtitle");
    setSelectOptions("sceneDetectMethod", options.scene_detect_methods, "pyscenedetect");
  } catch (error) {
    log(`Options could not be loaded: ${error.message}`);
  }
}

function setSelectOptions(id, values, selected) {
  if (!Array.isArray(values)) return;
  const select = $(id);
  const current = select.value || "";
  if (id === "videoEncoderPreset" && !values.includes("auto")) {
    values = ["auto", ...values];
  }
  select.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value === "auto" ? "Auto (best for container)" : value;
    option.selected = value === (current || selected);
    select.appendChild(option);
  }
}

function splitPath(path) {
  const slash = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  const dir = slash >= 0 ? path.slice(0, slash + 1) : "";
  const file = slash >= 0 ? path.slice(slash + 1) : path;
  const dot = file.lastIndexOf(".");
  return { dir, name: dot > 0 ? file.slice(0, dot) : file };
}

function safeTag(value) {
  return String(value || "")
    .trim()
    .replace(/^.*[\\\/]/, "")
    .replace(/\.[^.]+$/, "")
    .replace(/[^\p{L}\p{N}._ -]+/gu, "-")
    .replace(/\s+/g, "_")
    .replace(/-+/g, "-")
    .replace(/^[-_.]+|[-_.]+$/g, "");
}

function selectedModelTags() {
  const tags = [];
  const rife = needsInterpolation() ? (safeTag(resolvedModelDestination("interpolation")) || "rife") : "no_rife";
  const upscale = safeTag(resolvedModelDestination("upscaler")) || `${$("contentType").value}_${$("scale").value}x`;
  if (rife) tags.push(rife);
  if (upscale) tags.push(upscale);
  return tags;
}

function audioFilenameTag() {
  const selected = $("audioEncoderPreset").value || "copy_audio";
  if (["copy", "copy_audio", ""].includes(selected)) {
    return sourceProbe?.audio_codec || "audio_copy";
  }
  if (selected === "opus") return "opus";
  if (selected === "vorbis") return "vorbis";
  return selected;
}

function outputResolutionTag() {
  const width = Number(sourceProbe?.width || 0);
  const height = Number(sourceProbe?.height || 0);
  const scale = Number($("overrideUpscaleScale").value || $("scale").value || 1);
  if (width > 0 && height > 0 && scale > 0) return `${Math.round(width * scale)}x${Math.round(height * scale)}`;
  return `${scale || 1}x`;
}

function buildOutputPath(inputPath) {
  const input = splitPath(inputPath.trim());
  if (!input.name) return "";
  const container = $("outputContainer").value || "mp4";
  const fps = `${$("targetFps").value || "source"}fps`;
  const tags = [
    ...selectedModelTags(),
    fps,
    outputResolutionTag(),
    $("audioBitrate").value || "copy",
    audioFilenameTag(),
    $("videoEncoderPreset").value || "video",
  ].map(safeTag).filter(Boolean);
  return `${input.dir}${input.name}${tags.map((tag) => `[${tag}]`).join("")}.${container}`;
}

async function startJob(event) {
  event.preventDefault();
  try {
    await loadSourceProbe($("input").value);
    const request = collectJob();
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify(request) });
    currentJob = job;
    startEncodePreview(job.id);
    log(`Job ${job.id}\nStatus: ${job.status}\nCommand: python ${job.args.join(" ")}\n`);
    attachJobEvents(job.id);
  } catch (error) {
    log(`Error: ${error.message}`);
  }
}

function attachJobEvents(id) {
  if (jobEvents) jobEvents.close();
  jobEvents = new EventSource(`/api/jobs/${id}/events`);
  let jobFinished = false;

  jobEvents.addEventListener("snapshot", (message) => renderJobEvent(JSON.parse(message.data)));
  jobEvents.addEventListener("status", (message) => renderJobEvent(JSON.parse(message.data)));
  jobEvents.addEventListener("log", (message) => renderJobEvent(JSON.parse(message.data)));
  for (const type of ["done", "error", "cancelled"]) {
    jobEvents.addEventListener(type, (message) => {
      jobFinished = true;
      renderJobEvent(JSON.parse(message.data));
      stopEncodePreview();
      jobEvents.close();
      jobEvents = null;
    });
  }
  jobEvents.onerror = () => {
    if (jobFinished) return;
    log("Log stream disconnected. Falling back to polling…");
    if (jobEvents) jobEvents.close();
    jobEvents = null;
    pollJob(id).catch((error) => log(error.message));
  };
}

function renderJobEvent(event) {
  const job = event.job;
  if (!job) return;
  currentJob = job;
  renderJobProgress(job);
  const lines = [`Job ${job.id}`, `Status: ${job.status}`, ...(job.logs || []), job.error ? `Error: ${job.error}` : ""].filter(Boolean);
  log(lines.slice(-maxVisibleLogLines).join("\n"), true);
}

async function pollJob(id) {
  const job = await api(`/api/jobs/${id}`);
  renderJobEvent({ job });
  if (!["done", "error", "cancelled"].includes(job.status)) {
    setTimeout(() => pollJob(id).catch((error) => log(error.message)), 1000);
  }
}

function log(text, keepScrolled) {
  const el = $("log");
  const wasNearBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 24;
  el.textContent = text;
  if (keepScrolled || wasNearBottom) el.scrollTop = el.scrollHeight;
}

function renderJobProgress(job) {
  const parsed = parseFrameProgress(job.logs || []);
  const etaEl = $("etaValue");
  const progressEl = $("progressValue");
  if (!etaEl || !progressEl) return;
  if (["done", "error", "cancelled"].includes(job.status)) {
    etaEl.textContent = job.status === "done" ? "Done" : job.status;
    progressEl.textContent = parsed ? `${parsed.done}/${parsed.total} frames` : `Status: ${job.status}`;
    return;
  }
  if (!parsed || !job.started_at) {
    etaEl.textContent = job.status || "Running";
    progressEl.textContent = "Waiting for frame progress…";
    return;
  }
  const started = Date.parse(job.started_at);
  const elapsed = Number.isFinite(started) ? Math.max(0, (Date.now() - started) / 1000) : 0;
  const rate = elapsed > 0 ? parsed.done / elapsed : 0;
  const remaining = rate > 0 ? Math.max(0, (parsed.total - parsed.done) / rate) : 0;
  const pct = parsed.total > 0 ? Math.min(100, Math.round((parsed.done / parsed.total) * 100)) : 0;
  etaEl.textContent = remaining > 0 ? formatDuration(remaining) : "Calculating…";
  progressEl.textContent = `${parsed.done}/${parsed.total} frames · ${pct}% · ${rate.toFixed(2)} fps`;
}

function parseFrameProgress(logs) {
  let total = 0;
  let done = 0;
  for (const line of logs) {
    let match = line.match(/output_frames=(\d+)/);
    if (match) total = Number(match[1]);
    match = line.match(/wrote\s+(\d+)\/(\d+)\s+frames/);
    if (match) {
      done = Number(match[1]);
      total = Number(match[2]);
    }
  }
  return total > 0 ? { done: Math.max(done, 0), total } : null;
}

function formatDuration(seconds) {
  seconds = Math.round(seconds);
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}


function startEncodePreview(jobID) {
  const panel = document.querySelector(".preview-panel");
  const img = $("encodePreview");
  const hint = $("previewHint");
  if (!img || !jobID) return;
  if (livePreviewTimer) clearInterval(livePreviewTimer);
  panel?.classList.add("encoding");
  img.classList.remove("ready");
  if (hint) hint.textContent = "Waiting for first encoded frame…";
  const refresh = async () => {
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(jobID)}/live-preview?t=${Date.now()}`, { cache: "no-store" });
      if (!response.ok) return;
      img.src = response.url;
      img.classList.add("ready");
      if (hint) hint.textContent = "Live preview of frames being encoded now.";
    } catch (_) {
      // Keep the previous frame visible; the next tick can recover.
    }
  };
  refresh();
  livePreviewTimer = setInterval(refresh, 1000);
}

function stopEncodePreview() {
  if (livePreviewTimer) {
    clearInterval(livePreviewTimer);
    livePreviewTimer = null;
  }
  document.querySelector(".preview-panel")?.classList.remove("encoding");
}

async function openOutputFolder() {
  try {
    const payload = currentJob?.id ? { job: currentJob.id } : { path: buildOutputPath($("input").value) };
    await api("/api/open-folder", { method: "POST", body: JSON.stringify(payload) });
  } catch (error) {
    log(`Open output folder failed: ${error.message}`, true);
  }
}

async function quitApp() {
  if (!confirm("Quit DaSiWa True Video Enhancer?")) return;
  log("Shutting down app…");
  try {
    await api("/api/quit", { method: "POST" });
  } catch (error) {
    log(`Shutdown requested, response failed: ${error.message}`);
  }
  setTimeout(() => {
    window.close();
    document.body.innerHTML = '<main class="shell"><section class="panel"><h1>App shut down</h1><p>You can now close this tab.</p></section></main>';
  }, 150);
}

// --- Event wiring (bottom of file) ---

$("checkBackendBtn").addEventListener("click", runBackendCheck);
$("installRuntime").addEventListener("click", installRuntime);
$("quitApp").addEventListener("click", quitApp);
$("jobForm").addEventListener("submit", startJob);
$("openOutputFolder").addEventListener("click", openOutputFolder);
$("input").addEventListener("change", function() { loadSourceProbe(this.value); });
$("targetFps").addEventListener("input", updateModelAutoLabels);
$("openTune").addEventListener("click", () => $("tuneDialog").showModal());
$("contentType").addEventListener("change", selectBestUpscalerModel);
$("scale").addEventListener("change", selectBestUpscalerModel);
$("overrideUpscaleScale").addEventListener("input", selectBestUpscalerModel);
$("upscaleModel").addEventListener("change", function() { setSelectedModel("upscaler", this.value || ""); updateModelAutoLabels(); });
$("rifeModel").addEventListener("change", function() { setSelectedModel("interpolation", this.value || ""); updateModelAutoLabels(); });

// File browser button -> open file picker dialog
let currentBrowsePath = "";
function getSavedBrowsePath() {
  const match = document.cookie.match(/(?:^|;\s*)last_browse_path=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
}

$("browseInput").addEventListener("click", () => {
  if (!currentBrowsePath) {
    const saved = getSavedBrowsePath();
    currentBrowsePath = saved || homeDir();
    openFileBrowser();
  } else {
    openFileBrowser();
  }
});

function setBrowsePathCookie(path) {
  const expires = new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toUTCString();
  document.cookie = `last_browse_path=${encodeURIComponent(path)}; expires=${expires}; path=/`;
}

function openFileBrowser() {
  $("fileSearchInput").value = "";
  browseDir(currentBrowsePath);
  $("fileBrowserDialog").showModal();
}

async function browseDir(path) {
  currentBrowsePath = path;
  setBrowsePathCookie(path);
  updatePathBar(path);
  try {
    const data = await api("/api/browse?path=" + encodeURIComponent(path));
    renderFileList(data.items, false);
  } catch (err) {
    $("fileList").innerHTML = `<p class="warning">Browse error: ${err.message}</p>`;
  }
}

async function searchFiles(query) {
  if (!query.trim()) {
    browseDir(currentBrowsePath);
    return;
  }
  updatePathBar("Search results");
  try {
    const data = await api("/api/search-files?q=" + encodeURIComponent(query) + "&path=" + encodeURIComponent(currentBrowsePath));
    renderFileList(data.items, true);
  } catch (err) {
    $("fileList").innerHTML = `<p class="warning">Search error: ${err.message}</p>`;
  }
}

function homeDir() {
  return window.__homeDir || "/home";
}

function updatePathBar(path) {
  const bar = $("fileBrowserPathBar");
  if (!bar) return;
  bar.innerHTML = "";
  if (path === "Search results") return;

  // Split into segments and build clickable breadcrumbs
  const segments = path.split("/").filter(Boolean);
  let accumulated = "";

  // "/" root segment
  const rootSeg = document.createElement("span");
  rootSeg.className = "file-browser-path-segment";
  rootSeg.textContent = "/";
  rootSeg.title = "Root";
  rootSeg.addEventListener("click", () => browseDir("/"));
  bar.appendChild(rootSeg);

  for (let i = 0; i < segments.length; i++) {
    accumulated += "/" + segments[i];
    const sep = document.createElement("span");
    sep.className = "file-browser-path-separator";
    sep.textContent = "/";
    bar.appendChild(sep);

    const seg = document.createElement("span");
    seg.className = "file-browser-path-segment";
    seg.textContent = segments[i];
    seg.title = accumulated;
    const target = accumulated;
    seg.addEventListener("click", () => browseDir(target));
    bar.appendChild(seg);
  }
}

function filepathDir(p) {
  return p.replace(/\/[^/]*$/, "") || "/";
}

function renderFileList(items, isSearch) {
  var box = $("fileList");
  box.innerHTML = "";
  if (!items.length) {
    box.textContent = "No files found.";
    return;
  }
  for (var i = 0; i < items.length; i++) {
    var item = items[i];
    var row = document.createElement("div");
    row.className = "file-browser-row";

    // Subtle icon placeholder (no glow emoji)
    var iconSpan = document.createElement("span");
    iconSpan.className = "file-browser-icon";
    if (item.is_dir) {
      iconSpan.textContent = "\u{1F4C1}"; // 📁 plain text
    } else {
      iconSpan.textContent = "\u{1F3AC}"; // 🎬 plain text
    }

    var nameSpan = document.createElement("span");
    nameSpan.className = "file-browser-name";
    nameSpan.textContent = item.name + (item.is_dir ? "/" : "");

    row.appendChild(iconSpan);
    row.appendChild(nameSpan);

    (function(it) {
      row.addEventListener("click", function() {
        if (it.is_dir) {
          browseDir(it.path);
        } else {
          $("input").value = it.path;
          loadSourceProbe(it.path);
          currentBrowsePath = filepathDir(it.path);
          $("fileBrowserDialog").close();
        }
      });
    })(item);

    box.appendChild(row);
  }
}

// Search input: debounce + enter key
$("fileSearchInput").addEventListener("input", function() {
  var q = this.value.trim();
  searchFiles(q);
});

// Close file browser dialog X button
$("closeFileBrowserX").addEventListener("click", () => $("fileBrowserDialog").close());
$("fileBrowserDialog").addEventListener("click", (e) => { if (e.target.id === "fileBrowserDialog") $("fileBrowserDialog").close(); });

$("closeDialogX").addEventListener("click", () => $("tuneDialog").close());
$("closeCheckX").addEventListener("click", () => $("checkDialog").close());
$("openModelsBtn").addEventListener("click", refreshRuntimeModels); // fetch fresh + open dialog
$("modelsDialog").addEventListener("click", (e) => { if (e.target.id === "closeModelsX") $("modelsDialog").close(); });

// Load models on startup (populates runtimeModels array; list renders when button is clicked)
loadRuntimeModels().then(() => applySelectionsToFields()); // auto-fill advanced settings from persisted selections
loadSupportedOptions();
refreshRuntime();
