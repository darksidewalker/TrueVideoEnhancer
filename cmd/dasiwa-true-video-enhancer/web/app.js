const $ = id => document.getElementById(id);
let jobEvents=null, runtimeModels=[], sourceProbe=null, sourceProbePath='', currentJob=null, livePreviewTimer=null;
let capabilities=null, capabilityError='', probeSerial=0, playbackMode='processed', runtimeModelsLoaded=false;
const maxVisibleLogLines=500;
const settingsKey='tve_studio_settings_v2';
let settings={...TVE.defaults};
function getSelectedModels() { return {upscaler:settings.upscaleModel,interpolation:settings.rifeModel}; }
function setSelectedModel(category,id) { settings[category==='upscaler'?'upscaleModel':'rifeModel']=id; syncSettings(); persistSettings(); }
function parseModelScale(m) { return TVE.scaleOf(m); }
function parseRate(rate) { return TVE.rate(rate); }
function needsInterpolation() { return TVE.increased(settings,sourceProbe); }
function applySelectionsToFields() { populateModels(); syncSettings(); }
function resolvedModelDestination(category) { return (category==='upscaler'?settings.spatialBackend==='ai':settings.temporalBackend==='rife')?TVE.model(category,settings,runtimeModels)?.destination || '':''; }
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

    await refreshRuntime(); await loadCapabilities(); await loadRuntimeModels(); applySelectionsToFields();
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
    runtimeModels = result.models || []; runtimeModelsLoaded=true;
  } catch (error) {
    log(`Model list could not be loaded: ${error.message}`);
    runtimeModels = [];
  }
}

async function refreshRuntimeModels() {
  try {
    const result = await api("/api/runtime/models");
    runtimeModels = result.models || []; runtimeModelsLoaded=true;
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
      if (scale && $("scale")) { settings=TVE.goal(settings,"scale",scale); $("scale").value = String(scale); }
    }
    renderModelList();
    readSettings(); syncSettings(); persistSettings();
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

function attachJobEvents(id) {
  if (jobEvents) jobEvents.close();
  jobEvents = new EventSource(`/api/jobs/${id}/events`);
  let jobFinished = false;

  jobEvents.addEventListener("snapshot", (message) => renderJobEvent(JSON.parse(message.data)));
  jobEvents.addEventListener("status", (message) => renderJobEvent(JSON.parse(message.data)));
  jobEvents.addEventListener("log", (message) => renderJobEvent(JSON.parse(message.data)));
  for (const type of ["done", "error", "cancelled"]) {
    jobEvents.addEventListener(type, (message) => {
      // EventSource also dispatches transport errors without a JSON payload.
      // Let onerror fall back to polling; only backend terminal events finish a job.
      if (typeof message.data !== "string") return;
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

  updatePlayback();
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
    currentBrowsePath=data.path; updatePathBar(data.path);
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

async function openOutputFolder() {
  try {
    const payload = currentJob?.id ? { job: currentJob.id } : { path: buildOutputPath($("input").value) };
    await api("/api/open-folder", { method: "POST", body: JSON.stringify(payload) });
  } catch (error) {
    log(`Open output folder failed: ${error.message}`, true);
  }
}

async function quitApp() {
  if (!confirm("Quit DaSiWa True Video Enhancer? Active and queued jobs will be cancelled.")) return;
  log("Shutting down app…");
  try {
    await api("/api/quit", { method: "POST" });
  } catch (error) {
    log(`Shutdown failed: ${error.message}`);
    return;
  }
  setTimeout(() => {
    window.close();
    document.body.innerHTML = '<main class="shell"><section class="panel"><h1>Shutdown requested</h1><p>The app is stopping its jobs. You can close this tab.</p></section></main>';
  }, 150);
}


function persistSettings() { try { localStorage.setItem(settingsKey,JSON.stringify(settings)); } catch (_) {} }
function restoreSettings() {
  try {
    const current=localStorage.getItem(settingsKey);
    if (current) settings=TVE.migrate(JSON.parse(current));
    else settings=TVE.migrate(JSON.parse(localStorage.getItem('tve_vfx_settings_v1')||'{}'));
    for (const key of ['tve_vfx_settings_v1','selected_models','model_selection_storage_version']) localStorage.removeItem(key);
  } catch (_) { settings={...TVE.defaults}; }
}
function readSettings() {
  for (const key of Object.keys(TVE.defaults)) {
    const el=$(key);
    if (!el || el.tagName==='BUTTON' || (!runtimeModelsLoaded && ['upscaleModel','rifeModel'].includes(key))) continue;
    settings[key]=el.type==='checkbox'?el.checked:el.value;
  }
}
function show(id,visible) { $(id).hidden=!visible; }
function pressed(id,value,disabled=false) {
  $(id).setAttribute('aria-pressed',String(!!value)); $(id).disabled=disabled;
}
function populateModels() {
  for (const [id,category] of [['upscaleModel','upscaler'],['rifeModel','interpolation']]) {
    $(id).replaceChildren(new Option('Auto · '+(category==='upscaler'?'content / scale':'general RIFE'),''));
    for (const m of runtimeModels.filter(m=>m.category===category && m.present && !/heavy/i.test(`${m.id} ${m.name}`))) $(id).add(new Option(m.name,m.id));
    if (runtimeModelsLoaded && !runtimeModels.some(m=>m.id===settings[id] && m.present)) settings[id]='';
  }
}
function fillOptions(id,values,label=v=>v) {
  const select=$(id), old=select.value;
  select.replaceChildren(...values.map(v=>new Option(label(v),v)));
  select.value=values.includes(old)?old:values[0]||'';
}
function syncSettings() {
  settings=TVE.normalize(settings,sourceProbe);
  const ai=settings.spatialBackend==='ai'||settings.temporalBackend==='rife', sdk=TVE.hasSDK(settings), v=capabilities?.vfx || {};
  for (const key of Object.keys(TVE.defaults)) {
    const el=$(key); if (!el || el.tagName==='BUTTON') continue;
    if (el.type==='checkbox') el.checked=!!settings[key]; else el.value=settings[key];
  }
  for (const option of $('outputContainer').options) option.disabled=!capabilities?.containers?.includes(option.value) || !TVE.codecChoices({...settings,outputContainer:option.value},capabilities).length;
  const choices=TVE.codecChoices(settings,capabilities);
  fillOptions('videoEncoderPreset',choices,v=>({auto:'Auto',av1_nvenc:'AV1 · NVENC',x265_nvenc:'HEVC · NVENC',x264_nvenc:'H.264 · NVENC',av1:'AV1 · CPU',libx265:'HEVC · CPU',libx264:'H.264 · CPU',vp9:'VP9',prores:'ProRes',ffv1:'FFV1'}[v]||v));
  if (choices.length && !choices.includes(settings.videoEncoderPreset)) settings.videoEncoderPreset=choices.includes('auto')?'auto':choices[0];
  $('videoEncoderPreset').value=settings.videoEncoderPreset;
  for (const id of ['denoiseQuality','deblurQuality','rtxUpscaleQuality']) {
    for (const option of $(id).options) option.disabled=sdk && !v.quality_levels?.some(level=>level.toLowerCase()===option.value.toLowerCase());
  }
  for (const option of $('rtxUpscale').options) option.disabled=option.value.startsWith('Streaming') && !v.streaming?.includes(option.value);
  $('rtxImageEncoding').options[1].disabled=!v.encoding || (settings.temporalBackend==='nvidia' && !v.frame_generation_encoding);
  const selected={aiUpscale:settings.spatialBackend==='ai',nvidiaUpscale:settings.spatialBackend==='nvidia',rifeInterpolation:settings.temporalBackend==='rife',nvidiaFrameGeneration:settings.temporalBackend==='nvidia',denoise:settings.denoise,deblur:settings.deblur,rtxHDR:settings.rtxHDR};
  for (const [id,on] of Object.entries(selected)) {
    const reason=TVE.featureUnavailable(id,settings,sourceProbe,capabilities,runtimeModels);
    // Unavailable selected features stay removable; never silently change the graph.
    pressed(id,on,!!reason && !on); $(id).title=reason;
  }
  show('aiControls',ai); show('nativeControls',sdk);
  show('upscaleModelControls',settings.spatialBackend==='ai'); show('rifeControls',settings.temporalBackend==='rife');
  $('tileSize').closest('label').hidden=settings.spatialBackend!=='ai';
  show('denoiseControls',settings.denoise); show('deblurControls',settings.deblur);
  for (const id of ['rtxDenoiseStrength','rtxDeblurStrength','rtxUpscaleStrength']) $(id).closest('label').hidden=!v.strength;
  show('sdkUpscaleControls',settings.spatialBackend==='nvidia'); show('fgControls',settings.temporalBackend==='nvidia');
  show('downsampleControls',settings.spatialBackend==='nvidia' && Number(settings.rtxUpscaleIntermediate)>0);
  show('hdrControls',settings.rtxHDR); show('precisionNote',settings.rtxHDR||(sdk&&settings.rtxImageEncoding==='rgb10a2'));
  $('contentType').closest('label').hidden=!ai;
  $('settingsTitle').textContent='Mix & match';
  $('pipelinePlan').textContent=TVE.stages(settings,sourceProbe,runtimeModels).join(' → ');
  for (const [id,category] of [['upscaleModel','upscaler'],['rifeModel','interpolation']]) {
    const m=TVE.model(category,{...settings,[id]:''},runtimeModels);
    if ($(id).options[0]) $(id).options[0].textContent=m?`Auto · ${m.name}`:'Auto · matching model not installed';
  }
  const seconds=TVE.previewSeconds(settings,sourceProbe);
  $('previewSeconds').max=String(Math.min(30,Math.max(1,Number(sourceProbe?.duration)||30)));
  $('previewSeconds').value=String(seconds); settings.previewSeconds=seconds;
  $('previewSeconds').disabled=!settings.preview; $('previewSecondsValue').textContent=`${seconds.toFixed(1).replace(/\.0$/,'')} s`;
  const errors=TVE.issues(settings,sourceProbePath===$('input').value.trim()?sourceProbe:null,capabilities,runtimeModels);
  if (!$('input').value.trim()) errors.unshift('Select an input video.');
  $('runJob').disabled=errors.length>0;
  $('runJob').textContent=settings.preview?'▶ Run preview':'▶ Run full video';
  $('requestStatus').textContent=errors.length?errors.join(' '):settings.preview?`Process the first ${seconds} seconds with these settings.`:'Ready · entire video, no duration limit.';
  $('requestStatus').classList.toggle('warning',errors.length>0);
  const fps=TVE.sourceFPS(sourceProbe), w=Number(sourceProbe?.width)||0,h=Number(sourceProbe?.height)||0;
  $('sourceDetails').textContent=sourceProbe?`${w} × ${h} · ${fps.toFixed(3).replace(/0+$/,'').replace(/\.$/,'')} fps · ${Number(sourceProbe.duration||0).toFixed(1)} s · ${sourceProbe.codec_name||'video'}`:'Select a readable video to inspect source FPS.';
}
async function loadCapabilities() {
  try { capabilities=await api('/api/capabilities'); capabilityError=''; }
  catch (error) { capabilities=null; capabilityError=error.message; }
  syncSettings();
  if (capabilityError) $('requestStatus').textContent=`Capabilities unavailable: ${capabilityError}. Retry ↻ before running.`;
}
async function probeForRequest(path) {
  const probe=await api(`/api/probe?path=${encodeURIComponent(path)}`);
  if (probe.error || !(TVE.sourceFPS(probe)>0)) throw new Error(probe.error || 'Could not determine source FPS.');
  return probe;
}
async function loadSourceProbe(path) {
  const serial=++probeSerial, input=(path||'').trim();
  sourceProbe=null; sourceProbePath=''; syncSettings(); updatePlayback();
  if (!input) return null;
  try {
    const probe=await probeForRequest(input);
    if (serial!==probeSerial || $('input').value.trim()!==input) return null;
    sourceProbe=probe; sourceProbePath=input; syncSettings(); updatePlayback(); return probe;
  } catch (error) { if (serial===probeSerial) { syncSettings(); $('sourceDetails').textContent=error.message; } return null; }
}
function captureSettings() { readSettings(); return {...settings}; }
function captureJobRequest(snapshot=captureSettings(),probe=sourceProbe,input=$('input').value.trim()) {
  return TVE.request(snapshot,probe,capabilities,runtimeModels,input);
}
function buildOutputPath(input) { return TVE.outputPath({input,output_container:settings.outputContainer}); }
function updatePlayback() {
  const job=currentJob, done=job?.status==='done', running=job?.status==='running';
  $('processedPlayback').disabled=!(done||running);
  pressed('sourcePlayback',playbackMode==='source'); pressed('processedPlayback',playbackMode==='processed',!(done||running));
  const video=$('videoPreview'), img=$('encodePreview');
  let url='';
  if (playbackMode==='processed') {
    if (done) url=`/api/jobs/${encodeURIComponent(job.id)}/preview?quality=native-v2`;
  } else if ($('input').value.trim()) url=`/api/stream?path=${encodeURIComponent($('input').value.trim())}`;
  const liveFrame=playbackMode==='processed' && running && img.dataset.previewJobId===job.id;
  video.hidden=!url; img.hidden=!liveFrame;
  $('placeholderFrame').hidden=!!url || liveFrame;
  $('placeholderFrame').textContent=playbackMode==='processed' && running?'Waiting for processed frames…':'Select a video to preview.';
  if (url && video.getAttribute('src')!==url) { video.src=url; video.load(); }
  if (!url) { video.pause(); video.removeAttribute('src'); }
  $('playbackStatus').textContent=playbackMode==='processed'?(done?'Processed · native resolution/FPS · first 20 s · SDR browser proxy (HDR tone-mapped).':'Live processed frames · native resolution · JPEG preview before video encoding.'):'Source · browser playback depends on its codec.';
}
function startEncodePreview(jobID) {
  stopEncodePreview();
  const refresh=()=>{ if (currentJob?.id===jobID && playbackMode==='processed' && currentJob.status==='running') $('encodePreview').src=`/api/jobs/${encodeURIComponent(jobID)}/live-preview?t=${Date.now()}`; };
  refresh(); livePreviewTimer=setInterval(refresh,1000);
}
function stopEncodePreview() { if (livePreviewTimer) clearInterval(livePreviewTimer); livePreviewTimer=null; }
function wireStudio() {
  restoreSettings(); populateModels(); syncSettings();
  for (const key of Object.keys(TVE.defaults)) {
    const el=$(key); if (!el || el.tagName==='BUTTON') continue;
    el.addEventListener('input',()=>{
      readSettings();
      if (key==='scale' || key==='targetFps') settings=TVE.goal(settings,key,settings[key]);
      syncSettings(); persistSettings();
    });
  }
  for (const id of ['aiUpscale','rifeInterpolation','nvidiaUpscale','nvidiaFrameGeneration','denoise','deblur','rtxHDR']) $(id).addEventListener('click',()=>{
    settings=TVE.toggle(settings,id,sourceProbe); syncSettings(); persistSettings();
  });
  $('retryCapabilities').addEventListener('click',loadCapabilities);
  $('input').addEventListener('input',()=>{sourceProbe=null; sourceProbePath=''; syncSettings();});
  $('input').addEventListener('change',()=>loadSourceProbe($('input').value));
  for (const mode of ['source','processed']) $(mode+'Playback').addEventListener('click',()=>{playbackMode=mode;updatePlayback();});
  $('videoPreview').addEventListener('error',()=>{$('playbackStatus').textContent='This video cannot play in this browser. The file remains available in the output folder.';});
  $('encodePreview').addEventListener('load',()=>{
    const image=$('encodePreview');
    const expected=`/api/jobs/${encodeURIComponent(currentJob?.id||'')}/live-preview`;
    if (image.naturalWidth>0 && new URL(image.currentSrc,location.href).pathname===expected) image.dataset.previewJobId=currentJob.id;
    updatePlayback();
  });
  $('encodePreview').addEventListener('error',()=>{delete $('encodePreview').dataset.previewJobId;updatePlayback();});
  $('browseInput').addEventListener('click',()=>{currentBrowsePath=currentBrowsePath||getSavedBrowsePath()||homeDir();openFileBrowser();});
  $('fileSearchInput').addEventListener('input',()=>searchFiles($('fileSearchInput').value));
  $('closeFileBrowserX').addEventListener('click',()=>$('fileBrowserDialog').close());
  $('checkBackendBtn').addEventListener('click',runBackendCheck); $('installRuntime').addEventListener('click',installRuntime);
  $('quitApp').addEventListener('click',quitApp); $('openOutputFolder').addEventListener('click',openOutputFolder);
  $('closeCheckX').addEventListener('click',()=>$('checkDialog').close());
  $('openModelsBtn').addEventListener('click',refreshRuntimeModels); $('closeModelsX').addEventListener('click',()=>$('modelsDialog').close());
  loadRuntimeModels().then(applySelectionsToFields); loadCapabilities(); refreshRuntime();
}
wireStudio();
