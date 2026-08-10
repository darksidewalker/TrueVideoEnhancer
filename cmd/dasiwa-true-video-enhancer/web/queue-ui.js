(() => {
  "use strict";

  // DaSiWa True Video Enhancer - FIFO Queue UI companion
  // Requires the FIFO worker manager.go modification.
  // Keeps submitted job IDs in localStorage so the UI can restore the queue
  // after a browser refresh while the Go server is still running.

  const STORAGE_KEY = "dasiwa_fifo_queue_jobs_v1";
  const MAX_REMEMBERED_JOBS = 100;
  const POLL_INTERVAL_MS = 1000;
  const TERMINAL = new Set(["done", "error", "cancelled"]);

  const queueJobs = new Map();
  let queueJobIDs = loadQueueIDs();
  let queueSelectedJobID = null;
  let queueRefreshTimer = null;
  let queueRefreshBusy = false;
  let queuePreviewJobID = null;
  let queueSelectionPinned = false;

  const legacyStartJob = startJob;
  const legacyRenderJobEvent = renderJobEvent;

  function loadQueueIDs() {
    try {
      const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      if (!Array.isArray(raw)) return [];
      return raw.filter((id) => typeof id === "string" && id.length > 0).slice(-MAX_REMEMBERED_JOBS);
    } catch (_) {
      return [];
    }
  }

  function saveQueueIDs() {
    queueJobIDs = queueJobIDs.slice(-MAX_REMEMBERED_JOBS);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(queueJobIDs));
  }

  function rememberJob(job) {
    if (!job || !job.id) return;
    queueJobs.set(job.id, job);
    if (!queueJobIDs.includes(job.id)) {
      queueJobIDs.push(job.id);
      saveQueueIDs();
    }
  }

  function forgetJob(id) {
    queueJobs.delete(id);
    queueJobIDs = queueJobIDs.filter((jobID) => jobID !== id);
    saveQueueIDs();
  }

  function isTerminal(job) {
    return !!job && TERMINAL.has(job.status);
  }

  function basename(path) {
    if (!path) return "(unknown input)";
    const parts = String(path).split(/[\\/]/);
    return parts[parts.length - 1] || path;
  }

  function shortID(id) {
    return String(id || "").slice(0, 8);
  }

  function progressFor(job) {
    if (!job) return null;

    // Prefer backend-provided progress summaries. The list endpoint returns these
    // without shipping every job's full log on each refresh.
    if (Number(job.progress_total) > 0) {
      return {
        done: Math.max(0, Number(job.progress_done) || 0),
        total: Number(job.progress_total),
        throughput: Math.max(0, Number(job.progress_throughput) || 0),
      };
    }

    const logs = job.logs || [];

    // Current backend format:
    // [PIPE] processed 180/302 frames throughput=0.15 fps
    for (let i = logs.length - 1; i >= 0; i--) {
      const match = String(logs[i]).match(
        /\bprocessed\s+(\d+)\/(\d+)\s+frames(?:\s+throughput=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)\s*fps)?/i
      );
      if (match) {
        return {
          done: Number(match[1]),
          total: Number(match[2]),
          throughput: Number(match[3]) || 0,
        };
      }
    }

    if (typeof parseFrameProgress === "function") {
      const parsed = parseFrameProgress(logs);
      if (parsed) return { ...parsed, throughput: 0 };
    }
    return null;
  }

  function progressPercent(job) {
    const parsed = progressFor(job);
    if (!parsed || !parsed.total) return 0;
    return Math.max(0, Math.min(100, Math.round((parsed.done / parsed.total) * 100)));
  }

  function etaFor(job) {
    if (!job || job.status !== "running") return "";
    const parsed = progressFor(job);
    if (!parsed || !parsed.total || !job.started_at) return "Calculating…";

    const started = Date.parse(job.started_at);
    if (!Number.isFinite(started)) return "Calculating…";

    const elapsed = Math.max(0, (Date.now() - started) / 1000);
    const measuredThroughput = Number(parsed.throughput) || 0;
    const rate = measuredThroughput > 0
      ? measuredThroughput
      : (elapsed > 0 ? parsed.done / elapsed : 0);
    if (rate <= 0) return "Calculating…";

    const remaining = Math.max(0, (parsed.total - parsed.done) / rate);
    return typeof formatDuration === "function"
      ? formatDuration(remaining)
      : `${Math.ceil(remaining)}s`;
  }

  function queuePosition(id) {
    const waiting = queueJobIDs.filter((jobID) => queueJobs.get(jobID)?.status === "queued");
    const pos = waiting.indexOf(id);
    return pos >= 0 ? pos + 1 : 0;
  }

  function statusLabel(job) {
    if (!job) return "UNKNOWN";
    switch (job.status) {
      case "running":
        return `RUNNING ${progressPercent(job)}%`;
      case "queued": {
        const pos = queuePosition(job.id);
        return pos ? `QUEUED #${pos}` : "QUEUED";
      }
      case "done":
        return "DONE";
      case "error":
        return "ERROR";
      case "cancelled":
        return "CANCELLED";
      default:
        return String(job.status || "UNKNOWN").toUpperCase();
    }
  }

  function injectQueueStyles() {
    if (document.getElementById("dasiwaQueueStyles")) return;
    const style = document.createElement("style");
    style.id = "dasiwaQueueStyles";
    style.textContent = `
      .queue-ui-toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin: 4px 0 12px;
        padding: 10px 12px;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: rgba(255,255,255,.025);
      }
      .queue-ui-summary {
        color: var(--muted);
        font-size: 13px;
        font-weight: 600;
      }
      .queue-ui-actions {
        display: flex;
        gap: 8px;
      }
      .queue-ui-actions button {
        width: auto;
        padding: 6px 10px;
        font-size: 12px;
      }
      .queue-ui-list {
        display: grid;
        gap: 8px;
        margin-bottom: 14px;
      }
      .queue-ui-empty {
        padding: 18px 14px;
        border: 1px dashed var(--line);
        border-radius: 12px;
        color: var(--muted);
        text-align: center;
      }
      .queue-ui-group-title {
        margin: 8px 2px 4px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .11em;
        text-transform: uppercase;
      }
      .queue-ui-job {
        display: grid;
        grid-template-columns: 12px minmax(0, 1fr) auto;
        gap: 10px;
        align-items: center;
        padding: 11px 12px;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: var(--panel-2);
        cursor: pointer;
        transition: border-color .12s ease, background .12s ease, transform .12s ease;
      }
      .queue-ui-job:hover {
        border-color: rgba(99,243,255,.45);
        background: color-mix(in srgb, var(--panel-2) 91%, var(--cyan) 9%);
      }
      .queue-ui-job.selected {
        border-color: var(--cyan);
        box-shadow: 0 0 0 1px rgba(99,243,255,.12);
      }
      .queue-ui-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: var(--muted);
        box-shadow: 0 0 0 3px rgba(169,191,211,.08);
      }
      .queue-ui-job.running .queue-ui-dot {
        background: var(--cyan);
        box-shadow: 0 0 10px rgba(99,243,255,.65);
        animation: queuePulse 1.2s ease-in-out infinite alternate;
      }
      .queue-ui-job.queued .queue-ui-dot { background: var(--amber); }
      .queue-ui-job.done .queue-ui-dot { background: var(--green); }
      .queue-ui-job.error .queue-ui-dot,
      .queue-ui-job.cancelled .queue-ui-dot { background: var(--rose); }
      .queue-ui-main { min-width: 0; }
      .queue-ui-name {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        color: var(--text);
        font-weight: 700;
      }
      .queue-ui-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 6px 10px;
        margin-top: 2px;
        color: var(--muted);
        font-size: 11px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      }
      .queue-ui-progress {
        height: 5px;
        margin-top: 8px;
        overflow: hidden;
        border-radius: 999px;
        background: rgba(255,255,255,.06);
      }
      .queue-ui-progress > span {
        display: block;
        height: 100%;
        width: 0;
        border-radius: inherit;
        background: linear-gradient(90deg, var(--cyan), #8c7dff);
        transition: width .25s ease;
      }
      .queue-ui-side {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .queue-ui-status {
        min-width: 82px;
        text-align: right;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .04em;
        color: var(--muted);
      }
      .queue-ui-job.running .queue-ui-status { color: var(--cyan); }
      .queue-ui-job.queued .queue-ui-status { color: var(--amber); }
      .queue-ui-job.done .queue-ui-status { color: var(--green); }
      .queue-ui-job.error .queue-ui-status,
      .queue-ui-job.cancelled .queue-ui-status { color: var(--rose); }
      .queue-ui-cancel {
        width: auto !important;
        padding: 4px 8px !important;
        border-color: rgba(255,95,131,.4) !important;
        color: var(--rose) !important;
        background: rgba(255,95,131,.07) !important;
        font-size: 11px !important;
      }
      .queue-ui-log-title {
        margin: 12px 0 6px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .1em;
        text-transform: uppercase;
      }
      .queue-ui-header-count {
        margin-left: auto;
        margin-right: 8px;
        padding: 2px 8px;
        border: 1px solid var(--line);
        border-radius: 999px;
        color: var(--muted);
        font-size: 11px;
        white-space: nowrap;
      }
      @keyframes queuePulse {
        from { opacity: .55; transform: scale(.9); }
        to { opacity: 1; transform: scale(1.12); }
      }
      @media (max-width: 720px) {
        .queue-ui-job { grid-template-columns: 10px minmax(0, 1fr); }
        .queue-ui-side { grid-column: 2; justify-content: space-between; }
        .queue-ui-status { text-align: left; }
        .queue-ui-toolbar { align-items: flex-start; flex-direction: column; }
      }
    `;
    document.head.appendChild(style);
  }

  function injectQueuePanel() {
    const panel = document.querySelector(".jobs-panel");
    const logEl = $("log");
    if (!panel || !logEl || $("queueUiList")) return;

    const panelHead = panel.querySelector(".panel-head");
    if (panelHead) {
      const badge = document.createElement("span");
      badge.id = "queueUiHeaderCount";
      badge.className = "queue-ui-header-count";
      badge.textContent = "0 jobs";
      const openButton = $("openOutputFolder");
      if (openButton) panelHead.insertBefore(badge, openButton);
      else panelHead.appendChild(badge);
    }

    const toolbar = document.createElement("div");
    toolbar.className = "queue-ui-toolbar";

    const summary = document.createElement("div");
    summary.id = "queueUiSummary";
    summary.className = "queue-ui-summary";
    summary.textContent = "Queue is empty.";

    const actions = document.createElement("div");
    actions.className = "queue-ui-actions";

    const refreshButton = document.createElement("button");
    refreshButton.type = "button";
    refreshButton.textContent = "Refresh";
    refreshButton.addEventListener("click", () => refreshQueueJobs(true));

    const clearButton = document.createElement("button");
    clearButton.type = "button";
    clearButton.textContent = "Clear finished";
    clearButton.addEventListener("click", clearFinishedJobs);

    actions.append(refreshButton, clearButton);
    toolbar.append(summary, actions);

    const list = document.createElement("div");
    list.id = "queueUiList";
    list.className = "queue-ui-list";

    const logTitle = document.createElement("div");
    logTitle.className = "queue-ui-log-title";
    logTitle.textContent = "Selected job log";

    panel.insertBefore(toolbar, logEl);
    panel.insertBefore(list, logEl);
    panel.insertBefore(logTitle, logEl);
  }

  function updateSummary() {
    const jobs = queueJobIDs.map((id) => queueJobs.get(id)).filter(Boolean);
    const running = jobs.filter((job) => job.status === "running").length;
    const queued = jobs.filter((job) => job.status === "queued").length;
    const done = jobs.filter((job) => job.status === "done").length;
    const errors = jobs.filter((job) => job.status === "error").length;
    const cancelled = jobs.filter((job) => job.status === "cancelled").length;

    const parts = [];
    if (running) parts.push(`${running} running`);
    if (queued) parts.push(`${queued} waiting`);
    if (done) parts.push(`${done} done`);
    if (errors) parts.push(`${errors} error`);
    if (cancelled) parts.push(`${cancelled} cancelled`);

    const summary = $("queueUiSummary");
    if (summary) summary.textContent = parts.length ? parts.join(" · ") : "Queue is empty.";

    const badge = $("queueUiHeaderCount");
    if (badge) {
      const active = running + queued;
      badge.textContent = active ? `${active} active / ${jobs.length} total` : `${jobs.length} jobs`;
    }
  }

  function createJobRow(job) {
    const row = document.createElement("div");
    row.className = `queue-ui-job ${job.status || "unknown"}${job.id === queueSelectedJobID ? " selected" : ""}`;
    row.dataset.jobId = job.id;
    row.title = job.output || job.input || job.id;
    row.addEventListener("click", () => selectQueueJob(job.id, true));

    const dot = document.createElement("span");
    dot.className = "queue-ui-dot";

    const main = document.createElement("div");
    main.className = "queue-ui-main";

    const name = document.createElement("div");
    name.className = "queue-ui-name";
    name.textContent = basename(job.input);

    const meta = document.createElement("div");
    meta.className = "queue-ui-meta";

    const idMeta = document.createElement("span");
    idMeta.textContent = `job ${shortID(job.id)}`;
    meta.appendChild(idMeta);

    if (job.status === "running") {
      const parsed = progressFor(job);
      if (parsed) {
        const frames = document.createElement("span");
        frames.textContent = `${parsed.done}/${parsed.total} frames`;
        meta.appendChild(frames);
      }
      const eta = etaFor(job);
      if (eta) {
        const etaMeta = document.createElement("span");
        etaMeta.textContent = `ETA ${eta}`;
        meta.appendChild(etaMeta);
      }
    } else if (job.status === "queued") {
      const pos = queuePosition(job.id);
      if (pos) {
        const waiting = document.createElement("span");
        waiting.textContent = `waiting position ${pos}`;
        meta.appendChild(waiting);
      }
    } else if (job.output) {
      const output = document.createElement("span");
      output.textContent = basename(job.output);
      meta.appendChild(output);
    }

    main.append(name, meta);

    if (job.status === "running") {
      const progress = document.createElement("div");
      progress.className = "queue-ui-progress";
      const fill = document.createElement("span");
      fill.style.width = `${progressPercent(job)}%`;
      progress.appendChild(fill);
      main.appendChild(progress);
    }

    if (job.error) {
      const err = document.createElement("div");
      err.className = "queue-ui-meta";
      err.style.color = "var(--rose)";
      err.textContent = job.error;
      main.appendChild(err);
    }

    const side = document.createElement("div");
    side.className = "queue-ui-side";

    const status = document.createElement("div");
    status.className = "queue-ui-status";
    status.textContent = statusLabel(job);
    side.appendChild(status);

    if (job.status === "running" || job.status === "queued") {
      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.className = "queue-ui-cancel";
      cancel.textContent = "Cancel";
      cancel.title = "Cancel this job";
      cancel.addEventListener("click", async (event) => {
        event.stopPropagation();
        await cancelQueueJob(job.id);
      });
      side.appendChild(cancel);
    }

    row.append(dot, main, side);
    return row;
  }

  function renderQueue() {
    const list = $("queueUiList");
    if (!list) return;
    list.innerHTML = "";

    const jobs = queueJobIDs.map((id) => queueJobs.get(id)).filter(Boolean);
    const active = jobs.filter((job) => !isTerminal(job));
    const finished = jobs.filter((job) => isTerminal(job)).reverse();

    if (!active.length && !finished.length) {
      const empty = document.createElement("div");
      empty.className = "queue-ui-empty";
      empty.textContent = "No queued jobs yet. Press Run several times with different input videos.";
      list.appendChild(empty);
      updateSummary();
      return;
    }

    if (active.length) {
      const title = document.createElement("div");
      title.className = "queue-ui-group-title";
      title.textContent = "Active queue";
      list.appendChild(title);
      active.forEach((job) => list.appendChild(createJobRow(job)));
    }

    if (finished.length) {
      const title = document.createElement("div");
      title.className = "queue-ui-group-title";
      title.textContent = "Completed / stopped";
      list.appendChild(title);
      finished.slice(0, 20).forEach((job) => list.appendChild(createJobRow(job)));
    }

    updateSummary();
  }

  function captureJobRequest() {
    const input = $("input").value.trim();
    return {
      input,
      output: "",
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

  async function probeForRequest(inputPath) {
    if (!inputPath) return null;
    try {
      const probe = await api(`/api/probe?path=${encodeURIComponent(inputPath)}`);
      return probe && !probe.error ? probe : null;
    } catch (_) {
      return null;
    }
  }

  function buildSnapshotOutputPath(request, probe) {
    const input = splitPath(request.input || "");
    if (!input.name) return "";

    const sourceFps = typeof parseRate === "function" ? parseRate(probe?.r_frame_rate) : 0;
    const needsRife = request.target_fps > 0
      && (!sourceFps || request.target_fps > sourceFps + 1e-6);

    const rifeTag = needsRife
      ? (safeTag(request.rife_model) || "rife")
      : "no_rife";
    const upscaleTag = safeTag(request.upscale_model)
      || `${request.content_type || "mixed"}_${request.scale || 1}x`;

    const scale = Number(request.override_upscale_scale || request.scale || 1);
    const width = Number(probe?.width || 0);
    const height = Number(probe?.height || 0);
    const resolutionTag = width > 0 && height > 0 && scale > 0
      ? `${Math.round(width * scale)}x${Math.round(height * scale)}`
      : `${scale || 1}x`;

    const audioPreset = request.audio_encoder_preset || "copy_audio";
    let audioTag;
    if (["copy", "copy_audio", ""].includes(audioPreset)) {
      audioTag = probe?.audio_codec || "audio_copy";
    } else if (audioPreset === "opus") {
      audioTag = "opus";
    } else if (audioPreset === "vorbis") {
      audioTag = "vorbis";
    } else {
      audioTag = audioPreset;
    }

    const videoCodecTag = request.video_encoder_preset || "auto";

    const tags = [
      rifeTag,
      upscaleTag,
      `${request.target_fps || "source"}fps`,
      resolutionTag,
      request.audio_bitrate || "copy",
      audioTag,
      videoCodecTag,
    ].map(safeTag).filter(Boolean);

    const container = request.output_container || "mp4";
    return `${input.dir}${input.name}${tags.map((tag) => `[${tag}]`).join("")}.${container}`;
  }

  function syncSelectedPreview(job) {
    if (!job || job.id !== queueSelectedJobID) return;

    if (job.status === "running") {
      if (queuePreviewJobID !== job.id) {
        queuePreviewJobID = job.id;
        startEncodePreview(job.id);
      }
      return;
    }

    if (queuePreviewJobID === job.id || queuePreviewJobID !== null) {
      stopEncodePreview();
      queuePreviewJobID = null;
    }
  }

  async function selectQueueJob(id, manualSelection = false) {
    let job = queueJobs.get(id);

    try {
      job = await api(`/api/jobs/${encodeURIComponent(id)}`);
      rememberJob(job);
    } catch (error) {
      // A stale localStorage ID usually means the app/server was restarted.
      if (/not found/i.test(error.message || "")) {
        forgetJob(id);
        if (queueSelectedJobID === id) {
          queueSelectedJobID = null;
          queueSelectionPinned = false;
        }
        renderQueue();
        return;
      }
      log(`Could not open job ${id}: ${error.message}`, true);
      return;
    }

    queueSelectedJobID = id;
    queueSelectionPinned = manualSelection;
    legacyRenderJobEvent({ job });
    syncSelectedPreview(job);
    renderQueue();

    if (isTerminal(job)) {
      if (jobEvents) {
        jobEvents.close();
        jobEvents = null;
      }
      return;
    }

    attachJobEvents(id);
  }

  async function queueAwareStartJob(event) {
    event.preventDefault();

    // Capture every setting immediately. Probing is asynchronous, so reading the DOM
    // after await could otherwise submit a different input/settings if the user has
    // already prepared the next queued job.
    const request = captureJobRequest();

    try {
      if (!request.input) throw new Error("input is required");

      const probe = await probeForRequest(request.input);
      request.output = buildSnapshotOutputPath(request, probe);

      const job = await api("/api/jobs", { method: "POST", body: JSON.stringify(request) });

      rememberJob(job);
      renderQueue();

      const selected = queueSelectedJobID ? queueJobs.get(queueSelectedJobID) : null;
      if (!selected || (!queueSelectionPinned && isTerminal(selected))) {
        await selectQueueJob(job.id, false);
      }

      // Refresh immediately so a job that changed queued -> running is reflected fast.
      refreshQueueJobs(true);
    } catch (error) {
      log(`Error: ${error.message}`);
    }
  }

  async function cancelQueueJob(id) {
    const job = queueJobs.get(id);
    const label = job ? basename(job.input) : id;
    if (!confirm(`Cancel job?\n\n${label}`)) return;

    try {
      await api(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
      const refreshed = await api(`/api/jobs/${encodeURIComponent(id)}`);
      rememberJob(refreshed);
      renderQueue();

      if (id === queueSelectedJobID) {
        legacyRenderJobEvent({ job: refreshed });
        syncSelectedPreview(refreshed);
      }

      setTimeout(() => refreshQueueJobs(true), 150);
    } catch (error) {
      log(`Cancel failed: ${error.message}`, true);
    }
  }

  function clearFinishedJobs() {
    const finishedIDs = queueJobIDs.filter((id) => isTerminal(queueJobs.get(id)));
    if (!finishedIDs.length) return;

    for (const id of finishedIDs) {
      queueJobs.delete(id);
    }
    queueJobIDs = queueJobIDs.filter((id) => !finishedIDs.includes(id));
    saveQueueIDs();

    if (finishedIDs.includes(queueSelectedJobID)) {
      queueSelectedJobID = null;
      queueSelectionPinned = false;
      currentJob = null;
      if (jobEvents) {
        jobEvents.close();
        jobEvents = null;
      }
      stopEncodePreview();
      queuePreviewJobID = null;
      log("No job selected.");
    }

    renderQueue();
    chooseBestJobToSelect();
  }

  async function refreshQueueJobs(_forceAll = false) {
    if (queueRefreshBusy) return;
    queueRefreshBusy = true;

    try {
      const data = await api("/api/jobs");
      const serverJobs = Array.isArray(data) ? data : (Array.isArray(data.jobs) ? data.jobs : []);
      const byID = new Map(serverJobs.map((job) => [job.id, job]));
      const stale = [];

      for (const id of [...queueJobIDs]) {
        const job = byID.get(id);
        if (job) {
          rememberJob(job);
        } else {
          stale.push(id);
        }
      }

      for (const id of stale) {
        forgetJob(id);
        if (queueSelectedJobID === id) {
          queueSelectedJobID = null;
          queueSelectionPinned = false;
        }
      }

      renderQueue();
      chooseBestJobToSelect();
    } catch (error) {
      // Keep the existing UI state on a transient refresh failure.
      console.warn("Queue refresh failed:", error);
    } finally {
      queueRefreshBusy = false;
    }
  }

  function chooseBestJobToSelect() {
    const selected = queueSelectedJobID ? queueJobs.get(queueSelectedJobID) : null;

    // A row explicitly selected by the user stays pinned, including completed jobs,
    // so its log can be inspected without the 1-second refresh jumping away.
    if (queueSelectionPinned && selected) return;
    if (selected && !isTerminal(selected)) return;

    const active = queueJobIDs
      .map((id) => queueJobs.get(id))
      .filter((job) => job && !isTerminal(job));

    const next = active.find((job) => job.status === "running")
      || active.find((job) => job.status === "queued");

    if (next && next.id !== queueSelectedJobID) {
      selectQueueJob(next.id, false);
    }
  }

  // Wrap the existing renderer so SSE updates also refresh the queue row.
  renderJobEvent = function queueAwareRenderJobEvent(event) {
    legacyRenderJobEvent(event);
    const job = event?.job;
    if (!job) return;

    rememberJob(job);
    if (!queueSelectedJobID) queueSelectedJobID = job.id;
    syncSelectedPreview(job);
    renderQueue();

    if (isTerminal(job)) {
      setTimeout(() => refreshQueueJobs(true), 100);
    }
  };

  // Replace the legacy submit handler. The original handler automatically selected
  // every newly-submitted job and therefore disconnected the previous job's SSE.
  // This version keeps the currently selected/running job visible while new jobs
  // are simply added to the queue.
  const form = $("jobForm");
  if (form) {
    form.removeEventListener("submit", legacyStartJob);
    form.addEventListener("submit", queueAwareStartJob);
  }

  injectQueueStyles();
  injectQueuePanel();
  renderQueue();

  // Restore job cards after browser refresh and remove stale IDs after server restart.
  refreshQueueJobs(true);

  if (queueRefreshTimer) clearInterval(queueRefreshTimer);
  queueRefreshTimer = setInterval(() => refreshQueueJobs(false), POLL_INTERVAL_MS);
})();
