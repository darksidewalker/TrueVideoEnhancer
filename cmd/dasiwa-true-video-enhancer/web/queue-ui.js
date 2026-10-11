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

  function progressPercent(job) {
    return TVEProgress.view(job).percent;
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
        return TVEProgress.view(job).label;
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

    if (job.status === "queued") {
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
    if (job.status === "running") {
      const detail = document.createElement("div");
      detail.className = "queue-ui-meta";
      detail.textContent = TVEProgress.view(job).detail;
      side.appendChild(detail);
    }

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
    const snapshot = captureSettings();
    const input = $("input").value.trim();

    try {
      const probe = await probeForRequest(input);
      const request = captureJobRequest(snapshot, probe, input);
      if (!request.input) throw new Error("input is required");

      request.output = TVE.outputPath(request);

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
      serverJobs.forEach(rememberJob);
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

  const form = $("jobForm");
  if (form) {
    form.addEventListener("submit", queueAwareStartJob);
  }

  injectQueuePanel();
  renderQueue();

  // ---- Batch folder queue -------------------------------------------------
  const batchDialog = $("batchDialog");
  let batchItems = [];

  async function loadBatchFolder() {
    const folder = $("batchFolder").value.trim();
    const box = $("batchFileList");
    const summary = $("batchSummary");
    batchItems = [];
    if (!folder) {
      box.innerHTML = '<p class="hint">Enter a folder path first.</p>';
      summary.textContent = "";
      return;
    }
    box.innerHTML = '<p class="hint">Scanning…</p>';
    try {
      const recursive = $("batchRecursive").checked ? "1" : "0";
      const data = await api(`/api/batch-folder?path=${encodeURIComponent(folder)}&recursive=${recursive}`);
      batchItems = data.items || [];
      box.innerHTML = "";
      if (!batchItems.length) {
        box.innerHTML = '<p class="hint">No video files found in this folder.</p>';
        summary.textContent = "";
        return;
      }
      for (const item of batchItems) {
        const row = document.createElement("label");
        row.className = "file-browser-row";
        row.style.display = "flex";
        row.style.alignItems = "center";
        row.style.gap = "8px";
        row.style.cursor = "pointer";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = true;
        cb.dataset.path = item.path;
        const name = document.createElement("span");
        name.className = "file-browser-name";
        const sizeMB = item.size ? ` (${(item.size / (1024 * 1024)).toFixed(1)} MB)` : "";
        name.textContent = item.name + sizeMB;
        row.appendChild(cb);
        row.appendChild(name);
        box.appendChild(row);
      }
      updateBatchSummary();
    } catch (err) {
      box.innerHTML = `<p class="warning">Folder scan failed: ${err.message}</p>`;
      summary.textContent = "";
    }
  }

  function updateBatchSummary() {
    const checked = [...$("batchFileList").querySelectorAll("input[type=checkbox]:checked")];
    $("batchSummary").textContent = checked.length
      ? `${checked.length} of ${batchItems.length} videos selected`
      : "No videos selected";
  }

  async function enqueueBatch() {
    const folder = $("batchFolder").value.trim();
    const checked = [...$("batchFileList").querySelectorAll("input[type=checkbox]:checked")].map((cb) => cb.dataset.path);
    if (!checked.length) {
      log("Batch: select at least one video first.", true);
      return;
    }
    const snapshot=captureSettings();
    const outputDir = $("batchOutputDir").value.trim() || `${folder}/enhanced`;
    try {
      // A flat batch request cannot carry different source-dependent model / FG
      // choices. Probe all inputs first and prepare validated one-input batches.
      const prepared=[];
      for (const input of checked) {
        const probe=await probeForRequest(input);
        const request=captureJobRequest(snapshot,probe,input);
        request.input=""; request.output="";
        // Preserve recursive source directories so equal basenames cannot collide.
        const normalized=input.replace(/\\/g,'/'), root=folder.replace(/\\/g,'/').replace(/\/$/,'');
        const relative=normalized.startsWith(root+'/')?normalized.slice(root.length+1):normalized.split('/').pop();
        const slash=relative.lastIndexOf('/');
        const subdir=slash>=0?relative.slice(0,slash):'';
        const filename=relative.slice(slash+1), stem=filename.replace(/\.[^.]+$/,'');
        const sameStem=checked.filter(path=>{
          const p=path.replace(/\\/g,'/');return p.slice(0,p.lastIndexOf('/'))===normalized.slice(0,normalized.lastIndexOf('/')) && p.split('/').pop().replace(/\.[^.]+$/,'')===stem;
        }).length>1;
        const collisionDir=sameStem?filename.replace(/[^\p{L}\p{N}._-]+/gu,'_'):'';
        prepared.push({...request,inputs:[input],output_dir:[outputDir,subdir,collisionDir].filter(Boolean).join('/')});
      }
      const result={jobs:[],skipped:[]};
      for (const payload of prepared) {
        const response=await api("/api/jobs/batch",{method:"POST",body:JSON.stringify(payload)});
        for (const job of response.jobs || []) { rememberJob(job); result.jobs.push(job); }
        result.skipped.push(...(response.skipped || []));
      }
      for (const job of result.jobs || []) {
        rememberJob(job);
      }
      renderQueue();
      refreshQueueJobs(true);
      const skippedNote = (result.skipped || []).length
        ? ` Skipped ${result.skipped.length} (output exists / invalid).`
        : "";
      log(`Batch queued ${ (result.jobs || []).length } job(s) → ${outputDir}.${skippedNote}`);
      batchDialog.close();
    } catch (err) {
      log(`Batch failed: ${err.message}`, true);
    }
  }

  if ($("openBatchDialog")) {
    $("openBatchDialog").addEventListener("click", () => {
      if (!$("batchFolder").value.trim()) {
        const input = $("input").value.trim();
        $("batchFolder").value = input ? filepathDir(input) : (currentBrowsePath || homeDir());
      }
      loadBatchFolder();
      batchDialog.showModal();
    });
  }
  if ($("closeBatchX")) $("closeBatchX").addEventListener("click", () => batchDialog.close());
  if ($("browseBatchFolder")) {
    $("browseBatchFolder").addEventListener("click", () => {
      openFileBrowser();
      log("Navigate to the folder in the browser, then click 'Use current browse folder'.");
    });
  }
  if ($("useCurrentBrowseFolder")) {
    $("useCurrentBrowseFolder").addEventListener("click", () => {
      $("batchFolder").value = currentBrowsePath || homeDir();
      loadBatchFolder();
    });
  }
  if ($("batchFolder")) $("batchFolder").addEventListener("change", loadBatchFolder);
  if ($("batchRecursive")) $("batchRecursive").addEventListener("change", loadBatchFolder);
  if ($("batchSelectAll")) {
    $("batchSelectAll").addEventListener("click", () => {
      const boxes = [...$("batchFileList").querySelectorAll("input[type=checkbox]")];
      const allChecked = boxes.every((cb) => cb.checked);
      for (const cb of boxes) cb.checked = !allChecked;
      $("batchSelectAll").textContent = allChecked ? "Select all" : "Select none";
      updateBatchSummary();
    });
  }
  if ($("batchFileList")) {
    $("batchFileList").addEventListener("change", updateBatchSummary);
  }
  if ($("batchEnqueue")) $("batchEnqueue").addEventListener("click", enqueueBatch);

  // Restore job cards after browser refresh and remove stale IDs after server restart.
  refreshQueueJobs(true);

  if (queueRefreshTimer) clearInterval(queueRefreshTimer);
  queueRefreshTimer = setInterval(() => refreshQueueJobs(false), POLL_INTERVAL_MS);
})();
