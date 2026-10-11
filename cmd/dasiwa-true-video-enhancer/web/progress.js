/* The worker summary is the sole progress source; never infer from queue age/logs. */
(function (root) {
  function progressFor(job) {
    if (!job) return null;
    const done = Number(job.progress_done || 0);
    const total = Number(job.progress_total || 0);
    const throughput = Number(job.progress_throughput || 0);
    if (![done, total, throughput].every(Number.isFinite) || done < 0 || total < done || throughput < 0) return null;
    return { done, total, throughput, stage: job.progress_stage,
      updated: Date.parse(job.progress_updated_at) };
  }
  function duration(seconds) {
    seconds = Math.max(1, Math.ceil(seconds));
    const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = seconds % 60;
    return h ? `${h}h ${m}m` : m ? `${m}m ${s}s` : `${s}s`;
  }
  function view(job, now = Date.now()) {
    const status = job?.status;
    if (status !== 'running') return { label: String(status || 'unknown').toUpperCase(), detail: '', percent: status === 'done' ? 100 : 0 };
    const p = progressFor(job);
    const percent = p?.total ? Math.min(99, Math.floor(p.done / p.total * 100)) : 0;
    if (!p || p.stage === 'preparing' || (!p.done && p.stage !== 'finalizing')) {
      return { label: 'PREPARING', detail: 'Model / decoder setup · ETA after measured frames', percent };
    }
    if (p.stage === 'finalizing' || (p.total && p.done >= p.total)) {
      return { label: 'FINALIZING', detail: 'Encoding / muxing', percent: 99 };
    }
    const frames = p.total ? `${p.done}/${p.total} frames` : `${p.done} frames`;
    const age = Number.isFinite(p.updated) ? Math.max(0, (now - p.updated) / 1000) : Infinity;
    // Allow slow individual frames, but never count down forever on stale samples.
    const staleAfter = Math.max(30, p.throughput > 0 ? 3 / p.throughput : 30);
    if (age > staleAfter) return { label: 'WAITING FOR PROGRESS', detail: `${frames} · ETA unavailable`, percent };
    if (!p.throughput || !p.total) return { label: 'RENDERING', detail: `${frames} · Measuring throughput`, percent };
    const eta = Math.max(1, (p.total - p.done) / p.throughput - age);
    return { label: `RENDERING ${percent}%`, detail: `${frames} · ${p.throughput.toFixed(2)} fps · ETA ≈ ${duration(eta)}`, percent };
  }
  const api = { progressFor, view };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.TVEProgress = api;
})(typeof window !== 'undefined' ? window : globalThis);
