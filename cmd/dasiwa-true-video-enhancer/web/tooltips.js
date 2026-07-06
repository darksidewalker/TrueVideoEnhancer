/**
 * Tooltips & Translations Module
 *
 * Separate import for hints, naming, and language support.
 * Each tooltip has: id, title (short), body (detailed explanation).
 * Titles can be translated; bodies contain technical context.
 *
 * Usage:
 *   <script src="tooltips.js"></script>
 *   <script src="app.js"></script>
 *
 * To add a new language:
 *   1. Copy TOOLTIPS object keys
 *   2. Translate titles only (bodies stay technical)
 *   3. Call setLanguage("de") or setLanguage("en")
 */

(function () {
  "use strict";

  // ─── Language Dictionary ──────────────────────────────────────────

  const LANGUAGES = {
    en: {
      name: "English",
      tooltips: {},
    },
    de: {
      name: "Deutsch",
      tooltips: {},
    },
    fr: {
      name: "Français",
      tooltips: {},
    },
    es: {
      name: "Español",
      tooltips: {},
    },
    ja: {
      name: "日本語",
      tooltips: {},
    },
  };

  // Default English tooltips (source of truth)
  const EN_TOOLTIPS = {
    // ── Header Buttons ──────────────────────────────────────────────
    "btn-check-backend": {
      title: "Run Diagnostics",
      body: "Checks the Python runtime, GPU availability, and all backend components (TensorRT, PyTorch, FFmpeg, etc.). Shows a pass/fail summary.",
    },
    "btn-install-runtime": {
      title: "Install Runtime",
      body: "Sets up an isolated Python environment managed by uv. Downloads dependencies (PyTorch, TensorRT, FFmpeg bindings) without touching your system Python.",
    },
    "btn-open-models": {
      title: "Manage Models",
      body: "Opens the model manager. Download, view, and select the AI models used for upscaling and frame interpolation.",
    },
    "btn-quit": {
      title: "Quit Application",
      body: "Gracefully shuts down the web server and all running jobs. Confirm to proceed.",
    },

    // ── Source Panel ────────────────────────────────────────────────
    "field-input": {
      title: "Input Video Path",
      body: "Full filesystem path to the source video file. Use the folder icon to browse, or paste a direct path like /home/user/video.mp4.",
    },
    "btn-browse": {
      title: "Browse Files",
      body: "Open a file browser starting at your home directory. Click folders to navigate, click a video file to select it.",
    },
    "field-content-type": {
      title: "Content Type",
      body: "Defines the visual style of your source material:\n• Anime — Line art, flat colors, high contrast\n• Mixed — Combination of animated and real footage\n• Realism — Live-action, photorealistic content",
    },
    "field-output-container": {
      title: "Output Container Format",
      body: "The wrapper format for the output video file. MP4 is universally compatible. MKV supports more metadata and multiple audio tracks.",
    },
    "field-target-fps": {
      title: "Target Frame Rate",
      body: "Desired FPS for the output video. Higher values produce smoother motion. Use values above source FPS for slow-motion effect via frame interpolation.",
    },
    "field-scale": {
      title: "Resolution Scale Factor",
      body: "Multiplies the video dimensions:\n• 1x — No change\n• 2x — Double width and height (4× pixels)\n• 4x — Quadruple (16× pixels)",
    },

    // ── Presets ─────────────────────────────────────────────────────
    "preset-fast": {
      title: "Fast Preset",
      body: "Optimized for speed. Uses lower-quality models and fewer processing passes. Suitable for quick previews or low-end hardware.",
    },
    "preset-balanced": {
      title: "Balanced Preset",
      body: "Default recommendation. Balances output quality with processing time. Good for most content types and typical GPU setups.",
    },
    "preset-best": {
      title: "Best Quality Preset",
      body: "Maximum quality output. Uses highest-tier models with ensemble passes and dynamic optical flow. Slowest but best results.",
    },

    // ── Preview Panel ───────────────────────────────────────────────
    "btn-open-tune": {
      title: "Advanced Settings",
      body: "Opens detailed controls: backend selection, model overrides, encoding parameters, scene detection, and experimental toggles.",
    },

    // ── Jobs Panel ──────────────────────────────────────────────────
    "field-log": {
      title: "Job Log",
      body: "Live feed of job progress, status changes, and error messages. Updates in real-time as the backend processes your video.",
    },

    // ── Advanced: Backend & Models ──────────────────────────────────
    "field-backend": {
      title: "Inference Backend",
      body: "Computing engine for AI models:\n• TensorRT — NVIDIA GPU acceleration, fastest (requires NVIDIA GPU)\n• PyTorch CUDA — Standard fallback, works on any GPU",
    },
    "field-rife-model": {
      title: "Frame Interpolation Model",
      body: "AI model for generating intermediate frames. Heavy variant adds detail for anime; standard variant is faster for live-action.",
    },
    "field-upscale-model": {
      title: "Upscaling Model",
      body: "AI model for increasing resolution. Different models are optimized for anime, mixed, or realism content.",
    },

    // ── Advanced: Encoding ──────────────────────────────────────────
    "field-crf": {
      title: "CRF / Constant Quality",
      body: "Rate control parameter (0–51). Lower = better quality, larger file. Typical range: 16–24. 18 is a good default.",
    },
    "field-cfg-scale": {
      title: "CFG Scale (Classifier-Free Guidance)",
      body: "Controls how strongly the model follows its training. Reserved until a model explicitly supports it. Leave disabled for now.",
    },
    "field-video-encoder": {
      title: "Video Encoder",
      body: "Compression codec for the output video:\n• NVENC variants — Hardware accelerated on NVIDIA GPUs\n• CPU variants — Software encoding, works everywhere",
    },
    "field-pixel-format": {
      title: "Pixel Format",
      body: "Color sampling and bit depth:\n• yuv420p — Standard 8-bit, widely compatible\n• yuv420p10le — 10-bit for HDR content\n• yuv444p — Full color detail (larger files)",
    },
    "field-audio-encoder": {
      title: "Audio Codec",
      body: "How to handle audio in the output:\n• Copy — Keep original audio unchanged\n• AAC/Opus/MP3 — Re-encode to specified format",
    },
    "field-audio-bitrate": {
      title: "Audio Bitrate",
      body: "Bits per second for audio encoding. Higher = better quality:\n• 64k — Mono, minimal\n• 128k–192k — Standard stereo\n• 256k–320k — High fidelity",
    },
    "field-subtitle-encoder": {
      title: "Subtitle Handling",
      body: "What to do with embedded subtitles:\n• Copy — Preserve original subtitle stream\n• SRT/ASS/WebVTT — Convert to specific format",
    },
    "field-tile-size": {
      title: "Tile Size",
      body: "Size of image tiles processed separately. Reduces VRAM usage for large images. Set to 0 for automatic tiling.",
    },

    // ── Advanced: TensorRT ──────────────────────────────────────────
    "field-tensorrt-profile": {
      title: "TensorRT Optimization Profile",
      body: "Build-time optimization level:\n• 1 — Fastest build, lowest quality\n• 3 — Balanced build time and quality (default)\n• 5 — Slowest build, best performance",
    },
    "toggle-tensorrt-dynamic-shapes": {
      title: "Dynamic Shapes",
      body: "Allows TensorRT to handle variable input sizes without rebuilding the network. Slower initial build but more flexible at runtime.",
    },

    // ── Advanced: Toggles ───────────────────────────────────────────
    "toggle-hdr-mode": {
      title: "HDR Colorspace Mode",
      body: "Preserves high-dynamic-range color information through the pipeline. Required for HDR source content.",
    },
    "toggle-uhd-mode": {
      title: "UHD/8K VRAM Saver",
      body: "Memory-saving mode for ultra-high-resolution processing. Reduces intermediate buffer sizes to fit within limited VRAM.",
    },
    "toggle-slomo-mode": {
      title: "Slow Motion Mode",
      body: "Lengthens the video duration by inserting interpolated frames between originals. Target FPS must be higher than source FPS.",
    },
    "toggle-ensemble": {
      title: "Ensemble Processing",
      body: "Runs multiple inference passes and combines results for higher quality. Slower but reduces artifacts.",
    },
    "toggle-dynamic-optical-flow": {
      title: "Dynamic Optical Flow",
      body: "Adapts the optical flow algorithm per-frame for better motion estimation. Higher quality but slower processing.",
    },

    // ── Advanced: Device & Timing ───────────────────────────────────
    "field-device": {
      title: "Compute Device",
      body: "Hardware accelerator for AI inference:\n• Auto — System default\n• CUDA — NVIDIA GPUs\n• MPS — Apple Silicon\n• XPU — Intel Arc GPUs",
    },
    "toggle-benchmark-mode": {
      title: "Benchmark Mode",
      body: "Runs timing measurements and reports performance metrics. Useful for comparing hardware configurations.",
    },
    "field-start-time": {
      title: "Start Time (seconds)",
      body: "Optional: begin processing from this timestamp. Leave empty to process from the beginning.",
    },
    "field-end-time": {
      title: "End Time (seconds)",
      body: "Optional: stop processing at this timestamp. Leaves remaining content unprocessed.",
    },
    "field-pytorch-gpu-id": {
      title: "PyTorch GPU Index",
      body: "Which GPU to use when multiple are present. 0 = first GPU.",
    },
    "field-ncnn-gpu-id": {
      title: "NCNN GPU Index",
      body: "Which GPU to use for NCNN-based inference paths. 0 = first GPU.",
    },
    "field-custom-encoder": {
      title: "Custom Encoder Arguments",
      body: "Expert override: extra flags passed directly to FFmpeg. Use cautiously — incorrect arguments may break the output.",
    },
    "toggle-dry-run": {
      title: "Dry Run Only",
      body: "Builds and displays the complete command that would be executed, without actually running it. Safe for testing configurations.",
    },

    // ── Advanced: Scene Detection ───────────────────────────────────
    "field-scene-detect": {
      title: "Scene Detection Method",
      body: "Algorithm for detecting scene cuts:\n• PySceneDetect — Robust open-source library\n• Mean — Average frame brightness comparison\n• Mean segmented — Block-based mean comparison\n• None — Skip scene detection entirely",
    },
    "field-scene-threshold": {
      title: "Scene Threshold",
      body: "Sensitivity for scene cut detection (0–10). Lower values detect more cuts. Typical range: 2–6. Adjust based on source content complexity.",
    },
    "field-override-upscale-scale": {
      title: "Override Upscale Scale",
      body: "Force a specific upscale factor regardless of the main Scale setting. Useful when you want different processing per model without changing the UI preset.",
    },

    // ── Actions ─────────────────────────────────────────────────────
    "btn-prepare-job": {
      title: "Prepare Job",
      body: "Submits the configured processing job to the queue. The backend will run the full pipeline: decode → denoise → upscale → interpolate → encode.",
    },

    // ── Dialogs ─────────────────────────────────────────────────────
    "dialog-check": {
      title: "Backend Status Check",
      body: "Comprehensive diagnostic report showing availability of Python, GPU, TensorRT, PyTorch, and FFmpeg components.",
    },
    "dialog-models": {
      title: "Model Manager",
      body: "Download and manage AI models. Click a model row to select it. Download arrows indicate missing models.",
    },
    "dialog-file-browser": {
      title: "File Browser",
      body: "Navigate your filesystem to find video files. Click folders to enter them, click a file to select it.",
    },
    "dialog-download": {
      title: "Download Progress",
      body: "Real-time log of model download progress. Do not close this dialog while downloading.",
    },
  };

  // ─── Apply translations ──────────────────────────────────────────

  function deepCopy(obj) {
    return JSON.parse(JSON.stringify(obj));
  }

  function buildTooltipMap(languageCode) {
    const lang = LANGUAGES[languageCode];
    if (!lang) return deepCopy(EN_TOOLTIPS);

    // If language has translations, merge over English defaults
    if (lang.tooltips && Object.keys(lang.tooltips).length > 0) {
      const merged = deepCopy(EN_TOOLTIPS);
      for (const [id, fields] of Object.entries(lang.tooltips)) {
        if (merged[id]) {
          if (fields.title) merged[id].title = fields.title;
          if (fields.body) merged[id].body = fields.body;
        }
      }
      return merged;
    }

    return deepCopy(EN_TOOLTIPS);
  }

  // ─── Tooltip Engine ──────────────────────────────────────────────

  let currentLang = "en";
  let activeTooltip = null;

  function getTooltips() {
    return buildTooltipMap(currentLang);
  }

  function showTooltip(element, tooltipId) {
    hideTooltip();

    const tooltips = getTooltips();
    const tt = tooltips[tooltipId];
    if (!tt) return;

    const tip = document.createElement("div");
    tip.className = "ds-tip";
    tip.innerHTML = `<strong>${escapeHtml(tt.title)}</strong><span class="ds-tip-body">${escapeHtml(tt.body).replace(/\n/g, "<br>")}</span>`;

    // Append to the nearest dialog ancestor if present — keeps tooltip inside
    // the dialog's stacking context so it renders above the backdrop overlay.
    const container = element.closest("dialog") || document.body;
    container.appendChild(tip);

    // Force visible immediately in next frame after layout settles
    requestAnimationFrame(function() {
      tip.classList.add("visible");
      tip.style.opacity = "1";
      
      // Measure AFTER visibility is set so dimensions are correct
      const rect = element.getBoundingClientRect();
      const tipRect = tip.getBoundingClientRect();

      let left = rect.left + rect.width / 2 - tipRect.width / 2;
      let top = rect.bottom + 8;

      // Keep within viewport
      if (left < 8) left = 8;
      if (left + tipRect.width > window.innerWidth - 8) {
        left = window.innerWidth - tipRect.width - 8;
      }
      if (top + tipRect.height > window.innerHeight - 8) {
        top = rect.top - tipRect.height - 8;
      }

      tip.style.left = left + "px";
      tip.style.top = top + "px";
    });

    activeTooltip = { el: tip, timeout: null };

    activeTooltip.timeout = setTimeout(hideTooltip, 8000);
  }

  function hideTooltip() {
    if (activeTooltip) {
      if (activeTooltip.el.parentNode) {
        activeTooltip.el.parentNode.removeChild(activeTooltip.el);
      }
      activeTooltip = null;
    }
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // Public API
  window.Tooltips = {
    /**
     * Switch active language. Pass any key from LANGUAGES.
     * Falls back to English if no translations exist yet.
     */
    setLanguage(code) {
      currentLang = code || "en";
    },

    /** Get currently active language code */
    getLanguage() {
      return currentLang;
    },

    /** List available language codes */
    getLanguages() {
      return Object.keys(LANGUAGES);
    },

    /** Show tooltip for a given element ID */
    show(id) {
      const el = document.getElementById(id);
      if (el) {
        showTooltip(el, id);
      }
    },

    /** Hide current tooltip */
    hide: hideTooltip,

    /** Register all elements with data-tooltip attributes */
    registerAll() {
      const TOOLTIP_DELAY_MS = 1000;
      let pendingTimer = null;

      document.querySelectorAll("[data-tooltip]").forEach(function (el) {
        const tid = el.getAttribute("data-tooltip");
        
        // For radio/checkbox inputs (zero-sized until focused), attach events
        // to the parent label so hover works reliably.
        let target = el;
        if ((el.type === "radio" || el.type === "checkbox") && el.closest("label")) {
          target = el.closest("label");
        }

        target.addEventListener("mouseenter", function () {
          clearTimeout(pendingTimer);
          pendingTimer = setTimeout(function () {
            showTooltip(target, tid);
          }, TOOLTIP_DELAY_MS);
        });
        target.addEventListener("mouseleave", function () {
          clearTimeout(pendingTimer);
          hideTooltip();
        });
      });
    },
  };

  // Auto-register on DOMContentLoaded
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      window.Tooltips.registerAll();
    });
  } else {
    window.Tooltips.registerAll();
  }
})();
