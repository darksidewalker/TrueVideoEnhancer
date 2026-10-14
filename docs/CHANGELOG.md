# Changelog

All notable changes to DaSiWa True Video Enhancer are recorded in this file.
The project has no release tags; entries are grouped by date. The most recent
entries are at the top.

## 2026-10-14 — NVIDIA internal upscale and downsample (`2026.10.14-nvidia-refine`)

- Added opt-in source-relative internal NVIDIA 2× / 4× with separate final output scale (including 1×). Default Off preserves the existing path; Area is the downsample default, with Lanczos and Bicubic choices.
- Persisted studio controls and one canonical single/batch request; Source plan names internal scale, reduction filter and final scale. No inactive setting leaks to AI/legacy backends.
- Added early API/CLI cross-field validation and intermediate 8K/host-memory guards. Downsample occurs once before temporal interpolation and HDR, retaining uint8/uint16 storage and final encoder dimensions; no extra encode or frame staging.
- Added frontend/browser, Go argv/JSON and CPU pipeline regressions. Real two-frame NVIDIA SDK renders verified 2×→1× Area and 4×→2× Lanczos / 4×→1× Bicubic, including actual FFV1 8/10-bit outputs and staging cleanup.
- Rebuilt Linux amd64 and Windows amd64/arm64 binaries directly with the feature version; running services remain untouched. Windows inference is not established by cross-compilation.

## 2026-10-11 — Processed preview fidelity fix

- Removed the completed-preview reduction to 480 pixels and 12 FPS; browser proxies retain output resolution/cadence and use H.264 CRF 18. Versioned cache paths and URLs bypass older low-quality copies.
- Live processed JPEGs retain native frame dimensions at quality 95. Enlarged the bounded worker transport to handle those frames without truncation.
- Clarified that playback is a first-20-seconds SDR browser proxy and that live JPEGs precede final video encoding; HDR previews remain tone-mapped rather than identical to the HDR output.
- Verified the actual HTTP proxy with real FFmpeg encode/decode: 960×540, 60 FPS, 60 frames, SSIM above 0.98. Added native 1920×1080 JPEG and oversized subprocess-transport regressions.

## 2026-10-11 — Composable processing studio

- Swapped Settings and Preview; moved output scale/target FPS into Settings and added a compact planned processing chain in Source.
- Replaced fixed pipeline choices with seven feature buttons: AI Upscale, RIFE Interpolation, NVIDIA Upscale, Frame Generation, Denoise, Deblur and SDR-to-HDR. One spatial and one temporal backend can be selected independently, with optional restoration and final HDR conversion.
- Added bounded custom processing with selective model forwarding, floating-point AI/RIFE input, high-precision storage, shared preview/progress and atomic output publication. Legacy API/CLI behavior remains available.
- Verified real AI Upscale → RIFE → NVIDIA HDR inference using CPU models and GPU HDR: four 256×256 HEVC 10-bit/PQ frames. A real browser job also verified custom NVIDIA upscale and a one-second, 24-frame preview. Heavy all-GPU combinations remain untested under current VRAM limits.

## 2026-10-11 — Compact studio and preview rendering

- Replaced the preset/advanced-dialog workflow with compact Source, Preview and always-visible Settings areas; feature buttons expose only relevant processing controls.
- Removed the redundant restore-only UI choice and Fast/Balanced/Best presets. AI, NVIDIA and Hybrid remain distinct pipelines.
- Added runtime capability discovery and compatible container/codec filtering, with the same shared matrices validated before inference.
- Added first-seconds preview rendering with a 1–30 second slider, bounded decode/inference/media mux and separate preview output names.
- Verified a real browser-submitted NVIDIA HDR preview: 60 video frames at 60 FPS from the first second of a three-second source, with bounded audio. Desktop/mobile layout checks found no JavaScript errors or horizontal overflow.

## 2026-10-11 — NVIDIA VFX pipeline

- Added a separate selectable NVIDIA VFX pipeline without replacing model-based upscaling or RIFE.
- Added arbitrary-target-FPS RTX Frame Generation, quality presets and SDK shot-change controls.
- Updated denoise/deblur/upscale SDK handling, separate strengths, RGB10A2 precision and optional final TrueHDR controls.
- Added precision-preserving rawvideo output, compatible 10-bit encoding, PQ/BT.2020 tagging and independent SDR browser/live previews.
- Added single/batch setting persistence, explicit zero/false forwarding, stale-model isolation and capability diagnostics.
- Declared and installed `nvidia-vfx==0.2.0.0`. Linux RTX 5090 small-input GPU and HTTP job renders verified; Windows executables cross-built, Windows VFX inference not tested.

## 2026-09-29

- **Batch folder queue.** `POST /api/jobs/batch` queues every listed input as
  sequential jobs; `GET /api/batch-folder` lists videos in a folder. The Queue
  panel gains a Batch Folder Queue dialog (browse → checklist → enqueue).
  Existing outputs are skipped, so a stopped batch resumes where it ended.
- **RTX VFX pipeline (`backend/rtx_vfx.py`).** NVIDIA RTX Video SDK
  (`nvvfx`) support ported from the DaSiWa RTX Upscaler & Refiner ComfyUI
  node protocol: quality-level attribute resolution, constructor probing,
  CUDA stream sync with DLPack clone on every effect handoff, direct
  SDK-output copy into the preallocated row, VFX SDK 1.3 strength parameter,
  and Streaming Medium/Ultra modes (Ampere+).
- **Restore pipeline.** `--pipeline restore` runs the RTX VFX denoise/deblur
  chain at source resolution with no upscale model and source FPS; the chain
  docks at exactly one streaming stage so effects never run twice.
- **Streaming memory tuning.** Reader/writer queue depth adapts to frame size
  vs available RAM; the resource-budget check runs before render start.
- **FFmpeg 7+ compatibility.** `-vsync` was removed in FFmpeg 7; decode and
  frame extraction now pick `-fps_mode passthrough` automatically.

## 2026-09-02

- **Windows support.** The Go server is fully portable (pure Go, no
  CGo) and all pinned runtime packages — `torch`/`torchvision` cu132,
  `torch_tensorrt`, `cupy-cuda13x`, `onnxruntime-gpu` — ship
  `win_amd64` wheels, so the uv-managed runtime installs and runs on
  Windows (venv Python resolved via `Scripts/python.exe`; uv bootstrap
  uses the official installer via PowerShell; "open folder" uses
  `explorer.exe`). Native Windows inference is supported on amd64; on
  Windows arm64 the server runs but no CUDA wheels exist.
- **Prebuilt binaries committed to the repository root**, one name per
  platform: `dasiwa-true-video-enhancer-linux-amd64`,
  `dasiwa-true-video-enhancer-windows-amd64.exe`,
  `dasiwa-true-video-enhancer-windows-arm64.exe`. `build.sh`
  cross-compiles all three; the Linux root copy was renamed from the
  bare `dasiwa-true-video-enhancer` to match the pattern.

## 2026-08-30

- **Fixed corrupted output from the persistent TensorRT upscaler engine cache
  (issue #2).** The durable Dynamo engine cache could leave stale constant
  weights behind on a refit, producing solid-color frames on the second job
  with some upscaler architectures (e.g. RCAN). The upstream fix, the
  "Simplified Refit Pipeline" in `torch_tensorrt` 2.13.0, rebuilds the
  weight mapping from the graph on every refit and eliminates the bug.
- **Upgraded the portable runtime** to `torch==2.13.0+cu132`,
  `torchvision==0.28.0+cu132`, `torch_tensorrt==2.13.0` (bundles
  `tensorrt-cu13` 11.0.0.114). The previous 2.12.x family was affected.
- **The upscaler engine cache is now ON by default.** Verified on the
  hardware with the three-phase runner `tools/verify_issue2_trt_cache.py`
  (Phases A/B/C in separate subprocesses). To fall back to a fresh compile
  per job, set `RVE_UPSCALER_TRT_ENGINE_CACHE=0` (also accepts
  `false`/`no`/`off`). The RIFE and ONNX TensorRT caches are unaffected.
- Added `tools/verify_issue2_trt_cache.py` and the evidence records
  `tve-verify.log` (corruption on 2.12.1, exit 1) and
  `tve-verify-2130.log` (clean, exit 0).

## 2026-08-10

- Sequential job queue: jobs run one at a time with a visible queue UI
  (`4d3d692`).
- Output filename now reflects the resolved video codec (`bf85018`).

## 2026-07-17

- Guarded 4x jobs against host-memory exhaustion with an 8K-UHD pixel
  budget check before the worker starts (`10b8208`).
- Added the persistent TensorRT engine cache (`f85121d`) — later gated
  behind `RVE_UPSCALER_TRT_ENGINE_CACHE` in August 2026.
- Kept iterative 2x→4x upscaling on the GPU (`574ad3d`).
- Added explicit FP16 typing support for the Torch-TensorRT 2.12 path
  (`8e83871`).
- Bundled the `2x-AnimeJaNai_HD_V3_Compact.safetensors` base upscaler
  (`35fe8f5`).
- Documented the supported video pipeline and model/project credits
  (`6881bc6`, `c96c65e`).

## 2026-07-12

- Replaced serial PNG frame sequences with a bounded FFmpeg rawvideo
  decode → AI → encode streaming pipeline (`6625712`).

## 2026-07-11

- Restored the `main.go` entry point and corrected repo-root resolution
  (`42c0950`).
- Replaced the HAT-L pipeline with native fast restoration models routed
  per content type (`acb0b89`).

## 2026-07-06

- Published compiled Linux amd64 binaries in releases (`0a1049e`).
- Frontend live file-size estimate, preview screenshot, and housekeeping
  cleanups.

## 2026-07-06 and earlier

- Initial DaSiWa True Video Enhancer release: local NVIDIA video upscaling
  and RIFE frame interpolation with a Go browser UI and a uv-managed
  Python runtime.
