# Changelog

All notable changes to DaSiWa True Video Enhancer are recorded in this file.
The project has no release tags; entries are grouped by date. The most recent
entries are at the top.

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
