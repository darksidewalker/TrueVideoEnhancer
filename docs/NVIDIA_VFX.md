# NVIDIA VFX pipeline

Activate the individual NVIDIA feature buttons in the studio's middle Settings area. NVIDIA upscale and frame generation can replace AI Upscale and RIFE independently; denoise, deblur and SDR-to-HDR can also accompany AI/RIFE. The Source plan shows the actual restore → spatial → temporal → HDR → encode order. Jobs and folder batches use the same settings. The legacy `--pipeline nvidia-vfx` remains available for old CLI callers; the studio uses `--processing_mode custom`.

## Controls

- Denoise and Deblur restore frames at source resolution. Each has its own quality tier and strength from 0 to 1.
- RTX upscale offers Off, VSR, High Bitrate, Streaming Medium and Streaming Ultra. VSR targets typical compressed material; High Bitrate is an SDK reconstruction preset, not an export-bitrate setting. Streaming presets are SDK quality modes, not a different file-I/O architecture.
- RTX Frame Generation interpolates between adjacent frames at the existing target FPS. LOW, MEDIUM and HIGH are frame-generation quality presets, distinct from upscale quality. Arbitrary FPS ratios use timestamp-derived timesteps rather than an integer-only multiplier.
- Automatic shot-change detection asks the SDK to detect cuts. The API-only force-shot flag marks every pair as a cut, not selected cut indices; the studio exposes automatic detection instead.
- RGB8 is the compatibility default. RGB10A2 is packed 10-bit SDR processing and does not turn SDR into HDR or restore precision lost in the source.
- TrueHDR is an optional final SDR-to-BT.2020/PQ conversion. Its contrast, saturation, middle-gray, peak-luminance and debanding controls are independent of upscale. It can be used without upscaling. TrueHDR is disabled by default.

Feature support depends on the installed `nvidia-vfx` binding, embedded SDK, GPU and driver. Unsupported explicit requests must produce an error rather than silently switching to another effect or lower precision. SDK availability alone is not proof that inference works on a particular device.

## Encoder preflight

Every offered encoder is smoke-tested before model/effect inference with one aligned 256×256 frame at 24 FPS and the production encoder profile/quality options. Strict pixel formats prevent an unsupported 10-bit request from silently becoming 8-bit. NVENC 10-bit probes use P010, software codecs use planar 10-bit, and ProRes uses 10-bit 4:2:2. The probe explicitly converts the test frame to that format, has a 20-second timeout and reports the actual FFmpeg failure details. This checks basic encoder availability, not memory requirements at the full requested resolution.

Run `TVE_ENCODER_SMOKE=1 runtime/venv/bin/python -m pytest backend/tests/test_encoder_preflight.py -q` for real preflight and encoded-bitstream verification of all ten presets and their seven advertised 10-bit modes, plus WebM Auto/manual AV1 selection. It opens small NVENC sessions; ordinary regression tests leave these hardware probes opt-in. H.264 and VP8 remain excluded from the existing precision/HDR contract; ProRes is intrinsically 10-bit even for SDR.

## Streaming and media

The implementation processes bounded frame queues and adjacent source pairs, not complete ComfyUI IMAGE batches. Source audio and subtitle handling remain in the FFmpeg stage. Increasing the target FPS does not intentionally change playback speed. The NVIDIA timestamp scheduler retains the final source-frame display interval to preserve source duration. CFR rounding can differ by less than one target-frame interval. Variable/interlaced cadence, HDR source input, trimming and custom-encoder flags are rejected explicitly in this pipeline rather than silently approximated.

Ten-bit processing requires a precision-preserving decode, processing and encode path. HDR output additionally requires a compatible 10-bit encoder and explicit BT.2020/PQ color tags. An SDR JPEG preview cannot validate HDR encoding or represent the original HDR signal faithfully.

The existing output-resolution and host-memory safeguards continue to apply. Bounded queues do not guarantee that every SDK mode or resolution fits GPU memory. Lower resolution or reduce enabled effects if SDK allocation fails; do not stop unrelated GPU services to make a job fit.

## Runtime and deployment

Use the project's Runtime installer to install the declared dependencies into `runtime/venv`. Existing environments need the new NVIDIA binding before running this pipeline. Backend changes take effect in the next Python job. Rebuild the Go executable for the new selector and embedded controls; running server instances retain their old embedded UI until restarted by the user.

No ComfyUI server or node pack is required at runtime. The integration adopts the NVIDIA SDK protocols from the DaSiWa RTX nodes, not their batch allocation or ComfyUI cache management.
