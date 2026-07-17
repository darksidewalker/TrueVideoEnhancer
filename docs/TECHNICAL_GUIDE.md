# Technical guide

This document describes the implementation that is present in this repository. It deliberately separates implemented behavior from UI fields or legacy-compatible arguments that do not currently affect rendering.

## 1. Runtime architecture

```
Browser UI
  │  HTTP + Server-Sent Events
  ▼
Go HTTP server and job manager
  │  one Python subprocess per submitted job
  ▼
Python render backend
  ├─ FFmpeg rawvideo decoder
  ├─ optional RIFE interpolation
  ├─ optional AI upscaling
  └─ FFmpeg encoder and source audio/subtitle mapping
```

The Go executable embeds the web assets. It resolves the repository root, creates a uv-managed runtime under `runtime/venv`, and starts an HTTP server on `DASIWA_PORT` (default `8612`). The worker launches `backend/rve-backend.py` for each job and converts backend output into job status, logs, progress, resolved video-codec state, and JPEG preview data.

The renderer is not an in-process Go ML implementation. Go owns orchestration and the local API; Python owns PyTorch, TensorRT, ONNX Runtime, model loading, and pixel processing.

## 2. Installation and health checks

The Runtime UI/API creates a clean virtual environment and installs `backend/requirements.txt` with uv. The intended Python version is 3.12, falling back to 3.11. The installed inference stack includes CUDA 13.2 PyTorch, Torch-TensorRT, TensorRT, ONNX Runtime GPU, Spandrel, OpenCV, Safetensors, and PySceneDetect.

`POST /api/runtime/check` runs `rve-backend.py --list-backends` and reports whether the backend script, Python environment, PyTorch, TensorRT, Safetensors, Spandrel, ONNX Runtime, OpenCV, and FFmpeg are available.

The runtime is CUDA-oriented. Do not interpret the UI's device options (`mps`, `xpu`) or detected NCNN entry as working alternative inference backends: the production backend accepts only `tensorrt`, `pytorch`, or `onnxruntime` modes, and its shipped dependency set is CUDA/NVIDIA based.

## 3. Local input and output

The application uses paths available to the machine that runs the server.

- `GET /api/browse` lists directories and supported video files.
- `GET /api/search-files` performs recursive filename search below a chosen path.
- `GET /api/stream` serves a selected local source for the browser player.
- `GET /api/probe` uses FFprobe to retrieve source metadata.

No HTTP video upload endpoint exists. This avoids an extra copy of large local files and upload-size limits.

The Go worker accepts these output containers: MP4, MKV, WebM, MOV, AVI, FLV, TS, and M4V. The backend derives final dimensions from requested scale/override scale and rounds them to even values for common YUV codecs.

## 4. Rendering pipeline

### 4.1 Frame streaming

The renderer starts FFmpeg as a raw BGR24 reader and an independent rawvideo writer. The reader, AI processor, and writer communicate through bounded queues of three frames. This provides backpressure and avoids complete decoded, interpolated, or upscaled image-sequence directories.

Each output frame follows this order:

1. Select source frame A/B from the timestamp schedule.
2. If needed, produce an interpolated frame with RIFE.
3. If an upscaler is selected, run AI upscaling.
4. If the model output does not exactly match the requested dimensions, use Lanczos only for the final exact-size adjustment.
5. Write BGR24 rawvideo to FFmpeg, which encodes the output while mapping source audio/subtitle streams.

The final Lanczos adjustment is not a replacement for a selected AI upscaler. In particular, a native 2x model at a 4x target runs twice before any final dimension adjustment.

### 4.2 Frame interpolation

Interpolation runs only if `target_fps > source_fps`. The scheduler is timestamp driven rather than fixed to 2x/30 FPS:

- For each output timestamp, it calculates the source-frame position.
- It selects the surrounding source-frame pair.
- It passes the fractional position as the RIFE timestep.

That supports ratios such as 24→60 and 29.97→60. If target FPS is missing or not greater than source FPS, no interpolation model is loaded and source cadence is retained.

The current built-in interpolation model is **RIFE v4.26 General**. RIFE Heavy is not in the available-model list and should not be documented as a selectable option.

### 4.3 Upscaling formats and execution modes

The production loader accepts:

- `.safetensors`: loaded through Spandrel; scale is taken from the model descriptor.
- `.onnx`: executed through ONNX Runtime GPU. With TensorRT selected, the renderer confirms that `TensorrtExecutionProvider` remains active after session creation.

For Safetensors with TensorRT selected, the underlying model is compiled with Torch-TensorRT. A static full-frame engine is attempted when appropriate. On a compile failure, automatic selection falls back to tiled inference. TensorRT tile batches are padded to a static batch shape; only the original tile count is used in the reconstructed output.

For CUDA Safetensors paths, the iterative flow keeps the first AI pass as a CUDA tensor and feeds it directly into the second pass. The frame is converted back to CPU BGR only after the final AI pass.

## 5. 2x, 4x, tiles, and resource safety

Models are trained for a native scale. The renderer uses this rule:

| Requested output | Selected native model | Actual processing |
|---|---|---|
| 2x | 2x | One AI pass |
| 4x | 4x | One AI pass |
| 4x | 2x | Two AI passes: source → AI 2x → AI 2x |

Iterative 4x has a large second-pass cost because it processes the first pass's higher-resolution output. For automatic 2x→4x TensorRT jobs, the backend starts with 256-pixel tile cores and retries 128-pixel cores when compilation fails. An explicit `--tilesize` remains an expert override.

Before worker setup, the renderer applies two host-safety checks:

1. Output must be at most 33,177,600 pixels (8K UHD: 7680×4320).
2. Available RAM must cover a conservative estimate for bounded source/output rawvideo buffers plus 512 MiB headroom.

This means 1080p→4x is allowed, while 4K→4x (15360×8640) is rejected before ML allocations begin. The limit is intentional: it prevents pathological memory requests from freezing or shutting down the desktop.

## 6. Built-in models and Auto routing

`GET /api/runtime/models` returns model metadata including category, content subcategory, local destination, download URL, and presence state. Downloads write to `models/` through a `.part` file and atomically rename only after completion.

Auto routing works from content type and requested model scale. The first preferred model in each content/scale bucket is selected, with manual dropdown selection taking precedence.

| Content type | Typical 2x Auto route | Typical 4x Auto route |
|---|---|---|
| Anime | AnimeJaNai HD V3 Compact | NomosUni SPAN Multi-JPEG |
| Mixed | AnimeJaNai HD V3 Compact | NomosUni SPAN Multi-JPEG |
| Realism | Public RealPLKSR Restoration | ClearRealityV1 SPAN |

Additional built-in options include AnimeSharp, HFA2k LUDVAE, UltraSharpV2-Lite, Nomos WebPhoto, RealPLKSR GAN, and slower HAT-L variants. Availability remains local: a model must be present or downloaded before it can be used.

## 7. Encoding, containers, audio, and subtitles

The backend queries FFmpeg's encoder list and runs a small test encode before accepting a requested/automatic video encoder. The Auto preference depends on the selected container.

Supported encoder presets include NVENC H.264, NVENC HEVC, NVENC AV1, libx264, libx265, SVT-AV1, VP9, ProRes, and FFV1. Actual availability depends on the installed FFmpeg binary and GPU driver.

Audio can be copied, AAC encoded, MP3 encoded, or Opus encoded. Subtitles can be copied or converted to SRT, ASS, or WebVTT.

### WebM normalization

WebM has stricter codec rules. Before expensive rendering begins, the backend changes incompatible settings:

- incompatible video codec → VP9;
- incompatible copied/requested audio → Opus;
- incompatible copied/requested subtitle → WebVTT.

This avoids discovering an invalid muxer combination only after AI processing has completed.

## 8. UI, jobs, progress, and previews

Jobs are submitted with `POST /api/jobs`. The manager records job state, bounded logs, elapsed timestamps, output path, resolved codec, and current live-preview JPEG. The job's Server-Sent Events endpoint is `GET /api/jobs/{id}/events`; cancellation is `POST /api/jobs/{id}/cancel`.

The backend emits progress after the first frame and then every ten frames. The UI can show parseable frame progress and throughput. It emits a live JPEG preview every ten processed frames.

After completion, `GET /api/jobs/{id}/preview` creates a browser-compatible proxy: the first 20 seconds, 12 FPS, width up to 480 pixels, no audio, H.264 MP4. This proxy is a convenience preview, not the final output.

## 9. Current non-features and inert compatibility fields

The UI/request type exposes several settings that are not part of the current renderer's effective processing path. They should be treated as unavailable rather than advertised features.

| Item | Current behavior |
|---|---|
| Denoise | No dedicated denoise model or stage exists. |
| HDR mode | Argument is accepted, but no HDR conversion/tone-map path is implemented. |
| Scene detection method/threshold | Arguments are accepted, but no scene-detection branch is called by the renderer. |
| Ensemble | Argument is accepted, but no ensemble inference is implemented. |
| Dynamic optical flow | Argument is accepted, but no dynamic-flow behavior is implemented. |
| Slow motion mode | Argument is accepted; higher target FPS is the actual interpolation control. |
| Start/end trim time | Arguments are accepted, but decode is not trimmed by the current streaming command. |
| Custom encoder field | Argument is accepted, but encoder selection uses the preset mapping. |
| TensorRT dynamic-shape/profile flags | Arguments are accepted; the Safetensors TensorRT path compiles fixed sample/tile shapes. |
| Multi-file batch jobs | Not implemented. Submit one source video per job. |

## 10. API summary

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Server health response |
| GET | `/api/options` | UI-selectable container/codec/options metadata |
| GET | `/api/runtime/status` | Runtime/OS/tool status |
| POST | `/api/runtime/check` | Backend dependency health check |
| POST / GET SSE | `/api/runtime/install`, `/api/runtime/install/stream` | Install runtime |
| GET | `/api/runtime/models` | Built-in model metadata and local presence |
| POST / GET SSE | `/api/models/download`, `/api/models/download/stream` | Download requested models |
| GET | `/api/browse`, `/api/search-files` | Local file navigation/search |
| GET | `/api/probe` | FFprobe metadata for a local video |
| GET | `/api/stream` | Serve a local source file to the browser |
| POST | `/api/jobs` | Start one render job |
| GET | `/api/jobs/{id}` | Fetch job state |
| GET | `/api/jobs/{id}/events` | Job Server-Sent Events |
| GET | `/api/jobs/{id}/live-preview` | Current JPEG preview |
| GET | `/api/jobs/{id}/preview` | Completed-job MP4 proxy preview |
| POST | `/api/jobs/{id}/cancel` | Cancel a job |

## 11. Development verification

Run the same checks used for backend and Go changes:

```bash
runtime/venv/bin/python -m pytest backend/tests -q
go test ./...
go build ./cmd/dasiwa-true-video-enhancer
```

For a source-only Python backend change, the next spawned job loads the new Python code. Go changes and embedded browser assets require rebuilding the user-facing Go binary.
