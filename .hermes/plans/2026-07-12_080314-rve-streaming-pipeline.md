# REAL-Video-Enhancer-Style Streaming Pipeline Implementation Plan

> **For Hermes:** Implement this plan task-by-task with strict TDD and verify every stage on the RTX 5090.

**Goal:** Replace the serial PNG intermediary pipeline with a bounded FFmpeg rawvideo streaming pipeline so decode, AI processing, and encode overlap and no full-frame PNG sequences are written to disk.

**Architecture:** FFmpeg decodes source frames to BGR24 rawvideo. A bounded reader queue feeds a timestamp-driven processor that performs RIFE and upscaling in one pass. A bounded writer queue feeds FFmpeg rawvideo stdin while audio/subtitles are mapped from the source. Backpressure bounds RAM; shutdown and errors propagate across all stages.

**Tech Stack:** Python 3.12, FFmpeg/ffprobe, OpenCV/NumPy, PyTorch/TensorRT, queue/threading/subprocess, pytest, Go worker integration.

---

## Current context

- `backend/rve-backend.py:579-633` currently writes source, generated RIFE, and upscaled PNG sequences before final encoding.
- `backend/upscale_inference.py:352-382` performs serial `imread -> inference -> imwrite`.
- REAL-Video-Enhancer commit `fa72f83e5958d50510628a6638e45bc8f73a1c38` uses independent FFmpeg read, render, and write threads with bounded queues.
- Full-frame AnimeSharpV4 TensorRT inference is verified at 1088x1920; pure inference is about 0.442 seconds/frame. Streaming removes pipeline overhead but cannot exceed the model's compute ceiling.
- Existing codec/container normalization, target-FPS timestamp semantics, model routing, preview, progress, and audio/subtitle behavior must remain compatible.

## Task 1: Add rawvideo reader/writer primitives

**Files:**
- Create: `backend/streaming_pipeline.py`
- Create: `backend/tests/test_streaming_pipeline.py`

**TDD steps:**
1. Add failing tests for exact BGR24 frame sizing, EOF handling, bounded queues, writer ordering, and child-process error propagation.
2. Run `runtime/venv/bin/python -m pytest backend/tests/test_streaming_pipeline.py -q`; expect failures because primitives do not exist.
3. Implement FFmpeg decode command generation and raw frame parsing.
4. Implement FFmpeg encode command generation with source audio/subtitle mapping.
5. Implement reader/writer worker threads, sentinels, bounded queues, cancellation, stderr capture, and deterministic cleanup.
6. Re-run focused tests; expect pass.

## Task 2: Add timestamp-driven streaming frame scheduler

**Files:**
- Modify: `backend/streaming_pipeline.py`
- Modify: `backend/tests/test_streaming_pipeline.py`

**TDD steps:**
1. Add failing tests for 24->60 timestamps, source-frame passthrough, fractional RIFE timesteps, final-frame behavior, and no-interpolation cadence.
2. Implement a two-frame sliding window; never retain the full video.
3. Emit exact CFR output frames in order without writing intermediates.
4. Verify focused tests.

## Task 3: Fuse RIFE and upscaler processing

**Files:**
- Modify: `backend/streaming_pipeline.py`
- Modify: `backend/upscale_inference.py`
- Modify: `backend/tests/test_streaming_pipeline.py`
- Modify: `backend/tests/test_upscale_inference.py`

**TDD steps:**
1. Add failing tests proving each scheduled frame flows directly through optional RIFE then optional upscaler and resize.
2. Add array/tensor conversion helpers that avoid PNG and avoid duplicate conversions.
3. Preserve full-frame TensorRT first with safe tiled fallback and static ONNX handling.
4. Emit previews periodically from processed output and progress after successful queueing.
5. Verify focused tests.

## Task 4: Replace production render orchestration

**Files:**
- Modify: `backend/rve-backend.py:524-633`
- Create/modify: `backend/tests/test_backend_streaming_integration.py`

**TDD steps:**
1. Add failing command/integration tests verifying no `%08d.png`, `TemporaryDirectory`, `extract_source_frames`, or `process_frames` path is used by `render()`.
2. Wire model setup before stream startup.
3. Start bounded decode and encode workers, run the fused processor, and close models/processes in `finally`.
4. Preserve requested/actual technique logging and JSON success output.
5. Verify focused tests.

## Task 5: Error handling and compatibility

**Files:**
- Modify: `backend/streaming_pipeline.py`
- Modify: `backend/rve-backend.py`
- Modify tests above

**TDD steps:**
1. Add tests for decoder failure, encoder failure, cancellation, broken pipe, unavailable stream, no audio/subtitles, WebM codec normalization, and odd dimensions.
2. Ensure every child is terminated and joined on failure.
3. Ensure output is removed/left clearly failed rather than reported successful.
4. Verify all backend tests.

## Task 6: Real execution and throughput verification

**Commands:**
- `runtime/venv/bin/python -m pytest backend/tests -q`
- `go test ./...`
- `runtime/venv/bin/python -m py_compile backend/*.py`
- Run a short no-AI rawvideo roundtrip and compare frame count/FPS/resolution with ffprobe.
- Run AnimeSharpV4 TensorRT on the real 1088x1920 sample, first 2-3 seconds, recording decode/AI/encode FPS, GPU utilization, temp disk use, output frame count, and visual frame checks.
- Run RIFE 24->60 plus AnimeSharp on a short clip and verify exact 60 fps output, audio presence, duration, and processed-frame previews.
- Confirm `/tmp/dasiwa-rife-*` does not grow with PNG sequences.

## Task 7: Build and app smoke test

**Files:**
- Use existing `build.sh`; modify only if verification exposes a real issue.

**Commands:**
- `./build.sh`
- Start root binary and verify `GET /api/health` and embedded UI.
- Confirm root and `dist/dasiwa-true-video-enhancer-linux-amd64` binaries match.

## Acceptance criteria

- No source/RIFE/upscale PNG intermediary sequence in the production path.
- Decode, processing, and encode use bounded queues and overlap in time.
- Peak RAM is bounded by queue capacity rather than video length.
- RIFE arbitrary target-FPS semantics remain exact.
- Upscale-only, RIFE-only, combined, and passthrough pipelines work.
- Audio/subtitles and container-specific codec normalization remain functional.
- Progress reports processed/queued frames, not temporary files.
- TensorRT technique and fallback are logged truthfully.
- Tests, builds, real FFmpeg smoke tests, and RTX 5090 smoke tests pass.

## Risks and tradeoffs

- Raw BGR24 at 4K/8K is bandwidth-heavy; queues must stay small (typically 2-4 frames).
- NVENC and TensorRT share GPU resources; overlap may improve wall time but can reduce instantaneous inference speed.
- Python raw pipes still require GPU-to-host output copies. A future CUDA/NVENC zero-copy path is separate work and not required for this migration.
- Exact 20-30 fps cannot be guaranteed for RCAN at 1088x1920 because measured model inference itself is about 0.442 seconds/frame. Success is parity in pipeline architecture and elimination of avoidable I/O, not fabricated benchmark parity.
