# DaSiWa True Video Enhancer

**NVIDIA-first video restoration studio.** One-page Go shell for RIFE safetensors, TensorRT upscaling, denoise, and target-FPS smoothing.

![DaSiWa True Video Enhancer Preview](assets/preview.webp)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Features](#features)
- [UI Reference](#ui-reference)
  - [Source Panel (01)](#source-panel-01)
  - [Preview Panel (02)](#preview-panel-02)
  - [Queue Panel (03)](#queue-panel-03)
  - [Advanced Settings](#advanced-settings)
- [Presets Explained](#presets-explained)
- [Tooltips & Translations](#tooltips--translations)
- [Runtime Installation](#runtime-installation)
- [Model Management](#model-management)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

DaSiWa True Video Enhancer is a hybrid application: a **Go web frontend** that orchestrates **Python AI inference backends**. It provides a single-page interface to:

- Upscale videos using super-resolution models (2×, 4×)
- Interpolate frames using RIFE optical flow networks
- Denoise and restore degraded footage
- Smooth motion via target-FPS generation
- Encode output with hardware-accelerated codecs (NVENC)
- Apply intelligent multi-pass smart upscaling with RTX VFX
- Stream live encode previews in real-time
- Manage AI models with automatic downloads
- Browse local video files with search functionality

The app runs as a standalone web server. Open the URL in any modern browser to access the full studio.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              Web Browser                     │
│   index.html · app.js · tooltips.js         │
└──────────────┬──────────────────────────────┘
               │ HTTP + SSE (EventSource)
┌──────────────▼──────────────────────────────┐
│           Go Web Server                      │
│   cmd/dasiwa-true-video-enhancer/web/       │
│   • Static file serving                      │
│   • REST API (/api/*)                        │
│   • SSE streaming (/stream)                  │
│   • Live preview streaming                   │
│   • File browser                             │
└──────────────┬──────────────────────────────┘
               │ subprocess (python)
┌──────────────▼──────────────────────────────┐
│       Python Runtime (uv-managed)            │
│   • PyTorch / TensorRT inference             │
│   • FFmpeg encoding                          │
│   • Scene detection (PySceneDetect)          │
│   • Model management                         │
│   • RTX upscale pipeline                     │
│   • Live preview frame emission              │
└─────────────────────────────────────────────┘
```

### Key Design Decisions

- **Isolated Python environment**: Uses `uv` to manage dependencies — never touches your system Python
- **GPU-first**: Optimized for NVIDIA GPUs with CUDA/TensorRT acceleration
- **Fallback support**: PyTorch CPU/CUDA works when TensorRT is unavailable
- **Cross-platform**: Works on Linux, Windows, macOS (with appropriate drivers)
- **Live preview**: Real-time base64-encoded JPEG frames streamed from Python to Go to browser
- **Portable runtime**: Self-contained Python environment managed by uv
- **Intelligent upscaling**: Multi-pass RTX VFX pipeline with automatic model selection

---

## Quick Start

### Prerequisites

1. **NVIDIA GPU** with driver ≥ 525 (for TensorRT) or ≥ 470 (for CUDA)
2. **FFmpeg** installed on the system (for encoding)
3. **Git** (for cloning)
4. **Go 1.24+** (for building the web server)

### Installation

```bash
# Clone the repository
git clone https://github.com/darksidewalker/DaSiWa-TrueVideoEnhancer.git
cd DaSiWa-TrueVideoEnhancer

# Build the web server
go build -o dasiwa-true-video-enhancer ./cmd/dasiwa-true-video-enhancer

# Run (installs Python runtime automatically on first launch)
./dasiwa-true-video-enhancer
```

The server starts at `http://localhost:8612`. On first run, click **Install Runtime** in the UI to set up the Python environment.

### Docker Deployment

```bash
docker build -t da-si-wa-tve .
docker run --gpus all -p 8080:8612 da-si-wa-tve
```

---

## Features

### Core Capabilities

- **Video Upscaling**: 2× and 4× resolution enhancement with auto-selected models per content type
- **Frame Interpolation**: RIFE optical flow network for smooth motion (target FPS generation)
- **Hardware Encoding**: NVENC GPU-accelerated encoding (H.264, H.265, AV1)
- **Multi-Container Support**: MP4, MKV, WebM, MOV, AVI, FLV, MPEG-TS, M4V
- **Audio/Subtitle Handling**: Copy, AAC, Opus, MP3, SRT, ASS, WebVTT
- **Smart Upscaling Pipeline**: Intelligent multi-pass RTX VFX with Lanczos fallback
- **Live Preview**: Real-time encode preview during processing
- **File Browser**: Navigate local directories to select videos
- **Model Management**: Automatic download and management of AI models
- **Scene Detection**: PySceneDetect integration for scene-aware processing
- **HDR Mode**: Preserve high-dynamic-range color through the pipeline
- **Slomo Mode**: Lengthen video by inserting interpolated frames
- **Ensemble Processing**: Multiple inference passes for higher quality
- **Dynamic Optical Flow**: Per-frame adaptive flow estimation
- **Benchmark Mode**: Timing metrics for hardware comparison
- **Dry-Run**: Build command without executing (safe testing)
- **Segment Selection**: Process only specific time ranges (start/end time)
- **Device Selection**: CUDA, MPS (Apple Silicon), XPU (Intel Arc)
- **VRAM Optimization**: UHD/8K VRAM saver for ultra-high resolutions
- **Custom Encoder Args**: Expert override for FFmpeg flags
- **Auto Fallback**: Automatically selects best available encoder per container

### Technical Highlights

- **SSE Streaming**: Real-time progress updates via Server-Sent Events
- **Base64 Frame Protocol**: Cross-process signaling between Go worker and Python subprocess
- **Portable Runtime**: uv-managed Python 3.12 environment, no system dependencies
- **Health Checks**: Application health monitoring endpoint
- **Backend Diagnostics**: Comprehensive component checking (`--list-backends`)
- **Tooltip System**: Centralized tooltip dictionary with i18n support
- **localStorage Persistence**: User preferences saved across sessions
- **Cookie-Based State**: Last browsed directory path persisted

---

## UI Reference

The interface is divided into three main panels plus an Advanced Settings dialog.

### Source Panel (01)

Where you configure input and basic output parameters.

| Element | Description |
|---------|-------------|
| **Input video** | Full path to source video. Use the 📂 button to browse, or paste a path directly. |
| **Content type** | Visual style selector: *Anime* (line art), *Mixed* (animated + real), *Realism* (live-action). Affects auto-selected models. |
| **Container** | Output wrapper format. MP4 for compatibility, MKV for metadata/audio tracks. |
| **Target FPS** | Desired frame rate. Set higher than source for slow-motion effect. |
| **Scale** | Resolution multiplier: 1× (no change), 2× (double dimensions), 4× (quadruple). |
| **Presets** | Quality presets (see below). |
| **Prepare Job** | Submits the configured job to the processing queue. |

### Preview Panel (02)

Visual representation of neural processing stages. The animated timeline shows the conceptual pipeline:

1. **Decode** → Input video decoded to frames
2. **Denoise** → Noise reduction pass
3. **Upscale** → Super-resolution enhancement
4. **Interpolate** → Frame generation via optical flow
5. **Encode** → Hardware-encoded output

Click **Advanced Settings** for detailed controls.

### Queue Panel (03)

Live feed of job progress. Shows:

- Job ID and status (queued, running, done, error, cancelled)
- Real-time log output from the backend
- Error messages with context
- ETA and estimated file size
- Open output folder button

### Advanced Settings

Opens a comprehensive configuration dialog with grouped sections:

#### Backend & Models

| Setting | Options | Explanation |
|---------|---------|-------------|
| **Backend** | TensorRT / PyTorch CUDA | Computing engine. TensorRT = fastest (NVIDIA only). PyTorch = universal fallback. |
| **RIFE safetensors** | Auto / manual selection | Frame interpolation model. Heavy variant for anime, standard for live-action. |
| **Upscale model** | Auto / manual selection | Super-resolution model. Different variants optimized per content type. |

#### Encoding Parameters

| Setting | Default | Explanation |
|---------|---------|-------------|
| **CRF / CQ base** | 18 | Quality vs file size tradeoff. Lower = better quality, larger file. Range: 0–51. |
| **Video encoder** | H.264 NVENC | Compression codec. NVENC = GPU hardware accelerated. CPU variants work everywhere. |
| **Pixel format** | yuv420p (8-bit 4:2:0) | Color sampling. Use 10-bit for HDR content, 4:4:4 for maximum color detail. |
| **Audio encoder** | Copy audio | How to handle audio. Copy preserves original; AAC/Opus/MP3 re-encode. |
| **Audio bitrate** | 128 kbps | Audio quality. Higher = better fidelity but larger files. |
| **Subtitle handling** | Copy subtitle | Preserve or convert embedded subtitles (SRT, ASS, WebVTT). |
| **Tile size** | 0 (auto) | Image tiling for VRAM management. Larger tiles = faster but more memory. |

#### TensorRT Optimization

| Setting | Default | Explanation |
|---------|---------|-------------|
| **Optimization profile** | 3 (balanced) | Build-time optimization. Profile 1 = fastest build, Profile 5 = best runtime performance. |
| **Dynamic shapes** | Off | Handle variable input sizes without rebuilding the network. Slower initial build. |

#### Processing Toggles

| Toggle | Effect |
|--------|--------|
| **HDR colorspace mode** | Preserves high-dynamic-range color through the pipeline. Required for HDR sources. |
| **UHD/8K VRAM saver** | Reduces intermediate buffers for ultra-high-resolution processing. |
| **Slomo Mode** | Lengthens video by inserting interpolated frames between originals. |
| **Ensemble** | Multiple inference passes combined for higher quality. Slower but fewer artifacts. |
| **Dynamic optical flow** | Per-frame adaptive flow estimation. Better motion quality, slower processing. |
| **Benchmark Mode** | Reports timing metrics. Useful for comparing hardware configurations. |
| **Dry-run only** | Builds and displays the command without executing. Safe for testing. |
| **RTX Upscale** | Enable intelligent multi-pass smart upscaling with RTX VFX |

#### Device & Timing

| Setting | Purpose |
|---------|---------|
| **Device** | Compute accelerator: Auto, CUDA (NVIDIA), MPS (Apple Silicon), XPU (Intel Arc) |
| **PyTorch GPU ID** | Which GPU when multiple are present (0 = first) |
| **NCNN GPU ID** | Which GPU for NCNN-based inference paths |
| **Start/End time** | Optional: process only a segment of the video (in seconds) |
| **Custom encoder args** | Expert override: extra flags passed directly to FFmpeg |

---

## Presets Explained

Three quality tiers balance speed against output quality:

### Fast ⚡

- **Use case:** Quick previews, low-end hardware, batch processing
- **Behavior:** Uses lower-tier models, skips ensemble passes, minimal post-processing
- **Speed:** ~3–5× faster than Balanced
- **Quality:** Good enough for rough cuts and previews

### Balanced ⚖️ *(Default)*

- **Use case:** General-purpose processing, most content types
- **Behavior:** Standard model selections, single-pass inference, balanced encoding
- **Speed:** Baseline reference point
- **Quality:** Strong results for most use cases

### Best 🏆

- **Use case:** Final delivery, archival, professional output
- **Behavior:** Highest-tier models, ensemble passes, dynamic optical flow, aggressive denoising
- **Speed:** ~2–3× slower than Balanced
- **Quality:** Maximum possible output quality

**Note:** Presets are starting points. You can override individual settings in Advanced Settings while keeping the preset's baseline choices.

---

## Tooltips & Translations

Every field, button, and toggle has an associated tooltip providing explanation text.

### Adding Translations

Tooltips are defined in `web/tooltips.js` as a centralized dictionary. To add a new language:

```javascript
// In web/tooltips.js, extend LANGUAGES:
const LANGUAGES = {
  en: { name: "English", tooltips: {} },
  de: { name: "Deutsch", tooltips: {
    "btn-check-backend": { title: "Diagnose ausführen", body: "Prüft Python-Laufzeitumgebung, GPU-Verfügbarkeit und alle Backend-Komponenten." },
    // ... translate other keys as needed
  }},
  // Add more languages here
};
```

Only titles need translation. Technical bodies remain in English for precision.

### Switching Language at Runtime

```javascript
window.Tooltips.setLanguage("de");  // Switch to German
window.Tooltips.getLanguages();     // List available codes: ["en", "de", "fr", ...]
```

### How Tooltips Work

Each interactive element carries a `data-tooltip="id"` attribute pointing to a key in the tooltip dictionary. The `tooltips.js` module auto-registers hover handlers and renders floating cards with the translated title and detailed body text.

---

## Runtime Installation

The first time you run the app, click **Install Runtime** to set up the Python environment:

1. **uv** installs a managed Python 3.12 distribution
2. Dependencies are downloaded: PyTorch 2.12+, TensorRT, FFmpeg bindings
3. The environment is isolated — no system packages modified
4. Progress streams live in the Jobs panel

### Manual Runtime Setup

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate the project environment
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install torch torchvision tensorrt ffmpeg-python pyscenedetect
```

---

## Model Management

### Available Model Categories

| Category | Purpose | Variants |
|----------|---------|----------|
| **Upscaler** | Increase resolution (2×, 4×) | Anime, Mixed, Realism subcategories |
| **Interpolation** | Generate intermediate frames | Heavy (anime/detail), Standard (fast) |

### Model Selection Logic

When **Auto** is selected (default):

1. **Upscaler:** Picks the best model matching your Content Type + Scale setting
2. **Interpolation:** Picks Heavy variant for Anime, Standard for others

You can manually override either selection in Advanced Settings. Selected models persist across sessions via localStorage.

### Downloading Models

1. Click **Models** in the header
2. Missing models show a download arrow (↓)
3. Click **Download** next to any missing model
4. Progress streams in the Download dialog — do not close during download

---

## API Endpoints

The Go server exposes these endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Application health check |
| GET | `/api/options` | Supported encoder/container options |
| GET | `/api/runtime/status` | Python + GPU availability |
| POST | `/api/runtime/check` | Full diagnostic report |
| POST | `/api/runtime/install` | Install runtime (blocking) |
| GET | `/api/runtime/install/stream` | Install progress (SSE) |
| GET | `/api/runtime/models` | Available AI models |
| POST | `/api/jobs` | Submit a new job |
| GET | `/api/jobs/:id` | Get job status |
| GET | `/api/jobs/:id/events` | Live job events (SSE) |
| GET | `/api/jobs/:id/live-preview` | Live encode preview (JPEG) |
| GET | `/api/jobs/:id/preview` | Post-job preview (MP4) |
| POST | `/api/jobs/:id/cancel` | Cancel running job |
| GET | `/api/browse?path=` | Directory listing |
| GET | `/api/search-files?q=&path=` | File search |
| POST | `/api/open-folder` | Open output folder |
| POST | `/api/quit` | Graceful shutdown |
| GET | `/api/models/download/stream?models=` | Model download (SSE) |
| POST | `/api/models/download` | Model download (blocking) |
| GET | `/api/probe` | Probe video file info |
| GET | `/api/stream?path=` | Stream raw video file |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8612` | Web server listen port |
| `HOME_DIR` | OS default | File browser root directory |
| `PYTHON_PATH` | Auto-detected | Override Python interpreter path |
| `UV_PATH` | Auto-detected | Override uv binary path |
| `MODEL_DIR` | `~/.cache/dasiwa/models` | AI models storage location |

### Persistent Settings

User preferences (selected models, last browse path) are stored in:

- **Browser:** localStorage (model selections)
- **Cookie:** Last browsed directory path (`last_browse_path`)

---

## Troubleshooting

### Common Issues

| Symptom | Solution |
|---------|----------|
| "Python not found" | Click **Install Runtime** to set up uv-managed Python |
| "GPU not detected" | Verify NVIDIA drivers: `nvidia-smi`. TensorRT requires driver ≥ 525 |
| "Out of memory" | Enable **UHD/8K VRAM saver** or reduce Tile Size |
| Slow processing | Try **Fast** preset or switch from TensorRT to PyTorch CPU for debugging |
| Black output frames | Check CRF value (try 18–22), verify pixel format matches source |
| Audio issues | Ensure audio encoder supports your source format (copy_audio safest) |
| No live preview | Check that backend is properly installed and GPU is accessible |

### Diagnostic Commands

```bash
# Check backend components
curl http://localhost:8612/api/runtime/check

# List available models
curl http://localhost:8612/api/runtime/models

# Test FFmpeg
ffmpeg -version

# Check GPU
nvidia-smi
```

---

## Contributing

### Project Structure

```
DaSiWa-TrueVideoEnhancer/
├── cmd/dasiwa-true-video-enhancer/
│   ├── main.go              # Application entry point
│   └── web/
│       ├── index.html        # Single-page UI
│       ├── app.js            # Frontend logic (jobs, API, file browser)
│       ├── tooltips.js       # Tooltip & translation module
│       ├── style.css         # Dark theme styling
│       └── assets/           # Images and media assets
├── internal/                 # Go backend (handlers, workers, config)
│   ├── server/               # HTTP routes and handlers
│   ├── worker/               # Job manager and execution
│   ├── runtimecheck/         # Runtime probe and installation
│   └── utils/                # Utility functions
├── backend/                  # Python inference scripts
│   ├── rve-backend.py        # Main backend CLI
│   ├── rtx_upscaler.py       # Smart upscaling pipeline
│   └── src/                  # Legacy backend modules
├── runtime/                  # Installed Python environment
│   └── venv/                 # uv-managed virtual environment
└── go.mod                    # Go module definition
```

### Development Workflow

```bash
# Watch for changes and rebuild
go build -watch ./cmd/dasiwa-true-video-enhancer

# Run with hot-reload
go run ./cmd/dasiwa-true-video-enhancer
```

### Adding Features

1. Extend the UI in `index.html` with proper `data-tooltip` attributes
2. Add handler logic in `internal/server/`
3. Update `app.js` for frontend interactions
4. Document new fields in this README under the relevant section

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

**Built with ❤️ by DaSiWa** — Restoring video, one frame at a time.
