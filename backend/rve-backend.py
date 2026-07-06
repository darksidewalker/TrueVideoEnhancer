#!/usr/bin/env python3
"""DaSiWa TrueVideoEnhancer backend entrypoint.

This is the CLI that the Go worker starts.  It keeps target-FPS handling in
one place: probe the source FPS, derive the interpolation cadence, generate a
monotonic output frame sequence, then mux it at the requested target FPS.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Iterable

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - reported by --list-backends
    cv2 = None
    np = None

try:
    import torch
except Exception:  # pragma: no cover - reported by --list-backends
    torch = None

try:
    import tensorrt as trt  # noqa: F401
    HAS_TRT = True
except Exception:
    HAS_TRT = False

try:
    import safetensors  # noqa: F401
    HAS_SAFETENSORS = True
except Exception:
    HAS_SAFETENSORS = False

try:
    import nvvfx  # noqa: F401
    HAS_NVVFX = True
except Exception:
    HAS_NVVFX = False

HAS_CUDA = False
try:
    import torch
    HAS_CUDA = torch.cuda.is_available() if torch is not None else False
except ImportError:
    pass

try:
    from rve_backend import ModelLoader
except Exception as exc:  # pragma: no cover - full error emitted on actual use
    ModelLoader = None
    MODEL_LOADER_ERROR = exc
else:
    MODEL_LOADER_ERROR = None

VERSION = "0.2.0"
SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}

_COLOR = sys.stderr.isatty() or os.environ.get("FORCE_COLOR") in {"1", "true", "yes"}
_COLORS = {
    "RESET": "\033[0m", "BOLD": "\033[1m", "DIM": "\033[2m",
    "PIPE": "\033[96m", "MODEL": "\033[95m", "TECH": "\033[94m",
    "FPS": "\033[92m", "RES": "\033[92m", "FRAMES": "\033[96m",
    "ENCODE": "\033[94m", "DONE": "\033[92m", "WARN": "\033[93m", "ERROR": "\033[91m",
}


def log(tag: str, message: str, *, detail: str | None = None) -> None:
    color = _COLORS.get(tag, "") if _COLOR else ""
    reset = _COLORS["RESET"] if _COLOR else ""
    dim = _COLORS["DIM"] if _COLOR else ""
    line = f"{color}[{tag}]{reset} {message}"
    if detail:
        line += f" {dim}{detail}{reset}"
    print(line, file=sys.stderr)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found on PATH")
    return path


def _ffprobe() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise RuntimeError("ffprobe not found on PATH")
    return path


def _ratio_to_float(value: str) -> float:
    try:
        return float(Fraction(value))
    except Exception:
        return 0.0


def probe_video(input_path: str) -> dict:
    cmd = [
        _ffprobe(), "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", input_path,
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
    data = json.loads(proc.stdout or "{}")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    subtitle = next((s for s in data.get("streams", []) if s.get("codec_type") == "subtitle"), None)
    if not video:
        raise RuntimeError("input contains no video stream")
    fps = _ratio_to_float(video.get("avg_frame_rate", "0/1")) or _ratio_to_float(video.get("r_frame_rate", "0/1"))
    if fps <= 0:
        raise RuntimeError("could not determine source FPS")
    width = int(video.get("width", 0) or 0)
    height = int(video.get("height", 0) or 0)
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
        "audio_codec": audio.get("codec_name", "") if audio else "",
        "subtitle_codec": subtitle.get("codec_name", "") if subtitle else "",
    }


def derive_target_fps(source_fps: float, target_fps: float | None) -> tuple[float, float]:
    if not target_fps or target_fps <= 0:
        return source_fps, 1.0
    if target_fps <= source_fps + 1e-6:
        log("FPS", f"target_fps={target_fps:g} <= source_fps={source_fps:.6g}; keeping source FPS")
        return source_fps, 1.0
    factor = target_fps / source_fps
    log("FPS", f"source={source_fps:.6g}, target={target_fps:.6g}, interpolation_factor={factor:.6g}")
    return target_fps, factor


def resolve_model_path(path_or_id: str) -> str:
    if not path_or_id:
        return ""
    p = Path(path_or_id).expanduser()
    if p.exists():
        return str(p)
    repo_model = Path("models") / path_or_id
    if repo_model.exists():
        return str(repo_model)
    repo_model_st = Path("models") / f"{path_or_id}.safetensors"
    if repo_model_st.exists():
        return str(repo_model_st)
    raise FileNotFoundError(f"RIFE model not found: {path_or_id}")


def extract_source_frames(input_path: str, frame_dir: Path) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(frame_dir / "%08d.png")
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", input_path, "-vsync", "0", pattern]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {proc.stderr.strip()}")
    frames = sorted(frame_dir.glob("*.png"))
    if len(frames) < 1:
        raise RuntimeError("no frames extracted")
    log("FRAMES", f"extracted={len(frames)}", detail=f"dir={frame_dir}")
    return frames


def _to_tensor(frame_bgr):
    if torch is None:
        raise RuntimeError("PyTorch is required for RIFE interpolation")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0


def _to_bgr(tensor):
    arr = tensor.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def select_interpolation_model(args) -> str:
    model_path = resolve_model_path(args.interpolate_model)
    name = Path(model_path).name.lower()
    if "heavy" in name:
        general = Path(model_path).with_name("rife_v4.26.safetensors")
        if general.exists():
            log("WARN", "RIFE Heavy is disabled due to known errors; using general RIFE", detail=f"selected={Path(model_path).name} using={general.name}")
            return str(general)
        log("WARN", "RIFE Heavy is disabled but general RIFE model is missing", detail=f"selected={Path(model_path).name}")
    return model_path


def build_interpolator(args, width: int, height: int):
    model_path = select_interpolation_model(args)
    log("MODEL", f"interpolation={Path(model_path).name}", detail=f"path={model_path}")
    log("TECH", f"requested_backend={args.backend} precision={args.precision}", detail=f"profile={args.tensorrt_opt_profile} dynamic_shapes={args.tensorrt_dynamic_shapes} input={width}x{height}")
    if ModelLoader is None:
        raise RuntimeError(f"RIFE model loader unavailable: {MODEL_LOADER_ERROR}")
    loader = ModelLoader(device=args.device)
    return loader.load_interpolator(
        model_path,
        backend=args.backend,
        precision=args.precision,
        opt_profile=args.tensorrt_opt_profile,
        dynamic_shapes=args.tensorrt_dynamic_shapes,
        resolution=(height, width),
    )


def write_target_frames(source_frames: list[Path], output_dir: Path, source_fps: float,
                        target_fps: float, interpolator) -> int:
    """Generate a CFR frame sequence at target_fps from source frame times.

    For each output timestamp, choose the adjacent source pair and the exact
    fractional timestep between them.  This supports arbitrary target FPS such
    as 24->60 (2.5x), not only integer 2x/4x interpolation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required for video processing")

    total_duration = (len(source_frames) - 1) / source_fps if len(source_frames) > 1 else 1.0 / source_fps
    out_count = max(1, int(math.floor(total_duration * target_fps + 1e-6)) + 1)
    technique = getattr(interpolator, "technique", type(interpolator).__name__)
    log("PIPE", f"output_frames={out_count} target_fps={target_fps:.6g}", detail=f"interpolation={technique}")

    cache_index = -1
    frame_a = frame_b = None
    for out_idx in range(out_count):
        t = out_idx / target_fps
        pos = min(t * source_fps, len(source_frames) - 1)
        left = int(math.floor(pos))
        right = min(left + 1, len(source_frames) - 1)
        timestep = float(pos - left)

        if left != cache_index:
            frame_a = cv2.imread(str(source_frames[left]), cv2.IMREAD_COLOR)
            frame_b = cv2.imread(str(source_frames[right]), cv2.IMREAD_COLOR)
            cache_index = left
        elif right == left:
            frame_b = frame_a

        if frame_a is None or frame_b is None:
            raise RuntimeError(f"failed to read source frame pair {left}/{right}")

        if right == left or timestep <= 1e-5:
            out = frame_a
        elif timestep >= 1.0 - 1e-5:
            out = frame_b
        else:
            a = _to_tensor(frame_a)
            b = _to_tensor(frame_b)
            out = _to_bgr(interpolator.interpolate(a, b, timestep=timestep))

        out_path = output_dir / f"{out_idx + 1:08d}.png"
        if not cv2.imwrite(str(out_path), out):
            raise RuntimeError(f"failed to write output frame: {out_path}")
        emit_live_preview(out)
        if (out_idx + 1) % 50 == 0 or out_idx + 1 == out_count:
            log("PIPE", f"wrote {out_idx + 1}/{out_count} frames")
    return out_count


def write_source_cadence_frames(source_frames: list[Path], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(source_frames)
    log("PIPE", f"output_frames={total} target_fps=source", detail="interpolation=skipped")
    for idx, frame in enumerate(source_frames, start=1):
        shutil.copy2(frame, output_dir / f"{idx:08d}.png")
        if cv2 is not None:
            emit_live_preview(cv2.imread(str(frame), cv2.IMREAD_COLOR))
        if idx % 50 == 0 or idx == total:
            log("PIPE", f"wrote {idx}/{total} frames")
    return total


_PREVIEW_MARKER = "<PREVIEW>"


def emit_live_preview(frame) -> None:
    """Emit the latest processed frame as a base64-encoded JPEG on stderr.

    Format: ``<PREVIEW><base64-jpeg-bytes>\\n``.  The Go worker filters this
    marker from log output and stores the decoded bytes in memory for live
    streaming over HTTP.
    """
    if frame is None or cv2 is None:
        return
    height, width = frame.shape[:2]
    if width > 640:
        scale = 640 / float(width)
        frame = cv2.resize(frame, (640, max(2, int(height * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return
    import base64
    encoded = base64.b64encode(buf.tobytes()).decode("ascii")
    print(f"{_PREVIEW_MARKER}{encoded}", file=sys.stderr, flush=True)


def derive_output_resolution(width: int, height: int, scale: int, override_scale: int = 0) -> tuple[int, int]:
    factor = override_scale if override_scale > 0 else scale
    if factor <= 0:
        factor = 1
    out_w = max(2, int(round(width * factor)))
    out_h = max(2, int(round(height * factor)))
    # Most delivery codecs / yuv420 formats require even dimensions.  Keeping
    # this universal avoids NVENC/libx264 failures on odd-sized source videos.
    out_w += out_w % 2
    out_h += out_h % 2
    return out_w, out_h


def encoder_args(encoder: str, crf: str, pix_fmt: str) -> list[str]:
    enc = (encoder or "libx264").strip()
    quality = crf or "18"
    pix = pix_fmt or "yuv420p"
    if enc in {"x264_nvenc", "h264_nvenc"}:
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", quality, "-pix_fmt", pix]
    if enc in {"x265_nvenc", "hevc_nvenc"}:
        return ["-c:v", "hevc_nvenc", "-preset", "p5", "-cq", quality, "-pix_fmt", pix]
    if enc == "av1_nvenc":
        return ["-c:v", "av1_nvenc", "-preset", "p5", "-cq", quality, "-pix_fmt", pix]
    if enc == "libx265":
        return ["-c:v", "libx265", "-preset", "medium", "-crf", quality, "-pix_fmt", pix]
    if enc == "vp9":
        return ["-c:v", "libvpx-vp9", "-crf", quality, "-b:v", "0", "-pix_fmt", pix]
    if enc == "av1":
        return ["-c:v", "libsvtav1", "-crf", quality, "-preset", "6", "-pix_fmt", pix]
    if enc == "prores":
        return ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
    if enc == "ffv1":
        return ["-c:v", "ffv1", "-level", "3", "-pix_fmt", pix]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", quality, "-pix_fmt", pix]


def available_ffmpeg_encoders() -> set[str]:
    proc = _run([_ffmpeg(), "-hide_banner", "-encoders"])
    if proc.returncode != 0:
        log("WARN", "Could not query FFmpeg encoders; using conservative fallback", detail=proc.stderr.strip())
        return set()
    encoders: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def ffmpeg_codec_name(preset: str) -> str:
    enc = (preset or "").strip()
    if enc in {"x264_nvenc", "h264_nvenc"}:
        return "h264_nvenc"
    if enc in {"x265_nvenc", "hevc_nvenc"}:
        return "hevc_nvenc"
    if enc == "av1":
        return "libsvtav1"
    if enc == "vp9":
        return "libvpx-vp9"
    if enc == "vp8":
        return "libvpx"
    return enc


def container_name(output_video: str) -> str:
    return Path(output_video).suffix.lower().lstrip(".")


def container_video_candidates(container: str) -> list[str]:
    # Preference order: modern/efficient first, then broadly compatible fallbacks.
    if container == "webm":
        return ["av1_nvenc", "vp9", "vp8", "av1"]
    if container in {"mp4", "m4v"}:
        return ["av1_nvenc", "x265_nvenc", "x264_nvenc", "libx265", "libx264", "av1"]
    if container in {"mov", "mkv"}:
        return ["av1_nvenc", "x265_nvenc", "x264_nvenc", "libx265", "libx264", "av1", "prores", "ffv1"]
    if container == "ts":
        return ["x265_nvenc", "x264_nvenc", "libx265", "libx264"]
    if container == "flv":
        return ["x264_nvenc", "libx264"]
    if container == "avi":
        return ["x264_nvenc", "libx264", "ffv1"]
    return ["av1_nvenc", "x265_nvenc", "x264_nvenc", "av1", "libx265", "libx264"]


def is_video_preset_supported(preset: str, encoders: set[str]) -> bool:
    codec = ffmpeg_codec_name(preset)
    if encoders and codec not in encoders:
        return False
    proc = _run([
        _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1",
        "-frames:v", "1", "-c:v", codec, "-f", "null", "-",
    ])
    return proc.returncode == 0


def select_auto_video_encoder(output_video: str, encoders: set[str]) -> str:
    container = container_name(output_video)
    for preset in container_video_candidates(container):
        if is_video_preset_supported(preset, encoders):
            log("TECH", "auto video encoder selected", detail=f"container={container or 'unknown'} encoder={preset} ffmpeg={ffmpeg_codec_name(preset)}")
            return preset
    log("WARN", "No preferred encoder available; falling back to libx264", detail=f"container={container or 'unknown'}")
    return "libx264"


def normalize_video_encoder(output_video: str, encoder: str) -> str:
    requested = (encoder or "auto").strip()
    encoders = available_ffmpeg_encoders()
    if requested in {"", "auto"}:
        return select_auto_video_encoder(output_video, encoders)
    if is_video_preset_supported(requested, encoders):
        return requested
    fallback = select_auto_video_encoder(output_video, encoders)
    log("WARN", "Requested FFmpeg video encoder is unavailable; using auto fallback", detail=f"requested={requested} fallback={fallback}")
    return fallback


def webm_safe_settings(output_video: str, encoder: str, audio_encoder: str,
                       subtitle_encoder: str, input_audio_codec: str,
                       input_subtitle_codec: str) -> tuple[str, str, str]:
    """Normalize codec settings for WebM before the final mux.

    FFmpeg rejects H.264/H.265/AAC/etc. in WebM only after the expensive frame
    processing stage. Adjust WebM jobs before encoding so completed RIFE work is
    written instead of failing at header creation.
    """
    if container_name(output_video) != "webm":
        return encoder, audio_encoder, subtitle_encoder

    requested_video = (encoder or "libx264").strip()
    requested_audio = (audio_encoder or "copy_audio").strip()
    requested_subtitle = (subtitle_encoder or "copy_subtitle").strip()

    webm_video = {"vp8", "vp9", "av1", "av1_nvenc", "libvpx", "libvpx-vp9", "libsvtav1", "libaom-av1"}
    webm_audio = {"opus", "libopus", "vorbis", "libvorbis"}
    webm_subtitle = {"webvtt"}

    safe_video = requested_video
    if requested_video not in webm_video:
        safe_video = "vp9"
        log("WARN", "WebM does not support requested video codec; using VP9", detail=f"requested={requested_video}")

    safe_audio = requested_audio
    if requested_audio in {"", "copy", "copy_audio"}:
        if input_audio_codec and input_audio_codec not in {"opus", "vorbis"}:
            safe_audio = "opus"
            log("WARN", "WebM cannot copy input audio codec; using Opus", detail=f"input_audio={input_audio_codec}")
    elif requested_audio not in webm_audio:
        safe_audio = "opus"
        log("WARN", "WebM does not support requested audio codec; using Opus", detail=f"requested={requested_audio}")

    safe_subtitle = requested_subtitle
    if requested_subtitle in {"", "copy", "copy_subtitle"}:
        if input_subtitle_codec and input_subtitle_codec not in webm_subtitle:
            safe_subtitle = "webvtt"
            log("WARN", "WebM cannot copy input subtitle codec; using WebVTT", detail=f"input_subtitle={input_subtitle_codec}")
    elif requested_subtitle not in webm_subtitle:
        safe_subtitle = "webvtt"
        log("WARN", "WebM does not support requested subtitle codec; using WebVTT", detail=f"requested={requested_subtitle}")

    return safe_video, safe_audio, safe_subtitle


def normalize_encode_settings(output_video: str, encoder: str, audio_encoder: str,
                              subtitle_encoder: str, input_audio_codec: str,
                              input_subtitle_codec: str) -> tuple[str, str, str]:
    video = normalize_video_encoder(output_video, encoder)
    return webm_safe_settings(
        output_video,
        video,
        audio_encoder,
        subtitle_encoder,
        input_audio_codec,
        input_subtitle_codec,
    )


def audio_args(audio_encoder: str, audio_bitrate: str | None) -> list[str]:
    enc = (audio_encoder or "copy_audio").strip()
    if enc in {"", "copy", "copy_audio"}:
        return ["-c:a", "copy"]
    if enc == "opus":
        enc = "libopus"
    if enc == "vorbis":
        enc = "libvorbis"
    args = ["-c:a", enc]
    if audio_bitrate:
        args += ["-b:a", audio_bitrate]
    return args


def subtitle_args(subtitle_encoder: str) -> list[str]:
    enc = (subtitle_encoder or "copy_subtitle").strip()
    if enc in {"", "copy", "copy_subtitle"}:
        return ["-c:s", "copy"]
    return ["-c:s", enc]


def encode_video(input_video: str, output_video: str, frame_dir: Path, fps: float,
                 crf: str, encoder: str, pix_fmt: str, audio_encoder: str,
                 subtitle_encoder: str, audio_bitrate: str | None,
                 output_width: int, output_height: int):
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    fps_str = f"{fps:.6f}".rstrip("0").rstrip(".")
    cmd = [
        _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", fps_str,
        "-i", str(frame_dir / "%08d.png"),
        "-i", input_video,
        "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
        "-vf", f"scale={output_width}:{output_height}:flags=lanczos",
    ]
    cmd += encoder_args(encoder, crf, pix_fmt)
    cmd += audio_args(audio_encoder, audio_bitrate)
    cmd += subtitle_args(subtitle_encoder)
    cmd += ["-shortest", output_video]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.strip()}")
    log("ENCODE", f"video_encoder={encoder_args(encoder, crf, pix_fmt)[1]} audio={audio_encoder or 'copy'} subtitles={subtitle_encoder or 'copy'}", detail=f"quality={crf} pix_fmt={pix_fmt} fps={fps_str}")
    log("DONE", f"output={output_video}", detail=f"resolution={output_width}x{output_height} requested_encoder={encoder}")


def render(args) -> None:
    log("PIPE", "DaSiWa TrueVideoEnhancer backend start", detail=f"version={VERSION}")
    log("PIPE", f"input={args.input}", detail=f"output={args.output}")
    info = probe_video(args.input)
    log("TECH", "video_probe", detail=f"source={info['width']}x{info['height']} fps={info['fps']:.6g} duration={info['duration']:.3f}s")
    args.video_encoder_preset, args.audio_encoder_preset, args.subtitle_encoder_preset = normalize_encode_settings(
        args.output,
        args.video_encoder_preset,
        args.audio_encoder_preset,
        args.subtitle_encoder_preset,
        info.get("audio_codec", ""),
        info.get("subtitle_codec", ""),
    )
    # Emit resolved video codec immediately so the frontend can update the
    # output filename tag from "[auto]" to the actual codec in real-time.
    resolved_codec = encoder_args(args.video_encoder_preset, args.crf, args.video_pixel_format)[1]
    print(f"<VIDEO_CODEC>{resolved_codec}", file=sys.stderr, flush=True)
    output_fps, factor = derive_target_fps(info["fps"], args.target_fps)
    
    # Apply RTX upscale if enabled
    if hasattr(args, 'rtx_upscale') and args.rtx_upscale and args.upscale_model:
        log("UPSCALE", f"RTX upscale enabled with model: {args.upscale_model}")
        
        # Temporarily extract frames for upscaling
        temp_dir = tempfile.mkdtemp(prefix="dasiwa-upscale-")
        src_frames = extract_source_frames(args.input, Path(temp_dir) / "source")
        
        target_scale = args.override_upscale_scale or args.scale
        preview_cb = lambda frame: emit_live_preview(frame)
        
        count, out_w, out_h = apply_smart_upscale(
            source_frames=src_frames,
            upscale_model=args.upscale_model,
            target_scale=target_scale,
            output_dir=Path(temp_dir) / "frames",
            device_id=args.pytorch_gpu_id,
            preview_cb=preview_cb,
            enable_final_rtx=getattr(args, 'enable_final_rtx', False),
        )
        
        output_width, output_height = out_w, out_h
        log("RES", f"After RTX upscale: {out_w}x{out_h}", detail=f"frames={count}")
        
        shutil.rmtree(temp_dir)
    else:
        output_width, output_height = derive_output_resolution(
            info["width"], info["height"], args.scale, args.override_upscale_scale,
        )
        log("RES", f"source={info['width']}x{info['height']} target={output_width}x{output_height}", 
            detail=f"scale={args.override_upscale_scale or args.scale} upscale_model={args.upscale_model or 'none'}")
    if args.interpolate_model and factor <= 1.0:
        log("WARN", "interpolation model selected but no FPS increase requested; skipping RIFE")
    if factor > 1.0 and not args.interpolate_model:
        raise RuntimeError("--interpolate_model is required for frame interpolation")

    with tempfile.TemporaryDirectory(prefix="dasiwa-rife-") as td:
        temp = Path(td)
        src = extract_source_frames(args.input, temp / "source")
        frame_dir = temp / "frames"
        if factor > 1.0:
            interpolator = build_interpolator(args, info["width"], info["height"])
            count = write_target_frames(src, frame_dir, info["fps"], output_fps, interpolator)
        else:
            count = write_source_cadence_frames(src, frame_dir)
        encode_video(
            args.input, args.output, frame_dir, output_fps,
            args.crf, args.video_encoder_preset, args.video_pixel_format,
            args.audio_encoder_preset, args.subtitle_encoder_preset, args.audio_bitrate,
            output_width, output_height,
        )
    print(json.dumps({"status": "success", "output": args.output, "target_fps": output_fps, "frames": count, "width": output_width, "height": output_height}))


def print_backends() -> None:
    checks = [
        ("PyTorch", torch is not None),
        ("TensorRT", HAS_TRT),
        ("safetensors", HAS_SAFETENSORS),
        ("OpenCV", cv2 is not None),
        ("FFmpeg", shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None),
    ]
    for name, ok in checks:
        print(f"[{'OK' if ok else 'XX'}] {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DaSiWa TrueVideoEnhancer RIFE backend")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--list-backends", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--backend", choices=["pytorch", "tensorrt", "onnxruntime"], default="tensorrt")
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--upscale_model", default="")
    parser.add_argument("--interpolate_model", default="")
    parser.add_argument("--target_fps", type=float, default=0.0)
    parser.add_argument("--content_type", default="mixed", choices=["anime", "mixed", "realism"])
    parser.add_argument("--crf", default="18")
    parser.add_argument("--video_encoder_preset", default="auto")
    parser.add_argument("--video_pixel_format", default="yuv420p")
    parser.add_argument("--audio_encoder_preset", default="copy")
    parser.add_argument("--subtitle_encoder_preset", default="copy")
    parser.add_argument("--audio_bitrate", default="")
    parser.add_argument("--tilesize", type=int, default=0)
    parser.add_argument("--tensorrt_dynamic_shapes", action="store_true")
    parser.add_argument("--tensorrt_opt_profile", type=int, default=3)
    parser.add_argument("--scene_detect_method", default="none")
    parser.add_argument("--scene_detect_threshold", type=float, default=0.0)
    parser.add_argument("--custom_encoder", default="")
    parser.add_argument("--override_upscale_scale", type=int, default=0)
    parser.add_argument("--hdr_mode", action="store_true")
    parser.add_argument("--UHD_mode", action="store_true")
    parser.add_argument("--slomo_mode", action="store_true")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--dynamic_scaled_optical_flow", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--start_time", type=float, default=0.0)
    parser.add_argument("--end_time", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pytorch_gpu_id", type=int, default=0)
    parser.add_argument("--ncnn_gpu_id", type=int, default=0)
    parser.add_argument("--enable_final_rtx", action="store_true", help="Apply optional final RTX VFX pass after upscaling")
    parser.add_argument("--rtx_upscale", action="store_true", help="Enable RTX VFX upscaling with Lanczos fallback")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(VERSION)
        return 0
    if args.list_backends:
        print_backends()
        return 0
    if not args.input or not args.output:
        parser.error("--input and --output are required")
    if Path(args.output).exists() and not args.overwrite:
        raise RuntimeError(f"output exists (use --overwrite): {args.output}")
    render(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
