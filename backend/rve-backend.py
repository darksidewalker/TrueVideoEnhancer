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
    if not video:
        raise RuntimeError("input contains no video stream")
    fps = _ratio_to_float(video.get("avg_frame_rate", "0/1")) or _ratio_to_float(video.get("r_frame_rate", "0/1"))
    if fps <= 0:
        raise RuntimeError("could not determine source FPS")
    width = int(video.get("width", 0) or 0)
    height = int(video.get("height", 0) or 0)
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0.0)
    return {"fps": fps, "width": width, "height": height, "duration": duration}


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
    content_type = (args.content_type or "mixed").strip().lower()
    name = Path(model_path).name.lower()
    if content_type != "anime" and "heavy" in name:
        general = Path(model_path).with_name("rife_v4.26.safetensors")
        if general.exists():
            log("WARN", "Heavy RIFE is anime-specialized; switching to general RIFE for non-anime content", detail=f"content_type={content_type} selected={Path(model_path).name} using={general.name}")
            return str(general)
        log("WARN", "Heavy RIFE is anime-specialized but general RIFE model is missing", detail=f"content_type={content_type} selected={Path(model_path).name}")
    elif content_type == "anime" and "heavy" in name:
        log("TECH", "Anime content selected: keeping RIFE Heavy Anime model")
    elif content_type == "anime":
        log("WARN", "Anime content selected but non-heavy RIFE model is configured", detail=f"selected={Path(model_path).name}")
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
        if (out_idx + 1) % 50 == 0 or out_idx + 1 == out_count:
            log("PIPE", f"wrote {out_idx + 1}/{out_count} frames")
    return out_count


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


def audio_args(audio_encoder: str, audio_bitrate: str | None) -> list[str]:
    enc = (audio_encoder or "copy_audio").strip()
    if enc in {"", "copy", "copy_audio"}:
        return ["-c:a", "copy"]
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
    output_fps, factor = derive_target_fps(info["fps"], args.target_fps)
    output_width, output_height = derive_output_resolution(
        info["width"], info["height"], args.scale, args.override_upscale_scale,
    )
    log("RES", f"source={info['width']}x{info['height']} target={output_width}x{output_height}", detail=f"scale={args.override_upscale_scale or args.scale} upscale_model={args.upscale_model or 'none'}")
    if args.interpolate_model and factor <= 1.0:
        log("WARN", "interpolation model selected but no FPS increase requested; copying source cadence")
    if not args.interpolate_model:
        raise RuntimeError("--interpolate_model is required for frame interpolation")

    with tempfile.TemporaryDirectory(prefix="dasiwa-rife-") as td:
        temp = Path(td)
        src = extract_source_frames(args.input, temp / "source")
        interpolator = build_interpolator(args, info["width"], info["height"])
        count = write_target_frames(src, temp / "frames", info["fps"], output_fps, interpolator)
        encode_video(
            args.input, args.output, temp / "frames", output_fps,
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
        print(f"[{'✓' if ok else '✗'}] {name}")


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
    parser.add_argument("--video_encoder_preset", default="fast")
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
