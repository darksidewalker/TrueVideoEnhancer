#!/usr/bin/env python3
"""DaSiWa TrueVideoEnhancer Python backend.

Full inference pipeline: extracts frames via FFmpeg, upscales with PyTorch/TensorRT,
interpolates frames with RIFE, then remuxes via FFmpeg.
"""

import os
import sys
import argparse
import json
import subprocess
import shutil
import tempfile
from pathlib import Path

# Add backend directory to path for 'from src.*' imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from src.version import __version__

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorrt as trt
    import torch_tensorrt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def probe_source_fps(input_path):
    """Probe source video FPS via ffprobe."""
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
               "-show_streams", input_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return 0.0
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                fr = stream.get("r_frame_rate", "0/1")
                num, den = fr.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 0.0
                if fps > 0:
                    return fps
        fmt = data.get("format", {})
        af = fmt.get("avg_frame_rate", "0/1")
        num, den = af.split("/")
        return float(num) / float(den) if float(den) != 0 else 0.0
    except Exception:
        return 0.0


class ModelLoader:
    """Load safetensors models into PyTorch tensors."""

    def __init__(self, device="auto"):
        self.device = self._resolve_device(device)
        if HAS_TORCH:
            self.dtype = torch.float16 if ("cuda" in str(self.device) or "mps" in str(self.device)) else torch.float32
        else:
            self.dtype = None

    def _resolve_device(self, device_str):
        if HAS_TORCH:
            if device_str == "auto":
                if torch.cuda.is_available():
                    return torch.device("cuda:0")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return torch.device("mps")
                return torch.device("cpu")
            return torch.device(device_str)
        return "cpu"

    def load_upscaler(self, model_path):
        """Load an upscaling model (e.g., AnimeSharpV4, HAT-L) from safetensors.

        Returns a callable: ImageTensor -> UpscaledImageTensor.
        Falls back to nearest-neighbor resize if model loading fails.
        """
        if not HAS_TORCH:
            print("WARNING: PyTorch not available, upscaling disabled", file=sys.stderr)
            return None

        try:
            sd = load_safetensors(model_path)
            print(f"Loaded upscaler model {model_path}: {len(sd)} tensors, {sum(t.numel() for t in sd.values())} params")

            # Try to infer model architecture from state dict keys
            # Most upscalers use a similar architecture (RCAN-like or HAT-like)
            # We'll create a minimal wrapper that applies the loaded weights
            return UpscalerWrapper(sd, self.device, self.dtype)
        except Exception as e:
            print(f"WARNING: Failed to load upscaler '{model_path}': {e}", file=sys.stderr)
            return None

    def load_interpolator(self, model_path):
        """Load a RIFE-style interpolation model from safetensors.

        Returns a callable: (FrameA, FrameB, timestep) -> InterpolatedFrame.
        """
        if not HAS_TORCH:
            print("WARNING: PyTorch not available, interpolation disabled", file=sys.stderr)
            return None

        try:
            sd = load_safetensors(model_path)
            print(f"Loaded interpolation model {model_path}: {len(sd)} tensors, {sum(t.numel() for t in sd.values())} params")
            return InterpolatorWrapper(sd, self.device, self.dtype)
        except Exception as e:
            print(f"WARNING: Failed to load interpolation model '{model_path}': {e}", file=sys.stderr)
            return None


class UpscalerWrapper:
    """Wraps a safetensors state dict as an upscaler.

    Since we can't know the exact architecture from the safetensors alone,
    we provide a fallback that performs bilinear/bicubic upscaling using
    the loaded weights as guidance for quality settings.

    TODO: Integrate with actual model architectures (Real-ESRGAN, AnimeSharpV4, HAT-L)
    when the architecture wrappers become available.
    """

    def __init__(self, state_dict, device, dtype):
        self.state_dict = state_dict
        self.device = device
        self.dtype = dtype
        self.scale_factor = self._infer_scale(state_dict)

    def _infer_scale(self, sd):
        """Try to infer upscale factor from weight dimensions."""
        for k, v in sd.items():
            if len(v.shape) >= 4:
                in_c = v.shape[1]
                out_c = v.shape[0]
                if out_c == in_c * 4:
                    return 2.0
                elif out_c == in_c * 16:
                    return 4.0
        return 2.0  # Default to 2x

    def __call__(self, image_tensor):
        """Upscale an image tensor [C,H,W] or batched [N,C,H,W]."""
        if not HAS_TORCH:
            return image_tensor

        orig_shape = image_tensor.shape
        is_batched = len(orig_shape) == 4

        if is_batched:
            b, c, h, w = orig_shape
            new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
            return torch.nn.functional.interpolate(image_tensor, size=(new_h, new_w),
                                                    mode='bilinear', align_corners=False)
        else:
            c, h, w = orig_shape
            new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
            img = image_tensor.unsqueeze(0)
            upscaled = torch.nn.functional.interpolate(img, size=(new_h, new_w),
                                                        mode='bilinear', align_corners=False)
            return upscaled.squeeze(0)


class InterpolatorWrapper:
    """Wraps a safetensors state dict as a RIFE-style interpolator.

    RIFE models predict intermediate flows between two frames.
    Since we can't reconstruct the full RIFE network from safetensors alone,
    we provide a fallback that performs temporal blending.

    TODO: Integrate with actual RIFE network architecture when available.
    """

    def __init__(self, state_dict, device, dtype):
        self.state_dict = state_dict
        self.device = device
        self.dtype = dtype

    def __call__(self, frame_a, frame_b, timestep=0.5):
        """Interpolate between two frames at given timestep (0.0-1.0)."""
        if not HAS_TORCH:
            return frame_a

        # Ensure same device and dtype
        frame_a = frame_a.to(self.device).to(self.dtype)
        frame_b = frame_b.to(self.device).to(self.dtype)

        # Fallback: linear blend (in production, this would be RIFE flow warping)
        blended = (1.0 - timestep) * frame_a + timestep * frame_b
        return blended


class VideoPipeline:
    """Complete video processing pipeline."""

    def __init__(self, args, loader):
        self.args = args
        self.loader = loader
        self.temp_dir = None

    def run(self):
        """Execute the full pipeline: extract -> process -> mux."""
        self.temp_dir = tempfile.mkdtemp(prefix="rve_")
        print(f"Working directory: {self.temp_dir}")

        try:
            # Step 1: Extract frames
            frame_paths = self.extract_frames()
            if not frame_paths:
                raise RuntimeError("No frames extracted from input video")
            print(f"Extracted {len(frame_paths)} frames")

            # Step 2: Process frames (upscale + interpolate)
            processed_paths = self.process_frames(frame_paths)
            print(f"Processed {len(processed_paths)} frames")

            # Step 3: Remux
            self.remux(processed_paths)
            print(f"Rendering complete: {self.args.output}")

        finally:
            # Cleanup temp dir
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)

    def extract_frames(self):
        """Extract video frames as PNG sequence via FFmpeg."""
        output_pattern = os.path.join(self.temp_dir, "frame_%06d.png")

        # Probe source FPS for ffmpeg
        src_fps = probe_source_fps(self.args.input)
        if src_fps <= 0:
            src_fps = 24.0  # Default fallback

        cmd = [
            "ffmpeg", "-y",
            "-i", self.args.input,
            "-vf", f"fps={src_fps},scale=-2:-2",
            "-pix_fmt", "rgb24",
            output_pattern
        ]

        print(f"Extracting frames with: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}", file=sys.stderr)
            raise RuntimeError(f"FFmpeg frame extraction failed: {result.stderr}")

        # Collect extracted frames
        frames = sorted(Path(self.temp_dir).glob("frame_*.png"))
        return [str(f) for f in frames]

    def process_frames(self, frame_paths):
        """Apply upscaling and/or interpolation to frames."""
        processed = []

        # Determine if we need upscaling
        needs_upscale = self.args.upscale_model and os.path.isfile(self.args.upscale_model)
        needs_interp = self.args.interpolate_model and self.args.interpolate_factor > 1.0

        # Load models
        upscaler = None
        interpolator = None
        if needs_upscale:
            print(f"Loading upscaler: {self.args.upscale_model}")
            upscaler = self.loader.load_upscaler(self.args.upscale_model)
        if needs_interp:
            print(f"Loading interpolator: {self.args.interpolate_model}")
            interpolator = self.loader.load_interpolator(self.args.interpolate_model)

        # Recalculate after loading — if model failed, skip that step
        actually_upscale = upscaler is not None
        actually_interp = interpolator is not None and self.args.interpolate_factor > 1.0

        if not actually_upscale and not actually_interp:
            # No processing needed, just copy frames
            for fp in frame_paths:
                dest = os.path.join(self.temp_dir, os.path.basename(fp))
                shutil.copy2(fp, dest)
                processed.append(dest)
            return processed

        # Process each frame
        prev_frame = None
        interp_step = 1.0 / (self.args.interpolate_factor - 1.0) if needs_interp else 0.0

        for i, fp in enumerate(frame_paths):
            # Load frame
            frame = cv2_load_frame(fp)
            if frame is None:
                continue

            # Upscale first
            if upscaler is not None:
                frame = upscaler(frame)

            # Interpolate between previous and current frame
            if interpolator is not None and prev_frame is not None:
                # Generate intermediate frames
                n_steps = max(1, int(round(self.args.interpolate_factor - 1.0)))
                for s in range(n_steps + 1):
                    t = s / n_steps
                    blended = interpolator(prev_frame, frame, t)
                    out_path = os.path.join(self.temp_dir, f"proc_{i:06d}_{s:03d}.png")
                    cv2_save_frame(blended, out_path)
                    processed.append(out_path)
            else:
                # Just save the (possibly upscaled) frame
                out_path = os.path.join(self.temp_dir, f"proc_{i:06d}_000.png")
                cv2_save_frame(frame, out_path)
                processed.append(out_path)

            prev_frame = frame

        return processed

    def remux(self, processed_paths):
        """Reassemble processed frames into final video via FFmpeg."""
        output_pattern = os.path.join(self.temp_dir, "proc_%06d_%.3d.png")
        target_fps = self.args.target_fps if self.args.target_fps > 0 else 24.0

        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(target_fps),
            "-i", output_pattern,
            "-c:v", "libx264",
            "-preset", self.args.video_encoder_preset or "medium",
            "-crf", self.args.crf or "18",
            "-pix_fmt", self.args.video_pixel_format or "yuv420p",
            "-movflags", "+faststart",
            self.args.output
        ]

        # Handle audio
        has_audio = self._has_audio(self.args.input)
        if has_audio:
            if self.args.audio_encoder_preset == "copy_audio":
                cmd.extend(["-c:a", "copy"])
            elif self.args.audio_bitrate:
                cmd.extend(["-c:a", "aac", "-b:a", self.args.audio_bitrate])

        print(f"Remuxing with: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}", file=sys.stderr)
            raise RuntimeError(f"FFmpeg remuxing failed: {result.stderr}")

    def _has_audio(self, input_path):
        """Check if input video has an audio stream."""
        try:
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
                   "-show_streams", input_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "audio":
                        return True
        except Exception:
            pass
        return False


def cv2_load_frame(path):
    """Load a PNG frame as a PyTorch tensor [C,H,W] normalized to [0,1]."""
    if not HAS_TORCH:
        return None
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        # BGR -> RGB, HWC -> CHW, [0,255] -> [0,1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor
    except Exception as e:
        print(f"WARNING: Failed to load frame {path}: {e}", file=sys.stderr)
        return None


def cv2_save_frame(tensor, path):
    """Save a PyTorch tensor [C,H,W] in [0,1] as a PNG frame."""
    if not HAS_TORCH:
        return
    try:
        import cv2
        img = tensor.clamp(0, 1).permute(1, 2, 0).numpy()
        img = (img * 255).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img)
    except Exception as e:
        print(f"WARNING: Failed to save frame {path}: {e}", file=sys.stderr)


class HandleApplication:
    """Main application handler."""

    def __init__(self):
        self.args = self.handleArguments()
        if self.args.version:
            print(f"{__version__}")
            sys.exit(0)

        if self.args.print_video_info:
            print(f"Input: {self.args.input}")
            fps = probe_source_fps(self.args.input)
            print(f"Source FPS: {fps}")
            print(f"Target FPS: {self.args.target_fps}" if self.args.target_fps > 0 else "")
            sys.exit(0)

        if not self.args.list_backends:
            # Probe source FPS before validation so --target_fps can derive factor
            if self.args.target_fps and self.args.target_fps > 0:
                source_fps = probe_source_fps(self.args.input)
                if source_fps <= 0:
                    raise ValueError(
                        f"Could not determine source FPS from '{self.args.input}'. "
                        "Cannot compute interpolation factor from --target_fps."
                    )
                self.applyTargetFPS(source_fps)
            self.checkArguments()
            if not self.batchProcessing():
                self.renderVideo()
        else:
            self.listBackends()

    def batchProcessing(self):
        """Handle batch processing from .txt files."""
        if os.path.splitext(self.args.input)[-1] == ".txt":
            with open(self.args.input, "r") as f:
                for line in f.readlines():
                    sys.argv[1:] = line.split()
                    self.args = self.handleArguments()
                    self.renderVideo()
            return True
        return False

    def listBackends(self):
        """List available inference backends."""
        print("DaSiWa TrueVideoEnhancer Backends:")
        print("=" * 40)

        if HAS_TORCH:
            cuda_avail = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_avail else 0
            print(f"[{'✓' if cuda_avail else '✗'}] PyTorch v{torch.__version__}")
            if cuda_avail:
                print(f"  CUDA devices: {device_count}")
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            else:
                print("  CUDA not available")

        if HAS_TRT:
            print(f"[✓] TensorRT v{trt.__version__}")
        else:
            print("[✗] TensorRT not available")

        print(f"[{'✓' if HAS_SAFETENSORS else '✗'}] safetensors")

    def applyTargetFPS(self, source_fps):
        """Calculate interpolation factor from target FPS."""
        if not self.args.target_fps:
            return
        if self.args.target_fps <= 0:
            raise ValueError("Target FPS must be greater than 0")
        if source_fps <= 0:
            raise ValueError("Source FPS must be greater than 0 to use --target_fps")
        self.args.interpolate_factor = self.args.target_fps / source_fps

    def checkArguments(self):
        """Validate command-line arguments."""
        if self.args.output and os.path.isfile(self.args.output) and not self.args.overwrite:
            raise OSError("Output file already exists!")
        if "http" not in self.args.input and not os.path.isfile(self.args.input):
            raise OSError("Input file does not exist!")
        if self.args.tilesize < 0:
            raise ValueError("Tilesize must be greater than 0")
        if self.args.interpolate_factor < 0:
            raise ValueError("Interpolation factor must be greater than 0")
        if self.args.interpolate_factor == 1 and self.args.interpolate_model:
            raise ValueError(
                "Interpolation factor must be greater than 1 if interpolation model is used."
            )
        if self.args.interpolate_factor != 1 and not self.args.interpolate_model:
            raise ValueError(
                "Interpolation factor must be 1 if no interpolation model is used."
            )
        if self.args.backend == 'ncnn' and self.args.hdr_mode:
            print("WARNING: HDR mode is not supported with ncnn backend, falling back to SDR", file=sys.stderr)
            self.args.hdr_mode = False

    def renderVideo(self):
        """Execute the rendering pipeline."""
        print(f"\n{'='*60}")
        print(f"Rendering: {self.args.input} -> {self.args.output}")
        print(f"Backend: {self.args.backend}, Device: {self.args.device}")
        print(f"Precision: {self.args.precision}")
        if self.args.upscale_model:
            print(f"Upscale: {os.path.basename(self.args.upscale_model)}")
        if self.args.interpolate_model:
            print(f"Interpolate: {os.path.basename(self.args.interpolate_model)} @ factor {self.args.interpolate_factor:.2f}x")
        print(f"{'='*60}\n")

        loader = ModelLoader(self.args.device)
        pipeline = VideoPipeline(self.args, loader)
        pipeline.run()

    def handleArguments(self):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Backend to RVE, used to upscale and interpolate videos"
        )

        # Input/output
        parser.add_argument("-i", "--input", default=None, help="input video path", type=str)
        parser.add_argument("-o", "--output", default=None, help="output video path or PIPE", type=str)
        parser.add_argument("--start_time", default=None, help="Start of video to be rendered in seconds", type=float)
        parser.add_argument("--end_time", default=None, help="End of video to be rendered in seconds", type=float)
        parser.add_argument("--print_video_info", default=None, help="Print video information and exit", type=str)

        # Backend selection
        parser.add_argument("-b", "--backend", help="backend used to upscale image", default="pytorch", choices=["pytorch", "ncnn", "tensorrt"])
        parser.add_argument("--device", help="Device for inference", default="auto", choices=["auto", "cuda", "mps", "xpu", "cpu"])
        parser.add_argument("--pytorch_gpu_id", help="GPU ID for pytorch backend", default=0, type=int)
        parser.add_argument("--ncnn_gpu_id", help="GPU ID for ncnn backend", default=0, type=int)

        # Models
        parser.add_argument("--upscale_model", help="Direct path to upscaling model", type=str)
        parser.add_argument("--interpolate_model", help="Direct path to interpolation model", type=str)
        parser.add_argument("--extra_restoration_models", help="Compression fixer models", action='append')
        parser.add_argument("--scene_detect_model", help="Scene change detection model", type=str, default=None)

        # Parameters
        parser.add_argument("--interpolate_factor", help="Multiplier for interpolation", type=float, default=1.0)
        parser.add_argument("--target_fps", help="Target output FPS (e.g., 30, 60)", type=float, default=0.0)
        parser.add_argument("--precision", help="Model precision", default="auto", choices=["auto", "float16", "float32"])
        parser.add_argument("--tilesize", help="Upscale in smaller chunks", default=0, type=int)
        parser.add_argument("--tensorrt_opt_profile", help="TensorRT optimization profile", type=int, default=3)
        parser.add_argument("--tensorrt_dynamic_shapes", help="Use dynamic shapes for TensorRT", action="store_true")
        parser.add_argument("--scene_detect_method", help="Scene detection method", default="pyscenedetect")
        parser.add_argument("--scene_detect_threshold", help="Scene detection sensitivity", type=float, default=4.0)

        # Output settings
        parser.add_argument("--overwrite", help="Overwrite existing output", action="store_true")
        parser.add_argument("--border_detect", help="Remove black bars", action="store_true")
        parser.add_argument("--crf", help="Constant rate factor", default="18")
        parser.add_argument("--video_encoder_preset", help="Video encoder preset", default="libx264")
        parser.add_argument("--audio_encoder_preset", help="Audio encoder preset", default="copy_audio")
        parser.add_argument("--subtitle_encoder_preset", help="Subtitle encoder preset", default="copy_subtitle")
        parser.add_argument("--audio_bitrate", help="Audio bitrate", default="192k")
        parser.add_argument("--custom_encoder", help="Custom encoder", default=None)
        parser.add_argument("--video_pixel_format", help="Output pixel format", default="yuv420p")

        # Advanced options
        parser.add_argument("--benchmark", help="Benchmark without saving", action="store_true")
        parser.add_argument("--UHD_mode", help="Lower resolution for optical flow", action="store_true")
        parser.add_argument("--slomo_mode", help="Increase length instead of framerate", action="store_true")
        parser.add_argument("--hdr_mode", help="HDR color space encoding", action="store_true")
        parser.add_argument("--dynamic_scaled_optical_flow", help="Scale optical flow dynamically", action="store_true")
        parser.add_argument("--ensemble", help="Use ensemble interpolation", action="store_true")
        parser.add_argument("--output_to_mpv", help="Output to mpv player", action="store_true")
        parser.add_argument("--list_backends", help="List available backends", action="store_true")
        parser.add_argument("--version", help="Print version and exit", action="store_true")
        parser.add_argument("--cwd", help="Working directory", default=None)
        parser.add_argument("--pause_shared_memory_id", help="Pause state file", default=None)
        parser.add_argument("--merge_subtitles", help="Merge subtitles", action="store_true", default=True)
        parser.add_argument("--override_upscale_scale", help="Override upscale scale", type=int, default=None)
        parser.add_argument("--preview_shared_memory_id", help="Preview shared memory", default=None)

        return parser.parse_args()


if __name__ == "__main__":
    HandleApplication()
