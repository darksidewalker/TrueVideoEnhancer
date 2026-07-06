"""NVIDIA RTX VFX Upscaler with Intelligent Scaling.

Applies upscale model only when needed (target factor ≠ model factor).
Supports optional final RTX pass for maximum quality.

Pipeline:
1. Detect model factor from filename (2x, 4x)
2. If target_scale ≠ model_factor → upscale to model_factor, then downscale to target
3. Optional: Apply RTX VFX again at end for refinement
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np


class RTXUpscaler:
    """RTX VFX Upscaler with automatic Lanczos fallback."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.vsr_effect = None
        self.has_rtx = False
        self._try_init_rtx()

    def _try_init_rtx(self) -> None:
        """Attempt to initialize RTX VFX, fall through silently if unavailable."""
        try:
            import torch
            if torch.cuda.is_available():
                import nvvfx
                self.has_rtx = True
                log_info(f"RTX VFX initialized on GPU {self.device_id}")
        except ImportError:
            log_warn("RTX VFX not available — will use Lanczos fallback")

    def close(self) -> None:
        """Clean up RTX resources."""
        if self.vsr_effect:
            try:
                self.vsr_effect.close()
            except Exception:
                pass
            self.vsr_effect = None

    def upscale_frame(self, frame_bgr: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """Upscale a single frame using RTX or Lanczos."""
        target_width = max(8, (target_width // 8) * 8)
        target_height = max(8, (target_height // 8) * 8)

        if self.has_rtx:
            return self._upscale_rtx(frame_bgr, target_width, target_height)
        return self._upscale_lanczos(frame_bgr, target_width, target_height)

    def _upscale_rtx(self, frame_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
        """Apply NVIDIA RTX VFX Video Super Resolution."""
        import torch

        if self.vsr_effect is None:
            try:
                from nvvfx import VideoSuperRes
                self.vsr_effect = VideoSuperRes(quality="HIGH", device=self.device_id)
                self.vsr_effect.output_width = w
                self.vsr_effect.output_height = h
            except Exception as exc:
                log_error(f"VSR init failed: {exc}")
                self.has_rtx = False
                return self._upscale_lanczos(frame_bgr, w, h)

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
            tensor = tensor.to(f"cuda:{self.device_id}")

            result = self.vsr_effect.run(tensor.unsqueeze(0))

            arr = result[0].permute(1, 2, 0).cpu().numpy()
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            log_error(f"VSR frame processing failed: {exc}")
            self.has_rtx = False
            return self._upscale_lanczos(frame_bgr, w, h)

    def _upscale_lanczos(self, frame_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
        """High-quality Lanczos resize (no GPU required)."""
        return cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)


def detect_upscale_factor(model_path: str) -> int:
    """Detect the scale factor from an upscale model filename.
    
    Patterns: "2x-anime", "anime-2x-safetensors", "RealESRGAN_x4plus"
    Returns: 2 or 4 (default 2 if not detected)
    """
    name = Path(model_path).stem.lower()
    
    # Try regex first
    match = __import__("re").search(r"(\d)x", name)
    if match:
        factor = int(match.group(1))
        if factor in (2, 4):
            return factor
    
    # Check for common 4x patterns
    if any(x in name for x in ("x4", "_4x", "-4x")):
        return 4
    
    # Default to 2x
    return 2


def needs_upscale(target_scale: int, model_factor: int) -> bool:
    """Check if upscaling is needed based on target vs model factor."""
    return target_scale != model_factor


def apply_smart_upscale(
    source_frames: list[Path],
    upscale_model: str,
    target_scale: int,
    output_dir: Path,
    device_id: int = 0,
    preview_cb: Optional[Callable[[np.ndarray], None]] = None,
    enable_final_rtx: bool = False,
) -> tuple[int, int, int]:
    """Apply upscale model intelligently with optional final RTX pass.

    Pipeline:
      1. Detect model factor (e.g., 2x)
      2. If target_scale ≠ model_factor → upscale to model_factor, then downscale to target
      3. If target_scale == model_factor → apply model directly
      4. Optionally apply RTX VFX again at end for refinement

    Returns: (frame_count, output_width, output_height)
    """
    if not upscale_model:
        return len(source_frames), 0, 0

    model_factor = detect_upscale_factor(upscale_model)
    log_info(f"Smart upscale: model={model_factor}x, target={target_scale}x")

    # Read first frame to get source dimensions
    first_frame = cv2.imread(str(source_frames[0]))
    if first_frame is None:
        raise RuntimeError(f"Cannot read first frame: {source_frames[0]}")

    src_h, src_w = first_frame.shape[:2]

    # Calculate resolutions
    model_w = src_w * model_factor
    model_h = src_h * model_factor
    target_w = src_w * target_scale
    target_h = src_h * target_scale

    log_info(f"Source: {src_w}x{src_h} | Model output: {model_w}x{model_h} | Target: {target_w}x{target_h}")

    # Initialize upscaler
    upscaler = RTXUpscaler(device_id=device_id)

    try:
        # Phase 1: Apply upscale model
        tmp_dir = output_dir / "_smart_upscale_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        total = len(source_frames)
        
        if needs_upscale(target_scale, model_factor):
            # Case A: Upscale to model factor, then downscale to target
            log_info(f"Upscaling {src_w}x{src_h} → {model_w}x{model_h}, then downscaling to {target_w}x{target_h}")
            
            for i, fp in enumerate(source_frames):
                frame = cv2.imread(str(fp))
                if frame is None:
                    log_warn(f"Skipping unreadable frame {i}")
                    continue

                # Upscale to model factor
                upscaled = upscaler.upscale_frame(frame, model_w, model_h)

                # Downscale to target (if different)
                if target_scale < model_factor:
                    downscaled = cv2.resize(upscaled, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                    out_frame = downscaled
                else:
                    out_frame = upscaled

                cv2.imwrite(str(tmp_dir / f"{i + 1:08d}.png"), out_frame)

                if preview_cb:
                    try:
                        preview_cb(out_frame)
                    except Exception:
                        pass

                if (i + 1) % 50 == 0 or i + 1 == total:
                    log_info(f"Processed {i + 1}/{total} frames")

            final_w, final_h = target_w, target_h
        else:
            # Case B: Apply model directly (target == model factor)
            log_info(f"Applying model directly: {src_w}x{src_h} → {model_w}x{model_h}")

            for i, fp in enumerate(source_frames):
                frame = cv2.imread(str(fp))
                if frame is None:
                    log_warn(f"Skipping unreadable frame {i}")
                    continue

                upscaled = upscaler.upscale_frame(frame, model_w, model_h)
                cv2.imwrite(str(tmp_dir / f"{i + 1:08d}.png"), upscaled)

                if preview_cb:
                    try:
                        preview_cb(upscaled)
                    except Exception:
                        pass

                if (i + 1) % 50 == 0 or i + 1 == total:
                    log_info(f"Upscaled {i + 1}/{total} frames")

            final_w, final_h = model_w, model_h

        # Phase 2: Optional final RTX pass
        if enable_final_rtx and upscaler.has_rtx:
            log_info("Applying final RTX VFX pass...")
            rtx_dir = output_dir / "_final_rtx"
            rtx_dir.mkdir(parents=True, exist_ok=True)

            sorted_frames = sorted(tmp_dir.glob("*.png"))
            for i, fp in enumerate(sorted_frames):
                frame = cv2.imread(str(fp))
                if frame is None:
                    continue

                # Apply RTX VFX to final resolution
                rtx_result = upscaler.upscale_frame(frame, final_w, final_h)
                cv2.imwrite(str(rtx_dir / f"{i + 1:08d}.png"), rtx_result)

                if preview_cb:
                    try:
                        preview_cb(rtx_result)
                    except Exception:
                        pass

            # Replace temp with final
            import shutil
            shutil.rmtree(tmp_dir)
            tmp_dir.rename(output_dir / "_final_output")
            tmp_dir = output_dir / "_final_output"

        final_count = len(list(tmp_dir.glob("*.png")))
        log_info(f"Final output: {final_w}x{final_h}, {final_count} frames")

        return final_count, final_w, final_h

    finally:
        upscaler.close()


# --- Logging helpers (avoid circular imports with rve_backend) ---

_COLOR = sys.stderr.isatty() or __import__("os").environ.get("FORCE_COLOR") in {"1", "true", "yes"}
_COLORS = {
    "RESET": "\033[0m", "DIM": "\033[2m",
    "INFO": "\033[96m", "WARN": "\033[93m", "ERROR": "\033[91m",
}


def _log(tag: str, msg: str) -> None:
    color = _COLORS.get(tag, "") if _COLOR else ""
    reset = _COLORS["RESET"] if _COLOR else ""
    print(f"{color}[{tag}]{reset} {msg}", file=sys.stderr)


def log_info(msg: str) -> None:
    _log("INFO", msg)


def log_warn(msg: str) -> None:
    _log("WARN", msg)


def log_error(msg: str) -> None:
    _log("ERROR", msg)
