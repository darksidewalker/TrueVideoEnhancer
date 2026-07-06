"""NVIDIA RTX VFX Upscaler with Lanczos Fallback.

Provides high-quality upscaling/refinement for DaSiWa TVE.
When used with Scale=1, applies the upscale model then downscales back
to original resolution for pure quality improvement without size change.

Primary path: NVIDIA RTX VFX Video Super Resolution (if available)
Fallback: OpenCV Lanczos resampling (high quality, no GPU dependency)
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
    """Detect the scale factor from an upscale model filename."""
    name = Path(model_path).stem.lower()
    match = __import__("re").search(r"(\d)x", name)
    if match:
        return int(match.group(1))
    if any(x in name for x in ("x4", "_4x", "-4x")):
        return 4
    return 2


def apply_upscale_refine(
    source_frames: list[Path],
    upscale_model: str,
    target_scale: int,
    output_dir: Path,
    device_id: int = 0,
    preview_cb: Optional[Callable[[np.ndarray], None]] = None,
) -> tuple[int, int, int]:
    """Process frames through upscale model, optionally downscaling back.

    Pipeline:
      1. Detect model factor (e.g., 2x)
      2. Upscale all frames to model factor × source resolution
      3. If target_scale < model_factor, downscale to target resolution
      4. Return (frame_count, output_width, output_height)
    """
    if not upscale_model:
        return len(source_frames), 0, 0

    model_factor = detect_upscale_factor(upscale_model)
    log_info(f"Upscale model: {model_factor}x, target scale: {target_scale}x")

    first_frame = cv2.imread(str(source_frames[0]))
    if first_frame is None:
        raise RuntimeError(f"Cannot read first frame: {source_frames[0]}")

    src_h, src_w = first_frame.shape[:2]

    # Intermediate resolution after applying upscale model
    mid_w = src_w * model_factor
    mid_h = src_h * model_factor

    # Final target resolution
    final_w = src_w * target_scale
    final_h = src_h * target_scale

    log_info(f"Source: {src_w}x{src_h} → Model output: {mid_w}x{mid_h} → Final: {final_w}x{final_h}")

    upscaler = RTXUpscaler(device_id=device_id)

    try:
        # Phase 1: Upscale
        tmp_dir = output_dir / "_upscale_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        total = len(source_frames)
        for i, fp in enumerate(source_frames):
            frame = cv2.imread(str(fp))
            if frame is None:
                log_warn(f"Skipping unreadable frame {i}")
                continue

            upscaled = upscaler.upscale_frame(frame, mid_w, mid_h)

            cv2.imwrite(str(tmp_dir / f"{i + 1:08d}.png"), upscaled)

            if preview_cb:
                try:
                    preview_cb(upscaled)
                except Exception:
                    pass

            if (i + 1) % 50 == 0 or i + 1 == total:
                log_info(f"Upscaled {i + 1}/{total} frames")

        # Phase 2: Downscale back to target if needed
        if target_scale < model_factor:
            tgt_dir = output_dir / "_resize_tmp"
            tgt_dir.mkdir(parents=True, exist_ok=True)

            resized_count = 0
            sorted_frames = sorted(tmp_dir.glob("*.png"))
            for i, fp in enumerate(sorted_frames):
                frame = cv2.imread(str(fp))
                if frame is None:
                    continue
                resized = cv2.resize(frame, (final_w, final_h), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(str(tgt_dir / f"{i + 1:08d}.png"), resized)
                resized_count += 1

                if preview_cb:
                    try:
                        preview_cb(resized)
                    except Exception:
                        pass

                if (i + 1) % 50 == 0 or i + 1 == len(sorted_frames):
                    log_info(f"Downscaled to target: {i + 1}/{resized_count} frames")

            # Clean up intermediate
            import shutil
            shutil.rmtree(tmp_dir)

            return resized_count, final_w, final_h
        else:
            # No downscale needed — use upscaled frames directly
            import shutil
            shutil.rmtree(tmp_dir)
            return total, mid_w, mid_h

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
