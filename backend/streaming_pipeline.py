from __future__ import annotations

import math
import os
import queue
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, cast

import cv2
import numpy as np

from upscale_inference import Upscaler, upscale_frame_tiled

_SENTINEL = object()
MAX_SAFE_OUTPUT_PIXELS = 33_177_600  # 8K UHD; preserves 1080p→8K 4x jobs.
_HOST_MEMORY_HEADROOM = 512 << 20


def available_host_memory() -> int | None:
    """Return currently available host RAM without counting reclaim-unsafe swap."""
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None


def validate_resource_budget(*, source_width: int, source_height: int,
                             output_width: int, output_height: int,
                             available_memory: int | None = None, bytes_per_channel: int = 1) -> int:
    """Fail before spawning ML/FFmpeg workers for a frame the host cannot safely hold.

    Rawvideo streaming deliberately keeps only bounded queues, but each 4x frame is
    still held by the processor and encoder at the same time.  A hard output-pixel
    limit avoids pathological allocations that can bypass Python's normal OOM path
    and stall the desktop before the kernel can reclaim memory.
    """
    output_pixels = output_width * output_height
    if output_pixels > MAX_SAFE_OUTPUT_PIXELS:
        raise RuntimeError(
            f"requested {output_width}x{output_height} output exceeds the 8K safety limit "
            f"({MAX_SAFE_OUTPUT_PIXELS:,} pixels); use a smaller scale or source resolution"
        )
    source_bytes = source_width * source_height * 3 * bytes_per_channel
    output_bytes = output_pixels * 3 * bytes_per_channel
    # Reader queue, writer queue, the current processed output, and conversion
    # buffers.  This intentionally excludes GPU tensors.
    required_memory = _HOST_MEMORY_HEADROOM + source_bytes * 3 + output_bytes * 5
    if available_memory is None:
        available_memory = available_host_memory()
    if available_memory is not None and available_memory < required_memory:
        raise RuntimeError(
            f"insufficient host memory for bounded streaming buffers: need at least "
            f"{required_memory / (1 << 30):.1f} GiB free, have "
            f"{available_memory / (1 << 30):.1f} GiB"
        )
    return required_memory


def queue_size_for_frame(source_bytes: int, output_bytes: int,
                         available_memory: int | None = None) -> int:
    """Adaptive rawvideo queue depth (2-3) from frame footprint and free RAM.

    Mirrors the RTX node's memory discipline: large 4K->8K frames shrink the
    in-flight buffer so decode+AI+encode never out-run reclaimable RAM.
    """
    if available_memory is None:
        available_memory = available_host_memory()
    if available_memory is None:
        return 3
    per_frame = source_bytes + output_bytes
    # Keep at most ~25% of free RAM in flight, with a 1 GiB floor of headroom.
    usable = max(0, available_memory - (1 << 30)) // 4
    depth = usable // max(1, per_frame)
    return max(2, min(3, depth))


class TensorUpscaler(Upscaler, Protocol):
    supports_tensor_pipeline: bool

    def frame_to_tensor(self, frame: np.ndarray) -> Any: ...
    def upscale_tensor_tiled(self, tensor: Any, tile_size: int) -> Any: ...
    def tensor_to_frame(self, tensor: Any) -> np.ndarray: ...


class VfxStage(Protocol):
    """RTX VFX stage: process one BGR frame into a preallocated output buffer."""

    def process(self, frame_bgr: np.ndarray, out_bgr: np.ndarray) -> np.ndarray: ...
    def close(self) -> None: ...


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found on PATH")
    return path


@lru_cache(maxsize=1)
def _vsync_args() -> list[str]:
    """FFmpeg 7.0 removed -vsync; newer builds need -fps_mode passthrough."""
    try:
        out = subprocess.run(
            [_ffmpeg(), "-hide_banner", "-version"],
            capture_output=True, text=True, check=True,
        ).stdout
        match = re.search(r"ffmpeg version\s+n?(\d+)", out)
        major = int(match.group(1)) if match else 6
    except Exception:
        major = 6
    if major >= 7:
        return ["-fps_mode", "passthrough"]
    return ["-vsync", "0"]


def decode_command(input_video: str, *, width: int, height: int, pixel_format: str = "bgr24", duration: float = 0) -> list[str]:
    return [
        _ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", input_video,
        *(["-t", _fps_string(duration)] if duration else []),
        "-map", "0:v:0", *_vsync_args(), "-pix_fmt", pixel_format, "-f", "rawvideo", "-",
    ]


def _fps_string(fps: float) -> str:
    return f"{fps:.6f}".rstrip("0").rstrip(".")


def resolve_output_codec_path(output_video: str, requested_tag: str, resolved_codec: str) -> str:
    """Replace the final UI-generated codec tag with the codec actually used.

    Arbitrary CLI output names are left untouched. Only a final ``[requested]``
    tag immediately before the extension is replaced.
    """
    path = Path(output_video)
    suffix = path.suffix
    name = path.name
    stem = name[:-len(suffix)] if suffix else name
    import re
    preview_match = re.search(r'-preview-[0-9]+(?:\.[0-9]+)?s$', stem)
    preview_suffix = preview_match.group(0) if preview_match else ''
    if preview_suffix:
        stem = stem[:-len(preview_suffix)]

    requested = (requested_tag or "auto").strip() or "auto"
    marker = f"[{requested}]"
    if not stem.endswith(marker):
        return output_video

    codec = "".join(
        ch if ch.isalnum() or ch in "._-" else "-"
        for ch in (resolved_codec or "").strip()
    ).strip("-_.")
    if not codec:
        return output_video

    resolved_name = f"{stem[:-len(marker)]}[{codec}]{preview_suffix}{suffix}"
    return str(path.with_name(resolved_name))


def encode_command(input_video: str, output_video: str, *, width: int, height: int,
                   fps: float, video_args: list[str], audio_args: list[str],
                   subtitle_args: list[str], pixel_format: str = "bgr24", duration: float = 0) -> list[str]:
    return [
        _ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", pixel_format, "-s", f"{width}x{height}",
        "-r", _fps_string(fps), "-i", "-",
        *(["-t", _fps_string(duration)] if duration else []), "-i", input_video,
        "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
        *video_args, *audio_args, *subtitle_args,
        # -t alone does not clip a copied subtitle cue crossing the boundary.
        # Clamp packet display duration without re-encoding source media.
        *(["-bsf:s", f"setts=duration='min(DURATION,max(0,{_fps_string(duration)}/TB-PTS))'",
           "-bsf:a", f"setts=duration='min(DURATION,max(0,{_fps_string(duration)}/TB-PTS))'"] if duration else []),
        *(["-t", _fps_string(duration)] if duration else []), output_video,
    ]


def frame_from_bytes(data: bytes, *, width: int, height: int, pixel_format: str = "bgr24") -> np.ndarray:
    dtype = np.dtype("<u2") if pixel_format == "bgr48le" else np.dtype("u1")
    expected = width * height * 3 * dtype.itemsize
    if len(data) != expected:
        raise ValueError(f"raw frame has {len(data)} bytes, expected {expected}")
    return np.frombuffer(data, dtype=dtype).reshape(height, width, 3).copy()


def output_schedule(*, source_count: int, source_fps: float,
                    target_fps: float) -> Iterable[tuple[int, int, float]]:
    duration = (source_count - 1) / source_fps if source_count > 1 else 0.0
    count = max(1, int(math.floor(duration * target_fps + 1e-6)) + 1)
    for output_index in range(count):
        position = min(output_index * source_fps / target_fps, source_count - 1)
        left = int(math.floor(position))
        right = min(left + 1, source_count - 1)
        timestep = float(position - left) if right != left else 0.0
        yield left, right, timestep


@dataclass
class FrameProcessor:
    interpolator: object | None
    upscaler: Upscaler | None
    tile_size: int
    target_size: tuple[int, int]
    to_tensor: Callable[[np.ndarray], object]
    to_bgr: Callable[[object], np.ndarray]
    restorer: VfxStage | None = None
    vfx_upscaler: VfxStage | None = None
    output_dtype: Any = np.uint8
    force_upscale: bool = False

    def restore(self, frame: np.ndarray) -> np.ndarray:
        """Denoise/deblur at source resolution (RTX VFX, modes 8-15)."""
        if self.restorer is None:
            return frame
        buffer = np.empty_like(frame)
        return self.restorer.process(frame, buffer)

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        if self.vfx_upscaler is not None:
            buffer = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=self.output_dtype)
            return self.vfx_upscaler.process(frame, buffer)
        if self.upscaler is None:
            return frame
        if getattr(self.upscaler, "supports_tensor_pipeline", False):
            tensor_upscaler = cast(TensorUpscaler, self.upscaler)
            output = tensor_upscaler.frame_to_tensor(frame)
            first = self.force_upscale
            while first or output.shape[-1] < self.target_size[0] or output.shape[-2] < self.target_size[1]:
                first = False
                output = tensor_upscaler.upscale_tensor_tiled(output, self.tile_size)
            return tensor_upscaler.tensor_to_frame(output)
        output = frame
        first = self.force_upscale
        while first or output.shape[1] < self.target_size[0] or output.shape[0] < self.target_size[1]:
            first = False
            output = (upscale_frame_tiled(output, self.upscaler, tile_size=self.tile_size)
                      if self.tile_size > 0 else self.upscaler.upscale(output))
        return output

    def process(self, frame_a: np.ndarray, frame_b: np.ndarray, timestep: float) -> np.ndarray:
        if self.interpolator is not None and 1e-5 < timestep < 1.0 - 1e-5:
            output = self.to_bgr(self.interpolator.interpolate(
                self.to_tensor(frame_a), self.to_tensor(frame_b), timestep=timestep,
            ))
        elif timestep >= 1.0 - 1e-5:
            output = frame_b
        else:
            output = frame_a
        output = self.restore(output)
        output = self.upscale(output)
        if (output.shape[1], output.shape[0]) != self.target_size:
            output = cv2.resize(output, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        return np.ascontiguousarray(output)


class _ProcessWorker:
    def __init__(self, command: list[str], *, read: bool, queue_size: int):
        self.command = command
        self.read = read
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.error: BaseException | None = None
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.stopped = threading.Event()
        self.stderr_tail = bytearray()
        self.stderr_thread: threading.Thread | None = None

    def drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while True:
            data = self.process.stderr.read(4096)
            if not data:
                return
            self.stderr_tail.extend(data)
            del self.stderr_tail[:-65536]

    def start_stderr(self) -> None:
        self.stderr_thread = threading.Thread(target=self.drain_stderr, daemon=True)
        self.stderr_thread.start()

    def enqueue(self, item) -> None:
        while not self.stopped.is_set():
            self.raise_if_failed()
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                if self.process is not None and self.process.poll() is not None and not self.read:
                    raise RuntimeError(f"encoder exited: {self.stderr_tail.decode(errors='replace')}")
        raise RuntimeError("rawvideo worker stopped")

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError(str(self.error)) from self.error


class RawVideoReader(_ProcessWorker):
    def __init__(self, input_video: str, *, width: int, height: int, queue_size: int = 3, pixel_format: str = "bgr24", duration: float = 0):
        self.pixel_format = pixel_format
        super().__init__(decode_command(input_video, width=width, height=height, pixel_format=pixel_format, duration=duration), read=True,
                         queue_size=queue_size)
        self.width, self.height = width, height

    def start(self) -> None:
        self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.start_stderr()
        self.thread = threading.Thread(target=self._run, name="ffmpeg-raw-reader", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        size = self.width * self.height * 3 * (2 if self.pixel_format == "bgr48le" else 1)
        try:
            while True:
                data = self.process.stdout.read(size)
                if not data:
                    break
                if len(data) != size:
                    raise RuntimeError(f"decoder returned partial raw frame ({len(data)}/{size} bytes)")
                self.enqueue(frame_from_bytes(data, width=self.width, height=self.height, pixel_format=self.pixel_format))
            code = self.process.wait()
            if self.stderr_thread:
                self.stderr_thread.join(timeout=5)
            stderr = self.stderr_tail.decode(errors="replace")
            if code != 0:
                raise RuntimeError(f"ffmpeg decode failed: {stderr.strip()}")
        except BaseException as exc:
            self.error = exc
        finally:
            while not self.stopped.is_set():
                try:
                    self.queue.put(_SENTINEL, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def frames(self):
        while True:
            item = self.queue.get()
            if item is _SENTINEL:
                self.raise_if_failed()
                return
            yield item

    def close(self) -> None:
        self.stopped.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.thread:
            self.thread.join(timeout=5)


class RawVideoWriter(_ProcessWorker):
    def __init__(self, command: list[str], *, queue_size: int = 3):
        super().__init__(command, read=False, queue_size=queue_size)

    def start(self) -> None:
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.start_stderr()
        self.thread = threading.Thread(target=self._run, name="ffmpeg-raw-writer", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        assert self.process is not None and self.process.stdin is not None
        try:
            while not self.stopped.is_set():
                try:
                    item = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is _SENTINEL:
                    break
                self.process.stdin.write(item.tobytes())
            self.process.stdin.close()
            code = self.process.wait()
            if self.stderr_thread:
                self.stderr_thread.join(timeout=5)
            stderr = self.stderr_tail.decode(errors="replace")
            if code != 0:
                raise RuntimeError(f"ffmpeg encode failed: {stderr.strip()}")
        except BaseException as exc:
            self.error = exc
            if self.process.poll() is None:
                self.process.terminate()

    def put(self, frame: np.ndarray) -> None:
        self.raise_if_failed()
        self.enqueue(frame)

    def finish(self) -> None:
        self.enqueue(_SENTINEL)
        if self.thread:
            self.thread.join()
        self.raise_if_failed()

    def close(self) -> None:
        self.stopped.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)


def stream_frames(reader: RawVideoReader, writer: RawVideoWriter, processor: Any,
                  *, source_fps: float, target_fps: float,
                  progress_cb: Callable[[int], None] | None = None,
                  preview_cb: Callable[[np.ndarray], None] | None = None,
                  preserve_duration: bool = False, duration: float = 0) -> int:
    frames = iter(reader.frames())
    try:
        current = next(frames)
    except StopIteration:
        raise RuntimeError("decoder produced no video frames")
    source_index = 0
    output_index = 0
    next_output_position = 0.0
    def within_duration():
        return not duration or output_index / target_fps < duration - 1e-9
    for following in frames:
        while next_output_position < source_index + 1.0 - 1e-9 and within_duration():
            timestep = next_output_position - source_index
            output = processor.process(current, following, timestep)
            writer.put(output)
            output_index += 1
            if preview_cb is not None:
                preview_cb(output)
            if progress_cb is not None:
                progress_cb(output_index)
            next_output_position = output_index * source_fps / target_fps
        current = following
        source_index += 1
    while (next_output_position < source_index + 1.0 - 1e-9 if preserve_duration
           else next_output_position <= source_index + 1e-6) and within_duration():
        output = processor.process(current, current, 0.0)
        writer.put(output)
        output_index += 1
        if preview_cb is not None:
            preview_cb(output)
        if progress_cb is not None:
            progress_cb(output_index)
        next_output_position = output_index * source_fps / target_fps
    return output_index
