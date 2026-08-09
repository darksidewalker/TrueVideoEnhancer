from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release.*")
warnings.filterwarnings("ignore", message="Both operands of the binary elementwise op.*")
warnings.filterwarnings("ignore", message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*")


class Upscaler(Protocol):
    scale: int
    technique: str

    def upscale(self, frame: np.ndarray) -> np.ndarray: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class UpscaleResult:
    frames: list[Path]
    width: int
    height: int
    technique: str


def choose_tile_size(width: int, height: int, *, requested: int, backend: str) -> int:
    if requested > 0:
        return requested
    return 0


def tensorrt_compile_candidates(width: int, height: int, *, requested_tile: int,
                                fixed_input_size: tuple[int, int] | None = None
                                ) -> list[tuple[int, tuple[int, int]]]:
    if fixed_input_size is not None:
        if fixed_input_size[0] != fixed_input_size[1] or fixed_input_size[0] <= 32:
            raise ValueError(f"unsupported static ONNX input shape: {fixed_input_size}")
        return [(fixed_input_size[0] - 32, fixed_input_size)]
    if requested_tile > 0:
        return [(requested_tile, (requested_tile + 32, requested_tile + 32))]
    if width * height > 512 * 512:
        return [(0, (width, height)), (128, (160, 160))]
    return [(0, (width, height))]


def tensorrt_batch_size(input_size: tuple[int, int]) -> int:
    return 8 if input_size[0] * input_size[1] <= 256 * 256 else 1


def tensorrt_workspace_size(input_size: tuple[int, int]) -> int:
    return (8 if input_size[0] * input_size[1] > 512 * 512 else 1) << 30


def tensorrt_engine_cache_kwargs(model_path: str | Path, *, compiler=None) -> dict[str, object]:
    """Disable persistent Dynamo engine reuse for safetensors upscalers.

    Cached/refitted Torch-TensorRT engines produced corrupted frames in
    reproducible multi-job testing with RCAN safetensors upscaling, accompanied
    by missing CONSTANT weight-mapping warnings. Compile the upscaler engine
    fresh for each backend job. RIFE and ONNX TensorRT caches are unaffected.
    """
    return {}

def static_onnx_input_size(model_path: str | Path) -> tuple[int, int] | None:
    """Return a model's fixed NCHW input size without creating an ORT session."""
    path = Path(model_path)
    if path.suffix.lower() != ".onnx":
        return None
    try:
        import importlib
        onnx = importlib.import_module("onnx")
        graph = onnx.load(str(path), load_external_data=False).graph
        dims = graph.input[0].type.tensor_type.shape.dim
        height, width = dims[-2].dim_value, dims[-1].dim_value
    except Exception:
        return None
    return (int(width), int(height)) if width > 0 and height > 0 else None


def _device_name(device: str, gpu_id: int) -> str:
    if device == "auto":
        try:
            import torch
            return f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if device == "cuda":
        return f"cuda:{gpu_id}"
    return device


def create_upscaler(model_path: str | Path, *, backend: str, device: str = "auto",
                    gpu_id: int = 0, precision: str = "float16",
                    input_size: tuple[int, int] | None = None) -> Upscaler:
    path = Path(model_path)
    suffix = path.suffix.lower()
    if suffix not in {".safetensors", ".onnx"}:
        raise ValueError("upscale models must use .safetensors or .onnx")
    if not path.is_file():
        raise FileNotFoundError(f"upscale model not found: {path}")
    if suffix == ".onnx":
        return ONNXUpscaler(path, backend=backend, gpu_id=gpu_id)
    return SafetensorsUpscaler(
        path,
        backend=backend,
        device=_device_name(device, gpu_id),
        precision=precision,
        input_size=input_size,
    )


def create_optimized_upscaler(model_path: str | Path, *, width: int, height: int,
                              backend: str, requested_tile: int = 0,
                              device: str = "auto", gpu_id: int = 0,
                              precision: str = "float16", fallback_tile_sizes: tuple[int, ...] = (),
                              factory=None):
    factory = factory or create_upscaler
    fixed_size = static_onnx_input_size(model_path)
    if backend == "tensorrt":
        candidates = tensorrt_compile_candidates(
            width, height, requested_tile=requested_tile, fixed_input_size=fixed_size,
        )
        if requested_tile > 0 and fallback_tile_sizes:
            candidates += [
                (tile_size, (tile_size + 32, tile_size + 32))
                for tile_size in fallback_tile_sizes if tile_size != requested_tile
            ]
    else:
        tile_size = fixed_size[0] - 32 if fixed_size else requested_tile
        compile_size = fixed_size or (width, height)
        candidates = [(tile_size, compile_size)]
    failures = []
    for index, (tile_size, compile_size) in enumerate(candidates):
        try:
            upscaler = factory(
                model_path, backend=backend, device=device, gpu_id=gpu_id,
                precision=precision, input_size=compile_size,
            )
            return upscaler, tile_size, compile_size, failures
        except Exception as exc:
            failures.append((compile_size, str(exc)))
            if index + 1 >= len(candidates):
                raise
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    raise RuntimeError("no upscaler compile candidates available")


class SafetensorsUpscaler:
    def __init__(self, model_path: Path, *, backend: str, device: str,
                 precision: str, input_size: tuple[int, int] | None):
        import torch
        try:
            from spandrel import ImageModelDescriptor, ModelLoader
        except ImportError as exc:
            raise RuntimeError("spandrel is required for safetensors upscalers") from exc

        descriptor = ModelLoader().load_from_file(model_path)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise RuntimeError(f"model is not an image upscaler: {model_path.name}")
        self.scale = int(descriptor.scale)
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = torch.float16 if precision == "float16" and self.device.type == "cuda" else torch.float32
        try:
            self.model = descriptor.to(self.device, dtype=self.dtype).eval()
        except Exception as exc:
            if self.dtype != torch.float16 or "half precision" not in str(exc):
                raise
            self.dtype = torch.float32
            self.model = descriptor.to(self.device, dtype=self.dtype).eval()
        self.technique = f"{descriptor.architecture.name} + PyTorch"
        self.batch_size = 1
        self.supports_tensor_pipeline = self.device.type == "cuda"

        if backend == "tensorrt":
            if self.device.type != "cuda":
                raise RuntimeError("TensorRT upscaling requires a CUDA device")
            if input_size is None:
                raise ValueError("input_size is required for TensorRT upscaling")
            self.batch_size = tensorrt_batch_size(input_size)
            self.model = self._compile_tensorrt(input_size)
            self.technique = f"{descriptor.architecture.name} + TensorRT"

    def _compile_tensorrt(self, input_size: tuple[int, int]):
        import torch
        try:
            import torch_tensorrt
        except ImportError as exc:
            raise RuntimeError("torch-tensorrt is required for TensorRT upscaling") from exc
        width, height = input_size
        sample = torch.zeros((self.batch_size, 3, height, width), device=self.device, dtype=self.dtype)
        compile_target = getattr(self.model, "model", self.model)
        model_path = getattr(self, "model_path", None)
        cache_kwargs = tensorrt_engine_cache_kwargs(model_path) if model_path is not None else {}
        return torch_tensorrt.compile(
            compile_target,
            ir="dynamo",
            inputs=[sample],
            workspace_size=tensorrt_workspace_size(input_size),
            min_block_size=1,
            **cache_kwargs,
        )

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        return self.upscale_batch([frame])[0]

    def frame_to_tensor(self, frame: np.ndarray):
        """Upload one BGR frame through pinned host memory for iterative CUDA paths."""
        import torch
        rgb = np.ascontiguousarray(frame[..., ::-1].transpose(2, 0, 1)[None])
        tensor = torch.from_numpy(rgb)
        if self.device.type == "cuda":
            tensor = tensor.pin_memory().to(self.device, dtype=self.dtype, non_blocking=True)
        else:
            tensor = tensor.to(self.device, dtype=self.dtype)
        return tensor.div_(255.0)

    def tensor_to_frame(self, tensor) -> np.ndarray:
        """Download a normalized NCHW RGB tensor once after all iterative passes."""
        import torch
        output = tensor[0].detach().float().clamp_(0, 1).mul_(255.0).round_().to("cpu")
        rgb = output.to(torch.uint8).numpy().transpose(1, 2, 0)
        return np.ascontiguousarray(rgb[..., ::-1])

    def _upscale_tensor_batch(self, tensor):
        import torch
        count = tensor.shape[0]
        if count > self.batch_size:
            raise ValueError(f"tensor batch {count} exceeds compiled TensorRT batch {self.batch_size}")
        if count < self.batch_size:
            tensor = torch.cat((tensor, tensor[-1:].expand(self.batch_size - count, -1, -1, -1)), dim=0)
        if self.model is None:
            raise RuntimeError("upscaler is closed")
        with torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output[:count]

    def upscale_tensor_tiled(self, tensor, tile_size: int, *, overlap: int = 16):
        """Run a tiled CUDA inference pass without a CPU/NumPy round trip.

        TensorRT engines are static, so tile batches are padded to the compiled
        batch size inside _upscale_tensor_batch. The assembled result remains on
        CUDA and can immediately feed the next native 2x pass.
        """
        import torch
        import torch.nn.functional as F
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise ValueError(f"expected one NCHW RGB tensor, got {tuple(tensor.shape)}")
        if tile_size <= 0:
            return self._upscale_tensor_batch(tensor)
        _, _, height, width = tensor.shape
        scale = self.scale
        rows = (height + tile_size - 1) // tile_size
        cols = (width + tile_size - 1) // tile_size
        padded_height = rows * tile_size
        padded_width = cols * tile_size
        # Extra context on every edge matches the CPU tiled path's reflected border.
        padded = F.pad(
            tensor,
            (overlap, padded_width - width + overlap, overlap, padded_height - height + overlap),
            mode="reflect",
        )
        output = torch.empty((1, 3, height * scale, width * scale), device=tensor.device, dtype=tensor.dtype)
        pending = []
        for top in range(0, padded_height, tile_size):
            for left in range(0, padded_width, tile_size):
                pending.append((
                    padded[:, :, top:top + tile_size + 2 * overlap, left:left + tile_size + 2 * overlap],
                    top, min(top + tile_size, height), left, min(left + tile_size, width),
                ))
        for start in range(0, len(pending), self.batch_size):
            items = pending[start:start + self.batch_size]
            tiles = torch.cat([item[0] for item in items], dim=0)
            upscaled = self._upscale_tensor_batch(tiles)
            for item, tile in zip(items, upscaled, strict=True):
                _, top, bottom, left, right = item
                output[:, :, top * scale:bottom * scale, left * scale:right * scale] = tile[
                    :, overlap * scale:(overlap + bottom - top) * scale,
                    overlap * scale:(overlap + right - left) * scale,
                ]
        return output

    def upscale_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        import torch
        if not frames:
            return []
        count = len(frames)
        padded = frames + [frames[-1]] * (self.batch_size - count)
        rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) for frame in padded]
        tensor = torch.from_numpy(np.ascontiguousarray(np.stack(rgb)))
        tensor = tensor.to(self.device, dtype=self.dtype).div_(255.0)
        with torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        arrays = output[:count].detach().float().clamp_(0, 1).cpu().numpy()
        return [
            cv2.cvtColor((array.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
            for array in arrays
        ]

    def close(self) -> None:
        self.model = None
        try:
            import torch
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass


class ONNXUpscaler:
    def __init__(self, model_path: Path, *, backend: str, gpu_id: int):
        self._preload_nvidia_libraries()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime-gpu is required for ONNX upscalers") from exc
        available = set(ort.get_available_providers())
        if backend == "tensorrt":
            if "TensorrtExecutionProvider" not in available:
                raise RuntimeError("ONNX Runtime TensorRT execution provider is unavailable")
            providers = [
                ("TensorrtExecutionProvider", {
                    "device_id": gpu_id,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(model_path.parent / ".trt-cache"),
                    "trt_fp16_enable": True,
                }),
                ("CUDAExecutionProvider", {"device_id": gpu_id}),
                "CPUExecutionProvider",
            ]
            self.technique = "ONNX + TensorRT"
        else:
            providers = [("CUDAExecutionProvider", {"device_id": gpu_id}), "CPUExecutionProvider"]
            self.technique = "ONNX Runtime"
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        if backend == "tensorrt" and "TensorrtExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("ONNX Runtime failed to activate the TensorRT execution provider")
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = self.session.get_inputs()[0].type
        self.scale = self._detect_scale()

    @staticmethod
    def _preload_nvidia_libraries() -> None:
        import ctypes
        import site
        try:
            import torch
            torch.cuda.init()
        except Exception:
            pass
        for root in site.getsitepackages():
            base = Path(root)
            libraries = [
                base / "tensorrt_libs" / "libnvinfer.so.10",
                base / "tensorrt_libs" / "libnvonnxparser.so.10",
                base / "tensorrt_libs" / "libnvinfer_plugin.so.10",
                base / "nvidia" / "cudnn" / "lib" / "libcudnn.so.9",
            ]
            for library in libraries:
                if library.exists():
                    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)

    def _detect_scale(self) -> int:
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape
        if all(isinstance(v, int) and v > 0 for v in (input_shape[-2], output_shape[-2])):
            scale = int(output_shape[-2] // input_shape[-2])
            if scale > 0:
                return scale
        name = Path(self.session._model_path).stem.lower()
        import re
        match = re.search(r"(?:^|[_-])(\d+)x|x(\d+)(?:[_-]|$)", name)
        if match:
            return int(match.group(1) or match.group(2))
        raise RuntimeError("cannot determine ONNX upscaler scale")

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dtype = np.float16 if "float16" in self.input_type else np.float32
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype(dtype) / dtype(255.0)
        output = self.session.run(None, {self.input_name: tensor})[0]
        array = np.asarray(output)[0].clip(0, 1).transpose(1, 2, 0)
        return cv2.cvtColor((array * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self.session = None


def upscale_frame_tiled(frame: np.ndarray, upscaler: Upscaler, *, tile_size: int,
                        overlap: int = 16) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = upscaler.scale
    output = np.empty((height * scale, width * scale, 3), dtype=np.uint8)
    pending = []
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            padded_top = max(0, top - overlap)
            padded_left = max(0, left - overlap)
            padded_bottom = min(height, bottom + overlap)
            padded_right = min(width, right + overlap)
            tile = frame[padded_top:padded_bottom, padded_left:padded_right]
            pad_top = max(0, overlap - top)
            pad_left = max(0, overlap - left)
            pad_bottom = max(0, top + tile_size + overlap - height)
            pad_right = max(0, left + tile_size + overlap - width)
            missing_height = tile_size + 2 * overlap - tile.shape[0] - pad_top - pad_bottom
            missing_width = tile_size + 2 * overlap - tile.shape[1] - pad_left - pad_right
            pad_bottom += max(0, missing_height)
            pad_right += max(0, missing_width)
            tile = cv2.copyMakeBorder(
                tile, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101,
            )
            crop_top = (top - padded_top + pad_top) * scale
            crop_left = (left - padded_left + pad_left) * scale
            crop_bottom = crop_top + (bottom - top) * scale
            crop_right = crop_left + (right - left) * scale
            pending.append((tile, top, bottom, left, right, crop_top, crop_bottom, crop_left, crop_right))
    batch_size = max(1, int(getattr(upscaler, "batch_size", 1)))
    batch_method = getattr(upscaler, "upscale_batch", None)
    for start in range(0, len(pending), batch_size):
        items = pending[start:start + batch_size]
        tiles = [item[0] for item in items]
        upscaled_tiles = batch_method(tiles) if batch_method else [upscaler.upscale(tile) for tile in tiles]
        for item, upscaled in zip(items, upscaled_tiles, strict=True):
            _, top, bottom, left, right, crop_top, crop_bottom, crop_left, crop_right = item
            output[top * scale:bottom * scale, left * scale:right * scale] = (
                upscaled[crop_top:crop_bottom, crop_left:crop_right]
            )
    return output


def process_frames(source_frames: list[Path], output_dir: Path, upscaler: Upscaler,
                   *, target_scale: int, preview_cb=None, progress_cb=None,
                   tile_size: int = 0) -> UpscaleResult:
    if not source_frames:
        raise ValueError("no source frames to upscale")
    if target_scale < 1:
        raise ValueError("target_scale must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(source_frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"cannot read frame: {source_frames[0]}")
    source_height, source_width = first.shape[:2]
    target_size = (source_width * target_scale, source_height * target_scale)
    written: list[Path] = []
    for index, path in enumerate(source_frames, start=1):
        frame = first if index == 1 else cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"cannot read frame: {path}")
        output = frame
        while output.shape[1] < target_size[0] or output.shape[0] < target_size[1]:
            output = (upscale_frame_tiled(output, upscaler, tile_size=tile_size)
                      if tile_size > 0 else upscaler.upscale(output))
        if (output.shape[1], output.shape[0]) != target_size:
            output = cv2.resize(output, target_size, interpolation=cv2.INTER_LANCZOS4)
        output_path = output_dir / f"{index:08d}.png"
        if not cv2.imwrite(str(output_path), output, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
            raise RuntimeError(f"cannot write upscaled frame: {output_path}")
        written.append(output_path)
        if preview_cb is not None:
            preview_cb(output)
        if progress_cb is not None:
            progress_cb(index, len(source_frames))
    return UpscaleResult(written, target_size[0], target_size[1], upscaler.technique)
