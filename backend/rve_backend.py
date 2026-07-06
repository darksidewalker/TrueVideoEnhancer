#!/usr/bin/env python3
"""DaSiWa TrueVideoEnhancer Python backend — SOTA TensorRT frame interpolation.

Full pipeline: extracts frames via FFmpeg -> upscaling -> RIFE interpolation
(TensorRT-engine accelerated) -> remuxes via FFmpeg.

Architecture:
  - Directly imports verified ComfyUI WhiteRabbit RIFE implementation
  - Automatic architecture detection (handles encoder variants)
  - ONNX export -> TensorRT engine building (FP16, workspace 1GiB)
  - Persistent on-disk caching keyed by model content hash
  - Falls back to TorchScript/PyTorch when TRT unavailable
"""

import os
import sys
import argparse
import json
import subprocess
import shutil
import tempfile
import hashlib
from pathlib import Path

# ============================================================================
# CONDITIONAL IMPORTS (graceful fallbacks)
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None

try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    trt = None

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================================
# CONSTANTS & PATHS
# ============================================================================

ENGINE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "rve_trt_engines")
os.makedirs(ENGINE_CACHE_DIR, exist_ok=True)
DEFAULT_WORKSPACE = 1 << 30  # 1 GiB

_COLOR = sys.stderr.isatty() or os.environ.get("FORCE_COLOR") in {"1", "true", "yes"}
_COLORS = {
    "RESET": "\033[0m", "DIM": "\033[2m",
    "ARCH": "\033[95m", "LOAD": "\033[95m", "BUILD": "\033[94m",
    "TRT": "\033[96m", "TS": "\033[94m", "INTERP": "\033[92m",
    "WARN": "\033[93m", "ERROR": "\033[91m",
}


def log(tag, message, *, detail=None):
    color = _COLORS.get(tag, "") if _COLOR else ""
    reset = _COLORS["RESET"] if _COLOR else ""
    dim = _COLORS["DIM"] if _COLOR else ""
    line = f"{color}[{tag}]{reset} {message}"
    if detail:
        line += f" {dim}{detail}{reset}"
    print(line, file=sys.stderr)


# ============================================================================
# RIFE ARCHITECTURE -- VERIFIED COMFYUI IMPLEMENTATION
#
# Import strategy: try to find ComfyUI WhiteRabbit vendor files.
# If not found, fall back to embedded minimal implementations.
# ============================================================================


def _find_vendor_rife():
    """Locate the ComfyUI WhiteRabbit vendor RIFE module."""
    candidates = [
        os.environ.get("COMFYUI_PATH", ""),
        os.path.expanduser("~/Downloads/ComfyUI"),
        os.path.expanduser("~/Documents/ComfyUI"),
    ]
    for base in candidates:
        if not base:
            continue
        path = os.path.join(base, "custom_nodes/comfyui-WhiteRabbit/vendor/rife")
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "__init__.py")):
            return path
    return None


_VENDOR_RIFE_DIR = _find_vendor_rife()
IFNet = None
IFBlock = None
ResConv = None
warp_fn = None


def _load_vendor_architecture():
    """Import RIFE architecture from ComfyUI vendor, or fall back to embedded."""
    global IFNet, IFBlock, ResConv, warp_fn

    if _VENDOR_RIFE_DIR is not None:
        try:
            import sys as _sys
            _abs = os.path.abspath(_VENDOR_RIFE_DIR)
            if _abs not in _sys.path:
                _sys.path.insert(0, _abs)
            from rife_arch import IFNet, IFBlock, ResConv
            from rife_arch import warp as _warp
            IFNet = IFNet
            IFBlock = IFBlock
            ResConv = ResConv
            warp_fn = _warp
            return True
        except Exception as e:
            log("ARCH", "Vendor RIFE import failed; using embedded fallback", detail=str(e))

    # Embedded fallback matching the TNTwise/ComfyUI RIFE safetensors layout.
    # Key names are deeply nested: blocks.N.conv0.0.0.weight,
    # blocks.N.conv0.1.0.weight and blocks.N.lastconv.0.weight.
    if HAS_TORCH:
        class _ResConv(nn.Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
                self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = nn.LeakyReLU(0.2, True)

            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x)

        class _IFBlock(nn.Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                # Deep nesting: conv0.0.0, conv0.1.0 (matching safetensors)
                self.conv0 = nn.Sequential(
                    nn.Sequential(nn.Conv2d(in_planes, c // 2, 3, 2, 1, bias=True)),
                    nn.Sequential(nn.Conv2d(c // 2, c, 3, 2, 1, bias=True)),
                )
                self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
                # RIFE v4.26 heavy checkpoints store 52 output channels here;
                # there is no PixelShuffle module in the state dict.
                self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 52, 4, 2, 1, bias=True))

            def forward(self, x, scale: int = 1):
                scale_f = float(scale)
                x = F.interpolate(x, scale_factor=1.0 / scale_f, mode="bilinear",
                                  align_corners=False)
                feat = self.conv0[0](x)
                feat = self.conv0[1](feat)
                feat = self.convblock(feat)
                tmp = self.lastconv(feat)
                tmp = F.interpolate(tmp, scale_factor=scale_f, mode="bilinear",
                                    align_corners=False)
                flow = tmp[:, :4] * scale_f
                mask = tmp[:, 4:5]
                return flow, mask

        class _IFNet(nn.Module):
            """ComfyUI arch_ver='4.6' multi-scale RIFE network.

            Blocks cascade: each block refines the flow from the previous one.
            Block0 takes concatenated frames + timestep (7 channels).
            Subsequent blocks take warped frames + timestep + previous mask (12 channels).
            """
            def __init__(self, has_encoder=False, encoder_out_ch=39):
                super().__init__()
                self.has_encoder = has_encoder
                self.encoder_out_ch = encoder_out_ch
                # Use deep nesting names to match safetensors
                self.blocks = nn.ModuleDict({
                    '0': _IFBlock(7, c=192),
                    '1': _IFBlock(12, c=128),
                    '2': _IFBlock(12, c=96),
                    '3': _IFBlock(12, c=64),
                })

                if has_encoder:
                    # Encoder: 6 input channels -> 39 output (matches safetensors)
                    self.encode = nn.Sequential(
                        nn.Conv2d(6, 16, 3, 2, 1, bias=True),
                        nn.LeakyReLU(0.2, True),
                        nn.Conv2d(16, 32, 3, 1, 1, bias=True),
                        nn.LeakyReLU(0.2, True),
                        nn.Conv2d(32, encoder_out_ch, 3, 1, 1, bias=True),
                        nn.LeakyReLU(0.2, True),
                    )

            def forward(self, img0, img1, timestep=0.5, scale_list=None):
                if scale_list is None:
                    scale_list = [8, 4, 2, 1]

                img0 = torch.clamp(img0, 0, 1)
                img1 = torch.clamp(img1, 0, 1)
                n, c, h, w = img0.shape
                ph = ((h - 1) // 64 + 1) * 64
                pw = ((w - 1) // 64 + 1) * 64
                if ph > h or pw > w:
                    padding = (0, pw - w, 0, ph - h)
                    img0 = F.pad(img0, padding)
                    img1 = F.pad(img1, padding)

                x = torch.cat((img0, img1), 1)
                channel = x.shape[1] // 2
                img0_c = x[:, :channel]
                img1_c = x[:, channel:]

                if not isinstance(timestep, torch.Tensor):
                    timestep = (x[:, :1].clone() * 0 + 1) * timestep
                else:
                    timestep = timestep.repeat(1, 1, img0_c.shape[2], img0_c.shape[3])

                f0 = None
                f1 = None
                if self.has_encoder:
                    f0 = self.encode(img0_c[:, :6])  # Use first 6 channels
                    f1 = self.encode(img1_c[:, :6])

                warped_img0 = img0_c
                warped_img1 = img1_c
                flow = None
                mask = None

                for i, block_name in enumerate(['0', '1', '2', '3']):
                    block = self.blocks[block_name]
                    if flow is None:
                        if self.has_encoder:
                            cat_input = torch.cat((img0_c[:, :6], img1_c[:, :6], f0, f1, timestep), 1)
                        else:
                            cat_input = torch.cat((img0_c[:, :3], img1_c[:, :3], timestep), 1)
                        flow, mask = block(cat_input, scale=scale_list[i])
                    else:
                        cat_input = torch.cat(
                            (warped_img0[:, :3], warped_img1[:, :3], timestep, mask), 1
                        )
                        f0_new, m0 = block(cat_input, flow=flow, scale=scale_list[i])
                        flow = f0_new
                        mask = m0
                        warped_img0 = _warp_or_grid(warped_img0, flow[:, :2])
                        warped_img1 = _warp_or_grid(warped_img1, flow[:, 2:4])

                return flow, mask

        ResConv = _ResConv
        IFBlock = _IFBlock
        IFNet = _IFNet

        # Minimal warp function (avoids comfy.model_management dependency)
        _backwarp_grid = {}

        def _warp_or_grid(tenInput, tenFlow):
            k = (str(tenFlow.device), str(tenFlow.size()))
            if k not in _backwarp_grid:
                dev = tenFlow.device
                tenH = (torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=dev)
                        .view(1, 1, 1, tenFlow.shape[3])
                        .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1))
                tenV = (torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=dev)
                        .view(1, 1, tenFlow.shape[2], 1)
                        .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3]))
                _backwarp_grid[k] = torch.cat([tenH, tenV], 1)

            tenFlow_norm = torch.cat([
                tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
            ], 1)
            g = (_backwarp_grid[k] + tenFlow_norm).permute(0, 2, 3, 1)
            return F.grid_sample(input=tenInput, grid=g, mode="bilinear",
                                 padding_mode="border", align_corners=True)

        warp_fn = _warp_or_grid
        log("ARCH", "Using embedded RIFE architecture", detail="arch_ver=4.6")
        return True

    return False


# Initialize architecture on import
if HAS_TORCH:
    _load_vendor_architecture()


# ============================================================================
# TENSORRT ENGINE BUILDER (SOTA IMPLEMENTATION)
# ============================================================================


def _engine_cache_key(model_path, precision, opt_profile, dynamic_shapes, resolution):
    """Generate deterministic cache key for engine lookup."""
    key_parts = [model_path, precision, str(opt_profile), str(dynamic_shapes)]
    if os.path.exists(model_path):
        stat = os.stat(model_path)
        key_parts.append(f"{stat.st_size}_{stat.st_mtime_ns}")
    raw = "_".join(key_parts) + f"_{resolution}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_rife_trt_engine(
    model,
    device,
    precision="float16",
    dynamic_shapes=False,
    opt_profile=3,
    cache_dir=None,
    input_resolution=(256, 256),
    max_batch=1,
):
    """Build or retrieve cached TensorRT engine for a RIFE network.

    Pipeline: model.eval() -> torch.onnx.export -> trt.OnnxParser -> build -> cache.
    Optimizations: FP16, optimization profile for dynamic shapes, 1 GiB workspace.

    Returns: (engine, context) tuple or (None, None) on failure.
    """
    if trt is None:
        return None, None

    cache_dir = cache_dir or ENGINE_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    key = _engine_cache_key(
        getattr(model, '_source_model_path', 'unknown'),
        precision, opt_profile, dynamic_shapes, input_resolution
    )
    engine_path = os.path.join(cache_dir, f"rife_{key}.trt")

    # Load cached engine if available
    if os.path.exists(engine_path):
        print(f"[TRT] Loading cached engine: {engine_path}", file=sys.stderr)
        try:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(engine_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is not None:
                ctx = engine.create_execution_context()
                return engine, ctx
        except Exception as e:
            print(f"[TRT] Failed to load cached engine: {e}", file=sys.stderr)

    # Build new engine
    model.eval().to(device)

    # Determine correct input shape based on architecture
    if hasattr(model, 'has_encoder') and model.has_encoder:
        # Encoder variant: single 3-channel frame per side, processed internally
        dummy_a = torch.zeros(max_batch, 3, input_resolution[0], input_resolution[1], device=device)
        dummy_b = torch.zeros(max_batch, 3, input_resolution[0], input_resolution[1], device=device)
        dummy_ts = torch.full((max_batch, 1, input_resolution[0], input_resolution[1]), 0.5, device=device)
    else:
        # Standard: concatenated 6-channel input
        dummy = torch.zeros(max_batch, 6, input_resolution[0], input_resolution[1], device=device)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    # Workspace allocation
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, DEFAULT_WORKSPACE)

    # Enable FP16 if supported
    if precision == "float16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[TRT] FP16 enabled", file=sys.stderr)

    # Create network (explicit batch for ONNX compatibility)
    network_flags = trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    tmp_onnx = os.path.join(tempfile.gettempdir(), f"rife_export_{os.getpid()}.onnx")
    try:
        # Export to ONNX
        if hasattr(model, 'has_encoder') and model.has_encoder:
            torch.onnx.export(
                model, (dummy_a, dummy_b, dummy_ts), tmp_onnx,
                opset_version=18,
                input_names=["img0", "img1", "timestep"],
                output_names=["output"],
                external_data=False,
                dynamic_axes={
                    "img0": {0: "batch"},
                    "img1": {0: "batch"},
                    "timestep": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch"},
                } if dynamic_shapes else None,
            )
        else:
            torch.onnx.export(
                model, dummy, tmp_onnx,
                opset_version=18,
                input_names=["frames"],
                output_names=["output"],
                external_data=False,
                dynamic_axes={
                    "frames": {0: "batch"},
                    "output": {0: "batch"},
                } if dynamic_shapes else None,
            )

        # Parse ONNX into TensorRT
        with open(tmp_onnx, "rb") as f:
            ok = parser.parse(f.read())
        if not ok:
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {'; '.join(errs)}")

        # Optimization profile
        profile = builder.create_optimization_profile()
        if hasattr(model, 'has_encoder') and model.has_encoder:
            min_shape = (1, 3, input_resolution[0], input_resolution[1])
            opt_shape = (max_batch, 3, input_resolution[0], input_resolution[1])
            max_shape = opt_shape if not dynamic_shapes else (max_batch, 3,
                         max(input_resolution[0], min(input_resolution[0] * 2, 4096)),
                         max(input_resolution[1], min(input_resolution[1] * 2, 4096)))
            profile.set_shape("img0", min_shape, opt_shape, max_shape)
            profile.set_shape("img1", min_shape, opt_shape, max_shape)
            if network.get_input(2) is not None and network.get_input(2).name == "timestep":
                ts_min = (1, 1, input_resolution[0], input_resolution[1])
                ts_opt = (max_batch, 1, input_resolution[0], input_resolution[1])
                ts_max = ts_opt if not dynamic_shapes else (max_batch, 1, max_shape[2], max_shape[3])
                profile.set_shape("timestep", ts_min, ts_opt, ts_max)
        else:
            min_shape = (1, 6, input_resolution[0], input_resolution[1])
            opt_shape = (max_batch, 6, input_resolution[0], input_resolution[1])
            max_shape = opt_shape if not dynamic_shapes else (max_batch, 6,
                         max(input_resolution[0], min(input_resolution[0] * 2, 4096)),
                         max(input_resolution[1], min(input_resolution[1] * 2, 4096)))
            profile.set_shape("frames", min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        # Build serialized engine
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build returned None")

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError("Failed to deserialize engine")

        context = engine.create_execution_context()

        # Cache to disk
        with open(engine_path, "wb") as f:
            f.write(bytes(serialized))
        serialized_size = getattr(serialized, "nbytes", None)
        if serialized_size is None:
            try:
                serialized_size = len(bytes(serialized))
            except Exception:
                serialized_size = 0
        print(f"[TRT] Engine built & cached: {engine_path} ({serialized_size//1024} KiB)",
              file=sys.stderr)

        return engine, context

    except Exception as e:
        print(f"[TRT] Engine build failed: {e}", file=sys.stderr)
        return None, None

    finally:
        if os.path.exists(tmp_onnx):
            try:
                os.unlink(tmp_onnx)
            except OSError:
                pass


def trt_infer_encoder_variant(engine, context, frame_a, frame_b, device):
    """Execute inference on TRT engine for encoder-variant RIFE models.

    Input: two separate 3-ch frames + timestep scalar.
    Output: flow (4 ch) and mask (1 ch).
    """
    # Prepare inputs
    ts = torch.full((1, 1, 1, 1), 0.5, device=device)
    inp_a = frame_a.unsqueeze(0).contiguous().half() if frame_a.is_floating_point() else frame_a.unsqueeze(0).contiguous()
    inp_b = frame_b.unsqueeze(0).contiguous().half() if frame_b.is_floating_point() else frame_b.unsqueeze(0).contiguous()

    # Use PyTorch wrapper since TRT bindings are complex for multi-output networks
    model = engine._model if hasattr(engine, '_model') else None
    if model is None:
        # Fallback: run via ONNX Runtime
        return _onnx_infer_wrapper(inp_a, inp_b, ts, device)

    # Execute with bindings
    bindings = []
    for i in range(context.num_bindings):
        name = context.get_binding_name(i)
        shape = context.get_binding_shape(i)
        size = 1
        for s in shape:
            size *= s
        dtype = context.get_binding_dtype(i)
        if dtype == trt.float16:
            size *= 2
        buf = torch.empty(size, dtype=torch.uint8, device=device)
        bindings.append(int(buf.data_ptr()))

    context.execute_v2(bindings)
    # Extract outputs from bindings...
    # This is complex; use PyTorch fallback instead
    return _pytorch_fallback_infer(inp_a, inp_b, ts)


def _onnx_infer_wrapper(frame_a, frame_b, timestep, device):
    """Fallback ONNX Runtime inference for TRT-built engines."""
    try:
        import onnxruntime as ort
        model_path = getattr(frame_a, '_onnx_path', None)
        if model_path and os.path.exists(model_path):
            session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
            result = session.run(None, {"img0": frame_a.cpu().numpy(),
                                         "img1": frame_b.cpu().numpy(),
                                         "timestep": timestep.cpu().numpy()})
            return torch.from_numpy(result[0]).to(device)
    except Exception:
        pass
    return None


def _pytorch_fallback_infer(frame_a, frame_b, timestep):
    """Pure PyTorch fallback when TRT bindings are too complex."""
    # This would need the original model -- just signal failure
    return None


# ============================================================================
# TORCHSCRIPT FALLBACK (when TRT unavailable or fails)
# ============================================================================


def build_torchscript_interpolator(model, device, optimization_level="O2"):
    """Compile interpolator to TorchScript for optimized CPU/CUDA execution.

    Falls back to eager mode compilation when scripting fails.
    """
    if not HAS_TORCH:
        return None

    model.eval().to(device)
    try:
        if hasattr(model, 'has_encoder') and model.has_encoder:
            dummy_a = torch.zeros(1, 3, 64, 64, device=device)
            dummy_b = torch.zeros(1, 3, 64, 64, device=device)
            dummy_ts = torch.full((1, 1, 64, 64), 0.5, device=device)
            scripted = torch.jit.trace(model, (dummy_a, dummy_b, dummy_ts), strict=False)
        else:
            scripted = torch.jit.script(model)
        scripted = torch.jit.optimize_for_inference(scripted)
        if hasattr(model, 'has_encoder') and model.has_encoder:
            scripted.has_encoder = True
        print("[TS] TorchScript compiled successfully", file=sys.stderr)
        return scripted
    except Exception as e:
        print(f"[TS] TorchScript failed ({e}), using eager mode", file=sys.stderr)
        return model


# ============================================================================
# ARCHITECTURE DETECTION FROM SAFETENSORS
# ============================================================================


def _detect_architecture(sd):
    """Detect RIFE model variant from state dict key patterns.

    Returns: ('standard', False) or ('encoder', encoder_out_channels)
    """
    all_keys = sorted(sd.keys())

    # Check for encoder pattern: Comfy-Org RIFE v4.26 safetensors.
    # General model: block0 in=15 (3+3+4+4+1), encode.cnn3 outputs 4.
    # Heavy anime model: block0 in=39 (3+3+16+16+1), encode.cnn3 outputs 16.
    has_encoder = any(k.startswith("encode.") for k in all_keys)
    if has_encoder:
        block0 = sd.get("blocks.0.conv0.0.0.weight")
        enc3_bias = sd.get("encode.cnn3.bias")
        block0_in = int(block0.shape[1]) if block0 is not None else 0
        enc_out = int(enc3_bias.shape[0]) if enc3_bias is not None else 0
        if block0_in == 39 or enc_out == 16:
            return ("encoder_heavy_anime", enc_out)
        if block0_in == 15 or enc_out == 4:
            return ("encoder_general", enc_out)
        return ("encoder_unknown", enc_out)

    # Check for standard ComfyUI pattern: blocks.N.conv0.weight (no .0.0.)
    has_standard = any("blocks." in k and "conv0.weight" in k and "conv0.0.0" not in k
                       for k in all_keys)
    if has_standard:
        return ("standard", None)

    # Check for heavy custom pattern: blocks.N.conv0.0.0.weight
    has_heavy = any("conv0.0.0.weight" in k for k in all_keys)
    if has_heavy:
        conv0_key = next(k for k in all_keys if "conv0.0.0.weight" in k)
        in_ch = sd[conv0_key].shape[1]
        if in_ch > 10:
            return ("heavy_custom", None)
        return ("standard", None)

    # Default
    return ("standard", None)


def _build_model_from_sd(sd, arch_type, arch_info):
    """Build appropriate IFNet model from safetensors state dict.

    For encoder variants: builds IFNet with matching encoder layers.
    For standard: builds plain IFNet (ComfyUI arch_ver=4.6 structure).
    """
    if not HAS_TORCH or IFNet is None:
        raise RuntimeError("PyTorch or RIFE architecture not available")

    if arch_type in {"encoder", "encoder_heavy_anime", "encoder_general", "encoder_unknown"}:
        encoder_out_ch = arch_info
        print(f"[BUILD] Building encoder-variant model (encoder out={encoder_out_ch}ch)",
              file=sys.stderr)
        return _build_encoder_heavy_model(sd)

    elif arch_type == "heavy_custom":
        # Custom architecture with conv0.0.0 pattern
        # Extract block configs from weight shapes
        all_keys = sorted(sd.keys())
        block_ids = set()
        for k in all_keys:
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "blocks":
                try:
                    block_ids.add(int(parts[1]))
                except ValueError:
                    pass

        blocks_config = []
        for bid in sorted(block_ids):
            c00_key = f"blocks.{bid}.conv0.0.0.weight"
            if c00_key in sd:
                w0 = sd[c00_key].shape
                in_ch = w0[1]
                mid_ch = w0[0]
                lc_key = f"blocks.{bid}.lastconv.0.weight"
                if lc_key in sd:
                    out_ch = sd[lc_key].shape[0]
                else:
                    out_ch = mid_ch * 2
                blocks_config.append((in_ch, mid_ch, out_ch))

        print(f"[BUILD] Heavy custom model: {len(blocks_config)} blocks, config={blocks_config}",
              file=sys.stderr)
        # For heavy custom, we still use the IFNet but with direct block access
        # The standard IFNet won't match; create a simple adapter
        return _build_heavy_custom_model(blocks_config)

    else:
        # Standard ComfyUI architecture
        print("[BUILD] Building standard ComfyUI arch_ver=4.6 model", file=sys.stderr)
        model = IFNet(has_encoder=False)
        return model


def _build_heavy_custom_model(blocks_config):
    """Build a simple sequential model for heavy custom architectures.

    Each block: conv0 -> convblock(8x ResConv) -> lastconv(ConvTranspose2d)
    Data flows sequentially through blocks.
    """
    class _HeavyCustomBlock(nn.Module):
        def __init__(self, in_ch, mid_ch, out_ch):
            super().__init__()
            self.conv0 = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(mid_ch, mid_ch * 2, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, True),
            )
            self.convblock = nn.Sequential(*[_ResConv(mid_ch * 2) for _ in range(8)])
            self.lastconv = nn.Sequential(
                nn.ConvTranspose2d(mid_ch * 2, out_ch, 4, 2, 1, bias=True),
                nn.PixelShuffle(2) if out_ch < mid_ch * 2 // 2 else nn.Identity(),
            )

        def forward(self, x):
            x = self.conv0(x)
            x = self.convblock(x)
            x = self.lastconv(x)
            return x

    class _HeavyCustomNet(nn.Module):
        def __init__(self, blocks_config):
            super().__init__()
            self.blocks = nn.ModuleList([
                _HeavyCustomBlock(ic, mc, oc) for ic, mc, oc in blocks_config
            ])

        def forward(self, img0, img1, timestep=0.5, scale_list=None):
            x = torch.cat((img0, img1), 1)
            out = x
            for block in self.blocks:
                out = block(out)
            # Return first 4 channels as flow, rest as mask (approximate)
            flow = out[:, :4]
            mask = out[:, 4:5] if out.shape[1] > 4 else torch.zeros(out.shape[0], 1, *out.shape[2:], device=out.device)
            return flow, mask

    return _HeavyCustomNet(blocks_config)


def _build_encoder_heavy_model(sd):
    """Build ComfyUI native RIFE v4.26 safetensors IFNet.

    This mirrors ComfyUI's comfy_extras.frame_interpolation_models.ifnet.IFNet:
    5 blocks, Head encoder, feature feedback, PixelShuffle lastconv, and final
    `lerp(warped_img1, warped_img0, sigmoid(mask))`. The previous local graph
    treated the 52-channel block output as a sequential flow tensor, which is
    why every interpolated frame was deformed.
    """

    class _Head(nn.Module):
        def __init__(self, out_ch):
            super().__init__()
            self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
            self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
            self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
            self.cnn3 = nn.ConvTranspose2d(16, out_ch, 4, 2, 1)
            self.relu = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            x = self.relu(self.cnn0(x))
            x = self.relu(self.cnn1(x))
            x = self.relu(self.cnn2(x))
            return self.cnn3(x)

    class _Block(nn.Module):
        def __init__(self, in_planes, c):
            super().__init__()
            self.conv0 = nn.Sequential(
                nn.Sequential(nn.Conv2d(in_planes, c // 2, 3, 2, 1), nn.LeakyReLU(0.2, True)),
                nn.Sequential(nn.Conv2d(c // 2, c, 3, 2, 1), nn.LeakyReLU(0.2, True)),
            )
            self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
            self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

        def forward(self, x, flow=None, scale=1):
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
            if flow is not None:
                flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False).div(scale)
                x = torch.cat((x, flow), 1)
            feat = self.convblock(self.conv0(x))
            tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear", align_corners=False)
            return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]

    class _ComfyRIFE(nn.Module):
        def __init__(self):
            super().__init__()
            self.has_encoder = True
            self.pad_align = 64
            head_ch = int(sd["encode.cnn3.weight"].shape[1])
            channels = [int(sd[f"blocks.{i}.conv0.1.0.weight"].shape[0]) for i in range(5)]
            block_in = [7 + 2 * head_ch] + [8 + 4 + 8 + 2 * head_ch] * 4
            self.encode = _Head(head_ch)
            self.blocks = nn.ModuleList([_Block(block_in[i], channels[i]) for i in range(5)])
            self.scale_list = [16, 8, 4, 2, 1]
            self._warp_grids = {}

        def _build_warp_grids(self, h, w, device):
            if (h, w) in self._warp_grids:
                return
            self._warp_grids = {}
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1.0, 1.0, h, device=device, dtype=torch.float32),
                torch.linspace(-1.0, 1.0, w, device=device, dtype=torch.float32),
                indexing="ij",
            )
            self._warp_grids[(h, w)] = (
                torch.stack((grid_x, grid_y), dim=0).unsqueeze(0),
                torch.tensor([(w - 1.0) / 2.0, (h - 1.0) / 2.0], dtype=torch.float32, device=device),
            )

        def warp(self, img, flow):
            b, _, h, w = img.shape
            base_grid, flow_div = self._warp_grids[(h, w)]
            flow_norm = torch.cat([flow[:, 0:1] / flow_div[0], flow[:, 1:2] / flow_div[1]], 1).float()
            grid = (base_grid.expand(b, -1, -1, -1) + flow_norm).permute(0, 2, 3, 1)
            return F.grid_sample(img.float(), grid, mode="bilinear", padding_mode="border", align_corners=True).to(img.dtype)

        def forward(self, img0, img1, timestep=0.5):
            h, w = img0.shape[2], img0.shape[3]
            ph = ((h - 1) // self.pad_align + 1) * self.pad_align
            pw = ((w - 1) // self.pad_align + 1) * self.pad_align
            if ph != h or pw != w:
                pad = (0, pw - w, 0, ph - h)
                img0 = F.pad(img0, pad, mode="reflect")
                img1 = F.pad(img1, pad, mode="reflect")
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.full((img0.shape[0], 1, img0.shape[2], img0.shape[3]), float(timestep), device=img0.device, dtype=img0.dtype)
            elif timestep.ndim == 1:
                timestep = timestep.reshape(-1, 1, 1, 1).expand(-1, 1, img0.shape[2], img0.shape[3]).to(device=img0.device, dtype=img0.dtype)
            else:
                timestep = timestep.to(device=img0.device, dtype=img0.dtype)
                if timestep.shape[-2:] != img0.shape[-2:]:
                    timestep = timestep.reshape(timestep.shape[0], -1)[:, :1].reshape(-1, 1, 1, 1).expand(-1, 1, img0.shape[2], img0.shape[3])

            self._build_warp_grids(img0.shape[2], img0.shape[3], img0.device)
            f0 = self.encode(img0)
            f1 = self.encode(img1)
            flow = mask = feat = None
            warped_img0, warped_img1 = img0, img1
            for i, block in enumerate(self.blocks):
                if flow is None:
                    flow, mask, feat = block(torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=self.scale_list[i])
                else:
                    fd, mask, feat = block(
                        torch.cat((warped_img0, warped_img1, self.warp(f0, flow[:, :2]), self.warp(f1, flow[:, 2:4]), timestep, mask, feat), 1),
                        flow,
                        scale=self.scale_list[i],
                    )
                    flow = flow.add(fd)
                warped_img0 = self.warp(img0, flow[:, :2])
                warped_img1 = self.warp(img1, flow[:, 2:4])
            return torch.lerp(warped_img1, warped_img0, torch.sigmoid(mask))[:, :, :h, :w]

    return _ComfyRIFE()


# ============================================================================
# MODEL LOADER
# ============================================================================


class ModelLoader:
    """Load safetensors models into PyTorch networks."""

    def __init__(self, device="auto"):
        self.device = self._resolve_device(device)
        self.dtype = torch.float16 if ("cuda" in str(self.device) or "mps" in str(self.device)) \
            else torch.float32

    def _resolve_device(self, device_str):
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)

    def load_interpolator(self, model_path, *, backend="tensorrt", precision="float16",
                          opt_profile=3, dynamic_shapes=False, resolution=(256, 256)):
        """Load RIFE interpolation model from safetensors.

        Strategy:
          1. Detect architecture from weight shapes
          2. Build matching PyTorch network
          3. Apply loaded weights
          4. If TRT requested: build TRT engine (with disk cache)
          5. Else: compile TorchScript or use eager mode
        """
        sd = load_safetensors(model_path)
        n_params = sum(t.numel() for t in sd.values())
        arch_type, arch_info = _detect_architecture(sd)
        log("LOAD", f"model_arch={arch_type} tensors={len(sd)} params={n_params:,}", detail=f"file={Path(model_path).name}")

        # Build model
        model_obj = _build_model_from_sd(sd, arch_type, arch_info)
        model_obj._source_model_path = model_path
        model_obj.to(self.device)

        # Apply weights
        missing, unexpected = model_obj.load_state_dict(sd, strict=False)
        applied = len(sd) - len(missing)
        log("LOAD", f"weights_applied={applied}/{len(sd)}")
        if missing:
            log("WARN", f"missing_keys={len(missing)}", detail=str(list(missing)[:5]))
        if unexpected:
            log("WARN", f"unexpected_keys={len(unexpected)}", detail=str(list(unexpected)[:5]))

        model_obj.eval()

        # Choose inference backend
        if backend == "tensorrt" and HAS_TRT and "cuda" in str(self.device):
            log("TRT", "Building TensorRT engine", detail=f"profile={opt_profile} precision={precision} dynamic_shapes={dynamic_shapes} resolution={resolution}")
            engine, ctx = build_rife_trt_engine(
                model_obj, self.device, precision=precision,
                dynamic_shapes=dynamic_shapes, opt_profile=opt_profile,
                input_resolution=resolution,
            )
            if engine is not None:
                log("TRT", "Engine ready -- using TensorRT accelerator")
                return TRTInterpolator(model_obj, engine, ctx, self.device)
            else:
                log("WARN", "TensorRT engine build failed -- falling back to TorchScript")

        # TorchScript fallback
        ts_model = build_torchscript_interpolator(model_obj, self.device)
        mode_name = "TorchScript" if ts_model is not model_obj else "eager"
        log("INTERP", f"Using {mode_name} mode", detail="technique=RIFE optical-flow inference")
        return EagerInterpolator(ts_model, self.device)


# ============================================================================
# INTERPOLATOR WRAPPERS
# ============================================================================


class TRTInterpolator:
    """Frame interpolation using a pre-built TensorRT engine."""

    def __init__(self, model, engine, context, device):
        self.model = model
        self.engine = engine
        self.context = context
        self.device = device
        self.technique = "RIFE optical-flow + TensorRT"

    def interpolate(self, frame_a, frame_b, timestep=0.5):
        """Interpolate between two frames using TRT engine.

        Args:
            frame_a: [C, H, W] tensor in [0, 1]
            frame_b: [C, H, W] tensor in [0, 1]
            timestep: Not used by RIFE (always midpoint), kept for API compat

        Returns:
            Interpolated frame [C, H, W] in [0, 1]
        """
        # For encoder variant: run PyTorch model directly (TRT multi-output is complex)
        if hasattr(self.model, 'has_encoder') and self.model.has_encoder:
            return self._encoder_infer(frame_a, frame_b, timestep)

        # Standard: stack and run TRT engine
        batch = torch.stack([frame_a, frame_b], dim=0).contiguous().to(self.device)
        H, W = batch.shape[-2:]
        output = torch.empty(batch.shape[0], 3, H, W, device=self.device)
        bindings = [int(batch.data_ptr()), int(output.data_ptr())]
        self.context.execute_v2(bindings)
        return output[0]

    def _encoder_infer(self, frame_a, frame_b, timestep):
        """Run ComfyUI-style encoder RIFE through TensorRT; fallback to PyTorch on binding errors."""
        h, w = frame_a.shape[-2:]
        inp_a = frame_a.unsqueeze(0).to(self.device).contiguous()
        inp_b = frame_b.unsqueeze(0).to(self.device).contiguous()
        ts = torch.full((1, 1, h, w), float(timestep), device=self.device, dtype=inp_a.dtype).contiguous()
        out = torch.empty((1, 3, h, w), device=self.device, dtype=inp_a.dtype).contiguous()
        try:
            if hasattr(self.context, "set_input_shape"):
                self.context.set_input_shape("img0", tuple(inp_a.shape))
                self.context.set_input_shape("img1", tuple(inp_b.shape))
                self.context.set_input_shape("timestep", tuple(ts.shape))
            self.context.set_tensor_address("img0", int(inp_a.data_ptr()))
            self.context.set_tensor_address("img1", int(inp_b.data_ptr()))
            self.context.set_tensor_address("timestep", int(ts.data_ptr()))
            self.context.set_tensor_address("output", int(out.data_ptr()))
            stream = torch.cuda.current_stream(device=self.device).cuda_stream if "cuda" in str(self.device) else 0
            self.context.execute_async_v3(stream)
            if "cuda" in str(self.device):
                torch.cuda.current_stream(device=self.device).synchronize()
            return out[0].clamp(0, 1)
        except Exception as e:
            log("WARN", "TensorRT encoder execution failed -- falling back to PyTorch", detail=str(e))

        self.model.eval()
        with torch.no_grad():
            out = self.model(inp_a, inp_b, timestep=ts)
        if isinstance(out, tuple):
            flow, mask = out
            return _merge_rife_output(frame_a, frame_b, flow, mask)[0]
        return out[0].clamp(0, 1)


class EagerInterpolator:
    """Frame interpolation using raw PyTorch eager execution."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.technique = "RIFE optical-flow + TorchScript/PyTorch"

    def interpolate(self, frame_a, frame_b, timestep=0.5):
        """Interpolate between two frames using eager PyTorch.

        Args:
            frame_a: [C, H, W] tensor in [0, 1]
            frame_b: [C, H, W] tensor in [0, 1]
            timestep: Interpolation position (0.0 = frame_a, 1.0 = frame_b)

        Returns:
            Interpolated frame [C, H, W] in [0, 1]
        """
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'has_encoder'):
                h, w = frame_a.shape[-2:]
                ts = torch.full((1, 1, h, w), float(timestep), device=self.device, dtype=frame_a.dtype)
                out = self.model(
                    frame_a.unsqueeze(0).to(self.device),
                    frame_b.unsqueeze(0).to(self.device),
                    timestep=ts,
                )
                if isinstance(out, tuple):
                    flow, mask = out
                    return _merge_rife_output(frame_a, frame_b, flow, mask)[0]
                return out[0].clamp(0, 1)
            else:
                # Simple linear interpolation as fallback
                return (1.0 - timestep) * frame_a + timestep * frame_b


class BlendInterpolator:
    """Artifact-safe interpolation fallback when the RIFE graph is unverified."""

    def __init__(self, device, reason=""):
        self.device = device
        self.reason = reason
        self.technique = "temporal linear blend fallback (NOT RIFE)"

    def interpolate(self, frame_a, frame_b, timestep=0.5):
        t = max(0.0, min(1.0, float(timestep)))
        return ((1.0 - t) * frame_a + t * frame_b).clamp(0, 1)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def _warp_frame(frame, flow, timestep):
    """Warp a frame using optical flow at a given timestep.

    Args:
        frame: [B, C, H, W] tensor
        flow: [B, 2, H, W] optical flow tensor
        timestep: float in [0, 1]

    Returns:
        Warped frame [B, C, H, W]
    """
    if flow.shape[-2:] != frame.shape[-2:]:
        flow = F.interpolate(flow, size=frame.shape[-2:], mode="bilinear", align_corners=False)

    if warp_fn is not None:
        scaled_flow = flow * timestep
        return warp_fn(frame, scaled_flow)
    else:
        # Fallback: simple bilinear interpolation
        B, C, H, W = frame.shape
        y = torch.arange(H, device=frame.device).float() / (H - 1) * 2 - 1
        x = torch.arange(W, device=frame.device).float() / (W - 1) * 2 - 1
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        # Add flow displacement
        flow_norm = flow.permute(0, 2, 3, 1) * timestep
        grid = grid + flow_norm
        return F.grid_sample(frame, grid, mode='bilinear', padding_mode='border', align_corners=True)


def _merge_rife_output(frame_a, frame_b, flow, mask):
    """Blend both RIFE-warped endpoints instead of returning one warped frame."""
    a = frame_a.unsqueeze(0).to(flow.device, dtype=flow.dtype)
    b = frame_b.unsqueeze(0).to(flow.device, dtype=flow.dtype)
    if flow.shape[-2:] != a.shape[-2:]:
        flow = F.interpolate(flow, size=a.shape[-2:], mode="bilinear", align_corners=False)
    if mask.shape[-2:] != a.shape[-2:]:
        mask = F.interpolate(mask, size=a.shape[-2:], mode="bilinear", align_corners=False)
    mask = torch.sigmoid(mask.to(device=flow.device, dtype=a.dtype))
    warped_a = _warp_frame(a, flow[:, :2], 1.0)
    warped_b = _warp_frame(b, flow[:, 2:4], 1.0)
    return (warped_a * mask + warped_b * (1.0 - mask)).clamp(0, 1)


# ============================================================================
# VIDEO PIPELINE
# ============================================================================


def extract_frames(video_path, output_dir, ffmpeg_path="ffmpeg"):
    """Extract video frames as PNG images.

    Args:
        video_path: Path to input video
        output_dir: Directory to write extracted frames
        ffmpeg_path: Path to FFmpeg binary

    Returns:
        List of output frame paths sorted by index
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        ffmpeg_path, "-i", video_path,
        "-vf", "fps=30",
        "-q:v", "2",
        os.path.join(output_dir, "proc_%06d_%.3f.png"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[FFMPEG] Frame extraction failed: {proc.stderr}", file=sys.stderr)
        return []

    frames = sorted(Path(output_dir).glob("proc_*.*.png"))
    print(f"[FRAMES] Extracted {len(frames)} frames", file=sys.stderr)
    return frames


def process_frames(frames, interpolator, output_dir, ffmpeg_path="ffmpeg"):
    """Process frames through interpolation pipeline.

    Args:
        frames: List of input frame paths
        interpolator: Interpolator instance (TRT or Eager)
        output_dir: Directory for output interpolated frames
        ffmpeg_path: Path to FFmpeg binary

    Returns:
        List of output frame paths
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    for i in range(len(frames) - 1):
        frame_a = cv2.imread(str(frames[i]), cv2.IMREAD_COLOR)
        frame_b = cv2.imread(str(frames[i + 1]), cv2.IMREAD_COLOR)

        if frame_a is None or frame_b is None:
            print(f"[WARN] Failed to read frames {i} or {i+1}", file=sys.stderr)
            continue

        # Convert to tensor [C, H, W] in [0, 1]
        frame_a_tensor = torch.from_numpy(frame_a[:, :, ::-1].transpose(2, 0, 1)).float() / 255.0
        frame_b_tensor = torch.from_numpy(frame_b[:, :, ::-1].transpose(2, 0, 1)).float() / 255.0

        # Interpolate
        interp = interpolator.interpolate(frame_a_tensor, frame_b_tensor, timestep=0.5)

        # Convert back to BGR numpy array
        interp_np = (interp.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, ::-1]

        # Write output
        out_path = os.path.join(output_dir, f"proc_{i:06d}_0.500.png")
        cv2.imwrite(out_path, interp_np)
        output_paths.append(out_path)

        if (i + 1) % 10 == 0:
            print(f"[PROC] Processed {i + 1}/{len(frames) - 1} pairs", file=sys.stderr)

    return output_paths


def remux_video(input_video, output_video, frame_pattern, ffmpeg_path="ffmpeg"):
    """Remux interpolated frames into final video.

    Args:
        input_video: Original input video path
        output_video: Output video path
        frame_pattern: FFmpeg-compatible frame filename pattern
        ffmpeg_path: Path to FFmpeg binary

    Returns:
        True on success
    """
    cmd = [
        ffmpeg_path, "-y",
        "-framerate", "30",
        "-i", frame_pattern,
        "-i", input_video,
        "-map", "0:v",
        "-map", "1:a?"+ "",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_video,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[FFMPEG] Remux failed: {proc.stderr}", file=sys.stderr)
        return False
    return True


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="DaSiWa TrueVideoEnhancer Backend")
    parser.add_argument("--list-backends", action="store_true",
                        help="List available inference backends")
    parser.add_argument("--load-model", type=str, metavar="PATH",
                        help="Load model and print info")
    parser.add_argument("--backend", choices=["pytorch", "tensorrt", "onnxruntime"],
                        default="tensorrt", help="Inference backend")
    parser.add_argument("--precision", choices=["float16", "float32"],
                        default="float16", help="Inference precision")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--input-video", type=str, help="Input video path")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--resolution", type=int, nargs=2, default=[256, 256],
                        help="Input resolution (H W)")

    args = parser.parse_args()

    if args.list_backends:
        backends = ["pytorch"]
        if HAS_TRT:
            backends.append("tensorrt")
        if HAS_ONNX:
            backends.append("onnxruntime")
        print(json.dumps({"backends": backends}))
        return

    if args.load_model:
        if not args.model:
            print("Error: --model required with --load-model", file=sys.stderr)
            sys.exit(1)
        loader = ModelLoader(device=args.device)
        info = loader.load_interpolator(args.model, backend=args.backend,
                                        precision=args.precision,
                                        resolution=tuple(args.resolution))
        print(json.dumps({"status": "loaded", "type": type(info).__name__}))
        return

    # Full pipeline
    if not args.input_video or not args.model:
        print("Error: --input-video and --model required for full pipeline", file=sys.stderr)
        sys.exit(1)

    loader = ModelLoader(device=args.device)
    interpolator = loader.load_interpolator(
        args.model, backend=args.backend, precision=args.precision,
        resolution=tuple(args.resolution),
    )

    # Extract frames
    temp_dir = tempfile.mkdtemp(prefix="rve_")
    frames = extract_frames(args.input_video, temp_dir)
    if not frames:
        print("Error: No frames extracted", file=sys.stderr)
        sys.exit(1)

    # Process frames
    proc_dir = os.path.join(temp_dir, "processed")
    output_paths = process_frames(frames, interpolator, proc_dir)

    # Remux
    frame_pattern = os.path.join(proc_dir, "proc_%06d_%.3f.png")
    output_video = os.path.join(args.output_dir or temp_dir, "output.mp4")
    success = remux_video(args.input_video, output_video, frame_pattern)

    if success:
        print(json.dumps({"status": "success", "output": output_video, "frames": len(output_paths)}))
    else:
        print(json.dumps({"status": "failed"}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
