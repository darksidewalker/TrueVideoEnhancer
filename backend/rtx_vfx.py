"""Bounded NVIDIA Video SDK stages. No Torch/CUDA initialization at import.

SDK 0.2 native calls use its verified blocking default-stream path. Every
borrowed SDK output is copied before reuse; no full-video allocations.
"""

from __future__ import annotations
import inspect
import math
from importlib import metadata
import numpy as np

QUALITY_LEVELS = ("Low", "Medium", "High", "Ultra")
UPSCALE_MODES = ("Off", "VSR", "High Bitrate", "Streaming Medium", "Streaming Ultra")


def _sdk():
    try:
        import nvvfx

        return nvvfx
    except Exception as exc:
        raise RuntimeError(
            f"NVIDIA VFX unavailable: {exc}; install nvidia-vfx in the runtime venv"
        ) from exc


def _params(obj):
    try:
        return inspect.signature(obj).parameters
    except (ValueError, TypeError):
        return {}


def capabilities():
    """API-only diagnostics, never create effects or allocate GPU memory."""
    out = dict(
        available=False,
        upscale=False,
        denoise=False,
        deblur=False,
        version="unavailable",
        strength=False,
        encoding=False,
        hdr=False,
        frame_generation=False,
        frame_generation_encoding=False,
        streaming=[],
        quality_levels=[],
        error=None,
    )
    try:
        sdk = _sdk()
        try:
            out["version"] = metadata.version("nvidia-vfx")
        except metadata.PackageNotFoundError:
            out["version"] = "unknown"
        vsr = getattr(sdk, "VideoSuperRes", None)
        quality = getattr(
            getattr(sdk, "effects", None),
            "QualityLevel",
            getattr(vsr, "QualityLevel", None),
        )
        enc = getattr(sdk, "ImageEncoding", None)
        fg = getattr(sdk, "VideoFrameGeneration", None)
        hdr = getattr(sdk, "TrueHDR", None)
        out.update(
            available=callable(vsr),
            upscale=callable(vsr),
            denoise=callable(vsr) and any(n.startswith('DENOISE_') for n in dir(quality)),
            deblur=callable(vsr) and any(n.startswith('DEBLUR_') for n in dir(quality)),
            strength="strength" in _params(vsr),
            encoding="image_encoding" in _params(vsr) and hasattr(enc, "RGB10A2"),
            hdr=callable(hdr)
            and all(
                n in _params(hdr)
                for n in (
                    "contrast",
                    "saturation",
                    "middle_gray",
                    "luminance",
                    "debanding_off",
                    "device",
                )
            ),
            frame_generation=callable(getattr(fg, "run_at_timestep", None)),
            frame_generation_encoding="image_encoding" in _params(fg)
            and hasattr(enc, "RGB10A2"),
            quality_levels=[n for n in dir(quality) if n.isupper()],
            streaming=[
                label
                for label, attr in [
                    ("Streaming Medium", "STREAMING_MEDIUM"),
                    ("Streaming Ultra", "STREAMING_ULTRA"),
                ]
                if hasattr(quality, attr)
            ],
        )
    except Exception as exc:
        out["error"] = str(exc)
    return out


def nvvfx_available():
    return capabilities()["available"]


def validate_options(options):
    for name, lo, hi in [
        ("rtx_denoise_strength", 0, 1),
        ("rtx_deblur_strength", 0, 1),
        ("rtx_upscale_strength", 0, 1),
        ("rtx_hdr_contrast", 0, 200),
        ("rtx_hdr_saturation", 0, 200),
        ("rtx_hdr_middle_gray", 10, 100),
        ("rtx_hdr_luminance", 400, 2000),
    ]:
        if name in options:
            value = float(options[name])
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{name} must be finite in [{lo}, {hi}]")
            if name.startswith("rtx_hdr_") and not value.is_integer():
                raise ValueError(
                    f"{name} must be an integer (SDK control); fractional values cannot be preserved"
                )


def pack_rgb10a2(rgb):
    import torch

    values = (rgb[:3].clamp(0, 1) * 1023).round().to(torch.int32)
    return (
        (values[0] | (values[1] << 10) | (values[2] << 20) | -1073741824)
        .view(torch.uint32)
        .contiguous()
    )


def unpack_rgb10a2(packed):
    import torch

    words = packed.view(torch.int32)
    return (
        torch.stack([(words >> s) & 1023 for s in (0, 10, 20)])
        .float()
        .div_(1023)
        .contiguous()
    )


def _quality_attr(mode, quality):
    if mode == "High Bitrate":
        return "HIGHBITRATE_" + quality.upper()
    if mode.startswith("Streaming"):
        return "STREAMING_" + mode.split()[-1].upper()
    if mode in ("Denoise", "Deblur"):
        return mode.upper() + "_" + quality.upper()
    return quality.upper()


def _resolve_quality_level(quality_enum, mode, quality):
    name = _quality_attr(mode, quality)
    if not hasattr(quality_enum, name):
        raise RuntimeError(f"NVIDIA VFX lacks requested quality {name}")
    return getattr(quality_enum, name), name


def _close(effect):
    if effect is not None:
        effect.close()


def _owned(result, packed):
    import torch

    value = torch.from_dlpack(result.image)
    return (unpack_rgb10a2(value) if packed else value).clone().contiguous()


def _input(frame, device):
    import torch

    divisor = 65535.0 if frame.dtype == np.uint16 else 255.0
    return (
        torch.from_numpy(np.ascontiguousarray(frame[:, :, ::-1].transpose(2, 0, 1)))
        .to(device=device, dtype=torch.float32)
        .div_(divisor)
    )


def _output(tensor, dtype):
    import torch

    scale = 65535.0 if dtype == np.uint16 else 255.0
    typ = torch.uint16 if dtype == np.uint16 else torch.uint8
    return np.ascontiguousarray(
        tensor.clamp(0, 1)
        .mul(scale)
        .round()
        .to(typ)
        .permute(1, 2, 0)
        .cpu()
        .numpy()[:, :, ::-1]
    )


class RTXVfxPipeline:
    def __init__(
        self,
        *,
        denoise=False,
        denoise_quality="Ultra",
        deblur=False,
        deblur_quality="Ultra",
        upscale_mode="Off",
        upscale_quality="Ultra",
        strength=1.0,
        denoise_strength=None,
        deblur_strength=None,
        upscale_strength=None,
        image_encoding="rgb8",
        hdr=False,
        hdr_contrast=100,
        hdr_saturation=100,
        hdr_middle_gray=50,
        hdr_luminance=650,
        hdr_debanding=True,
        device_index=0,
    ):
        self.device_index = device_index
        self.packed = image_encoding == "rgb10a2"
        self.hdr = hdr
        self.specs = []
        for enabled, mode, quality, s in [
            (denoise, "Denoise", denoise_quality, denoise_strength),
            (deblur, "Deblur", deblur_quality, deblur_strength),
            (upscale_mode != "Off", upscale_mode, upscale_quality, upscale_strength),
        ]:
            if enabled:
                self.specs.append((mode, quality, strength if s is None else s))
        self.hdr_options = dict(
            contrast=int(hdr_contrast),
            saturation=int(hdr_saturation),
            middle_gray=int(hdr_middle_gray),
            luminance=int(hdr_luminance),
            debanding_off=int(not hdr_debanding),
            device=device_index,
        )
        self._effects = []
        self._hdr = None
        self._shape = None

    @property
    def active(self):
        return bool(self.specs or self.hdr)

    def _ensure_loaded(self, source_w, source_h, target_w, target_h):
        shape = (source_w, source_h, target_w, target_h)
        if self._shape == shape:
            return
        if self._shape is not None:
            raise RuntimeError("VFX dimensions changed during streaming")
        sdk = _sdk()
        cap = capabilities()
        if self.packed and not cap["encoding"]:
            raise RuntimeError("NVIDIA VFX RGB10A2 unavailable")
        if self.hdr and not cap["hdr"]:
            raise RuntimeError("NVIDIA TrueHDR unavailable")
        quality = getattr(
            getattr(sdk, "effects", None),
            "QualityLevel",
            getattr(sdk.VideoSuperRes, "QualityLevel", None),
        )
        try:
            for mode, q, strength in self.specs:
                level, name = _resolve_quality_level(quality, mode, q)
                params = _params(sdk.VideoSuperRes)
                kwargs = dict(quality=level, device=self.device_index)
                if "strength" in params:
                    kwargs["strength"] = strength
                elif strength != 1.0:
                    raise RuntimeError("NVIDIA VFX strength unavailable")
                if "image_encoding" in params:
                    kwargs["image_encoding"] = getattr(
                        sdk.ImageEncoding, "RGB10A2" if self.packed else "RGB8"
                    )
                effect = sdk.VideoSuperRes(**kwargs)
                self._effects.append((effect, name))
                effect.output_width, effect.output_height = (
                    (source_w, source_h)
                    if mode in ("Denoise", "Deblur")
                    else (target_w, target_h)
                )
                effect.load()
            if self.hdr:
                self._hdr = sdk.TrueHDR(**self.hdr_options)
            self._shape = shape
        except BaseException:
            self.close()
            raise

    def process(self, frame_bgr, out_bgr):
        import torch

        device = torch.device(f"cuda:{self.device_index}")
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA VFX requires CUDA")
        caller = torch.cuda.current_stream(device)
        caller.synchronize()
        default = torch.cuda.default_stream(device)
        with torch.cuda.device(device), torch.inference_mode(), torch.cuda.stream(
            default
        ):
            try:
                # Native SDK 0.2 stream must be selected before load as well as run.
                self._ensure_loaded(
                    frame_bgr.shape[1],
                    frame_bgr.shape[0],
                    out_bgr.shape[1],
                    out_bgr.shape[0],
                )
                frame = _input(frame_bgr, device)
                for effect, name in self._effects:
                    frame = _owned(
                        effect.run(
                            pack_rgb10a2(frame) if self.packed else frame,
                            **(
                                {"stream_ptr": 0}
                                if "stream_ptr" in _params(effect.run)
                                else {}
                            ),
                        ),
                        self.packed,
                    )
                if tuple(frame.shape[-2:]) != tuple(out_bgr.shape[:2]):
                    frame = (
                        torch.nn.functional.interpolate(
                            frame.unsqueeze(0),
                            size=out_bgr.shape[:2],
                            mode="bicubic",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .clamp(0, 1)
                    )
                if self._hdr is not None:
                    frame = _owned(self._hdr.run(frame, stream_ptr=0), True)
                np.copyto(out_bgr, _output(frame, out_bgr.dtype))
                return out_bgr
            finally:
                default.synchronize()

    def close(self):
        effects, self._effects = self._effects, []
        hdr, self._hdr = self._hdr, None
        self._shape = None
        for effect, _ in effects:
            _close(effect)
        _close(hdr)


class RTXFrameInterpolator:
    technique = "NVIDIA VideoFrameGeneration run_at_timestep"

    def __init__(
        self,
        *,
        device_index=0,
        mode="MEDIUM",
        auto_shot=True,
        shot_change=False,
        image_encoding="rgb8",
    ):
        self.device_index = device_index
        self.mode = mode
        self.auto_shot = auto_shot
        self.shot_change = shot_change
        self.packed = image_encoding == "rgb10a2"
        self.effect = None
        self.shape = None

    def interpolate(self, a, b, *, timestep):
        import torch

        sdk = _sdk()
        cap = capabilities()
        if not cap["frame_generation"]:
            raise RuntimeError("NVIDIA frame generation unavailable")
        if self.packed and not cap["frame_generation_encoding"]:
            raise RuntimeError("NVIDIA frame generation RGB10A2 unavailable")
        device = torch.device(f"cuda:{self.device_index}")
        torch.cuda.synchronize(device)
        with torch.cuda.device(device), torch.inference_mode(), torch.cuda.stream(
            torch.cuda.default_stream(device)
        ):
            try:
                if self.effect is None:
                    kwargs = dict(
                        mode=getattr(sdk.VideoFrameGeneration.Mode, self.mode),
                        automatic_shot_change_detection_enabled=self.auto_shot,
                        device=self.device_index,
                    )
                    if "image_encoding" in _params(sdk.VideoFrameGeneration):
                        kwargs["image_encoding"] = getattr(
                            sdk.ImageEncoding, "RGB10A2" if self.packed else "RGB8"
                        )
                    self.effect = sdk.VideoFrameGeneration(**kwargs)
                    self.shape = tuple(a.shape[-2:])
                    self.effect.input_height, self.effect.input_width = self.shape
                    self.effect.frame_multiplier = 2
                    self.effect.load()
                if tuple(a.shape[-2:]) != self.shape:
                    raise RuntimeError("Frame generation dimensions changed")
                a = a.to(device=device, dtype=torch.float32).contiguous()
                b = b.to(device=device, dtype=torch.float32).contiguous()
                if self.packed:
                    a, b = pack_rgb10a2(a), pack_rgb10a2(b)
                return _owned(
                    self.effect.run_at_timestep(
                        a,
                        b,
                        float(timestep),
                        shot_change=self.shot_change,
                        non_blocking=False,
                        stream_ptr=0,
                    ),
                    self.packed,
                )
            finally:
                torch.cuda.synchronize(device)

    def close(self):
        effect, self.effect = self.effect, None
        _close(effect)


def precision_video_args(args, *, hdr=False):
    args = list(args)
    codec = args[args.index("-c:v") + 1]
    supported = {
        "hevc_nvenc",
        "av1_nvenc",
        "libx265",
        "libsvtav1",
        "libvpx-vp9",
        "ffv1",
        "prores_ks",
    }
    if codec not in supported:
        raise ValueError(
            f"{codec} cannot preserve requested NVIDIA VFX 10-bit/HDR output; select HEVC/AV1/VP9/FFV1/ProRes"
        )
    i = args.index("-pix_fmt")
    args[i + 1] = "yuv422p10le" if codec == "prores_ks" else "yuv420p10le"
    # Packed SDK outputs are normalized RGB signal values, NOT linear light.
    matrix = "bt2020nc" if hdr else "bt709"
    transfer = "smpte2084" if hdr else "bt709"
    primaries = "bt2020" if hdr else "bt709"
    args += [
        "-vf",
        f"zscale=matrixin=gbr:rangein=full:primariesin={primaries}:primaries={primaries}:transferin={transfer}:transfer={transfer}:matrix={matrix}:range=limited:dither=none",
        "-color_primaries",
        primaries,
        "-color_trc",
        transfer,
        "-colorspace",
        matrix,
        "-color_range",
        "tv",
    ]
    if codec == "libx265":
        args += [
            "-x265-params",
            f"colorprim={9 if hdr else 1}:transfer={16 if hdr else 1}:colormatrix={9 if hdr else 1}:range=limited:pools=2:frame-threads=1",
        ]
    return args
