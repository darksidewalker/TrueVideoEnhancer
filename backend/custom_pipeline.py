"""Composable bounded processing: restore -> spatial -> temporal -> SDR-to-HDR.

AI and RIFE consume normalized floating-point SDR, including 16-bit rawvideo.
HDR conversion is deliberately downstream of both models, never PQ model input.
"""
from collections import deque
import math
import numpy as np
from streaming_pipeline import FrameProcessor
from rtx_vfx import RTXVfxPipeline, RTXFrameInterpolator, _input, _output


def validate_selection(args, info=None):
    if not math.isfinite(args.target_fps) or args.target_fps < 0:
        raise ValueError('target_fps must be finite and nonnegative')
    if args.scale < 1 or args.override_upscale_scale < 0:
        raise ValueError('scale must be positive and override_upscale_scale nonnegative')
    if args.spatial_backend not in ('none', 'ai', 'nvidia'):
        raise ValueError('custom spatial_backend must be none, ai or nvidia')
    if args.temporal_backend not in ('none', 'rife', 'nvidia'):
        raise ValueError('custom temporal_backend must be none, rife or nvidia')
    if args.spatial_backend == 'ai' and not args.upscale_model:
        raise ValueError('AI upscale requires --upscale_model')
    if args.temporal_backend == 'rife' and not args.interpolate_model:
        raise ValueError('RIFE requires --interpolate_model')
    if args.spatial_backend == 'nvidia' and args.rtx_upscale == 'Off':
        raise ValueError('NVIDIA upscale requires an explicit --rtx_upscale quality mode')
    if args.spatial_backend != 'nvidia' and args.rtx_upscale != 'Off':
        raise ValueError('--rtx_upscale conflicts with selected spatial_backend')
    if args.rtx_frame_generation and args.temporal_backend != 'nvidia':
        raise ValueError('--rtx_frame_generation conflicts with selected temporal_backend')
    if args.vsr_strength != 0:
        raise ValueError('custom uses independent RTX strength controls, not legacy vsr_strength')
    if args.custom_encoder or args.hdr_mode or args.slomo_mode:
        raise ValueError('custom does not support custom_encoder/hdr_mode/slomo_mode; use --rtx_hdr for SDR-to-HDR')
    if info:
        scale = args.override_upscale_scale or args.scale
        if scale != 1 and args.spatial_backend == 'none':
            raise ValueError('requested scale requires an AI or NVIDIA spatial backend')
        if args.target_fps and abs(args.target_fps - info['fps']) > max(1e-6, info['fps'] * 1e-6):
            if args.temporal_backend == 'none':
                raise ValueError('requested FPS change requires RIFE or NVIDIA temporal backend')
            if args.target_fps < info['fps']:
                raise ValueError('custom interpolation cannot reduce source FPS')


def needs_nvidia(args):
    return bool(args.denoise or args.deblur or args.rtx_hdr or
                args.spatial_backend == 'nvidia' or args.temporal_backend == 'nvidia')


class ComposableProcessor:
    """Two owned source endpoints; restore/spatial once, HDR once per output."""
    def __init__(self, spatial, interpolator, hdr, dtype):
        self.spatial, self.interpolator, self.hdr, self.dtype = spatial, interpolator, hdr, dtype
        self.cache = deque(maxlen=2)

    def endpoint(self, frame):
        for original, processed in self.cache:
            if original is frame:
                return processed
        processed = self.spatial.upscale(self.spatial.restore(frame))
        if (processed.shape[1], processed.shape[0]) != self.spatial.target_size:
            import cv2
            processed = cv2.resize(processed, self.spatial.target_size, interpolation=cv2.INTER_LANCZOS4)
        processed = np.ascontiguousarray(processed)
        self.cache.append((frame, processed))
        return processed

    def process(self, a, b, timestep):
        left = self.endpoint(a)
        if self.interpolator is not None and 1e-5 < timestep < 1 - 1e-5:
            right = self.endpoint(b)
            output = _output(self.interpolator.interpolate(
                _input(left, 'cpu'), _input(right, 'cpu'), timestep=timestep), self.dtype)
        elif timestep >= 1 - 1e-5:
            output = self.endpoint(b)
        else:
            output = left
        if self.hdr is not None:
            output = self.hdr.process(output, np.empty_like(output, dtype=self.dtype))
        return np.ascontiguousarray(output)


def build_processor(args, cli, info, target, factor, dtype, resources):
    # Register each resource immediately, so a later model-load failure cleans it up.
    def own(value):
        resources.append(value)
        return value
    encoding = 'rgb10a2' if dtype == np.uint16 else args.rtx_image_encoding
    restorer = own(RTXVfxPipeline(
        denoise=args.denoise, denoise_quality=args.denoise_quality,
        deblur=args.deblur, deblur_quality=args.deblur_quality,
        denoise_strength=args.rtx_denoise_strength,
        deblur_strength=args.rtx_deblur_strength,
        image_encoding=encoding, device_index=args.pytorch_gpu_id,
    )) if args.denoise or args.deblur else None
    vfx = own(RTXVfxPipeline(
        upscale_mode=args.rtx_upscale, upscale_quality=args.rtx_upscale_quality,
        upscale_strength=args.rtx_upscale_strength, image_encoding=encoding,
        device_index=args.pytorch_gpu_id,
    )) if args.spatial_backend == 'nvidia' else None
    ai, tile = None, 0
    if args.spatial_backend == 'ai':
        path = cli.resolve_model_path(args.upscale_model)
        requested_tile = args.tilesize
        fallback = ()
        if (args.override_upscale_scale or args.scale) >= 4 and '2x' in path.lower() and not requested_tile:
            requested_tile, fallback = 256, (128,)
        ai, tile, size, failures = cli.create_optimized_upscaler(
            path, width=info['width'], height=info['height'], backend=args.backend,
            requested_tile=requested_tile, device=args.device, gpu_id=args.pytorch_gpu_id,
            precision=args.precision, fallback_tile_sizes=fallback,
        )
        own(ai)
        # Models run float16/float32 SDR; preserve >8-bit storage at every seam.
        ai.output_dtype = dtype
        cli.log('MODEL', f'custom AI upscale={path}', detail=f'tile={tile} input={size} precision={args.precision} storage={np.dtype(dtype)}')
        for failed, reason in failures:
            cli.log('WARN', f'AI compile fallback {failed}: {reason}')
    spatial = FrameProcessor(None, ai, tile, target,
                             lambda f: _input(f, 'cpu'), lambda t: _output(t, dtype),
                             restorer=restorer, vfx_upscaler=vfx, output_dtype=dtype,
                             force_upscale=ai is not None)
    temporal = None
    if factor > 1:
        if args.temporal_backend == 'rife':
            temporal = own(cli.build_interpolator(args, *target))
        elif args.temporal_backend == 'nvidia':
            temporal = own(RTXFrameInterpolator(
                device_index=args.pytorch_gpu_id, mode=args.rtx_fg_mode,
                auto_shot=args.rtx_auto_shot_detection, shot_change=args.rtx_shot_change,
                image_encoding=encoding,
            ))
    hdr = own(RTXVfxPipeline(
        hdr=True, image_encoding=encoding, hdr_contrast=args.rtx_hdr_contrast,
        hdr_saturation=args.rtx_hdr_saturation, hdr_middle_gray=args.rtx_hdr_middle_gray,
        hdr_luminance=args.rtx_hdr_luminance, hdr_debanding=args.rtx_hdr_debanding,
        device_index=args.pytorch_gpu_id,
    )) if args.rtx_hdr else None
    cli.log('TECH', f'custom: restoration={bool(restorer)} spatial={args.spatial_backend} temporal={args.temporal_backend}/{factor:g} HDR-last={bool(hdr)}',
            detail=f'AI/RIFE normalized floating-point SDR; raw storage={np.dtype(dtype)}')
    return ComposableProcessor(spatial, temporal, hdr, dtype)


def close_resources(resources):
    while resources:
        resource = resources.pop()
        close = getattr(resource, 'close', None)
        if close:
            close()
