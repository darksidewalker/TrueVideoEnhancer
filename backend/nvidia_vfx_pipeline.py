"""NVIDIA-only bounded streaming orchestration (legacy pipelines stay separate)."""

import math
import sys
import time
from pathlib import Path
import numpy as np
from rtx_vfx import (
    RTXVfxPipeline,
    RTXFrameInterpolator,
    capabilities,
    validate_options,
    precision_video_args,
    _input,
    _output,
)
from streaming_pipeline import (
    FrameProcessor,
    RawVideoReader,
    RawVideoWriter,
    encode_command,
    queue_size_for_frame,
    stream_frames,
    validate_resource_budget,
)


def preview_sdr(frame, hdr=False):
    signal = frame.astype(np.float32) / (65535 if frame.dtype == np.uint16 else 255)
    if hdr:
        # ST2084 EOTF -> display-referred SDR preview, never used for output.
        m1, m2 = 2610 / 16384, 2523 / 32
        c1, c2, c3 = 3424 / 4096, 2413 / 128, 2392 / 128
        p = np.power(signal, 1 / m2)
        linear = (
            np.power(np.maximum(p - c1, 0) / np.maximum(c2 - c3 * p, 1e-8), 1 / m1)
            * 10000
            / 100
        )
        rgb = linear[:, :, ::-1] @ np.array(
            [
                [1.6605, -0.1246, -0.0182],
                [-0.5876, 1.1329, -0.1006],
                [-0.0728, -0.0083, 1.1187],
            ],
            np.float32,
        )
        linear = np.maximum(rgb[:, :, ::-1], 0)
        signal = np.power(linear / (1 + linear), 1 / 2.2)
    return np.ascontiguousarray((np.clip(signal, 0, 1) * 255).round().astype(np.uint8))


def render_nvidia(args, cli):
    from contextlib import ExitStack
    with ExitStack() as cleanup:
        return _render_nvidia(args, cli, cleanup)


def _render_nvidia(args, cli, cleanup):
    custom = getattr(args, 'processing_mode', 'legacy') == 'custom'
    from custom_pipeline import validate_selection, needs_nvidia, build_processor, close_resources
    native = needs_nvidia(args) if custom else True
    if custom:
        validate_selection(args)
    from render_progress import RenderProgress, frame_total
    render_progress = RenderProgress()
    render_progress.preparing()
    from render_contract import validate_preview_seconds, preview_output_path
    duration = validate_preview_seconds(getattr(args, 'preview_seconds', 0))
    args.output = preview_output_path(args.output, duration)
    validate_options(vars(args))
    if args.vsr_strength != 0:
        raise ValueError(
            "nvidia-vfx uses independent --rtx_denoise_strength/--rtx_deblur_strength/--rtx_upscale_strength; --vsr_strength is legacy-only"
        )
    if not custom and (args.upscale_model or args.interpolate_model):
        raise ValueError(
            "nvidia-vfx pipeline does not use model files; clear --upscale_model/--interpolate_model"
        )
    if not custom and not (
        args.denoise
        or args.deblur
        or args.rtx_upscale != "Off"
        or args.rtx_hdr
        or args.rtx_frame_generation
    ):
        raise ValueError("nvidia-vfx requires at least one enabled VFX effect")
    if args.start_time or args.end_time or args.custom_encoder or args.hdr_mode:
        raise ValueError(
            "NVIDIA streaming does not support trim/custom_encoder/hdr_mode; TrueHDR uses --rtx_hdr"
        )
    info = cli.probe_video(args.input)
    if custom:
        validate_selection(args, info)
    if duration and info['duration'] > 0:
        duration = min(duration, info['duration'])
    # Fail codec/container validation before CUDA checks or effect construction.
    precise = (args.rtx_image_encoding == 'rgb10a2' or args.rtx_hdr or
               (custom and any(bits in args.video_pixel_format for bits in ('10', '12', '16'))))
    encoder, audio, subtitle = cli.normalize_encode_settings(
        args.output, args.video_encoder_preset, args.audio_encoder_preset,
        args.subtitle_encoder_preset, info['audio_codec'], info['subtitle_codec'],
        precise=precise, hdr=args.rtx_hdr,
    )
    timing = cli._run(
        [
            cli._ffprobe(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,avg_frame_rate,field_order",
            "-of",
            "json",
            args.input,
        ]
    )
    import json

    rates = json.loads(timing.stdout)["streams"][0]
    nominal = cli._ratio_to_float(rates.get("r_frame_rate", "0/1"))
    if (native or custom) and nominal and abs(nominal - info["fps"]) > max(0.001, info["fps"] * 0.0001):
        raise ValueError(
            "NVIDIA rawvideo scheduler requires CFR source; normalize variable/interlaced cadence to CFR explicitly before enhancement"
        )
    if (native or custom) and rates.get('field_order', 'unknown') not in ('unknown', 'progressive'):
        raise ValueError('streaming processing requires progressive CFR source; deinterlace explicitly first')
    # TrueHDR is SDR->HDR, not HDR->HDR enhancement. Avoid falsely interpreting PQ as SDR.
    probe = cli._run(
        [
            cli._ffprobe(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=color_transfer",
            "-of",
            "default=nw=1:nk=1",
            args.input,
        ]
    )
    if (native or custom) and probe.stdout.strip() in ("smpte2084", "arib-std-b67"):
        raise ValueError(
            "NVIDIA VFX expects SDR input; explicitly tone-map HDR source before enhancement"
        )
    cap = capabilities() if native else {'version': 'unused', 'available': True, 'encoding': True}
    if native and not cap["available"]:
        raise RuntimeError(cap["error"] or "NVIDIA VFX unavailable")
    if args.rtx_hdr and not cap["hdr"]:
        raise RuntimeError("TrueHDR API unavailable")
    if (args.rtx_frame_generation or (custom and args.temporal_backend == 'nvidia')) and not cap["frame_generation"]:
        raise RuntimeError("Frame generation API unavailable")
    if native and precise and not cap["encoding"]:
        raise RuntimeError("RGB10A2 API unavailable")
    import torch

    if native and (
        not torch.cuda.is_available()
        or not 0 <= args.pytorch_gpu_id < torch.cuda.device_count()
    ):
        raise RuntimeError("NVIDIA VFX requires a valid CUDA GPU index")
    if (
        args.rtx_upscale.startswith("Streaming")
        and torch.cuda.get_device_capability(args.pytorch_gpu_id)[0] < 8
    ):
        raise RuntimeError("NVIDIA streaming quality modes require Ampere or newer")
    target_fps, factor = cli.derive_target_fps(info["fps"], args.target_fps)
    if not custom and factor > 1 and not args.rtx_frame_generation:
        raise ValueError("FPS increase in nvidia-vfx requires --rtx_frame_generation")
    target = cli.derive_output_resolution(
        info["width"], info["height"], args.scale, args.override_upscale_scale
    )
    if not custom and target != (info["width"], info["height"]) and args.rtx_upscale == "Off":
        raise ValueError("NVIDIA scaling requires an explicit --rtx_upscale mode")
    raw = "bgr48le" if precise else "bgr24"
    dtype = np.uint16 if precise else np.uint8

    video = cli.encoder_args(encoder, args.crf, args.video_pixel_format)
    if precise:
        video = precision_video_args(video, hdr=args.rtx_hdr)
    codec = video[video.index("-c:v") + 1]
    output = cli.resolve_output_codec_path(
        args.output, args.video_encoder_preset, codec
    )
    from render_contract import validate_output_path
    validate_output_path(args.input, output, args.overwrite)
    print(f"<OUTPUT_PATH>{output}", file=sys.stderr, flush=True)
    print(f"<VIDEO_CODEC>{codec}", file=sys.stderr, flush=True)
    encode_output = output
    if custom:
        import os
        import tempfile
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fd, encode_output = tempfile.mkstemp(prefix='.tve-render-', suffix=Path(output).suffix, dir=Path(output).parent)
        os.close(fd)
        cleanup.callback(lambda: Path(encode_output).unlink(missing_ok=True))
    budget = validate_resource_budget(
        source_width=info["width"],
        source_height=info["height"],
        output_width=target[0],
        output_height=target[1],
        bytes_per_channel=2 if precise else 1,
    )
    queue = queue_size_for_frame(
        info["width"] * info["height"] * (6 if precise else 3),
        target[0] * target[1] * (6 if precise else 3),
    )
    resources = []
    cleanup.callback(close_resources, resources)
    vfx = fg = None
    try:
        if custom:
            processor = build_processor(args, cli, info, target, factor, dtype, resources)
        else:
            vfx = RTXVfxPipeline(
                denoise=args.denoise,
                denoise_quality=args.denoise_quality,
                deblur=args.deblur,
                deblur_quality=args.deblur_quality,
                upscale_mode=args.rtx_upscale,
                upscale_quality=args.rtx_upscale_quality,
                denoise_strength=args.rtx_denoise_strength,
                deblur_strength=args.rtx_deblur_strength,
                upscale_strength=args.rtx_upscale_strength,
                image_encoding=args.rtx_image_encoding,
                hdr=args.rtx_hdr,
                hdr_contrast=args.rtx_hdr_contrast,
                hdr_saturation=args.rtx_hdr_saturation,
                hdr_middle_gray=args.rtx_hdr_middle_gray,
                hdr_luminance=args.rtx_hdr_luminance,
                hdr_debanding=args.rtx_hdr_debanding,
                device_index=args.pytorch_gpu_id,
            )
            fg = (
                RTXFrameInterpolator(
                    device_index=args.pytorch_gpu_id,
                    mode=args.rtx_fg_mode,
                    auto_shot=args.rtx_auto_shot_detection,
                    shot_change=args.rtx_shot_change,
                    image_encoding=args.rtx_image_encoding,
                )
                if factor > 1
                else None
            )
            processor = FrameProcessor(
                interpolator=fg,
                upscaler=None,
                tile_size=0,
                target_size=target,
                to_tensor=lambda f: _input(f, "cpu"),
                to_bgr=lambda t: _output(t, dtype),
                vfx_upscaler=vfx,
                output_dtype=dtype,
            )
    except BaseException:
        close_resources(resources)
        if fg:
            fg.close()
        if vfx:
            vfx.close()
        raise
    reader = RawVideoReader(
        args.input,
        width=info["width"],
        height=info["height"],
        queue_size=queue,
        pixel_format=raw,
        duration=duration,
    )
    writer = RawVideoWriter(
        encode_command(
            args.input,
            encode_output,
            width=target[0],
            height=target[1],
            fps=target_fps,
            video_args=video,
            audio_args=cli.audio_args(audio, args.audio_bitrate),
            subtitle_args=cli.subtitle_args(subtitle),
            pixel_format=raw,
            duration=duration,
        ),
        queue_size=queue,
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cli.log(
        "TECH",
        (f'Custom streaming: spatial={args.spatial_backend} temporal={args.temporal_backend if factor > 1 else "none (source cadence)"} TrueHDR={args.rtx_hdr} SDK={cap["version"]}'
         if custom else f'NVIDIA VFX {cap["version"]}: FG={bool(fg)}/{args.rtx_fg_mode} denoise={args.denoise}/{args.denoise_quality}/{args.rtx_denoise_strength} deblur={args.deblur}/{args.deblur_quality}/{args.rtx_deblur_strength} upscale={args.rtx_upscale}/{args.rtx_upscale_quality}/{args.rtx_upscale_strength} TrueHDR={args.rtx_hdr}'),
        detail=f'encoding={"rgb10a2" if custom and precise else args.rtx_image_encoding} raw={raw} source_fps={info["fps"]} target_fps={target_fps} resolution={target} queue={queue} host_budget={budget} codec={codec}',
    )
    render_progress.total = frame_total(duration or info["duration"], target_fps)
    render_progress.rendering()
    progress = render_progress.update

    previews = 0

    def preview(frame):
        nonlocal previews
        if previews % 10 == 0:
            cli.emit_live_preview(preview_sdr(frame, args.rtx_hdr))
        previews += 1

    try:
        writer.start()
        reader.start()
        count = stream_frames(
            reader,
            writer,
            processor,
            source_fps=info["fps"],
            target_fps=target_fps,
            progress_cb=progress,
            preview_cb=preview,
            preserve_duration=True,
            duration=duration,
        )
        render_progress.finalizing(count)
        writer.finish()
    finally:
        reader.close()
        writer.close()
        if fg:
            fg.close()
        if vfx:
            vfx.close()
        close_resources(resources)
    # Validate actual bit depth and tags: no successful silent FFmpeg downgrade.
    if precise:
        import json

        result = cli._run(
            [
                cli._ffprobe(),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=pix_fmt,color_transfer,color_primaries,color_space",
                "-of",
                "json",
                encode_output,
            ]
        )
        stream = json.loads(result.stdout)["streams"][0]
        if (
            "10" not in stream["pix_fmt"]
            and "12" not in stream["pix_fmt"]
            and "16" not in stream["pix_fmt"]
        ):
            raise RuntimeError(f"Encoder lost requested precision: {stream}")
        if args.rtx_hdr and (
            stream.get("color_transfer") != "smpte2084"
            or stream.get("color_primaries") != "bt2020"
            or stream.get("color_space") != "bt2020nc"
        ):
            raise RuntimeError(f"Encoder lost HDR tags: {stream}")
    if custom:
        import os
        # Re-check source aliasing / overwrite policy at commit, not just setup.
        validate_output_path(args.input, output, args.overwrite)
        if args.overwrite:
            os.replace(encode_output, output)
        else:
            # Atomic create-if-absent on the same filesystem, no TOCTOU overwrite.
            os.link(encode_output, output)
            Path(encode_output).unlink()
    cli.log("PIPE", f"processed {count}/{count} frames", detail="finalized")
    import json

    print(
        json.dumps(
            dict(
                status="success",
                output=output,
                target_fps=target_fps,
                frames=count,
                width=target[0],
                height=target[1],
                pipeline="custom" if custom else "nvidia-vfx",
                hdr=args.rtx_hdr,
            )
        )
    )
