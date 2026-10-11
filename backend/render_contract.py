"""Pure render contracts shared by validation and capability discovery (no GPU)."""
import math
import re
from pathlib import Path

CODECS = {
    'av1_nvenc': 'av1_nvenc', 'x265_nvenc': 'hevc_nvenc',
    'x264_nvenc': 'h264_nvenc', 'libx265': 'libx265', 'libx264': 'libx264',
    'av1': 'libsvtav1', 'vp9': 'libvpx-vp9', 'vp8': 'libvpx',
    'prores': 'prores_ks', 'ffv1': 'ffv1',
}
ALIASES = {v: k for k, v in CODECS.items()}
MATRIX = {
    'mp4': ['av1_nvenc', 'x265_nvenc', 'x264_nvenc', 'libx265', 'libx264', 'av1'],
    'mkv': ['av1_nvenc', 'x265_nvenc', 'x264_nvenc', 'libx265', 'libx264', 'av1', 'vp9', 'vp8', 'prores', 'ffv1'],
    'webm': ['av1_nvenc', 'vp9', 'vp8', 'av1'],
    'mov': ['x265_nvenc', 'x264_nvenc', 'libx265', 'libx264', 'prores'],
    'ts': ['x265_nvenc', 'x264_nvenc', 'libx265', 'libx264'],
    # FFmpeg infers the restricted ipod muxer for .m4v, not generic MP4.
    'm4v': ['x264_nvenc', 'libx264'],
    'avi': ['x264_nvenc', 'libx264', 'ffv1'],
    'flv': ['x264_nvenc', 'libx264'],
}
TEN_BIT = {'av1_nvenc', 'x265_nvenc', 'libx265', 'av1', 'vp9', 'prores', 'ffv1'}
HDR = TEN_BIT  # ProRes MOV/MKV and FFV1 MKV retain 10-bit samples and HDR tags.


def canonical_preset(value: str) -> str:
    return ALIASES.get(value, value)


def parse_video_encoders(text):
    result = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0].startswith('V') and parts[1] != '=':
            result.add(parts[1])
    return result


def render_capabilities(encoders=None):
    # CLI fast path deliberately does not import inference modules or Torch.
    from rtx_vfx import capabilities
    if encoders is None:
        import shutil
        import subprocess
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg:
            proc = subprocess.run([ffmpeg, '-hide_banner', '-encoders'], capture_output=True, text=True, timeout=15)
            encoders = parse_video_encoders(proc.stdout) if proc.returncode == 0 else set()
        else:
            encoders = set()
    ordinary, hdr, precise = codec_matrices(encoders)
    return dict(vfx=capabilities(), encoders=sorted(encoders),
                processing_modes=['legacy', 'custom'],
                spatial_backends=['none', 'ai', 'nvidia'],
                temporal_backends=['none', 'rife', 'nvidia'],
                processing_order=['denoise', 'deblur', 'spatial', 'temporal', 'hdr', 'encode'],
                model_working_space='normalized-floating-point-sdr',
                pipelines=['upscale', 'nvidia-vfx', 'restore+upscale'],
                containers=list(MATRIX), codec_matrix=ordinary,
                hdr_codec_matrix=hdr, precise_codec_matrix=precise)


def codec_candidates(container, *, hdr=False, precise=False):
    candidates = MATRIX.get(container, [])
    if hdr or precise:
        if container not in {'mp4', 'mkv', 'webm', 'mov', 'ts', 'm4v'}:
            return []
        candidates = [p for p in candidates if p in (HDR if hdr else TEN_BIT)]
    return candidates


def codec_matrices(encoders):
    def matrix(**mode):
        result = {}
        for container in MATRIX:
            supported = [p for p in codec_candidates(container, **mode) if CODECS[p] in encoders]
            result[container] = ['auto', *supported] if supported else []
        return result
    return matrix(), matrix(hdr=True), matrix(precise=True)


def validate_preview_seconds(seconds):
    if not math.isfinite(seconds) or (seconds != 0 and not 1 <= seconds <= 30):
        raise ValueError('preview_seconds must be 0 (full render) or finite in range 1..30')
    return seconds


def validate_output_path(source, output, overwrite):
    import os
    if (Path(source).resolve() == Path(output).resolve()
            or (Path(source).exists() and Path(output).exists() and os.path.samefile(source, output))):
        raise ValueError('output must not overwrite the source')
    if Path(output).exists() and not overwrite:
        raise ValueError(f'output exists (use --overwrite): {output}')


def preview_output_path(output, seconds):
    validate_preview_seconds(seconds)
    if not seconds:
        return output
    path = Path(output)
    seconds_tag = str(float(seconds))
    if seconds_tag.endswith('.0'):
        seconds_tag = seconds_tag[:-2]
    marker = f'-preview-{seconds_tag}s'
    # Go and Python both protect raw API/CLI output names: make this idempotent.
    stem = re.sub(r'-preview-[0-9]+(?:\.[0-9]+)?s$', '', path.stem)
    return str(path.with_name(stem + marker + path.suffix))
