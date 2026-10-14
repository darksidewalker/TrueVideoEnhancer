#!/usr/bin/env python3
"""Explicit tiny NVIDIA SDK smoke; writes only to a new caller-selected directory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[1]
source = args.output_dir / 'source.mp4'
subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                'testsrc2=size=128x128:rate=4:duration=0.5', '-c:v', 'libx264', str(source)], check=True)
results = []
for internal, final, filter_name, precise in [(2, 1, 'area', False), (4, 2, 'lanczos', True), (4, 1, 'bicubic', True)]:
    name = f'internal{internal}-final{final}-{filter_name}'
    output = args.output_dir / (name + '.mkv')
    argv = [sys.executable, str(root / 'backend/rve-backend.py'), '--input', str(source),
            '--output', str(output), '--processing_mode', 'custom', '--spatial_backend', 'nvidia',
            '--temporal_backend', 'none', '--rtx_upscale', 'VSR', '--scale', str(final),
            '--rtx_upscale_intermediate', str(internal), '--rtx_downsample_filter', filter_name,
            '--video_encoder_preset', 'ffv1', '--rtx_image_encoding', 'rgb10a2' if precise else 'rgb8']
    run = subprocess.run(argv, capture_output=True, text=True)
    log = '\n'.join(line for line in (run.stdout + '\n' + run.stderr).splitlines() if not line.startswith('<PREVIEW>'))
    (args.output_dir / (name + '.log')).write_text(log)
    if run.returncode:
        raise RuntimeError(f'{name} failed ({run.returncode}): {log}')
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
        'stream=width,height,codec_name,pix_fmt,nb_read_frames', '-of', 'json', str(output)], text=True))['streams'][0]
    assert data['width'] == data['height'] == 128 * final, data
    assert data['nb_read_frames'] == '2' and data['codec_name'] == 'ffv1', data
    assert not precise or '10' in data['pix_fmt'], data
    assert f'internal={128*internal}x{128*internal} final={128*final}x{128*final} filter={filter_name}' in log
    assert not list(args.output_dir.glob('.tve-render-*'))
    results.append(dict(output=str(output), internal=internal, final=final, filter=filter_name, **data))
(args.output_dir / 'verification.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
