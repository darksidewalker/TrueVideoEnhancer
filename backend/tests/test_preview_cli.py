"""Real CPU CLI proof: bounded decode, media timestamps, and source preservation."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / 'rve-backend.py'


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True)


def probe(path):
    return json.loads(run('ffprobe', '-v', 'error', '-count_frames', '-show_entries',
        'format=duration:stream=codec_type,duration,start_time,nb_read_frames', '-of', 'json', str(path)).stdout)


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='FFmpeg required')
def test_real_preview_media_cap_and_short_source_no_padding(tmp_path):
    source = tmp_path / 'source.mp4'
    srt = tmp_path / 'cue.srt'
    srt.write_text('1\n00:00:00,000 --> 00:00:03,000\nBoundary cue\n')
    run('ffmpeg','-hide_banner','-loglevel','error','-y',
        '-f','lavfi','-i','testsrc2=size=128x128:rate=30:duration=3',
        '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=3',
        '-i',str(srt),'-map','0:v','-map','1:a','-map','2:s',
        '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-c:s','mov_text',str(source))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / 'full.mp4'
    output.write_bytes(b'full output sentinel')
    base = [sys.executable,str(BACKEND),'--input',str(source),'--output',str(output),
        '--pipeline','upscale','--backend','pytorch','--video_encoder_preset','libx264','--overwrite']
    result = run(*base,'--preview_seconds','1')
    assert '"frames": 30' in result.stdout
    records = [json.loads(line.removeprefix('<PROGRESS>')) for line in result.stderr.splitlines() if line.startswith('<PROGRESS>')]
    assert records[0]['stage'] == 'preparing'
    assert records[1]['stage'] == 'rendering' and records[1]['total'] == 30
    assert records[-1]['stage'] == 'finalizing' and records[-1]['done'] == records[-1]['total'] == 30
    assert any(r['done'] > 0 and r['throughput'] > 0 for r in records)
    data = probe(tmp_path / 'full-preview-1s.mp4')
    for stream in data['streams']:
        assert float(stream['start_time']) == 0
        assert abs(float(stream['duration']) - 1) <= 1/60
    assert data['streams'][0]['nb_read_frames'] == '30'
    assert abs(float(data['format']['duration']) - 1) <= 1/60
    assert output.read_bytes() == b'full output sentinel'
    # Preview longer than source must keep the real source endpoint, not pad.
    run(*base,'--preview_seconds','5')
    shorter = probe(tmp_path / 'full-preview-5s.mp4')
    assert shorter['streams'][0]['nb_read_frames'] == '90'
    assert abs(float(shorter['format']['duration']) - 3) <= 1/60
    # Default zero is full render and has no preview suffix.
    run(*base)
    assert probe(output)['streams'][0]['nb_read_frames'] == '90'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
