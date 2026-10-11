"""Compositional CPU mocks plus real FFmpeg integration; no model/GPU downloads."""
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import custom_pipeline as custom
import nvidia_vfx_pipeline as native
from streaming_pipeline import stream_frames
from upscale_inference import SafetensorsUpscaler, ONNXUpscaler

spec = importlib.util.spec_from_file_location('custom_test_cli', BACKEND / 'rve-backend.py')
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


def args(*flags):
    return cli.build_parser().parse_args(['--processing_mode', 'custom', *flags])


@pytest.mark.parametrize('spatial,temporal,hdr,restore', [
    (s, t, h, r) for s in ('none', 'ai', 'nvidia')
    for t in ('none', 'rife', 'nvidia') for h in (False, True) for r in (False, True)
])
def test_stage_composition_and_bounded_endpoints(monkeypatch, spatial, temporal, hdr, restore):
    events, created = [], []
    spatial_event, temporal_event = 'spatial-' + spatial, 'temporal-' + temporal

    class Stage:
        def __init__(self, **kw):
            self.names = (['denoise', 'deblur'] if kw.get('denoise') else [])
            self.names += [spatial_event] if kw.get('upscale_mode', 'Off') != 'Off' else []
            self.names += ['hdr'] if kw.get('hdr') else []
            self.closed = 0
            created.append(self)
        def process(self, frame, out):
            events.extend(self.names)
            assert frame.dtype == out.dtype == np.uint16
            import cv2
            np.copyto(out, cv2.resize(frame, (out.shape[1], out.shape[0])))
            return out
        def close(self):
            self.closed += 1

    class AI:
        scale = 2
        technique = 'CPU mock AI'
        def __init__(self):
            self.closed = 0
            created.append(self)
        def upscale(self, frame):
            events.append(spatial_event)
            return np.repeat(np.repeat(frame, 2, 0), 2, 1)
        def close(self):
            self.closed += 1

    class Temporal:
        def __init__(self, **kw):
            self.closed = 0
            created.append(self)
        def interpolate(self, a, b, timestep):
            events.append(temporal_event)
            assert tuple(a.shape[-2:]) == (4, 4)
            return a * (1 - timestep) + b * timestep
        def close(self):
            self.closed += 1

    monkeypatch.setattr(custom, 'RTXVfxPipeline', Stage)
    monkeypatch.setattr(custom, 'RTXFrameInterpolator', Temporal)
    monkeypatch.setattr(cli, 'resolve_model_path', lambda p: p)
    monkeypatch.setattr(cli, 'create_optimized_upscaler', lambda *a, **kw: (AI(), 0, (2, 2), []))
    monkeypatch.setattr(cli, 'build_interpolator', lambda *a: Temporal())
    a = args('--spatial_backend', spatial, '--temporal_backend', temporal,
             '--upscale_model', 'mock.safetensors', '--interpolate_model', 'mock-rife.safetensors',
             '--rtx_image_encoding', 'rgb10a2')
    a.rtx_upscale = 'VSR' if spatial == 'nvidia' else 'Off'
    a.rtx_hdr, a.denoise, a.deblur = hdr, restore, restore
    info = dict(width=2 if spatial != 'none' else 4, height=2 if spatial != 'none' else 4, fps=24)
    resources = []
    p = custom.build_processor(a, cli, info, (4, 4), 2.5 if temporal != 'none' else 1, np.uint16, resources)
    source = [np.full((info['height'], info['width'], 3), value, np.uint16) for value in (1000, 1100, 1200)]
    writer = SimpleNamespace(put=lambda frame: outputs.append(frame.copy()))
    outputs = []
    count = stream_frames(SimpleNamespace(frames=lambda: iter(source)), writer, p,
                          source_fps=24, target_fps=60 if temporal != 'none' else 24,
                          preserve_duration=True)
    assert count == (8 if temporal != 'none' else 3)
    assert all(f.dtype == np.uint16 and f.shape == (4, 4, 3) for f in outputs)
    assert len(p.cache) == 2
    if restore:
        assert events.count('denoise') == events.count('deblur') == 3
        assert events.index('denoise') < events.index('deblur')
    if spatial != 'none':
        assert events.count(spatial_event) == 3
        if restore:
            assert events.index('deblur') < events.index(spatial_event)
    if temporal != 'none' and spatial != 'none':
        assert events.index(spatial_event) < events.index(temporal_event)
    if hdr:
        assert events.count('hdr') == count
        # Fractional outputs execute temporal before their own HDR pass.
        for index, name in enumerate(events):
            if name == temporal_event and temporal != 'none':
                assert events[index + 1] == 'hdr'
    custom.close_resources(resources)
    custom.close_resources(resources)
    assert all(obj.closed == 1 for obj in created)


@pytest.mark.parametrize('flags,match', [
    (['--scale', '2'], 'scale requires'),
    (['--target_fps', '60'], 'FPS change requires'),
    (['--target_fps', '12', '--temporal_backend', 'nvidia'], 'reduce'),
    (['--spatial_backend', 'ai'], 'upscale_model'),
    (['--temporal_backend', 'rife'], 'interpolate_model'),
    (['--spatial_backend', 'nvidia'], 'explicit'),
    (['--rtx_upscale', 'VSR'], 'conflicts'),
    (['--rtx_frame_generation'], 'conflicts'),
    (['--target_fps', 'nan'], 'finite'),
    (['--hdr_mode'], 'does not support'),
])
def test_invalid_selection_fails_without_model_load(flags, match):
    with pytest.raises(ValueError, match=match):
        custom.validate_selection(args(*flags), dict(fps=24))


def test_explicit_no_features_source_cadence_is_valid():
    custom.validate_selection(args('--target_fps', '24'), dict(fps=24))


@pytest.mark.parametrize('precision', [torch.float16, torch.float32])
def test_safetensors_normalizes_16bit_before_half_and_never_quantizes_to_8bit(precision):
    obj = SafetensorsUpscaler.__new__(SafetensorsUpscaler)
    obj.device, obj.dtype, obj.output_dtype = torch.device('cpu'), precision, np.uint16
    obj.model, obj.batch_size = torch.nn.Identity(), 1
    frame = np.arange(300, dtype=np.uint16).reshape(10, 10, 3) + 30000
    tensor = obj.frame_to_tensor(frame)
    assert torch.isfinite(tensor).all() and tensor.max() > 0.4
    output = obj.tensor_to_frame(tensor)
    batch = obj.upscale_batch([frame])[0]
    assert output.dtype == batch.dtype == np.uint16
    tolerance = 20 if precision == torch.float16 else 1
    assert np.abs(output.astype(int) - frame).max() <= tolerance
    assert np.abs(batch.astype(int) - frame).max() <= tolerance
    # Arbitrary 16-bit samples survive, not just 8-bit multiples of 257.
    assert np.any(output % 257 != 0)


def test_onnx_high_precision_normalization():
    obj = ONNXUpscaler.__new__(ONNXUpscaler)
    obj.input_name, obj.input_type, obj.output_dtype = 'x', 'tensor(float16)', np.uint16
    obj.session = SimpleNamespace(run=lambda names, inputs: [inputs['x']])
    frame = np.full((2, 2, 3), 40000, np.uint16)
    output = obj.upscale(frame)
    assert output.dtype == np.uint16
    assert abs(int(output[0, 0, 0]) - 40000) < 20


def run(*argv):
    return subprocess.run(argv, capture_output=True, text=True, check=True)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'source.mp4'
    run('ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
        'testsrc2=size=128x128:rate=24:duration=2', '-c:v', 'libx264', str(path))
    return path


def test_real_custom_transcode_preview_totals_and_atomic_failure(source, tmp_path, monkeypatch, capsys):
    output = tmp_path / 'full.mp4'
    output.write_bytes(b'existing full output')
    a = args('--input', str(source), '--output', str(output), '--video_encoder_preset', 'libx264',
             '--preview_seconds', '1', '--overwrite')
    cli.render(a)
    preview = tmp_path / 'full-preview-1s.mp4'
    probe = json.loads(run('ffprobe', '-v', 'error', '-count_frames', '-show_entries',
                          'stream=nb_read_frames,avg_frame_rate', '-of', 'json', str(preview)).stdout)
    assert probe['streams'][0]['nb_read_frames'] == '24'
    records = [json.loads(line[len('<PROGRESS>'):]) for line in capsys.readouterr().err.splitlines() if line.startswith('<PROGRESS>')]
    assert records[-1]['done'] == records[-1]['total'] == 24
    assert output.read_bytes() == b'existing full output'
    assert not list(tmp_path.glob('.tve-render-*'))
    # Fail after actual encoder startup and the first frame; final output is untouched.
    def fail(reader, writer, processor, **kw):
        frame = next(reader.frames())
        writer.put(processor.process(frame, frame, 0))
        raise RuntimeError('injected processing failure')
    monkeypatch.setattr(native, 'stream_frames', fail)
    a = args('--input', str(source), '--output', str(output), '--video_encoder_preset', 'libx264', '--overwrite')
    with pytest.raises(RuntimeError, match='injected'):
        cli.render(a)
    assert output.read_bytes() == b'existing full output'
    assert not list(tmp_path.glob('.tve-render-*'))


def test_invalid_hdr_codec_rejected_before_loading_models(source, tmp_path, monkeypatch):
    monkeypatch.setattr(custom, 'build_processor', lambda *a: pytest.fail('loaded models before validation'))
    a = args('--input', str(source), '--output', str(tmp_path / 'bad.mp4'),
             '--video_encoder_preset', 'libx264', '--rtx_hdr')
    with pytest.raises(ValueError, match='incompatible'):
        cli.render(a)
    assert not list(tmp_path.glob('.tve-render-*'))


def test_real_cli_interrupt_cleans_staging_and_preserves_previous_output(tmp_path):
    source, output = tmp_path / 'long.mp4', tmp_path / 'existing.mp4'
    run('ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
        'testsrc2=size=128x128:rate=24:duration=20', '-c:v', 'libx264', str(source))
    output.write_bytes(b'previous completed output')
    process = subprocess.Popen([sys.executable, str(BACKEND / 'rve-backend.py'), '--input', str(source),
        '--output', str(output), '--processing_mode', 'custom', '--video_encoder_preset', 'libx264',
        '--overwrite'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for line in process.stderr:
            if line.startswith('<PREVIEW>'):
                process.send_signal(signal.SIGINT)
                break
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode != 0, (stdout, stderr)
        assert output.read_bytes() == b'previous completed output'
        assert not list(tmp_path.glob('.tve-render-*'))
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_real_composed_mock_models_high_precision_hdr_encode(source, tmp_path, monkeypatch):
    """Actual 10-bit FFmpeg encode; model/native effects intentionally CPU mocks."""
    class AI:
        scale = 2
        technique = 'CPU test mock'
        def upscale(self, frame):
            return np.repeat(np.repeat(frame, 2, 0), 2, 1)
        def close(self): pass
    class RIFE:
        def interpolate(self, a, b, timestep): return a * (1-timestep) + b * timestep
        def close(self): pass
    class HDR:
        def __init__(self, **kw): assert kw['hdr']
        def process(self, frame, output): np.copyto(output, frame); return output
        def close(self): pass
    monkeypatch.setattr(cli, 'resolve_model_path', lambda path: path)
    monkeypatch.setattr(cli, 'create_optimized_upscaler', lambda *a, **kw: (AI(), 0, (128,128), []))
    monkeypatch.setattr(cli, 'build_interpolator', lambda *a: RIFE())
    monkeypatch.setattr(custom, 'RTXVfxPipeline', HDR)
    monkeypatch.setattr(native, 'capabilities', lambda: dict(available=True, hdr=True, encoding=True, version='CPU mocks'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    output = tmp_path / 'composed.mkv'
    a = args('--input', str(source), '--output', str(output), '--video_encoder_preset', 'ffv1',
             '--spatial_backend', 'ai', '--temporal_backend', 'rife', '--upscale_model', 'mock',
             '--interpolate_model', 'mock', '--scale', '2', '--target_fps', '60', '--rtx_hdr', '--preview_seconds', '1')
    cli.render(a)
    data = json.loads(run('ffprobe', '-v', 'error', '-count_frames', '-show_entries',
        'stream=nb_read_frames,width,height,pix_fmt,color_transfer,color_primaries,color_space', '-of', 'json',
        str(tmp_path / 'composed-preview-1s.mkv')).stdout)['streams'][0]
    assert data['nb_read_frames'] == '60' and data['width'] == data['height'] == 256
    assert '10' in data['pix_fmt']
    assert (data['color_transfer'], data['color_primaries'], data['color_space']) == ('smpte2084', 'bt2020', 'bt2020nc')


@pytest.mark.skipif(os.environ.get('TVE_CUSTOM_HDR_SMOKE') != '1', reason='explicit one-feature GPU opt-in')
def test_real_gpu_custom_hdr_only(tmp_path):
    source, output = tmp_path / 'sdr.mp4', tmp_path / 'hdr.mkv'
    run('ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
        'testsrc2=size=128x128:rate=2:duration=1', '-c:v', 'libx264', str(source))
    result = run(sys.executable, str(BACKEND / 'rve-backend.py'), '--input', str(source),
        '--output', str(output), '--processing_mode', 'custom', '--spatial_backend', 'none',
        '--temporal_backend', 'none', '--rtx_hdr', '--video_encoder_preset', 'ffv1')
    assert json.loads(result.stdout.splitlines()[-1])['frames'] == 2
    data = json.loads(run('ffprobe', '-v', 'error', '-count_frames', '-show_entries',
        'stream=nb_read_frames,pix_fmt,color_transfer,color_primaries,color_space', '-of', 'json', str(output)).stdout)['streams'][0]
    assert data['nb_read_frames'] == '2' and '10' in data['pix_fmt']
    assert (data['color_transfer'], data['color_primaries'], data['color_space']) == ('smpte2084', 'bt2020', 'bt2020nc')
    raw = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(output),
                          '-frames:v', '1', '-pix_fmt', 'bgr48le', '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
    assert np.unique(np.frombuffer(raw, dtype='<u2')).size > 256
