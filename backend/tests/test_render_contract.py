import importlib.util
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streaming_pipeline import decode_command, encode_command


def cli():
    spec = importlib.util.spec_from_file_location('preview_cli', Path(__file__).resolve().parents[1] / 'rve-backend.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def test_decode_preview_before_inference_and_full_unchanged():
    full = decode_command('source.mp4', width=128, height=128)
    preview = decode_command('source.mp4', width=128, height=128, duration=1)
    assert '-t' not in full and '-ss' not in preview
    assert preview[preview.index('-t') + 1] == '1'


def test_mux_caps_source_audio_and_output_without_shortest():
    args = dict(width=128, height=128, fps=60, video_args=['-c:v', 'libx264'], audio_args=['-c:a', 'copy'], subtitle_args=['-c:s', 'copy'])
    full = encode_command('source.mp4', 'out.mp4', **args)
    cmd = encode_command('source.mp4', 'out.mp4', duration=1, **args)
    assert '-t' not in full and cmd.count('-t') == 2
    assert '-shortest' not in cmd and '-ss' not in cmd
    assert cmd[cmd.index('-map') + 1] == '0:v:0'


def test_preview_path_and_validation():
    from render_contract import preview_output_path, validate_preview_seconds
    assert preview_output_path('out.mp4', 1) == 'out-preview-1s.mp4'
    assert preview_output_path('out-preview-1s.mp4', 1) == 'out-preview-1s.mp4'
    assert preview_output_path('out.mp4', 0) == 'out.mp4'
    assert preview_output_path('out.mp4', 1.123456789) == 'out-preview-1.123456789s.mp4'
    for value in (-1, 0.5, 31, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            validate_preview_seconds(value)


def test_capability_matrices_filter_encoders():
    from render_contract import codec_matrices
    ordinary, hdr, precise = codec_matrices({'libx264', 'libx265', 'prores_ks', 'ffv1'})
    assert ordinary['webm'] == []
    assert hdr['ts'] == ['auto', 'libx265']
    assert 'prores' not in precise['mp4'] and 'ffv1' not in precise['mov']
    assert hdr['avi'] == [] and precise['flv'] == []
    assert 'prores' in hdr['mov'] and 'ffv1' in hdr['mkv']
    assert hdr['m4v'] == [] and precise['m4v'] == []
    assert 'av1_nvenc' not in ordinary['mov'] and 'av1' not in hdr['mov']
    assert 'x265_nvenc' not in precise['mkv']


@pytest.mark.parametrize('container,encoder,hdr,precise', [('ts','av1_nvenc',True,True), ('mp4','prores',False,True), ('mov','ffv1',False,True), ('avi','libx265',False,True), ('flv','auto',True,True), ('mp4','libx264',True,True), ('mov','av1_nvenc',True,True), ('m4v','libx265',True,True)])
def test_invalid_combinations_fail_before_effect(container, encoder, hdr, precise):
    m = cli()
    m.available_ffmpeg_encoders = lambda: {'libx265','libx264','av1_nvenc','prores_ks','ffv1'}
    m.is_video_preset_supported = lambda *a: (_ for _ in ()).throw(AssertionError('must validate before hardware probe'))
    with pytest.raises(ValueError):
        m.normalize_encode_settings('out.'+container, encoder, 'copy', 'copy', '', '', hdr=hdr, precise=precise)


def test_precise_explicit_unavailable_does_not_fallback():
    m = cli()
    m.available_ffmpeg_encoders = lambda: {'libx265'}
    with pytest.raises(ValueError):
        m.normalize_encode_settings('out.mp4','x265_nvenc','copy','copy','','',precise=True)


@pytest.mark.parametrize('pipeline', ['upscale', 'restore+upscale', 'nvidia-vfx'])
def test_render_rejects_codec_before_model_or_effect(monkeypatch, pipeline):
    m = cli()
    args = m.build_parser().parse_args(['--input','source.mp4','--output','out.mp4',
        '--pipeline',pipeline,'--video_encoder_preset','prores','--denoise'])
    m.probe_video = lambda _: dict(width=128,height=128,fps=30,duration=3,audio_codec='',subtitle_codec='')
    def forbidden(*args, **kwargs):
        raise AssertionError('model/effect work before codec validation')
    monkeypatch.setattr(m, 'create_optimized_upscaler', forbidden)
    import nvidia_vfx_pipeline, rtx_vfx
    monkeypatch.setattr(nvidia_vfx_pipeline, 'RTXVfxPipeline', forbidden)
    monkeypatch.setattr(nvidia_vfx_pipeline, 'capabilities', forbidden)
    monkeypatch.setattr(rtx_vfx, 'RTXVfxPipeline', forbidden)
    with pytest.raises(ValueError, match='incompatible'):
        m.render(args)


def test_precise_auto_container_safe_and_ten_bit_preflight():
    m = cli()
    m.available_ffmpeg_encoders = lambda: {'libx265', 'av1_nvenc'}
    probes = []
    def supported(preset, encoders, *, precise=False):
        probes.append((preset, precise))
        return True
    m.is_video_preset_supported = supported
    assert m.normalize_encode_settings('out.ts','auto','copy','copy','','',hdr=True,precise=True)[0] == 'libx265'
    assert probes == [('libx265', True)]


def test_codec_tag_resolves_with_preview_suffix():
    from streaming_pipeline import resolve_output_codec_path
    assert resolve_output_codec_path('out-[auto]-preview-1s.mp4', 'auto', 'av1_nvenc') == 'out-[av1_nvenc]-preview-1s.mp4'


def test_source_hardlink_cannot_be_overwritten(tmp_path):
    import os
    from render_contract import validate_output_path
    source, output = tmp_path / 'source.mp4', tmp_path / 'output.mp4'
    source.write_bytes(b'original')
    os.link(source, output)
    with pytest.raises(ValueError, match='source'):
        validate_output_path(source, output, True)


def test_render_capabilities_fast_path_never_initializes_cuda(monkeypatch, capsys):
    import json
    import runpy
    import torch
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('capability reporting touched CUDA')
    for name in ('is_available', 'device_count', '_lazy_init'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(sys, 'argv', ['rve-backend.py', '--render-capabilities'])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / 'rve-backend.py'), run_name='__main__')
    assert exc.value.code == 0
    output = capsys.readouterr()
    data = json.loads(output.out)
    assert set(data) == {'vfx','encoders','pipelines','containers','codec_matrix','hdr_codec_matrix','precise_codec_matrix',
                         'processing_modes', 'spatial_backends', 'temporal_backends', 'processing_order', 'model_working_space'}
    assert data['processing_modes'] == ['legacy', 'custom']
    assert data['spatial_backends'] == ['none', 'ai', 'nvidia']
    assert data['temporal_backends'] == ['none', 'rife', 'nvidia']
    assert data['processing_order'] == ['denoise', 'deblur', 'spatial', 'temporal', 'hdr', 'encode']
    assert '=' not in data['encoders'] and output.err == ''
    assert not calls
