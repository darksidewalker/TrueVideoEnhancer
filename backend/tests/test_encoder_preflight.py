"""Encoder probes must use valid dimensions and never silently drop precision."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from render_contract import CODECS, TEN_BIT


def cli():
    spec=importlib.util.spec_from_file_location('encoder_preflight_cli',Path(__file__).resolve().parents[1]/'rve-backend.py')
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('preset',list(CODECS))
@pytest.mark.parametrize('precise',[False,True])
def test_all_presets_use_valid_dimensions_and_strict_pixel_format(preset,precise,monkeypatch):
    module=cli()
    calls=[]
    def run(cmd,**kwargs):
        calls.append((cmd,kwargs))
        return subprocess.CompletedProcess(cmd,0,'','')
    monkeypatch.setattr(module,'_run',run)
    supported=module.is_video_preset_supported(preset,set(CODECS.values()),precise=precise)
    if precise and preset not in TEN_BIT:
        assert not supported and not calls
        return
    assert supported
    command,kwargs=calls[0]
    assert 'color=c=black:s=256x256:r=24' in command
    assert command[command.index('-c:v')+1]==CODECS[preset]
    pixel=command[command.index('-pix_fmt')+1]
    expected=('yuv422p10le' if preset=='prores' else 'p010le' if precise and preset.endswith('_nvenc') else 'yuv420p10le' if precise else 'yuv420p')
    assert pixel=='+'+expected
    assert command[command.index('-vf')+1]=='scale=256:256,format='+expected
    assert kwargs['timeout']==20


@pytest.mark.parametrize('alias',list(CODECS.values()))
def test_real_codec_aliases_use_the_same_encoder(alias,monkeypatch):
    module=cli()
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd,0,'','')
    monkeypatch.setattr(module,'_run',run)
    assert module.is_video_preset_supported(alias,set(CODECS.values()))
    assert calls[0][calls[0].index('-c:v')+1]==alias


def test_unknown_or_missing_encoder_is_not_probed(monkeypatch):
    module=cli()
    monkeypatch.setattr(module,'_run',lambda *a,**kw:pytest.fail('must not probe an unavailable encoder'))
    assert not module.is_video_preset_supported('not-an-encoder',set(CODECS.values()))
    assert not module.is_video_preset_supported('av1_nvenc',{'libx264'})


def test_driver_error_is_reported(monkeypatch,capsys):
    module=cli()
    monkeypatch.setattr(module,'_run',lambda cmd,**kw:subprocess.CompletedProcess(cmd,1,'','InitializeEncoder failed: unsupported device'))
    assert not module.is_video_preset_supported('av1_nvenc',{'av1_nvenc'},precise=True)
    error=capsys.readouterr().err
    assert 'InitializeEncoder failed: unsupported device' in error
    assert 'av1_nvenc' in error and 'p010le' in error


@pytest.mark.parametrize('failure',[subprocess.TimeoutExpired(['ffmpeg'],20),FileNotFoundError('ffmpeg missing')])
def test_probe_failures_are_bounded_and_reported(failure,monkeypatch,capsys):
    module=cli()
    def run(*args,**kwargs):raise failure
    monkeypatch.setattr(module,'_run',run)
    assert not module.is_video_preset_supported('libx264',{'libx264'})
    assert 'preflight failed' in capsys.readouterr().err


@pytest.mark.skipif(os.environ.get('TVE_ENCODER_SMOKE') != '1', reason='explicit opt-in: opens NVENC sessions')
@pytest.mark.parametrize('preset,precise',[(p,b) for p in CODECS for b in (False,True) if not b or p in TEN_BIT])
def test_real_encoder_preflight_and_encoded_precision(preset,precise,monkeypatch,tmp_path):
    import json
    module=cli()
    encoders=module.available_ffmpeg_encoders()
    assert CODECS[preset] in encoders, f'{preset} missing on this test host'
    original=module._run
    commands=[]
    def run(cmd,**kwargs):
        commands.append(cmd)
        return original(cmd,**kwargs)
    monkeypatch.setattr(module,'_run',run)
    assert module.is_video_preset_supported(preset,encoders,precise=precise)
    # Reuse the exact successful probe options but publish an actual bitstream.
    output=tmp_path/'probe.mkv'
    proc=original(commands[0][:-3]+[str(output)],timeout=30)
    assert proc.returncode==0,proc.stderr
    probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries',
                          'stream=codec_name,width,height,pix_fmt,nb_read_frames,bits_per_raw_sample',
                          '-of','json',str(output)],capture_output=True,text=True,check=True,timeout=15)
    stream=json.loads(probe.stdout)['streams'][0]
    assert stream['width']==stream['height']==256
    assert stream['nb_read_frames']=='1'
    expected_codec={'hevc_nvenc':'hevc','h264_nvenc':'h264','av1_nvenc':'av1',
                    'libx265':'hevc','libx264':'h264','libsvtav1':'av1',
                    'libvpx-vp9':'vp9','libvpx':'vp8','prores_ks':'prores','ffv1':'ffv1'}
    assert stream['codec_name']==expected_codec[CODECS[preset]]
    if precise:
        assert '10' in stream['pix_fmt'] or int(stream.get('bits_per_raw_sample') or 0)>=10


@pytest.mark.skipif(os.environ.get('TVE_ENCODER_SMOKE') != '1', reason='explicit opt-in: opens NVENC sessions')
@pytest.mark.parametrize('precise',[False,True])
def test_real_webm_auto_keeps_available_av1_nvenc(precise):
    module=cli()
    for requested in ('auto','av1_nvenc'):
        assert module.normalize_encode_settings('output.webm',requested,'copy','copy','aac','',
                                                hdr=precise,precise=precise)[0]=='av1_nvenc'
