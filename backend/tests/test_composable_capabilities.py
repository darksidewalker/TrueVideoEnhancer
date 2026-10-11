import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rtx_vfx


def test_independent_effect_flags_are_discovered_without_instantiation(monkeypatch):
    class SuperRes:
        def __init__(self, *args, **kwargs):
            raise AssertionError('Capability discovery must not instantiate effects')
    quality = SimpleNamespace(HIGH=0, DENOISE_HIGH=1, DEBLUR_HIGH=2)
    sdk = SimpleNamespace(VideoSuperRes=SuperRes, effects=SimpleNamespace(QualityLevel=quality))
    monkeypatch.setattr(rtx_vfx, '_sdk', lambda: sdk)
    caps = rtx_vfx.capabilities()
    assert caps.get('upscale') is True
    assert caps.get('denoise') is True
    assert caps.get('deblur') is True
    del quality.DEBLUR_HIGH
    assert rtx_vfx.capabilities().get('deblur') is False


def test_missing_sdk_explicitly_disables_each_effect(monkeypatch):
    def unavailable():
        raise RuntimeError('test SDK missing')
    monkeypatch.setattr(rtx_vfx, '_sdk', unavailable)
    caps = rtx_vfx.capabilities()
    for feature in ('upscale', 'denoise', 'deblur'):
        assert caps.get(feature) is False
