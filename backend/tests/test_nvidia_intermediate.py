"""Internal NVIDIA scale stays separate from final temporal/HDR/encode dimensions."""
import cv2
import numpy as np
import pytest
from test_custom_pipeline import args, cli, custom, native, source
from streaming_pipeline import validate_resource_budget


@pytest.mark.parametrize('internal,final', [(2,1),(4,1),(4,2)])
@pytest.mark.parametrize('filter_name,code', [('',cv2.INTER_AREA),('area',cv2.INTER_AREA),('lanczos',cv2.INTER_LANCZOS4),('bicubic',cv2.INTER_CUBIC)])
@pytest.mark.parametrize('dtype', [np.uint8,np.uint16])
def test_internal_resize_before_temporal_hdr(monkeypatch, internal, final, filter_name, code, dtype):
    events=[]
    size=4*final
    class Stage:
        def __init__(self, **kw): self.hdr=kw.get('hdr',False)
        def process(self, frame, out):
            if self.hdr:
                events.append('hdr')
                assert frame.shape==out.shape==(size,size,3)
                np.copyto(out,frame)
            else:
                events.append('upscale')
                assert out.shape==(4*internal,4*internal,3)
                np.copyto(out,cv2.resize(frame,(out.shape[1],out.shape[0])))
            assert out.dtype==dtype
            return out
        def close(self): pass
    class Temporal:
        def interpolate(self,a,b,timestep):
            events.append('temporal')
            assert tuple(a.shape[-2:])==(size,size)
            return a*(1-timestep)+b*timestep
        def close(self): pass
    resize=cv2.resize
    def tracked(frame,target,**kw):
        if target==(size,size):
            events.append('downsample')
            assert kw['interpolation']==code
        return resize(frame,target,**kw)
    monkeypatch.setattr(custom,'RTXVfxPipeline',Stage)
    monkeypatch.setattr(cli,'build_interpolator',lambda *a:Temporal())
    monkeypatch.setattr(cv2,'resize',tracked)
    a=args('--spatial_backend','nvidia','--rtx_upscale','VSR','--scale',str(final),
           '--rtx_upscale_intermediate',str(internal),
           '--temporal_backend','rife','--interpolate_model','mock','--rtx_hdr',
           *(['--rtx_downsample_filter',filter_name] if filter_name else []))
    custom.validate_selection(a)
    resources=[]
    p=custom.build_processor(a,cli,dict(width=4,height=4),(size,size),2,dtype,resources)
    x=np.arange(48,dtype=dtype).reshape(4,4,3)+(1000 if dtype==np.uint16 else 20)
    y=x+10
    output=p.process(x,y,0.5)
    assert events==['upscale','downsample','upscale','downsample','temporal','hdr']
    assert p.endpoint(x).shape==output.shape==(size,size,3)
    assert output.dtype==dtype
    assert events.count('upscale')==2  # Cached source endpoints, no extra pass.
    if dtype==np.uint16: assert np.any(output%257!=0)
    custom.close_resources(resources)


@pytest.mark.parametrize('flags,match', [
    (['--rtx_upscale_intermediate','2'], 'custom NVIDIA'),
    (['--spatial_backend','ai','--upscale_model','mock','--rtx_upscale_intermediate','2'], 'custom NVIDIA'),
    (['--spatial_backend','nvidia','--rtx_upscale','VSR','--scale','2','--rtx_upscale_intermediate','2'], 'exceed'),
    (['--spatial_backend','nvidia','--rtx_upscale','VSR','--scale','4','--rtx_upscale_intermediate','2'], 'exceed'),
    (['--rtx_downsample_filter','area'], 'requires intermediate'),
])
def test_bad_combinations(flags,match):
    with pytest.raises(ValueError,match=match): custom.validate_selection(args(*flags))


@pytest.mark.parametrize('value', [float('nan'),float('inf'),-1,3])
def test_invalid_scale(value):
    a=args();a.rtx_upscale_intermediate=value
    with pytest.raises(ValueError,match='must be'): custom.validate_intermediate(a)


def test_intermediate_budget_accounts_for_extra_buffers_and_cached_endpoints():
    dims=dict(source_width=128,source_height=128,output_width=128,output_height=128)
    baseline=validate_resource_budget(**dims,available_memory=1<<40)
    enabled=validate_resource_budget(**dims,intermediate_size=(512,512),available_memory=1<<40)
    assert enabled-baseline==512*512*3*3+128*128*3*2
    with pytest.raises(RuntimeError,match='host memory'):
        validate_resource_budget(**dims,intermediate_size=(512,512),available_memory=baseline)


def test_legacy_options_rejected_and_off_unchanged():
    a=args('--spatial_backend','nvidia','--rtx_upscale','VSR')
    custom.validate_selection(a)
    a.processing_mode='legacy';a.rtx_upscale_intermediate=2
    with pytest.raises(ValueError,match='custom NVIDIA'): cli.render(a)
    a.rtx_upscale_intermediate=0;a.rtx_downsample_filter='nearest'
    with pytest.raises(ValueError,match='must be'): cli.render(a)


def test_intermediate_8k_and_ram_budget_before_effects(source,tmp_path,monkeypatch):
    monkeypatch.setattr(native,'capabilities',lambda:dict(available=True,version='test',encoding=True))
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(custom,'build_processor',lambda *a:pytest.fail('SDK allocated before budget validation'))
    seen=[]
    def budget(**kw):
        seen.append(kw)
        return validate_resource_budget(**kw,available_memory=0)
    monkeypatch.setattr(native,'validate_resource_budget',budget)
    a=args('--input',str(source),'--output',str(tmp_path/'bad.mp4'),'--video_encoder_preset','libx264',
           '--spatial_backend','nvidia','--rtx_upscale','VSR','--rtx_upscale_intermediate','4')
    with pytest.raises(RuntimeError,match='host memory'): cli.render(a)
    assert seen[0]['intermediate_size']==(512,512)
    assert seen[0]['output_width']==seen[0]['output_height']==128
    assert not list(tmp_path.glob('.tve-render-*'))
    with pytest.raises(RuntimeError,match='8K'):
        validate_resource_budget(source_width=3840,source_height=2160,output_width=3840,output_height=2160,intermediate_size=(15360,8640))
