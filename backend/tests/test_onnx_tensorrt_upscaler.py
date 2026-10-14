import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import onnx_tensorrt_upscaler as native
import upscale_inference as upscaling


@pytest.mark.parametrize('backend', ['tensorrt', 'onnxruntime', 'pytorch'])
def test_factory_routes_onnx_without_changing_other_backends(tmp_path, monkeypatch, backend):
    path = tmp_path / 'model.onnx'
    path.write_bytes(b'model')
    calls = []
    def factory(path, **kwargs):
        calls.append(kwargs)
        return 'native'
    monkeypatch.setattr(native, 'TensorRTONNXUpscaler', factory)
    monkeypatch.setattr(upscaling, 'ONNXUpscaler', lambda *a, **kw: 'ort')
    result = upscaling.create_upscaler(path, backend=backend, input_size=(64, 48), precision='float32', gpu_id=2)
    assert result == ('native' if backend == 'tensorrt' else 'ort')
    assert calls == ([dict(input_size=(64, 48), precision='float32', gpu_id=2)] if backend == 'tensorrt' else [])


def test_native_failure_is_not_a_fallback(tmp_path, monkeypatch):
    path = tmp_path / 'model.onnx'
    path.write_bytes(b'model')
    def fail(*args, **kwargs):
        raise RuntimeError('native build failed')
    monkeypatch.setattr(native, 'TensorRTONNXUpscaler', fail)
    monkeypatch.setattr(upscaling, 'ONNXUpscaler', lambda *a, **kw: pytest.fail('ORT fallback'))
    with pytest.raises(RuntimeError, match='native build failed'):
        upscaling.create_upscaler(path, backend='tensorrt')


@pytest.mark.parametrize('declared,size,expected', [
    ((1, 3, 48, 64), None, (1, 3, 48, 64)),
    ((-1, 3, -1, -1), (64, 48), (1, 3, 48, 64)),
    ((1, 3, 48, -1), (64, 48), (1, 3, 48, 64)),
])
def test_fixed_profile_respects_static_and_dynamic_dimensions(declared, size, expected):
    assert native._input_shape(declared, size) == expected


@pytest.mark.parametrize('declared,size', [
    ((1, 3, 48, 64), (64, 64)), ((1, 3, -1, -1), None),
    ((2, 3, 48, 64), (64, 48)), ((1, 4, 48, 64), (64, 48)),
    ((1, 48, 64, 3), (64, 48)), ((1, 3, -1, -1), (0, 48)),
])
def test_rejects_unsupported_input_shapes(declared, size):
    with pytest.raises(ValueError):
        native._input_shape(declared, size)


@pytest.mark.parametrize('shape', [(1, 3, 96, 128), (1, 3, 48, 64)])
def test_resolves_integer_output_scale(shape):
    assert native._output_scale(shape, (1, 3, 48, 64)) == shape[2] // 48


@pytest.mark.parametrize('shape', [(1, 3, 96, 192), (1, 3, 95, 128), (1, 3, -1, -1), (1, 4, 96, 128), (2, 3, 96, 128)])
def test_rejects_unsupported_output_shapes(shape):
    with pytest.raises(ValueError):
        native._output_scale(shape, (1, 3, 48, 64))


def test_cache_fingerprint_includes_external_weights_and_settings(tmp_path):
    onnx = importlib.import_module('onnx')
    tensor = onnx.numpy_helper.from_array(np.ones((1, 3, 1, 1), np.float32), name='weight')
    graph = onnx.helper.make_graph([], 'external', [], [], [tensor])
    path = tmp_path / 'model.onnx'
    onnx.save_model(onnx.helper.make_model(graph), path, save_as_external_data=True,
                    all_tensors_to_one_file=True, location='weights.bin', size_threshold=0)
    first = native._cache_path(path, {'shape': [1, 3, 48, 64]})
    assert first == native._cache_path(path, {'shape': [1, 3, 48, 64]})
    assert first != native._cache_path(path, {'shape': [1, 3, 64, 64]})
    (tmp_path / 'weights.bin').write_bytes(b'changed weights')
    assert first != native._cache_path(path, {'shape': [1, 3, 48, 64]})


@pytest.mark.parametrize('input_dtype,output_dtype', [('int8', 'float32'), ('float16', 'int32')])
def test_production_constructor_rejects_nonfloating_io(tmp_path, monkeypatch, input_dtype, output_dtype):
    torch = importlib.import_module('torch')
    from contextlib import nullcontext
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device', lambda *a: nullcontext())
    network = types.SimpleNamespace(num_inputs=1, num_outputs=1,
        get_input=lambda i: types.SimpleNamespace(name='input', dtype=input_dtype),
        get_output=lambda i: types.SimpleNamespace(name='output', dtype=output_dtype))
    monkeypatch.setitem(sys.modules, 'tensorrt', types.SimpleNamespace(
        float16='float16', float32='float32', Logger=lambda *a: None,
        Builder=lambda *a: types.SimpleNamespace(create_network=lambda *a: network),
        OnnxParser=lambda *a: types.SimpleNamespace(parse_from_file=lambda *a: True)))
    sys.modules['tensorrt'].Logger = type('Logger', (), {'WARNING': 1, '__init__': lambda self, *a: None})
    with pytest.raises(ValueError, match='dtype must be float16 or float32'):
        native.TensorRTONNXUpscaler(tmp_path / 'model.onnx', input_size=(64, 48), precision='float16', gpu_id=0)


def test_static_rectangular_engine_accepts_matching_full_frame():
    assert upscaling.tensorrt_compile_candidates(64, 48, requested_tile=0, fixed_input_size=(64, 48)) == [(0, (64, 48))]


def test_static_model_never_retries_mismatching_fallback_tiles(monkeypatch):
    monkeypatch.setattr(upscaling, 'static_onnx_input_size', lambda *a: (64, 64))
    attempts = []
    def factory(*args, **kwargs):
        attempts.append(kwargs['input_size'])
        raise RuntimeError('build failed')
    with pytest.raises(RuntimeError, match='build failed'):
        upscaling.create_optimized_upscaler('model.onnx', width=512, height=512,
            backend='tensorrt', requested_tile=32, fallback_tile_sizes=(16,), factory=factory)
    assert attempts == [(64, 64)]


@pytest.mark.parametrize('boundary', ['parse', 'inputs', 'build', 'deserialize', 'context', 'shape', 'output'])
def test_native_constructor_checks_failure_boundaries(tmp_path, monkeypatch, boundary):
    torch = importlib.import_module('torch')
    from contextlib import nullcontext
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device', lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda *a: types.SimpleNamespace(name='fake',major=12,minor=0))
    monkeypatch.setattr(native, '_cache_path', lambda *a: tmp_path / 'cache.engine')
    inp = types.SimpleNamespace(name='input',dtype='float16',shape=(1,3,48,64))
    out = types.SimpleNamespace(name='output',dtype='float16')
    network = types.SimpleNamespace(num_inputs=2 if boundary == 'inputs' else 1,num_outputs=1,
        get_input=lambda i: inp,get_output=lambda i: out)
    context = types.SimpleNamespace(set_input_shape=lambda *a: boundary != 'shape',
        get_tensor_shape=lambda *a: (1,3,-1,-1) if boundary == 'output' else (1,3,96,128))
    engine = types.SimpleNamespace(create_execution_context=lambda: None if boundary == 'context' else context)
    config = types.SimpleNamespace(set_memory_pool_limit=lambda *a: None)
    builder = types.SimpleNamespace(create_network=lambda *a: network,create_builder_config=lambda: config,
        build_serialized_network=lambda *a: None if boundary == 'build' else b'engine')
    fake = types.SimpleNamespace(float16='float16',float32='float32',__version__='11',
        Logger=type('Logger', (), {'WARNING':1,'__init__':lambda self,*a:None}),
        Builder=lambda *a: builder,
        OnnxParser=lambda *a: types.SimpleNamespace(parse_from_file=lambda *a: boundary != 'parse',num_errors=1,get_error=lambda i:'parse error'),
        Runtime=lambda *a: types.SimpleNamespace(deserialize_cuda_engine=lambda *a: None if boundary == 'deserialize' else engine),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE=0),BuilderFlag=types.SimpleNamespace())
    monkeypatch.setitem(sys.modules,'tensorrt',fake)
    with pytest.raises((RuntimeError,ValueError)):
        native.TensorRTONNXUpscaler(tmp_path / 'model.onnx',input_size=(64,48),precision='float16',gpu_id=0)


@pytest.mark.parametrize('boundary', ['input', 'output', 'execute'])
def test_execution_checks_binding_and_enqueue_failures(monkeypatch, boundary):
    torch = importlib.import_module('torch')
    from contextlib import nullcontext
    monkeypatch.setattr(torch.cuda,'device',lambda *a:nullcontext())
    monkeypatch.setattr(torch.cuda,'stream',lambda *a:nullcontext())
    up = object.__new__(native.TensorRTONNXUpscaler)
    up.input_shape=(1,3,2,3)
    up.output_shape=(1,3,4,6)
    up.input_name,up.output_name='input','output'
    up.dtype=up.output_tensor_dtype=torch.float32
    up.device=torch.device('cpu')
    up.stream=types.SimpleNamespace(cuda_stream=123,synchronize=lambda:None)
    up.context=types.SimpleNamespace(set_tensor_address=lambda name,pointer:name != boundary,
                                    execute_async_v3=lambda stream:boundary != 'execute')
    with pytest.raises(RuntimeError,match='TensorRT'):
        up.upscale(np.zeros((2,3,3),np.uint8))
