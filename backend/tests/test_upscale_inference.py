import importlib
from pathlib import Path
import sys
import types

import cv2
import numpy as np
import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture
def module():
    return importlib.import_module("upscale_inference")


class FakeUpscaler:
    scale = 2
    technique = "fake model"

    def upscale(self, frame):
        return cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

    def close(self):
        pass


class FakeBatchUpscaler(FakeUpscaler):
    batch_size = 4

    def __init__(self):
        self.batch_calls = []

    def upscale_batch(self, frames):
        self.batch_calls.append(len(frames))
        return [self.upscale(frame) for frame in frames]


def test_rejects_legacy_pth_models(module, tmp_path):
    model = tmp_path / "legacy.pth"
    model.write_bytes(b"legacy")

    with pytest.raises(ValueError, match="safetensors.*onnx"):
        module.create_upscaler(model, backend="pytorch")


def test_tensorrt_compile_uses_model_dtype_without_enabled_precisions(module, monkeypatch):
    torch = importlib.import_module("torch")
    calls = []
    monkeypatch.setitem(sys.modules, "torch_tensorrt", types.SimpleNamespace(
        compile=lambda model, **kwargs: calls.append((model, kwargs)) or model,
    ))
    upscaler = object.__new__(module.SafetensorsUpscaler)
    upscaler.batch_size = 1
    upscaler.device = torch.device("cpu")
    upscaler.dtype = torch.float16
    upscaler.model = torch.nn.Identity()

    upscaler._compile_tensorrt((16, 16))

    assert calls[0][1]["inputs"][0].dtype == torch.float16
    assert "enabled_precisions" not in calls[0][1]


def test_process_frames_runs_model_and_returns_persistent_output(module, tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 0] = (10, 20, 30)
    cv2.imwrite(str(source_dir / "00000001.png"), frame)

    result = module.process_frames(
        sorted(source_dir.glob("*.png")),
        output_dir,
        FakeUpscaler(),
        target_scale=2,
    )

    written = cv2.imread(str(result.frames[0]))
    assert result.width == 6
    assert result.height == 4
    assert result.technique == "fake model"
    assert written.shape[:2] == (4, 6)
    assert tuple(written[0, 0]) == (10, 20, 30)


def test_process_frames_resizes_native_model_output_to_requested_scale(module, tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    frame = np.full((4, 5, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(source_dir / "00000001.png"), frame)

    result = module.process_frames(
        sorted(source_dir.glob("*.png")),
        output_dir,
        FakeUpscaler(),
        target_scale=1,
    )

    written = cv2.imread(str(result.frames[0]))
    assert written.shape[:2] == (4, 5)
    assert result.width == 5
    assert result.height == 4


def test_tiled_upscale_covers_frame_without_changing_dimensions(module):
    frame = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)

    output = module.upscale_frame_tiled(frame, FakeUpscaler(), tile_size=4, overlap=1)

    expected = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    assert output.shape == expected.shape
    assert np.array_equal(output, expected)


def test_tiled_upscale_batches_tiles_for_gpu_throughput(module):
    frame = np.arange(17 * 17 * 3, dtype=np.uint8).reshape(17, 17, 3)
    upscaler = FakeBatchUpscaler()

    output = module.upscale_frame_tiled(frame, upscaler, tile_size=4, overlap=1)

    expected = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(output, expected)
    assert upscaler.batch_calls == [4, 4, 4, 4, 4, 4, 1]


def test_auto_tile_avoids_full_frame_tensorrt_compile_for_large_input(module):
    assert module.choose_tile_size(704, 896, requested=0, backend="tensorrt") == 0
    assert module.choose_tile_size(32, 24, requested=0, backend="tensorrt") == 0
    assert module.choose_tile_size(704, 896, requested=256, backend="tensorrt") == 256


def test_tensorrt_compile_candidates_try_full_frame_then_safe_tiles(module):
    assert module.tensorrt_compile_candidates(1088, 1920, requested_tile=0) == [
        (0, (1088, 1920)),
        (128, (160, 160)),
    ]


def test_explicit_tile_does_not_attempt_full_frame(module):
    assert module.tensorrt_compile_candidates(1088, 1920, requested_tile=256) == [
        (256, (288, 288)),
    ]


def test_static_onnx_shape_uses_only_matching_tiles(module):
    assert module.tensorrt_compile_candidates(
        1088, 1920, requested_tile=0, fixed_input_size=(256, 256),
    ) == [(224, (256, 256))]


def test_full_frame_engine_uses_batch_one_while_tiles_use_batch_eight(module):
    assert module.tensorrt_batch_size((1088, 1920)) == 1
    assert module.tensorrt_batch_size((160, 160)) == 8


def test_full_frame_engine_gets_larger_tactic_workspace(module):
    assert module.tensorrt_workspace_size((1088, 1920)) == 8 << 30
    assert module.tensorrt_workspace_size((160, 160)) == 1 << 30


def test_tensorrt_engine_cache_is_disabled_for_safetensors_upscalers(module, tmp_path, monkeypatch):
    def compiler(*, cache_built_engines, reuse_cached_engines, engine_cache_dir,
                 engine_cache_size, immutable_weights):
        pass

    monkeypatch.delenv("RVE_UPSCALER_TRT_ENGINE_CACHE", raising=False)
    options = module.tensorrt_engine_cache_kwargs(
        tmp_path / "model.safetensors",
        compiler=compiler,
    )

    assert options == {}
    assert not (tmp_path / ".tensorrt-engine-cache").exists()


def test_tensorrt_engine_cache_uses_durable_model_adjacent_directory(module, tmp_path, monkeypatch):
    def compiler(*, cache_built_engines, reuse_cached_engines, engine_cache_dir, engine_cache_size, immutable_weights):
        pass

    monkeypatch.setenv("RVE_UPSCALER_TRT_ENGINE_CACHE", "1")
    options = module.tensorrt_engine_cache_kwargs(tmp_path / "model.safetensors", compiler=compiler)

    assert options == {
        "cache_built_engines": True,
        "reuse_cached_engines": True,
        "engine_cache_dir": str(tmp_path / ".tensorrt-engine-cache"),
        "engine_cache_size": 5 << 30,
        "immutable_weights": False,
    }
    assert (tmp_path / ".tensorrt-engine-cache").is_dir()


def test_tensorrt_engine_cache_stays_disabled_for_unsupported_compiler(module, tmp_path):
    def compiler(*, engine_cache_dir):
        pass

    options = module.tensorrt_engine_cache_kwargs(
        tmp_path / "model.safetensors",
        compiler=compiler,
    )

    assert options == {}
    assert not (tmp_path / ".tensorrt-engine-cache").exists()

def test_create_optimized_upscaler_falls_back_from_full_frame_to_tiles(module, tmp_path):
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    attempts = []

    def factory(path, **kwargs):
        attempts.append(kwargs["input_size"])
        if kwargs["input_size"] == (1088, 1920):
            raise RuntimeError("TensorRT out of memory")
        return FakeBatchUpscaler()

    upscaler, tile_size, compile_size, failures = module.create_optimized_upscaler(
        model, width=1088, height=1920, backend="tensorrt", requested_tile=0,
        factory=factory,
    )

    assert isinstance(upscaler, FakeBatchUpscaler)
    assert attempts == [(1088, 1920), (160, 160)]
    assert tile_size == 128
    assert compile_size == (160, 160)
    assert failures == [((1088, 1920), "TensorRT out of memory")]


def test_create_optimized_upscaler_falls_back_to_smaller_adaptive_tiles(module, tmp_path):
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model")
    attempts = []

    def factory(path, **kwargs):
        attempts.append(kwargs["input_size"])
        if kwargs["input_size"] == (544, 544):
            raise RuntimeError("TensorRT out of memory")
        return FakeBatchUpscaler()

    _, tile_size, compile_size, failures = module.create_optimized_upscaler(
        model, width=1088, height=1920, backend="tensorrt", requested_tile=512,
        fallback_tile_sizes=(256, 128), factory=factory,
    )

    assert attempts == [(544, 544), (288, 288)]
    assert tile_size == 256
    assert compile_size == (288, 288)
    assert failures == [((544, 544), "TensorRT out of memory")]


def test_static_onnx_input_size_detects_fixed_nchw_shape(module, tmp_path):
    onnx = importlib.import_module("onnx")
    helper = onnx.helper
    TensorProto = onnx.TensorProto

    path = tmp_path / "fixed.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "fixed",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT16, [1, 3, 256, 256])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 3, 256, 256])],
    )
    onnx.save(helper.make_model(graph), path)

    assert module.static_onnx_input_size(path) == (256, 256)
    assert module.static_onnx_input_size(tmp_path / "model.safetensors") is None


def test_process_frames_reports_actual_upscale_progress(module, tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for index in range(2):
        cv2.imwrite(str(source_dir / f"{index + 1:08d}.png"), np.full((2, 2, 3), index, np.uint8))
    progress = []

    module.process_frames(
        sorted(source_dir.glob("*.png")), tmp_path / "output", FakeUpscaler(),
        target_scale=2, progress_cb=lambda done, total: progress.append((done, total)),
    )

    assert progress == [(1, 2), (2, 2)]


def test_process_frames_runs_a_2x_model_twice_for_a_4x_target(module, tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    cv2.imwrite(str(source_dir / "00000001.png"), np.full((2, 3, 3), 127, dtype=np.uint8))

    result = module.process_frames(
        sorted(source_dir.glob("*.png")), output_dir, FakeUpscaler(), target_scale=4,
    )

    written = cv2.imread(str(result.frames[0]))
    assert written.shape[:2] == (8, 12)
    assert (result.width, result.height) == (12, 8)


def test_process_frames_uses_fast_lossless_temp_png_writes(module, tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), np.zeros((2, 2, 3), np.uint8))
    real_imwrite = cv2.imwrite
    options = []

    def recording_imwrite(path, frame, params=None):
        options.append(params)
        return real_imwrite(path, frame, params or [])

    monkeypatch.setattr(module.cv2, "imwrite", recording_imwrite)
    module.process_frames([source], tmp_path / "output", FakeUpscaler(), target_scale=2)

    assert options == [[cv2.IMWRITE_PNG_COMPRESSION, 0]]


def test_tensorrt_engine_cache_disabled_by_default(module, tmp_path, monkeypatch):
    def compiler(*, cache_built_engines, reuse_cached_engines, engine_cache_dir, engine_cache_size, immutable_weights):
        pass

    monkeypatch.delenv("RVE_UPSCALER_TRT_ENGINE_CACHE", raising=False)
    options = module.tensorrt_engine_cache_kwargs(tmp_path / "model.safetensors", compiler=compiler)

    assert options == {}
    assert not (tmp_path / ".tensorrt-engine-cache").exists()


def test_tensorrt_engine_cache_requires_explicit_opt_in(module, tmp_path, monkeypatch):
    def compiler(*, cache_built_engines, reuse_cached_engines, engine_cache_dir, engine_cache_size, immutable_weights):
        pass

    monkeypatch.setenv("RVE_UPSCALER_TRT_ENGINE_CACHE", "1")
    options = module.tensorrt_engine_cache_kwargs(tmp_path / "model.safetensors", compiler=compiler)

    assert options == {
        "cache_built_engines": True,
        "reuse_cached_engines": True,
        "engine_cache_dir": str(tmp_path / ".tensorrt-engine-cache"),
        "engine_cache_size": 5 << 30,
        "immutable_weights": False,
    }
    assert (tmp_path / ".tensorrt-engine-cache").is_dir()
