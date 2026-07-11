import importlib
from pathlib import Path
import sys

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


def test_rejects_legacy_pth_models(module, tmp_path):
    model = tmp_path / "legacy.pth"
    model.write_bytes(b"legacy")

    with pytest.raises(ValueError, match="safetensors.*onnx"):
        module.create_upscaler(model, backend="pytorch")


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


def test_auto_tile_avoids_full_frame_tensorrt_compile_for_large_input(module):
    assert module.choose_tile_size(704, 896, requested=0, backend="tensorrt") == 128
    assert module.choose_tile_size(32, 24, requested=0, backend="tensorrt") == 0
    assert module.choose_tile_size(704, 896, requested=256, backend="tensorrt") == 256


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
