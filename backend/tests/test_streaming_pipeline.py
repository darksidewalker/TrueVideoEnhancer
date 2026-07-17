import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


def module():
    return importlib.import_module("streaming_pipeline")


def test_decode_command_outputs_raw_bgr_frames():
    command = module().decode_command("input.mp4", width=1088, height=1920)
    assert command[-5:] == ["-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    assert "%08d.png" not in " ".join(command)


def test_encode_command_accepts_rawvideo_and_maps_source_media():
    command = module().encode_command(
        "input.mp4", "output.mp4", width=2176, height=3840, fps=60,
        video_args=["-c:v", "av1_nvenc"], audio_args=["-c:a", "copy"],
        subtitle_args=["-c:s", "copy"],
    )
    joined = " ".join(command)
    assert "-f rawvideo -pix_fmt bgr24 -s 2176x3840 -r 60 -i -" in joined
    assert "-map 0:v:0 -map 1:a? -map 1:s?" in joined
    assert command[-1] == "output.mp4"


def test_output_schedule_supports_fractional_target_fps():
    schedule = list(module().output_schedule(source_count=3, source_fps=24, target_fps=60))
    assert [(left, right) for left, right, _ in schedule[:6]] == [
        (0, 1), (0, 1), (0, 1), (1, 2), (1, 2), (2, 2),
    ]
    assert [step for _, _, step in schedule[:6]] == pytest.approx([0.0, 0.4, 0.8, 0.2, 0.6, 0.0])


def test_raw_frame_roundtrip_uses_exact_frame_size():
    frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    restored = module().frame_from_bytes(frame.tobytes(), width=6, height=4)
    assert np.array_equal(restored, frame)


def test_fused_processor_runs_interpolation_then_upscale():
    calls = []

    class Interpolator:
        technique = "fake RIFE"
        def interpolate(self, a, b, timestep):
            calls.append(("rife", timestep))
            return a

    class Upscaler:
        scale = 2
        technique = "fake upscale"
        def upscale(self, frame):
            calls.append(("upscale", frame.shape))
            return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)

    a = np.zeros((2, 3, 3), np.uint8)
    b = np.ones((2, 3, 3), np.uint8)
    processor = module().FrameProcessor(
        interpolator=Interpolator(), upscaler=Upscaler(), tile_size=0,
        target_size=(6, 4), to_tensor=lambda frame: frame,
        to_bgr=lambda frame: frame,
    )
    output = processor.process(a, b, 0.5)
    assert output.shape == (4, 6, 3)
    assert calls == [("rife", 0.5), ("upscale", (2, 3, 3))]


def test_fused_processor_runs_a_2x_model_twice_for_a_4x_target():
    calls = []

    class Upscaler:
        scale = 2
        technique = "fake upscale"

        def upscale(self, frame):
            calls.append(frame.shape)
            return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)

    frame = np.zeros((2, 3, 3), np.uint8)
    processor = module().FrameProcessor(
        interpolator=None, upscaler=Upscaler(), tile_size=0,
        target_size=(12, 8), to_tensor=lambda value: value, to_bgr=lambda value: value,
    )
    output = processor.process(frame, frame, 0.0)

    assert output.shape == (8, 12, 3)
    assert calls == [(2, 3, 3), (4, 6, 3)]
