import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtx_vfx import (
    pack_rgb10a2,
    unpack_rgb10a2,
    validate_options,
    capabilities,
    precision_video_args,
)


def test_packed_roundtrip():
    import torch

    x = (torch.arange(1024).float() / 1023).repeat(3, 2, 1)
    y = unpack_rgb10a2(pack_rgb10a2(x))
    assert torch.equal(y, x)
    assert pack_rgb10a2(x).dtype == torch.uint32


@pytest.mark.parametrize(
    "name,value",
    [
        ("rtx_denoise_strength", -0.1),
        ("rtx_hdr_luminance", float("nan")),
        ("rtx_hdr_contrast", 201),
    ],
)
def test_invalid_controls(name, value):
    with pytest.raises(ValueError):
        validate_options({name: value})


def test_hdr_encoding_rejects_eight_bit():
    with pytest.raises(ValueError):
        precision_video_args(["-c:v", "libx264", "-pix_fmt", "yuv420p"], hdr=True)


def test_hdr_encoding_tags_and_conversion():
    args = precision_video_args(["-c:v", "libx265", "-pix_fmt", "yuv420p"], hdr=True)
    assert args[args.index("-pix_fmt") + 1] == "yuv420p10le"
    assert "smpte2084" in args and "bt2020" in args
    assert "matrixin=gbr" in args[args.index("-vf") + 1]


def test_capabilities_no_effect_construction(monkeypatch):
    import types

    class VSR:
        def __init__(self, quality=None, strength=1.0, device=0, image_encoding=None):
            raise AssertionError("must not construct")

    monkeypatch.setitem(sys.modules, "nvvfx", types.SimpleNamespace(VideoSuperRes=VSR))
    assert capabilities()["available"]


def test_deblur_only_and_owned_output(monkeypatch):
    import torch, types, rtx_vfx

    class Effect:
        def __init__(self, quality=None, strength=1.0, device=0, image_encoding=None):
            self.closed = False

        def load(self):
            pass

        def run(self, x, stream_ptr=0):
            return types.SimpleNamespace(image=x + 0.1)

        def close(self):
            self.closed = True

    sdk = types.SimpleNamespace(
        VideoSuperRes=Effect,
        effects=types.SimpleNamespace(
            QualityLevel=types.SimpleNamespace(DEBLUR_ULTRA=15)
        ),
        ImageEncoding=types.SimpleNamespace(RGB8=0, RGB10A2=1),
    )
    monkeypatch.setattr(rtx_vfx, "_sdk", lambda: sdk)
    p = rtx_vfx.RTXVfxPipeline(deblur=True)
    p._ensure_loaded(8, 8, 8, 8)
    assert len(p._effects) == 1
    assert p._effects[0][1] == "DEBLUR_ULTRA"
    p.close()
    p.close()


def test_owned_sdk_output_survives_reuse():
    import torch, types
    from rtx_vfx import _owned

    borrowed = torch.ones(3, 2, 2)
    saved = _owned(types.SimpleNamespace(image=borrowed), False)
    borrowed.zero_()
    assert saved.sum() == 12


def test_precise_processor_never_allocates_uint8():
    from streaming_pipeline import FrameProcessor

    class VFX:
        def process(self, frame, out):
            assert out.dtype == np.uint16
            out[:] = 1025
            return out

    p = FrameProcessor(
        None,
        None,
        0,
        (4, 4),
        lambda x: x,
        lambda x: x,
        vfx_upscaler=VFX(),
        output_dtype=np.uint16,
    )
    x = np.zeros((2, 2, 3), np.uint16)
    assert p.process(x, x, 0).dtype == np.uint16


def test_arbitrary_cadence_preserves_duration_and_audio_mapping():
    from streaming_pipeline import stream_frames

    class Reader:
        def frames(self):
            yield from range(3)

    class Writer:
        def __init__(self):
            self.frames = []

        def put(self, x):
            self.frames.append(x)

    class Processor:
        def process(self, a, b, t):
            return (a, b, t)

    w = Writer()
    assert (
        stream_frames(
            Reader(),
            w,
            Processor(),
            source_fps=24,
            target_fps=60,
            preserve_duration=True,
        )
        == 8
    )
    assert [f[2] for f in w.frames[:5]] == pytest.approx([0, 0.4, 0.8, 0.2, 0.6])
    assert w.frames[-1] == (2, 2, 0)


def test_old_sdk_nondefault_strength_rejected(monkeypatch):
    import types, rtx_vfx

    class Effect:
        def __init__(self, quality=None, device=0):
            raise AssertionError("must reject before constructing")

    sdk = types.SimpleNamespace(
        VideoSuperRes=Effect,
        effects=types.SimpleNamespace(
            QualityLevel=types.SimpleNamespace(DEBLUR_ULTRA=15)
        ),
    )
    monkeypatch.setattr(rtx_vfx, "_sdk", lambda: sdk)
    p = rtx_vfx.RTXVfxPipeline(deblur=True, deblur_strength=0.5)
    with pytest.raises(RuntimeError, match="strength unavailable"):
        p._ensure_loaded(8, 8, 8, 8)


def test_bool_parser_cli_contract():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_vfx_cli", Path(__file__).resolve().parents[1] / "rve-backend.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    args = cli.build_parser().parse_args(
        [
            "--pipeline",
            "nvidia-vfx",
            "--rtx_auto_shot_detection",
            "false",
            "--rtx_hdr_debanding",
            "true",
        ]
    )
    assert not args.rtx_auto_shot_detection and args.rtx_hdr_debanding
    assert args.rtx_fg_mode == "MEDIUM" and args.rtx_denoise_strength == 1
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--rtx_hdr_debanding", "garbage"])


def test_capability_cli_is_lazy():
    import subprocess

    script = Path(__file__).resolve().parents[1] / "rve-backend.py"
    code = "import sys,runpy,atexit; sys.argv=[sys.argv[1],'--vfx-capabilities']; atexit.register(lambda: print('TORCH_IMPORTED='+str('torch' in sys.modules))); runpy.run_path(sys.argv[0],run_name='__main__')"
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "TORCH_IMPORTED=False" in result.stdout


def test_packed_channel_layout_and_cpu_color_conversion():
    import torch
    from rtx_vfx import _input, _output

    rgb = torch.eye(3).reshape(3, 1, 3)
    words = pack_rgb10a2(rgb).view(torch.int32)
    assert torch.equal(words & 1023, torch.tensor([[1023, 0, 0]]))
    assert torch.equal((words >> 10) & 1023, torch.tensor([[0, 1023, 0]]))
    assert torch.equal((words >> 20) & 1023, torch.tensor([[0, 0, 1023]]))
    bgr = np.array([[[32, 64, 128]]], dtype=np.uint8)
    assert np.array_equal(_output(_input(bgr, "cpu"), np.uint8), bgr)


def test_writer_failure_never_deadlocks_a_full_queue():
    from streaming_pipeline import RawVideoWriter

    writer = RawVideoWriter(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('intentional failure'); sys.exit(1)",
        ],
        queue_size=2,
    )
    try:
        writer.start()
        with pytest.raises(RuntimeError):
            for _ in range(20):
                writer.put(np.zeros((128, 128, 3), np.uint8))
            writer.finish()
    finally:
        writer.close()
    assert not writer.thread.is_alive()


def test_reader_close_unblocks_full_queue():
    import time
    from streaming_pipeline import RawVideoReader

    reader = RawVideoReader("unused", width=2, height=2, queue_size=2)
    reader.command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'x'*1200)",
    ]
    reader.start()
    deadline = time.monotonic() + 3
    while not reader.queue.full() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert reader.queue.full()
    reader.close()
    assert not reader.thread.is_alive()


def test_uint16_raw_frames():
    from streaming_pipeline import frame_from_bytes, encode_command, decode_command

    x = np.arange(12, dtype="<u2").reshape(2, 2, 3)
    assert np.array_equal(
        frame_from_bytes(x.tobytes(), width=2, height=2, pixel_format="bgr48le"), x
    )
    cmd = encode_command(
        "in.mp4",
        "out.mkv",
        width=2,
        height=2,
        fps=60,
        video_args=[],
        audio_args=[],
        subtitle_args=[],
        pixel_format="bgr48le",
    )
    assert cmd[cmd.index("-pix_fmt") + 1] == "bgr48le"
    assert "bgr48le" in decode_command("in", width=2, height=2, pixel_format="bgr48le")
