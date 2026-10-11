"""Opt-in, bounded 128px GPU smoke. Run with TVE_VFX_GPU_SMOKE=1."""

import os
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtx_vfx import RTXVfxPipeline, RTXFrameInterpolator, _input, _output


@pytest.mark.skipif(
    os.environ.get("TVE_VFX_GPU_SMOKE") != "1", reason="explicit GPU opt-in required"
)
def test_real_gpu_effects_and_stream_handoff():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # Never reclaim another process or allocate a full video.
    y, x = np.mgrid[:128, :128]
    a = np.stack((x * 2, y * 2, (x + y) % 256), axis=-1).astype(np.uint8)
    b = np.roll(a, 12, axis=1).copy()
    stream = torch.cuda.Stream()
    for mode in ("Off", "VSR", "High Bitrate", "Streaming Medium", "Streaming Ultra"):
        for encoding in ("rgb8", "rgb10a2"):
            p = RTXVfxPipeline(
                denoise=True,
                deblur=True,
                upscale_mode=mode,
                image_encoding=encoding,
                denoise_strength=0.5,
                deblur_strength=0.6,
                upscale_strength=0.7,
                hdr=True,
            )
            try:
                # Refinement at native size avoids large smoke outputs.
                with torch.cuda.stream(stream):
                    out1 = p.process(a, np.empty_like(a, dtype=np.uint16)).copy()
                    out2 = p.process(b, np.empty_like(a, dtype=np.uint16)).copy()
                assert out1.dtype == np.uint16
                assert np.isfinite(out1).all()
                assert np.mean(np.abs(out1.astype(float) - out2)) > 1
                assert np.unique(out1).size > 256
            finally:
                p.close()
    for encoding in ("rgb8", "rgb10a2"):
        fg = RTXFrameInterpolator(image_encoding=encoding)
        try:
            with torch.cuda.stream(stream):
                v1 = fg.interpolate(_input(a, "cpu"), _input(b, "cpu"), timestep=0.4)
                saved = v1.clone()
                v2 = fg.interpolate(_input(b, "cpu"), _input(a, "cpu"), timestep=0.7)
            assert torch.equal(v1, saved)
            assert v1.shape == (3, 128, 128)
            assert torch.isfinite(v1).all()
            assert float((v1 - v2).abs().mean()) > 1e-5
        finally:
            fg.close()
