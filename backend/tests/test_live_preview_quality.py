"""Live image transport must not discard processed-frame resolution."""
import base64
import importlib.util
from pathlib import Path

import cv2
import numpy as np


def test_live_preview_preserves_native_resolution(capsys):
    spec = importlib.util.spec_from_file_location('live_preview_cli', Path(__file__).resolve().parents[1] / 'rve-backend.py')
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    capsys.readouterr()  # discard backend import diagnostics
    # Real high-detail pixels and actual JPEG encode/decode, not a codec mock.
    frame = np.random.default_rng(42).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    cli.emit_live_preview(frame)
    line = capsys.readouterr().err.strip()
    assert line.startswith('<PREVIEW>')
    jpeg = base64.b64decode(line[len('<PREVIEW>'):])
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == frame.shape
    assert len(line) > 1024 * 1024  # worker must handle native-size transport
