import importlib
from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


def test_tensor_rt_11_builder_without_fp16_capability_attribute_defaults_to_supported():
    module = importlib.import_module("rve_backend")

    class TensorRT11Builder:
        pass

    assert module._supports_fast_fp16(TensorRT11Builder())


def test_legacy_builder_fp16_capability_is_respected():
    module = importlib.import_module("rve_backend")

    class LegacyBuilder:
        platform_has_fast_fp16 = False

    assert not module._supports_fast_fp16(LegacyBuilder())
