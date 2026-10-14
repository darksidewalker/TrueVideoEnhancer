"""Native ONNX TensorRT inference, independent of ONNX Runtime's TensorRT ABI."""
from pathlib import Path
import hashlib
import json
import os

import numpy as np


def _input_shape(declared, input_size):
    if len(declared) != 4 or declared[0] not in (1, -1) or declared[1] != 3:
        raise ValueError(f"ONNX upscaler requires batch-one NCHW RGB input: {declared}")
    if input_size is None:
        input_size = (declared[3], declared[2])
    width, height = input_size
    if not all(isinstance(v, int) and v > 0 for v in (width, height)):
        raise ValueError("positive input_size is required for dynamic ONNX TensorRT upscaling")
    shape = (1, 3, height, width)
    if any(d != -1 and d != actual for d, actual in zip(declared, shape)):
        raise ValueError(f"input_size {input_size} does not match static ONNX shape {declared}")
    return shape


def _output_scale(shape, input_shape):
    if len(shape) != 4 or tuple(shape[:2]) != (1, 3) or any(v <= 0 for v in shape):
        raise ValueError(f"ONNX upscaler requires resolved batch-one NCHW RGB output: {shape}")
    h, w = input_shape[-2:]
    if shape[2] % h or shape[3] % w or shape[2] // h != shape[3] // w:
        raise ValueError(f"ONNX output must have the same positive integer H/W scale: {shape}")
    return shape[2] // h


def _cache_path(model_path, settings):
    import onnx
    from onnx.external_data_helper import _get_all_tensors
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    model = onnx.load(str(model_path), load_external_data=False)
    files = {model_path.resolve()}
    for tensor in _get_all_tensors(model):
        for entry in tensor.external_data:
            if entry.key == "location":
                files.add((model_path.parent / entry.value).resolve())
    for path in sorted(files):
        digest.update(str(path).encode())
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
    return model_path.parent / ".trt-cache" / (digest.hexdigest() + ".engine")


class TensorRTONNXUpscaler:
    batch_size = 1
    technique = "ONNX + native TensorRT"

    def __init__(self, model_path: Path, *, input_size, precision: str, gpu_id: int):
        import torch
        import tensorrt as trt
        from upscale_inference import tensorrt_workspace_size

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT ONNX upscaling requires CUDA")
        self.device = torch.device(f"cuda:{gpu_id}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        with torch.cuda.device(self.device):
            builder = trt.Builder(self.logger)
            if builder is None:
                raise RuntimeError("TensorRT builder creation failed")
            network = builder.create_network(0)
            if network is None:
                raise RuntimeError("TensorRT network creation failed")
            parser = trt.OnnxParser(network, self.logger)
            if not parser.parse_from_file(str(model_path.resolve())):
                errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise RuntimeError(f"TensorRT ONNX parse failed: {errors}")
            if network.num_inputs != 1 or network.num_outputs != 1:
                raise ValueError("ONNX upscaler requires exactly one input and one output")
            inp, out = network.get_input(0), network.get_output(0)
            self.input_name, self.output_name = inp.name, out.name
            dtypes = {trt.float16: torch.float16, trt.float32: torch.float32}
            if inp.dtype not in dtypes or out.dtype not in dtypes:
                raise ValueError("ONNX upscaler input/output dtype must be float16 or float32")
            self.dtype, self.output_tensor_dtype = dtypes[inp.dtype], dtypes[out.dtype]
            self.input_shape = _input_shape(tuple(inp.shape), input_size)
            props = torch.cuda.get_device_properties(self.device)
            self.cache_path = _cache_path(model_path, {
                "implementation": 1, "trt": trt.__version__, "shape": self.input_shape,
                "precision": precision, "gpu_id": gpu_id, "gpu": props.name,
                "capability": [props.major, props.minor],
            })
            self.runtime = trt.Runtime(self.logger)
            if self.runtime is None:
                raise RuntimeError("TensorRT runtime creation failed")
            self.engine = None
            if self.cache_path.is_file():
                self.engine = self.runtime.deserialize_cuda_engine(self.cache_path.read_bytes())
            cache_hit = self.engine is not None
            if self.engine is None:
                config = builder.create_builder_config()
                if config is None:
                    raise RuntimeError("TensorRT builder config creation failed")
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                             tensorrt_workspace_size((self.input_shape[3], self.input_shape[2])))
                # TensorRT 11 is strongly typed; the ONNX graph owns its dtype.
                fp16 = getattr(trt.BuilderFlag, "FP16", None)
                if precision == "float16" and fp16 is not None:
                    config.set_flag(fp16)
                if -1 in tuple(inp.shape):
                    profile = builder.create_optimization_profile()
                    if profile is None:
                        raise RuntimeError("TensorRT input profile creation failed")
                    # TRT 11 returns None on success; older releases return bool.
                    if profile.set_shape(inp.name, self.input_shape, self.input_shape, self.input_shape) is False:
                        raise RuntimeError("TensorRT input profile shape failed")
                    if not profile:
                        raise RuntimeError("TensorRT input profile is invalid")
                    if config.add_optimization_profile(profile) < 0:
                        raise RuntimeError("TensorRT input profile activation failed")
                serialized = builder.build_serialized_network(network, config)
                if serialized is None:
                    raise RuntimeError("TensorRT ONNX engine build failed")
                self.engine = self.runtime.deserialize_cuda_engine(serialized)
                if self.engine is None:
                    raise RuntimeError("TensorRT engine deserialization failed")
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.cache_path.with_suffix(f".{os.getpid()}.tmp")
                try:
                    temporary.write_bytes(bytes(serialized))
                    temporary.replace(self.cache_path)
                finally:
                    temporary.unlink(missing_ok=True)
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("TensorRT execution context creation failed")
            if not self.context.set_input_shape(self.input_name, self.input_shape):
                raise RuntimeError("TensorRT input shape binding failed")
            self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))
            self.scale = _output_scale(self.output_shape, self.input_shape)
            self.stream = torch.cuda.Stream(device=self.device)
            import sys
            print(f"[TECH] ONNX + native TensorRT {trt.__version__} "
                  f"input={self.input_shape} dtype={self.dtype} scale={self.scale} "
                  f"cache={'hit' if cache_hit else 'built'}", file=sys.stderr, flush=True)

    def upscale(self, frame):
        import torch
        if self.context is None:
            raise RuntimeError("upscaler is closed")
        if frame.dtype not in (np.uint8, np.uint16) or frame.shape != (*self.input_shape[2:], 3):
            raise ValueError(f"expected uint8/uint16 BGR frame of shape {(*self.input_shape[2:], 3)}")
        maximum = 65535.0 if frame.dtype == np.uint16 else 255.0
        rgb = np.ascontiguousarray(frame[..., ::-1].transpose(2, 0, 1)[None])
        try:
            with torch.cuda.device(self.device), torch.inference_mode(), torch.cuda.stream(self.stream):
                tensor = torch.from_numpy(rgb).float().div_(maximum).to(self.device, dtype=self.dtype)
                output = torch.empty(self.output_shape, device=self.device, dtype=self.output_tensor_dtype)
                for name, value in ((self.input_name, tensor), (self.output_name, output)):
                    if not self.context.set_tensor_address(name, value.data_ptr()):
                        raise RuntimeError(f"TensorRT tensor address binding failed: {name}")
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("TensorRT ONNX execution failed")
                dtype = getattr(self, "output_dtype", frame.dtype)
                maximum = 65535.0 if dtype == np.uint16 else 255.0
                array = output[0].float().clamp_(0, 1).mul_(maximum).round_().cpu().numpy()
        finally:
            self.stream.synchronize()
        return np.ascontiguousarray(array.transpose(1, 2, 0)[..., ::-1].astype(dtype))

    def close(self):
        if getattr(self, "stream", None) is not None:
            self.stream.synchronize()
        self.context = None
        self.engine = None
        self.runtime = None
        self.stream = None
