"""Backend detection bridge for Go backend migration.

The actual GPU and inference framework detection has been migrated to Go
(internal/utils/backdetect.go). This module provides compatibility shims
for code that hasn't been updated yet.
"""

import subprocess
import platform


class BackendDetect:
    """Compatibility wrapper for Go's BackendList function."""
    
    def __init__(self):
        self.pytorch_device = None
        self.pytorch_version = None
        self.tensorrt_version = None
        self.ncnn_version = None
        
    def get_tensorrt(self):
        """Check TensorRT availability (compatibility shim)."""
        try:
            result = subprocess.run(['python3', '-c', 'import tensorrt; print(tensorrt.__version__)'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.tensorrt_version = result.stdout.strip()
                return self.tensorrt_version
        except:
            pass
        return None
        
    def get_ncnn(self):
        """Check NCNN availability (compatibility shim)."""
        # NCNN is typically accessed via C++ or Python bindings
        try:
            result = subprocess.run(['find', '/usr/lib', '/usr/local/lib', '-name', '*ncnn*', '-type', 'f'],
                                  capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                self.ncnn_version = "system"
                return self.ncnn_version
        except:
            pass
        return None
        
    def get_half_precision(self):
        """Check if half precision is supported (compatibility shim)."""
        # Check GPU compute capability for half precision support
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                caps = result.stdout.strip().split('\n')
                for cap in caps:
                    major = int(cap.split('.')[0])
                    if major >= 6:  # Pascal and newer support FP16
                        return True
        except:
            pass
        return False
        
    def get_gpus_torch(self):
        """Get PyTorch-visible GPUs (compatibility shim)."""
        try:
            import torch
            if torch.cuda.is_available():
                gpus = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append(f"{props.name} ({props.total_mem / 1024**3:.1f}GB)")
                return gpus
        except:
            pass
        return []
        
    def get_gpus_ncnn(self):
        """Get NCNN-compatible GPUs (compatibility shim)."""
        # NCNN uses Vulkan, so we check for compatible GPUs
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        memory = int(parts[1].strip())
                        if memory >= 2048:  # At least 2GB VRAM
                            gpus.append(f"{name} ({memory}MB)")
                return gpus
        except:
            pass
        return []
