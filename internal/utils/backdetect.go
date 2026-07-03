package utils

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

// Backend represents an inference framework that can process videos.
type Backend struct {
	Name     string `json:"name"`
	Version  string `json:"version,omitempty"`
	Devices  []GPUDetails `json:"devices,omitempty"`
	Supported bool `json:"supported"`
	Error    string `json:"error,omitempty"`
}

// GPUDetails holds information about a single GPU device.
type GPUDetails struct {
	Index  int    `json:"index"`
	Name   string `json:"name"`
	VRAM   uint64 `json:"vram_mb,omitempty"`
	Compute string `json:"compute,omitempty"`
}

// BackendList returns all available inference backends with their capabilities.
// This replaces the Python BackendDetect class logic.
func BackendList() []Backend {
	var backends []Backend

	// Check PyTorch/CUDA
	backends = append(backends, detectPyTorch())

	// Check TensorRT
	backends = append(backends, detectTensorRT())

	// Check NCNN
	backends = append(backends, detectNCNN())

	return backends
}

// FormatBackends formats backend list as human-readable string.
func FormatBackends(backends []Backend) string {
	var sb strings.Builder
	sb.WriteString("DaSiWa TrueVideoEnhancer Backends:\n")
	sb.WriteString("==============================\n\n")

	for _, b := range backends {
		status := "✓"
		if !b.Supported {
			status = "✗"
		}

		sb.WriteString(fmt.Sprintf("[%s] %s", status, b.Name))
		if b.Version != "" {
			sb.WriteString(fmt.Sprintf(" v%s", b.Version))
		}
		sb.WriteString("\n")

		if b.Error != "" {
			sb.WriteString(fmt.Sprintf("  Error: %s\n", b.Error))
			continue
		}

		if len(b.Devices) > 0 {
			sb.WriteString("  Devices:\n")
			for _, dev := range b.Devices {
				sb.WriteString(fmt.Sprintf("    - %s (GPU %d)", dev.Name, dev.Index))
				if dev.VRAM > 0 {
					sb.WriteString(fmt.Sprintf(", %d MB VRAM", dev.VRAM))
				}
				if dev.Compute != "" {
					sb.WriteString(fmt.Sprintf(", Compute %s", dev.Compute))
				}
				sb.WriteString("\n")
			}
		} else {
			sb.WriteString("  No devices detected\n")
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// detectPyTorch checks for PyTorch installation and CUDA availability.
func detectPyTorch() Backend {
	b := Backend{Name: "PyTorch"}

	// Check if torch is installed by trying to run python -c "import torch"
	cmd := exec.Command("python3", "-c", "import torch; print(torch.__version__); print('CUDA_AVAILABLE=' + str(torch.cuda.is_available())); print('DEVICE_COUNT=' + str(torch.cuda.device_count()))")
	output, err := cmd.Output()
	if err != nil {
		b.Supported = false
		b.Error = fmt.Sprintf("PyTorch not found: %v", err)
		return b
	}

	lines := strings.Split(string(output), "\n")
	if len(lines) < 3 {
		b.Supported = false
		b.Error = "Unexpected torch output format"
		return b
	}

	b.Version = strings.TrimSpace(lines[0])

	// Parse CUDA availability
	cudaLines := strings.Split(lines[1], "=")
	if len(cudaLines) >= 2 {
		cudaAvail := strings.TrimSpace(cudaLines[1]) == "True"
		if !cudaAvail {
			b.Supported = false
			b.Error = "PyTorch installed but CUDA not available"
			return b
		}
	}

	// Parse device count
	deviceLines := strings.Split(lines[2], "=")
	if len(deviceLines) >= 2 {
		var deviceCount int
		fmt.Sscanf(strings.TrimSpace(deviceLines[1]), "%d", &deviceCount)

		// Query each GPU via nvidia-smi
		b.Devices = queryNvidiaSMIGPUs(deviceCount)
		b.Supported = true
	}

	return b
}

// detectTensorRT checks for TensorRT installation.
func detectTensorRT() Backend {
	b := Backend{Name: "TensorRT"}

	// Check if tensorrt is installed
	cmd := exec.Command("python3", "-c", "import tensorrt; print(tensorrt.__version__)")
	output, err := cmd.Output()
	if err != nil {
		b.Supported = false
		b.Error = fmt.Sprintf("TensorRT not found: %v", err)
		return b
	}

	b.Version = strings.TrimSpace(string(output))
	b.Supported = true

	// Note: Full TensorRT engine detection would require loading actual engines
	// For now we just verify the library is present
	return b
}

// detectNCNN checks for NCNN library availability.
func detectNCNN() Backend {
	b := Backend{Name: "NCNN"}

	// Check if ncnn is installed via Python bindings or system library
	// NCNN is typically used via Python wrapper or direct C++ calls
	cmd := exec.Command("find", "/usr/lib", "/usr/local/lib", "-name", "*ncnn*", "-type", "f")
	output, err := cmd.Output()
	if err != nil || len(bytes.TrimSpace(output)) == 0 {
		// Also check for Python bindings
		pyCmd := exec.Command("python3", "-c", "import ncnn")
		err = pyCmd.Run()
		if err != nil {
			b.Supported = false
			b.Error = "NCNN not found (neither library nor Python bindings)"
			return b
		}
	}

	b.Version = "system"
	b.Supported = true

	// NCNN doesn't have direct Python API for GPU listing like PyTorch
	// Fall back to nvidia-smi for GPU info
	b.Devices = queryNvidiaSMIGPUs(-1) // -1 means query all available

	return b
}

// queryNvidiaSMIGPUs queries NVIDIA GPUs using nvidia-smi.
// If maxGpus <= 0, queries all available GPUs.
func queryNvidiaSMIGPUs(maxGpus int) []GPUDetails {
	var gpus []GPUDetails

	cmd := exec.Command("nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err != nil {
		return gpus // Return empty if nvidia-smi fails
	}

	scanner := bufio.NewScanner(bytes.NewReader(output))
	lineNum := 0

	for scanner.Scan() && (maxGpus <= 0 || lineNum < maxGpus) {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		fields := strings.Split(line, ",")
		if len(fields) < 4 {
			continue
		}

		gpu := GPUDetails{}

		// Parse index
		fmt.Sscanf(strings.TrimSpace(fields[0]), "%d", &gpu.Index)

		// Parse name
		gpu.Name = strings.TrimSpace(fields[1])

		// Parse memory (already in MB due to --no-units)
		fmt.Sscanf(strings.TrimSpace(fields[2]), "%d", &gpu.VRAM)

		// Parse compute capability
		gpu.Compute = fmt.Sprintf("%s.%s",
			strings.TrimSpace(fields[3]),
			getComputeArch(gpu.Compute))

		gpus = append(gpus, gpu)
		lineNum++
	}

	return gpus
}

// getComputeArch returns the architecture name for a compute capability version.
func getComputeArch(capability string) string {
	parts := strings.Split(capability, ".")
	if len(parts) < 2 {
		return ""
	}

	major, _ := fmt.Sscanf(parts[0], "%d", new(int))
	switch major {
	case 7:
		return "Volta/Turing"
	case 8:
		return "Ampere"
	case 9:
		return "Hopper"
	default:
		return "Unknown"
	}
}

// HasCUDA returns true if CUDA-capable GPU is available.
func HasCUDA() bool {
	cmd := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
	return cmd.Run() == nil
}

// GetOSInfo returns basic OS and runtime information.
func GetOSInfo() map[string]string {
	info := make(map[string]string)
	info["os"] = runtime.GOOS
	info["arch"] = runtime.GOARCH

	// Get Go version
	cmd := exec.Command("go", "version")
	output, err := cmd.Output()
	if err == nil {
		info["go_version"] = strings.TrimSpace(string(output))
	}

	// Get Python version if available
	pyCmd := exec.Command("python3", "--version")
	output, err = pyCmd.Output()
	if err == nil {
		info["python_version"] = strings.TrimSpace(string(output))
	}

	return info
}

// PrintBackendSummary prints a formatted summary of available backends.
func PrintBackendSummary() {
	backends := BackendList()
	fmt.Print(FormatBackends(backends))
}

// SaveBackendReport saves backend detection results to a JSON file.
func SaveBackendReport(filename string) error {
	backends := BackendList()
	osInfo := GetOSInfo()

	report := struct {
		Timestamp string      `json:"timestamp"`
		OSInfo    map[string]string `json:"os_info"`
		Backends  []Backend   `json:"backends"`
	}{
		Timestamp: fmt.Sprintf("%d", time.Now().Unix()),
		OSInfo:    osInfo,
		Backends:  backends,
	}

	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal backend report: %w", err)
	}

	return os.WriteFile(filename, data, 0644)
}
