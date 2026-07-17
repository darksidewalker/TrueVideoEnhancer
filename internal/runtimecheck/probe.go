package runtimecheck

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

type runner func(context.Context, string, ...string) ([]byte, error)
type lookPathFunc func(string) (string, error)

type Probe struct {
	rootDir  string
	lookPath lookPathFunc
	run      runner
}

type InstallOptions struct {
	Models []string `json:"models"`
}

type ModelChoice struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Category    string `json:"category"`              // "upscaler" or "interpolation"
	SubCategory string `json:"subcategory,omitempty"` // upscaler: "anime", "mixed", "realism"; interpolation: ""
	File        string `json:"file"`
	URL         string `json:"url"`
	Destination string `json:"destination"`
	Present     bool   `json:"present,omitempty"`
}

type Option func(*Probe)

type Status struct {
	OS            string `json:"os"`
	Arch          string `json:"arch"`
	Python        Tool   `json:"python"`
	Backend       Tool   `json:"backend"`
	FFmpeg        Tool   `json:"ffmpeg"`
	NvidiaSMI     Tool   `json:"nvidia_smi"`
	Uv            Tool   `json:"uv"`
	RuntimeRoot   string `json:"runtime_root"`
	TensorRTMode  string `json:"tensorrt_mode"`
	OfflineBundle string `json:"offline_bundle"`
}

type Tool struct {
	Available bool   `json:"available"`
	Command   string `json:"command,omitempty"`
	Version   string `json:"version,omitempty"`
	Error     string `json:"error,omitempty"`
}

type InstallPlan struct {
	RootDir      string     `json:"root_dir"`
	Python       string     `json:"python"`
	Uv           string     `json:"uv"`
	Requirements string     `json:"requirements"`
	ModelsDir    string     `json:"models_dir"`
	Models       []string   `json:"models"`
	Commands     [][]string `json:"commands"`
}

type InstallResult struct {
	Status string      `json:"status"`
	Error  string      `json:"error,omitempty"`
	Plan   InstallPlan `json:"plan"`
	Logs   []string    `json:"logs"`
}

// BackendCheckResult holds the structured output of a backend health check.
type BackendCheckResult struct {
	Status      string      `json:"status"` // "ok" or "fail"
	Version     string      `json:"version,omitempty"`
	Timestamp   int64       `json:"timestamp"`
	Items       []CheckItem `json:"items"`
	PassedCount int         `json:"passed_count"`
	TotalCount  int         `json:"total_count"`
}

// CheckItem represents a single check result.
type CheckItem struct {
	Name   string `json:"name"`
	Pass   bool   `json:"pass"`
	Detail string `json:"detail,omitempty"` // shown when pass=true
	Error  string `json:"error,omitempty"`  // shown when pass=false
}

func hasLine(lines []string, contains string) bool {
	for _, l := range lines {
		if strings.Contains(l, contains) {
			return true
		}
	}
	return false
}

func NewProbe(options ...Option) *Probe {
	p := &Probe{lookPath: exec.LookPath, run: defaultRun}
	p.rootDir = filepath.Join(".", "runtime")
	for _, option := range options {
		option(p)
	}
	return p
}

func WithRootDir(root string) Option      { return func(p *Probe) { p.rootDir = root } }
func WithLookPath(fn lookPathFunc) Option { return func(p *Probe) { p.lookPath = fn } }
func WithRunner(fn runner) Option         { return func(p *Probe) { p.run = fn } }

func (p *Probe) RootDir() string { return p.rootDir }

func (p *Probe) Status() Status {
	pythonCmd := p.PythonCommand()
	uvCmd := p.UvCommand()
	python := p.toolVersion(pythonCmd, "--version")
	return Status{
		OS:            runtime.GOOS,
		Arch:          runtime.GOARCH,
		Python:        python,
		Backend:       p.backendVersion(pythonCmd),
		FFmpeg:        p.toolVersion("ffmpeg", "-version"),
		NvidiaSMI:     p.toolVersion("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"),
		Uv:            p.toolVersion(uvCmd, "--version"),
		RuntimeRoot:   p.rootDir,
		TensorRTMode:  "uv-managed-runtime",
		OfflineBundle: "blocked-until-complete",
	}
}

func (p *Probe) PythonCommand() string {
	// Prefer venv Python — it has PyTorch, TensorRT, and all inference packages
	venvPython := p.venvPython()
	if _, err := os.Stat(venvPython); err == nil {
		return venvPython
	}

	// Fallback: try to get uv-managed Python (bare, no packages — will fail at import)
	uv := p.UvCommand()
	out, err := p.run(context.Background(), uv, "python", "find", "3.12")
	if err == nil {
		path := strings.TrimSpace(string(out))
		if path != "" {
			return path
		}
	}

	// Last resort: system Python (requires installation)
	for _, candidate := range []string{"python3.13", "python3.12", "python3", "python"} {
		if _, err := p.lookPath(candidate); err == nil {
			return candidate
		}
	}
	return "python3"
}

func (p *Probe) UvCommand() string {
	local := filepath.Join(p.rootDir, "bin", "uv")
	if _, err := os.Stat(local); err == nil {
		return local
	}
	if uv, err := p.lookPath("uv"); err == nil {
		return uv
	}
	return "uv"
}

func (p *Probe) InstallPlan(repoRoot string, modelIDs ...string) InstallPlan {
	uv := p.UvCommand()
	python := p.venvPython()
	requirements := filepath.Join(repoRoot, "backend", "requirements.txt")
	modelFiles := make([]string, 0, len(modelIDs))
	for _, model := range p.ModelChoices(modelIDs) {
		modelFiles = append(modelFiles, model.Destination)
	}

	venvPath := filepath.Join(p.rootDir, "venv")
	pythonVersion := p.bestAvailablePythonVersion()
	return InstallPlan{
		RootDir:      p.rootDir,
		Python:       python,
		Uv:           uv,
		Requirements: requirements,
		ModelsDir:    filepath.Join(repoRoot, "models"),
		Models:       modelFiles,
		Commands: [][]string{
			{uv, "venv", "--python", pythonVersion, venvPath},
			{uv, "pip", "install", "--python", python, "--index-strategy", "unsafe-best-match", "--force-reinstall", "-r", requirements},
		},
	}
}

// bestAvailablePythonVersion returns the highest available Python version string (e.g. "3.12", "3.11").
// Uses uv's built-in Python management for portability - no system-wide Python required.
func (p *Probe) bestAvailablePythonVersion() string {
	// Try to use uv's managed Python 3.12 (PyTorch 2.12.0 supports up to 3.12)
	if err := p.ensureUvPython("3.12"); err == nil {
		return "3.12"
	}

	// Fallback: try uv-managed Python 3.11
	if err := p.ensureUvPython("3.11"); err == nil {
		return "3.11"
	}

	// Ultimate fallback - but this requires system Python
	for _, cmd := range []string{"python3", "python"} {
		if path, _ := p.lookPath(cmd); path != "" {
			out, err := p.run(context.Background(), path, "--version")
			if err == nil {
				parts := strings.Fields(strings.TrimSpace(string(out)))
				if len(parts) >= 2 && strings.HasPrefix(parts[1], "3.") {
					return parts[1]
				}
			}
		}
	}
	return "3.12" // Ultimate fallback
}

// ensureUvPython installs and verifies a specific Python version via uv.
// Returns nil if successful, error otherwise.
func (p *Probe) ensureUvPython(version string) error {
	uv := p.UvCommand()

	// Install Python via uv if not already installed
	_, err := p.run(context.Background(), uv, "python", "install", version)
	if err != nil {
		return fmt.Errorf("failed to install Python %s via uv: %w", version, err)
	}

	// Verify installation by finding the exact path
	out, err := p.run(context.Background(), uv, "python", "find", version)
	if err != nil {
		return fmt.Errorf("failed to find Python %s: %w", version, err)
	}

	path := strings.TrimSpace(string(out))
	if path == "" {
		return fmt.Errorf("uv returned empty path for Python %s", version)
	}

	// Verify the Python binary works
	_, err = p.run(context.Background(), path, "--version")
	if err != nil {
		return fmt.Errorf("Python %s verification failed: %w", version, err)
	}

	return nil
}

func (p *Probe) BackendCheck(repoRoot string) BackendCheckResult {
	result := BackendCheckResult{Timestamp: time.Now().Unix()}
	pythonCmd := p.PythonCommand()
	backendScript := filepath.Join(repoRoot, "backend", "rve-backend.py")

	// Check if backend script exists
	if _, err := os.Stat(backendScript); err != nil {
		result.Status = "fail"
		result.Items = append(result.Items, CheckItem{Name: "backend_script", Pass: false, Error: "Backend script not found"})
		return result
	}

	// Run backend --list-backends
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := p.run(ctx, pythonCmd, backendScript, "--list-backends")
	if err != nil {
		result.Status = "fail"
		result.Items = append(result.Items, CheckItem{Name: "python_execution", Pass: false, Error: err.Error()})
		return result
	}

	output := strings.TrimSpace(string(out))
	lines := strings.Split(output, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || line == "DaSiWa TrueVideoEnhancer Backends:" || line == strings.Repeat("=", len(line)) {
			continue
		}

		checkItem := CheckItem{}
		if strings.HasPrefix(line, "[✓]") {
			checkItem.Pass = true
			checkItem.Name = strings.TrimPrefix(line, "[✓] ")
		} else if strings.HasPrefix(line, "[✗]") {
			checkItem.Pass = false
			checkItem.Name = strings.TrimPrefix(line, "[✗] ")
		} else {
			continue
		}

		result.Items = append(result.Items, checkItem)
		result.TotalCount++
		if checkItem.Pass {
			result.PassedCount++
		}
	}

	if result.PassedCount == result.TotalCount && result.TotalCount > 0 {
		result.Status = "ok"
	} else {
		result.Status = "fail"
	}

	return result
}

func (p *Probe) Install(ctx context.Context, repoRoot string, options InstallOptions, logf func(string)) error {
	if logf == nil {
		logf = func(string) {}
	}
	if err := os.MkdirAll(p.rootDir, 0755); err != nil {
		return err
	}

	// Completely remove existing venv to ensure clean upgrade
	venvPath := filepath.Join(p.rootDir, "venv")
	os.RemoveAll(venvPath)
	logf("Removed existing virtual environment for clean upgrade")

	uv := p.UvCommand()
	if _, err := p.lookPath(uv); err != nil {
		if err := p.bootstrapUv(ctx, logf); err != nil {
			return fmt.Errorf("uv bootstrap failed: %w", err)
		}
		uv = p.UvCommand()
	}
	plan := p.InstallPlan(repoRoot, options.Models...)
	for _, command := range plan.Commands {
		if len(command) == 0 {
			continue
		}
		logf("Running: " + strings.Join(command, " "))
		out, err := p.run(ctx, command[0], command[1:]...)
		if text := strings.TrimSpace(string(out)); text != "" {
			logf(text)
		}
		if err != nil {
			return fmt.Errorf("%s failed: %w", strings.Join(command, " "), err)
		}
	}
	return nil
}

func (p *Probe) DownloadModels(ctx context.Context, modelIDs []string, logf func(string)) error {
	for _, model := range p.ModelChoices(modelIDs) {
		if err := downloadModel(ctx, model, logf); err != nil {
			return err
		}
	}
	return nil
}

func (p *Probe) AvailableModels(repoRoot string) []ModelChoice {
	choices := builtInModelChoices(filepath.Join(repoRoot, "models"))
	return choices
}

func (p *Probe) ModelChoices(ids []string) []ModelChoice {
	if len(ids) == 0 {
		return nil
	}
	all := builtInModelChoices(filepath.Join(filepath.Dir(p.rootDir), "models"))
	byID := make(map[string]ModelChoice, len(all))
	for _, model := range all {
		byID[model.ID] = model
	}
	selected := make([]ModelChoice, 0, len(ids))
	for _, id := range ids {
		if model, ok := byID[id]; ok {
			selected = append(selected, model)
		}
	}
	return selected
}

func builtInModelChoices(modelsDir string) []ModelChoice {
	const phhofmBase = "https://github.com/Phhofm/models/releases/download/"
	const hfBase = "https://huggingface.co/"
	const hfFrameInterp = "https://huggingface.co/Comfy-Org/frame_interpolation/resolve/main/frame_interpolation/"

	files := []struct{ id, name, file, category, subcategory, urlOverride string }{
		// Recommended video upscalers come first in each content/scale bucket: Auto routing picks the first exact match.
		// Anime: compression-trained native models; the 4x ONNX graph has a fixed 256x256 input.
		{"animejanai-hd-v3-compact-2x", "AnimeJaNai HD V3 Compact 2x (Ultra Fast Anime/Mixed)", "2x-AnimeJaNai_HD_V3_Compact.safetensors", "upscaler", "anime", ""},
		{"anime-sharp-v4-2x", "AnimeSharpV4 Fast RCAN-PU 2x (Video)", "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors", "upscaler", "anime", hfBase + "Kim2091/2x-AnimeSharpV4/resolve/main/2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"},
		{"anime-sharp-v4-quality-2x", "AnimeSharpV4 RCAN 2x (Quality)", "2x-AnimeSharpV4_RCAN.safetensors", "upscaler", "anime", hfBase + "Kim2091/2x-AnimeSharpV4/resolve/main/2x-AnimeSharpV4_RCAN.safetensors"},
		{"nomosuni-span-anime-4x", "NomosUni SPAN Multi-JPEG 4x (Fast)", "4xNomosUni_span_multijpg_fp16_opset17.onnx", "upscaler", "anime", phhofmBase + "4xNomosUni_span_multijpg/4xNomosUni_span_multijpg_fp16_opset17.onnx"},
		{"anime-restoration-4x", "HFA2k LUDVAE RealPLKSR 4x (Video)", "4xHFA2k_ludvae_realplksr_dysample_256_fp16_fullyoptimized.onnx", "upscaler", "anime", phhofmBase + "4xHFA2k_ludvae_realplksr_dysample/4xHFA2k_ludvae_realplksr_dysample_256_fp16_fullyoptimized.onnx"},

		// Mixed: fast SPAN restoration for 2x and broad-content RealPLKSR-Lite restoration for 4x.
		{"animejanai-hd-v3-compact-mixed-2x", "AnimeJaNai HD V3 Compact 2x (Ultra Fast Anime/Mixed)", "2x-AnimeJaNai_HD_V3_Compact.safetensors", "upscaler", "mixed", ""},
		{"nomosuni-span-2x", "NomosUni SPAN Multi-JPEG 2x (Fast)", "2xNomosUni_span_multijpg_fp16_opset17.onnx", "upscaler", "mixed", phhofmBase + "2xNomosUni_span_multijpg/2xNomosUni_span_multijpg_fp16_opset17.onnx"},
		{"nomosuni-span-4x", "NomosUni SPAN Multi-JPEG 4x (Fast)", "4xNomosUni_span_multijpg_fp16_opset17.onnx", "upscaler", "mixed", phhofmBase + "4xNomosUni_span_multijpg/4xNomosUni_span_multijpg_fp16_opset17.onnx"},
		{"ultrasharpv2-4x", "UltraSharpV2-Lite RealPLKSR 4x (Video)", "4x-UltraSharpV2_Lite.safetensors", "upscaler", "mixed", hfBase + "Kim2091/UltraSharpV2/resolve/main/4x-UltraSharpV2_Lite.safetensors"},

		// Realism: degradation-trained RealPLKSR 2x and artifact-resistant SPAN 4x.
		{"realplksr-restoration-2x", "Public RealPLKSR Restoration 2x (Video)", "2xPublic_realplksr_dysample_layernorm_real.safetensors", "upscaler", "realism", phhofmBase + "2xPublic_realplksr_dysample_layernorm_real/2xPublic_realplksr_dysample_layernorm_real.safetensors"},
		{"clearreality-4x", "ClearRealityV1 SPAN 4x (Video)", "4x-ClearRealityV1.safetensors", "upscaler", "realism", hfBase + "Kim2091/ClearRealityV1/resolve/main/4x-ClearRealityV1.safetensors"},
		{"nomos-webphoto-4x", "Nomos WebPhoto RealPLKSR 4x (Restoration)", "4xNomosWebPhoto_RealPLKSR.safetensors", "upscaler", "realism", phhofmBase + "4xNomosWebPhoto_RealPLKSR/4xNomosWebPhoto_RealPLKSR.safetensors"},

		// Optional slower/legacy quality choices; never selected before the video defaults above.
		{"hat-l-sharp-4x", "HAT-L Sharp Anime 4x (Very Slow)", "4xBHI_small_hat-l_sharp.safetensors", "upscaler", "anime", phhofmBase + "4xBHI_small_hat-l/4xBHI_small_hat-l_sharp.safetensors"},
		{"realplksr-gan-2x", "RealPLKSR GAN 2x (Clean Inputs)", "2xPublic_realplksr_dysample_layernorm_gan.safetensors", "upscaler", "realism", phhofmBase + "2xPublic_realplksr_dysample_layernorm_gan/2xPublic_realplksr_dysample_layernorm_gan.safetensors"},
		{"hat-l-4x", "HAT-L Realism 4x (Very Slow)", "4xBHI_small_hat-l.safetensors", "upscaler", "realism", phhofmBase + "4xBHI_small_hat-l/4xBHI_small_hat-l.safetensors"},

		// Interpolation models (RIFE v4.26 remains current SOTA for speed-quality balance)
		{"rife-v4.26", "RIFE v4.26 General", "rife_v4.26.safetensors", "interpolation", "", hfFrameInterp + "rife_v4.26.safetensors"},
	}

	const rveBase = "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/"
	choices := make([]ModelChoice, 0, len(files))
	for _, file := range files {
		dest := filepath.Join(modelsDir, file.file)
		present := modelIsPresent(dest, modelsDir)

		url := ""
		if file.urlOverride != "" {
			url = file.urlOverride
		} else {
			url = rveBase + file.file
		}

		choices = append(choices, ModelChoice{ID: file.id, Name: file.name, Category: file.category, SubCategory: file.subcategory, File: file.file, URL: url, Destination: dest, Present: present})
	}
	return choices
}

// modelIsPresent returns true if the destination file exists or, for .tar.gz archives,
// a corresponding extracted .safetensors was found inside modelsDir.
func modelIsPresent(dest, modelsDir string) bool {
	if _, err := os.Stat(dest); err == nil {
		return true
	}
	if strings.HasSuffix(dest, ".tar.gz") {
		baseName := strings.TrimSuffix(filepath.Base(dest), ".tar.gz") + ".safetensors"
		files, _ := filepath.Glob(filepath.Join(modelsDir, "*.safetensors"))
		for _, f := range files {
			if filepath.Base(f) == baseName {
				return true
			}
		}
	}
	return false
}

func downloadModel(ctx context.Context, model ModelChoice, logf func(string)) error {
	if _, err := os.Stat(model.Destination); err == nil {
		logf("Model already present: " + model.Destination)
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(model.Destination), 0755); err != nil {
		return err
	}
	logf("Downloading model: " + model.Name)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, model.URL, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("download %s failed: %w", model.ID, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("download %s failed: %s", model.ID, resp.Status)
	}
	tmp := model.Destination + ".part"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, resp.Body)
	closeErr := out.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	if err := os.Rename(tmp, model.Destination); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	logf("Model saved: " + model.Destination)
	if strings.HasSuffix(model.Destination, ".tar.gz") {
		if err := extractTarGZ(model.Destination, filepath.Dir(model.Destination)); err != nil {
			return fmt.Errorf("extract %s failed: %w", model.File, err)
		}
		logf("Model extracted: " + filepath.Dir(model.Destination))
	}
	return nil
}

func extractTarGZ(path, dest string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	gz, err := gzip.NewReader(file)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		header, err := tr.Next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		cleanDest := filepath.Clean(dest)
		target := filepath.Join(cleanDest, filepath.Clean(header.Name))
		if target != cleanDest && !strings.HasPrefix(target, cleanDest+string(os.PathSeparator)) {
			return fmt.Errorf("unsafe tar path: %s", header.Name)
		}
		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
				return err
			}
			out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.FileMode(header.Mode))
			if err != nil {
				return err
			}
			_, copyErr := io.Copy(out, tr)
			closeErr := out.Close()
			if copyErr != nil {
				return copyErr
			}
			if closeErr != nil {
				return closeErr
			}
		}
	}
}

func (p *Probe) bootstrapUv(ctx context.Context, logf func(string)) error {
	if runtime.GOOS == "windows" {
		return errors.New("uv was not found; install uv first from https://docs.astral.sh/uv/getting-started/installation/")
	}
	installDir := filepath.Join(p.rootDir, "bin")
	if err := os.MkdirAll(installDir, 0755); err != nil {
		return err
	}
	cmd := "sh"
	args := []string{"-c", "curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR='" + installDir + "' sh -s -- -y"}
	logf("uv not found; bootstrapping uv into " + installDir)
	out, err := p.run(ctx, cmd, args...)
	if text := strings.TrimSpace(string(out)); text != "" {
		logf(text)
	}
	return err
}

func (p *Probe) venvPython() string {
	if runtime.GOOS == "windows" {
		return filepath.Join(p.rootDir, "venv", "Scripts", "python.exe")
	}
	return filepath.Join(p.rootDir, "venv", "bin", "python")
}

func (p *Probe) backendVersion(python string) Tool {
	return p.commandVersion(python, "backend/rve-backend.py", "--version")
}

func (p *Probe) toolVersion(command string, args ...string) Tool {
	if _, err := p.lookPath(command); err != nil && !filepath.IsAbs(command) {
		return Tool{Available: false, Command: command, Error: err.Error()}
	}
	return p.commandVersion(command, args...)
}

func (p *Probe) commandVersion(command string, args ...string) Tool {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := p.run(ctx, command, args...)
	tool := Tool{Available: err == nil, Command: command, Version: firstLine(string(out))}
	if err != nil {
		tool.Error = err.Error()
	}
	return tool
}

func defaultRun(ctx context.Context, command string, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, command, args...).CombinedOutput()
}

func firstLine(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	line, _, _ := strings.Cut(text, "\n")
	return strings.TrimSpace(line)
}
