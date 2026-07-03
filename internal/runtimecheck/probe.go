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
	Category    string `json:"category"` // "upscaler" or "interpolation"
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
	Status     string      `json:"status"` // "ok" or "fail"
	Version    string      `json:"version,omitempty"`
	Timestamp  int64       `json:"timestamp"`
	Items      []CheckItem `json:"items"`
	PassedCount int        `json:"passed_count"`
	TotalCount  int        `json:"total_count"`
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
	venvPython := p.venvPython()
	if _, err := os.Stat(venvPython); err == nil {
		return venvPython
	}
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
	pythonVersion := p.bestAvailablePythonVersion()
	return InstallPlan{
		RootDir:      p.rootDir,
		Python:       python,
		Uv:           uv,
		Requirements: requirements,
		ModelsDir:    filepath.Join(repoRoot, "models"),
		Models:       modelFiles,
		Commands: [][]string{
			{uv, "venv", "--python", pythonVersion, "--clear", filepath.Join(p.rootDir, "venv")},
			{uv, "pip", "install", "--python", python, "--index-strategy", "unsafe-best-match", "-r", requirements},
		},
	}
}

// bestAvailablePythonVersion returns the highest available Python version string (e.g. "3.13", "3.12").
// Prefers newer versions; falls back gracefully if a specific version isn't installed.
func (p *Probe) bestAvailablePythonVersion() string {
	preferredOrder := []string{"python3.13", "python3.12", "python3.11"}
	for _, name := range preferredOrder {
		if _, err := p.lookPath(name); err == nil {
			// Extract major.minor from the command name
			var ver string
			if strings.HasPrefix(name, "python") {
				ver = strings.TrimPrefix(name, "python")
			}
			if ver != "" {
				return ver
			}
		}
	}
	// Fallback: try any python3.x and extract version
	for _, cmd := range []string{"python3", "python"} {
		if path, _ := p.lookPath(cmd); path != "" {
			out, err := p.run(context.Background(), path, "--version")
			if err == nil {
				// Output like "Python 3.12.5" or "Python 3.13.0"
				parts := strings.Fields(strings.TrimSpace(string(out)))
				if len(parts) >= 2 && strings.HasPrefix(parts[1], "3.") {
					return parts[1]
				}
			}
		}
	}
	return "3.12" // Ultimate fallback
}

func (p *Probe) Install(ctx context.Context, repoRoot string, options InstallOptions, logf func(string)) error {
	if logf == nil {
		logf = func(string) {}
	}
	if err := os.MkdirAll(p.rootDir, 0755); err != nil {
		return err
	}
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
		// Anime upscalers (AnimeSharpV4 Fast RCAN for 2x, HAT-L Sharp for 4x)
		{"anime-sharp-v4-2x", "AnimeSharpV4-Fast RCAN PU 2x",   "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors",      "upscaler", "anime", hfBase + "Kim2091/2x-AnimeSharpV4/resolve/main/2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"},
		{"hat-l-sharp-4x",    "HAT-L Sharp Anime 4x",            "4xBHI_small_hat-l_sharp.safetensors",          "upscaler", "anime", phhofmBase + "4xBHI_small_hat-l/4xBHI_small_hat-l_sharp.safetensors"},

		// Mixed content upscalers (UltraSharpV2 — RealPLKSR-lite architecture, general purpose)
		{"ultrasharpv2-4x","UltraSharpV2-Lite 4x",                   "4x-UltraSharpV2_Lite.safetensors",              "upscaler", "mixed", hfBase + "Kim2091/UltraSharpV2/resolve/main/4x-UltraSharpV2_Lite.safetensors"},

		// Realism upscalers (photorealistic / live-action)
		{"realplksr-gan-2x","RealPLKSR-DySample GAN 2x",    "2xPublic_realplksr_dysample_layernorm_gan.safetensors", "upscaler", "realism", phhofmBase + "2xPublic_realplksr_dysample_layernorm_gan/2xPublic_realplksr_dysample_layernorm_gan.safetensors"},
		{"hat-l-4x",        "HAT-L Realism 4x",             "4xBHI_small_hat-l.safetensors",            "upscaler", "realism", phhofmBase + "4xBHI_small_hat-l/4xBHI_small_hat-l.safetensors"},

		// Interpolation models (RIFE v4.26 remains current SOTA for speed-quality balance)
		{"rife-v4.26",      "RIFE v4.26 General",            "rife_v4.26.safetensors",                  "interpolation", "", hfFrameInterp + "rife_v4.26.safetensors"},
		{"rife-v4.26-heavy","RIFE v4.26 Heavy Anime",        "rife_v4.26_heavy.safetensors",            "interpolation", "", hfFrameInterp + "rife_v4.26_heavy.safetensors"},
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

// BackendCheck runs the Python backend with --list-backends and returns structured check results.
func (p *Probe) BackendCheck(repoRoot string) BackendCheckResult {
	pythonCmd := p.PythonCommand()
	backendScript := filepath.Join(repoRoot, "backend", "rve-backend.py")

	items := []CheckItem{}
	timestamp := time.Now().Unix()

	// Check 1: Python available + version
	pyTool := p.toolVersion(pythonCmd, "--version")
	if pyTool.Available {
		items = append(items, CheckItem{Name: "Python runtime", Pass: true, Detail: pyTool.Version})
	} else {
		items = append(items, CheckItem{Name: "Python runtime", Pass: false, Error: pyTool.Error})
		timestamp = 0 // skip timestamp if python isn't available
	}

	// Check 2: Backend script exists
	if _, err := os.Stat(backendScript); err == nil {
		items = append(items, CheckItem{Name: "Backend script", Pass: true, Detail: filepath.Base(backendScript)})
	} else {
		items = append(items, CheckItem{Name: "Backend script", Pass: false, Error: err.Error()})
	}

	// Run --list-backends to get TensorRT / PyTorch / NCNN + GPU info
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	out, runErr := p.run(ctx, pythonCmd, backendScript, "--list_backends")

	lines := strings.Split(string(out), "\n")
	hasTensorRT := false
	hasPyTorch := false
	hasGPU := false
	hasFP16Line := false

	if len(out) > 0 {
		for _, line := range lines {
			line = strings.TrimSpace(line)

			if strings.HasPrefix(line, "RVE Backend Version:") {
				v := strings.TrimPrefix(line, "RVE Backend Version: ")
				items = append(items, CheckItem{Name: "Backend version", Pass: true, Detail: v})
			} else if strings.HasPrefix(line, "TensorRT Version:") {
				hasTensorRT = true
				v := strings.TrimPrefix(line, "TensorRT Version: ")
				items = append(items, CheckItem{Name: "TensorRT backend", Pass: true, Detail: v})
			} else if strings.Contains(line, "Cannot use tensorrt") && !hasTensorRT {
				hasFP16Line = false // irrelevant without TRT
				errMsg := line
				if idx := strings.Index(errMsg, ":"); idx >= 0 {
					errMsg = errMsg[idx+1:]
				}
				items = append(items, CheckItem{Name: "TensorRT backend", Pass: false, Error: strings.TrimSpace(errMsg)})
			} else if strings.HasPrefix(line, "PyTorch Version:") {
				hasPyTorch = true
				v := strings.TrimPrefix(line, "PyTorch Version: ")
				items = append(items, CheckItem{Name: "PyTorch backend", Pass: true, Detail: v})
			} else if strings.HasPrefix(line, "PyTorch GPU 0:") {
				hasGPU = true
				gpu := strings.TrimPrefix(line, "PyTorch GPU 0: ")
				items = append(items, CheckItem{Name: "GPU detected", Pass: true, Detail: gpu})
			} else if strings.HasPrefix(line, "NCNN Version:") {
				v := strings.TrimPrefix(line, "NCNN Version: ")
				items = append(items, CheckItem{Name: "NCNN backend", Pass: true, Detail: v})
			} else if strings.Contains(line, "Half precision support:") {
				hasFP16Line = true
				if strings.Contains(line, ": True") || strings.Contains(line, ":true") {
					items = append(items, CheckItem{Name: "FP16 / Half-precision", Pass: true, Detail: "Supported"})
				} else if strings.Contains(line, ": False") || strings.Contains(line, ":false") {
					items = append(items, CheckItem{Name: "FP16 / Half-precision", Pass: false, Error: "Not supported on this GPU (requires RTX 20+)"})
				}
			}
		}

		// Add missing components if --list-backends ran but didn't report them
		if runErr == nil { // command succeeded at least partially
			if !hasTensorRT && len(items) > 0 {
				items = append(items, CheckItem{Name: "TensorRT backend", Pass: false, Error: "Not installed"})
			}
			if hasPyTorch && !hasGPU {
				items = append(items, CheckItem{Name: "GPU detected", Pass: false, Error: "No GPU found by PyTorch (CPU-only mode)"})
			}
			if !hasFP16Line {
				// FP16 status unknown — only add if we got partial output without it
				items = append(items, CheckItem{Name: "FP16 / Half-precision", Pass: false, Error: "Status not reported"})
			}
		}

		if !hasPyTorch && runErr == nil {
			items = append(items, CheckItem{Name: "PyTorch backend", Pass: false, Error: "Not detected in --list-backends output"})
		}
	} else if runErr != nil {
		// Command failed completely — report it
		items = append(items, CheckItem{
			Name:   "Backend diagnostics (--list-backends)",
			Pass:   false,
			Error:  fmt.Sprintf("Command failed: %v", runErr),
		})
	} else {
		// ran but empty output — something is wrong with the script
		items = append(items, CheckItem{
			Name:   "Backend diagnostics (--list-backends)",
			Pass:   false,
			Error:  "--list-backends returned no output",
		})
	}

	passCount := 0
	for _, item := range items {
		if item.Pass {
			passCount++
		}
	}

	status := "ok"
	for _, item := range items {
		if !item.Pass {
			status = "fail"
			break
		}
	}

	return BackendCheckResult{
		Status:      status,
		Timestamp:   timestamp,
		Items:       items,
		PassedCount: passCount,
		TotalCount:  len(items),
	}
}
