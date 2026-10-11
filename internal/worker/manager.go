package worker

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Config struct {
	BackendScript string
	Python        string
	VenvDir       string // Path to virtual environment directory (e.g., "runtime/venv")
}

type Manager struct {
	cfg         Config
	mu          sync.Mutex
	jobs        map[string]*Job
	subscribers map[string]map[chan Event]struct{}
	queue       chan queuedJob
	stopQueue   chan struct{}
	queueDone   chan struct{}
	closing     bool
	runJob      func(context.Context, string, []string)
}

type queuedJob struct {
	ctx  context.Context
	id   string
	args []string
}

const maxStoredLogLines = 800
const _PREVIEW_MARKER = "<PREVIEW>"
const _VIDEO_CODEC_MARKER = "<VIDEO_CODEC>"
const _OUTPUT_PATH_MARKER = "<OUTPUT_PATH>"

type Event struct {
	Type string `json:"type"`
	Job  *Job   `json:"job"`
	Line string `json:"line,omitempty"`
}

type Job struct {
	ID                 string    `json:"id"`
	Input              string    `json:"input"`
	Output             string    `json:"output"`
	Status             string    `json:"status"`
	StartedAt          time.Time `json:"started_at"`
	EndedAt            time.Time `json:"ended_at,omitempty"`
	Args               []string  `json:"args"`
	Logs               []string  `json:"logs"`
	Error              string    `json:"error,omitempty"`
	LivePreview        []byte    `json:"-"`                     // Base64-decoded JPEG frame (in-memory only)
	VideoCodec         string    `json:"video_codec,omitempty"` // Resolved video codec from backend
	ProgressDone       int       `json:"progress_done,omitempty"`
	ProgressTotal      int       `json:"progress_total,omitempty"`
	ProgressThroughput float64   `json:"progress_throughput,omitempty"`
	ProgressUpdatedAt  time.Time `json:"progress_updated_at,omitempty"`
	ProgressStage      string    `json:"progress_stage,omitempty"`
	structuredProgress bool
	cancel             context.CancelFunc
}

type JobSummary struct {
	ID                 string    `json:"id"`
	Input              string    `json:"input"`
	Output             string    `json:"output"`
	Status             string    `json:"status"`
	StartedAt          time.Time `json:"started_at"`
	EndedAt            time.Time `json:"ended_at,omitempty"`
	Error              string    `json:"error,omitempty"`
	ProgressDone       int       `json:"progress_done,omitempty"`
	ProgressTotal      int       `json:"progress_total,omitempty"`
	ProgressThroughput float64   `json:"progress_throughput,omitempty"`
	ProgressUpdatedAt  time.Time `json:"progress_updated_at,omitempty"`
	ProgressStage      string    `json:"progress_stage,omitempty"`
}

var (
	processedProgressRE = regexp.MustCompile(`(?i)\bprocessed\s+(\d+)/(\d+)\s+frames(?:\s+throughput=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)\s*fps)?`)
	wroteProgressRE     = regexp.MustCompile(`(?i)\bwrote\s+(\d+)/(\d+)\s+frames`)
	outputFramesRE      = regexp.MustCompile(`(?i)\boutput_frames[=≈](\d+)`)
	throughputRE        = regexp.MustCompile(`(?i)throughput=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)\s*fps`)
)

type Request struct {
	ProcessingMode        string  `json:"processing_mode"`
	SpatialBackend        string  `json:"spatial_backend"`
	TemporalBackend       string  `json:"temporal_backend"`
	Precision             string  `json:"precision"`
	Preview               bool    `json:"preview"`
	PreviewSeconds        float64 `json:"preview_seconds"`
	previewSecondsSet     bool
	Input                 string   `json:"input"`
	Output                string   `json:"output"`
	OutputContainer       string   `json:"output_container"`
	Backend               string   `json:"backend"`
	Preset                string   `json:"preset"`
	ContentType           string   `json:"content_type"`
	TargetFPS             float64  `json:"target_fps"`
	Scale                 int      `json:"scale"`
	UpscaleModel          string   `json:"upscale_model"`
	RIFEModel             string   `json:"rife_model"`
	CRF                   string   `json:"crf"`
	VideoEncoderPreset    string   `json:"video_encoder_preset"`
	VideoPixelFormat      string   `json:"video_pixel_format"`
	AudioEncoderPreset    string   `json:"audio_encoder_preset"`
	SubtitleEncoderPreset string   `json:"subtitle_encoder_preset"`
	AudioBitrate          string   `json:"audio_bitrate"`
	TileSize              int      `json:"tile_size"`
	TensorRTDynamicShapes bool     `json:"tensorrt_dynamic_shapes"`
	TensorRTOptProfile    int      `json:"tensorrt_opt_profile"`
	SceneDetectMethod     string   `json:"scene_detect_method"`
	SceneDetectThreshold  float64  `json:"scene_detect_threshold"`
	CustomEncoder         string   `json:"custom_encoder"`
	OverrideUpscaleScale  int      `json:"override_upscale_scale"`
	HDRMode               bool     `json:"hdr_mode"`
	UHDMode               bool     `json:"uhd_mode"`
	SloMoMode             bool     `json:"slomo_mode"`
	Ensemble              bool     `json:"ensemble"`
	DynamicOpticalFlow    bool     `json:"dynamic_optical_flow"`
	Benchmark             bool     `json:"benchmark"`
	StartTm               float64  `json:"start_time"`
	EndTm                 float64  `json:"end_time"`
	Device                string   `json:"device"`
	PytorchGPUID          int      `json:"pytorch_gpu_id"`
	NcnnGPUID             int      `json:"ncnn_gpu_id"`
	ExtraArgs             []string `json:"extra_args"`
	DryRun                bool     `json:"dry_run"`
	Pipeline              string   `json:"pipeline"`
	Denoise               bool     `json:"denoise"`
	DenoiseQuality        string   `json:"denoise_quality"`
	Deblur                bool     `json:"deblur"`
	DeblurQuality         string   `json:"deblur_quality"`
	RtxUpscale            string   `json:"rtx_upscale"`
	RtxUpscaleQuality     string   `json:"rtx_upscale_quality"`
	VsrStrength           float64  `json:"vsr_strength"`
	RtxFrameGeneration    bool     `json:"rtx_frame_generation"`
	RtxFGMode             string   `json:"rtx_fg_mode"`
	RtxAutoShotDetection  *bool    `json:"rtx_auto_shot_detection"`
	RtxShotChange         bool     `json:"rtx_shot_change"`
	RtxImageEncoding      string   `json:"rtx_image_encoding"`
	RtxDenoiseStrength    *float64 `json:"rtx_denoise_strength"`
	RtxDeblurStrength     *float64 `json:"rtx_deblur_strength"`
	RtxUpscaleStrength    *float64 `json:"rtx_upscale_strength"`
	RtxHDR                bool     `json:"rtx_hdr"`
	RtxHDRContrast        *float64 `json:"rtx_hdr_contrast"`
	RtxHDRSaturation      *float64 `json:"rtx_hdr_saturation"`
	RtxHDRMiddleGray      *float64 `json:"rtx_hdr_middle_gray"`
	RtxHDRLuminance       *float64 `json:"rtx_hdr_luminance"`
	RtxHDRDebanding       *bool    `json:"rtx_hdr_debanding"`
}

// BatchRequest enqueues many input videos with one shared settings template.
// OutputDir receives one output file per input, named after the input stem.
type BatchRequest struct {
	Request
	Inputs    []string `json:"inputs"`
	OutputDir string   `json:"output_dir"`
}

// Decode defaults without accepting an explicitly supplied zero preview duration.
func (req *Request) UnmarshalJSON(data []byte) error {
	type plain Request
	value := plain{PreviewSeconds: 5}
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return err
	}
	*req = Request(value)
	_, present := fields["preview_seconds"]
	req.previewSecondsSet = present && req.Preview && req.PreviewSeconds == 0
	return nil
}

// Request has custom JSON defaults; decode the flat batch envelope separately.
func (req *BatchRequest) UnmarshalJSON(data []byte) error {
	var envelope struct {
		Inputs    []string `json:"inputs"`
		OutputDir string   `json:"output_dir"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &req.Request); err != nil {
		return err
	}
	req.Inputs, req.OutputDir = envelope.Inputs, envelope.OutputDir
	return nil
}

// BatchResult reports the enqueued jobs and the inputs that were skipped
// because their output file already exists (resume-friendly batches).
type BatchResult struct {
	Jobs    []*Job   `json:"jobs"`
	Skipped []string `json:"skipped"`
}

func NewManager(cfg Config) *Manager {
	m := &Manager{
		cfg:         cfg,
		jobs:        make(map[string]*Job),
		subscribers: make(map[string]map[chan Event]struct{}),
		queue:       make(chan queuedJob, 256),
		stopQueue:   make(chan struct{}),
		queueDone:   make(chan struct{}),
	}
	m.runJob = m.run
	go m.queueWorker()
	return m
}

func (m *Manager) Start(req Request) (*Job, error) {
	req.Output = applyOutputContainer(req.Output, req.OutputContainer)
	if req.Preview && req.PreviewSeconds == 0 && !req.previewSecondsSet {
		req.PreviewSeconds = 5
	}
	if req.Preview {
		req.Output = previewOutputPath(req.Output, req.PreviewSeconds)
	}
	if err := validateRequest(req); err != nil {
		return nil, err
	}
	jobID := newID()
	args := m.buildArgs(req)

	var ctx context.Context
	var cancel context.CancelFunc
	if !req.DryRun {
		ctx, cancel = context.WithCancel(context.Background())
	}

	job := &Job{
		ID:        jobID,
		Input:     req.Input,
		Output:    req.Output,
		Status:    "queued",
		StartedAt: time.Now(),
		Args:      args,
		cancel:    cancel,
	}

	m.mu.Lock()
	if m.closing {
		m.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		return nil, errors.New("worker manager is shutting down")
	}
	m.jobs[job.ID] = job
	m.mu.Unlock()

	if req.DryRun {
		m.appendLog(job.ID, "Dry run: backend command was built but not executed.")
		m.finish(job.ID, "done", "")
		return job, nil
	}

	m.appendLog(job.ID, "Queued: waiting for the current video job to finish.")
	select {
	case m.queue <- queuedJob{ctx: ctx, id: job.ID, args: args}:
	case <-m.stopQueue:
		_ = m.Cancel(job.ID)
		return nil, errors.New("worker manager is shutting down")
	}
	return job, nil
}

// StartBatch enqueues one job per input video, all sharing the settings of
// the embedded template Request. Outputs are named "<stem>.<container>" inside
// OutputDir. Inputs whose output file already exists are skipped so a batch
// can be resumed after a crash without re-rendering finished videos.
func (m *Manager) StartBatch(req BatchRequest) (*BatchResult, error) {
	if req.Preview {
		if req.PreviewSeconds == 0 && !req.previewSecondsSet {
			req.PreviewSeconds = 5
		}
		if !isFinite(req.PreviewSeconds) || req.PreviewSeconds < 1 || req.PreviewSeconds > 30 {
			return nil, errors.New("preview_seconds must be finite in range 1..30")
		}
	}
	if len(req.Inputs) == 0 {
		return nil, errors.New("inputs are required")
	}
	if req.OutputDir == "" {
		return nil, errors.New("output_dir is required")
	}
	if err := os.MkdirAll(req.OutputDir, 0755); err != nil {
		return nil, fmt.Errorf("output folder cannot be created: %w", err)
	}
	result := &BatchResult{}
	for _, input := range req.Inputs {
		template := req.Request
		template.Input = input
		template.Output = ""
		stem := strings.TrimSuffix(filepath.Base(input), filepath.Ext(input))
		if stem == "" {
			result.Skipped = append(result.Skipped, input)
			continue
		}
		container := template.OutputContainer
		if !isSupportedContainer(container) {
			container = "mp4"
			template.OutputContainer = container
		}
		template.Output = filepath.Join(req.OutputDir, stem+"."+container)
		if template.Preview {
			seconds := template.PreviewSeconds
			if seconds == 0 {
				seconds = 5
			}
			template.Output = previewOutputPath(template.Output, seconds)
		}
		if _, err := os.Stat(template.Output); err == nil {
			result.Skipped = append(result.Skipped, input)
			continue
		}
		job, err := m.Start(template)
		if err != nil {
			result.Skipped = append(result.Skipped, input)
			continue
		}
		result.Jobs = append(result.Jobs, job)
	}
	if len(result.Jobs) == 0 && len(result.Skipped) == 0 {
		return nil, errors.New("no video inputs to enqueue")
	}
	return result, nil
}

func (m *Manager) queueWorker() {
	defer close(m.queueDone)
	for {
		select {
		case <-m.stopQueue:
			return
		case item := <-m.queue:
			if item.ctx == nil || item.ctx.Err() != nil || !m.beginRun(item.id) {
				continue
			}
			m.runJob(item.ctx, item.id, item.args)
		}
	}
}

// Shutdown rejects new work, cancels owned jobs and waits for backend cleanup.
// Closing stopQueue also releases submissions blocked on a full queue.
func (m *Manager) Shutdown(ctx context.Context) error {
	m.mu.Lock()
	var active []string
	if !m.closing {
		m.closing = true
		close(m.stopQueue)
		for id, job := range m.jobs {
			if job.Status == "queued" || job.Status == "running" {
				active = append(active, id)
			}
		}
	}
	m.mu.Unlock()
	for _, id := range active {
		_ = m.Cancel(id)
	}
	select {
	case <-m.queueDone:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *Manager) beginRun(id string) bool {
	var event Event
	m.mu.Lock()
	job := m.jobs[id]
	if m.closing || job == nil || job.Status != "queued" {
		m.mu.Unlock()
		return false
	}
	job.Status = "running"
	job.StartedAt = time.Now()
	event = Event{Type: "status", Job: cloneJob(job)}
	m.mu.Unlock()
	m.broadcast(id, event)
	return true
}

func (m *Manager) Get(id string) (*Job, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	job, ok := m.jobs[id]
	if !ok {
		return nil, false
	}
	return cloneJob(job), true
}

func (m *Manager) List() []*JobSummary {
	m.mu.Lock()
	defer m.mu.Unlock()

	jobs := make([]*JobSummary, 0, len(m.jobs))
	for _, job := range m.jobs {
		jobs = append(jobs, summarizeJob(job))
	}
	sort.Slice(jobs, func(i, j int) bool {
		return jobs[i].ID < jobs[j].ID
	})
	return jobs
}

func (m *Manager) Subscribe(id string) (<-chan Event, func(), bool) {
	m.mu.Lock()
	job, ok := m.jobs[id]
	if !ok {
		m.mu.Unlock()
		return nil, nil, false
	}
	ch := make(chan Event, 32)
	if m.subscribers[id] == nil {
		m.subscribers[id] = make(map[chan Event]struct{})
	}
	m.subscribers[id][ch] = struct{}{}
	initial := Event{Type: "snapshot", Job: cloneJob(job)}
	m.mu.Unlock()

	ch <- initial
	unsubscribe := func() {
		m.mu.Lock()
		defer m.mu.Unlock()
		if subs := m.subscribers[id]; subs != nil {
			delete(subs, ch)
			if len(subs) == 0 {
				delete(m.subscribers, id)
			}
		}
	}
	return ch, unsubscribe, true
}

func (m *Manager) Cancel(id string) error {
	m.mu.Lock()
	job, ok := m.jobs[id]
	if !ok {
		m.mu.Unlock()
		return errors.New("job not found")
	}
	cancel := job.cancel
	status := job.Status
	m.mu.Unlock()

	if status == "done" || status == "error" || status == "cancelled" {
		return nil
	}
	if cancel != nil {
		cancel()
	}
	m.finishIfActive(id, "cancelled", "")
	return nil
}

// venvPythonPath returns the platform-specific virtualenv Python executable.
func venvPythonPath(venvDir string) string {
	if runtime.GOOS == "windows" {
		return filepath.Join(venvDir, "Scripts", "python.exe")
	}
	return filepath.Join(venvDir, "bin", "python")
}

func (m *Manager) run(ctx context.Context, id string, args []string) {

	// Use venv python if available
	pythonPath := m.cfg.Python
	if m.cfg.VenvDir != "" {
		venvPython := venvPythonPath(m.cfg.VenvDir)
		if _, err := os.Stat(venvPython); err == nil {
			pythonPath = venvPython
		}
	}

	cmd := exec.CommandContext(ctx, pythonPath, args...)
	// Cancel only this app's backend. On Windows include its FFmpeg children.
	cmd.WaitDelay = 5 * time.Second
	if runtime.GOOS != "windows" {
		cmd.Cancel = func() error { return cmd.Process.Signal(os.Interrupt) }
	} else {
		cmd.Cancel = func() error {
			killCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			if err := exec.CommandContext(killCtx, "taskkill", "/PID", strconv.Itoa(cmd.Process.Pid), "/T", "/F").Run(); err != nil {
				return cmd.Process.Kill()
			}
			return nil
		}
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		m.finishRunError(ctx, id, err)
		return
	}

	// Separate stderr pipe for preview data parsing
	stderr, err := cmd.StderrPipe()
	if err != nil {
		m.finishRunError(ctx, id, err)
		return
	}

	if err := cmd.Start(); err != nil {
		m.finishRunError(ctx, id, err)
		return
	}

	// Scan stdout and stderr concurrently so live preview and log lines
	// arrive in real time instead of only after the process exits.
	var wg sync.WaitGroup
	wg.Add(2)

	// Scan stdout for logs (final JSON summary etc.)
	go func() {
		defer wg.Done()
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			m.appendLog(id, scanner.Text())
		}
		if err := scanner.Err(); err != nil {
			m.appendLog(id, "log stream error: "+err.Error())
		}
	}()

	// Scan stderr for preview data (<PREVIEW><base64>) and log lines
	go func() {
		defer wg.Done()
		previewScanner := bufio.NewScanner(stderr)
		previewScanner.Buffer(make([]byte, 0, 64*1024), 64*1024*1024) // bounded native-resolution JPEG/base64 transport
		for previewScanner.Scan() {
			line := previewScanner.Text()
			if strings.HasPrefix(line, _PREVIEW_MARKER) {
				base64Data := line[len(_PREVIEW_MARKER):]
				if decoded, err := base64.StdEncoding.DecodeString(base64Data); err == nil {
					m.setLivePreview(id, decoded)
				}
			} else if strings.HasPrefix(line, _VIDEO_CODEC_MARKER) {
				codec := line[len(_VIDEO_CODEC_MARKER):]
				m.setVideoCodec(id, codec)
			} else if strings.HasPrefix(line, _OUTPUT_PATH_MARKER) {
				output := strings.TrimSpace(line[len(_OUTPUT_PATH_MARKER):])
				m.setOutputPath(id, output)
			} else if len(line) > 0 {
				// Non-preview stderr lines are also logged
				m.appendLog(id, line)
			}
		}
	}()

	wg.Wait()
	if err := cmd.Wait(); err != nil {
		m.finishRunError(ctx, id, err)
		return
	}
	if ctx.Err() != nil {
		m.finishIfActive(id, "cancelled", "")
		return
	}
	m.finishIfActive(id, "done", "")
}

func (m *Manager) finishRunError(ctx context.Context, id string, err error) {
	if ctx.Err() != nil {
		m.finishIfActive(id, "cancelled", "")
		return
	}
	m.finishIfActive(id, "error", err.Error())
}

func (m *Manager) buildArgs(req Request) []string {
	backend := req.Backend
	if backend == "" {
		backend = "tensorrt"
	}
	args := []string{m.cfg.BackendScript, "--input", req.Input, "--output", req.Output, "--backend", backend, "--overwrite"}
	if req.Preview {
		seconds := req.PreviewSeconds
		if seconds == 0 {
			seconds = 5
		}
		args = append(args, "--preview_seconds", strconv.FormatFloat(seconds, 'f', -1, 64))
	}
	if req.Scale > 0 {
		args = append(args, "--scale", strconv.Itoa(req.Scale))
	}
	if req.UpscaleModel != "" && ((req.ProcessingMode == "custom" && req.SpatialBackend == "ai") || (req.ProcessingMode != "custom" && req.Pipeline != "nvidia-vfx")) {
		args = append(args, "--upscale_model", req.UpscaleModel)
	}
	if req.RIFEModel != "" && ((req.ProcessingMode == "custom" && req.TemporalBackend == "rife") || (req.ProcessingMode != "custom" && req.Pipeline != "nvidia-vfx")) {
		args = append(args, "--interpolate_model", req.RIFEModel)
	}
	if req.ContentType != "" {
		args = append(args, "--content_type", req.ContentType)
	}
	if req.TargetFPS > 0 {
		args = append(args, "--target_fps", strconv.FormatFloat(req.TargetFPS, 'f', -1, 64))
	}
	args = append(args, presetArgs(req.Preset)...)
	args = append(args, advancedArgs(req)...)
	// Filter out preview_dir and other flags that aren't supported by the Python backend
	filtered := make([]string, 0, len(req.ExtraArgs))
	for i := 0; i < len(req.ExtraArgs); i += 2 {
		if req.ExtraArgs[i] != "--preview_dir" {
			filtered = append(filtered, req.ExtraArgs[i])
			if i+1 < len(req.ExtraArgs) {
				filtered = append(filtered, req.ExtraArgs[i+1])
			}
		}
	}
	args = append(args, filtered...)
	return args
}

func advancedArgs(req Request) []string {
	var args []string
	args = appendStringArg(args, "--precision", req.Precision)
	if req.ProcessingMode == "custom" {
		args = append(args, "--processing_mode", "custom", "--spatial_backend", req.SpatialBackend, "--temporal_backend", req.TemporalBackend)
	}
	args = appendStringArg(args, "--crf", req.CRF)
	args = appendStringArg(args, "--video_encoder_preset", req.VideoEncoderPreset)
	args = appendStringArg(args, "--video_pixel_format", req.VideoPixelFormat)
	args = appendStringArg(args, "--audio_encoder_preset", req.AudioEncoderPreset)
	args = appendStringArg(args, "--subtitle_encoder_preset", req.SubtitleEncoderPreset)
	args = appendStringArg(args, "--audio_bitrate", req.AudioBitrate)
	args = appendIntArg(args, "--tilesize", req.TileSize)
	if req.TensorRTDynamicShapes {
		args = append(args, "--tensorrt_dynamic_shapes")
	}
	args = appendIntArg(args, "--tensorrt_opt_profile", req.TensorRTOptProfile)
	if req.Pipeline != "nvidia-vfx" {
		args = appendStringArg(args, "--scene_detect_method", req.SceneDetectMethod)
		if req.SceneDetectThreshold > 0 {
			args = append(args, "--scene_detect_threshold", strconv.FormatFloat(req.SceneDetectThreshold, 'f', -1, 64))
		}
	}
	args = appendStringArg(args, "--custom_encoder", req.CustomEncoder)
	args = appendIntArg(args, "--override_upscale_scale", req.OverrideUpscaleScale)
	if req.HDRMode && req.Pipeline != "nvidia-vfx" {
		args = append(args, "--hdr_mode")
	}
	if req.UHDMode {
		args = append(args, "--UHD_mode")
	}
	if req.SloMoMode {
		args = append(args, "--slomo_mode")
	}
	if req.Ensemble {
		args = append(args, "--ensemble")
	}
	if req.DynamicOpticalFlow {
		args = append(args, "--dynamic_scaled_optical_flow")
	}
	if req.Benchmark {
		args = append(args, "--benchmark")
	}

	if device := validateDevice(req.Device); device != "" {
		args = append(args, "--device", device)
	}
	args = appendIntArg(args, "--pytorch_gpu_id", req.PytorchGPUID)
	args = appendIntArg(args, "--ncnn_gpu_id", req.NcnnGPUID)
	if pipeline := validatePipeline(req.Pipeline); pipeline != "" && req.ProcessingMode != "custom" {
		args = append(args, "--pipeline", pipeline)
	}
	if req.Denoise {
		args = append(args, "--denoise")
	}
	args = appendStringArg(args, "--denoise_quality", req.DenoiseQuality)
	if req.Deblur {
		args = append(args, "--deblur")
	}
	args = appendStringArg(args, "--deblur_quality", req.DeblurQuality)
	if mode := validateRtxUpscale(req.RtxUpscale); mode != "" && mode != "Off" && (req.ProcessingMode != "custom" || req.SpatialBackend == "nvidia") {
		args = append(args, "--rtx_upscale", mode)
		args = appendStringArg(args, "--rtx_upscale_quality", req.RtxUpscaleQuality)
		if req.VsrStrength > 0 && req.Pipeline != "nvidia-vfx" {
			args = append(args, "--vsr_strength", strconv.FormatFloat(min(req.VsrStrength, 1.0), 'f', -1, 64))
		}
	}
	if req.Pipeline == "nvidia-vfx" || req.ProcessingMode == "custom" {
		args = append(args, nvidiaVFXArgs(req)...)
	}
	return args
}

// Preserve explicit false/zero while giving older request payloads SDK defaults.
func nvidiaVFXArgs(req Request) []string {
	var args []string
	for _, flag := range []struct {
		name    string
		enabled bool
	}{
		{"--rtx_frame_generation", req.RtxFrameGeneration},
		{"--rtx_shot_change", req.RtxShotChange},
		{"--rtx_hdr", req.RtxHDR},
	} {
		if flag.enabled {
			args = append(args, flag.name)
		}
	}
	mode := req.RtxFGMode
	if mode == "" {
		mode = "MEDIUM"
	}
	encoding := req.RtxImageEncoding
	if encoding == "" {
		encoding = "rgb8"
	}
	args = append(args, "--rtx_fg_mode", mode, "--rtx_image_encoding", encoding)
	for _, value := range []struct {
		name  string
		value *bool
	}{
		{"--rtx_auto_shot_detection", req.RtxAutoShotDetection},
		{"--rtx_hdr_debanding", req.RtxHDRDebanding},
	} {
		enabled := true
		if value.value != nil {
			enabled = *value.value
		}
		args = append(args, value.name, strconv.FormatBool(enabled))
	}
	for _, value := range []struct {
		name     string
		value    *float64
		fallback float64
	}{
		{"--rtx_denoise_strength", req.RtxDenoiseStrength, 1},
		{"--rtx_deblur_strength", req.RtxDeblurStrength, 1},
		{"--rtx_upscale_strength", req.RtxUpscaleStrength, 1},
		{"--rtx_hdr_contrast", req.RtxHDRContrast, 100},
		{"--rtx_hdr_saturation", req.RtxHDRSaturation, 100},
		{"--rtx_hdr_middle_gray", req.RtxHDRMiddleGray, 50},
		{"--rtx_hdr_luminance", req.RtxHDRLuminance, 650},
	} {
		number := value.fallback
		if value.value != nil {
			number = *value.value
		}
		args = append(args, value.name, strconv.FormatFloat(number, 'f', -1, 64))
	}
	return args
}

func validatePipeline(pipeline string) string {
	switch pipeline {
	case "upscale", "restore", "restore+upscale", "nvidia-vfx":
		return pipeline
	default:
		return ""
	}
}

func validateRtxUpscale(mode string) string {
	switch mode {
	case "Off", "VSR", "High Bitrate", "Streaming Medium", "Streaming Ultra":
		return mode
	default:
		return ""
	}
}

func validateDevice(device string) string {
	switch strings.ToLower(strings.TrimSpace(device)) {
	case "auto":
		return "auto"
	case "cuda":
		return "cuda"
	case "mps":
		return "mps"
	case "xpu":
		return "xpu"
	default:
		return ""
	}
}

func appendStringArg(args []string, name, value string) []string {
	if value == "" {
		return args
	}
	return append(args, name, value)
}

func appendIntArg(args []string, name string, value int) []string {
	if value <= 0 {
		return args
	}
	return append(args, name, strconv.Itoa(value))
}

func applyOutputContainer(output, container string) string {
	if output == "" || !isSupportedContainer(container) {
		return output
	}
	ext := "." + container
	current := filepath.Ext(output)
	if current == "" {
		return output + ext
	}
	return output[:len(output)-len(current)] + ext
}

func isSupportedContainer(container string) bool {
	switch container {
	case "mp4", "mkv", "webm", "mov", "avi", "flv", "ts", "m4v":
		return true
	default:
		return false
	}
}

func validateRequest(req Request) error {
	if req.ProcessingMode != "" && req.ProcessingMode != "legacy" && req.ProcessingMode != "custom" {
		return errors.New("processing_mode must be legacy or custom")
	}
	if !isFinite(req.TargetFPS) || req.TargetFPS < 0 || req.Scale < 0 || req.OverrideUpscaleScale < 0 {
		return errors.New("target_fps must be finite nonnegative and scales nonnegative")
	}
	if req.Precision != "" && req.Precision != "float16" && req.Precision != "float32" {
		return errors.New("precision must be float16 or float32")
	}
	if req.ProcessingMode == "custom" {
		switch req.SpatialBackend {
		case "none", "ai", "nvidia":
		default:
			return errors.New("spatial_backend must be none, ai or nvidia")
		}
		switch req.TemporalBackend {
		case "none", "rife", "nvidia":
		default:
			return errors.New("temporal_backend must be none, rife or nvidia")
		}
		if req.SpatialBackend == "ai" && req.UpscaleModel == "" {
			return errors.New("AI upscale requires upscale_model")
		}
		if req.TemporalBackend == "rife" && req.RIFEModel == "" {
			return errors.New("RIFE requires rife_model")
		}
		if req.SpatialBackend == "none" && (req.Scale > 1 || req.OverrideUpscaleScale > 1) {
			return errors.New("requested scale requires a spatial backend")
		}
		if req.SpatialBackend == "nvidia" && (validateRtxUpscale(req.RtxUpscale) == "" || req.RtxUpscale == "Off") {
			return errors.New("NVIDIA upscale requires an explicit rtx_upscale mode")
		}
		if req.SpatialBackend != "nvidia" && req.RtxUpscale != "" && req.RtxUpscale != "Off" {
			return errors.New("rtx_upscale conflicts with spatial_backend")
		}
		if req.RtxFrameGeneration && req.TemporalBackend != "nvidia" {
			return errors.New("rtx_frame_generation conflicts with temporal_backend")
		}
		if req.HDRMode || req.CustomEncoder != "" || req.SloMoMode || req.VsrStrength != 0 {
			return errors.New("custom mode does not support legacy hdr_mode/custom_encoder/slomo_mode/vsr_strength")
		}
	}
	if req.Preview && (!isFinite(req.PreviewSeconds) || req.PreviewSeconds < 1 || req.PreviewSeconds > 30) {
		return errors.New("preview_seconds must be finite in range 1..30")
	}
	if req.StartTm != 0 || req.EndTm != 0 {
		return errors.New("start_time/end_time are unsupported; preview renders the first selected seconds")
	}
	for _, arg := range req.ExtraArgs {
		name := strings.SplitN(arg, "=", 2)[0]
		switch name {
		case "--preview_seconds", "--input", "--output", "--start_time", "--end_time", "--processing_mode", "--spatial_backend", "--temporal_backend":
			return fmt.Errorf("extra_args may not override managed flag %s", name)
		}
	}
	if req.Input == "" {
		return errors.New("input is required")
	}
	if req.Output == "" {
		return errors.New("output is required")
	}
	in, _ := filepath.Abs(req.Input)
	out, _ := filepath.Abs(req.Output)
	if in == out {
		return errors.New("output must not overwrite the source")
	}
	if !req.DryRun {
		if _, err := os.Stat(req.Input); err != nil {
			return fmt.Errorf("input is not readable: %w", err)
		}
		if err := os.MkdirAll(filepath.Dir(req.Output), 0755); err != nil {
			return fmt.Errorf("output folder cannot be created: %w", err)
		}
	}
	return nil
}

func presetArgs(preset string) []string {
	switch preset {
	case "best":
		return []string{"--precision", "float16", "--tensorrt_opt_profile", "5"}
	case "fast":
		return []string{"--precision", "float16", "--tensorrt_opt_profile", "2"}
	default:
		return nil // CLI defaults apply; legacy named presets remain accepted.
	}
}

func isFinite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func previewOutputPath(output string, seconds float64) string {
	if output == "" {
		return ""
	}
	ext := filepath.Ext(output)
	stem := strings.TrimSuffix(output, ext)
	stem = regexp.MustCompile(`-preview-[0-9]+(?:\.[0-9]+)?s$`).ReplaceAllString(stem, "")
	return stem + "-preview-" + strconv.FormatFloat(seconds, 'f', -1, 64) + "s" + ext
}

func (m *Manager) setStatus(id, status string) {
	var event Event
	m.mu.Lock()
	if job := m.jobs[id]; job != nil {
		job.Status = status
		event = Event{Type: "status", Job: cloneJob(job)}
	}
	m.mu.Unlock()
	m.broadcast(id, event)
}

func (m *Manager) appendLog(id, line string) {
	var event Event
	m.mu.Lock()
	if job := m.jobs[id]; job != nil {
		updateProgressFromLog(job, line)
		if !strings.HasPrefix(line, "<PROGRESS>") {
			job.Logs = append(job.Logs, line)
		}
		if len(job.Logs) > maxStoredLogLines {
			job.Logs = append([]string(nil), job.Logs[len(job.Logs)-maxStoredLogLines:]...)
		}
		event = Event{Type: "log", Job: cloneJob(job), Line: line}
	}
	m.mu.Unlock()
	m.broadcast(id, event)
}

func (m *Manager) finish(id, status, message string) {
	var event Event
	m.mu.Lock()
	if job := m.jobs[id]; job != nil {
		job.Status = status
		job.EndedAt = time.Now()
		job.Error = message
		event = Event{Type: status, Job: cloneJob(job)}
	}
	m.mu.Unlock()
	m.broadcast(id, event)
}

func (m *Manager) finishIfActive(id, status, message string) {
	var event Event
	m.mu.Lock()
	if job := m.jobs[id]; job != nil {
		if job.Status == "done" || job.Status == "error" || job.Status == "cancelled" {
			m.mu.Unlock()
			return
		}
		job.Status = status
		job.EndedAt = time.Now()
		job.Error = message
		event = Event{Type: status, Job: cloneJob(job)}
	}
	m.mu.Unlock()
	m.broadcast(id, event)
}

func (m *Manager) broadcast(id string, event Event) {
	if event.Type == "" || event.Job == nil {
		return
	}
	m.mu.Lock()
	subs := make([]chan Event, 0, len(m.subscribers[id]))
	for ch := range m.subscribers[id] {
		subs = append(subs, ch)
	}
	m.mu.Unlock()
	for _, ch := range subs {
		select {
		case ch <- event:
		default:
		}
	}
}

func (m *Manager) setLivePreview(id string, data []byte) {
	m.mu.Lock()
	if job := m.jobs[id]; job != nil {
		job.LivePreview = data
	}
	m.mu.Unlock()
}

func (m *Manager) setVideoCodec(id string, codec string) {
	m.mu.Lock()
	if job := m.jobs[id]; job != nil {
		job.VideoCodec = codec
	}
	m.mu.Unlock()
}

func (m *Manager) setOutputPath(id string, output string) {
	if output == "" {
		return
	}
	m.mu.Lock()
	if job := m.jobs[id]; job != nil {
		job.Output = output
	}
	m.mu.Unlock()
}

func updateProgressFromLog(job *Job, line string) {
	updateProgressAt(job, line, time.Now())
}

func updateProgressAt(job *Job, line string, now time.Time) {
	if strings.HasPrefix(line, "<PROGRESS>") {
		var p struct {
			Done       int     `json:"done"`
			Total      int     `json:"total"`
			Throughput float64 `json:"throughput"`
			Stage      string  `json:"stage"`
		}
		if json.Unmarshal([]byte(strings.TrimPrefix(line, "<PROGRESS>")), &p) != nil ||
			p.Done < 0 || p.Total < p.Done || p.Throughput < 0 || math.IsNaN(p.Throughput) || math.IsInf(p.Throughput, 0) ||
			(p.Stage != "preparing" && p.Stage != "rendering" && p.Stage != "finalizing") {
			return
		}
		if !job.structuredProgress || p.Done != job.ProgressDone || p.Stage != job.ProgressStage {
			job.ProgressUpdatedAt = now
		}
		job.ProgressDone, job.ProgressTotal = p.Done, p.Total
		job.ProgressThroughput, job.ProgressStage = p.Throughput, p.Stage
		job.structuredProgress = true
		return
	}
	if job.structuredProgress {
		return
	}
	done, total := job.ProgressDone, job.ProgressTotal
	rate := 0.0
	match := processedProgressRE.FindStringSubmatch(line)
	if match == nil {
		match = wroteProgressRE.FindStringSubmatch(line)
	}
	if match != nil {
		done, _ = strconv.Atoi(match[1])
		total, _ = strconv.Atoi(match[2])
	} else if match = outputFramesRE.FindStringSubmatch(line); match != nil {
		n, _ := strconv.Atoi(match[1])
		// Old native output_frames=N throughput=... means DONE, not TOTAL.
		if throughputRE.MatchString(line) {
			done = n
		} else {
			total = n
		}
	} else {
		return
	}
	if m := throughputRE.FindStringSubmatch(line); m != nil {
		rate, _ = strconv.ParseFloat(m[1], 64)
	} else if done > job.ProgressDone && !job.ProgressUpdatedAt.IsZero() {
		elapsed := now.Sub(job.ProgressUpdatedAt).Seconds()
		if elapsed > 0 {
			rate = float64(done-job.ProgressDone) / elapsed
		}
	}
	if math.IsNaN(rate) || math.IsInf(rate, 0) {
		rate = 0
	}
	if done > job.ProgressDone {
		if rate > 0 {
			if job.ProgressThroughput > 0 {
				rate = 0.35*rate + 0.65*job.ProgressThroughput
			}
			job.ProgressThroughput = rate
		}
		job.ProgressUpdatedAt = now
		job.ProgressStage = "rendering"
	}
	job.ProgressDone, job.ProgressTotal = done, total
	if total > 0 && done >= total {
		job.ProgressStage = "finalizing"
	}
}

func summarizeJob(job *Job) *JobSummary {
	return &JobSummary{
		ID:                 job.ID,
		Input:              job.Input,
		Output:             job.Output,
		Status:             job.Status,
		StartedAt:          job.StartedAt,
		EndedAt:            job.EndedAt,
		Error:              job.Error,
		ProgressDone:       job.ProgressDone,
		ProgressTotal:      job.ProgressTotal,
		ProgressThroughput: job.ProgressThroughput,
		ProgressUpdatedAt:  job.ProgressUpdatedAt,
		ProgressStage:      job.ProgressStage,
	}
}

func cloneJob(job *Job) *Job {
	copy := *job
	copy.Logs = append([]string(nil), job.Logs...)
	return &copy
}

func newID() string {
	return strconv.FormatInt(time.Now().UnixNano(), 36)
}
