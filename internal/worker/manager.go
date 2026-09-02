package worker

import (
	"bufio"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
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
}

var (
	processedProgressRE = regexp.MustCompile(`(?i)\bprocessed\s+(\d+)/(\d+)\s+frames(?:\s+throughput=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)\s*fps)?`)
	wroteProgressRE     = regexp.MustCompile(`(?i)\bwrote\s+(\d+)/(\d+)\s+frames`)
	outputFramesRE      = regexp.MustCompile(`(?i)\boutput_frames=(\d+)`)
)

type Request struct {
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
}

func NewManager(cfg Config) *Manager {
	m := &Manager{
		cfg:         cfg,
		jobs:        make(map[string]*Job),
		subscribers: make(map[string]map[chan Event]struct{}),
		queue:       make(chan queuedJob, 256),
	}
	m.runJob = m.run
	go m.queueWorker()
	return m
}

func (m *Manager) Start(req Request) (*Job, error) {
	req.Output = applyOutputContainer(req.Output, req.OutputContainer)
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
	m.jobs[job.ID] = job
	m.mu.Unlock()

	if req.DryRun {
		m.appendLog(job.ID, "Dry run: backend command was built but not executed.")
		m.finish(job.ID, "done", "")
		return job, nil
	}

	m.appendLog(job.ID, "Queued: waiting for the current video job to finish.")
	m.queue <- queuedJob{ctx: ctx, id: job.ID, args: args}
	return job, nil
}

func (m *Manager) queueWorker() {
	for item := range m.queue {
		if item.ctx == nil || item.ctx.Err() != nil {
			continue
		}
		if !m.beginRun(item.id) {
			continue
		}
		m.runJob(item.ctx, item.id, item.args)
	}
}

func (m *Manager) beginRun(id string) bool {
	var event Event
	m.mu.Lock()
	job := m.jobs[id]
	if job == nil || job.Status != "queued" {
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
		previewScanner.Buffer(make([]byte, 0, 64*1024), 1024*1024) // handle large base64 frames
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
	if req.Scale > 0 {
		args = append(args, "--scale", strconv.Itoa(req.Scale))
	}
	if req.UpscaleModel != "" {
		args = append(args, "--upscale_model", req.UpscaleModel)
	}
	if req.RIFEModel != "" {
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
	args = appendStringArg(args, "--scene_detect_method", req.SceneDetectMethod)
	if req.SceneDetectThreshold > 0 {
		args = append(args, "--scene_detect_threshold", strconv.FormatFloat(req.SceneDetectThreshold, 'f', -1, 64))
	}
	args = appendStringArg(args, "--custom_encoder", req.CustomEncoder)
	args = appendIntArg(args, "--override_upscale_scale", req.OverrideUpscaleScale)
	if req.HDRMode {
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
	if req.StartTm > 0 {
		args = append(args, "--start_time", strconv.FormatFloat(req.StartTm, 'f', -1, 64))
	}
	if req.EndTm > 0 {
		args = append(args, "--end_time", strconv.FormatFloat(req.EndTm, 'f', -1, 64))
	}
	if device := validateDevice(req.Device); device != "" {
		args = append(args, "--device", device)
	}
	args = appendIntArg(args, "--pytorch_gpu_id", req.PytorchGPUID)
	args = appendIntArg(args, "--ncnn_gpu_id", req.NcnnGPUID)
	return args
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
	if req.Input == "" {
		return errors.New("input is required")
	}
	if req.Output == "" {
		return errors.New("output is required")
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
		return []string{"--precision", "float16", "--tensorrt_opt_profile", "3"}
	}
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
		job.Logs = append(job.Logs, line)
		updateProgressFromLog(job, line)
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
	if match := processedProgressRE.FindStringSubmatch(line); match != nil {
		if done, err := strconv.Atoi(match[1]); err == nil {
			job.ProgressDone = done
		}
		if total, err := strconv.Atoi(match[2]); err == nil {
			job.ProgressTotal = total
		}
		if len(match) > 3 && match[3] != "" {
			if throughput, err := strconv.ParseFloat(match[3], 64); err == nil {
				job.ProgressThroughput = throughput
			}
		}
		return
	}

	if match := wroteProgressRE.FindStringSubmatch(line); match != nil {
		if done, err := strconv.Atoi(match[1]); err == nil {
			job.ProgressDone = done
		}
		if total, err := strconv.Atoi(match[2]); err == nil {
			job.ProgressTotal = total
		}
		return
	}

	if match := outputFramesRE.FindStringSubmatch(line); match != nil {
		if total, err := strconv.Atoi(match[1]); err == nil {
			job.ProgressTotal = total
		}
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
