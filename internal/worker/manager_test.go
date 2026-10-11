package worker

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func makeTestRequest(t *testing.T, dir, name string) Request {
	t.Helper()

	input := filepath.Join(dir, name)
	if err := os.WriteFile(input, []byte("test"), 0644); err != nil {
		t.Fatalf("write input: %v", err)
	}
	return Request{
		Input:  input,
		Output: filepath.Join(dir, "out", name),
	}
}

func waitForStatus(t *testing.T, m *Manager, id string, wanted ...string) *Job {
	t.Helper()

	allowed := make(map[string]bool, len(wanted))
	for _, status := range wanted {
		allowed[status] = true
	}

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if job, ok := m.Get(id); ok && allowed[job.Status] {
			return job
		}
		time.Sleep(5 * time.Millisecond)
	}
	job, _ := m.Get(id)
	t.Fatalf("job %s did not reach %v; final state: %#v", id, wanted, job)
	return nil
}

func TestBoundedPreviewContract(t *testing.T) {
	m := NewManager(Config{BackendScript: "backend.py"})
	for _, pipeline := range []string{"upscale", "nvidia-vfx", "restore+upscale"} {
		job, err := m.Start(Request{Input: "source.mp4", Output: "full.mp4", Preview: true, PreviewSeconds: 1.5, DryRun: true, Pipeline: pipeline})
		if err != nil {
			t.Fatal(err)
		}
		if job.Output != "full-preview-1.5s.mp4" || !strings.Contains(strings.Join(job.Args, " "), "--preview_seconds 1.5") {
			t.Fatalf("preview contract: %#v", job)
		}
	}
	job, err := m.Start(Request{Input: "source.mp4", Output: "full.mp4", Preview: true, DryRun: true})
	if err != nil || job.Output != "full-preview-5s.mp4" {
		t.Fatalf("default: %#v %v", job, err)
	}
	job, err = m.Start(Request{Input: "source.mp4", Output: "full.mp4", PreviewSeconds: 12, DryRun: true})
	if err != nil || job.Output != "full.mp4" || strings.Contains(strings.Join(job.Args, " "), "--preview_seconds") {
		t.Fatalf("full: %#v %v", job, err)
	}
	for _, seconds := range []float64{-1, .5, 31} {
		if _, err := m.Start(Request{Input: "source.mp4", Output: "full.mp4", Preview: true, PreviewSeconds: seconds, DryRun: true}); err == nil {
			t.Fatalf("accepted %v", seconds)
		}
	}
}

func TestPreviewJSONDefaultsAndExplicitZero(t *testing.T) {
	m := NewManager(Config{})
	for _, body := range []string{`{"input":"in.mp4","output":"out.mp4","preview":true,"dry_run":true}`, `{"input":"in.mp4","output":"out.mp4","preview":true,"preview_seconds":0,"dry_run":true}`} {
		var req Request
		if err := json.Unmarshal([]byte(body), &req); err != nil {
			t.Fatal(err)
		}
		_, err := m.Start(req)
		if strings.Contains(body, `"preview_seconds":0`) && err == nil {
			t.Fatal("explicit zero accepted")
		}
		if !strings.Contains(body, `"preview_seconds":0`) && err != nil {
			t.Fatal(err)
		}
	}
	var batch BatchRequest
	if err := json.Unmarshal([]byte(`{"inputs":["a.mp4"],"output_dir":"out","preview":true,"preview_seconds":1,"rtx_hdr_contrast":0,"rtx_auto_shot_detection":false}`), &batch); err != nil {
		t.Fatal(err)
	}
	if len(batch.Inputs) != 1 || batch.OutputDir != "out" || batch.PreviewSeconds != 1 || batch.RtxHDRContrast == nil || *batch.RtxHDRContrast != 0 || batch.RtxAutoShotDetection == nil || *batch.RtxAutoShotDetection {
		t.Fatalf("flat batch lost fields: %#v", batch)
	}
}

func TestBatchPreviewDoesNotSkipFullOutput(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "source.mp4")
	if err := os.WriteFile(input, []byte("source"), 0600); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "out")
	if err := os.MkdirAll(out, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "source.mp4"), []byte("full"), 0600); err != nil {
		t.Fatal(err)
	}
	m := NewManager(Config{})
	req := BatchRequest{Request: Request{Preview: true, PreviewSeconds: 1, DryRun: true}, Inputs: []string{input}, OutputDir: out}
	result, err := m.StartBatch(req)
	if err != nil || len(result.Jobs) != 1 || len(result.Skipped) != 0 {
		t.Fatalf("%#v %v", result, err)
	}
	if err := os.WriteFile(filepath.Join(out, "source-preview-1s.mp4"), []byte("preview"), 0600); err != nil {
		t.Fatal(err)
	}
	result, err = m.StartBatch(req)
	if err != nil || len(result.Jobs) != 0 || len(result.Skipped) != 1 {
		t.Fatalf("%#v %v", result, err)
	}
}

func TestQueueRunsSequentiallyInFIFOOrder(t *testing.T) {
	dir := t.TempDir()
	m := NewManager(Config{})

	var mu sync.Mutex
	running := 0
	maxRunning := 0
	var order []string

	m.runJob = func(_ context.Context, id string, _ []string) {
		job, ok := m.Get(id)
		if !ok {
			t.Errorf("job %s disappeared", id)
			return
		}

		mu.Lock()
		running++
		if running > maxRunning {
			maxRunning = running
		}
		order = append(order, filepath.Base(job.Input))
		mu.Unlock()

		time.Sleep(25 * time.Millisecond)

		mu.Lock()
		running--
		mu.Unlock()
		m.finishIfActive(id, "done", "")
	}

	names := []string{"01.mp4", "02.mp4", "03.mp4"}
	jobs := make([]*Job, 0, len(names))
	for _, name := range names {
		job, err := m.Start(makeTestRequest(t, dir, name))
		if err != nil {
			t.Fatalf("Start(%s): %v", name, err)
		}
		jobs = append(jobs, job)
	}

	for _, job := range jobs {
		waitForStatus(t, m, job.ID, "done")
	}

	mu.Lock()
	defer mu.Unlock()
	if maxRunning != 1 {
		t.Fatalf("max concurrent jobs = %d, want 1", maxRunning)
	}
	if !reflect.DeepEqual(order, names) {
		t.Fatalf("run order = %v, want %v", order, names)
	}
}

func TestCancelledQueuedJobNeverRuns(t *testing.T) {
	dir := t.TempDir()
	m := NewManager(Config{})

	firstStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	var once sync.Once
	var mu sync.Mutex
	var ran []string

	m.runJob = func(_ context.Context, id string, _ []string) {
		job, _ := m.Get(id)
		name := filepath.Base(job.Input)

		mu.Lock()
		ran = append(ran, name)
		mu.Unlock()

		if name == "first.mp4" {
			once.Do(func() { close(firstStarted) })
			<-releaseFirst
		}
		m.finishIfActive(id, "done", "")
	}

	first, err := m.Start(makeTestRequest(t, dir, "first.mp4"))
	if err != nil {
		t.Fatal(err)
	}
	<-firstStarted

	second, err := m.Start(makeTestRequest(t, dir, "second.mp4"))
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Cancel(second.ID); err != nil {
		t.Fatalf("Cancel: %v", err)
	}

	close(releaseFirst)
	waitForStatus(t, m, first.ID, "done")
	waitForStatus(t, m, second.ID, "cancelled")

	time.Sleep(25 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()
	if !reflect.DeepEqual(ran, []string{"first.mp4"}) {
		t.Fatalf("executed jobs = %v, queued cancelled job must not run", ran)
	}
}

func TestErrorDoesNotBlockFollowingJob(t *testing.T) {
	dir := t.TempDir()
	m := NewManager(Config{})

	var mu sync.Mutex
	var ran []string

	m.runJob = func(_ context.Context, id string, _ []string) {
		job, _ := m.Get(id)
		name := filepath.Base(job.Input)

		mu.Lock()
		ran = append(ran, name)
		mu.Unlock()

		if name == "bad.mp4" {
			m.finishIfActive(id, "error", "synthetic failure")
			return
		}
		m.finishIfActive(id, "done", "")
	}

	bad, err := m.Start(makeTestRequest(t, dir, "bad.mp4"))
	if err != nil {
		t.Fatal(err)
	}
	good, err := m.Start(makeTestRequest(t, dir, "good.mp4"))
	if err != nil {
		t.Fatal(err)
	}

	waitForStatus(t, m, bad.ID, "error")
	waitForStatus(t, m, good.ID, "done")

	mu.Lock()
	defer mu.Unlock()
	if !reflect.DeepEqual(ran, []string{"bad.mp4", "good.mp4"}) {
		t.Fatalf("run order = %v", ran)
	}
}

func TestSetOutputPathUpdatesJob(t *testing.T) {
	m := NewManager(Config{})
	m.jobs["job"] = &Job{ID: "job", Output: "/tmp/video[auto].mp4"}

	m.setOutputPath("job", "/tmp/video[libx265].mp4")

	job, ok := m.Get("job")
	if !ok {
		t.Fatal("job not found")
	}
	if job.Output != "/tmp/video[libx265].mp4" {
		t.Fatalf("output = %q, want resolved codec path", job.Output)
	}
}

func TestProgressParsing(t *testing.T) {
	job := &Job{}

	updateProgressFromLog(job, "[PIPE] processed 180/302 frames throughput=0.15 fps")
	if job.ProgressDone != 180 || job.ProgressTotal != 302 {
		t.Fatalf("progress = %d/%d, want 180/302", job.ProgressDone, job.ProgressTotal)
	}
	if job.ProgressThroughput != 0.15 {
		t.Fatalf("throughput = %v, want 0.15", job.ProgressThroughput)
	}

	updateProgressFromLog(job, "wrote 200/302 frames")
	if job.ProgressDone != 200 || job.ProgressTotal != 302 {
		t.Fatalf("wrote progress = %d/%d, want 200/302", job.ProgressDone, job.ProgressTotal)
	}

	updateProgressFromLog(job, "output_frames=400")
	if job.ProgressTotal != 400 {
		t.Fatalf("output frame total = %d, want 400", job.ProgressTotal)
	}
}

func TestBatchEnqueuesSequentialJobsWithDerivedOutputs(t *testing.T) {
	dir := t.TempDir()
	m := NewManager(Config{})

	var mu sync.Mutex
	running := 0
	maxRunning := 0

	m.runJob = func(_ context.Context, id string, _ []string) {
		mu.Lock()
		running++
		if running > maxRunning {
			maxRunning = running
		}
		running--
		mu.Unlock()
		m.finishIfActive(id, "done", "")
	}

	inputs := []string{
		filepath.Join(dir, "a.mp4"),
		filepath.Join(dir, "b.mkv"),
		filepath.Join(dir, "c.mp4"),
	}
	for _, path := range inputs {
		if err := os.WriteFile(path, []byte("test"), 0644); err != nil {
			t.Fatalf("write input: %v", err)
		}
	}
	outDir := filepath.Join(dir, "out")

	result, err := m.StartBatch(BatchRequest{
		Request:   Request{OutputContainer: "mp4"},
		Inputs:    inputs,
		OutputDir: outDir,
	})
	if err != nil {
		t.Fatalf("StartBatch: %v", err)
	}
	if len(result.Jobs) != 3 {
		t.Fatalf("jobs = %d, want 3", len(result.Jobs))
	}
	if len(result.Skipped) != 0 {
		t.Fatalf("skipped = %v, want none", result.Skipped)
	}
	want := []string{
		filepath.Join(outDir, "a.mp4"),
		filepath.Join(outDir, "b.mp4"),
		filepath.Join(outDir, "c.mp4"),
	}
	for i, job := range result.Jobs {
		if job.Output != want[i] {
			t.Fatalf("job %d output = %s, want %s", i, job.Output, want[i])
		}
	}

	for _, job := range result.Jobs {
		waitForStatus(t, m, job.ID, "done")
	}
	mu.Lock()
	defer mu.Unlock()
	if maxRunning != 1 {
		t.Fatalf("max concurrent jobs = %d, want 1", maxRunning)
	}
}

func TestBatchSkipsExistingOutputs(t *testing.T) {
	dir := t.TempDir()
	m := NewManager(Config{})
	m.runJob = func(_ context.Context, id string, _ []string) {
		m.finishIfActive(id, "done", "")
	}

	for _, name := range []string{"a.mp4", "b.mp4"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("test"), 0644); err != nil {
			t.Fatalf("write input: %v", err)
		}
	}
	outDir := filepath.Join(dir, "out")
	if err := os.MkdirAll(outDir, 0755); err != nil {
		t.Fatalf("mkdir out: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outDir, "a.mp4"), []byte("done"), 0644); err != nil {
		t.Fatalf("write existing output: %v", err)
	}

	result, err := m.StartBatch(BatchRequest{
		Request:   Request{OutputContainer: "mp4"},
		Inputs:    []string{filepath.Join(dir, "a.mp4"), filepath.Join(dir, "b.mp4")},
		OutputDir: outDir,
	})
	if err != nil {
		t.Fatalf("StartBatch: %v", err)
	}
	if len(result.Jobs) != 1 {
		t.Fatalf("jobs = %d, want 1", len(result.Jobs))
	}
	if len(result.Skipped) != 1 || result.Skipped[0] != filepath.Join(dir, "a.mp4") {
		t.Fatalf("skipped = %v, want a.mp4", result.Skipped)
	}
}

func TestRestoreAndRtxArgs(t *testing.T) {
	req := Request{
		Input:             "in.mp4",
		Output:            "out.mp4",
		Pipeline:          "restore",
		Denoise:           true,
		DenoiseQuality:    "High",
		Deblur:            true,
		DeblurQuality:     "Ultra",
		RtxUpscale:        "Streaming Ultra",
		RtxUpscaleQuality: "Medium",
		VsrStrength:       0.7,
	}
	args := strings.Join(advancedArgs(req), " ")
	for _, want := range []string{
		"--pipeline restore", "--denoise", "--denoise_quality High",
		"--deblur", "--deblur_quality Ultra",
		"--rtx_upscale Streaming Ultra", "--rtx_upscale_quality Medium",
		"--vsr_strength 0.7",
	} {
		if !strings.Contains(args, want) {
			t.Fatalf("args %q missing %q", args, want)
		}
	}

	// Off mode and invalid values must not emit flags.
	args = strings.Join(advancedArgs(Request{RtxUpscale: "Off", Pipeline: "bogus"}), " ")
	if strings.Contains(args, "--rtx_upscale") || strings.Contains(args, "--pipeline") {
		t.Fatalf("args %q must not contain rtx_upscale/pipeline", args)
	}
}

func TestNvidiaVFXDefaultsAndLegacyIsolation(t *testing.T) {
	want := []string{
		"--pipeline", "nvidia-vfx", "--rtx_fg_mode", "MEDIUM", "--rtx_image_encoding", "rgb8",
		"--rtx_auto_shot_detection", "true", "--rtx_hdr_debanding", "true",
		"--rtx_denoise_strength", "1", "--rtx_deblur_strength", "1", "--rtx_upscale_strength", "1",
		"--rtx_hdr_contrast", "100", "--rtx_hdr_saturation", "100", "--rtx_hdr_middle_gray", "50", "--rtx_hdr_luminance", "650",
	}
	if got := advancedArgs(Request{Pipeline: "nvidia-vfx"}); !reflect.DeepEqual(got, want) {
		t.Fatalf("default argv = %v, want %v", got, want)
	}
	for _, pipeline := range []string{"", "upscale", "restore", "restore+upscale"} {
		base := Request{Pipeline: pipeline, HDRMode: true, SceneDetectMethod: "mean", SceneDetectThreshold: 4}
		withNative := base
		withNative.RtxFrameGeneration, withNative.RtxHDR, withNative.RtxShotChange = true, true, true
		withNative.RtxFGMode, withNative.RtxImageEncoding = "HIGH", "rgb10a2"
		if !reflect.DeepEqual(advancedArgs(base), advancedArgs(withNative)) {
			t.Fatalf("native settings changed legacy pipeline %q", pipeline)
		}
	}
}

func TestNvidiaVFXJSONRoundTripSingleAndBatch(t *testing.T) {
	payload := []byte(`{"pipeline":"nvidia-vfx","rtx_frame_generation":true,"rtx_fg_mode":"HIGH","rtx_auto_shot_detection":false,"rtx_shot_change":true,"rtx_image_encoding":"rgb10a2","rtx_denoise_strength":0,"rtx_deblur_strength":0,"rtx_upscale_strength":0,"rtx_hdr":true,"rtx_hdr_contrast":0,"rtx_hdr_saturation":125,"rtx_hdr_middle_gray":40,"rtx_hdr_luminance":1000,"rtx_hdr_debanding":false,"denoise":true,"denoise_quality":"High","deblur":true,"deblur_quality":"Medium","rtx_upscale":"Streaming Ultra","rtx_upscale_quality":"Low","vsr_strength":0.7,"hdr_mode":true,"scene_detect_method":"mean","scene_detect_threshold":9,"rife_model":"stale-rife","upscale_model":"stale-ai","target_fps":59.94,"dry_run":true}`)
	var req Request
	if err := json.Unmarshal(payload, &req); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip Request
	if err := json.Unmarshal(encoded, &roundTrip); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(req, roundTrip) {
		t.Fatal("request JSON roundtrip changed native settings")
	}
	dir := t.TempDir()
	file := makeTestRequest(t, dir, "native.mp4")
	req.Input, req.Output = file.Input, file.Output
	m := NewManager(Config{BackendScript: "backend.py"})
	single, err := m.Start(req)
	if err != nil {
		t.Fatal(err)
	}
	// Decode the actual flat batch JSON surface, not a nested request object.
	var batch BatchRequest
	if err := json.Unmarshal(payload, &batch); err != nil {
		t.Fatal(err)
	}
	batch.Inputs, batch.OutputDir = []string{file.Input}, filepath.Join(dir, "batch")
	result, err := m.StartBatch(batch)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Jobs) != 1 || len(result.Skipped) != 0 {
		t.Fatalf("unexpected batch: %#v", result)
	}
	for _, job := range []*Job{single, result.Jobs[0]} {
		args := strings.Join(job.Args, " ")
		for _, want := range []string{"--pipeline nvidia-vfx", "--rtx_frame_generation", "--rtx_fg_mode HIGH", "--rtx_auto_shot_detection false", "--rtx_shot_change", "--rtx_image_encoding rgb10a2", "--rtx_denoise_strength 0", "--rtx_deblur_strength 0", "--rtx_upscale_strength 0", "--rtx_hdr ", "--rtx_hdr_contrast 0", "--rtx_hdr_saturation 125", "--rtx_hdr_middle_gray 40", "--rtx_hdr_luminance 1000", "--rtx_hdr_debanding false", "--denoise_quality High", "--deblur_quality Medium", "--rtx_upscale Streaming Ultra", "--rtx_upscale_quality Low", "--target_fps 59.94"} {
			if !strings.Contains(args, want) {
				t.Fatalf("argv %q missing %q", args, want)
			}
		}
		for _, forbidden := range []string{"--interpolate_model", "--upscale_model", "--hdr_mode", "--scene_detect_method", "--scene_detect_threshold", "--vsr_strength"} {
			if strings.Contains(args, forbidden) {
				t.Fatalf("argv retained stale option %s: %s", forbidden, args)
			}
		}
	}
}

func TestNvidiaVFXFrameGenerationModes(t *testing.T) {
	for _, mode := range []string{"LOW", "MEDIUM", "HIGH"} {
		for _, encoding := range []string{"rgb8", "rgb10a2"} {
			req := Request{Pipeline: "nvidia-vfx", RtxFGMode: mode, RtxImageEncoding: encoding, RIFEModel: "rife"}
			m := &Manager{}
			args := strings.Join(m.buildArgs(req), " ")
			if !strings.Contains(args, "--rtx_fg_mode "+mode) || !strings.Contains(args, "--rtx_image_encoding "+encoding) || strings.Contains(args, "--interpolate_model") {
				t.Fatalf("NVIDIA pipeline must suppress stale RIFE and retain requested modes: %s", args)
			}
			for _, arg := range m.buildArgs(req) {
				if arg == "--rtx_frame_generation" || arg == "--rtx_hdr" || arg == "--rtx_shot_change" {
					t.Fatalf("false flag emitted: %s", arg)
				}
			}
		}
	}
}

func TestBuildArgsRestorePlusUpscalePipeline(t *testing.T) {
	args := strings.Join(advancedArgs(Request{
		Pipeline:   "restore+upscale",
		Denoise:    true,
		RtxUpscale: "Off",
	}), " ")
	for _, want := range []string{"--pipeline restore+upscale", "--denoise"} {
		if !strings.Contains(args, want) {
			t.Fatalf("args %q missing %q", args, want)
		}
	}
	if strings.Contains(args, "--rtx_upscale") {
		t.Fatalf("args %q must not contain rtx_upscale for Off mode", args)
	}
}
