package worker

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
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
