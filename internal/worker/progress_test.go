package worker

import (
	"encoding/json"
	"testing"
	"time"
)

func TestCanonicalProgressSummary(t *testing.T) {
	job := &Job{ID: "progress", Status: "running"}
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	updateProgressAt(job, `<PROGRESS>{"done":0,"total":0,"throughput":0,"stage":"preparing"}`, now)
	updateProgressAt(job, `<PROGRESS>{"done":1,"total":300,"throughput":0,"stage":"rendering"}`, now.Add(10*time.Minute))
	updateProgressAt(job, `<PROGRESS>{"done":2,"total":300,"throughput":1,"stage":"rendering"}`, now.Add(601*time.Second))
	updateProgressAt(job, `output_frames=999 throughput=0.001fps`, now.Add(time.Hour))
	if job.ProgressDone != 2 || job.ProgressTotal != 300 || job.ProgressThroughput != 1 {
		t.Fatalf("canonical progress overwritten: %+v", job)
	}
	before := job.ProgressUpdatedAt
	updateProgressAt(job, `<PROGRESS>{"done":2,"total":300,"throughput":1,"stage":"rendering"}`, now.Add(time.Hour))
	if !job.ProgressUpdatedAt.Equal(before) {
		t.Fatal("unchanged sample hid stalled rendering")
	}
	updateProgressAt(job, `<PROGRESS>{"done":300,"total":300,"throughput":4,"stage":"finalizing"}`, now.Add(time.Hour))
	summary := summarizeJob(job)
	data, err := json.Marshal(summary)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]any
	if err := json.Unmarshal(data, &fields); err != nil {
		t.Fatal(err)
	}
	if fields["progress_done"] != float64(300) || fields["progress_total"] != float64(300) || fields["progress_throughput"] != float64(4) || fields["progress_stage"] != "finalizing" || fields["progress_updated_at"] == nil || fields["status"] != "running" {
		t.Fatalf("summary contract: %s", data)
	}
	for _, line := range []string{
		`<PROGRESS>{"done":-1,"total":300,"throughput":1,"stage":"rendering"}`,
		`<PROGRESS>{"done":400,"total":300,"throughput":1,"stage":"rendering"}`,
		`<PROGRESS>{"done":2,"total":300,"throughput":1e999,"stage":"rendering"}`,
		`<PROGRESS>{"done":2,"total":300,"throughput":1,"stage":"bogus"}`,
	} {
		updateProgressAt(job, line, now)
	}
	if job.ProgressDone != 300 || job.ProgressStage != "finalizing" {
		t.Fatal("invalid progress accepted")
	}
}

func TestLegacyNativeProgressCountsAndRate(t *testing.T) {
	now := time.Now()
	job := &Job{}
	updateProgressAt(job, "streaming output_frames≈300 target_fps=60", now)
	updateProgressAt(job, "output_frames=10 throughput=2.0fps", now.Add(time.Minute))
	if job.ProgressDone != 10 || job.ProgressTotal != 300 || job.ProgressThroughput != 2 {
		t.Fatalf("native done was treated as total: %+v", job)
	}
	updateProgressAt(job, "processed 20/300 frames throughput=4e0 fps", now.Add(65*time.Second))
	if job.ProgressThroughput <= 2 || job.ProgressThroughput >= 4 {
		t.Fatal("legacy rate must adapt with smoothing")
	}
	updateProgressAt(job, "wrote 300/300 frames", now.Add(2*time.Minute))
	if job.ProgressStage != "finalizing" {
		t.Fatal("last frame is not job completion")
	}
}

func TestProgressProtocolDoesNotFillLog(t *testing.T) {
	m := NewManager(Config{})
	m.jobs["p"] = &Job{ID: "p", Status: "running"}
	m.appendLog("p", `<PROGRESS>{"done":10,"total":30,"throughput":2,"stage":"rendering"}`)
	if len(m.jobs["p"].Logs) != 0 || m.jobs["p"].ProgressDone != 10 {
		t.Fatal("progress must update summary without duplicating logs")
	}
}
