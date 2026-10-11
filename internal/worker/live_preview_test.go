package worker

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func TestNativeLivePreviewTransport(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 unavailable")
	}
	dir := t.TempDir()
	script := filepath.Join(dir, "preview.py")
	// Oversized transport payload exercises the real subprocess scanner.
	code := "import base64, sys\nprint('<PREVIEW>' + base64.b64encode(b'x' * (2 * 1024 * 1024)).decode(), file=sys.stderr, flush=True)\nprint('<PROGRESS>{\"done\":1,\"total\":1,\"throughput\":1,\"stage\":\"finalizing\"}', file=sys.stderr, flush=True)\n"
	if err := os.WriteFile(script, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	input := filepath.Join(dir, "input.mp4")
	if err := os.WriteFile(input, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	m := NewManager(Config{Python: python, BackendScript: script})
	job, err := m.Start(Request{Input: input, Output: filepath.Join(dir, "out.mp4")})
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		got, _ := m.Get(job.ID)
		if got.Status == "done" {
			if len(got.LivePreview) != 2*1024*1024 || got.ProgressDone != 1 {
				t.Fatalf("native preview truncated: bytes=%d progress=%d logs=%v", len(got.LivePreview), got.ProgressDone, got.Logs)
			}
			return
		}
		if got.Status == "failed" || time.Now().After(deadline) {
			t.Fatalf("worker did not finish: status=%s logs=%v", got.Status, got.Logs)
		}
		time.Sleep(10 * time.Millisecond)
	}
}
