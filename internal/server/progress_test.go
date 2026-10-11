package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/darksidewalker/da-si-wa-true-video-enhancer/internal/worker"
)

// Real subprocess/HTTP contract test. Synthetic progress producer, not inference.
func TestJobListProgressContract(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 unavailable")
	}
	dir := t.TempDir()
	script := filepath.Join(dir, "progress.py")
	code := `import sys
for payload in ['{"done":0,"total":0,"throughput":0,"stage":"preparing"}', '{"done":10,"total":30,"throughput":2,"stage":"rendering"}', '{"done":30,"total":30,"throughput":3,"stage":"finalizing"}']:
    print('<PROGRESS>' + payload, file=sys.stderr, flush=True)
`
	if err := os.WriteFile(script, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	input := filepath.Join(dir, "input.mp4")
	if err := os.WriteFile(input, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	m := worker.NewManager(worker.Config{Python: python, BackendScript: script})
	job, err := m.Start(worker.Request{Input: input, Output: filepath.Join(dir, "out.mp4")})
	if err != nil {
		t.Fatal(err)
	}
	s := New(Config{Jobs: m, Web: os.DirFS(dir)})
	deadline := time.Now().Add(5 * time.Second)
	for {
		w := httptest.NewRecorder()
		s.Routes().ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/jobs", nil))
		var body struct {
			Jobs []worker.JobSummary `json:"jobs"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
			t.Fatal(err)
		}
		if len(body.Jobs) == 1 && body.Jobs[0].Status == "done" {
			p := body.Jobs[0]
			if p.ID != job.ID || p.ProgressDone != 30 || p.ProgressTotal != 30 || p.ProgressThroughput != 3 || p.ProgressStage != "finalizing" || p.ProgressUpdatedAt.IsZero() {
				t.Fatalf("list contract: %s", w.Body.String())
			}
			var raw struct {
				Jobs []map[string]any `json:"jobs"`
			}
			if err := json.Unmarshal(w.Body.Bytes(), &raw); err != nil {
				t.Fatal(err)
			}
			if _, ok := raw.Jobs[0]["logs"]; ok {
				t.Fatal("list shipped logs instead of summary")
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("worker did not finish: %s", w.Body.String())
		}
		time.Sleep(10 * time.Millisecond)
	}
}
