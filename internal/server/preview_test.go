package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/darksidewalker/da-si-wa-true-video-enhancer/internal/worker"
)

func TestProcessedPreviewPreservesResolutionAndCadence(t *testing.T) {
	for _, name := range []string{"ffmpeg", "ffprobe", "python3"} {
		if _, err := exec.LookPath(name); err != nil {
			t.Skip(name + " unavailable")
		}
	}
	dir := t.TempDir()
	output := filepath.Join(dir, "output.mkv")
	cmd := exec.Command("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=960x540:rate=60", "-frames:v", "60", "-c:v", "ffv1", output)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("source: %v: %s", err, out)
	}
	// Only job completion is synthetic; the preview handler, encode and probe are real.
	script := filepath.Join(dir, "done.py")
	if err := os.WriteFile(script, []byte("pass\n"), 0600); err != nil {
		t.Fatal(err)
	}
	m := worker.NewManager(worker.Config{Python: "python3", BackendScript: script})
	input := filepath.Join(dir, "input.mkv")
	if err := os.WriteFile(input, []byte("unused synthetic worker input"), 0600); err != nil {
		t.Fatal(err)
	}
	job, err := m.Start(worker.Request{Input: input, Output: output})
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		current, _ := m.Get(job.ID)
		if current.Status == "done" {
			break
		}
		if current.Status == "failed" || time.Now().After(deadline) {
			t.Fatalf("job: %+v", current)
		}
		time.Sleep(10 * time.Millisecond)
	}
	s := New(Config{Jobs: m, Web: os.DirFS(dir)})
	w := httptest.NewRecorder()
	s.Routes().ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/jobs/"+job.ID+"/preview", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("preview: %d %s", w.Code, w.Body.String())
	}
	proxy := filepath.Join(dir, "proxy.mp4")
	if err := os.WriteFile(proxy, w.Body.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	data, err := exec.Command("ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", proxy).Output()
	if err != nil {
		t.Fatal(err)
	}
	var probe struct {
		Streams []struct {
			Width, Height int
			Rate          string `json:"avg_frame_rate"`
			Frames        string `json:"nb_read_frames"`
		}
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		t.Fatal(err)
	}
	if len(probe.Streams) != 1 || probe.Streams[0].Width != 960 || probe.Streams[0].Height != 540 || probe.Streams[0].Rate != "60/1" || probe.Streams[0].Frames != "60" {
		t.Fatalf("preview degraded output: %s", data)
	}
	// Fine detail must survive the browser transcode, not merely its dimensions.
	ssim, err := exec.Command("ffmpeg", "-hide_banner", "-i", output, "-i", proxy, "-lavfi", "ssim", "-f", "null", "-").CombinedOutput()
	if err != nil {
		t.Fatalf("SSIM: %v: %s", err, ssim)
	}
	match := regexp.MustCompile(`All:([0-9.]+)`).FindSubmatch(ssim)
	if len(match) != 2 {
		t.Fatalf("SSIM missing: %s", ssim)
	}
	quality, err := strconv.ParseFloat(string(match[1]), 64)
	if err != nil || quality < 0.98 {
		t.Fatalf("preview detail loss: SSIM=%v: %s", quality, ssim)
	}
	t.Logf("Native-resolution proxy: %s; SSIM=%f", data, quality)
}

func TestPreviewVideoFilterFFmpeg(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg unavailable")
	}
	for _, transfer := range []string{"bt709", "smpte2084", "arib-std-b67"} {
		t.Run(transfer, func(t *testing.T) {
			inputFilter := "format=yuv420p10le,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709"
			if transfer != "bt709" {
				inputFilter = "format=yuv420p10le,setparams=color_primaries=bt2020:color_trc=" + transfer + ":colorspace=bt2020nc"
			}
			cmd := exec.Command("ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=128x128:rate=12", "-frames:v", "3", "-vf", inputFilter+","+previewVideoFilter(transfer), "-f", "null", "-")
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("preview filter failed: %v: %s", err, out)
			}
		})
	}
}

func TestPreviewVideoFilter(t *testing.T) {
	for _, transfer := range []string{"smpte2084", "arib-std-b67"} {
		got := previewVideoFilter(transfer)
		if !strings.Contains(got, "tonemap=tonemap=mobius") || !strings.HasSuffix(got, "format=yuv420p") {
			t.Fatalf("HDR %s must tone-map browser proxy: %s", transfer, got)
		}
	}
	for _, transfer := range []string{"", "unknown", "bt709"} {
		got := previewVideoFilter(transfer)
		if strings.Contains(got, "tonemap") || !strings.HasSuffix(got, "format=yuv420p") {
			t.Fatalf("SDR %s must not tone-map: %s", transfer, got)
		}
	}
}
