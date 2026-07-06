package server

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/darksidewalker/da-si-wa-true-video-enhancer/internal/runtimecheck"
	"github.com/darksidewalker/da-si-wa-true-video-enhancer/internal/worker"
)

type Config struct {
	Name    string
	Version string
	Web     fs.FS
	Runtime *runtimecheck.Probe
	Jobs    *worker.Manager
	Quit    func()
}

type Server struct{ cfg Config }

func New(cfg Config) *Server { return &Server{cfg: cfg} }

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", s.health)
	mux.HandleFunc("GET /api/options", s.options)
	mux.HandleFunc("GET /api/runtime/status", s.runtimeStatus)
	mux.HandleFunc("POST /api/runtime/check", s.backendCheckHandler)
	mux.HandleFunc("GET /api/runtime/models", s.runtimeModels)
	mux.HandleFunc("GET /api/probe", probeVideo)
	mux.HandleFunc("POST /api/runtime/install", s.installRuntime)
	mux.HandleFunc("GET /api/runtime/install/stream", s.installRuntimeStream)
	mux.HandleFunc("GET /api/models/download/stream", s.downloadModelStream)
	mux.HandleFunc("POST /api/models/download", s.downloadModels)
	mux.HandleFunc("POST /api/quit", s.quit)
	mux.HandleFunc("GET /api/browse", s.browseFiles)
	mux.HandleFunc("GET /api/search-files", s.searchFiles)
	mux.HandleFunc("GET /api/stream", s.streamVideo)
	mux.HandleFunc("POST /api/open-folder", s.openFolder)
	mux.HandleFunc("POST /api/jobs", s.startJob)
	mux.HandleFunc("GET /api/jobs/{id}", s.getJob)
	mux.HandleFunc("GET /api/jobs/{id}/events", s.jobEvents)
	mux.HandleFunc("GET /api/jobs/{id}/live-preview", s.jobLivePreview)
	mux.HandleFunc("GET /api/jobs/{id}/preview", s.jobPreview)
	mux.HandleFunc("POST /api/jobs/{id}/cancel", s.cancelJob)
	mux.Handle("/", http.FileServer(http.FS(s.cfg.Web)))
	return withNoCache(mux)
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"name": s.cfg.Name, "version": s.cfg.Version, "status": "ok"})
}

func (s *Server) runtimeStatus(w http.ResponseWriter, _ *http.Request) {
	status := s.cfg.Runtime.Status()
	writeJSON(w, http.StatusOK, status)
}

// backendCheckHandler runs the Python backend --list-backends and returns structured check results.
func (s *Server) backendCheckHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}

	var req struct {
		RepoRoot string `json:"repo_root"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)

	repo := req.RepoRoot
	if repo == "" {
		repo = "."
	}

	result := s.cfg.Runtime.BackendCheck(repo)
	writeJSON(w, http.StatusOK, result)
}

// installRuntimeStream streams the installation progress via SSE so the UI can show real-time output.
func (s *Server) installRuntimeStream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache, no-store")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher.Flush()

	// Parse optional models from query string
	var req runtimecheck.InstallOptions
	if raw := r.URL.Query().Get("models"); raw != "" {
		json.Unmarshal([]byte(raw), &req)
	}

	plan := s.cfg.Runtime.InstallPlan(".", req.Models...)
	if err := writeSSE(w, "plan", map[string]any{"type": "plan", "plan": plan}); err != nil {
		return
	}
	flusher.Flush()

	errCh := make(chan error, 1)
	go func() {
		errCh <- s.cfg.Runtime.Install(r.Context(), ".", req, func(line string) {
			if serr := writeSSE(w, "log", map[string]any{"type": "log", "line": line}); serr != nil {
				return // client disconnected
			}
			flusher.Flush()
		})
	}()

	select {
	case <-r.Context().Done():
		return
	case err := <-errCh:
		if err != nil {
			_ = writeSSE(w, "error", map[string]any{"type": "error", "error": err.Error()})
		} else {
			_ = writeSSE(w, "done", map[string]any{"type": "done"})
		}
		flusher.Flush()
	}
}

func (s *Server) installRuntime(w http.ResponseWriter, r *http.Request) {
	var req runtimecheck.InstallOptions
	if r.Body != nil {
		_ = json.NewDecoder(r.Body).Decode(&req)
	}
	var logs []string
	err := s.cfg.Runtime.Install(r.Context(), ".", req, func(line string) {
		logs = append(logs, line)
	})
	plan := s.cfg.Runtime.InstallPlan(".", req.Models...)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, runtimecheck.InstallResult{Status: "error", Error: err.Error(), Plan: plan, Logs: append(logs, err.Error())})
		return
	}
	writeJSON(w, http.StatusOK, runtimecheck.InstallResult{Status: "ok", Plan: plan, Logs: logs})
}

func (s *Server) downloadModels(w http.ResponseWriter, r *http.Request) {
	var req runtimecheck.InstallOptions
	if r.Body != nil {
		_ = json.NewDecoder(r.Body).Decode(&req)
	}
	var logs []string
	err := s.cfg.Runtime.DownloadModels(r.Context(), req.Models, func(line string) {
		logs = append(logs, line)
	})
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"status": "error", "error": err.Error(), "logs": logs})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "logs": logs})
}

// downloadModelStream streams model downloads via SSE so the UI shows real-time progress.
func (s *Server) downloadModelStream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache, no-store")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher.Flush()

	var req runtimecheck.InstallOptions
	if raw := r.URL.Query().Get("models"); raw != "" {
		json.Unmarshal([]byte(raw), &req)
	}

	errCh := make(chan error, 1)
	go func() {
		errCh <- s.cfg.Runtime.DownloadModels(r.Context(), req.Models, func(line string) {
			if serr := writeSSE(w, "log", map[string]any{"type": "log", "line": line}); serr != nil {
				return // client disconnected
			}
			flusher.Flush()
		})
	}()

	select {
	case <-r.Context().Done():
		return
	case err := <-errCh:
		if err != nil {
			_ = writeSSE(w, "error", map[string]any{"type": "error", "error": err.Error()})
		} else {
			_ = writeSSE(w, "done", map[string]any{"type": "done"})
		}
		flusher.Flush()
	}
}

func (s *Server) runtimeModels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"models": s.cfg.Runtime.AvailableModels(".")})
}

func (s *Server) options(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"video_encoder_presets":    []string{"auto", "av1_nvenc", "x265_nvenc", "x264_nvenc", "av1", "libx265", "libx264", "vp9", "prores", "ffv1"},
		"output_containers":        []string{"mp4", "mkv", "webm", "mov", "avi", "flv", "ts", "m4v"},
		"video_pixel_formats":      []string{"yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "yuv444p10le"},
		"audio_encoder_presets":    []string{"copy_audio", "aac", "libmp3lame", "opus"},
		"subtitle_encoder_presets": []string{"copy_subtitle", "srt", "ass", "webvtt"},
		"scene_detect_methods":     []string{"pyscenedetect", "mean", "mean_segmented", "none"},
		"inference_devices":        []string{"auto", "cuda", "mps", "xpu"},
		"tensorrt_profiles":        []int{1, 2, 3, 4, 5},
		"cfg_support":              "reserved-for-supported-models-only",
		"custom_model_policy":      "unsupported-architectures-blocked",
	})
}

func (s *Server) quit(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "shutting-down"})
	if s.cfg.Quit != nil {
		go s.cfg.Quit()
	}
}

// homeBrowsePath returns the user's home directory as a default browse starting point.
func homeBrowsePath() string {
	if home, err := os.UserHomeDir(); err == nil {
		return home
	}
	return "."
}

// browseFiles lists directories in the user's home directory, filtering to video files.
func (s *Server) browseFiles(w http.ResponseWriter, r *http.Request) {
	// Check for saved browse path cookie
	path := cleanPath(r.URL.Query().Get("path"), "")
	if path == "" {
		if cookie, err := r.Cookie("last_browse_path"); err == nil && cookie.Value != "" {
			path = cookie.Value
		} else {
			path = homeBrowsePath()
		}
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	type item struct {
		Name  string `json:"name"`
		Path  string `json:"path"`
		IsDir bool   `json:"is_dir"`
	}
	items := make([]item, 0, len(entries))
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if !info.IsDir() && !isVideoFile(e.Name()) {
			continue
		}
		items = append(items, item{
			Name:  e.Name(),
			Path:  filepath.Join(path, e.Name()),
			IsDir: info.IsDir(),
		})
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].IsDir != items[j].IsDir {
			return items[i].IsDir
		}
		return strings.ToLower(items[i].Name) < strings.ToLower(items[j].Name)
	})
	parent := filepath.Dir(path)
	writeJSON(w, http.StatusOK, map[string]any{"path": path, "parent": parent, "items": items})
}

// searchFiles searches for video files by name.
func (s *Server) searchFiles(w http.ResponseWriter, r *http.Request) {
	type itemSearch struct {
		Name string `json:"name"`
		Path string `json:"path"`
	}
	path := cleanPath(r.URL.Query().Get("path"), s.cfg.Runtime.RootDir())
	query := strings.TrimSpace(r.URL.Query().Get("q"))
	if query == "" {
		writeJSON(w, http.StatusOK, map[string]any{"path": path, "query": "", "items": []itemSearch{}})
		return
	}
	queryLower := strings.ToLower(query)
	var results []itemSearch
	err := filepath.WalkDir(path, func(fp string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		name := d.Name()
		if strings.HasPrefix(name, ".") {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if !d.IsDir() && !isVideoFile(name) {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		if strings.Contains(strings.ToLower(name), queryLower) {
			results = append(results, itemSearch{Name: name, Path: fp})
		}
		return nil
	})
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	sort.Slice(results, func(i, j int) bool {
		return strings.ToLower(results[i].Name) < strings.ToLower(results[j].Name)
	})
	writeJSON(w, http.StatusOK, map[string]any{"path": path, "query": query, "items": results})
}

func isVideoFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts", ".m4v", ".zip", ".rar":
		return true
	default:
		return false
	}
}

func (s *Server) streamVideo(w http.ResponseWriter, r *http.Request) {
	path := cleanPath(r.URL.Query().Get("path"), "")
	if path == "" {
		writeError(w, http.StatusBadRequest, "path is required")
		return
	}
	info, err := os.Stat(path)
	if err != nil {
		writeError(w, http.StatusNotFound, "video is not readable: "+err.Error())
		return
	}
	if info.IsDir() || !isVideoFile(path) {
		writeError(w, http.StatusBadRequest, "path is not a supported video file")
		return
	}
	w.Header().Set("Cache-Control", "private, max-age=60")
	http.ServeFile(w, r, path)
}

func (s *Server) openFolder(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Path string `json:"path"`
		Job  string `json:"job"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	path := req.Path
	if req.Job != "" {
		if job, ok := s.cfg.Jobs.Get(req.Job); ok {
			path = job.Output
		}
	}
	path = cleanPath(path, "")
	if path == "" {
		writeError(w, http.StatusBadRequest, "path is required")
		return
	}
	if info, err := os.Stat(path); err == nil && !info.IsDir() {
		path = filepath.Dir(path)
	}
	if info, err := os.Stat(path); err != nil || !info.IsDir() {
		writeError(w, http.StatusNotFound, "folder is not readable")
		return
	}
	if err := exec.Command("xdg-open", path).Start(); err != nil {
		writeError(w, http.StatusInternalServerError, "open folder failed: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "opened", "path": path})
}

func cleanPath(value, fallback string) string {
	if value == "" {
		value = fallback
	}
	value = os.ExpandEnv(value)
	if strings.HasPrefix(value, "~") {
		if home, err := os.UserHomeDir(); err == nil {
			value = filepath.Join(home, strings.TrimPrefix(value, "~"))
		}
	}
	if abs, err := filepath.Abs(value); err == nil {
		value = abs
	}
	if real, err := filepath.EvalSymlinks(value); err == nil {
		value = real
	}
	return value
}

func (s *Server) startJob(w http.ResponseWriter, r *http.Request) {
	var req worker.Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	job, err := s.cfg.Jobs.Start(req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, job)
}

func (s *Server) getJob(w http.ResponseWriter, r *http.Request) {
	job, ok := s.cfg.Jobs.Get(r.PathValue("id"))
	if !ok {
		writeError(w, http.StatusNotFound, "job not found")
		return
	}
	writeJSON(w, http.StatusOK, job)
}

func (s *Server) jobLivePreview(w http.ResponseWriter, r *http.Request) {
	job, ok := s.cfg.Jobs.Get(r.PathValue("id"))
	if !ok {
		writeError(w, http.StatusNotFound, "job not found")
		return
	}
	if len(job.LivePreview) == 0 {
		writeError(w, http.StatusAccepted, "live preview is not ready yet")
		return
	}
	w.Header().Set("Content-Type", "image/jpeg")
	w.Header().Set("Cache-Control", "no-store")
	w.Write(job.LivePreview)
}

func (s *Server) jobPreview(w http.ResponseWriter, r *http.Request) {
	job, ok := s.cfg.Jobs.Get(r.PathValue("id"))
	if !ok {
		writeError(w, http.StatusNotFound, "job not found")
		return
	}
	if job.Status != "done" {
		writeError(w, http.StatusAccepted, "preview is available after the job finishes")
		return
	}
	if job.Output == "" {
		writeError(w, http.StatusNotFound, "job output is empty")
		return
	}
	info, err := os.Stat(job.Output)
	if err != nil {
		writeError(w, http.StatusNotFound, "job output is not readable: "+err.Error())
		return
	}
	preview := filepath.Join(os.TempDir(), "dasiwa-previews", job.ID+"-"+fmt.Sprintf("%x", info.ModTime().UnixNano())+".mp4")
	if _, err := os.Stat(preview); err != nil {
		if err := os.MkdirAll(filepath.Dir(preview), 0755); err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}
		cmd := exec.Command("ffmpeg",
			"-hide_banner", "-loglevel", "error", "-y",
			"-i", job.Output,
			"-t", "20",
			"-vf", "fps=12,scale='min(480,iw)':-2",
			"-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
			"-movflags", "+faststart", preview,
		)
		if out, err := cmd.CombinedOutput(); err != nil {
			writeError(w, http.StatusInternalServerError, "preview encode failed: "+strings.TrimSpace(string(out)))
			return
		}
	}
	w.Header().Set("Content-Type", "video/mp4")
	w.Header().Set("Cache-Control", "private, max-age=3600")
	http.ServeFile(w, r, preview)
}

func (s *Server) cancelJob(w http.ResponseWriter, r *http.Request) {
	if err := s.cfg.Jobs.Cancel(r.PathValue("id")); err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "cancelled"})
}

func (s *Server) jobEvents(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}
	ch, unsubscribe, ok := s.cfg.Jobs.Subscribe(r.PathValue("id"))
	if !ok {
		writeError(w, http.StatusNotFound, "job not found")
		return
	}
	defer unsubscribe()

	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache, no-store")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher.Flush()

	heartbeat := time.NewTicker(15 * time.Second)
	defer heartbeat.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-heartbeat.C:
			_, _ = fmt.Fprint(w, "event: heartbeat\ndata: ping\n\n")
			flusher.Flush()
		case event := <-ch:
			if err := writeSSE(w, event.Type, event); err != nil {
				return
			}
			flusher.Flush()
			if event.Job != nil && isTerminalJobStatus(event.Job.Status) {
				return
			}
		}
	}
}

func isTerminalJobStatus(status string) bool {
	switch status {
	case "done", "error", "cancelled":
		return true
	default:
		return false
	}
}

func writeSSE(w http.ResponseWriter, eventType string, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "event: %s\n", eventType); err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", data)
	return err
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func withNoCache(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") {
			w.Header().Set("Cache-Control", "no-store")
		}
		next.ServeHTTP(w, r)
	})
}
