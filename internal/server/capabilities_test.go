package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/darksidewalker/da-si-wa-true-video-enhancer/internal/runtimecheck"
)

const testCapabilities = `{"vfx":{"available":false,"version":"unavailable","strength":false,"encoding":false,"hdr":false,"frame_generation":false,"frame_generation_encoding":false,"streaming":[],"quality_levels":[],"error":"SDK not installed"},"encoders":["libx264"],"pipelines":["upscale"],"containers":["mp4","avi"],"codec_matrix":{"mp4":["auto","libx264"],"avi":["auto","libx264"]},"hdr_codec_matrix":{"mp4":[],"avi":[]},"precise_codec_matrix":{"mp4":[],"avi":[]}}`

func capabilityTestServer(t *testing.T) (*Server, string, string) {
	t.Helper()
	repo := t.TempDir()
	root := filepath.Join(repo, "runtime")
	python := filepath.Join(root, "venv", "bin", "python")
	if runtime.GOOS == "windows" {
		python = filepath.Join(root, "venv", "Scripts", "python.exe")
	}
	script := filepath.Join(repo, "backend", "rve-backend.py")
	for _, path := range []string{python, script} {
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, nil, 0755); err != nil {
			t.Fatal(err)
		}
	}
	probe := runtimecheck.NewProbe(runtimecheck.WithRootDir(root), runtimecheck.WithRunner(func(context.Context, string, ...string) ([]byte, error) {
		t.Error("PythonCommand must not fall back to uv or system Python")
		return nil, errors.New("unexpected fallback")
	}))
	return New(Config{Runtime: probe, Web: os.DirFS(repo)}), python, script
}

func requestCapabilities(s *Server, ctx context.Context) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	// Caller-controlled repo and path are deliberately ignored.
	req := httptest.NewRequest(http.MethodGet, "/api/capabilities?repo_root=/wrong&path=/wrong", nil).WithContext(ctx)
	s.Routes().ServeHTTP(w, req)
	return w
}

func TestCapabilitiesInvocationAndCache(t *testing.T) {
	s, python, script := capabilityTestServer(t)
	var calls atomic.Int32
	s.capabilitiesRun = func(ctx context.Context, dir, command string, args ...string) ([]byte, error) {
		calls.Add(1)
		if command != python || dir != filepath.Dir(filepath.Dir(script)) || len(args) != 2 || args[0] != script || args[1] != "--render-capabilities" {
			t.Errorf("unexpected invocation: %q %q %q", dir, command, args)
		}
		if deadline, ok := ctx.Deadline(); !ok || time.Until(deadline) > capabilityTimeout {
			t.Error("missing bounded request deadline")
		}
		return []byte(testCapabilities), nil
	}
	var wg sync.WaitGroup
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			w := requestCapabilities(s, context.Background())
			if w.Code != http.StatusOK {
				t.Errorf("status %d: %s", w.Code, w.Body.String())
				return
			}
			var got, want any
			if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
				t.Error(err)
			}
			_ = json.Unmarshal([]byte(testCapabilities), &want)
			gotJSON, _ := json.Marshal(got)
			wantJSON, _ := json.Marshal(want)
			if string(gotJSON) != string(wantJSON) {
				t.Errorf("capability contract not preserved: %s", gotJSON)
			}
		}()
	}
	wg.Wait()
	if calls.Load() != 1 {
		t.Fatalf("concurrent success probes = %d", calls.Load())
	}
	s.capabilitiesState.mu.Lock()
	s.capabilitiesState.expires = time.Now().Add(-time.Second)
	s.capabilitiesState.mu.Unlock()
	if w := requestCapabilities(s, context.Background()); w.Code != http.StatusOK {
		t.Fatal(w.Body.String())
	}
	if calls.Load() != 2 {
		t.Fatal("expired success not re-probed")
	}
}

func TestCapabilitiesMissingRuntimeAndBackend(t *testing.T) {
	for _, missing := range []string{"config", "venv", "backend"} {
		t.Run(missing, func(t *testing.T) {
			s, python, script := capabilityTestServer(t)
			switch missing {
			case "config":
				s.cfg.Runtime = nil
			case "venv":
				if err := os.Remove(python); err != nil {
					t.Fatal(err)
				}
			case "backend":
				if err := os.Remove(script); err != nil {
					t.Fatal(err)
				}
			}
			s.capabilitiesRun = func(context.Context, string, string, ...string) ([]byte, error) {
				t.Error("must not run with missing prerequisites")
				return nil, nil
			}
			w := requestCapabilities(s, context.Background())
			if w.Code != http.StatusServiceUnavailable || !strings.Contains(w.Body.String(), "error") || strings.Contains(w.Body.String(), "available") && strings.Contains(w.Body.String(), "\"vfx\"") {
				t.Fatalf("expected actionable failure, got %d %s", w.Code, w.Body.String())
			}
		})
	}
}

func TestCapabilitiesFailuresNotCached(t *testing.T) {
	for _, bad := range []string{"garbage", "{}", "null", `{"vfx":{"available":true}}`, strings.Replace(testCapabilities, `"hdr":false`, `"hdr":null`, 1), strings.Replace(testCapabilities, `"pipelines":["upscale"]`, `"pipelines":null`, 1), testCapabilities + " {}", strings.Repeat("x", capabilityOutputLimit+1), "exit-error"} {
		t.Run(bad[:min(len(bad), 35)], func(t *testing.T) {
			s, _, _ := capabilityTestServer(t)
			calls := 0
			s.capabilitiesRun = func(context.Context, string, string, ...string) ([]byte, error) {
				calls++
				if calls > 1 {
					return []byte(testCapabilities), nil
				}
				if bad == "exit-error" {
					return nil, errors.New("broken interpreter")
				}
				return []byte(bad), nil
			}
			w := requestCapabilities(s, context.Background())
			if w.Code != http.StatusServiceUnavailable || strings.Contains(w.Body.String(), `"vfx":`) {
				t.Fatalf("invalid support assumed: %d %s", w.Code, w.Body.String())
			}
			if w := requestCapabilities(s, context.Background()); w.Code != http.StatusOK {
				t.Fatalf("failure cached: %d %s", w.Code, w.Body.String())
			}
		})
	}
}

func TestCapabilitiesTimeout(t *testing.T) {
	s, _, _ := capabilityTestServer(t)
	s.capabilitiesRun = func(ctx context.Context, _, _ string, _ ...string) ([]byte, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	w := requestCapabilities(s, ctx)
	if w.Code != http.StatusGatewayTimeout || strings.Contains(w.Body.String(), `"vfx":`) {
		t.Fatalf("timeout assumed support: %d %s", w.Code, w.Body.String())
	}
	s.capabilitiesRun = func(context.Context, string, string, ...string) ([]byte, error) { return []byte(testCapabilities), nil }
	if w := requestCapabilities(s, context.Background()); w.Code != http.StatusOK {
		t.Fatal("timeout was cached")
	}
}

func TestCapabilityCommandBoundAndTimeout(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX fake interpreter")
	}
	dir := t.TempDir()
	interpreter := filepath.Join(dir, "python")
	for _, tc := range []struct{ name, script string }{
		{"success", "#!/bin/sh\nprintf '%s' '" + testCapabilities + "'\n"},
		{"oversize", "#!/bin/sh\nwhile :; do printf '%01024d' 0; done\n"},
		{"timeout", "#!/bin/sh\nwhile :; do :; done\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := os.WriteFile(interpreter, []byte(tc.script), 0755); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
			defer cancel()
			out, err := runCapabilityCommand(ctx, dir, interpreter, "--render-capabilities")
			if tc.name == "success" {
				if err != nil || string(out) != testCapabilities {
					t.Fatalf("%s %v", out, err)
				}
			} else if err == nil {
				t.Fatalf("%s must fail", tc.name)
			}
		})
	}
}
