package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

const capabilityTimeout = 15 * time.Second
const capabilityTTL = 30 * time.Second
const capabilityOutputLimit = 256 * 1024

// Cache only validated successes. Concurrent callers share an in-flight probe,
// but can cancel independently while waiting for it.
type capabilityCache struct {
	mu      sync.Mutex
	data    json.RawMessage
	expires time.Time
	pending chan struct{}
}

func (s *Server) capabilities(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), capabilityTimeout)
	defer cancel()
	data, err := s.renderCapabilities(ctx)
	if err != nil {
		status := http.StatusServiceUnavailable
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			status = http.StatusGatewayTimeout
		}
		writeError(w, status, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, data)
}

func (s *Server) renderCapabilities(ctx context.Context) (json.RawMessage, error) {
	c := &s.capabilitiesState
	for {
		c.mu.Lock()
		if c.data != nil && time.Now().Before(c.expires) {
			data := c.data
			c.mu.Unlock()
			return data, nil
		}
		if c.pending != nil {
			pending := c.pending
			c.mu.Unlock()
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-pending:
				continue
			}
		}
		c.pending = make(chan struct{})
		c.mu.Unlock()
		data, err := s.probeRenderCapabilities(ctx)
		c.mu.Lock()
		if err == nil {
			c.data = data
			c.expires = time.Now().Add(capabilityTTL)
		}
		close(c.pending)
		c.pending = nil
		c.mu.Unlock()
		return data, err
	}
}

func (s *Server) probeRenderCapabilities(ctx context.Context) (json.RawMessage, error) {
	if s.cfg.Runtime == nil || s.cfg.Runtime.RootDir() == "" {
		return nil, errors.New("render capabilities unavailable: configure and install the Python runtime")
	}
	root, err := filepath.Abs(s.cfg.Runtime.RootDir())
	if err != nil {
		return nil, fmt.Errorf("resolve runtime directory: %w", err)
	}
	// Require the installed venv before PythonCommand: its fallback can invoke uv
	// without a request context and bare/system Python has different packages.
	python := filepath.Join(root, "venv", "bin", "python")
	if runtime.GOOS == "windows" {
		python = filepath.Join(root, "venv", "Scripts", "python.exe")
	}
	if info, err := os.Stat(python); err != nil || info.IsDir() {
		return nil, fmt.Errorf("render capabilities unavailable: install the runtime venv (missing interpreter %s)", python)
	}
	python, err = filepath.Abs(s.cfg.Runtime.PythonCommand())
	if err != nil {
		return nil, fmt.Errorf("resolve runtime interpreter: %w", err)
	}
	repo := filepath.Dir(root)
	script := filepath.Join(repo, "backend", "rve-backend.py")
	if info, err := os.Stat(script); err != nil || info.IsDir() {
		return nil, fmt.Errorf("render capabilities unavailable: production backend missing at %s; restore the backend source", script)
	}
	run := s.capabilitiesRun
	if run == nil {
		run = runCapabilityCommand
	}
	// This is the backend's lightweight, API-only flag. Do not run diagnostics,
	// import models, or construct any CUDA/VFX effects here.
	out, err := run(ctx, repo, python, script, "--render-capabilities")
	if ctx.Err() != nil {
		return nil, fmt.Errorf("render capability probe did not complete: %w", ctx.Err())
	}
	if err != nil {
		return nil, fmt.Errorf("render capability probe failed; check the runtime installation and backend: %w", err)
	}
	if len(out) > capabilityOutputLimit {
		return nil, errors.New("render capability probe exceeded the output limit")
	}
	if err := validateRenderCapabilities(out); err != nil {
		return nil, fmt.Errorf("invalid render capability response; update or repair the backend: %w", err)
	}
	return json.RawMessage(bytes.TrimSpace(out)), nil
}

// Each stream is bounded, including stderr on failures. exec's WaitDelay also
// bounds inherited pipes if a subprocess exits leaving a child with them open.
type capabilityBuffer struct{ bytes.Buffer }

func (b *capabilityBuffer) Write(p []byte) (int, error) {
	if len(p) > capabilityOutputLimit-b.Len() {
		return 0, errors.New("capability output limit exceeded")
	}
	return b.Buffer.Write(p)
}
func runCapabilityCommand(ctx context.Context, dir, python string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, python, args...)
	cmd.Dir = dir
	cmd.WaitDelay = time.Second
	var stdout, stderr capabilityBuffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		detail := strings.TrimSpace(stderr.String())
		if len(detail) > 2048 {
			detail = detail[:2048]
		}
		return nil, fmt.Errorf("%w: %s", err, detail)
	}
	return stdout.Bytes(), nil
}

// Require the complete contract rather than decoding missing fields as false
// or silently treating a random JSON object as support. Preserve backend fields
// verbatim; this endpoint is not another source of codec compatibility policy.
func validateRenderCapabilities(data []byte) error {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	require := func(object map[string]json.RawMessage, key string, dst any) error {
		raw, ok := object[key]
		if !ok || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			return fmt.Errorf("missing %s", key)
		}
		if err := json.Unmarshal(raw, dst); err != nil {
			return fmt.Errorf("%s: %w", key, err)
		}
		return nil
	}
	var vfx map[string]json.RawMessage
	if err := require(obj, "vfx", &vfx); err != nil {
		return err
	}
	for _, key := range []string{"available", "strength", "encoding", "hdr", "frame_generation", "frame_generation_encoding"} {
		var value bool
		if err := require(vfx, key, &value); err != nil {
			return err
		}
	}
	var version string
	if err := require(vfx, "version", &version); err != nil {
		return err
	}
	// The backend uses null for no VFX error.
	if raw, ok := vfx["error"]; !ok {
		return errors.New("missing vfx.error")
	} else if string(raw) != "null" {
		var value string
		if err := json.Unmarshal(raw, &value); err != nil {
			return fmt.Errorf("vfx.error: %w", err)
		}
	}
	for _, key := range []string{"streaming", "quality_levels"} {
		var values []string
		if err := require(vfx, key, &values); err != nil {
			return err
		}
	}
	for _, key := range []string{"encoders", "pipelines", "containers"} {
		var values []string
		if err := require(obj, key, &values); err != nil {
			return err
		}
	}
	for _, key := range []string{"codec_matrix", "hdr_codec_matrix", "precise_codec_matrix"} {
		var matrix map[string][]string
		if err := require(obj, key, &matrix); err != nil {
			return err
		}
		for container, values := range matrix {
			if values == nil {
				return fmt.Errorf("%s.%s must be an array", key, container)
			}
		}
	}
	return nil
}
