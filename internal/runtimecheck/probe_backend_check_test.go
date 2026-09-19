package runtimecheck

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestBackendCheckParsesASCIIStatusMarkers(t *testing.T) {
	repoRoot := t.TempDir()
	backendDir := filepath.Join(repoRoot, "backend")
	if err := os.MkdirAll(backendDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(backendDir, "rve-backend.py"), nil, 0o644); err != nil {
		t.Fatal(err)
	}

	runtimeDir := filepath.Join(repoRoot, "runtime")
	venvBin := filepath.Join(runtimeDir, "venv", "bin")
	if err := os.MkdirAll(venvBin, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(venvBin, "python"), nil, 0o755); err != nil {
		t.Fatal(err)
	}

	probe := NewProbe(
		WithRootDir(runtimeDir),
		WithRunner(func(context.Context, string, ...string) ([]byte, error) {
			return []byte("[OK] PyTorch\n[XX] TensorRT\n"), nil
		}),
	)
	result := probe.BackendCheck(repoRoot)

	if result.TotalCount != 2 || result.PassedCount != 1 {
		t.Fatalf("counts = %d total, %d passed; want 2 total, 1 passed", result.TotalCount, result.PassedCount)
	}
	if result.Status != "fail" {
		t.Fatalf("status = %q, want fail", result.Status)
	}
	if got := result.Items[0]; !got.Pass || got.Name != "PyTorch" {
		t.Fatalf("first item = %#v, want passing PyTorch", got)
	}
	if got := result.Items[1]; got.Pass || got.Name != "TensorRT" {
		t.Fatalf("second item = %#v, want failing TensorRT", got)
	}
}
