package main

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/darksidewalker/da-si-wa-true-video-enhancer/internal/runtimecheck"
	"github.com/darksidewalker/da-si-wa-true-video-enhancer/internal/server"
	"github.com/darksidewalker/da-si-wa-true-video-enhancer/internal/utils"
	"github.com/darksidewalker/da-si-wa-true-video-enhancer/internal/worker"
)

//go:embed web/*
var webFiles embed.FS

// version is populated at build time via -ldflags "-X main.version=..."
var version = "dev"

func main() {
	if err := run(); err != nil {
		utils.LogError(fmt.Sprintf("server failed: %v", err))
		os.Exit(1)
	}
}

func run() error {
	port := getEnvOrDefault("DASIWA_PORT", "8612")
	repoRoot := resolveRepoRoot()

	log.Printf("%s v%s starting", appName(), version)
	log.Printf("repo root: %s", repoRoot)

	// Build web FS from embedded files (works in compiled binary)
	webFS, err := fs.Sub(webFiles, "web")
	if err != nil {
		return fmt.Errorf("embed web files: %w", err)
	}

	// Probe for runtime environment
	probe := runtimecheck.NewProbe(
		runtimecheck.WithRootDir(filepath.Join(repoRoot, "runtime")),
	)

	// Worker manager config
	backendScript := filepath.Join(repoRoot, "backend", "rve-backend.py")
	venvDir := filepath.Join(repoRoot, "runtime", "venv")
	workerMgr := worker.NewManager(worker.Config{
		BackendScript: backendScript,
		Python:        probe.PythonCommand(),
		VenvDir:       venvDir,
	})

	srv := server.New(server.Config{
		Name:    appName(),
		Version: version,
		Web:     webFS,
		Runtime: probe,
		Jobs:    workerMgr,
		Quit:    func() {},
	})

	mux := srv.Routes()
	addr := ":" + port
	httpServer := &http.Server{Addr: addr, Handler: mux}

	// Graceful shutdown
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		<-ctx.Done()
		log.Println("shutting down gracefully...")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(shutdownCtx); err != nil {
			log.Printf("shutdown error: %v", err)
		}
	}()

	log.Printf("DaSiWa True Video Enhancer v%s listening on http://127.0.0.1:%s", version, port)
	url := "http://127.0.0.1:" + port
	if shouldOpenBrowser() {
		go openBrowserWhenReady(url)
	}

	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("listen: %w", err)
	}
	return nil
}

func shouldOpenBrowser() bool {
	value := strings.ToLower(strings.TrimSpace(os.Getenv("DASIWA_NO_BROWSER")))
	return value != "1" && value != "true" && value != "yes"
}

func openBrowserWhenReady(url string) {
	client := http.Client{Timeout: 500 * time.Millisecond}
	for attempt := 0; attempt < 20; attempt++ {
		response, err := client.Get(url + "/api/health")
		if err == nil {
			_ = response.Body.Close()
			if response.StatusCode == http.StatusOK {
				break
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	var command *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		command = exec.Command("open", url)
	case "windows":
		command = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		command = exec.Command("xdg-open", url)
	}
	if err := command.Start(); err != nil {
		log.Printf("could not open browser automatically: %v", err)
	}
}

// appName returns the binary name.
func appName() string {
	name := filepath.Base(os.Args[0])
	return strings.TrimSuffix(name, filepath.Ext(name))
}

// resolveRepoRoot walks up from the current working directory (or the binary's
// location) until it finds a go.mod file. This ensures that even when the
// binary runs from dist/, models are downloaded into the project-root models/
// folder instead of dist/models/.
func resolveRepoRoot() string {
	candidates := []string{
		".",                      // cwd
		filepath.Dir(os.Args[0]), // next to the binary
		filepath.Dir(filepath.Dir(os.Args[0])),
	}
	for _, c := range candidates {
		dir, err := filepath.Abs(c)
		if err != nil {
			continue
		}
		abs, _ := filepath.EvalSymlinks(dir)
		if abs == "" {
			continue
		}
		// Walk up from this candidate looking for go.mod
		for dir != "/" {
			if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
				return dir
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	// Ultimate fallback: cwd
	if cwd, err := os.Getwd(); err == nil {
		return cwd
	}
	return "."
}

func getEnvOrDefault(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}
