package utils

import (
	"os"
	"os/exec"
	"path/filepath"
)

// FindFFMpeg locates the FFmpeg executable by checking candidate paths
// in priority order. Falls back to PATH lookup via exec.LookPath.
//
// Order: custom_path → local bin/ → system locations → PATH.
func FindFFMpeg(customPath string) string {
	if customPath != "" && IsExecutable(customPath) {
		return customPath
	}

	candidates := []string{
		filepath.Join(".", "bin", "ffmpeg"),
		"/usr/bin/ffmpeg",
		"/usr/local/bin/ffmpeg",
		filepath.Join(os.Getenv("HOME"), "bin", "ffmpeg"),
	}

	for _, c := range candidates {
		if IsExecutable(c) {
			return c
		}
	}

	// Try system PATH
	if path, err := exec.LookPath("ffmpeg"); err == nil {
		return path
	}

	return "ffmpeg" // assume available in PATH
}

// ValidateFFMpeg checks whether the given FFmpeg binary responds to -version.
func ValidateFFMpeg(ffmpegPath string) bool {
	cmd := exec.Command(ffmpegPath, "-version")
	return cmd.Run() == nil
}

// GetFFMpegVersion returns the first line of `ffmpeg -version` output,
// e.g. "ffmpeg version 6.1". Empty string on failure.
func GetFFMpegVersion(ffmpegPath string) string {
	cmd := exec.Command(ffmpegPath, "-version")
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	for i, ch := range out {
		if ch == '\n' {
			return string(out[:i])
		}
	}
	return string(out)
}

// IsExecutable returns true if path exists and has at least execute permission.
func IsExecutable(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !info.IsDir() && info.Mode().Perm()&0o111 != 0
}
