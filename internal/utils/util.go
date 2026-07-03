// Package utils provides core utility functions for DaSiWa TrueVideoEnhancer.
// Includes logging, colors, file/folder operations, and common helpers.
package utils

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Colors holds ANSI escape codes for terminal output.
type Colors struct{}

var Color = Colors{}

const (
	Red     = "\033[91m"
	Green   = "\033[92m"
	Yellow  = "\033[93m"
	Blue    = "\033[94m"
	Reset   = "\033[0m"
	Bold    = "\033[1m"
	Dim     = "\033[2m"
	Underline = "\033[4m"
)

// LogError writes an error message prefixed with [ERROR] to stderr.
func LogError(msg string) {
	fmt.Fprintf(os.Stderr, "%s[ERROR]%s %s\n", Red, Reset, msg)
}

// LogWarn writes a warning message prefixed with [WARN] to stderr.
func LogWarn(msg string) {
	fmt.Fprintf(os.Stderr, "%s[WARN]%s %s\n", Yellow, Reset, msg)
}

// LogInfo writes an informational message to stdout.
func LogInfo(msg string) {
	fmt.Println(msg)
}

// Log renders a formatted log line with optional severity prefix.
func Log(severity, msg string) {
	switch strings.ToUpper(severity) {
	case "ERROR":
		LogError(msg)
	case "WARN":
		LogWarn(msg)
	default:
		fmt.Fprintln(os.Stderr, msg)
	}
}

// FileExists returns true if the given path exists and is a regular file.
func FileExists(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !info.IsDir()
}

// DirExists returns true if the given path exists and is a directory.
func DirExists(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return info.IsDir()
}

// RemoveFile removes a single file if it exists. Returns nil on success or
// when the file doesn't exist.
func RemoveFile(path string) error {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil
	}
	return os.Remove(path)
}

// RemoveDir recursively removes a directory tree. Returns nil on success or
// when the directory doesn't exist.
func RemoveDir(path string) error {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil
	}
	return os.RemoveAll(path)
}

// EnsureDir creates the parent directory of path if it doesn't exist.
func EnsureDir(path string) error {
	return os.MkdirAll(filepath.Dir(path), 0o755)
}

// FormatBytes formats byte count to human-readable form (B, KB, MB, GB).
func FormatBytes(bytes uint64) string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
	)
	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(MB))
	case bytes >= KB:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(KB))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

// Percent calculates percentage of part relative to whole.
func Percent(part, whole float64) float64 {
	if whole == 0 {
		return 0
	}
	return (part / whole) * 100
}

// PadString pads s on both sides so that total length equals n.
// If padding is odd, extra character goes to the right.
func PadString(s string, n int, pad rune) string {
	if len(s) >= n {
		return s
	}
	padTotal := n - len(s)
	leftPad := padTotal / 2
	rightPad := padTotal - leftPad
	return strings.Repeat(string(pad), leftPad) + s + strings.Repeat(string(pad), rightPad)
}

// Truncate truncates s to maxLen characters, appending "..." if cut off.
func Truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen < 3 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}
