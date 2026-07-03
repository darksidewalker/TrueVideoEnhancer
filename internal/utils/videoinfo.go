// Package utils provides core utility functions for DaSiWa TrueVideoEnhancer.
// Includes logging, colors, file/folder operations, and common helpers.
package utils

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// VideoInfo holds parsed metadata from ffprobe.
type VideoInfo struct {
	InputPath       string  `json:"input_path"`
	Duration        float64 `json:"duration"`
	FPS             float64 `json:"fps"`
	Width           int     `json:"width"`
	Height          int     `json:"height"`
	Codec           string  `json:"codec"`
	Format          string  `json:"format"`
	PixelFormat     string  `json:"pixel_format"`
	NbFrames        int     `json:"nb_frames"`
	HasAudio        bool    `json:"has_audio"`
	AudioCodec      string  `json:"audio_codec,omitempty"`
	AudioSampleRate int     `json:"audio_sample_rate,omitempty"`
}

// ProbeFFProbe is a configurable runner for ffprobe commands.
type ProbeFFProbe func(args ...string) ([]byte, error)

// DefaultRunner runs ffprobe via exec.CommandContext with a 30-second timeout.
func DefaultRunner(ctx context.Context, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, "ffprobe", args...)
	return cmd.Output()
}

// GetVideoInfo extracts comprehensive video information using ffprobe.
// Returns nil if the probe fails or no video stream is found.
func GetVideoInfo(inputPath, ffmpegPath string) (*VideoInfo, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	args := []string{
		"-v", "quiet",
		"-print_format", "json",
		"-show_format",
		"-show_streams",
		inputPath,
	}

	data, err := DefaultRunner(ctx, args...)
	if err != nil {
		return nil, fmt.Errorf("ffprobe failed: %w", err)
	}

	var result struct {
		Streams []map[string]interface{} `json:"streams"`
		Format  map[string]interface{}   `json:"format"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("parse ffprobe JSON: %w", err)
	}

	videoStream := findStream(result.Streams, "video")
	if videoStream == nil {
		return nil, fmt.Errorf("no video stream found in %s", inputPath)
	}

	width := toInt(videoStream["width"])
	height := toInt(videoStream["height"])
	fps := parseFPS(videoStream["r_frame_rate"])
	nbFrames := toInt(videoStream["nb_frames"])

	audioStream := findStream(result.Streams, "audio")

	info := &VideoInfo{
		InputPath:   inputPath,
		Duration:    parseFloat(result.Format["duration"]),
		FPS:         fps,
		Width:       width,
		Height:      height,
		Codec:       toString(videoStream["codec_name"]),
		Format:      toString(result.Format["format_name"]),
		PixelFormat: toString(videoStream["pix_fmt"]),
		NbFrames:    nbFrames,
	}

	if audioStream != nil {
		info.HasAudio = true
		info.AudioCodec = toString(audioStream["codec_name"])
		info.AudioSampleRate = toInt(audioStream["sample_rate"])
	}

	return info, nil
}

// PrintVideoInfo pretty-prints video metadata to stdout.
func PrintVideoInfo(info *VideoInfo) {
	fmt.Printf("Input file: %s\n", info.InputPath)
	fmt.Printf("Duration: %.2fs\n", info.Duration)
	fmt.Printf("Resolution: %dx%d\n", info.Width, info.Height)
	fmt.Printf("FPS: %.2f\n", info.FPS)
	fmt.Printf("Codec: %s\n", info.Codec)
	fmt.Printf("Format: %s\n", info.Format)
	fmt.Printf("Pixel format: %s\n", info.PixelFormat)
	fmt.Printf("Frames: %d\n", info.NbFrames)

	if info.HasAudio {
		fmt.Printf("Audio: %s @ %dHz\n", info.AudioCodec, info.AudioSampleRate)
	} else {
		fmt.Println("Audio: none")
	}
}

// Helpers

func findStream(streams []map[string]interface{}, codecType string) map[string]interface{} {
	for _, s := range streams {
		if ct, ok := s["codec_type"].(string); ok && ct == codecType {
			return s
		}
	}
	return nil
}

func parseFPS(rFrameRate interface{}) float64 {
	s, ok := rFrameRate.(string)
	if !ok || s == "" {
		return 0
	}
	parts := strings.SplitN(s, "/", 2)
	if len(parts) == 2 {
		num, err1 := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
		den, err2 := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		if err1 == nil && err2 == nil && den != 0 {
			return num / den
		}
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return f
	}
	return 0
}

func toString(v interface{}) string {
	if v == nil {
		return ""
	}
	return fmt.Sprintf("%v", v)
}

func parseFloat(v interface{}) float64 {
	if v == nil {
		return 0
	}
	switch x := v.(type) {
	case float64:
		return x
	case string:
		f, _ := strconv.ParseFloat(x, 64)
		return f
	}
	return 0
}

func toInt(v interface{}) int {
	if v == nil {
		return 0
	}
	switch x := v.(type) {
	case float64:
		return int(x)
	case string:
		n, _ := strconv.Atoi(x)
		return n
	}
	return 0
}
