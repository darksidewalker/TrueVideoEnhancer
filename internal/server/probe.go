package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strings"
)

// ProbeMeta holds ffprobe-derived metadata for a video file.
type ProbeMeta struct {
	InputPath      string  `json:"input_path"`
	Duration       float64 `json:"duration,omitempty"`
	RFrameRate     string  `json:"r_frame_rate"`
	Width          int     `json:"width,omitempty"`
	Height         int     `json:"height,omitempty"`
	CodecName      string  `json:"codec_name"`
	AudioCodec     string  `json:"audio_codec,omitempty"`
	SampleRate     string  `json:"sample_rate,omitempty"`
	BitRate        string  `json:"bitrate,omitempty"`
	ColorPrimaries string  `json:"color_primaries,omitempty"`
	ColorTransfer  string  `json:"color_transfer,omitempty"`
	NBStreams      int     `json:"nb_streams"`
	Error          string  `json:"error,omitempty"`
}

func probeVideo(w http.ResponseWriter, r *http.Request) {
	input := strings.TrimSpace(r.URL.Query().Get("path"))
	if input == "" {
		writeError(w, http.StatusBadRequest, "path parameter required")
		return
	}
	meta := runProbe(input)
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(meta) // always sends ProbeMeta (with error field if needed)
	if meta.Error != "" {
		http.NotFound(w, r)
		return
	}
}

func runProbe(path string) ProbeMeta {
	var meta ProbeMeta
	meta.InputPath = path

	cmd := exec.Command("ffprobe",
		"-v", "quiet",
		"-print_format", "json",
		"-show_format",
		"-show_streams",
		path,
	)
	out, err := cmd.Output()
	if err != nil {
		return ProbeMeta{InputPath: path, Error: fmt.Sprintf("ffprobe failed: %v", err)}
	}

	var probe struct {
		Format  map[string]any   `json:"format"`
		Streams []map[string]any `json:"streams"`
	}
	if err := json.Unmarshal(out, &probe); err != nil {
		return ProbeMeta{InputPath: path, Error: fmt.Sprintf("parse ffprobe JSON: %v", err)}
	}

	meta.NBStreams = len(probe.Streams)

	for _, s := range probe.Streams {
		ct, _ := s["codec_type"].(string)
		if ct == "video" && meta.Width == 0 {
			if w, ok := s["width"].(float64); ok {
				meta.Width = int(w)
			}
			if h, ok := s["height"].(float64); ok {
				meta.Height = int(h)
			}
			meta.CodecName, _ = s["codec_name"].(string)
			if fr, ok := s["r_frame_rate"].(string); ok {
				meta.RFrameRate = fr
			}
			meta.ColorPrimaries, _ = s["color_primaries"].(string)
			meta.ColorTransfer, _ = s["color_transfer"].(string)
		}
		if ct == "audio" && meta.SampleRate == "" {
			meta.AudioCodec, _ = s["codec_name"].(string)
			meta.SampleRate, _ = s["sample_rate"].(string)
		}
	}

	if dur, ok := probe.Format["duration"].(string); ok {
		var d float64
		fmt.Sscanf(dur, "%f", &d)
		meta.Duration = d
	}
	if br, ok := probe.Format["bit_rate"].(string); ok {
		meta.BitRate = br
	}

	return meta
}
