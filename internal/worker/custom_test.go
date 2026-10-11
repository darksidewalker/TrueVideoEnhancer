package worker

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestCustomModelForwardingAndLegacyIsolation(t *testing.T) {
	m := &Manager{cfg: Config{BackendScript: "backend/rve-backend.py"}}
	for _, s := range []string{"none", "ai", "nvidia"} {
		for _, temporal := range []string{"none", "rife", "nvidia"} {
			req := Request{Input: "in.mp4", Output: "out.mkv", DryRun: true, ProcessingMode: "custom", SpatialBackend: s, TemporalBackend: temporal, Pipeline: "nvidia-vfx", UpscaleModel: "up.safetensors", RIFEModel: "rife.safetensors", Precision: "float32", Scale: 1, RtxHDR: true}
			if s == "nvidia" {
				req.RtxUpscale = "VSR"
			}
			if err := validateRequest(req); err != nil {
				t.Fatal(err)
			}
			line := strings.Join(m.buildArgs(req), " ")
			for _, part := range []string{"--processing_mode custom", "--spatial_backend " + s, "--temporal_backend " + temporal, "--precision float32", "--rtx_hdr"} {
				if !strings.Contains(line, part) {
					t.Fatalf("missing %s: %s", part, line)
				}
			}
			if strings.Contains(line, "--upscale_model") != (s == "ai") {
				t.Fatalf("wrong AI forwarding: %s", line)
			}
			if strings.Contains(line, "--interpolate_model") != (temporal == "rife") {
				t.Fatalf("wrong RIFE forwarding: %s", line)
			}
			if strings.Contains(line, "--pipeline") {
				t.Fatalf("legacy pipeline leaked: %s", line)
			}
		}
	}
}

func TestCustomValidation(t *testing.T) {
	base := Request{Input: "in.mp4", Output: "out.mp4", DryRun: true, ProcessingMode: "custom", SpatialBackend: "none", TemporalBackend: "none", Scale: 1}
	tests := []func(*Request){
		func(r *Request) { r.SpatialBackend = "ai+nvidia" },
		func(r *Request) { r.TemporalBackend = "rife+nvidia" },
		func(r *Request) { r.SpatialBackend = "ai" },
		func(r *Request) { r.TemporalBackend = "rife" },
		func(r *Request) { r.Scale = 2 },
		func(r *Request) { r.TargetFPS = math.NaN() },
		func(r *Request) { r.SpatialBackend = "nvidia" },
		func(r *Request) { r.RtxUpscale = "VSR" },
		func(r *Request) { r.RtxFrameGeneration = true },
		func(r *Request) { r.HDRMode = true },
		func(r *Request) { r.ExtraArgs = []string{"--spatial_backend=ai"} },
	}
	for index, change := range tests {
		req := base
		change(&req)
		if err := validateRequest(req); err == nil {
			t.Fatalf("case %d accepted", index)
		}
	}
	if err := validateRequest(base); err != nil {
		t.Fatal(err)
	}
}

func TestCustomFlatBatchJSONPreservesFalseZero(t *testing.T) {
	var b BatchRequest
	if err := json.Unmarshal([]byte(`{"processing_mode":"custom","spatial_backend":"ai","temporal_backend":"nvidia","upscale_model":"up.safetensors","rife_model":"stale.safetensors","inputs":["one.mp4"],"output_dir":"out","rtx_hdr":true,"rtx_denoise_strength":0,"rtx_auto_shot_detection":false,"rtx_hdr_debanding":false}`), &b); err != nil {
		t.Fatal(err)
	}
	line := strings.Join((&Manager{cfg: Config{BackendScript: "backend.py"}}).buildArgs(b.Request), " ")
	for _, token := range []string{"--processing_mode custom", "--upscale_model up.safetensors", "--rtx_hdr", "--rtx_denoise_strength 0", "--rtx_auto_shot_detection false", "--rtx_hdr_debanding false"} {
		if !strings.Contains(line, token) {
			t.Fatalf("missing %s: %s", token, line)
		}
	}
	if strings.Contains(line, "--interpolate_model") {
		t.Fatal(line)
	}
}
