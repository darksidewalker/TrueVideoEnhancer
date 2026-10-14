package worker

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestNvidiaIntermediateRequestAndArgs(t *testing.T) {
	for _, filter := range []string{"", "area", "lanczos", "bicubic"} {
		for _, pair := range [][2]int{{2, 1}, {4, 1}, {4, 2}} {
			req := Request{Input: "in.mp4", Output: "out.mp4", DryRun: true, ProcessingMode: "custom", SpatialBackend: "nvidia", TemporalBackend: "none", RtxUpscale: "VSR", Scale: pair[1], RtxUpscaleIntermediate: pair[0], RtxDownsampleFilter: filter}
			if err := validateRequest(req); err != nil {
				t.Fatal(err)
			}
			argv := strings.Join(advancedArgs(req), " ")
			if !strings.Contains(argv, "--rtx_upscale_intermediate") || (filter != "" && !strings.Contains(argv, "--rtx_downsample_filter "+filter)) {
				t.Fatal(argv)
			}
		}
	}
	off := Request{RtxUpscale: "VSR", ProcessingMode: "custom", SpatialBackend: "nvidia"}
	if strings.Contains(strings.Join(advancedArgs(off), " "), "--rtx_upscale_intermediate") {
		t.Fatal("off changed argv")
	}
}

func TestNvidiaIntermediateRejectsInvalidRequests(t *testing.T) {
	base := Request{Input: "in", Output: "out", DryRun: true, ProcessingMode: "custom", SpatialBackend: "nvidia", TemporalBackend: "none", RtxUpscale: "VSR", Scale: 1, RtxUpscaleIntermediate: 2}
	for name, change := range map[string]func(*Request){
		"bad scale":  func(r *Request) { r.RtxUpscaleIntermediate = 3 },
		"bad filter": func(r *Request) { r.RtxDownsampleFilter = "nearest" },
		"equal":      func(r *Request) { r.Scale = 2 },
		"smaller":    func(r *Request) { r.Scale = 4 },
		"override":   func(r *Request) { r.OverrideUpscaleScale = 4 },
		"legacy":     func(r *Request) { r.ProcessingMode = "legacy" },
		"AI":         func(r *Request) { r.SpatialBackend = "ai" },
		"none":       func(r *Request) { r.SpatialBackend = "none" },
		"off filter": func(r *Request) { r.RtxUpscaleIntermediate = 0; r.RtxDownsampleFilter = "area" },
		"extra":      func(r *Request) { r.ExtraArgs = []string{"--rtx_upscale_intermediate=4"} },
	} {
		t.Run(name, func(t *testing.T) {
			r := base
			change(&r)
			if validateRequest(r) == nil {
				t.Fatal("accepted invalid request")
			}
		})
	}
	for _, body := range []string{`{"rtx_upscale_intermediate":NaN}`, `{"rtx_upscale_intermediate":2.5}`, `{"rtx_upscale_intermediate":"2"}`} {
		var r Request
		if json.Unmarshal([]byte(body), &r) == nil {
			t.Fatal("accepted invalid numeric JSON", body)
		}
	}
	var batch BatchRequest
	if err := json.Unmarshal([]byte(`{"inputs":["in"],"rtx_upscale_intermediate":4,"rtx_downsample_filter":"area"}`), &batch); err != nil {
		t.Fatal(err)
	}
	if batch.RtxUpscaleIntermediate != 4 || batch.RtxDownsampleFilter != "area" {
		t.Fatal("batch lost settings")
	}
}
