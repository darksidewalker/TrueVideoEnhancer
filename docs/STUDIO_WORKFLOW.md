# Studio workflow

Three compact areas remain visible: **Source** on the left, **Settings** in the middle and **Preview** on the right. Queue and logs remain below them. There are no global Fast/Balanced/Best presets, fixed pipeline dropdown or separate advanced-settings dialog.

## Plan the processing

Select a local source and export container in Source. Content controls AI model routing when AI Upscale is enabled. Scale and target FPS live in Settings. The source probe supplies dimensions, cadence and duration.

Activate the desired processing steps with seven feature buttons:

- **AI Upscale** or **NVIDIA Upscale**: one spatial backend for the target scale.
- **RIFE Interpolation** or **NVIDIA Frame Generation**: one temporal backend for the target FPS.
- **NVIDIA Denoise**, **NVIDIA Deblur** and **NVIDIA SDR → HDR**: independent optional stages, including alongside AI/RIFE.

The two upscalers are alternatives, as are the two interpolation engines. Selecting an alternative replaces the previous engine without doubling scale or FPS. Turning the active spatial/temporal engine off restores source scale/cadence; selecting a larger goal re-enables the remembered backend. FPS reductions are not interpolation; choose 0 to retain source cadence.

Source displays the **planned processing order**, including model names, scale/FPS and export format:

`NVIDIA Denoise → NVIDIA Deblur → AI/NVIDIA Upscale → RIFE/NVIDIA Frame Generation → SDR→HDR → Encode`

Disabled stages are omitted. Interpolation at or below source cadence is not represented as an executed step; lower target FPS is rejected. With no stages and source scale/FPS, a plain transcode is valid. The same plan is resolved separately for each file in a batch. Legacy fixed pipeline API/CLI identifiers remain supported for old callers, but are not duplicate studio choices.

## Contextual settings and precision

Only controls relevant to active stages appear. AI/RIFE models and engine settings do not disappear merely because an NVIDIA restoration or HDR stage is also active. NVIDIA quality and strength controls belong to their individual effects. Missing SDK APIs, incompatible source cadence/color and unavailable models are reported rather than replaced with guessed capabilities.

TrueHDR runs last, after AI/upscaling and interpolation, on SDR frames. AI/RIFE use normalized floating-point SDR, not PQ model input. The high-precision path retains uint16 frame storage and suitable 10-bit export with BT.2020/PQ tags. Ten-bit SDR is not HDR. Encoder/container matrices filter choices in the UI and validate requests before inference. Installed API/encoder availability is not a guarantee that all enabled effects will fit GPU memory.

The preview can show a previously selected completed job while the plan is edited. Its processed view identifies that selected job; planned stages and targets are shown only in Source, not repeated beside the player. With no selected video or no loaded live frame, the player displays a text placeholder rather than an empty image element.

## Render a short preview

Enable Preview in Source and set its duration with the seconds slider. This processes the first 1–30 seconds with the selected stages and export settings; shorter sources finish at their actual length. Decode, inference, scheduling and audio/subtitle mux are bounded, not just playback.

Preview files have a distinct duration-marked name and do not replace intended full outputs. Disable Preview for the whole source. Source/Processed playback is selectable. Browser playback can use an SDR-compatible proxy, which is not the original HDR output.

## Progress and ETA

Progress and ETA appear once, on each active Queue row, not in the media preview. Every processing path publishes completed/total output frames, recent throughput, processing stage and measurement time. Preview jobs use the capped total.

Queue waiting, preparation and compilation do not count toward render speed. A five-second-half-life moving average adapts to measured throughput; polling updates remaining time every second. Stale progress hides the estimate. The final frame enters Finalizing while encoding/muxing finishes; only successful worker exit marks Done. Queued and terminal jobs have no ETA.

## Quit and deployment

Quit stops the app instance serving the current browser tab, independent of executable name or root/dist location. Its confirmation warns that active and queued jobs are cancelled. Shutdown rejects new jobs, waits for owned backend cleanup and closes the HTTP server before the app process exits. On Windows, backend cancellation targets its own process tree; Windows binaries are cross-built here, not native-runtime verified. A failed Quit request leaves the UI open and does not claim the app has stopped.

Backend changes are loaded by the next Python worker. The embedded interface requires rebuilding the Go executable and a user-controlled restart. Do not stop active jobs or unrelated GPU services just to activate the interface or make a smoke test fit.
