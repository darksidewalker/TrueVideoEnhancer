'use strict';
// Native, keyboard-accessible help for controls that exist in the studio.
const controlHelp={
  aiUpscale:'Upscale with an installed AI model. Replaces NVIDIA Upscale, without changing restoration or interpolation.',
  rifeInterpolation:'Interpolate with an installed RIFE model. Replaces NVIDIA Frame Generation.',
  nvidiaUpscale:'Upscale with the NVIDIA SDK. Replaces AI Upscale.',
  nvidiaFrameGeneration:'Generate intermediate frames with the NVIDIA SDK. Replaces RIFE Interpolation.',
  targetFps:'0 preserves source cadence and turns off interpolation. Increasing the target restores the last chosen temporal feature.',
  contentType:'Routes AI models by content. NVIDIA SDK processing has no content-specific modes.',
  scale:'A scale above 1× requires upscaling. Turn off Upscale to restore 1×.',
  preview:'Preview runs the first selected seconds using the same processing settings. Turn it off to render the entire video.',
  rtxHDR:'Converts SDR to BT.2020 / PQ HDR. Only available on NVIDIA with compatible SDK, encoder and container.',
  rtxImageEncoding:'10-bit SDR is not HDR. Both require a precision-compatible encoder and container.',
  crf:'Lower values retain more detail and usually increase output size.',
  tileSize:'0 selects automatic full-frame inference with a tiled fallback. Set a non-zero tile size to override.',
};
for(const [id,text] of Object.entries(controlHelp)){const el=document.getElementById(id);if(el)el.title=text;}
