'use strict';
// Canonical composable processing contract shared by single and per-input batch jobs.
const TVE = (() => {
  const defaults = {processingMode:'custom', spatialBackend:'ai', temporalBackend:'rife', lastSpatialBackend:'ai', lastTemporalBackend:'rife',
    contentType:'mixed', outputContainer:'mp4', scale:2, targetFps:60,
    backend:'tensorrt', upscaleModel:'', rifeModel:'', videoEncoderPreset:'auto', crf:18, tileSize:0,
    denoise:false, deblur:false, denoiseQuality:'Ultra', deblurQuality:'Ultra', rtxUpscaleQuality:'Ultra',
    rtxUpscale:'VSR', rtxFGMode:'MEDIUM', rtxAutoShotDetection:true,
    rtxImageEncoding:'rgb8', rtxDenoiseStrength:1, rtxDeblurStrength:1, rtxUpscaleStrength:1,
    rtxHDR:false, rtxHDRContrast:100, rtxHDRSaturation:100, rtxHDRMiddleGray:50, rtxHDRLuminance:650,
    rtxHDRDebanding:true, preview:false, previewSeconds:5};
  const rate = value => { const [a,b] = String(value || 0).split('/').map(Number); return b ? a/b : Number.isFinite(a) ? a : 0; };
  const scaleOf = m => Number(`${m.id} ${m.name} ${m.file}`.match(/(?:^|[^0-9])([24])x(?:[^0-9]|$)/)?.[1] || 0);
  const sourceFPS = p => rate(p?.avg_frame_rate || p?.r_frame_rate);
  const increased = (s,p) => sourceFPS(p) > 0 && Number(s.targetFps) > sourceFPS(p) + 1e-6;
  const hasSDK = s => s.denoise || s.deblur || s.rtxHDR || s.spatialBackend==='nvidia' || s.temporalBackend==='nvidia';
  function migrate(saved = {}) {
    const s={...defaults};
    for (const key of Object.keys(defaults)) if (Object.hasOwn(saved,key)) s[key]=saved[key];
    if (!Object.hasOwn(saved,'spatialBackend')) {
      const native=saved.pipeline==='nvidia-vfx', restore=saved.pipeline==='restore';
      s.spatialBackend=restore || Number(s.scale)===1?'none':native?'nvidia':'ai';
      s.temporalBackend=restore || Number(s.targetFps)===0?'none':native?'nvidia':'rife';
      if (restore) { s.scale=1; s.targetFps=0; s.denoise=true; }
      // Earlier AI-only forms hid these SDK options; they were not active processing stages.
      if (saved.pipeline==='upscale') { s.denoise=false; s.deblur=false; s.rtxHDR=false; }
      if (saved.pipeline==='restore+upscale') s.rtxHDR=false;
      s.lastSpatialBackend=native?'nvidia':'ai'; s.lastTemporalBackend=native?'nvidia':'rife';
    }
    return normalize(s);
  }
  function normalize(settings) {
    const s={...defaults,...settings};
    if (!['none','ai','nvidia'].includes(s.spatialBackend)) s.spatialBackend='none';
    if (!['none','rife','nvidia'].includes(s.temporalBackend)) s.temporalBackend='none';
    if (Number(s.scale)===1) s.spatialBackend='none';
    if (Number(s.targetFps)===0) s.temporalBackend='none';
    if (s.spatialBackend!=='none') s.lastSpatialBackend=s.spatialBackend;
    if (s.temporalBackend!=='none') s.lastTemporalBackend=s.temporalBackend;
    if (s.spatialBackend==='nvidia' && s.rtxUpscale==='Off') s.rtxUpscale='VSR';
    delete s.pipeline; delete s.rtxFrameGeneration;
    return s;
  }
  function goal(settings,key,value) {
    const s={...settings,[key]:value};
    if (key==='scale') s.spatialBackend=Number(value)===1?'none':settings.spatialBackend!=='none'?settings.spatialBackend:settings.lastSpatialBackend||'ai';
    if (key==='targetFps') s.temporalBackend=Number(value)===0?'none':settings.temporalBackend!=='none'?settings.temporalBackend:settings.lastTemporalBackend||'rife';
    return normalize(s);
  }
  function toggle(settings,feature,p) {
    const s=normalize(settings);
    const spatial={aiUpscale:'ai',nvidiaUpscale:'nvidia'}, temporal={rifeInterpolation:'rife',nvidiaFrameGeneration:'nvidia'};
    if (spatial[feature]) {
      const selected=spatial[feature]; s.lastSpatialBackend=selected;
      if (s.spatialBackend===selected) { s.spatialBackend='none'; s.scale=1; }
      else { s.spatialBackend=selected; if (!(Number(s.scale)>1)) s.scale=2; }
    } else if (temporal[feature]) {
      const selected=temporal[feature]; s.lastTemporalBackend=selected;
      if (s.temporalBackend===selected) { s.temporalBackend='none'; s.targetFps=0; }
      else { s.temporalBackend=selected; if (!(Number(s.targetFps)>sourceFPS(p))) s.targetFps=sourceFPS(p)>=60?sourceFPS(p)*2:60; }
    } else if (['denoise','deblur','rtxHDR'].includes(feature)) s[feature]=!s[feature];
    return normalize(s);
  }
  function model(category,s,models) {
    const available = models.filter(m => m.category===category && m.present && m.destination && !/heavy/i.test(`${m.id} ${m.name} ${m.file}`));
    const manual = category==='upscaler' ? s.upscaleModel : s.rifeModel;
    if (manual) return available.find(m => m.id===manual) || null;
    if (category==='interpolation') return available[0] || null;
    const preferred = {anime:{2:'anime-sharp-v4-2x',4:'nomosuni-span-anime-4x'},mixed:{2:'nomosuni-span-2x',4:'nomosuni-span-4x'},realism:{2:'realplksr-restoration-2x',4:'clearreality-4x'}};
    const exact = available.filter(m => scaleOf(m)===Number(s.scale));
    return exact.find(m => m.id===preferred[s.contentType]?.[s.scale]) || exact.find(m => m.subcategory===s.contentType)
      || (s.contentType==='mixed' ? exact.find(m => m.subcategory==='realism') : null) || null;
  }
  function codecChoices(s,caps) {
    if (!caps) return [];
    const matrix=s.rtxHDR ? caps.hdr_codec_matrix : hasSDK(s) && s.rtxImageEncoding==='rgb10a2' ? caps.precise_codec_matrix : caps.codec_matrix;
    const supported=['auto','av1_nvenc','x265_nvenc','x264_nvenc','av1','libx265','libx264','vp9','prores','ffv1'];
    return Array.isArray(matrix?.[s.outputContainer]) ? matrix[s.outputContainer].filter(codec=>supported.includes(codec)) : [];
  }
  function sdkSourceIssue(p) {
    if (['smpte2084','arib-std-b67'].includes(p?.color_transfer)) return 'NVIDIA expects SDR input. Tone-map this HDR source first.';
    if (p?.field_order && !['unknown','progressive'].includes(p.field_order)) return 'NVIDIA requires progressive source frames. Deinterlace this source first.';
    if (p?.vfr || (p?.avg_frame_rate && p?.r_frame_rate && Math.abs(rate(p.r_frame_rate)-rate(p.avg_frame_rate))>0.001)) return 'NVIDIA requires a constant-frame-rate source.';
    return '';
  }
  function featureUnavailable(feature,s,p,caps,models=[]) {
    if (!caps) return 'Runtime capability discovery required';
    if (!caps.processing_modes?.includes('custom')) return 'This backend does not support custom processing. Update the runtime.';
    if (feature==='aiUpscale') return models.some(m=>m.category==='upscaler' && m.present && m.destination && !/heavy/i.test(`${m.id} ${m.name}`)) && (!caps.spatial_backends || caps.spatial_backends.includes('ai'))?'':'Download an available AI upscale model in Models';
    if (feature==='rifeInterpolation') return model('interpolation',s,models) && (!caps.temporal_backends || caps.temporal_backends.includes('rife'))?'':'Download an available general RIFE model in Models';
    if (feature==='nvidiaUpscale' && caps.spatial_backends && !caps.spatial_backends.includes('nvidia')) return 'NVIDIA upscale backend unavailable';
    if (feature==='nvidiaFrameGeneration' && caps.temporal_backends && !caps.temporal_backends.includes('nvidia')) return 'NVIDIA temporal backend unavailable';
    const v=caps.vfx || {};
    if (!v.available) return v.error || 'NVIDIA SDK unavailable';
    const source=sdkSourceIssue(p); if (source) return source;
    const flags={nvidiaUpscale:'upscale',nvidiaFrameGeneration:'frame_generation',denoise:'denoise',deblur:'deblur',rtxHDR:'hdr'};
    if (!v[flags[feature]]) return 'This NVIDIA feature is unavailable';
    if (feature==='rtxHDR' && !v.encoding) return 'HDR output requires unavailable 10-bit SDK precision';
    if (feature==='nvidiaFrameGeneration' && s.rtxImageEncoding==='rgb10a2' && !v.frame_generation_encoding) return 'Frame generation cannot use 10-bit working precision on this SDK';
    return '';
  }
  function previewSeconds(s,p) {
    const n=Number(s.previewSeconds);
    return Math.max(1,Math.min(30,Number.isFinite(n)?n:5,Number(p?.duration)>0?Math.max(1,Number(p.duration)):30));
  }
  function issues(settings,p,caps,models,requireSource=true) {
    const s=normalize(settings), errors=[], sdk=hasSDK(s), v=caps?.vfx || {};
    if (!caps) errors.push('Capabilities unavailable. Retry discovery before running.');
    else {
      if (!caps.processing_modes?.includes('custom')) errors.push('This backend does not support custom processing. Update the runtime.');
      if (caps.spatial_backends && !caps.spatial_backends.includes(s.spatialBackend)) errors.push('Selected spatial backend is unavailable.');
      if (caps.temporal_backends && !caps.temporal_backends.includes(s.temporalBackend)) errors.push('Selected temporal backend is unavailable.');
      if (!codecChoices(s,caps).includes(s.videoEncoderPreset)) errors.push('Choose a compatible available codec and container.');
    }
    if (requireSource && (!p || !(sourceFPS(p)>0))) errors.push('Select a readable video to determine source FPS.');
    if (![1,2,4].includes(Number(s.scale))) errors.push('Select 1×, 2× or 4× scale.');
    if (!Number.isFinite(Number(s.targetFps)) || Number(s.targetFps)<0) errors.push('Target FPS must be finite and non-negative.');
    if (Number(s.targetFps)>0 && Number(s.targetFps)<sourceFPS(p)-1e-6) errors.push('Interpolation cannot reduce source FPS. Use 0 to preserve source cadence.');
    if (!Number.isFinite(Number(s.crf)) || Number(s.crf)<0 || Number(s.crf)>51) errors.push('CRF / CQ must be between 0 and 51.');
    if ((s.spatialBackend==='ai'||s.temporalBackend==='rife') && (!Number.isFinite(Number(s.tileSize)) || Number(s.tileSize)<0)) errors.push('Tile size must be non-negative.');
    if (Number(s.scale)>1 && s.spatialBackend==='none') errors.push('Select an upscale feature for the output scale.');
    if (increased(s,p) && s.temporalBackend==='none') errors.push('Select an interpolation feature for the target FPS.');
    if (s.spatialBackend==='ai' && (!model('upscaler',s,models) || scaleOf(model('upscaler',s,models))!==Number(s.scale))) errors.push('Choose or download a matching content / scale model in Models.');
    if (s.temporalBackend==='rife' && increased(s,p) && !model('interpolation',s,models)) errors.push('Download a general RIFE model in Models for this FPS increase.');
    if (sdk) {
      if (!v.available) errors.push(v.error || 'NVIDIA effects are unavailable.');
      const source=sdkSourceIssue(p); if (source) errors.push(source);
      for (const [on,feature] of [[s.denoise,'denoise'],[s.deblur,'deblur'],[s.rtxHDR,'rtxHDR'],[s.spatialBackend==='nvidia','nvidiaUpscale'],[s.temporalBackend==='nvidia' && increased(s,p),'nvidiaFrameGeneration']]) {
        if (on) { const why=featureUnavailable(feature,s,p,caps,models); if (why && why!==source && v.available) errors.push(why); }
      }
      if (s.rtxImageEncoding==='rgb10a2' && !v.encoding) errors.push('10-bit working precision is unavailable.');
      const quality=v.quality_levels || [];
      for (const [enabled,value] of [[s.denoise,s.denoiseQuality],[s.deblur,s.deblurQuality],[s.spatialBackend==='nvidia',s.rtxUpscaleQuality]]) {
        if (enabled && !quality.some(q=>q.toLowerCase()===String(value).toLowerCase())) errors.push('Selected SDK quality level is unavailable.');
      }
      if (v.strength) for (const [on,key] of [[s.denoise,'rtxDenoiseStrength'],[s.deblur,'rtxDeblurStrength'],[s.spatialBackend==='nvidia','rtxUpscaleStrength']]) {
        if (on && (!Number.isFinite(Number(s[key])) || Number(s[key])<0 || Number(s[key])>1)) errors.push('Effect strengths must be between 0 and 1.');
      }
      if (s.spatialBackend==='nvidia' && s.rtxUpscale.startsWith('Streaming') && !v.streaming?.includes(s.rtxUpscale)) errors.push('Selected NVIDIA upscale mode is unavailable.');
    }
    if (s.rtxHDR) for (const [key,lo,hi] of [['rtxHDRContrast',0,200],['rtxHDRSaturation',0,200],['rtxHDRMiddleGray',10,100],['rtxHDRLuminance',400,2000]]) {
      if (!Number.isFinite(Number(s[key])) || Number(s[key])<lo || Number(s[key])>hi) errors.push(`HDR setting ${key} must be between ${lo} and ${hi}.`);
    }
    return [...new Set(errors)];
  }
  function stages(settings,p,models=[]) {
    const s=normalize(settings), result=[];
    if (s.denoise) result.push('NVIDIA Denoise');
    if (s.deblur) result.push('NVIDIA Deblur');
    if (s.spatialBackend!=='none') {
      const m=s.spatialBackend==='ai'?model('upscaler',s,models):null;
      result.push(`${s.spatialBackend==='ai'?'AI Upscale':'NVIDIA Upscale'} (${s.scale}x${m?', '+m.name:''})`);
    }
    if (s.temporalBackend!=='none' && increased(s,p)) {
      const m=s.temporalBackend==='rife'?model('interpolation',s,models):null;
      result.push(`${s.temporalBackend==='rife'?'RIFE':'NVIDIA Frame Generation'} (${s.targetFps}fps${m?', '+m.name:''})`);
    }
    if (s.rtxHDR) result.push('SDR→HDR');
    const codec={auto:'Auto codec',av1_nvenc:'AV1',av1:'AV1',x265_nvenc:'HEVC',libx265:'HEVC',x264_nvenc:'H.264',libx264:'H.264',vp9:'VP9',prores:'ProRes',ffv1:'FFV1'}[s.videoEncoderPreset] || s.videoEncoderPreset;
    result.push(`${codec}/${s.outputContainer.toUpperCase()}${s.rtxHDR?' · 10-bit PQ':hasSDK(s)&&s.rtxImageEncoding==='rgb10a2'?' · 10-bit SDR':''}`);
    return result;
  }
  function request(settings,p,caps,models,input,requireSource=true) {
    const s=normalize(settings), errors=issues(s,p,caps,models,requireSource);
    if (errors.length) throw new Error(errors.join('\n'));
    const ai=s.spatialBackend==='ai'||s.temporalBackend==='rife', sdk=hasSDK(s);
    // Custom dispatch owns the graph; the legacy field is a safe compatibility default, never a UI choice.
    return {input,output:'',output_container:s.outputContainer,pipeline:'upscale',processing_mode:'custom',
      spatial_backend:s.spatialBackend,temporal_backend:increased(s,p)?s.temporalBackend:'none',backend:ai?s.backend:'tensorrt',
      content_type:s.contentType,scale:Number(s.scale),target_fps:Number(s.targetFps),
      upscale_model:s.spatialBackend==='ai'?model('upscaler',s,models)?.destination || '':'',
      rife_model:s.temporalBackend==='rife' && increased(s,p)?model('interpolation',s,models)?.destination || '':'',
      video_encoder_preset:s.videoEncoderPreset,video_pixel_format:s.rtxHDR || (sdk && s.rtxImageEncoding==='rgb10a2')?'yuv420p10le':'yuv420p',
      crf:String(s.crf),audio_encoder_preset:'copy_audio',subtitle_encoder_preset:'copy_subtitle',audio_bitrate:'192k',
      scene_detect_method:s.temporalBackend==='rife'?'pyscenedetect':'none',scene_detect_threshold:s.temporalBackend==='rife'?4:0,tile_size:ai?Number(s.tileSize):0,
      denoise:!!s.denoise,deblur:!!s.deblur,denoise_quality:s.denoiseQuality,deblur_quality:s.deblurQuality,
      rtx_upscale:s.spatialBackend==='nvidia'?s.rtxUpscale:'Off',rtx_upscale_quality:s.rtxUpscaleQuality,vsr_strength:0,hdr_mode:false,
      rtx_frame_generation:s.temporalBackend==='nvidia' && increased(s,p),rtx_fg_mode:s.rtxFGMode,
      rtx_auto_shot_detection:!!s.rtxAutoShotDetection,rtx_shot_change:false,rtx_image_encoding:sdk?s.rtxImageEncoding:'rgb8',
      rtx_denoise_strength:caps.vfx?.strength?Number(s.rtxDenoiseStrength):1,rtx_deblur_strength:caps.vfx?.strength?Number(s.rtxDeblurStrength):1,rtx_upscale_strength:caps.vfx?.strength?Number(s.rtxUpscaleStrength):1,
      rtx_hdr:!!s.rtxHDR,rtx_hdr_contrast:Number(s.rtxHDRContrast),rtx_hdr_saturation:Number(s.rtxHDRSaturation),
      rtx_hdr_middle_gray:Number(s.rtxHDRMiddleGray),rtx_hdr_luminance:Number(s.rtxHDRLuminance),rtx_hdr_debanding:!!s.rtxHDRDebanding,
      preview:!!s.preview,preview_seconds:previewSeconds(s,p)};
  }
  // Backend owns the preview suffix. Never add it here or cap a full render.
  function outputPath(req) {
    const path=String(req.input||''), slash=Math.max(path.lastIndexOf('/'),path.lastIndexOf('\\'));
    const dir=path.slice(0,slash+1), file=path.slice(slash+1), dot=file.lastIndexOf('.');
    const name=dot>0?file.slice(0,dot):file;
    return name ? `${dir}${name}-enhanced.${req.output_container}` : '';
  }
  return {defaults,rate,scaleOf,sourceFPS,increased,migrate,model,codecChoices,featureUnavailable,hasSDK,normalize,goal,toggle,stages,previewSeconds,issues,request,outputPath};
})();
if (typeof module!=='undefined') module.exports=TVE;
