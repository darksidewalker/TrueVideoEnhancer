'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const T=require('./processing.js');
// Explicit test fixtures, never runtime capability fallbacks.
const caps={processing_modes:['legacy','custom'],spatial_backends:['none','ai','nvidia'],temporal_backends:['none','rife','nvidia'],vfx:{available:true,upscale:true,denoise:true,deblur:true,strength:true,encoding:true,hdr:true,frame_generation:true,frame_generation_encoding:false,quality_levels:['LOW','MEDIUM','HIGH','ULTRA'],streaming:['Streaming Medium']},pipelines:['upscale','nvidia-vfx','restore+upscale'],containers:['mp4','mkv','webm','mov','avi','flv','ts','m4v'],
 codec_matrix:{mp4:['auto','av1_nvenc','x264_nvenc'],mkv:['auto','av1_nvenc','ffv1'],webm:['auto','av1_nvenc','vp9'],mov:['auto','prores'],avi:['auto','libx264'],flv:['auto','libx264'],ts:['auto','x265_nvenc'],m4v:['auto','x265_nvenc']},
 hdr_codec_matrix:{mp4:['auto','av1_nvenc'],mkv:['auto','ffv1'],webm:['auto','av1_nvenc'],mov:['auto','prores'],ts:['auto','x265_nvenc'],m4v:['auto','x265_nvenc']},
 precise_codec_matrix:{mp4:['auto','av1_nvenc'],mkv:['auto','ffv1'],webm:['auto','vp9'],mov:['auto','prores'],ts:['auto','x265_nvenc'],m4v:['auto','x265_nvenc']}};
const models=[{id:'mixed-2x',name:'Mixed 2x',file:'mixed-2x.safetensors',category:'upscaler',subcategory:'mixed',present:true,destination:'/models/mixed2.safetensors'},
{id:'anime-2x',name:'Anime 2x',file:'anime-2x.safetensors',category:'upscaler',subcategory:'anime',present:true,destination:'/models/anime2.safetensors'},
{id:'rife-general',name:'RIFE general',category:'interpolation',present:true,destination:'/models/rife.safetensors'}];
const probe={input_path:'/video/a.mp4',width:640,height:360,r_frame_rate:'24000/1001',duration:8,codec_name:'h264'};
const s=overrides=>({...T.defaults,...overrides});
for(const [internal,final] of [[2,1],[4,1],[4,2]])for(const filter of ['area','lanczos','bicubic']) {
 const settings=s({spatialBackend:'nvidia',rtxUpscaleIntermediate:internal,rtxDownsampleFilter:filter,scale:final,targetFps:0});
 const r=T.request(settings,probe,caps,models,'/video/a.mp4');
 assert.equal(r.spatial_backend,'nvidia');assert.equal(r.rtx_upscale_intermediate,internal);assert.equal(r.rtx_downsample_filter,filter);
 assert.equal(T.goal(settings,'scale',1).spatialBackend,'nvidia');
 assert.match(T.stages(settings,probe,models).join(' → '),new RegExp(`NVIDIA Upscale .*${internal}x source, internal.*Downsample.*${filter}.*final ${final}x`));
 assert.equal(T.migrate(JSON.parse(JSON.stringify(settings))).rtxUpscaleIntermediate,internal);
}
for(const spatial of ['ai','none']) {
 const r=T.request(s({spatialBackend:spatial,scale:spatial==='ai'?2:1,targetFps:0,rtxUpscaleIntermediate:4,rtxDownsampleFilter:'lanczos'}),probe,caps,models,'in');
 assert.equal(r.rtx_upscale_intermediate,0);assert.equal(r.rtx_downsample_filter,'');
}
for(const change of [{rtxUpscaleIntermediate:2,scale:2},{rtxUpscaleIntermediate:4,scale:4},{rtxUpscaleIntermediate:3},{rtxUpscaleIntermediate:NaN},{rtxDownsampleFilter:'nearest'}]) {
 assert.throws(()=>T.request(s({spatialBackend:'nvidia',rtxUpscaleIntermediate:4,scale:2,...change}),probe,caps,models,'in'),/Internal NVIDIA|downsampling/);
}
assert.equal(T.normalize(s({spatialBackend:'ai',scale:1,rtxUpscaleIntermediate:4})).spatialBackend,'none');
assert.equal(T.normalize(s({spatialBackend:'nvidia',scale:1,rtxUpscaleIntermediate:0})).spatialBackend,'none');

const req=(overrides={},p=probe,c=caps)=>T.request(s(overrides),p,c,models,'/video/a.mp4');
assert.equal(req().processing_mode,'custom');assert.equal(req().pipeline,'upscale');assert.equal(req().spatial_backend,'ai');assert.equal(req().temporal_backend,'rife');
assert.equal(req().upscale_model,'/models/mixed2.safetensors');
assert.equal(req({contentType:'anime'}).upscale_model,'/models/anime2.safetensors');
assert.equal(req({targetFps:0}).rife_model,'');assert.throws(()=>req({targetFps:20}),/cannot reduce source FPS/);assert.equal(req({scale:1}).upscale_model,'');
assert.equal(req({targetFps:60},{...probe,r_frame_rate:'60/1'}).temporal_backend,'none');
assert.equal(req({targetFps:60},{...probe,r_frame_rate:'60/1'}).rife_model,'');
assert.equal(req({scale:1,targetFps:0}).rtx_frame_generation,false); // Plain transcode is valid.
for(const spatial of ['none','ai','nvidia'])for(const temporal of ['none','rife','nvidia']){
 const options={spatialBackend:spatial,temporalBackend:temporal,scale:spatial==='none'?1:2,targetFps:temporal==='none'?0:60,denoise:true,deblur:true,rtxHDR:true};
 const r=req(options);assert.equal(r.spatial_backend,spatial);assert.equal(r.temporal_backend,temporal);
 assert.equal(r.denoise,true);assert.equal(r.deblur,true);assert.equal(r.rtx_hdr,true);assert.equal(r.video_pixel_format,'yuv420p10le');
 assert.equal(!!r.upscale_model,spatial==='ai');assert.equal(!!r.rife_model,temporal==='rife');assert.equal(r.rtx_frame_generation,temporal==='nvidia');assert.equal(r.rtx_upscale,spatial==='nvidia'?'VSR':'Off');
 const plan=T.stages(s(options),probe,models);assert.equal(plan[0],'NVIDIA Denoise');assert.equal(plan[1],'NVIDIA Deblur');
 assert.equal(plan.at(-2),'SDR→HDR');assert.match(plan.at(-1),/10-bit PQ/);
 if(spatial!=='none'&&temporal!=='none')assert.ok(plan.findIndex(x=>/Upscale/.test(x))<plan.findIndex(x=>/fps/.test(x)));
}
let selection=T.toggle(s({scale:1,targetFps:0}),'nvidiaUpscale',probe);
assert.equal(selection.scale,2);assert.equal(selection.spatialBackend,'nvidia');
selection=T.toggle(selection,'aiUpscale',probe);assert.equal(selection.spatialBackend,'ai');assert.equal(selection.scale,2);
selection=T.toggle(selection,'aiUpscale',probe);assert.equal(selection.spatialBackend,'none');assert.equal(selection.scale,1);
selection=T.goal(selection,'scale',4);assert.equal(selection.spatialBackend,'ai');
selection=T.toggle(selection,'nvidiaFrameGeneration',probe);assert.equal(selection.targetFps,60);assert.equal(selection.temporalBackend,'nvidia');
selection=T.toggle(selection,'rifeInterpolation',probe);assert.equal(selection.temporalBackend,'rife');
selection=T.toggle(selection,'rifeInterpolation',probe);assert.equal(selection.targetFps,0);assert.equal(selection.temporalBackend,'none');
assert.equal(T.goal(selection,'targetFps',120).temporalBackend,'rife');
assert.equal(T.toggle(s({targetFps:0}),'rifeInterpolation',{r_frame_rate:'60/1'}).targetFps,120);
const manual=s({upscaleModel:'anime-2x',rifeModel:'rife-general',rtxDenoiseStrength:0,rtxAutoShotDetection:false,rtxHDRDebanding:false});
assert.equal(T.toggle(manual,'nvidiaUpscale',probe).upscaleModel,'anime-2x');assert.equal(T.toggle(manual,'denoise',probe).rifeModel,'rife-general');
assert.equal(req({...manual,denoise:true}).rtx_denoise_strength,0);assert.equal(req(manual).rtx_auto_shot_detection,false);assert.equal(req(manual).rtx_hdr_debanding,false);
for(const pipeline of ['upscale','nvidia-vfx','restore+upscale','restore']){
 const legacy={pipeline,upscaleModel:'anime-2x',rifeModel:'rife-general',rtxDenoiseStrength:0,rtxAutoShotDetection:false,scale:2,targetFps:60};
 const migrated=T.migrate(legacy);assert.equal(migrated.upscaleModel,'anime-2x');assert.equal(migrated.rifeModel,'rife-general');assert.equal(migrated.rtxDenoiseStrength,0);assert.equal(migrated.rtxAutoShotDetection,false);
 assert.equal(migrated.spatialBackend,pipeline==='restore'?'none':pipeline==='nvidia-vfx'?'nvidia':'ai');assert.equal(migrated.temporalBackend,pipeline==='restore'?'none':pipeline==='nvidia-vfx'?'nvidia':'rife');
}
assert.equal(T.migrate({pipeline:'nvidia-vfx',scale:1,targetFps:0,denoise:false}).denoise,false);
assert.throws(()=>req({},null),/source FPS/);assert.throws(()=>req({},probe,null),/Capabilities unavailable/);
assert.throws(()=>req({},probe,{...caps,processing_modes:['legacy']}),/does not support custom/);
assert.throws(()=>req({denoise:true}, {...probe,color_transfer:'smpte2084'}),/SDR input/);
assert.throws(()=>req({rtxHDR:true}, {...probe,avg_frame_rate:'20/1'}),/constant-frame/);
assert.throws(()=>req({temporalBackend:'nvidia',rtxImageEncoding:'rgb10a2'}),/cannot use 10-bit/);
assert.equal(req({temporalBackend:'nvidia',rtxHDR:true}).rtx_hdr,true); // HDR is after FG, not its working format.
assert.throws(()=>req({rtxHDR:true,outputContainer:'avi'}),/compatible/);
assert.throws(()=>req({denoise:true},probe,{...caps,vfx:{...caps.vfx,denoise:false}}),/feature is unavailable/);
assert.throws(()=>req({spatialBackend:'nvidia'},probe,{...caps,vfx:{...caps.vfx,upscale:undefined}}),/feature is unavailable/);
assert.throws(()=>req({rtxHDR:true,rtxHDRLuminance:0}),/400 and 2000/);
assert.throws(()=>req({rtxHDR:true},probe,{...caps,vfx:{...caps.vfx,encoding:false}}),/10-bit SDK precision/);
assert.match(T.featureUnavailable('nvidiaUpscale',s(),{...probe,field_order:'tt'},caps,models),/progressive/);
assert.match(T.featureUnavailable('aiUpscale',s(),probe,caps,[]),/Download/);
assert.match(T.featureUnavailable('rifeInterpolation',s(),probe,caps,[]),/Download/);
assert.throws(()=>req({scale:4,upscaleModel:'mixed-2x'}),/matching content/);
assert.throws(()=>req({spatialBackend:'nvidia',rtxUpscale:'Streaming Ultra'}),/mode is unavailable/);
for(const container of caps.containers){
 const config=s({rtxHDR:true,outputContainer:container});assert.deepEqual(T.codecChoices(config,caps),caps.hdr_codec_matrix[container]||[]);
 for(const codec of T.codecChoices(config,caps))assert.equal(req({...config,videoEncoderPreset:codec}).video_pixel_format,'yuv420p10le');
}
assert.equal(req({preview:true,previewSeconds:30}).preview_seconds,8);assert.equal(req({preview:true,previewSeconds:0}).preview_seconds,1);
assert.equal(req({preview:true,previewSeconds:NaN}).preview_seconds,5);assert.equal(req({preview:false,previewSeconds:3}).preview,false);
assert.equal(req().start_time,undefined);assert.equal(req().end_time,undefined);assert.equal(req().preset,undefined);
assert.equal(T.outputPath(req({preview:true})),'/video/a-enhanced.mp4');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8'), app=fs.readFileSync(path.join(__dirname,'app.js'),'utf8'),queue=fs.readFileSync(path.join(__dirname,'queue-ui.js'),'utf8');
for(const removed of ['tuneDialog','openTune','name="preset"','id="pipeline"','id="upscaleFeature"','id="fgFeature"','cfgScale','benchmarkMode'])assert.ok(!html.includes(removed),removed);
assert.ok(html.indexOf('settings-panel')<html.indexOf('preview-panel'));assert.ok(!queue.includes('function captureJobRequest'));
for(const match of [...app.matchAll(/\$\("([^"]+)"\)/g),...queue.matchAll(/\$\("([^"]+)"\)/g)]){
 const id=match[1];assert.ok(html.includes(`id="${id}"`)||queue.includes(`id = "${id}"`),`Missing canonical HTML ID ${id}`);
}
console.log('Frontend composable contract passed: all spatial/temporal pairs, restoration/HDR ordering, model isolation, exclusive selection, goals, capabilities, codec matrices, legacy migration and preview/full.');
module.exports={caps,models,probe};
