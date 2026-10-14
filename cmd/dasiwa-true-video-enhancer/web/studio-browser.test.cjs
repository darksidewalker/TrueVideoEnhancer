'use strict';
// Production frontend browser smoke with explicit API fixtures; never touches the live server.
// PLAYWRIGHT_MODULE=/path/to/playwright node .../studio-browser.test.cjs
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {caps,models,probe}=require('./vfx-ui.test.cjs');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'/home/darksidewalker/.hermes/hermes-agent/node_modules/playwright');
const requests=[],jobs=[];
let failCapabilities=false, legacyCapabilities=false;
const server=http.createServer((req,res)=>{
 const url=new URL(req.url,'http://localhost');
 const json=data=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data));};
 if(url.pathname==='/api/capabilities'){if(failCapabilities){res.statusCode=503;return json({error:'SDK discovery offline'});}return json(legacyCapabilities?{...caps,processing_modes:['legacy']}:caps);}
 if(url.pathname==='/api/runtime/status')return json({python:{available:true},nvidia_smi:{available:true}});
 if(url.pathname==='/api/runtime/models')return json({models});
 if(url.pathname==='/api/probe')return json({...probe,input_path:url.searchParams.get('path'),r_frame_rate:url.searchParams.get('path').includes('60fps')?'60/1':probe.r_frame_rate});
 if(url.pathname==='/api/batch-folder')return json({items:[{name:'a.mp4',path:'/video/a.mp4'},{name:'60fps.mp4',path:'/video/60fps.mp4'},{name:'sub/a.mp4',path:'/video/sub/a.mp4'},{name:'sub/a.mkv',path:'/video/sub/a.mkv'}]});
 if(url.pathname==='/api/jobs' && req.method==='GET')return json(jobs);
 if((url.pathname==='/api/jobs'||url.pathname==='/api/jobs/batch')&&req.method==='POST'){
  let body='';req.on('data',b=>body+=b);req.on('end',()=>{const data=JSON.parse(body);requests.push({url:url.pathname,body:data});const job={id:String(jobs.length+1),input:data.input||data.inputs[0],output:'/video/a-enhanced'+(data.preview?'[preview_5s]':'')+'.mp4',status:'done',logs:['output_frames=10','wrote 10/10 frames']};jobs.push(job);json(url.pathname.endsWith('/batch')?{jobs:[job],skipped:[]}:job);});return;
 }
 const id=url.pathname.match(/^\/api\/jobs\/(\d+)$/)?.[1];if(id)return json(jobs.find(j=>j.id===id));
 // Optional real browser-playable synthetic media fixture, not inference output.
 if(process.env.TVE_BROWSER_MEDIA && (url.pathname==='/api/stream' || /^\/api\/jobs\/\d+\/preview$/.test(url.pathname))){
  res.setHeader('Content-Type','video/mp4');return res.end(fs.readFileSync(process.env.TVE_BROWSER_MEDIA));
 }
 if(url.pathname.startsWith('/api/')){res.statusCode=404;return json({error:'Media intentionally omitted in frontend fixture'});}
 const file=path.join(__dirname,url.pathname==='/'?'index.html':path.basename(url.pathname));
 if(!fs.existsSync(file)){res.statusCode=404;return res.end();}
 res.setHeader('Content-Type',file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':'text/html');res.end(fs.readFileSync(file));
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const browser=await chromium.launch({headless:true});
 try{
 const page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(`http://127.0.0.1:${server.address().port}`);
 await page.waitForFunction(()=>document.querySelector('#runtimeSummary').textContent.includes('Python'));
 assert.equal(await page.locator('#runJob').isDisabled(),true);
 await page.locator('#input').fill('/video/a.mp4'); await page.locator('#input').press('Tab');
 await page.waitForFunction(()=>!document.querySelector('#runJob').disabled);
 assert.equal(await page.locator('.settings-panel').isVisible(),true);
 assert.equal(await page.locator('#rifeInterpolation').getAttribute('aria-pressed'),'true');
 assert.deepEqual(await page.locator('.studio-grid > .panel').evaluateAll(nodes=>nodes.slice(0,3).map(n=>n.className)),['panel source-panel','panel settings-panel','panel preview-panel']);
 assert.equal(await page.locator('.source-panel #scale,.source-panel #targetFps').count(),0);
 assert.equal(await page.locator('.settings-panel #scale,.settings-panel #targetFps').count(),2);
 assert.equal(await page.locator('#pipeline').count(),0);
 await page.locator('#upscaleModel').selectOption('mixed-2x');await page.locator('#rifeModel').selectOption('rife-general');
 await page.locator('#nvidiaUpscale').click();await page.locator('#nvidiaFrameGeneration').click();
 assert.equal(await page.locator('#aiUpscale').getAttribute('aria-pressed'),'false');
 assert.equal(await page.locator('#rifeInterpolation').getAttribute('aria-pressed'),'false');
 assert.equal(await page.locator('#upscaleModel').inputValue(),'mixed-2x');assert.equal(await page.locator('#rifeModel').inputValue(),'rife-general');
 assert.equal(await page.locator('#aiControls').isVisible(),false);
 assert.equal(await page.locator('#contentType').isVisible(),false);
 assert.equal(await page.locator('#nvidiaUpscale').getAttribute('aria-pressed'),'true');
 await page.locator('#denoise').click();await page.locator('#rtxDenoiseStrength').fill('0');
 await page.locator('#rtxAutoShotDetection').uncheck();
 await page.locator('#preview').check();
 await page.locator('#previewSeconds').fill('5');
 await page.locator('#runJob').click(); await page.waitForFunction(()=>document.querySelector('#processedPlayback').disabled===false);
 assert.equal(requests[0].body.processing_mode,'custom');assert.equal(requests[0].body.spatial_backend,'nvidia');assert.equal(requests[0].body.temporal_backend,'nvidia');
 assert.equal(requests[0].body.preview,true);assert.equal(requests[0].body.preview_seconds,5);
 assert.equal(requests[0].body.rtx_denoise_strength,0);assert.equal(requests[0].body.rtx_auto_shot_detection,false);
 assert.equal(requests[0].body.rtx_frame_generation,true); assert.equal(requests[0].body.upscale_model,'');
 assert.ok(!requests[0].body.output.includes('preview'));assert.equal(requests[0].body.preset,undefined);
 await page.locator('#processedPlayback').click();assert.ok((await page.locator('#videoPreview').getAttribute('src')).includes('/api/jobs/1/preview'));
 await page.locator('#sourcePlayback').click();assert.ok((await page.locator('#videoPreview').getAttribute('src')).includes('/api/stream?path='));
 await page.locator('#nvidiaFrameGeneration').click(); assert.equal(await page.locator('#targetFps').inputValue(),'0');
 await page.locator('#rtxHDR').click();
 assert.equal(await page.locator('#hdrControls').isVisible(),true);assert.equal(await page.locator('#precisionNote').isVisible(),true);
 await page.locator('#outputContainer').selectOption('webm');
 assert.deepEqual(await page.locator('#videoEncoderPreset option').evaluateAll(o=>o.map(x=>x.value)),['auto','av1_nvenc']);
 await page.locator('#deblur').click();
 // Production DOM plan: restore first, spatial then temporal, HDR last, explicit export precision.
 await page.locator('#aiUpscale').click();await page.locator('#rifeInterpolation').click();
 assert.equal(await page.locator('#nvidiaUpscale').getAttribute('aria-pressed'),'false');
 assert.equal(await page.locator('#nvidiaFrameGeneration').getAttribute('aria-pressed'),'false');
 assert.equal(await page.locator('#upscaleModel').inputValue(),'mixed-2x');assert.equal(await page.locator('#rifeModel').inputValue(),'rife-general');
 assert.equal(await page.locator('#contentType').isVisible(),true);
 const plan=await page.locator('#pipelinePlan').innerText();
 assert.match(plan,/NVIDIA Denoise → NVIDIA Deblur → AI Upscale \(2x, Mixed 2x\) → RIFE \(60fps, RIFE general\) → SDR→HDR → Auto codec\/WEBM · 10-bit PQ/);
 await page.locator('#aiUpscale').click();assert.equal(await page.locator('#scale').inputValue(),'1');
 await page.locator('#scale').selectOption('2');assert.equal(await page.locator('#aiUpscale').getAttribute('aria-pressed'),'true');
 await page.locator('#rifeInterpolation').click();assert.equal(await page.locator('#targetFps').inputValue(),'0');
 await page.locator('#targetFps').fill('60');assert.equal(await page.locator('#rifeInterpolation').getAttribute('aria-pressed'),'true');
 await page.locator('#nvidiaUpscale').click();await page.locator('#nvidiaFrameGeneration').click();
 if(process.env.TVE_BROWSER_MEDIA){await page.waitForFunction(()=>document.querySelector('#videoPreview').readyState>=2);await page.locator('#videoPreview').evaluate(video=>video.currentTime=0.5);await page.waitForFunction(()=>!document.querySelector('#videoPreview').seeking && document.querySelector('#videoPreview').readyState>=2);}
 await page.locator('#rtxUpscaleIntermediate').selectOption('4');
 await page.locator('#rtxDownsampleFilter').selectOption('lanczos');
 await page.locator('#scale').selectOption('1');
 assert.equal(await page.locator('#nvidiaUpscale').getAttribute('aria-pressed'),'true');
 assert.match(await page.locator('#pipelinePlan').innerText(),/4x source, internal.*Downsample.*lanczos.*final 1x/);
 await page.locator('#scale').selectOption('4');
 assert.equal(await page.locator('#runJob').isDisabled(),true);
 await page.locator('#scale').selectOption('2');
 const output=process.env.TVE_SCREENSHOT_DIR||process.env.TMPDIR;
 assert.ok(output,'Set TVE_SCREENSHOT_DIR or TMPDIR for screenshot artifacts');fs.mkdirSync(output,{recursive:true});
 for(const [width,height] of [[1440,900],[1920,1080],[390,844]]){
  await page.setViewportSize({width,height});
  const dims=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,width:innerWidth,height:document.documentElement.scrollHeight}));
  assert.ok(dims.scroll<=width,`Horizontal overflow at ${width}: ${dims.scroll}`);
  if(width>=1440) assert.ok(dims.height<=height,`Desktop viewport overflow at ${width}: ${dims.height}`);
  await page.screenshot({path:path.join(output,`tve-studio-${width}.png`),fullPage:true});
  console.log(`Screenshot ${width}×${height}: document height ${dims.height}`);
 }
 await page.setViewportSize({width:1440,height:900});
 await page.locator('#rtxHDR').click(); await page.locator('#targetFps').fill('60');
 await page.locator('#preview').uncheck(); await page.locator('#runJob').click();
 await page.waitForFunction(()=>document.querySelectorAll('.queue-ui-job').length>=2);
 assert.equal(requests[1].body.rtx_upscale_intermediate,4);assert.equal(requests[1].body.rtx_downsample_filter,'lanczos');
 assert.equal(requests[1].body.preview,false);assert.equal(requests[1].body.start_time,undefined);assert.equal(requests[1].body.end_time,undefined);
 await page.locator('#preview').check(); await page.locator('#openBatchDialog').click(); await page.locator('#batchFolder').fill('/video'); await page.locator('#batchFolder').press('Tab');
 await page.waitForFunction(()=>document.querySelectorAll('#batchFileList input').length===4);await page.locator('#batchEnqueue').click();
 await page.waitForFunction(()=>!document.querySelector('#batchDialog').open);
 const batch=requests.filter(r=>r.url==='/api/jobs/batch'); assert.equal(batch.length,4);
 for(const r of batch){assert.equal(r.body.rtx_upscale_intermediate,4);assert.equal(r.body.rtx_downsample_filter,'lanczos');}
 assert.equal(batch[0].body.rtx_frame_generation,true);assert.equal(batch[1].body.rtx_frame_generation,false);
 assert.equal(batch[0].body.preview,true);assert.equal(batch[1].body.preview,true);assert.equal(batch[0].body.request,undefined);
 assert.equal(batch[2].body.output_dir,'/video/enhanced/sub/a.mp4');assert.equal(batch[3].body.output_dir,'/video/enhanced/sub/a.mkv');
 await page.reload();await page.waitForFunction(()=>document.querySelector('#runtimeSummary').textContent.includes('Python'));
 assert.equal(await page.locator('#upscaleModel').inputValue(),'mixed-2x');assert.equal(await page.locator('#rifeModel').inputValue(),'rife-general');
 assert.equal(await page.locator('#rtxUpscaleIntermediate').inputValue(),'4');assert.equal(await page.locator('#rtxDownsampleFilter').inputValue(),'lanczos');
 assert.equal(await page.locator('#rtxDenoiseStrength').inputValue(),'0');assert.equal(await page.locator('#rtxAutoShotDetection').isChecked(),false);
 failCapabilities=true;await page.locator('#retryCapabilities').click();await page.waitForFunction(()=>document.querySelector('#requestStatus').textContent.includes('SDK discovery offline'));
 assert.equal(await page.locator('#runJob').isDisabled(),true);
 failCapabilities=false;legacyCapabilities=true;
 await page.locator('#retryCapabilities').click();await page.waitForFunction(()=>document.querySelector('#requestStatus').textContent.includes('does not support custom'));
 assert.equal(await page.locator('#runJob').isDisabled(),true);
 legacyCapabilities=false;
 await page.evaluate(()=>localStorage.setItem('tve_studio_settings_v2',JSON.stringify({pipeline:'restore+upscale',scale:2,targetFps:60,denoise:true,upscaleModel:'mixed-2x',rifeModel:'rife-general',rtxDenoiseStrength:0,rtxAutoShotDetection:false})));
 await page.reload();await page.waitForFunction(()=>document.querySelector('#upscaleModel').value==='mixed-2x');
 assert.equal(await page.locator('#aiUpscale').getAttribute('aria-pressed'),'true');assert.equal(await page.locator('#rifeInterpolation').getAttribute('aria-pressed'),'true');
 assert.equal(await page.locator('#denoise').getAttribute('aria-pressed'),'true');assert.equal(await page.locator('#rtxDenoiseStrength').inputValue(),'0');assert.equal(await page.locator('#rtxAutoShotDetection').isChecked(),false);
 assert.deepEqual(errors,[],'Browser runtime errors');
 console.log('Browser smoke passed: real DOM interactions, snapshots, single/full/batch requests, source-dependent FG, playback selection, persistence, capability failure and responsive screenshots.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(()=>server.close());