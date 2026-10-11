'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {caps,models}=require('./vfx-ui.test.cjs');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
let preview=null;
const server=http.createServer((req,res)=>{
 const url=new URL(req.url,'http://localhost');
 const json=data=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data));};
 if(url.pathname==='/api/capabilities')return json(caps);
 if(url.pathname==='/api/runtime/models')return json({models});
 if(url.pathname==='/api/runtime/status')return json({python:{available:true},nvidia_smi:{available:true}});
 if(url.pathname==='/api/jobs')return json({jobs:[]});
 if(url.pathname.endsWith('/live-preview')){
  if(preview){res.setHeader('Content-Type','image/png');return res.end(preview);}
  res.statusCode=404;return json({error:'No processed frame yet'});
 }
 if(url.pathname.startsWith('/api/')){res.statusCode=404;return json({error:'Unimplemented test fixture'});}
 const file=path.join(__dirname,url.pathname==='/'?'index.html':path.basename(url.pathname));
 if(!fs.existsSync(file)){res.statusCode=404;return res.end();}
 res.setHeader('Content-Type',file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':'text/html');res.end(fs.readFileSync(file));
});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  assert.equal(await page.locator('#placeholderFrame').isVisible(),true);
  assert.equal(await page.locator('#encodePreview').isVisible(),false,'No broken image on empty startup');
  assert.equal(await page.locator('#videoPreview').isVisible(),false);
  assert.equal(await page.locator('.preview-panel #outputSummary').count(),0,'Planned output only belongs to Source');
  await page.evaluate(()=>{renderJobEvent({job:{id:'test-job',status:'running',logs:[]}});});
  await page.locator('#processedPlayback').click();
  await page.evaluate(()=>startEncodePreview('test-job'));
  await page.waitForFunction(()=>document.getElementById('placeholderFrame').textContent==='Waiting for processed frames…');
  assert.equal(await page.locator('#encodePreview').isVisible(),false,'404 live frame cannot display a broken image');
  assert.equal(await page.locator('#placeholderFrame').isVisible(),true);
  const image=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=c.height=2;c.getContext('2d').fillRect(0,0,2,2);return c.toDataURL('image/png');});
  preview=Buffer.from(image.split(',')[1],'base64'); // Explicit generated fixture, not inference.
  await page.evaluate(()=>startEncodePreview('test-job'));
  await page.waitForFunction(()=>!document.getElementById('encodePreview').hidden&&document.getElementById('encodePreview').naturalWidth===2);
  assert.equal(await page.locator('#placeholderFrame').isVisible(),false);
  await page.evaluate(()=>{stopEncodePreview();renderJobEvent({job:{id:'new-job',status:'running',logs:[]}});});
  assert.equal(await page.locator('#encodePreview').isVisible(),false,'Do not display previous job frames while new job prepares');
  assert.equal(await page.locator('#placeholderFrame').isVisible(),true);
  await page.locator('#sourcePlayback').click();
  assert.equal(await page.locator('#placeholderFrame').textContent(),'Select a video to preview.');
  assert.equal(await page.locator('#videoPreview').isVisible(),false);
  assert.deepEqual(errors,[]);
  page.once('dialog',d=>d.accept());
  await page.locator('#quitApp').click();
  await page.waitForFunction(()=>document.getElementById('log').textContent.includes('Shutdown failed:'));
  await page.waitForTimeout(250);
  assert.equal(await page.locator('.preview-panel').isVisible(),true,'Failed Quit must not claim the app has stopped');
  console.log('Preview regressions passed: empty startup, single Source plan, failed live frame, valid live frame, job switch, return to empty Source and failed Quit acknowledgement.');
 }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});
