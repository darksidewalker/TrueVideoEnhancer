'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {caps,models}=require('./vfx-ui.test.cjs');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'/home/darksidewalker/.hermes/hermes-agent/node_modules/playwright');
let jobs=['done','error','cancelled','running','queued'].map(status=>({id:status,status,input:`/video/${status}.mp4`,output:`/video/${status}-out.mp4`,logs:[`${status} log line 1`,'line 2']}));
const server=http.createServer((req,res)=>{
 const url=new URL(req.url,'http://localhost');
 const json=data=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data));};
 if(url.pathname==='/api/capabilities')return json(caps);
 if(url.pathname==='/api/runtime/models')return json({models});
 if(url.pathname==='/api/runtime/status')return json({python:{available:true},nvidia_smi:{available:true}});
 if(url.pathname==='/api/jobs')return json(jobs);
 const job=jobs.find(j=>url.pathname===`/api/jobs/${j.id}`);if(job)return json(job);
 if(url.pathname.startsWith('/api/')){res.statusCode=404;return json({error:'Fixture has no media or events'});}
 const file=path.join(__dirname,url.pathname==='/'?'index.html':path.basename(url.pathname));
 if(!fs.existsSync(file)){res.statusCode=404;return res.end();}
 res.setHeader('Content-Type',file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':'text/html');res.end(fs.readFileSync(file));
});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.waitForFunction(()=>document.querySelectorAll('.queue-ui-job').length===5);
  await page.locator('[data-job-id="done"]').click();
  await page.waitForFunction(()=>document.getElementById('log').textContent.includes('done log line 1'));
  const text=await page.locator('#log').textContent();
  await page.evaluate(()=>{window.copiedLog=null;Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>{window.copiedLog=text;}}});});
  await page.getByRole('button',{name:'Copy log',exact:true}).click();
  assert.equal(await page.evaluate(()=>window.copiedLog),text);
  assert.equal(await page.locator('#log').textContent(),text,'Copy must preserve the log');
  assert.equal(await page.locator('#copyLog svg rect').count(),2,'Copy icon has two overlapping squares');
  await page.evaluate(()=>{navigator.clipboard.writeText=async()=>{throw new Error('Permission denied');};});
  await page.getByRole('button',{name:'Copy log',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('copyLogStatus').textContent.includes('failed'));
  assert.equal(await page.locator('#log').textContent(),text,'Clipboard failure must preserve the log');
  await page.getByRole('button',{name:'Clear finished',exact:true}).click();
  await page.waitForFunction(()=>document.querySelectorAll('.queue-ui-job').length===2);
  assert.deepEqual(await page.locator('.queue-ui-job').evaluateAll(rows=>rows.map(r=>r.dataset.jobId).sort()),['queued','running']);
  await page.waitForTimeout(1300);
  assert.equal(await page.locator('.queue-ui-job').count(),2,'Polling must not restore cleared jobs');
  await page.evaluate(()=>renderJobEvent({job:{id:'done',status:'done',logs:['late event']}}));
  assert.equal(await page.locator('.queue-ui-job').count(),2,'Late events must not restore cleared jobs');
  await page.reload();
  await page.waitForFunction(()=>document.querySelectorAll('.queue-ui-job').length===2);
  jobs=jobs.map(j=>({...j,status:'done'}));
  await page.waitForFunction(()=>[...document.querySelectorAll('.queue-ui-job')].every(r=>r.classList.contains('done')));
  await page.getByRole('button',{name:'Clear finished',exact:true}).click();
  await page.waitForFunction(()=>document.querySelectorAll('.queue-ui-job').length===0);
  assert.equal(await page.locator('#queueUiSummary').textContent(),'Queue is empty.');
  assert.equal(await page.locator('#queueUiHeaderCount').textContent(),'0 jobs');
  assert.equal(await page.locator('#log').textContent(),'No job selected.');
  await page.reload();await page.waitForTimeout(1300);
  assert.equal(await page.locator('.queue-ui-job').count(),0,'Empty queue stays empty after reload');
  assert.deepEqual(errors,[]);
  console.log('Queue actions passed: two-square copy, exact text, clipboard failure, terminal-only clear, polling/SSE/reload persistence and empty state.');
 }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});
