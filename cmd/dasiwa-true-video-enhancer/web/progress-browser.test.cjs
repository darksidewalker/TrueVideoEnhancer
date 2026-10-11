'use strict';
// Explicit API fixtures exercise the shipped queue DOM and actual polling timer.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),http=require('node:http');
const {caps,models}=require('./vfx-ui.test.cjs');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'/home/darksidewalker/.hermes/hermes-agent/node_modules/playwright');
const now=Date.parse('2026-01-01T00:10:00Z');
const job={id:'eta-job',input:'/fixture/video.mp4',output:'/fixture/out.mp4',status:'running',logs:[],started_at:'2026-01-01T00:00:00Z',progress_done:10,progress_total:100,progress_throughput:2,progress_stage:'rendering',progress_updated_at:new Date(now).toISOString()};
const queued={id:'waiting-job',input:'/fixture/wait.mp4',status:'queued',started_at:'2020-01-01T00:00:00Z'};
const server=http.createServer((req,res)=>{
 const url=new URL(req.url,'http://localhost'),json=x=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(x));};
 if(url.pathname==='/api/runtime/status')return json({python:{available:true},nvidia_smi:{available:true}});
 if(url.pathname==='/api/runtime/models')return json({models});
 if(url.pathname==='/api/capabilities')return json(caps);
 if(url.pathname==='/api/jobs')return json({jobs:[job,queued]});
 if(url.pathname==='/api/jobs/eta-job')return json(job);
 if(url.pathname==='/api/jobs/waiting-job')return json(queued);
 if(url.pathname.startsWith('/api/')){res.statusCode=404;return json({error:'Fixture omits media'});}
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
  await page.addInitScript(n=>{window.fixtureNow=n;Date.now=()=>window.fixtureNow;},now);
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const row=page.locator('.queue-ui-job').filter({hasText:'video.mp4'});
  await row.getByText('ETA ≈ 45s',{exact:false}).waitFor();
  assert.equal(await page.locator('.preview-panel').getByText(/ETA|fps|frames/i).count(),0);
  assert.equal((await row.innerText()).match(/ETA ≈/g).length,1);
  assert.doesNotMatch(await page.locator('.queue-ui-job').filter({hasText:'wait.mp4'}).innerText(),/ETA/);
  await page.evaluate(()=>window.fixtureNow+=1000);
  await row.getByText('ETA ≈ 44s',{exact:false}).waitFor();
  job.progress_done=20;job.progress_throughput=4;job.progress_updated_at=new Date(now+1000).toISOString();
  await row.getByText('ETA ≈ 20s',{exact:false}).waitFor();
  await page.evaluate(()=>window.fixtureNow+=31000);
  await row.getByText('WAITING FOR PROGRESS',{exact:true}).waitFor();
  job.progress_done=100;job.progress_stage='finalizing';job.progress_updated_at=new Date(now+32000).toISOString();
  await row.getByText('FINALIZING',{exact:true}).waitFor();
  assert.doesNotMatch(await row.innerText(),/ETA ≈|100%/);
  job.status='done';await row.getByText('DONE',{exact:true}).waitFor();
  assert.doesNotMatch(await row.innerText(),/ETA/);
  for(const width of [1440,390]){
   await page.setViewportSize({width,height:900});
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'no horizontal overflow');
  }
  assert.deepEqual(errors,[]);
  console.log('ETA browser polling smoke passed: one queue ETA, no preview counters, countdown, throughput change, queued, stale, finalizing, done, compact responsive DOM.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(()=>server.close());
