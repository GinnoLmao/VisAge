// Isolated headless Chrome rendering for the local app; no personal profile.
import {spawn} from 'node:child_process';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {join, resolve} from 'node:path';
import {randomUUID} from 'node:crypto';

const root = process.cwd(), base = process.argv[2] || 'http://127.0.0.1:8041';
if (!/^http:\/\/127\.0\.0\.1:\d+$/.test(base)) throw Error('Local QA origin required');
const folder = resolve(root, process.argv[3] || 'reports/pretrained/web_app/screenshots');
const reportMode = process.argv.includes('--report');
await mkdir(folder, {recursive:true});
const profile = join(root, '.cache', `visual_chrome_${randomUUID()}`);
const chrome = spawn('C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', [
  '--headless', '--no-first-run', '--disable-background-networking', '--remote-debugging-port=0',
  '--remote-debugging-address=127.0.0.1', `--user-data-dir=${profile}`, 'about:blank'
], {windowsHide:true, stdio:'ignore'});
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
let socket, serial = 0;
const pending = new Map(), errors = [], captures = [];
function call(method, params = {}) {
  const id = ++serial;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {pending.delete(id); reject(Error(`Timed out: ${method}`));}, 30000);
    pending.set(id, {resolve, reject, timer});
    socket.send(JSON.stringify({id, method, params}));
  });
}
async function evaluate(expression) {
  const r = await call('Runtime.evaluate', {expression, returnByValue:true, awaitPromise:true});
  if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails));
  return r.result.value;
}
async function until(expression) {
  for (let i = 0; i < 150; i++) {
    if (await evaluate(expression)) return;
    await pause(100);
  }
  throw Error(`UI condition did not complete: ${expression}`);
}
async function capture(name, width, height) {
  await evaluate('document.activeElement?.blur();window.scrollTo(0,0);true');
  await pause(150); // Let chart/overlay ResizeObservers settle before rendering.
  const size = await evaluate('({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight})');
  if (size.width !== width || size.scrollWidth > width + 1) throw Error(`Viewport overflow: ${JSON.stringify(size)}`);
  const result = await call('Page.captureScreenshot', {format:'png', captureBeyondViewport:true,
    clip:{x:0,y:0,width,height:Math.max(height,size.height),scale:1}});
  await writeFile(join(folder, `${name}.png`), Buffer.from(result.data, 'base64'));
  captures.push({name, ...size, file:join(folder, `${name}.png`)});
}
try {
  let address;
  for (let i = 0; i < 150; i++) {
    try {
      const [port] = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).trim().split('\n');
      const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      address = pages.find(p => p.type === 'page')?.webSocketDebuggerUrl;
      if (address) break;
    } catch {}
    await pause(100);
  }
  if (!address) throw Error('Chrome did not expose its isolated test page');
  socket = new WebSocket(address);
  await new Promise((resolve,reject) => {socket.onopen=resolve; socket.onerror=reject;});
  socket.onmessage = event => {
    const r = JSON.parse(event.data);
    if (r.method === 'Runtime.exceptionThrown') errors.push(r.params.exceptionDetails);
    if (!r.id || !pending.has(r.id)) return;
    const p = pending.get(r.id); pending.delete(r.id); clearTimeout(p.timer);
    if (r.error) p.reject(Error(JSON.stringify(r.error))); else p.resolve(r.result);
  };
  await call('Page.enable'); await call('Runtime.enable');
  for (const [prefix, width, height] of (reportMode ? [['desktop',1440,1080]] : [['desktop',1440,1080],['mobile',390,844]])) {
    await call('Emulation.setDeviceMetricsOverride', {width,height,deviceScaleFactor:1,mobile:prefix==='mobile'});
    await call('Page.navigate', {url:base});
    await until('document.getElementById("service-status")?.dataset.state === "ready"');
    await capture(`${prefix}_analysis`,width,height);
    const doc = await call('DOM.getDocument');
    const input = await call('DOM.querySelector',{nodeId:doc.root.nodeId,selector:'#photo-input'});
    await call('DOM.setFileInputFiles',{nodeId:input.nodeId,files:[resolve(root,'.cache/web_qa/single.jpg')]});
    await until('document.getElementById("analyze-photo")?.disabled === false');
    await evaluate('document.getElementById("method-select").value="all";document.getElementById("method-select").dispatchEvent(new Event("change"));document.getElementById("analyze-photo").click();true');
    await until('document.getElementById("model-results")?.children.length === 3 && !document.getElementById("prediction-results").hidden');
    await until('document.querySelectorAll("#model-results .score-row").length === 9');
    if (reportMode) {
      await evaluate('document.getElementById("photo-preview").style.visibility="hidden";const n=document.createElement("span");n.textContent="Validation photo omitted from public report";n.style.cssText="position:absolute;inset:42% 12%;text-align:center;font:14px system-ui;color:#596b84";document.getElementById("photo-stage").append(n);true');
      const rect = await evaluate('(()=>{const r=document.querySelector(".model-result").getBoundingClientRect();return {x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height};})()');
      const screenshot = await call('Page.captureScreenshot', {format:'png',captureBeyondViewport:true,clip:{...rect,scale:1}});
      await writeFile(join(folder,'model_scores.png'),Buffer.from(screenshot.data,'base64'));
    }
    await capture(`${prefix}_comparison`,width,height);
    await evaluate('location.hash="evaluation";true');
    await until('document.querySelectorAll("#eval-classification .chart-select button").length === 3');
    await capture(`${prefix}_evaluation`,width,height);
    await evaluate('document.getElementById("eval-detection-tab").click();true');
    await capture(`${prefix}_detection`,width,height);
    await evaluate('document.getElementById("eval-pipeline-tab").click();true');
    await capture(`${prefix}_pipeline`,width,height);
  }
  if (errors.length) throw Error(JSON.stringify(errors));
  await writeFile(join(folder,'render_manifest.json'),JSON.stringify({ok:true,captures,uncaught_errors:errors,scope:reportMode ? 'Isolated headless Chrome, 1440px desktop; face photo hidden for public reporting. Validation fixture only.' : 'Isolated headless Chrome, true 390px mobile viewport and 1440px desktop. Validation fixture only.'},null,2));
  console.log(JSON.stringify({ok:true,captures:captures.length}));
} finally {
  if (socket?.readyState === WebSocket.OPEN) {
    await call('Browser.close').catch(()=>{}); socket.close();
  }
  if (chrome.exitCode === null) await new Promise(resolve => {chrome.once('exit',resolve);setTimeout(()=>{chrome.kill();resolve();},5000).unref();});
}
