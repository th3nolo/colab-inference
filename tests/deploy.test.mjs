import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { MARKER, parseArgs, selectTarget, buildCode, updateCell, evaluateResult, waitForExecution, connect } from '../deploy.mjs';

const key = 'https://colab.research.google.com/drive/test-notebook';
const tab = { id: 'ABC', type: 'page', url: key, webSocketDebuggerUrl: 'ws://127.0.0.1:9222/devtools/page/ABC' };
const revision = 'a'.repeat(40);
const args = ['--notebook-url', key];
function cell(text) {
  return { text, executions: 0, getText() { return this.text; }, setText(t) { this.text = t; }, executeCellCommand() { this.executions++; } };
}
let statusListener;
function fixture(cells) {
  globalThis.addEventListener = (_event, listener) => { statusListener = listener; };
  globalThis.removeEventListener = () => {};
  globalThis.location = { origin: 'https://colab.research.google.com', pathname: '/drive/test-notebook' };
  globalThis.colab = { global: { notebook: { cells } } };
  return { key, marker: MARKER, code: MARKER + '\nprint("new")', token: 'test-token' };
}
afterEach(() => { delete globalThis.location; delete globalThis.colab; delete globalThis.__colabInferenceDeploy; delete globalThis.addEventListener; delete globalThis.removeEventListener; });

test('explicit selector required, ambiguous and missing arguments rejected', () => {
  for (const a of [[], ['--port'], ['--bad','x'], [...args,'--tab-id','ABC'], [...args,'--notebook-url',key]]) assert.throws(() => parseArgs(a));
  assert.equal(parseArgs(args).port, 3000);
  assert.equal(parseArgs(args).model, undefined);
});
test('validated model/revision pair and numeric limits', () => {
  assert.equal(parseArgs([...args,'--model','Qwen/Qwen3-8B','--revision',revision]).revision, revision);
  for (const a of [['--model','Qwen/Qwen3-8B'], ['--revision',revision], ['--model',"x/y');evil()",'--revision',revision], ['--model','x/y','--revision','main'], ['--port','0'], ['--port','65536'], ['--timeout','NaN']]) assert.throws(() => parseArgs([...args,...a]));
});
test('exact origin/path, unique target, loopback CDP endpoint', () => {
  assert.equal(selectTarget([tab],parseArgs(args)).id,'ABC');
  assert.equal(selectTarget([tab],{tabId:'ABC'}).key,key);
  for (const tabs of [[], [tab,tab], [{...tab,url:'https://evil.test/?'+key}], [{...tab,url:'https://colab.research.google.com.evil.test/drive/test-notebook'}], [{...tab,webSocketDebuggerUrl:'ws://evil.test:9222/devtools/page/ABC'}]]) assert.throws(() => selectTarget(tabs,parseArgs(args)));
});
test('updates marked cell, preserves first cell and all unrelated cells', async () => {
  const first = cell('personal code'), managed = cell(MARKER), last = cell('notes');
  await updateCell(fixture([first,managed,last]));
  assert.equal(first.text,'personal code'); assert.equal(last.text,'notes');
  assert.equal(managed.executions,1); assert.equal(first.executions,0);
  assert.equal(colab.global.notebook.cells.length,3);
  globalThis.__colabInferenceDeploy.status = 'complete';
  await updateCell(fixture([first,managed,last]));
  assert.equal(colab.global.notebook.cells.length,3);
});
test('missing, duplicated, misplaced markers fail without changing cells', async () => {
  for (const cells of [[cell('user')], [cell(MARKER),cell(MARKER)], [cell('print(1)\n'+MARKER)]]) {
    const before = cells.map(c=>c.text);
    await assert.rejects(updateCell(fixture(cells)),/Expected exactly one/);
    assert.deepEqual(cells.map(c=>c.text),before);
    assert.ok(cells.every(c=>c.executions===0));
  }
});
test('navigation, ownership race, unsupported API and pending execution fail closed', async () => {
  let c = cell(MARKER), p = fixture([c]);
  location.pathname = '/drive/other';
  await assert.rejects(updateCell(p),/navigated/);
  c = cell(MARKER); p = fixture([c]); c.setText = function(t) { this.text=t; colab.global.notebook.cells=[]; };
  await assert.rejects(updateCell(p),/ownership/); assert.equal(c.executions,0);
  c = cell(MARKER); p = fixture([c]); delete c.executeCellCommand;
  await assert.rejects(updateCell(p),/Unsupported/);
  c = cell(MARKER); p = fixture([c]); globalThis.__colabInferenceDeploy={status:'pending'};
  await assert.rejects(updateCell(p),/already/); assert.equal(c.text,MARKER);
});
test('sync and async execution failures become explicit errors', async () => {
  const c=cell(MARKER), p=fixture([c]); c.executeCellCommand=()=>{throw new Error('sync error');};
  await assert.rejects(updateCell(p),/sync error/); assert.equal(globalThis.__colabInferenceDeploy.status,'error');
  c.executeCellCommand=()=>Promise.reject(new Error('async error'));
  await updateCell(p); await Promise.resolve();
  assert.match(globalThis.__colabInferenceDeploy.detail,/async error/);
});
test('CDP protocol and page errors propagate', () => {
  for (const result of [{error:{message:'protocol'}},{result:{exceptionDetails:{text:'exception'}}},{result:{result:{subtype:'error',description:'page error'}}}]) assert.throws(()=>evaluateResult(result));
  assert.deepEqual(evaluateResult({result:{result:{value:{submitted:true}}}}),{submitted:true});
});
test('JSON round-trip protects nested Python/JS input from code injection', async () => {
  const source = 'CONFIG = {"model_id": "default", "model_revision": "default"}\n# quotes \' " ` ${danger} \\ unicode 雪\n';
  const code=buildCode(source,'org/model','token',revision);
  const encoded=code.match(/b64decode\('([^']+)'\)/)[1];
  const data=JSON.parse(Buffer.from(encoded,'base64').toString());
  assert.equal(data.source,source); assert.equal(data.model,'org/model'); assert.equal(data.revision,revision);
  const c=cell(MARKER), payload=fixture([c]); payload.code=code;
  await vm.runInThisContext(`(${updateCell.toString()})(${JSON.stringify(payload)})`);
  assert.equal(c.text,code); assert.equal(c.executions,1);
});

test('output-frame status requires exact channel/token and trusted origin', async () => {
  await updateCell(fixture([cell(MARKER)]));
  const data={channel:'colab-inference:deploy-status:v1',token:'test-token',status:'complete',detail:'done'};
  statusListener({origin:'https://evil.test',data});
  assert.equal(globalThis.__colabInferenceDeploy.status,'pending');
  statusListener({origin:'https://output-colab.googleusercontent.com',data:{...data,token:'other'}});
  assert.equal(globalThis.__colabInferenceDeploy.status,'pending');
  statusListener({origin:'https://output-colab.googleusercontent.com',data});
  assert.equal(globalThis.__colabInferenceDeploy.status,'complete');
});

test('completion polling returns only matching-token completion and rejects unknown outcomes', async () => {
  let tick=0;
  const timing={now:()=>tick,sleep:async ms=>{tick+=ms;}};
  const states=[{token:'t',status:'running'},{token:'t',status:'complete',detail:'done'}];
  assert.equal(await waitForExecution({evaluate:async()=>states.shift()},key,'t',5,timing),'done');
  for (const state of [undefined,{token:'other',status:'complete'},{token:'t',status:'error',detail:'Python traceback'}]) {
    await assert.rejects(waitForExecution({evaluate:async()=>state},key,'t',5,timing));
  }
  tick=0;
  await assert.rejects(waitForExecution({evaluate:async()=>({token:'t',status:'pending'})},key,'t',2,timing),/timed out/);
});
test('CDP transport correlates replies, ignores events and propagates disconnects', async () => {
  const original=globalThis.WebSocket;
  let socket;
  class MockSocket extends EventTarget {
    constructor() { super(); socket=this; queueMicrotask(()=>this.dispatchEvent(new Event('open'))); }
    send(raw) {
      const request=JSON.parse(raw);
      this.dispatchEvent(new MessageEvent('message',{data:JSON.stringify({method:'Runtime.consoleAPICalled'})}));
      this.dispatchEvent(new MessageEvent('message',{data:JSON.stringify({id:request.id+20,result:{result:{value:'wrong'}}})}));
      queueMicrotask(()=>this.dispatchEvent(new MessageEvent('message',{data:JSON.stringify({id:request.id,result:{result:{value:'matched'}}})})));
    }
    close() { this.dispatchEvent(new Event('close')); }
  }
  globalThis.WebSocket=MockSocket;
  try {
    const client=await connect('ws://fixture.invalid');
    assert.equal(await client.evaluate('1'),'matched');
    socket.send=()=>{};
    const pending=client.evaluate('2'); socket.close();
    await assert.rejects(pending,/closed/);
  } finally { globalThis.WebSocket=original; }
});
test('concurrent same-cell submissions execute at most once', async () => {
  const c=cell(MARKER), p=fixture([c]);
  const outcomes=await Promise.allSettled([updateCell(p),updateCell(p)]);
  assert.equal(outcomes.filter(r=>r.status==='fulfilled').length,1);
  assert.equal(c.executions,1);
});
