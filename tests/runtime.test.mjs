import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createInterface } from 'node:readline';
import { setTimeout as sleep } from 'node:timers/promises';
import { startProxy } from '../proxy.mjs';

const token = 'test-only-token-' + 'x'.repeat(32);
const payload = { messages: [{ role: 'user', content: 'hello' }] };
const request = (url, options = {}) => fetch(url, { ...options, signal: AbortSignal.timeout(3000) });

async function fixture(t, entrypoint) {
  assert.ok(process.env.TEST_PYTHON, 'Set TEST_PYTHON to the isolated locked Python interpreter');
  const child = spawn(process.env.TEST_PYTHON, ['-u', 'tests/runtime_fixture.py', entrypoint], {
    stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true,
  });
  const exited = once(child, 'exit');
  let errors = '';
  child.stderr.setEncoding('utf8').on('data', data => { errors += data; });
  t.after(async () => {
    child.stdin.end();
    const timer = setTimeout(() => child.kill(), 4000);
    try {
      const [code] = await exited;
      assert.equal(code, 0, errors);
    } finally { clearTimeout(timer); }
  });
  const lines = createInterface({ input: child.stdout });
  const ready = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Uvicorn startup timed out: ${errors}`)), 8000);
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.once('exit', code => { clearTimeout(timer); reject(new Error(`Fixture exited ${code}: ${errors}`)); });
    lines.on('line', line => {
      if (line.startsWith('FIXTURE_READY ')) {
        clearTimeout(timer);
        resolve(JSON.parse(line.slice('FIXTURE_READY '.length)));
      }
    });
  });
  const upstream = `http://127.0.0.1:${ready.port}`;
  const proxy = startProxy({ tunnel: upstream, apiToken: token, upstreamTimeoutMs: 2000 }, 0);
  await once(proxy, 'listening');
  t.after(() => { proxy.closeAllConnections(); proxy.close(); lines.close(); });
  assert.equal(proxy.address().address, '127.0.0.1');
  return { ...ready, upstream, url: `http://127.0.0.1:${proxy.address().port}` };
}

for (const entrypoint of ['server', 'notebook']) {
  test(`real ${entrypoint} startup, HTTP routing and loopback proxy with a model fixture`, { timeout: 20000 }, async t => {
    const f = await fixture(t, entrypoint);
    const post = (changes = {}, url = f.url) => request(url + '/v1/chat/completions', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: 'Bearer caller-token' },
      body: JSON.stringify({ ...payload, ...changes }),
    });
    for (const path of ['/health', '/v1/models', '/v1/chat/completions']) {
      const unauthorized = await request(f.upstream + path);
      assert.equal(unauthorized.status, 401);
      assert.equal(unauthorized.headers.get('www-authenticate'), 'Bearer');
      await unauthorized.text();
    }
    // A malformed unauthenticated body still takes the authentication path.
    assert.equal((await request(f.upstream + '/v1/chat/completions', { method: 'POST', body: '{bad' })).status, 401);
    const health = await request(f.url + '/health');
    assert.equal(health.status, 200);
    assert.equal((await health.json()).model, f.model);
    assert.equal((await (await request(f.url + '/v1/models')).json()).data[0].id, f.model);
    assert.equal(f.loads.length, 2);
    for (const [, model, options] of f.loads) {
      assert.equal(model, f.model);
      assert.match(options.revision, /^[0-9a-f]{40}$/);
      assert.equal(options.trust_remote_code, false);
    }
    assert.equal(f.loads[0][2].revision, f.loads[1][2].revision);
    assert.equal(f.loads[0][2].use_safetensors, true);
    const response = await post({ max_tokens: 2 });
    assert.equal(response.status, 200);
    const result = await response.json();
    assert.equal(result.model, f.model);
    assert.equal(result.object, 'chat.completion');
    assert.deepEqual(result.choices[0].message, { role: 'assistant', content: 'fixture answer' });
    assert.equal(result.choices[0].finish_reason, 'stop');
    assert.equal(result.usage.prompt_tokens, 2);
    assert.equal(result.usage.completion_tokens, 2);
    assert.equal(result.usage.total_tokens, 4);
    for (const [changes, param] of [[{ model: 'wrong/model' }, 'model'], [{ stream: true }, 'stream']]) {
      const rejected = await post(changes);
      assert.equal(rejected.status, 400);
      assert.equal((await rejected.json()).detail.param, param);
    }
    const invalid = await post({ max_tokens: true });
    assert.equal(invalid.status, 422);
    assert.deepEqual(await invalid.json(), { detail: 'Invalid request parameters' });
    const failed = await post({ messages: [{ role: 'user', content: 'fail' }] });
    assert.equal(failed.status, 500);
    await failed.text();
    assert.equal((await post()).status, 200, 'model errors must release the inference slot');

    const deadlineProxy = startProxy({ tunnel: f.upstream, apiToken: token, upstreamTimeoutMs: 100 }, 0);
    await once(deadlineProxy, 'listening');
    t.after(() => { deadlineProxy.closeAllConnections(); deadlineProxy.close(); });
    const timedOut = await post({ messages: [{ role: 'user', content: 'slow' }] }, `http://127.0.0.1:${deadlineProxy.address().port}`);
    assert.equal(timedOut.status, 504);
    await timedOut.text();
    // A proxy disconnect does not cancel blocking model.generate. Assert its
    // real lifecycle: occupied until the fixture returns, then reusable.
    const busy = await post();
    assert.equal(busy.status, 429);
    assert.equal(busy.headers.get('retry-after'), '1');
    await busy.text();
    const deadline = Date.now() + 2500;
    let recovered;
    do {
      await sleep(50);
      recovered = await post();
      await recovered.text();
    } while (recovered.status === 429 && Date.now() < deadline);
    assert.equal(recovered.status, 200);
  });
}
