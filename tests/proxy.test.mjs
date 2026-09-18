import { test } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import { startProxy, createProxy } from '../proxy.mjs';

const apiToken = 'test-only-' + 'x'.repeat(32);

async function fixture(t, handler, options = {}) {
  const upstream = http.createServer(handler);
  upstream.listen(0, '127.0.0.1');
  await once(upstream, 'listening');
  const proxy = startProxy({ apiToken, tunnel: `http://127.0.0.1:${upstream.address().port}`, ...options }, 0);
  await once(proxy, 'listening');
  t.after(() => { proxy.closeAllConnections(); proxy.close(); upstream.closeAllConnections(); upstream.close(); });
  return { proxy, port: proxy.address().port, url: `http://127.0.0.1:${proxy.address().port}` };
}
function request(port, { method = 'POST', headers = {}, chunks = [] } = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port, path: '/v1/chat/completions', method, headers }, res => {
      let body = '';
      res.on('data', chunk => { body += chunk; });
      res.on('end', () => resolve({ status: res.statusCode, body }));
    });
    req.on('error', reject);
    for (const chunk of chunks) req.write(chunk);
    req.end();
  });
}

test('loopback bind and CLI-style requests preserve method, path, bytes and upstream status', async t => {
  const f = await fixture(t, async (req, res) => {
    assert.equal(req.method, 'POST');
    assert.equal(req.url, '/v1/chat/completions');
    let body = '';
    for await (const chunk of req) body += chunk;
    assert.equal(body, '{"text":"café"}');
    res.writeHead(201, { 'Content-Type': 'application/json' });
    res.end('{"ok":true}');
  });
  assert.equal(f.proxy.address().address, '127.0.0.1');
  assert.deepEqual(await request(f.port, { chunks: ['{"text":"café"}'] }), { status: 201, body: '{"ok":true}' });
});

test('origin and Host checks reject before any upstream request, including simple POST', async t => {
  let calls = 0;
  const f = await fixture(t, (req, res) => { calls++; res.end('{}'); }, { allowedOrigins: ['http://localhost:5173'] });
  for (const origin of ['https://evil.example', 'null', 'http://localhost:5173.evil.example']) {
    assert.equal((await request(f.port, { headers: { Origin: origin }, chunks: ['hello'] })).status, 403);
  }
  assert.equal((await request(f.port, { headers: { Host: `evil.example:${f.port}` } })).status, 403);
  assert.equal(calls, 0);
  const allowed = await fetch(f.url, { headers: { Origin: 'http://localhost:5173' } });
  assert.equal(allowed.headers.get('access-control-allow-origin'), 'http://localhost:5173');
  assert.equal(await allowed.text(), '{}');
  const same = await fetch(f.url, { headers: { Origin: f.url } });
  assert.equal(same.status, 200);
  await same.text();
});

test('preflight only permits supported methods and headers', async t => {
  const f = await fixture(t, () => assert.fail('Preflight reached upstream'));
  for (const [method, header, expected] of [['POST', 'content-type, authorization', 204], ['DELETE', 'content-type', 403], ['POST', 'x-unsafe', 403]]) {
    const response = await fetch(f.url, { method: 'OPTIONS', headers: { Origin: f.url,
      'Access-Control-Request-Method': method, 'Access-Control-Request-Headers': header } });
    assert.equal(response.status, expected);
  }
  assert.equal((await fetch(f.url, { method: 'DELETE' })).status, 405);
});

test('declared and chunked bodies are capped in bytes; exact limit succeeds', async t => {
  let calls = 0;
  const f = await fixture(t, (req, res) => { calls++; res.end('{}'); }, { maxBodyBytes: 4 });
  assert.equal((await request(f.port, { headers: { 'Content-Length': '5' }, chunks: ['12345'] })).status, 413);
  assert.equal((await request(f.port, { chunks: ['123', '45'] })).status, 413);
  assert.equal((await request(f.port, { chunks: ['ééé'] })).status, 413);
  assert.equal(calls, 0);
  assert.equal((await request(f.port, { chunks: ['éé'] })).status, 200);
  assert.equal(calls, 1);
});

test('deadline aborts upstream before headers with a 504', async t => {
  let closed;
  const disconnected = new Promise(resolve => { closed = resolve; });
  const f = await fixture(t, (req, res) => { res.on('close', closed); }, { upstreamTimeoutMs: 100 });
  const response = await fetch(f.url);
  assert.equal(response.status, 504);
  await response.text();
  await disconnected;
});

test('deadline covers response streaming and closes stalled upstream', async t => {
  let closed;
  const disconnected = new Promise(resolve => { closed = resolve; });
  const f = await fixture(t, (req, res) => { res.write('partial'); res.on('close', closed); }, { upstreamTimeoutMs: 100 });
  const response = await fetch(f.url);
  await assert.rejects(response.text());
  await disconnected;
});

test('client disconnect cancels upstream while awaiting headers', async t => {
  let entered, closed;
  const started = new Promise(resolve => { entered = resolve; });
  const disconnected = new Promise(resolve => { closed = resolve; });
  const f = await fixture(t, (req, res) => { res.on('close', closed); entered(); });
  const req = http.get(f.url);
  req.on('error', () => {});
  await started;
  req.destroy();
  await disconnected;
});

test('redirects are not followed', async t => {
  const f = await fixture(t, (req, res) => { res.writeHead(302, { Location: '/other' }); res.end(); });
  assert.equal((await fetch(f.url)).status, 502);
});

test('configuration rejects wildcard origins and unbounded limits', () => {
  const tunnel = 'http://127.0.0.1:1234';
  for (const allowedOrigins of [['*'], ['null'], ['https://example.com/path']]) {
    assert.throws(() => createProxy({ apiToken, tunnel, allowedOrigins }));
  }
  assert.throws(() => createProxy({ apiToken, tunnel, upstreamTimeoutMs: 0 }));
  assert.throws(() => createProxy({ apiToken, tunnel: `${tunnel}/path` }));
});

test('client disconnect cancels an upstream response already streaming', async t => {
  let closed;
  const disconnected = new Promise(resolve => { closed = resolve; });
  const f = await fixture(t, (req, res) => { res.on('close', closed); res.write('partial'); });
  const req = http.get(f.url);
  req.on('error', () => {});
  const [res] = await once(req, 'response');
  await once(res, 'data');
  res.destroy();
  await disconnected;
});

test('aborted partial uploads never reach upstream', async t => {
  let calls = 0;
  const f = await fixture(t, (req, res) => { calls++; res.end('{}'); });
  const received = once(f.proxy, 'request');
  const req = http.request(f.url, { method: 'POST', headers: { 'Content-Length': '100' } });
  req.on('error', () => {});
  req.write('partial');
  const [incoming] = await received;
  const aborted = once(incoming, 'aborted');
  req.destroy();
  await aborted;
  assert.equal(calls, 0);
});


test('configured bearer token replaces client auth on every upstream request', async t => {
  const f = await fixture(t, (req, res) => {
    assert.equal(req.headers.authorization, `Bearer ${apiToken}`);
    res.writeHead(429, { 'Retry-After': '1', 'WWW-Authenticate': 'Bearer' });
    res.end('{}');
  });
  for (const path of ['/health', '/v1/models', '/v1/chat/completions']) {
    const response = await fetch(f.url + path, { headers: { Authorization: 'Bearer caller-value' } });
    assert.equal(response.status, 429);
    assert.equal(response.headers.get('retry-after'), '1');
    assert.equal(response.headers.get('www-authenticate'), 'Bearer');
    await response.text();
  }
});

test('missing or malformed configured token fails closed without echoing it', () => {
  for (const value of [undefined, '', 'short-secret', apiToken + '\n', 'x'.repeat(257), 'bad token' + 'x'.repeat(32)]) {
    assert.throws(() => createProxy({ tunnel: 'https://example.com', apiToken: value }), error => {
      assert.match(error.message, /COLAB_API_TOKEN/);
      if (value) assert.ok(!error.message.includes(value));
      return true;
    });
  }
});
