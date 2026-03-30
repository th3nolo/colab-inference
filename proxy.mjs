#!/usr/bin/env node
/**
 * Local proxy server for Colab inference.
 * Forwards requests from localhost to the cloudflared tunnel.
 *
 * Usage: node proxy.mjs <tunnel-url> [port]
 * Example: node proxy.mjs https://abc-xyz.trycloudflare.com 3000
 */

import http from 'http';

const TUNNEL = process.argv[2];
const PORT = parseInt(process.argv[3] || '3000', 10);

if (!TUNNEL) {
  console.error('Usage: node proxy.mjs <tunnel-url> [port]');
  console.error('Example: node proxy.mjs https://abc-xyz.trycloudflare.com 3000');
  process.exit(1);
}

const server = http.createServer((req, res) => {
  // CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    });
    return res.end();
  }

  let body = '';
  req.on('data', c => (body += c));
  req.on('end', () => {
    const url = `${TUNNEL}${req.url}`;
    const t0 = Date.now();

    fetch(url, {
      method: req.method,
      headers: { 'Content-Type': 'application/json' },
      ...(body ? { body } : {}),
    })
      .then(async r => {
        const text = await r.text();
        const ms = Date.now() - t0;
        console.log(`${req.method} ${req.url} -> ${r.status} (${ms}ms)`);
        res.writeHead(r.status, {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        });
        res.end(text);
      })
      .catch(e => {
        console.error(`${req.method} ${req.url} -> ERROR: ${e.message}`);
        res.writeHead(502);
        res.end(JSON.stringify({ error: e.message }));
      });
  });
});

server.listen(PORT, () => {
  console.log(`Colab inference proxy`);
  console.log(`  Local:  http://localhost:${PORT}`);
  console.log(`  Remote: ${TUNNEL}`);
  console.log(`\nEndpoints:`);
  console.log(`  GET  /v1/models`);
  console.log(`  POST /v1/chat/completions`);
  console.log(`  GET  /health`);
});
