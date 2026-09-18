#!/usr/bin/env node
/** Local loopback proxy. Usage: node proxy.mjs <tunnel-url> [port] */
import http from 'node:http';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { pathToFileURL } from 'node:url';

export function createProxy({ tunnel, allowedOrigins = [], maxBodyBytes = 1024 * 1024,
  upstreamTimeoutMs = 300_000 } = {}) {
  const target = new URL(tunnel);
  if (!['http:', 'https:'].includes(target.protocol) || target.username || target.password ||
      target.pathname !== '/' || target.search || target.hash) {
    throw new Error('Tunnel must be an HTTP(S) origin without credentials, path, query or fragment');
  }
  for (const value of [maxBodyBytes, upstreamTimeoutMs]) {
    if (!Number.isSafeInteger(value) || value <= 0) throw new Error('Limits must be positive integers');
  }
  if (upstreamTimeoutMs > 2_147_483_647) throw new Error('Timeout exceeds Node timer range');
  const origins = new Set(allowedOrigins.map(origin => {
    const parsed = new URL(origin);
    if (!['http:', 'https:'].includes(parsed.protocol) || parsed.origin !== origin) {
      throw new Error('Allowed origins must be exact HTTP(S) origins');
    }
    return origin;
  }));
  return http.createServer(async (req, res) => {
    const reply = (status, error) => {
      if (res.destroyed || res.writableEnded) return;
      res.writeHead(status, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error }));
    };
    // Validate Host as well as Origin to reject browser DNS-rebinding requests.
    const port = req.socket.localPort;
    const hosts = [`127.0.0.1:${port}`, `localhost:${port}`];
    if (port === 80) hosts.push('127.0.0.1', 'localhost');
    if (!hosts.includes(req.headers.host)) return reply(403, 'Host not allowed');
    res.setHeader('Vary', 'Origin');
    const origin = req.headers.origin;
    const sameOrigins = [`http://127.0.0.1:${port}`, `http://localhost:${port}`];
    if (origin && !origins.has(origin) && !sameOrigins.includes(origin)) {
      return reply(403, 'Origin not allowed');
    }
    if (origin) res.setHeader('Access-Control-Allow-Origin', origin);
    if (!['GET', 'POST', 'OPTIONS'].includes(req.method)) {
      res.setHeader('Allow', 'GET, POST, OPTIONS');
      return reply(405, 'Method not allowed');
    }
    if (req.method === 'OPTIONS') {
      const requestedMethod = req.headers['access-control-request-method'];
      const headers = (req.headers['access-control-request-headers'] || '').toLowerCase()
        .split(',').map(value => value.trim()).filter(Boolean);
      if ((requestedMethod && !['GET', 'POST'].includes(requestedMethod)) ||
          headers.some(value => !['content-type', 'authorization'].includes(value))) {
        return reply(403, 'Preflight not allowed');
      }
      res.writeHead(204, {
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      });
      return res.end();
    }
    if (!req.url.startsWith('/') || req.url.startsWith('//')) return reply(400, 'Invalid path');
    if (Number(req.headers['content-length']) > maxBodyBytes) {
      res.setHeader('Connection', 'close');
      reply(413, 'Request body too large');
      req.resume();
      return;
    }
    const chunks = [];
    let bytes = 0;
    let oversized = false;
    req.on('data', chunk => {
      bytes += chunk.length;
      if (bytes > maxBodyBytes) {
        if (!oversized) {
          oversized = true;
          chunks.length = 0;
          res.setHeader('Connection', 'close');
          reply(413, 'Request body too large');
        }
      } else if (!oversized) chunks.push(chunk);
    });
    req.on('error', () => {});
    req.on('end', async () => {
      if (oversized || res.destroyed) return;
      if (req.method === 'GET' && bytes) return reply(400, 'GET requests cannot have a body');
      const controller = new AbortController();
      let timedOut = false;
      const timer = setTimeout(() => { timedOut = true; controller.abort(); }, upstreamTimeoutMs);
      const cancel = () => { if (!res.writableFinished) controller.abort(); };
      res.on('close', cancel);
      try {
        const upstream = await fetch(`${target.origin}${req.url}`, {
          method: req.method,
          headers: { 'Content-Type': 'application/json' },
          ...(bytes ? { body: Buffer.concat(chunks, bytes) } : {}),
          signal: controller.signal,
          redirect: 'error',
        });
        if (res.destroyed) return;
        res.writeHead(upstream.status, {
          'Content-Type': upstream.headers.get('content-type') || 'application/json',
        });
        if (upstream.body) await pipeline(Readable.fromWeb(upstream.body), res);
        else res.end();
      } catch {
        if (!res.headersSent) reply(timedOut ? 504 : 502,
          timedOut ? 'Upstream timed out' : 'Upstream request failed');
        else res.destroy();
      } finally {
        clearTimeout(timer);
        res.off('close', cancel);
      }
    });
  });
}

export function startProxy(options, port = 3000) {
  const server = createProxy(options);
  server.listen(port, '127.0.0.1');
  return server;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const port = Number(process.argv[3] || '3000');
    if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid port');
    const server = startProxy({ tunnel: process.argv[2],
      allowedOrigins: (process.env.PROXY_ALLOWED_ORIGINS || '').split(',').filter(Boolean),
      upstreamTimeoutMs: Number(process.env.PROXY_UPSTREAM_TIMEOUT_MS || 300_000),
    }, port);
    server.on('error', error => { console.error(error.message); process.exitCode = 1; });
    server.on('listening', () => console.log(`Colab inference proxy: http://127.0.0.1:${port}`));
  } catch (error) {
    console.error(`${error.message}\nUsage: node proxy.mjs <tunnel-url> [port]`);
    process.exitCode = 1;
  }
}
