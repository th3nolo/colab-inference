# Colab Inference Server

Run LLMs on Google Colab's free GPUs and serve them locally via an OpenAI-compatible API.

## Prerequisites

- **Chromium/Chrome** with remote debugging enabled
- **Node.js** (for the local proxy)
- **uv** (auto-installed by deploy script if missing)
- **Google account** logged in on Chromium (Colab requires auth)

## Quick Start

### 1. Start Chromium with debugging

```bash
chromium --remote-debugging-port=9222 --remote-allow-origins=*
```

> `--remote-allow-origins=*` is **required** — without it, CDP WebSocket connections get rejected with 403.
>
> You **must be logged into Google** in the browser. Colab uses your Google session cookies for authentication.

### 2. Open a Colab notebook and set GPU runtime

Navigate to https://colab.research.google.com/#create=true

Then: **Runtime -> Change runtime type -> T4 GPU**

### 3. Deploy the server

```bash
cd ~/colab-inference
./deploy.sh
```

### 4. Start the local proxy

Once you see the tunnel URL in the Colab output:

```bash
node proxy.mjs https://your-tunnel.trycloudflare.com 3000
```

### 5. Use it

```bash
# List models
curl http://localhost:3000/v1/models

# Chat
curl -s http://localhost:3000/v1/chat/completions \
  -X POST -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

Works with any OpenAI-compatible client using `base_url = "http://localhost:3000/v1"`.

## Manual Setup

If you prefer, just copy `colab_server.py` into a Colab cell and run it. Edit the `CONFIG` dict at the top to change the model.

## Files

- `colab_server.py` — Single-cell Colab script (model + FastAPI + cloudflared)
- `proxy.mjs` — Local Node.js proxy (forwards to tunnel)
- `deploy.sh` — Auto-deploy via Chrome DevTools Protocol

## Changing Models

Edit `CONFIG["model_id"]` in `colab_server.py`, or:

```bash
./deploy.sh --model "meta-llama/Llama-3.2-1B-Instruct"
```
