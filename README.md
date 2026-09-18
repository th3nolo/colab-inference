# Colab Inference Server

Run LLMs on Google Colab's free GPUs and serve them locally via an OpenAI-compatible API.

## Prerequisites

- **Chromium/Chrome** with remote debugging enabled
- **Node.js 22.15+** (built-in WebSocket for deployment and the local proxy)
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

### 3. Configure the shared token and deploy

Generate a fresh random token locally for each temporary session (for example, use
`uv run --no-project python -c "import secrets; print(secrets.token_urlsafe(32))"`).
Store it in **Colab Secrets** as `COLAB_API_TOKEN` and enable notebook access.
Alternatively, provision that environment variable in the Colab runtime.
The server fails before installation/model loading if the token is missing or malformed.
Never paste the token into notebook source, CONFIG, committed files, URLs, or saved outputs.

Set the same token in the local proxy environment without putting it in shell history:

```bash
read -rsp 'Colab API token: ' COLAB_API_TOKEN; echo
export COLAB_API_TOKEN
```

PowerShell (7+) equivalent:

```powershell
$env:COLAB_API_TOKEN = Read-Host 'Colab API token' -MaskInput
```

This preserves the short-lived public HTTPS tunnel workflow: only callers with the
shared token can use the Colab server. Stop the Colab runtime/tunnel when finished
(e.g. after your two-hour session), unset the local variable, and use a new token
next time. There is no automatic two-hour expiry. Colab Secrets persist until changed.

Deploy after adding the Colab secret:

Save the notebook first. Create one dedicated **code cell** whose first line is exactly:

```python
# colab-inference:managed-deploy:v1
```

Then run:

```bash
cd ~/colab-inference
./deploy.sh --notebook-url "https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID"
```

After that one-time opt-in, rerun the same command for deployment. It updates only this marked cell with the current local `colab_server.py`; all other cells are preserved. Do not put personal code in the managed cell. Missing or duplicate markers stop deployment. A URL open in multiple tabs is ambiguous: close duplicates or use `--tab-id ID` from the local CDP `http://127.0.0.1:9222/json/list` instead. Exactly one selector is required; query strings and fragments do not identify different notebooks.

Identical reruns skip duplicate server startup in the same runtime. For changed code/model, **restart the Colab runtime**, then run the same deploy command; no new opt-in is needed. After an interrupted or timed-out deployment, inspect the notebook before retrying. If the browser still reports pending/running after a runtime restart, reload that notebook tab to clear the stale status. The tool does not dismiss security/authorization prompts or change runtimes.

The command waits up to 600 seconds (override with `--timeout SECONDS`, maximum 3600), reports Python tracebacks and CDP errors, and exits nonzero on failure or unknown status. Completion means the server script finished and reported a tunnel URL, not that the remote endpoint has been independently health-checked. `--port` only changes the suggested local proxy command; it does not start the proxy.

Colab's page API is private and may change. Unsupported cell APIs fail closed; use manual setup if necessary. This automation is covered by local mocks, not a live notebook test.

### 4. Start the local proxy

Once you see the tunnel URL in the Colab output:

```bash
node proxy.mjs https://your-tunnel.trycloudflare.com 3000
```

The proxy listens only on `127.0.0.1`. Use `http://127.0.0.1:3000/v1` if
`localhost` resolves to IPv6 in your client. Command-line and SDK callers need
no CORS configuration. Browser requests from other origins are rejected before
forwarding, including simple POST requests. For a local web app, allow its exact
origin explicitly (no wildcard):

```bash
PROXY_ALLOWED_ORIGINS=http://localhost:5173 node proxy.mjs https://your-tunnel.trycloudflare.com 3000
```

In PowerShell, set `$env:PROXY_ALLOWED_ORIGINS = "http://localhost:5173"` before
running the existing `node proxy.mjs ...` command. Multiple origins can be
comma-separated. Only add origins you trust to make inference requests.

Request bodies are limited to 1 MiB, including chunked uploads; larger requests
receive HTTP 413. Upstream requests have a five-minute total deadline covering
headers and response streaming. Set `PROXY_UPSTREAM_TIMEOUT_MS` to a positive
integer in milliseconds to adjust it for slower inference (maximum 2147483647).
Timeouts before response headers return 504; a timeout after streaming starts
closes the response. Closing the local client connection aborts the upstream
request; this does not guarantee cancellation of GPU work already started by
the remote server. Upstream redirects are rejected.

Loopback binding and browser restrictions do not authenticate local processes.
The proxy holds the shared token and supplies it to the authenticated public server.

Proxy regression tests use only local mock HTTP servers, with no dependencies,
model downloads, or public tunnel:

```bash
node --test --test-timeout=5000 tests/proxy.test.mjs
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

The local proxy supplies `Authorization: Bearer $COLAB_API_TOKEN` upstream.
Treat access to this proxy as access to the model; its loopback bind and browser
origin restrictions limit who can use it.

For direct HTTPS tunnel access, use the shared token as your provider/client API key:

```bash
curl "$TUNNEL_URL/v1/models" -H "Authorization: Bearer $COLAB_API_TOKEN"
```

```python
import os
from openai import OpenAI  # if your client already uses the OpenAI SDK
client = OpenAI(base_url=os.environ["TUNNEL_URL"] + "/v1",
                api_key=os.environ["COLAB_API_TOKEN"])
```

All server routes, including `/health` and `/v1/models`, require the token; missing,
invalid, or duplicate authorization headers return `401`. Secrets are never printed
by the server or proxy. Documentation endpoints are disabled.

### Inference limits

Adjust these positive integers in CONFIG in either deployment format:

| Setting | Default | Effect |
| --- | ---: | --- |
| `max_input_chars` | 16000 | Total message content and role characters before tokenization |
| `max_messages` | 64 | Maximum messages per request |
| `max_input_tokens` | 4096 | Prompt token cap, including chat-template overhead |
| `max_output_tokens` | 1024 | Maximum allowed client `max_tokens` |
| `max_new_tokens` | 512 | Default output budget, no larger than `max_output_tokens` |
| `max_concurrent_requests` | 1 | Concurrent tokenization/generation operations |

Oversized prompts return `413` (schema length violations return `422`); invalid
output budgets return `422`. Prompt plus output must fit the model's advertised
`max_position_embeddings`, when available. Busy inference returns `429` with
`Retry-After: 1`; clients should retry with backoff. Tokenization and generation
share the slot, and exceptions always release it. Keep concurrency at one on a
small Colab GPU unless you have measured available memory. These are inference
limits; proxy transport/body/time limits are maintained by the proxy hardening PR.

### Chat API behavior

Use the exact loaded model ID from `/v1/models`, or omit `model` to use the default.
Unsupported model IDs and `stream=true` return HTTP 400 before tokenization or
acquiring an inference slot. Streaming is not supported; use `stream=false` (the
default). Responses, model listing and health identify the model captured at load
time, even if CONFIG is edited without reloading it.

`finish_reason` is `length` when generation exhausts `max_tokens` without ending
on an EOS token, and `stop` for EOS or earlier stopping. EOS on the final allowed
token counts as `stop`. The authenticated API's existing schema enforces positive
output limits and its configured maximum.

Offline API contract tests (standard library only; no model downloads or GPU):

```bash
uv run --offline --no-project python -m unittest discover -s tests -p test_api_contract.py -v
```

These tests exercise server and notebook handler definitions using dependency
doubles. The separate authentication tests exercise real FastAPI routing with
mocked inference; neither suite starts a live Colab runtime.

## Supported Models

Use Hugging Face models supported by the pinned `AutoModelForCausalLM` + `AutoTokenizer` runtime.
Change both `model_id` and `model_revision` (a reviewed 40-character commit SHA) in the CONFIG cell or `colab_server.py`. The same revision pins the tokenizer. Models must support the pinned runtime and safetensors without remote Python code.

### How to choose a model

**Key constraint: Colab free tier gives you a T4 GPU with 16GB VRAM.**

| VRAM needed | What fits |
|-------------|-----------|
| ~2-3 GB | 1B models (fp16) |
| ~4-6 GB | 3-4B models (fp16), 7B models (4-bit — measured 4.8 GB for Mistral 7B) |
| ~8-10 GB | 14B models (4-bit quantized) |
| ~14-16 GB | 7-8B models (fp16) — tight: Mistral 7B bf16 measured 14.5 GB and spills 5 of 36 modules to CPU on the T4's 15.0 GB, dropping to ~2 tok/s |

Rule of thumb: **fp16 uses ~2GB per 1B parameters**. Quantized (4-bit) cuts that in half.

### Recommended models (T4 friendly)

These all fit on a T4 and work out of the box with this toolkit.

#### General purpose

| Model | Params | HuggingFace ID | Notes |
|-------|--------|----------------|-------|
| **Qwen 3** | 4B | `Qwen/Qwen3-4B` | Top tier at this size. Code, reasoning, multilingual |
| **Qwen 3** | 8B | `Qwen/Qwen3-8B` | Best overall quality that fits T4 |
| **SmolLM3** | 3B | `HuggingFaceTB/SmolLM3-3B` | Beats Llama-3.2-3B and Qwen2.5-3B |
| **Phi-4 mini** | 3.8B | `microsoft/phi-4-mini-instruct` | Strong reasoning for its size |
| **Gemma 3** | 4B | `google/gemma-3-4b-it` | Google's best small model, multimodal |
| **Mistral 7B Instruct** | 7B | `mistralai/Mistral-7B-Instruct-v0.3` | Use 4-bit on T4: 4.8 GB, ~6 tok/s (verified); fp16 spills to CPU and drops to ~2 tok/s |

#### Code focused

| Model | Params | HuggingFace ID | Notes |
|-------|--------|----------------|-------|
| **Qwen 2.5 Coder** | 7B | `Qwen/Qwen2.5-Coder-7B-Instruct` | Best small coding model |
| **CodeGemma** | 7B | `google/codegemma-7b-it` | Good for code completion |

#### Reasoning / thinking

| Model | Params | HuggingFace ID | Notes |
|-------|--------|----------------|-------|
| **Qwen3** | 4B/8B | `Qwen/Qwen3-4B` | Built-in thinking mode via `/think` |
| **DeepSeek R1 Distill** | 7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Chain-of-thought reasoning |

#### Lightweight / fast

| Model | Params | HuggingFace ID | Notes |
|-------|--------|----------------|-------|
| **Qwen 3.5** | 0.6B | `Qwen/Qwen3.5-0.6B` | Tiny but capable |
| **SmolLM3** | 3B | `HuggingFaceTB/SmolLM3-3B` | Great speed/quality ratio |
| **LFM 2.5** | 1.2B | `LiquidAI/LFM2.5-1.2B-Instruct` | Non-transformer, fast on CPU too |

### How to deploy a different model

**Option A: Edit the notebook**

Change the CONFIG cell:
```python
CONFIG = {
    "model_id": "Qwen/Qwen3-8B",
    "model_revision": "<reviewed 40-character commit SHA>",
    ...
}
```

**Option B: Via deploy script**

```bash
./deploy.sh --notebook-url "https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID" \
  --model "Qwen/Qwen3-8B" --revision "<40-hex-model-commit>"
```

Custom models require both an immutable Hugging Face commit (not a branch/tag) and a server version with `CONFIG.model_revision` support; model and tokenizer use the same revision. Without these flags, deployment preserves the server's checked-in model defaults.

**Option C: 4-bit quantization** (for larger models)

If a model barely fits or doesn't fit in 16GB, use 4-bit loading.
Add this to the model loading cell:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_id"],
    device_map="auto",
    revision=CONFIG["model_revision"],
    trust_remote_code=False,
    use_safetensors=True,
    quantization_config=quantization_config,
)
```

Quantization is an optional extension, outside the locked baseline. Before enabling it, select and audit an explicit compatible `bitsandbytes` version, add it to `pyproject.toml`, regenerate the lock and validate it on your GPU. No unpinned installation command is provided.

### Finding models on HuggingFace

1. Go to [huggingface.co/models](https://huggingface.co/models)
2. Filter by: **Text Generation**, sort by **Trending** or **Most Downloads**
3. Check the model card for:
   - **Size** — does it fit in 16GB? (check VRAM requirements)
   - **License** — is it permissive for your use case?
   - **`trust_remote_code=True`** — some models need this flag added to `from_pretrained()`
4. Copy the model ID (e.g. `Qwen/Qwen3-8B`) and use it in CONFIG

### Troubleshooting models

| Problem | Fix |
|---------|-----|
| `OutOfMemoryError` | Model too large. Use a smaller variant or enable 4-bit quantization |
| `trust_remote_code` error | Choose a model natively supported by the pinned Transformers version |
| `tokenizer.apply_chat_template` fails | Model may not have a chat template. Check model card for correct prompt format |
| Slow inference | Make sure GPU runtime is enabled. Check with `!nvidia-smi` |
| Gibberish output | Use the `-Instruct` or `-it` variant, not the base model |

## Manual Setup

Copy `colab_server.py` into a fresh **Linux x86_64 / Python 3.12** Colab cell and run it. Edit both the model ID and immutable revision in `CONFIG` when changing the model. Setup installs the embedded hash-locked runtime and verifies the uv/cloudflared executable downloads. Other Python versions fail before installation.

See [Reproducible runtime baseline](REPRODUCIBILITY.md) for exact versions, release-age/advisory evidence, offline tests, maintenance commands and the remaining Colab validation limits. Deployment PR #4 must accompany this change to remove the old deployment installer.

## Files

- `colab_inference_server.ipynb` — Notebook ready to open in Colab
- `colab_server.py` — Single-cell version (copy-paste into any notebook)
- `proxy.mjs` — Local Node.js proxy (forwards to tunnel)
- `deploy.sh` — Auto-deploy via Chrome DevTools Protocol


## Authentication regression tests

With FastAPI, Pydantic v2 and httpx available in your existing Python environment:

```bash
uv run --no-project python -m unittest discover -s tests -p 'test_auth.py' -v
node --test --test-timeout=5000 tests/proxy.test.mjs
```

Python tests execute only configuration and API definitions with mocked model,
tokenizer and torch objects. They never execute dependency installation, model
loading, server startup, or tunnel startup. Both suites use local mocks only.

## Deployment regression tests

Run `node --test tests/deploy.test.mjs`. Tests use notebook/CDP fixtures and never contact a real notebook, download a model, or start a tunnel. No packages are installed by the deployment launcher.

Test the generated Python cell with an existing interpreter: `uv run --no-project --no-python-downloads --python /path/to/python tests/deploy_wrapper_test.py`. This uses only the Python standard library and Node; it mocks Colab output and server startup.
