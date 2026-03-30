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

## Supported Models

Any HuggingFace model that works with `AutoModelForCausalLM` + `AutoTokenizer` works here.
Just change the `model_id` in the CONFIG cell (notebook) or `colab_server.py`.

### How to choose a model

**Key constraint: Colab free tier gives you a T4 GPU with 16GB VRAM.**

| VRAM needed | What fits |
|-------------|-----------|
| ~2-3 GB | 1B models (fp16) |
| ~4-6 GB | 3-4B models (fp16) |
| ~8-10 GB | 7-8B models (fp16) |
| ~14-16 GB | 7-8B models (fp16 + long context), 14B models (4-bit quantized) |

Rule of thumb: **fp16 uses ~2GB per 1B parameters**. Quantized (4-bit) cuts that in half.

### Recommended models (T4 friendly)

These all fit on a T4, deliver strong quality, and work out of the box with this toolkit.

#### General purpose

| Model | Params | HuggingFace ID | Notes |
|-------|--------|----------------|-------|
| **Qwen 3** | 4B | `Qwen/Qwen3-4B` | Top tier at this size. Code, reasoning, multilingual |
| **Qwen 3** | 8B | `Qwen/Qwen3-8B` | Best overall quality that fits T4 |
| **SmolLM3** | 3B | `HuggingFaceTB/SmolLM3-3B` | Beats Llama-3.2-3B and Qwen2.5-3B |
| **Phi-4 mini** | 3.8B | `microsoft/phi-4-mini-instruct` | Strong reasoning for its size |
| **Gemma 3** | 4B | `google/gemma-3-4b-it` | Google's best small model, multimodal |
| **Mistral Small 3** | 7B | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Fast, great instruction following |

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
    "model_id": "Qwen/Qwen3-8B",  # <-- change this
    ...
}
```

**Option B: Via deploy script**

```bash
./deploy.sh --model "Qwen/Qwen3-8B"
```

**Option C: 4-bit quantization** (for larger models)

If a model barely fits or doesn't fit in 16GB, use 4-bit loading.
Add this to the model loading cell:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_id"],
    device_map="auto",
    quantization_config=quantization_config,
)
```

This lets you run 14B models on a T4. Install `bitsandbytes` first:
```python
!pip install -q bitsandbytes
```

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
| `trust_remote_code` error | Add `trust_remote_code=True` to both `from_pretrained()` calls |
| `tokenizer.apply_chat_template` fails | Model may not have a chat template. Check model card for correct prompt format |
| Slow inference | Make sure GPU runtime is enabled. Check with `!nvidia-smi` |
| Gibberish output | Use the `-Instruct` or `-it` variant, not the base model |

## Manual Setup

If you prefer, just copy `colab_server.py` into a Colab cell and run it. Edit the `CONFIG` dict at the top to change the model.

## Files

- `colab_inference_server.ipynb` — Clean notebook ready to open in Colab
- `colab_server.py` — Single-cell version (copy-paste into any notebook)
- `proxy.mjs` — Local Node.js proxy (forwards to tunnel)
- `deploy.sh` — Auto-deploy via Chrome DevTools Protocol
