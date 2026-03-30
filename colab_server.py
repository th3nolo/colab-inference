"""
Colab Inference Server
======================
Paste this entire cell into a Google Colab notebook.
It will: install deps, load a model, start a FastAPI server, and expose it via cloudflared.

Usage:
  1. Open Google Colab (GPU runtime recommended)
  2. Paste this code into a cell and run it
  3. Copy the tunnel URL from the output
  4. Use it as an OpenAI-compatible API endpoint

Configuration: edit the CONFIG dict below.
"""

# ── Config ──────────────────────────────────────────────────────────────────────

CONFIG = {
    "model_id": "LiquidAI/LFM2.5-1.2B-Instruct",
    "dtype": "bfloat16",
    "max_new_tokens": 512,
    "temperature": 0.1,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "port": 8000,
}

# ── Step 1: Install dependencies ────────────────────────────────────────────────

import subprocess, sys

def install(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)

install(["transformers", "torch", "accelerate", "fastapi", "uvicorn"])
subprocess.run(
    ["wget", "-q", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "/tmp/cloudflared"],
    check=True,
)
subprocess.run(["chmod", "+x", "/tmp/cloudflared"], check=True)
print("[1/4] Dependencies installed")

# ── Step 2: Load model ──────────────────────────────────────────────────────────

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_id"],
    device_map="auto",
    dtype=CONFIG["dtype"],
)
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_id"])
print(f"[2/4] Model loaded: {CONFIG['model_id']} on {next(model.parameters()).device}")

# ── Step 3: Start API server ────────────────────────────────────────────────────

import threading, uuid, time as _time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Colab Inference Server")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = CONFIG["model_id"]
    messages: list[ChatMessage]
    max_tokens: int = CONFIG["max_new_tokens"]
    temperature: float = CONFIG["temperature"]
    top_k: int = CONFIG["top_k"]
    stream: bool = False

@app.get("/v1/models")
def list_models():
    return {"data": [{"id": CONFIG["model_id"], "object": "model"}]}

@app.get("/health")
def health():
    return {"status": "ok", "model": CONFIG["model_id"], "device": str(next(model.parameters()).device)}

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()

    t0 = _time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=req.temperature > 0,
            temperature=max(req.temperature, 0.01),
            top_k=req.top_k,
            repetition_penalty=CONFIG["repetition_penalty"],
            max_new_tokens=req.max_tokens,
        )

    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = _time.perf_counter() - t0
    gen_tokens = output.shape[1] - input_len
    response_text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": gen_tokens,
            "total_tokens": input_len + gen_tokens,
            "tokens_per_second": round(gen_tokens / elapsed, 1),
        },
    }

threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=CONFIG["port"], log_level="warning"),
    daemon=True,
).start()
_time.sleep(2)
print(f"[3/4] API server running on port {CONFIG['port']}")

# ── Step 4: Start tunnel ────────────────────────────────────────────────────────

import subprocess, re, os

os.system(f"nohup /tmp/cloudflared tunnel --url http://localhost:{CONFIG['port']} > /tmp/cf.log 2>&1 &")
_time.sleep(8)

r = subprocess.run(
    ["grep", "-oE", r"https://[a-z0-9-]+\.trycloudflare\.com", "/tmp/cf.log"],
    capture_output=True, text=True,
)
tunnel_url = r.stdout.strip().split("\n")[0] if r.stdout.strip() else None

if tunnel_url:
    print(f"[4/4] Tunnel ready!\n")
    print(f"  URL:      {tunnel_url}")
    print(f"  Models:   {tunnel_url}/v1/models")
    print(f"  Chat:     {tunnel_url}/v1/chat/completions")
    print(f"  Health:   {tunnel_url}/health")
    print(f"\n  curl -s -X POST {tunnel_url}/v1/chat/completions \\")
    print(f'    -H "Content-Type: application/json" \\')
    print(f"    -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hello!\"}}]}}'")
else:
    print("[4/4] Tunnel failed. Check /tmp/cf.log")
    print("  Server is still accessible at http://localhost:8000 within this notebook")
