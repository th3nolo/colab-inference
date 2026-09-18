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
    "max_input_chars": 16000,
    "max_input_tokens": 4096,
    "max_output_tokens": 1024,
    "max_messages": 64,
    "max_concurrent_requests": 1,
    "temperature": 0.1,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "port": 8000,
}

# Read the shared secret before installing dependencies or loading the model.
import os, re, secrets

API_TOKEN = os.environ.get("COLAB_API_TOKEN")
if API_TOKEN is None:
    try:
        from google.colab import userdata
        API_TOKEN = userdata.get("COLAB_API_TOKEN")
    except Exception:
        raise RuntimeError("Set COLAB_API_TOKEN in the environment or Colab Secrets and enable notebook access") from None
if not isinstance(API_TOKEN, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", API_TOKEN):
    raise RuntimeError("COLAB_API_TOKEN must contain 32-256 URL-safe characters; generate it with secrets.token_urlsafe(32)")
for key in ("max_new_tokens", "max_input_chars", "max_input_tokens", "max_output_tokens", "max_messages", "max_concurrent_requests"):
    if type(CONFIG[key]) is not int or CONFIG[key] < 1:
        raise RuntimeError(f"CONFIG[{key!r}] must be a positive integer")
if CONFIG["max_new_tokens"] > CONFIG["max_output_tokens"]:
    raise RuntimeError("max_new_tokens must not exceed max_output_tokens")

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
LOADED_MODEL_ID = CONFIG["model_id"]
print(f"[2/4] Model loaded: {CONFIG['model_id']} on {next(model.parameters()).device}")

# ── Step 3: Start API server ────────────────────────────────────────────────────

import threading, uuid, time as _time
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(title="Colab Inference Server", docs_url=None, redoc_url=None, openapi_url=None)
inference_slots = threading.BoundedSemaphore(CONFIG["max_concurrent_requests"])

@app.middleware("http")
async def authenticate(request: Request, call_next):
    values = request.headers.getlist("authorization")
    parts = values[0].split(" ") if len(values) == 1 else []
    if (len(parts) != 2 or parts[0].lower() != "bearer"
            or not secrets.compare_digest(parts[1].encode(), API_TOKEN.encode())):
        return JSONResponse({"detail": "Invalid or missing bearer token"}, status_code=401,
                            headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)

@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # Do not echo input (which may contain private prompts) in validation errors.
    return JSONResponse({"detail": "Invalid request parameters"}, status_code=422)

class ChatMessage(BaseModel):
    role: str = Field(max_length=32)
    content: str = Field(max_length=CONFIG["max_input_chars"])

class ChatRequest(BaseModel):
    model: str = LOADED_MODEL_ID
    messages: list[ChatMessage] = Field(min_length=1, max_length=CONFIG["max_messages"])
    max_tokens: int = Field(default=CONFIG["max_new_tokens"], strict=True, ge=1, le=CONFIG["max_output_tokens"])
    temperature: float = Field(default=CONFIG["temperature"], ge=0, le=2, allow_inf_nan=False)
    top_k: int = Field(default=CONFIG["top_k"], strict=True, ge=1, le=1000)
    stream: bool = False

@app.get("/v1/models")
def list_models():
    return {"data": [{"id": LOADED_MODEL_ID, "object": "model"}]}

@app.get("/health")
def health():
    return {"status": "ok", "model": LOADED_MODEL_ID, "device": str(next(model.parameters()).device)}

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    if req.model != LOADED_MODEL_ID:
        raise HTTPException(status_code=400, detail={
            "param": "model",
            "message": f"Unsupported model {req.model!r}; loaded model is {LOADED_MODEL_ID!r}.",
        })
    if req.stream:
        raise HTTPException(status_code=400, detail={
            "param": "stream",
            "message": "Streaming is not supported; use stream=false.",
        })
    if sum(len(m.content) + len(m.role) for m in req.messages) > CONFIG["max_input_chars"]:
        raise HTTPException(413, "Input exceeds max_input_chars")
    if not inference_slots.acquire(blocking=False):
        raise HTTPException(429, "Inference is busy; retry shortly", headers={"Retry-After": "1"})
    try:
        return generate_completion(req)
    finally:
        inference_slots.release()


def generate_completion(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=CONFIG["max_input_tokens"] + 1)
    input_len = inputs["input_ids"].shape[1]
    if input_len > CONFIG["max_input_tokens"]:
        raise HTTPException(413, "Input exceeds max_input_tokens")
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(context_limit, int) and input_len + req.max_tokens > context_limit:
        raise HTTPException(413, "Input plus requested output exceeds model context")
    inputs = inputs.to(model.device)

    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()

    # Use the same EOS IDs for generation and finish-reason classification.
    eos_token_id = model.generation_config.eos_token_id
    eos_ids = [] if eos_token_id is None else (
        [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
    )
    t0 = _time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=req.temperature > 0,
            temperature=max(req.temperature, 0.01),
            top_k=req.top_k,
            repetition_penalty=CONFIG["repetition_penalty"],
            max_new_tokens=req.max_tokens,
            eos_token_id=eos_token_id,
        )

    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = _time.perf_counter() - t0
    gen_tokens = output.shape[1] - input_len
    generated = output[0][input_len:]
    ended_on_eos = gen_tokens > 0 and generated[-1].item() in eos_ids
    finish_reason = "length" if gen_tokens >= req.max_tokens and not ended_on_eos else "stop"
    response_text = tokenizer.decode(generated, skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": LOADED_MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": finish_reason,
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
    print('    -H "Authorization: Bearer $COLAB_API_TOKEN" \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f"    -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hello!\"}}]}}'")
else:
    print("[4/4] Tunnel failed. Check /tmp/cf.log")
    print("  Server is still accessible at http://localhost:8000 within this notebook")
