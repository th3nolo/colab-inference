"""Real config/model-load/API cells; replace only the GPU/model boundary.

The socket fixture runs production's thread/startup statement with real Uvicorn.
Its bind adapter constrains the listener to an ephemeral loopback socket. No
installer, model download, Colab API or public tunnel is executed.
"""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "test-only-token-" + "x" * 32


def cells(entrypoint):
    if entrypoint == "server":
        source = (ROOT / "colab_server.py").read_text(encoding="utf-8")
        return (
            source[source.index("CONFIG ="):source.index("# BEGIN GENERATED SETUP")],
            source[source.index("from transformers import"):source.index("import threading")],
            source[source.index("import threading"):source.index("import subprocess, re, os")],
        )
    notebook = json.loads((ROOT / "colab_inference_server.ipynb").read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    return tuple(next(cell for cell in code if cell.startswith(prefix)) for prefix in (
        "CONFIG =", "from transformers import", "import threading",
    ))


class Scalar(int):
    def item(self):
        return int(self)


class Tensor:
    def __init__(self, tokens):
        self.tokens = tokens
        self.shape = (1, len(tokens))

    def __getitem__(self, index):
        return [Scalar(token) for token in self.tokens]


class Inputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

    def __call__(self, text, **kwargs):
        # Special prompts control only the external inference boundary.
        return Inputs(input_ids=Tensor([10, 11]), fixture_prompt=text)

    def decode(self, tokens, **kwargs):
        return "fixture answer"


class Model:
    device = SimpleNamespace(type="cpu")
    config = SimpleNamespace(max_position_embeddings=8192)
    generation_config = SimpleNamespace(eos_token_id=2)

    def parameters(self):
        yield SimpleNamespace(device=self.device)

    def generate(self, **kwargs):
        if kwargs["fixture_prompt"] == "slow":
            time.sleep(0.5)
        elif kwargs["fixture_prompt"] == "fail":
            raise RuntimeError("external model fixture failure")
        return Tensor([10, 11] + [7, 2][:kwargs["max_new_tokens"]])


def model_modules(loads):
    def factory(kind, instance):
        def from_pretrained(model_id, **kwargs):
            loads.append((kind, model_id, kwargs))
            return instance
        return SimpleNamespace(from_pretrained=from_pretrained)
    return {
        "torch": SimpleNamespace(no_grad=nullcontext),
        "transformers": SimpleNamespace(
            AutoModelForCausalLM=factory("model", Model()),
            AutoTokenizer=factory("tokenizer", Tokenizer()),
        ),
    }


def serve(entrypoint):
    config, load, api = cells(entrypoint)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = None

    def loopback_run(app, **options):
        nonlocal server
        # Preserve the production startup contract; restrict exposure in tests.
        if options != {"host": "0.0.0.0", "port": port, "log_level": "warning"}:
            raise RuntimeError(f"Unexpected startup contract: {options}")
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical"))
        server.run(sockets=[sock])

    namespace = {"__name__": "socket_fixture"}
    loads = []
    try:
        with patch.dict(os.environ, {"COLAB_API_TOKEN": TOKEN}), patch.dict(sys.modules, model_modules(loads)), patch.object(uvicorn, "run", loopback_run):
            exec(compile(config, entrypoint + ":config", "exec"), namespace)
            namespace["CONFIG"]["port"] = port
            exec(compile(load, entrypoint + ":model", "exec"), namespace)
            exec(compile(api, entrypoint + ":api", "exec"), namespace)
            if server is None or not server.started:
                raise RuntimeError("Production startup did not start Uvicorn")
            print("FIXTURE_READY " + json.dumps({"port": port, "model": namespace["LOADED_MODEL_ID"], "loads": loads}), flush=True)
            # Parent closes stdin; deadline also bounds a crashed parent's child.
            import threading
            finished = threading.Event()
            threading.Thread(target=lambda: (sys.stdin.read(), finished.set()), daemon=True).start()
            finished.wait(30)
    finally:
        if server is not None:
            server.should_exit = True
            deadline = time.monotonic() + 3
            while server.started and sock.fileno() >= 0 and time.monotonic() < deadline:
                time.sleep(0.02)
        sock.close()


if __name__ == "__main__":
    serve(sys.argv[1])
