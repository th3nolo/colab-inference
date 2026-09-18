# Reproducible runtime baseline

This baseline targets **CPython 3.12, Linux x86_64, glibc >= 2.28**, with the
PyTorch CUDA 12.6 build. It fails before downloading anything on other Python
versions/platforms. Select a compatible Colab runtime; do not bypass that check.
Start in a fresh session before importing torch, transformers, accelerate,
FastAPI or Pydantic. Setup reinstalls the locked packages into the notebook's
Python interpreter. Other packages bundled by Colab, its driver, GPU allocation,
OS image and Python patch version remain outside this lock. A pinned package
graph is not a reproducible Colab machine image.

## Sources and explicit versions

| Component | Pin | Purpose / source |
| --- | --- | --- |
| Transformers | 5.16.1 | LFM2 model/tokenizer loading; PyPI `transformers`, Hugging Face |
| PyTorch | 2.13.0+cu126 | CUDA inference; exact CPython 3.12 wheel from `download.pytorch.org/whl/cu126` |
| Accelerate | 1.15.0 | `device_map="auto"`; PyPI `accelerate`, Hugging Face |
| FastAPI | 0.141.1 | HTTP API; PyPI `fastapi` |
| Starlette | 1.3.1 | Explicit patched ASGI dependency; PyPI `starlette` |
| Uvicorn | 0.40.0 | ASGI server; PyPI `uvicorn` |
| uv | 0.12.9 | Installer; official `astral-sh/uv` release archive |
| cloudflared | 2026.8.0 | Tunnel executable; official `cloudflare/cloudflared` release |
| Model + tokenizer | `0f604ada3f766f9f257460c4c9f0b5d6f69d431b` | Same reviewed commit in `LiquidAI/LFM2.5-1.2B-Instruct` |

`pyproject.toml` records direct requirements and scopes the PyTorch index to
PyTorch only. `requirements.lock` locks all 62 runtime distributions, including
transitive dependencies, with SHA-256 hashes. Runtime installation uses binary
wheels only, disables dependency resolution, requires hashes and explicitly
reinstalls the lock so existing unverified installations are not silently reused.
PyTorch's exact wheel URL prevents selecting the default CUDA 13 distribution.
Lock compilation installs no packages. CI and local HTTP tests use 16 audited,
hash-locked CPU dependencies in `work/test-env`; no GPU packages are installed.
`scripts/test_environment.py` derives this subset without resolving new versions
and rechecks every artifact against PyPI before installation. See the README's
complete environment and check commands.

The uv archive and cloudflared binary are downloaded from versioned official
release URLs and checked against hardcoded SHA-256 digests **before execution**.
uv extraction reads only its expected regular-file archive entry. Both binaries
live in a newly created private temporary directory; the tunnel uses that verified
path rather than a potentially stale `/tmp/cloudflared`. No moving installer is
used. Setup downloads roughly 60 MB of installer/tunnel artifacts in addition to
the much larger runtime wheels and model when a user actually runs it in Colab.

## Review evidence (2026-09-18)

- `dependency-audit.json` records each exact package version, official registry
  metadata URL, project-source URLs, release timestamp and PyPI/OSV advisory
  result. All 62 runtime distributions plus uv have no published advisories in
  that snapshot. This is an advisory-feed result, not a guarantee of safety.
- All selected releases are older than 72 hours. Resolution uses the fixed
  cutoff `2026-09-14T00:00:00Z`; the audit also checks every locked PyPI artifact
  hash against its registry digest, upload age and yanked status. The CUDA wheel
  hash comes from the official PyTorch index; its public 2.13.0 release date is
  2026-07-08. Its 38 KB PEP 658 metadata was read without fetching the GPU wheel.
- Candidates Transformers 4.57.6, torch 2.10.0, Accelerate 1.14.0 and uv 0.10.4
  were rejected after advisory checks. FastAPI 0.128.0 itself had no advisory,
  but constrained Starlette to an affected version, so the baseline uses
  FastAPI 0.141.1 and explicitly pins patched Starlette 1.3.1.
- `artifact-audit.json` records official GitHub release dates, URLs, sizes and
  SHA-256 digests. Both uv and cloudflared were downloaded locally, their actual
  bytes matched those digests, and neither Linux executable was run locally.
  GitHub's public cloudflared advisories GHSA-7mjv-x3jf-545x (Windows installer, fixed 2023.3.1) and GHSA-hgwp-4vp4-qmm2 (fixed 2020.8.1) do not affect the selected 2026.8.0 Linux binary. Embedded Go dependencies were not independently scanned.
- Provenance here means official owner-controlled registries/repositories,
  immutable model commit and publisher-provided artifact hashes over HTTPS.
  It does **not** claim independent reproducible-build or signing-attestation
  verification. A compromised publisher could publish malicious code and hashes.
- Model metadata at the pinned commit identifies `Lfm2ForCausalLM` / `lfm2`.
  Both model and tokenizer loads use that commit, explicitly disable remote
  Python code, and model loading requires safetensors. The native `Lfm2ForCausalLM` class was also confirmed in the tagged Transformers v5.16.1 source. Custom models must use a
  reviewed immutable commit and work with this dependency graph; changing an ID
  alone is insufficient.

## Validation and limits

Offline unit tests cover digest rejection, size limits, valid downloads,
unsupported runtimes, mutable model revision rejection, exact locked install
arguments, private executable paths, and generated notebook/server parity.
The CPU gate uses FastAPI 0.141.1, Starlette 1.3.1, Pydantic 2.13.5 and HTTPX
0.28.1 on pinned Python 3.12.10. Starlette emits a test-client deprecation notice
for HTTPX; this compatible, locked dependency is retained. Real Uvicorn startup
and proxy requests run on bounded loopback sockets for both notebook and server
source. Only external model/GPU behavior is substituted in that path. Generated
deployment-cell tests verify custom model/revision propagation into actual
model-load code and API responses. This is **not** a completed Colab/T4 inference test: no GPU
package installation, model download, live notebook execution or public tunnel
was performed. Driver compatibility, preinstalled optional Colab packages,
model execution, precision support and inference output still need a fresh
Colab runtime smoke test. An initial direct-URL resolution began downloading a
PyTorch wheel and was stopped; subsequent resolution uses index metadata.

Run offline checks with an existing Python interpreter:

```bash
python -m unittest discover -s tests -p test_setup.py -v
```

Generated copies remain pasteable without repository access. Edit
`scripts/runtime_setup.py` and the reviewed lock, then run:

```bash
python scripts/sync_setup.py
```

To deliberately refresh/revalidate the lock using an already installed, reviewed
uv (original compilation used uv 0.11.19), run:

```bash
python scripts/lock_runtime.py
```

The script preserves existing resolved versions, resolves via the explicitly
scoped package indexes, checks registry metadata/advisories, and regenerates
setup/config/load cells. It does not install packages. Review all lock changes,
source identities, release ages, advisories and wheel compatibility before using
a changed lock. For an updated date policy, change the cutoff deliberately in
both the resolver and audit script. The generator intentionally leaves API,
authentication and tunnel cells owned by other changes intact.

## Manual GPU acceptance (opt-in)

There is no Colab service or GPU runner configured in GitHub Actions. The manual
workflow trigger runs CPU contracts only. The following procedure is for an
already available, user-controlled free Colab GPU session; it does not provision
paid compute. Allow at most 15 minutes for setup and stop on any unsupported
runtime, unavailable GPU, OOM, installation failure or timeout. Record the step
and error as a blocked/failed acceptance rather than substituting CPU fixtures.

1. Open the notebook from the exact PR commit in a fresh compatible CPython 3.12
   Linux x86_64 session. Confirm the allocated GPU and driver with `nvidia-smi`.
   Record the commit, Python version, GPU, driver, and model/revision. Do not
   bypass runtime or artifact checks. Keep the existing shared token in Colab
   Secrets/environment; never paste it into an issue, transcript or test log.
2. Execute the notebook's actual config, locked setup, model-load, API and tunnel
   cells in order once. The default model and tokenizer must use the committed
   revision, safetensors and `trust_remote_code=False`. Confirm the loaded
   device is CUDA. Installation and model download are real work here and are
   outside the CPU CI result. A custom model is a separate acceptance run with
   a reviewed immutable revision and sufficient free memory.
3. Run this bounded cell after successful startup. It makes two real completion
   requests (at most 16 new tokens each), one through the local API and one
   through the existing tunnel. Use only the public synthetic prompt below.

   ```python
   import json, urllib.request, urllib.error
   assert next(model.parameters()).device.type == "cuda"
   assert tunnel_url and tunnel_url.startswith("https://")
   for base in (f"http://127.0.0.1:{CONFIG['port']}", tunnel_url):
       def call(path, data=None, authenticated=True):
           headers = {"Content-Type": "application/json"}
           if authenticated:
               headers["Authorization"] = "Bearer " + API_TOKEN
           request = urllib.request.Request(base + path, data=data, headers=headers)
           try:
               with urllib.request.urlopen(request, timeout=30) as response:
                   return response.status, json.load(response)
           except urllib.error.HTTPError as error:
               return error.code, json.load(error)
       assert call("/health", authenticated=False)[0] == 401
       status, health = call("/health")
       assert status == 200 and health["model"] == LOADED_MODEL_ID
       status, models = call("/v1/models")
       assert status == 200 and models["data"][0]["id"] == LOADED_MODEL_ID
       body = {"model": LOADED_MODEL_ID, "messages": [{"role": "user", "content": "Reply with the word hello."}],
               "max_tokens": 16, "temperature": 0, "stream": False}
       status, result = call("/v1/chat/completions", json.dumps(body).encode())
       assert status == 200 and result["model"] == LOADED_MODEL_ID
       assert result["choices"][0]["message"]["content"].strip()
       assert result["choices"][0]["finish_reason"] in ("stop", "length")
       usage = result["usage"]
       assert 0 < usage["completion_tokens"] <= 16
       assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
       for changes, expected in (({"stream": True}, 400), ({"model": "invalid/model"}, 400), ({"max_tokens": 0}, 422)):
           assert call("/v1/chat/completions", json.dumps(body | changes).encode())[0] == expected
   print("Real GPU API and tunnel smoke checks passed")
   ```

4. From the local machine, start the existing proxy with the same environment
   token and returned tunnel URL. Repeat `/health`, `/v1/models` and one
   completion capped at 16 tokens using the documented client requests. Use a
   30-second client timeout. Confirm loopback binding and that disallowed
   browser origins receive 403. Stop the task-owned proxy after testing.
5. If testing `deploy.mjs`, use an explicitly selected notebook and its marked
   managed cell in a separate fresh session. Bound `--timeout` to 900 seconds;
   confirm completion is reported only after Python startup succeeds. This
   checks the current private Colab page API and must be recorded separately
   from manual notebook execution. Never overwrite unrelated cells.

Save a redacted result with the exact commit, environment/model identifiers,
pass/fail per step, token counts and errors. Report GPU, public tunnel, local
proxy and automated notebook deployment acceptance separately. Disconnect the
test session when finished. A proxy timeout closes HTTP transport but does not
guarantee cancellation of GPU generation already in progress.
