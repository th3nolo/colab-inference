"""Pinned, hash-checked setup embedded into the pasteable server and notebook."""
import hashlib
import io
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

UV_VERSION = "0.12.9"
UV_URL = "https://github.com/astral-sh/uv/releases/download/0.12.9/uv-x86_64-unknown-linux-gnu.tar.gz"
UV_SHA256 = "ec7a99cd05e0cd7f80243f135ce1361c76835cb0ee60055d14d20eba8eba1460"
CLOUDFLARED_VERSION = "2026.8.0"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/download/2026.8.0/cloudflared-linux-amd64"
CLOUDFLARED_SHA256 = "14ecae0dd17ba74f8055e22b8f5b5acc3cbb5a9c3be4e7d6507fe1c4eadaea95"
# Replaced by scripts/sync_setup.py from the reviewed requirements.lock.
LOCK_TEXT = "__LOCK_TEXT__"


def verified_download(url, expected_sha256, limit):
    """Never return executable bytes before validating their pinned digest."""
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read(limit + 1)
    if len(content) > limit:
        raise RuntimeError("Download exceeded the expected size limit")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise RuntimeError("SHA-256 mismatch: " + url)
    return content


def setup_runtime(config):
    if (sys.version_info[:2] != (3, 12) or platform.system() != "Linux"
            or platform.machine() not in ("x86_64", "AMD64")):
        raise RuntimeError("This lock requires a Linux x86_64 Python 3.12 runtime")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", config.get("model_revision", "")):
        raise ValueError("Set model_revision to a reviewed 40-character Hugging Face commit SHA")
    if any(name in sys.modules for name in ("torch", "transformers", "accelerate", "fastapi", "pydantic")):
        raise RuntimeError("Restart the Colab session before setup: runtime packages are already imported")
    # Private directory avoids reusing or executing a stale /tmp binary.
    directory = Path(tempfile.mkdtemp(prefix="colab-inference-"))
    uv_archive = verified_download(UV_URL, UV_SHA256, 30_000_000)
    with tarfile.open(fileobj=io.BytesIO(uv_archive), mode="r:gz") as archive:
        member = archive.getmember("uv-x86_64-unknown-linux-gnu/uv")
        if not member.isfile() or member.size > 100_000_000:
            raise RuntimeError("Unexpected uv executable archive entry")
        uv_binary = archive.extractfile(member).read()
    uv_path = directory / "uv"
    uv_path.write_bytes(uv_binary)
    uv_path.chmod(0o700)
    lock_path = directory / "requirements.lock"
    lock_path.write_text(LOCK_TEXT, encoding="utf-8")
    subprocess.check_call([
        str(uv_path), "pip", "install", "--python", sys.executable,
        "--no-config", "--index-url", "https://pypi.org/simple",
        "--require-hashes", "--only-binary", ":all:", "--no-deps",
        "--reinstall", "-r", str(lock_path),
    ], env={key: value for key, value in os.environ.items()
            if not key.startswith(("UV_", "PIP_"))})
    cloudflared = directory / "cloudflared"
    cloudflared.write_bytes(verified_download(CLOUDFLARED_URL, CLOUDFLARED_SHA256, 60_000_000))
    cloudflared.chmod(0o700)
    return str(cloudflared)
