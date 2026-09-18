"""Offline CI gate: both test filename conventions, no skips, bounded Node tests."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    if sys.version_info[:3] != (3, 12, 10):
        raise RuntimeError("Use the pinned CPU test runtime: CPython 3.12.10")
    node = os.environ.get("NODE_BINARY", "node")
    version = subprocess.check_output([node, "--version"], text=True, timeout=10).strip()
    if version != "v24.14.0":
        raise RuntimeError(f"Use pinned Node v24.14.0, found {version}")
    subprocess.run([sys.executable, "scripts/test_environment.py"], check=True, timeout=10)
    # Regenerate then fail on drift, including config/load/setup notebook cells.
    subprocess.run([sys.executable, "scripts/sync_setup.py"], check=True, timeout=10)
    subprocess.run(["git", "diff", "--exit-code", "--", "colab_server.py", "colab_inference_server.ipynb"], check=True, timeout=10)
    for source in [ROOT / "colab_server.py", *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py")]:
        compile(source.read_text(encoding="utf-8"), str(source), "exec")
    for source in ["proxy.mjs", "deploy.mjs", *map(str, ROOT.glob("tests/*.mjs"))]:
        subprocess.run([node, "--check", source], check=True, timeout=10)
    subprocess.run(["git", "diff", "--check"], check=True, timeout=10)
    suite = unittest.defaultTestLoader.discover("tests", pattern="*test*.py")
    if suite.countTestCases() < 31:
        raise RuntimeError("Discovery omitted existing Python tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit("Python gate failed (skips are not acceptance)")
    env = dict(os.environ, TEST_PYTHON=sys.executable)
    subprocess.run([node, "--test", "--test-timeout=20000", *map(str, sorted(ROOT.glob("tests/*.test.mjs")))],
                   check=True, timeout=90, env=env)


if __name__ == "__main__":
    main()
