#!/usr/bin/env bash
#
# deploy.sh - Deploy inference server to Google Colab via CDP
#
# Prerequisites:
#   - Chromium running with --remote-debugging-port=9222 --remote-allow-origins=*
#   - A Colab notebook open in the browser
#   - uv installed (for websocket-client)
#
# Usage:
#   ./deploy.sh                    # Auto-detect Colab tab, deploy, start proxy on :3000
#   ./deploy.sh --port 8080        # Use custom local port
#   ./deploy.sh --model "org/model" # Use a different model
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=3000
MODEL="LiquidAI/LFM2.5-1.2B-Instruct"

while [[ $# -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Preflight checks ───────────────────────────────────────────────────────────

echo "==> Preflight checks..."

# Check Chromium is running with debugging
if ! curl -s http://localhost:9222/json/version > /dev/null 2>&1; then
  echo "ERROR: Chromium not running with remote debugging."
  echo ""
  echo "  Start it with:"
  echo "  chromium --remote-debugging-port=9222 --remote-allow-origins=*"
  echo ""
  echo "  NOTE: --remote-allow-origins=* is required or CDP connections get 403."
  echo "  NOTE: You must be logged into Google in the browser for Colab access."
  exit 1
fi

# Check uv is available
if ! command -v uv &> /dev/null; then
  echo "    Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
fi

echo "    Chromium: OK"
echo "    uv: OK"

echo "==> Finding Colab tab..."
TAB_ID=$(curl -s http://localhost:9222/json/list | python3 -c "
import sys, json
tabs = json.load(sys.stdin)
colab = [t for t in tabs if t['type'] == 'page' and 'colab.research.google.com' in t.get('url','')]
if colab:
    print(colab[0]['id'])
else:
    print('NONE')
")

if [ "$TAB_ID" = "NONE" ]; then
  echo "ERROR: No Colab tab found. Open a Colab notebook in Chromium first."
  exit 1
fi
echo "    Tab: $TAB_ID"

echo "==> Reading colab_server.py..."
COLAB_CODE=$(python3 -c "
with open('$SCRIPT_DIR/colab_server.py') as f:
    code = f.read()
# Patch model if needed
code = code.replace('LiquidAI/LFM2.5-1.2B-Instruct', '$MODEL')
import json
print(json.dumps(code))
")

echo "==> Injecting into Colab notebook..."
uv run --with websocket-client python3 << PYEOF
import json, time, websocket

ws = websocket.create_connection("ws://localhost:9222/devtools/page/$TAB_ID")
ws.settimeout(5)
ws.send(json.dumps({"id":1, "method":"Runtime.enable"}))
try:
    while True: ws.recv()
except: pass

ws.settimeout(15)

code_js = $COLAB_CODE

expr = f"""(function() {{
    var nb = colab.global.notebook;
    var cell = nb.cells[0];
    cell.setText({json.dumps(code_js)});
    cell.executeCellCommand();
    return "deployed to cell 0";
}})()"""

ws.send(json.dumps({"id":99, "method":"Runtime.evaluate", "params":{"expression": expr, "returnByValue": True}}))
try:
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == 99:
            print("   ", r.get("result",{}).get("result",{}).get("value",""))
            break
except Exception as e:
    print(f"    Error: {e}")

ws.close()
PYEOF

echo "==> Waiting for model to load and tunnel to start..."
echo "    This takes ~60-90 seconds (model download + cloudflared)"
echo ""
echo "    IMPORTANT: Make sure the Colab notebook is set to GPU runtime:"
echo "    Runtime -> Change runtime type -> T4 GPU"
echo ""
echo "    Watch the Colab notebook for progress."
echo "    When you see the tunnel URL, copy it and run:"
echo ""
echo "    node $SCRIPT_DIR/proxy.mjs <tunnel-url> $PORT"
echo ""
echo "    Then use: http://localhost:$PORT/v1/chat/completions"
