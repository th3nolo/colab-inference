#!/usr/bin/env bash
# Explicit-target Colab deployment. See README.md for the one-time cell opt-in.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec node "$SCRIPT_DIR/deploy.mjs" "$@"
