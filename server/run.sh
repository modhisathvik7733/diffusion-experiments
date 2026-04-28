#!/usr/bin/env bash
# Start the DreamOn inference server on the Vast.ai box.
# Expects:
#   - Model deps already installed via diffusion-experiments/scripts/00_remote_setup.sh
#   - DIFFUSION_SHARED_SECRET set in env (or read from /workspace/.diffusion_token)
#   - cloudflared running separately (see README.md)
set -euo pipefail

ROOT=${ROOT:-/workspace/diffusion-llm}
SERVER_DIR="$ROOT/diffusion-experiments/server"

cd "$SERVER_DIR"

# Load shared secret from disk if not already in env
if [ -z "${DIFFUSION_SHARED_SECRET:-}" ] && [ -f /workspace/.diffusion_token ]; then
  export DIFFUSION_SHARED_SECRET="$(cat /workspace/.diffusion_token)"
fi
if [ -z "${DIFFUSION_SHARED_SECRET:-}" ]; then
  echo "ERROR: DIFFUSION_SHARED_SECRET not set and /workspace/.diffusion_token missing"
  echo "Generate one with: openssl rand -hex 32 > /workspace/.diffusion_token"
  exit 1
fi

# Install server-only deps (idempotent)
pip install --quiet -r requirements.txt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PORT=${PORT:-8000}

echo "==> Starting uvicorn on 0.0.0.0:$PORT (single worker)"
exec uvicorn app:app --host 0.0.0.0 --port "$PORT" --workers 1 --log-level info
