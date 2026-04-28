#!/usr/bin/env bash
# Reproduce Open-dCoder 0.5B HumanEval Pass@1.
# Expected (from Open-dLLM README): Pass@1 ~ 20.8, Pass@10 ~ 38.4.
set -euo pipefail

ROOT=${ROOT:-/workspace/diffusion-llm}
EXPDIR="$ROOT/diffusion-experiments"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_baseline_humaneval"
RUN_DIR="$EXPDIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

source "$ROOT/.venv/bin/activate"

cd "$ROOT/Open-dLLM/eval/eval_completion"

echo "==> Running upstream eval; logs -> $RUN_DIR/eval.log"
{
  echo "git rev-parse: $(git -C "$ROOT/Open-dLLM" rev-parse HEAD)"
  echo "host: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  echo "----"
  bash run_eval.sh
} 2>&1 | tee "$RUN_DIR/eval.log"

echo "==> Done. Inspect: $RUN_DIR/"
