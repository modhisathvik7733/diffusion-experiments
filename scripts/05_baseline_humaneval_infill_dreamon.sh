#!/usr/bin/env bash
# DreamOn-v0-7B HumanEval-Infill baseline (single GPU).
# Uses DreamOn's own eval script (it knows how to drive the variable-length
# generation correctly, including <|expand|>/<|delete|> token handling).
#
# Self-contained: clones DreamOn repo if missing, installs extra deps,
# patches multi-GPU defaults, runs eval. Logs land under runs/.
set -euo pipefail

ROOT=${ROOT:-/workspace/diffusion-llm}
EXPDIR="$ROOT/diffusion-experiments"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_baseline_humaneval_infill_dreamon"
RUN_DIR="$EXPDIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

DREAMON_DIR="$ROOT/DreamOn"

if [ ! -d "$DREAMON_DIR" ]; then
  echo "==> Cloning DreamOn repo"
  git clone --depth 1 https://github.com/DreamLM/DreamOn.git "$DREAMON_DIR"
fi

echo "==> Install extra deps for DreamOn eval"
pip install --upgrade omegaconf

echo "==> Confirm human-eval-infilling is installed (Phase 1 set this up)"
which evaluate_infilling_functional_correctness || {
  echo "==> Re-running 03's setup steps to install human-eval-infilling"
  bash "$EXPDIR/scripts/03_baseline_infilling_open_dcoder.sh" \
    || true  # ignore eval failure, we just need installed package
}

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$DREAMON_DIR"
ls eval/ || true

echo "==> Inspecting DreamOn's eval script for multi-GPU defaults to override"
EVAL_SH="eval/eval_humaneval_infilling.sh"
if [ ! -f "$EVAL_SH" ]; then
  echo "ERROR: $DREAMON_DIR/$EVAL_SH not found"
  ls eval/ || true
  exit 1
fi
cat "$EVAL_SH"

echo "==> Running eval (NOTE: if it uses torchrun --nproc_per_node N, we override to 1)"
{
  echo "host: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  python -c "import torch; print('torch', torch.__version__, 'cap', torch.cuda.get_device_capability(0))"
  echo "----"
  # We deliberately do NOT exec the script blindly because it likely hardcodes
  # multi-GPU. Run with NPROC=1 and small batch override if those env vars are
  # respected; otherwise inspect cat output above and we patch in next iteration.
  NPROC_PER_NODE=1 BATCH_SIZE=2 bash "$EVAL_SH"
} 2>&1 | tee "$RUN_DIR/eval.log"

echo "==> Done. Tail:"
tail -50 "$RUN_DIR/eval.log"
