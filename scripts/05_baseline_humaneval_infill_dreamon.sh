#!/usr/bin/env bash
# DreamOn-v0-7B HumanEval-Infill baseline (single GPU).
# Bypasses upstream's hardcoded 8-GPU wrapper and calls torchrun --nproc_per_node 1
# with the same dotlist hyperparameters. First run does ONE (min_gen_len, max_gen_len)
# combo; full sweep over (4,8,16,32,64) can come later.
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

which evaluate_infilling_functional_correctness || {
  echo "ERROR: evaluate_infilling_functional_correctness missing — run 03 first to install human-eval-infilling"
  exit 1
}

CKPT=${CKPT:-Dream-org/DreamOn-v0-7B}
MIN_GEN_LEN=${MIN_GEN_LEN:-4}
MAX_GEN_LEN=${MAX_GEN_LEN:-64}
STEPS=${STEPS:-256}
BATCH_SIZE=${BATCH_SIZE:-1}
TEMPERATURE=${TEMPERATURE:-0.2}

OUTPUT_DIR="$DREAMON_DIR/results/$(basename "$(dirname "$CKPT")")/$(basename "$CKPT")"
mkdir -p "$OUTPUT_DIR"
RESULT_FILE="$OUTPUT_DIR/humaneval_infill_dynamic_min${MIN_GEN_LEN}_max${MAX_GEN_LEN}_expand.jsonl"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$DREAMON_DIR"

echo "==> Running single-GPU DreamOn HumanEval-Infill eval"
echo "    ckpt=$CKPT  min=$MIN_GEN_LEN  max=$MAX_GEN_LEN  steps=$STEPS  bs=$BATCH_SIZE"
echo "    output -> $RESULT_FILE"

{
  echo "host: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  python -c "import torch; print('torch', torch.__version__, 'cap', torch.cuda.get_device_capability(0))"
  echo "----"

  torchrun --nproc_per_node 1 --master_port 29514 \
    -m eval.evaluate --dotlist \
      max_prompt_len=2048 \
      min_gen_len=$MIN_GEN_LEN \
      max_gen_len=$MAX_GEN_LEN \
      batch_size=$BATCH_SIZE \
      steps=$STEPS \
      pad_to_max_len=false \
      temperature=$TEMPERATURE \
      mask_expansion=true \
      delete_eos_token=true \
      overwrite=true \
      alg=entropy \
      alg_temp=0.0 \
      top_p=0.9 \
      show_progress=true \
      ckpt=$CKPT \
      prediction_path=$RESULT_FILE \
      task=humaneval_infill

  echo "==> Scoring with evaluate_infilling_functional_correctness"
  evaluate_infilling_functional_correctness single-line "$RESULT_FILE"
} 2>&1 | tee "$RUN_DIR/eval.log"

echo "==> Done. Tail:"
tail -50 "$RUN_DIR/eval.log"
