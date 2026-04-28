#!/usr/bin/env bash
# Reproduce Open-dCoder 0.5B HumanEval Pass@1 on a single GPU.
# Upstream run_eval.sh assumes 4 GPUs (CUDA_VISIBLE_DEVICES=4,5,6,7) and runs
# HumanEval/HumanEval+/MBPP/MBPP+ in sequence. We override for 1× RTX 5090
# and only run HumanEval first (fastest baseline). Expected Pass@1 ~ 20.8.
set -euo pipefail

ROOT=${ROOT:-/workspace/diffusion-llm}
EXPDIR="$ROOT/diffusion-experiments"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_baseline_humaneval"
RUN_DIR="$EXPDIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

MODEL_PATH=${MODEL_PATH:-fredzzp/open-dcoder-0.5B}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
STEPS=${STEPS:-128}
TEMPERATURE=${TEMPERATURE:-0.8}
ALG=${ALG:-p2}
BATCH_SIZE=${BATCH_SIZE:-4}

export CUDA_VISIBLE_DEVICES=0
export HF_ALLOW_CODE_EVAL=1

cd "$ROOT/Open-dLLM/eval/eval_completion"

echo "==> HumanEval Pass@1 (1× GPU); logs -> $RUN_DIR/eval.log"
{
  echo "host: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  python -c "import torch; print('torch', torch.__version__, 'cap', torch.cuda.get_device_capability(0))"
  echo "model: $MODEL_PATH  steps=$STEPS  temp=$TEMPERATURE  alg=$ALG  bs=$BATCH_SIZE"
  echo "----"
  accelerate launch --num_processes 1 eval.py \
    --model custom_coder \
    --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
    --tasks humaneval \
    --num_fewshot 0 \
    --batch_size "$BATCH_SIZE" \
    --output_path "$RUN_DIR/humaneval" \
    --log_samples \
    --confirm_run_unsafe_code
} 2>&1 | tee "$RUN_DIR/eval.log"

echo "==> Done. Tail of log:"
tail -40 "$RUN_DIR/eval.log"
