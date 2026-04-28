#!/usr/bin/env bash
# DreamOn-v0-7B on SantaCoder-FIM (Python subset). Single-GPU.
# Independent infilling benchmark — confirms HumanEval-Infill score isn't
# benchmark-specific overfitting. Same hyperparameters as our 88.67 baseline.
set -euo pipefail

ROOT=${ROOT:-/workspace/diffusion-llm}
EXPDIR="$ROOT/diffusion-experiments"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_santacoder_fim"
RUN_DIR="$EXPDIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

DREAMON_DIR="$ROOT/DreamOn"
if [ ! -d "$DREAMON_DIR" ]; then
  echo "ERROR: DreamOn not cloned. Run scripts/05 first."
  exit 1
fi

CKPT=Dream-org/DreamOn-v0-7B
OUTPUT_DIR="$DREAMON_DIR/results/Dream-org/DreamOn-v0-7B"
mkdir -p "$OUTPUT_DIR"
RESULT_FILE="$OUTPUT_DIR/santacoder_fim.jsonl"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$DREAMON_DIR"

echo "==> Running SantaCoder-FIM eval (single GPU)"
{
  echo "host: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  python -c "import torch; print('torch', torch.__version__, 'cap', torch.cuda.get_device_capability(0))"
  echo "model: $CKPT"
  echo "----"

  torchrun --nproc_per_node 1 --master_port 29516 \
    -m eval.evaluate --dotlist \
      max_prompt_len=2048 \
      min_gen_len=4 \
      max_gen_len=64 \
      batch_size=1 \
      steps=256 \
      pad_to_max_len=false \
      temperature=0.2 \
      mask_expansion=true \
      delete_eos_token=true \
      overwrite=true \
      alg=entropy \
      alg_temp=0.0 \
      top_p=0.9 \
      show_progress=false \
      ckpt=$CKPT \
      prediction_path=$RESULT_FILE \
      task=santacoder-fim

  echo "==> Scoring (compute_em_santa)"
  python "$DREAMON_DIR/eval/compute_em_santa.py" "$RESULT_FILE"
} 2>&1 | tee "$RUN_DIR/eval.log"

echo ""
echo "==> Done. Reference numbers (HumanEval Infill Pass@1):"
echo "    LLaDA-8B: 35.1   Dream-7B: 40.7   DiffuCoder-7B: 38.8"
echo "    Dream-Coder-7B: 40.0   Open-dCoder 0.5B: 29.6 (oracle: 56.4)"
echo ""
echo "Tail:"
tail -30 "$RUN_DIR/eval.log"
