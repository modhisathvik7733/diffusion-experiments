#!/usr/bin/env bash
# Open-dCoder 0.5B HumanEval-Infill Pass@1 baseline (single GPU).
# Expected from Open-dLLM README: Pass@1 ~ 32.5 (or 77.4 with oracle length).
#
# Self-contained: patches the broken upstream `human-eval-infilling` console-script
# entry point, installs the package, then runs single-GPU infilling eval.
set -euo pipefail

ROOT=${ROOT:-/workspace/diffusion-llm}
EXPDIR="$ROOT/diffusion-experiments"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_baseline_infilling"
RUN_DIR="$EXPDIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

HEI_DIR="$ROOT/Open-dLLM/human-eval-infilling"

echo "==> Patch upstream setup.py entry point (\`:None\` -> \`:main\`)"
# Idempotent: if already :main, no change.
sed -i \
  's|human_eval_infilling.evaluate_functional_correctness"|human_eval_infilling.evaluate_functional_correctness:main"|' \
  "$HEI_DIR/setup.py"
grep -q "evaluate_functional_correctness:main" "$HEI_DIR/setup.py" || {
  echo "ERROR: patch did not apply"; exit 1;
}

echo "==> Install human-eval-infilling (no build isolation so pkg_resources works)"
pip install --no-build-isolation -e "$HEI_DIR"
which evaluate_infilling_functional_correctness

echo "==> Fetch missing HumanEval-Infill benchmark data files"
DATA_DIR="$HEI_DIR/data"
mkdir -p "$DATA_DIR"
for f in HumanEval-SingleLineInfilling HumanEval-MultiLineInfilling HumanEval-RandomSpanInfilling HumanEval-RandomSpanInfillingLight; do
  if [ ! -s "$DATA_DIR/$f.jsonl.gz" ]; then
    echo "    downloading $f.jsonl.gz"
    curl -fsSL -o "$DATA_DIR/$f.jsonl.gz" \
      "https://raw.githubusercontent.com/openai/human-eval-infilling/main/data/$f.jsonl.gz"
  fi
done
ls -la "$DATA_DIR"

MODEL_PATH=${MODEL_PATH:-fredzzp/open-dcoder-0.5B}
TEMPERATURE=${TEMPERATURE:-0.6}
STEPS=${STEPS:-64}
ALG=${ALG:-p2}
BATCH_SIZE=${BATCH_SIZE:-16}

export CUDA_VISIBLE_DEVICES=0

cd "$ROOT/Open-dLLM/eval/eval_infill"

echo "==> HumanEval-Infill Pass@1 (1× GPU); logs -> $RUN_DIR/eval.log"
{
  echo "host: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  python -c "import torch; print('torch', torch.__version__, 'cap', torch.cuda.get_device_capability(0))"
  echo "model: $MODEL_PATH  steps=$STEPS  temp=$TEMPERATURE  alg=$ALG  bs=$BATCH_SIZE"
  echo "----"
  python eval_infill.py \
    --model_path "$MODEL_PATH" \
    --task humaneval_infill \
    --temperature "$TEMPERATURE" \
    --steps "$STEPS" \
    --alg "$ALG" \
    --batch_size "$BATCH_SIZE" \
    --no_wandb
} 2>&1 | tee "$RUN_DIR/eval.log"

echo "==> Done. Tail:"
tail -40 "$RUN_DIR/eval.log"

echo "==> Predictions saved by upstream under: $ROOT/Open-dLLM/eval/eval_infill/infill_results/"
