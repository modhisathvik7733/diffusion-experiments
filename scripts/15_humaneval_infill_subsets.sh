#!/usr/bin/env bash
# Sweep DreamOn-v0-7B HumanEval-Infill across all 3 subsets:
#   - single-line  (already at 88.67 — should reproduce)
#   - multi-line   (harder — multi-line holes)
#   - random-span  (hardest — arbitrary spans)
# Patches DreamOn/eval/evaluate.py once to read benchmark_name from cfg dotlist,
# then runs eval per subset, restores on exit.
set -euo pipefail

ROOT=${ROOT:-/workspace/diffusion-llm}
EXPDIR="$ROOT/diffusion-experiments"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_humaneval_infill_subsets"
RUN_DIR="$EXPDIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

EVAL_PY="$ROOT/DreamOn/eval/evaluate.py"
BACKUP="$EVAL_PY.subsets.backup"

if [ ! -f "$EVAL_PY" ]; then
  echo "ERROR: DreamOn not cloned. Run scripts/05 first."
  exit 1
fi

if [ -f "$BACKUP" ]; then
  echo "==> Stale backup detected — restoring before patching"
  mv "$BACKUP" "$EVAL_PY"
fi
cp "$EVAL_PY" "$BACKUP"
trap 'mv "$BACKUP" "$EVAL_PY" 2>/dev/null && echo "==> Restored evaluate.py" || true' EXIT INT TERM

echo "==> Patching evaluate.py to read benchmark_name from cfg"
python - <<PYEOF
path = "$EVAL_PY"
with open(path) as f:
    code = f.read()
old = "problems = read_problems(benchmark_name='single-line')"
new = "problems = read_problems(benchmark_name=getattr(cfg, 'benchmark_name', 'single-line'))"
if old not in code:
    raise SystemExit("ERROR: target line not found — upstream code may have changed")
code = code.replace(old, new)
with open(path, "w") as f:
    f.write(code)
print("  Patched: benchmark_name now read from cfg.benchmark_name")
PYEOF

python -c "import ast; ast.parse(open('$EVAL_PY').read())"
echo "==> Patched evaluate.py parses OK"

CKPT=Dream-org/DreamOn-v0-7B
CSV="$RUN_DIR/results.csv"
echo "subset,pass_at_1,wall_seconds" > "$CSV"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$ROOT/DreamOn"

for SUBSET in single-line multi-line random-span; do
  echo ""
  echo "############################################"
  echo "## SUBSET=$SUBSET"
  echo "############################################"
  STAGE_LOG="$RUN_DIR/eval_${SUBSET}.log"
  OUTPUT_DIR="$ROOT/DreamOn/results/Dream-org/DreamOn-v0-7B"
  mkdir -p "$OUTPUT_DIR"
  RESULT_FILE="$OUTPUT_DIR/humaneval_infill_${SUBSET}.jsonl"

  start=$(date +%s)
  {
    torchrun --nproc_per_node 1 --master_port 29515 \
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
        task=humaneval_infill \
        benchmark_name=$SUBSET

    echo "==> Scoring $SUBSET"
    evaluate_infilling_functional_correctness "$SUBSET" "$RESULT_FILE"
  } 2>&1 | tee "$STAGE_LOG" || true
  end=$(date +%s)
  duration=$((end - start))

  pass1=$(grep -oE "'pass@1': np\.float64\([0-9]+\.[0-9]+\)" "$STAGE_LOG" \
          | grep -oE '[0-9]+\.[0-9]+' \
          | head -n 1)
  pass1=${pass1:-ERROR}

  echo "$SUBSET,$pass1,$duration" >> "$CSV"
  echo "  -> $SUBSET  pass@1=$pass1  wall=${duration}s"
done

echo ""
echo "############################################"
echo "## SUBSETS SWEEP COMPLETE"
echo "############################################"
column -t -s, "$CSV"
echo ""
echo "Reference: single-line was 88.67 in earlier vanilla run."
