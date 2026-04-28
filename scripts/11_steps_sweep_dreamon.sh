#!/usr/bin/env bash
# Step-count sweep on DreamOn-v0-7B HumanEval-Infill.
# Determines the lowest diffusion step count that preserves Pass@1 quality.
# Vanilla baseline is steps=256, Pass@1=0.8867 (88.67).
#
# Strategy: run scripts/05 with STEPS in {32, 64, 128}, parse Pass@1 + wall time
# from each run, print a comparison table. Total runtime ~25-35 min.
set -euo pipefail

ROOT=${ROOT:-/workspace/diffusion-llm}
EXPDIR="$ROOT/diffusion-experiments"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_steps_sweep"
RUN_DIR="$EXPDIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

# Override via: STEPS_LIST="16 32 64 128" bash scripts/11_steps_sweep_dreamon.sh
STEPS_LIST=${STEPS_LIST:-"32 64 128"}

CSV="$RUN_DIR/results.csv"
echo "steps,pass_at_1,wall_seconds" > "$CSV"

for S in $STEPS_LIST; do
  echo ""
  echo "############################################"
  echo "## STEPS=$S  (run -> $RUN_DIR/eval_steps${S}.log)"
  echo "############################################"
  STAGE_LOG="$RUN_DIR/eval_steps${S}.log"

  start=$(date +%s)
  # script 05 honors STEPS env var; let it run, capture log
  STEPS=$S bash "$EXPDIR/scripts/05_baseline_humaneval_infill_dreamon.sh" 2>&1 | tee "$STAGE_LOG" || true
  end=$(date +%s)
  duration=$((end - start))

  # Parse the upstream scorer's output: '{'pass@1': np.float64(0.8867...)'
  pass1=$(grep -oE "'pass@1': np\.float64\([0-9]+\.[0-9]+\)" "$STAGE_LOG" \
          | grep -oE '[0-9]+\.[0-9]+' \
          | head -n 1)
  if [ -z "$pass1" ]; then
    pass1="ERROR"
  fi

  echo "$S,$pass1,$duration" >> "$CSV"
  echo ""
  echo "  -> steps=$S  pass@1=$pass1  wall=${duration}s"
done

echo ""
echo "############################################"
echo "## SWEEP COMPLETE"
echo "############################################"
echo ""
echo "Results CSV: $CSV"
echo ""
column -t -s, "$CSV"
echo ""
echo "Reference (vanilla, steps=256):  pass@1 = 0.8867,  wall ~ 545s"
echo ""
echo "Decision rule:"
echo "  - Pass@1 stays ≥ 0.85  -> safe to ship at this step count"
echo "  - Pass@1 drops to 0.80-0.85 -> usable but consider distillation later"
echo "  - Pass@1 drops below 0.80  -> too lossy, use more steps"
