#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/home/wangziran/eap_auto/results/ioi_0312}"
MODEL="${MODEL:-gpt-5.2-2025-12-11}"
CONDA_ENV="${CONDA_ENV:-eap-ig}"
CAUSAL_DIRECTION="${CAUSAL_DIRECTION:-decrease}"

SIH_RECEIVER_DESC="9.6:${RESULTS_ROOT}/hypothesis/Name_Mover_Head/9.6_20260315_0536/best_hypothesis.json,9.9:${RESULTS_ROOT}/hypothesis/Name_Mover_Head/9.9_20260315_0726/best_hypothesis.json,10.0:${RESULTS_ROOT}/hypothesis/Name_Mover_Head/10.0_20260315_0752/best_hypothesis.json"
DTH_RECEIVER_DESC="7.9:${RESULTS_ROOT}/hypothesis/Middle_Head/7.9_20260313_0152/best_hypothesis.json,8.6:${RESULTS_ROOT}/hypothesis/Middle_Head/8.6_20260313_1418/best_hypothesis.json,8.10:${RESULTS_ROOT}/hypothesis/Middle_Head/8.10_20260312_1631/best_hypothesis.json,9.6:${RESULTS_ROOT}/hypothesis/Name_Mover_Head/9.6_20260315_0536/best_hypothesis.json,9.9:${RESULTS_ROOT}/hypothesis/Name_Mover_Head/9.9_20260315_0726/best_hypothesis.json,10.0:${RESULTS_ROOT}/hypothesis/Name_Mover_Head/10.0_20260315_0752/best_hypothesis.json"

run_sih() {
  local layer="$1"
  local head="$2"
  local mode="$3"
  local tag="${layer}.${head}_${mode}_${CAUSAL_DIRECTION}"
  bash "$REPO_ROOT/tests/experiments/run_single_middle_head_val_size_x2.sh" \
    --layer "$layer" \
    --head "$head" \
    --receiver-heads "9.6,9.9,10.0" \
    --receiver-desc "$SIH_RECEIVER_DESC" \
    --results-root "$RESULTS_ROOT" \
    --causal-direction "$CAUSAL_DIRECTION" \
    --optimize-only "$mode" \
    --model "$MODEL" \
    --conda-env "$CONDA_ENV" \
    --output-dir "$RESULTS_ROOT/hypothesis/Middle_Head/${tag}_$(date +%Y%m%d_%H%M)"
}

run_dth() {
  local layer="$1"
  local head="$2"
  local mode="$3"
  local head_tag="${layer}.${head}"
  local tag="${head_tag}_to_9_6__9_9__10_0_${mode}_${CAUSAL_DIRECTION}"
  bash "$REPO_ROOT/tests/experiments/run_single_middle_head_plus_val_size_x2.sh" \
    --layer "$layer" \
    --head "$head" \
    --intermediate-heads "7.9,8.6,8.10" \
    --target-heads "9.6,9.9,10.0" \
    --receiver-desc "$DTH_RECEIVER_DESC" \
    --results-root "$RESULTS_ROOT" \
    --causal-direction "$CAUSAL_DIRECTION" \
    --optimize-only "$mode" \
    --model "$MODEL" \
    --conda-env "$CONDA_ENV" \
    --output-dir "$RESULTS_ROOT/hypothesis/Middle_Head_Plus/${tag}_$(date +%Y%m%d_%H%M)"
}

for mode in causal attention; do
  run_sih 7 9 "$mode"
  run_sih 8 6 "$mode"
  run_sih 8 10 "$mode"
done

for mode in causal attention; do
  run_dth 0 1 "$mode"
  run_dth 0 10 "$mode"
  run_dth 3 0 "$mode"
done

echo "✅ IOI 0312 single-task batch finished."
