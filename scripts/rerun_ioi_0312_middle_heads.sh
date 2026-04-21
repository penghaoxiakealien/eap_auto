#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/home/wangziran/eap_auto/results/ioi_0312}"
MODEL="${MODEL:-gpt-5-chat}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"

RECEIVER_HEADS="9.6,9.9,10.0"
RECEIVER_DESC="9.6:${RESULTS_ROOT}/hypothesis/Name_Mover_Head/9.6_20260315_0536/best_hypothesis.json,9.9:${RESULTS_ROOT}/hypothesis/Name_Mover_Head/9.9_20260315_0726/best_hypothesis.json,10.0:${RESULTS_ROOT}/hypothesis/Name_Mover_Head/10.0_20260315_0752/best_hypothesis.json"

RUN_SCRIPT="${REPO_ROOT}/tests/experiments/run_middle_head_val_size_x2.sh"

run_one() {
  local layer="$1"
  local head="$2"
  local tag="${layer}.${head}_rerun_${STAMP}"

  bash "${RUN_SCRIPT}" \
    --layer "${layer}" \
    --head "${head}" \
    --receiver-heads "${RECEIVER_HEADS}" \
    --receiver-desc "${RECEIVER_DESC}" \
    --results-root "${RESULTS_ROOT}" \
    --output-dir "${RESULTS_ROOT}/hypothesis/Middle_Head/${tag}" \
    --optimize-only dual \
    --model "${MODEL}"
}

run_one 7 9
run_one 8 6
run_one 8 10

