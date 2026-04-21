#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/home/wangziran/eap_auto/results/ioi_0312}"
MODEL="${MODEL:-gpt-5.2-2025-12-11}"
CONDA_ENV="${CONDA_ENV:-eap-ig}"

run_nmh() {
  local layer="$1"
  local head="$2"
  local mode="$3"
  local family="$4"
  local task_prefix="$5"
  local tag="${layer}.${head}_${mode}"
  bash "$REPO_ROOT/tests/experiments/run_NMH.sh" \
    --layer "$layer" \
    --head "$head" \
    --results-root "$RESULTS_ROOT" \
    --conda-env "$CONDA_ENV" \
    --task-prefix "$task_prefix" \
    --output-family "$family" \
    --optimize-only "$mode" \
    --validate-every 2 \
    --validation-sample-size 25 \
    --test-sample-size 50 \
    --model "$MODEL" \
    --output-dir "$RESULTS_ROOT/hypothesis/${family}/${tag}_$(date +%Y%m%d_%H%M)"
}

for mode in causal attention; do
  run_nmh 9 6 "$mode" "Name_Mover_Head" "NMH"
  run_nmh 9 9 "$mode" "Name_Mover_Head" "NMH"
  run_nmh 10 0 "$mode" "Name_Mover_Head" "NMH"
  run_nmh 10 7 "$mode" "Negative_Name_Mover_Head" "NNMH"
  run_nmh 11 10 "$mode" "Negative_Name_Mover_Head" "NNMH"
done

echo "✅ IOI 0312 NMH/NNMH single-task batch finished."
