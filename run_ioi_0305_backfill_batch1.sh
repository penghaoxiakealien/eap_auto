#!/bin/bash
set -euo pipefail

# Batch 1: NNMH missing tasks

REPO_ROOT="/home/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/home/wangziran/eap_auto/results/ioi_0305}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
RUN_NMH="${REPO_ROOT}/tests/experiments/run_NMH_val_size_x2.sh"

LOG_DIR="${RESULTS_ROOT}/logs/backfill_manual_${STAMP}"
mkdir -p "$LOG_DIR"

run_one() {
  local name="$1"; shift
  local log_file="${LOG_DIR}/${name}.log"
  echo "[start] ${name}"
  (
    "$@"
  ) >"$log_file" 2>&1
  echo "[done] ${name}"
}

run_one "NNMH_10.7_att" \
  bash "$RUN_NMH" \
    --layer 10 --head 7 \
    --task-prefix NNMH \
    --output-family Negative_Name_Mover_Head \
    --results-root "$RESULTS_ROOT" \
    --optimize-only attention \
    --output-dir "${RESULTS_ROOT}/hypothesis/Negative_Name_Mover_Head/10.7_${STAMP}_att"

run_one "NNMH_11.10_causal" \
  bash "$RUN_NMH" \
    --layer 11 --head 10 \
    --task-prefix NNMH \
    --output-family Negative_Name_Mover_Head \
    --results-root "$RESULTS_ROOT" \
    --optimize-only causal \
    --output-dir "${RESULTS_ROOT}/hypothesis/Negative_Name_Mover_Head/11.10_${STAMP}_causal"

run_one "NNMH_11.10_att" \
  bash "$RUN_NMH" \
    --layer 11 --head 10 \
    --task-prefix NNMH \
    --output-family Negative_Name_Mover_Head \
    --results-root "$RESULTS_ROOT" \
    --optimize-only attention \
    --output-dir "${RESULTS_ROOT}/hypothesis/Negative_Name_Mover_Head/11.10_${STAMP}_att"

echo "✅ Batch1 done. Logs: ${LOG_DIR}"
