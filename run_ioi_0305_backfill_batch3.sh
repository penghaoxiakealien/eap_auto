#!/bin/bash
set -euo pipefail

# Batch 3: DTH(0.10 / 3.0) missing tasks

REPO_ROOT="/data31/private/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/data31/private/wangziran/eap_auto/results/ioi_0305}"
STANDARD_FILE="${STANDARD_FILE:-${RESULTS_ROOT}/path_patching/standard_ioi_data.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

RUN_MIDDLE_PLUS="${REPO_ROOT}/tests/experiments/run_middle_head_plus_val_size_x2.sh"

LOG_DIR="${RESULTS_ROOT}/logs/backfill_manual_${STAMP}"
mkdir -p "$LOG_DIR"

latest_all_best() {
  local family="$1"
  local head="$2"
  local pattern="${RESULTS_ROOT}/hypothesis/${family}/${head}_*_all/best_hypothesis.json"
  ls -1t $pattern 2>/dev/null | head -n 1 || true
}

build_desc_from_latest_all() {
  local family="$1"; shift
  local desc=""
  for h in "$@"; do
    local p
    p="$(latest_all_best "$family" "$h")"
    if [[ -z "$p" ]]; then
      echo "ERROR: missing latest _all best_hypothesis for ${family} ${h}" >&2
      return 1
    fi
    [[ -n "$desc" ]] && desc+=","
    desc+="${h}:${p}"
  done
  echo "$desc"
}

run_one() {
  local name="$1"; shift
  local log_file="${LOG_DIR}/${name}.log"
  echo "[start] ${name}"
  (
    "$@"
  ) >"$log_file" 2>&1
  echo "[done] ${name}"
}

NMH_DESC="$(build_desc_from_latest_all "Name_Mover_Head" 9.6 9.9 10.0)"
SIH_DESC="$(build_desc_from_latest_all "SIH" 7.9 8.6 8.10)"
DTH_DESC="${SIH_DESC},${NMH_DESC}"

run_dth() {
  local head="$1"
  local mode="$2"
  local opt="$3"
  local layer head_idx
  IFS='.' read -r layer head_idx <<< "$head"
  bash "$RUN_MIDDLE_PLUS" \
    --layer "$layer" \
    --head "$head_idx" \
    --intermediate-heads "7.9,8.6,8.10" \
    --target-heads "9.6,9.9,10.0" \
    --receiver-desc "$DTH_DESC" \
    --attention-position s2 \
    --standard-file "$STANDARD_FILE" \
    --results-root "$RESULTS_ROOT" \
    --optimize-only "$opt" \
    --output-dir "${RESULTS_ROOT}/hypothesis/DTH/${head}_${STAMP}_${mode}"
}

run_one "DTH_0.10_causal" run_dth "0.10" "causal" "causal"
run_one "DTH_0.10_att" run_dth "0.10" "att" "attention"
run_one "DTH_3.0_causal" run_dth "3.0" "causal" "causal"
run_one "DTH_3.0_att" run_dth "3.0" "att" "attention"

echo "✅ Batch3 done. Logs: ${LOG_DIR}"
