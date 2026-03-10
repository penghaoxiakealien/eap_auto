#!/bin/bash
set -euo pipefail

# Batch 2: SIH missing + DTH(0.1) missing

REPO_ROOT="/data31/private/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/data31/private/wangziran/eap_auto/results/ioi_0305}"
STANDARD_FILE="${STANDARD_FILE:-${RESULTS_ROOT}/path_patching/standard_ioi_data.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

RUN_MIDDLE="${REPO_ROOT}/tests/experiments/run_middle_head_val_size_x2.sh"
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

run_one "SIH_8.6_att" \
  bash "$RUN_MIDDLE" \
    --layer 8 --head 6 \
    --receiver-heads "9.6,9.9,10.0" \
    --receiver-desc "$NMH_DESC" \
    --standard-file "$STANDARD_FILE" \
    --results-root "$RESULTS_ROOT" \
    --optimize-only attention \
    --output-dir "${RESULTS_ROOT}/hypothesis/SIH/8.6_${STAMP}_att"

run_one "SIH_8.10_att" \
  bash "$RUN_MIDDLE" \
    --layer 8 --head 10 \
    --receiver-heads "9.6,9.9,10.0" \
    --receiver-desc "$NMH_DESC" \
    --standard-file "$STANDARD_FILE" \
    --results-root "$RESULTS_ROOT" \
    --optimize-only attention \
    --output-dir "${RESULTS_ROOT}/hypothesis/SIH/8.10_${STAMP}_att"

run_one "DTH_0.1_causal" \
  bash "$RUN_MIDDLE_PLUS" \
    --layer 0 --head 1 \
    --intermediate-heads "7.9,8.6,8.10" \
    --target-heads "9.6,9.9,10.0" \
    --receiver-desc "$DTH_DESC" \
    --attention-position s2 \
    --standard-file "$STANDARD_FILE" \
    --results-root "$RESULTS_ROOT" \
    --optimize-only causal \
    --output-dir "${RESULTS_ROOT}/hypothesis/DTH/0.1_${STAMP}_causal"

run_one "DTH_0.1_att" \
  bash "$RUN_MIDDLE_PLUS" \
    --layer 0 --head 1 \
    --intermediate-heads "7.9,8.6,8.10" \
    --target-heads "9.6,9.9,10.0" \
    --receiver-desc "$DTH_DESC" \
    --attention-position s2 \
    --standard-file "$STANDARD_FILE" \
    --results-root "$RESULTS_ROOT" \
    --optimize-only attention \
    --output-dir "${RESULTS_ROOT}/hypothesis/DTH/0.1_${STAMP}_att"

echo "✅ Batch2 done. Logs: ${LOG_DIR}"
