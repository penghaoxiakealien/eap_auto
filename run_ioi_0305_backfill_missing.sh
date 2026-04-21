#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/home/wangziran/eap_auto/results/ioi_0305}"
STANDARD_FILE="${STANDARD_FILE:-${RESULTS_ROOT}/path_patching/standard_ioi_data.json}"
MAX_JOBS="${MAX_JOBS:-3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

RUN_NMH="${REPO_ROOT}/tests/experiments/run_NMH_val_size_x2.sh"
RUN_MIDDLE="${REPO_ROOT}/tests/experiments/run_middle_head_val_size_x2.sh"
RUN_MIDDLE_PLUS="${REPO_ROOT}/tests/experiments/run_middle_head_plus_val_size_x2.sh"

LOG_DIR="${RESULTS_ROOT}/logs/backfill_${STAMP}"
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

echo "Using RESULTS_ROOT=${RESULTS_ROOT}"
echo "Using STANDARD_FILE=${STANDARD_FILE}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using MAX_JOBS=${MAX_JOBS}"
echo "Logs: ${LOG_DIR}"

if [[ ! -f "$STANDARD_FILE" ]]; then
  echo "ERROR: missing standard file: $STANDARD_FILE" >&2
  exit 1
fi

NMH_TARGET_HEADS=("9.6" "9.9" "10.0")
SIH_HEADS=("7.9" "8.6" "8.10")

NMH_DESC="$(build_desc_from_latest_all "Name_Mover_Head" "${NMH_TARGET_HEADS[@]}")"
SIH_DESC="$(build_desc_from_latest_all "SIH" "${SIH_HEADS[@]}")"
DTH_DESC="${SIH_DESC},${NMH_DESC}"

run_bg() {
  local name="$1"; shift
  local log_file="${LOG_DIR}/${name}.log"
  echo "[start] ${name}"
  (
    "$@"
  ) >"$log_file" 2>&1 &
}

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]]; do
    sleep 2
  done
}

# ---------------------------
# Missing NNMH tasks
# ---------------------------
wait_for_slot
run_bg "NNMH_10.7_att" \
  bash "$RUN_NMH" \
    --layer 10 --head 7 \
    --task-prefix NNMH \
    --output-family Negative_Name_Mover_Head \
    --results-root "$RESULTS_ROOT" \
    --optimize-only attention \
    --output-dir "${RESULTS_ROOT}/hypothesis/Negative_Name_Mover_Head/10.7_${STAMP}_att"

wait_for_slot
run_bg "NNMH_11.10_causal" \
  bash "$RUN_NMH" \
    --layer 11 --head 10 \
    --task-prefix NNMH \
    --output-family Negative_Name_Mover_Head \
    --results-root "$RESULTS_ROOT" \
    --optimize-only causal \
    --output-dir "${RESULTS_ROOT}/hypothesis/Negative_Name_Mover_Head/11.10_${STAMP}_causal"

wait_for_slot
run_bg "NNMH_11.10_att" \
  bash "$RUN_NMH" \
    --layer 11 --head 10 \
    --task-prefix NNMH \
    --output-family Negative_Name_Mover_Head \
    --results-root "$RESULTS_ROOT" \
    --optimize-only attention \
    --output-dir "${RESULTS_ROOT}/hypothesis/Negative_Name_Mover_Head/11.10_${STAMP}_att"

# ---------------------------
# Missing SIH tasks
# ---------------------------
wait_for_slot
run_bg "SIH_8.6_att" \
  bash "$RUN_MIDDLE" \
    --layer 8 --head 6 \
    --receiver-heads "9.6,9.9,10.0" \
    --receiver-desc "$NMH_DESC" \
    --standard-file "$STANDARD_FILE" \
    --results-root "$RESULTS_ROOT" \
    --optimize-only attention \
    --output-dir "${RESULTS_ROOT}/hypothesis/SIH/8.6_${STAMP}_att"

wait_for_slot
run_bg "SIH_8.10_att" \
  bash "$RUN_MIDDLE" \
    --layer 8 --head 10 \
    --receiver-heads "9.6,9.9,10.0" \
    --receiver-desc "$NMH_DESC" \
    --standard-file "$STANDARD_FILE" \
    --results-root "$RESULTS_ROOT" \
    --optimize-only attention \
    --output-dir "${RESULTS_ROOT}/hypothesis/SIH/8.10_${STAMP}_att"

# ---------------------------
# Missing DTH tasks
# ---------------------------
run_dth_mode() {
  local head="$1"
  local opt="$2"
  local mode_label="$3"
  IFS='.' read -r layer_idx head_idx <<< "$head"
  bash "$RUN_MIDDLE_PLUS" \
    --layer "$layer_idx" \
    --head "$head_idx" \
    --intermediate-heads "7.9,8.6,8.10" \
    --target-heads "9.6,9.9,10.0" \
    --receiver-desc "$DTH_DESC" \
    --attention-position s2 \
    --standard-file "$STANDARD_FILE" \
    --results-root "$RESULTS_ROOT" \
    --optimize-only "$opt" \
    --output-dir "${RESULTS_ROOT}/hypothesis/DTH/${head}_${STAMP}_${mode_label}"
}

for task in \
  "0.1 causal causal" \
  "0.1 attention att" \
  "0.10 causal causal" \
  "0.10 attention att" \
  "3.0 causal causal" \
  "3.0 attention att"
do
  read -r h opt m <<< "$task"
  wait_for_slot
  run_bg "DTH_${h}_${m}" run_dth_mode "$h" "$opt" "$m"
done

echo "Waiting for all backfill jobs..."
FAILED=0
for pid in $(jobs -rp); do
  if ! wait "$pid"; then
    FAILED=1
  fi
done

if [[ "$FAILED" -eq 1 ]]; then
  echo "⚠️ Some backfill jobs failed. Check logs in ${LOG_DIR}"
  exit 1
fi

echo "✅ Backfill complete. Logs in ${LOG_DIR}"
