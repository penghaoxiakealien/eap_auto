#!/bin/bash
set -euo pipefail

# 依赖：NMH 9.6/9.9/10.0 的 _all 已先跑完。
# 调度策略：
# 1) 先跑 SIH _all（用于给 DTH 提供 receiver descriptions）
# 2) 然后并行跑：
#    - SIH _causal + _att
#    - DTH _all + _causal + _att

REPO_ROOT="/data31/private/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/data31/private/wangziran/eap_auto/results/ioi_0301}"
RUN_MIDDLE="${REPO_ROOT}/tests/experiments/run_middle_head_val_size_x2.sh"
RUN_MIDDLE_PLUS="${REPO_ROOT}/tests/experiments/run_middle_head_plus_val_size_x2.sh"
STANDARD_FILE="${STANDARD_FILE:-${RESULTS_ROOT}/path_patching/standard_ioi_data.json}"
STAMP="$(date +%Y%m%d_%H%M)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

SIH_HEADS=("7.9" "8.6" "8.10")
NMH_TARGET_HEADS=("9.6" "9.9" "10.0")
DTH_HEADS=("0.1" "0.10" "3.0")

latest_all_best() {
  local family="$1"
  local head="$2"
  local pattern="${RESULTS_ROOT}/hypothesis/${family}/${head}_*_all/best_hypothesis.json"
  ls -1t $pattern 2>/dev/null | head -n 1 || true
}

build_desc_from_latest_all() {
  local family="$1"
  shift
  local desc=""
  for h in "$@"; do
    local p
    p="$(latest_all_best "$family" "$h")"
    if [[ -z "$p" ]]; then
      echo "ERROR: missing latest _all best_hypothesis for ${family} ${h}" >&2
      return 1
    fi
    if [[ -n "$desc" ]]; then
      desc+=","
    fi
    desc+="${h}:${p}"
  done
  echo "$desc"
}

run_sih_mode() {
  local mode_label="$1"
  local opt="$2"
  local nmh_desc="$3"
  for h in "${SIH_HEADS[@]}"; do
    IFS='.' read -r layer_idx head_idx <<< "$h"
    local out_dir="${RESULTS_ROOT}/hypothesis/SIH/${h}_${STAMP}_${mode_label}"
    echo "=== SIH ${h} ${mode_label} ==="
    bash "$RUN_MIDDLE" \
      --layer "$layer_idx" \
      --head "$head_idx" \
      --receiver-heads "$(IFS=,; echo "${NMH_TARGET_HEADS[*]}")" \
      --receiver-desc "$nmh_desc" \
      --standard-file "$STANDARD_FILE" \
      --results-root "$RESULTS_ROOT" \
      --optimize-only "$opt" \
      --output-dir "$out_dir"
  done
}

run_dth_modes() {
  local nmh_desc="$1"
  local sih_desc="$2"
  local dth_desc="${sih_desc},${nmh_desc}"

  local modes=("all:dual" "causal:causal" "att:attention")
  for mode in "${modes[@]}"; do
    local mode_label="${mode%%:*}"
    local opt="${mode##*:}"
    for h in "${DTH_HEADS[@]}"; do
      IFS='.' read -r layer_idx head_idx <<< "$h"
      local out_dir="${RESULTS_ROOT}/hypothesis/DTH/${h}_${STAMP}_${mode_label}"
      echo "=== DTH ${h} ${mode_label} ==="
      bash "$RUN_MIDDLE_PLUS" \
        --layer "$layer_idx" \
        --head "$head_idx" \
        --intermediate-heads "$(IFS=,; echo "${SIH_HEADS[*]}")" \
        --target-heads "$(IFS=,; echo "${NMH_TARGET_HEADS[*]}")" \
        --receiver-desc "$dth_desc" \
        --attention-position s2 \
        --standard-file "$STANDARD_FILE" \
        --results-root "$RESULTS_ROOT" \
        --optimize-only "$opt" \
        --output-dir "$out_dir"
    done
  done
}

echo "=== Step 0: load NMH _all descriptions (must exist) ==="
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
NMH_DESC="$(build_desc_from_latest_all "Name_Mover_Head" "${NMH_TARGET_HEADS[@]}")"

echo "=== Step 1: run SIH _all first (dependency for DTH) ==="
run_sih_mode "all" "dual" "$NMH_DESC"

echo "=== Step 2: build SIH _all descriptions ==="
SIH_DESC="$(build_desc_from_latest_all "SIH" "${SIH_HEADS[@]}")"

echo "=== Step 3: parallel launch: [SIH causal+att] || [DTH all+causal+att] ==="
(
  run_sih_mode "causal" "causal" "$NMH_DESC"
  run_sih_mode "att" "attention" "$NMH_DESC"
) &
PID_SIH=$!

(
  run_dth_modes "$NMH_DESC" "$SIH_DESC"
) &
PID_DTH=$!

wait "$PID_SIH"
wait "$PID_DTH"

echo "✅ SIH/DTH finished (DTH heads: 0.1, 0.10, 3.0)."
