#!/bin/bash
set -euo pipefail

# 重跑 ioi_0305 下所有 _all 任务（新逻辑）:
# NMH(_all) -> SIH(_all) -> DTH(_all)，并包含 NNMH(_all)。

REPO_ROOT="/home/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/home/wangziran/eap_auto/results/ioi_0307}"
STANDARD_FILE="${STANDARD_FILE:-${RESULTS_ROOT}/path_patching/standard_ioi_data.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

RUN_NMH="${REPO_ROOT}/tests/experiments/run_NMH_val_size_x2.sh"
RUN_MIDDLE="${REPO_ROOT}/tests/experiments/run_middle_head_val_size_x2.sh"
RUN_MIDDLE_PLUS="${REPO_ROOT}/tests/experiments/run_middle_head_plus_val_size_x2.sh"

NMH_HEADS=("9.6" "9.9" "10.0")
NNMH_HEADS=("10.7" "11.10")
SIH_HEADS=("7.9" "8.6" "8.10")
DTH_HEADS=("0.1" "0.10" "3.0")

join_by_comma() {
  local IFS=","
  echo "$*"
}

build_desc_for_stamp() {
  local family="$1"
  shift
  local desc=""
  for h in "$@"; do
    local p="${RESULTS_ROOT}/hypothesis/${family}/${h}_${STAMP}_all/best_hypothesis.json"
    if [[ ! -f "$p" ]]; then
      echo "ERROR: missing best_hypothesis for ${family} ${h}: $p" >&2
      return 1
    fi
    [[ -n "$desc" ]] && desc+=","
    desc+="${h}:${p}"
  done
  echo "$desc"
}

run_nmh_family_all() {
  local family="$1"
  local task_prefix="$2"
  shift 2
  local heads=("$@")
  for h in "${heads[@]}"; do
    IFS='.' read -r l hidx <<< "$h"
    local out_dir="${RESULTS_ROOT}/hypothesis/${family}/${h}_${STAMP}_all"
    echo "=== ${family} ${h} _all ==="
    bash "$RUN_NMH" \
      --layer "$l" \
      --head "$hidx" \
      --task-prefix "$task_prefix" \
      --output-family "$family" \
      --results-root "$RESULTS_ROOT" \
      --optimize-only dual \
      --output-dir "$out_dir"
  done
}

run_sih_all() {
  local nmh_desc="$1"
  for h in "${SIH_HEADS[@]}"; do
    IFS='.' read -r l hidx <<< "$h"
    local out_dir="${RESULTS_ROOT}/hypothesis/SIH/${h}_${STAMP}_all"
    echo "=== SIH ${h} _all ==="
    bash "$RUN_MIDDLE" \
      --layer "$l" \
      --head "$hidx" \
      --receiver-heads "$(join_by_comma "${NMH_HEADS[@]}")" \
      --receiver-desc "$nmh_desc" \
      --standard-file "$STANDARD_FILE" \
      --results-root "$RESULTS_ROOT" \
      --optimize-only dual \
      --output-dir "$out_dir"
  done
}

run_dth_all() {
  local desc="$1"
  for h in "${DTH_HEADS[@]}"; do
    IFS='.' read -r l hidx <<< "$h"
    local out_dir="${RESULTS_ROOT}/hypothesis/DTH/${h}_${STAMP}_all"
    echo "=== DTH ${h} _all ==="
    bash "$RUN_MIDDLE_PLUS" \
      --layer "$l" \
      --head "$hidx" \
      --intermediate-heads "$(join_by_comma "${SIH_HEADS[@]}")" \
      --target-heads "$(join_by_comma "${NMH_HEADS[@]}")" \
      --receiver-desc "$desc" \
      --attention-position s2 \
      --standard-file "$STANDARD_FILE" \
      --results-root "$RESULTS_ROOT" \
      --optimize-only dual \
      --output-dir "$out_dir"
  done
}

echo "=== ioi_0305 rerun all ==="
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "STANDARD_FILE=${STANDARD_FILE}"
echo "STAMP=${STAMP}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "=== Step 1/4: NMH _all ==="
run_nmh_family_all "Name_Mover_Head" "NMH" "${NMH_HEADS[@]}"

echo "=== Step 2/4: NNMH _all ==="
run_nmh_family_all "Negative_Name_Mover_Head" "NNMH" "${NNMH_HEADS[@]}"

echo "=== Step 3/4: SIH _all ==="
NMH_DESC="$(build_desc_for_stamp "Name_Mover_Head" "${NMH_HEADS[@]}")"
run_sih_all "$NMH_DESC"

echo "=== Step 4/4: DTH _all ==="
SIH_DESC="$(build_desc_for_stamp "SIH" "${SIH_HEADS[@]}")"
DTH_DESC="${SIH_DESC},${NMH_DESC}"
run_dth_all "$DTH_DESC"

echo "✅ Done. All _all jobs finished. stamp=${STAMP}"
