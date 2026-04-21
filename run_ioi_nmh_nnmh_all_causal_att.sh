#!/bin/bash
set -euo pipefail

# 分阶段运行：
# Phase A: 先跑 NMH 9.6/9.9/10.0 的 _all（dual）任务（供 SIH/DTH 依赖）。
# Phase B: 再跑 NMH/NNMH 的其余任务。

REPO_ROOT="/home/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/home/wangziran/eap_auto/results/ioi_0301}"
RUN_NMH="${REPO_ROOT}/tests/experiments/run_NMH_val_size_x2.sh"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

NMH_HEADS=("9.6" "9.9" "10.0")
NNMH_HEADS=("10.7" "11.10")

run_one() {
  local head="$1"
  local family="$2"
  local task_prefix="$3"
  local mode_label="$4"
  local opt="$5"
  IFS='.' read -r layer_idx head_idx <<< "$head"
  local out_dir="${RESULTS_ROOT}/hypothesis/${family}/${head}_${STAMP}_${mode_label}"
  echo "=== ${family} ${head} ${mode_label} ==="
  bash "$RUN_NMH" \
    --layer "$layer_idx" \
    --head "$head_idx" \
    --task-prefix "$task_prefix" \
    --output-family "$family" \
    --results-root "$RESULTS_ROOT" \
    --optimize-only "$opt" \
    --output-dir "$out_dir"
}

echo "=== Phase A: NMH _all first (dependency) ==="
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using RESULTS_ROOT=${RESULTS_ROOT}"
echo "Using STAMP=${STAMP}"
for h in "${NMH_HEADS[@]}"; do
  run_one "$h" "Name_Mover_Head" "NMH" "all" "dual"
done

echo "=== Phase B: remaining NMH/NNMH tasks ==="
# NMH 剩余两个模式
for mode in "causal:causal" "att:attention"; do
  mode_label="${mode%%:*}"
  opt="${mode##*:}"
  for h in "${NMH_HEADS[@]}"; do
    run_one "$h" "Name_Mover_Head" "NMH" "$mode_label" "$opt"
  done
done

# NNMH 全部三种模式
for mode in "all:dual" "causal:causal" "att:attention"; do
  mode_label="${mode%%:*}"
  opt="${mode##*:}"
  for h in "${NNMH_HEADS[@]}"; do
    run_one "$h" "Negative_Name_Mover_Head" "NNMH" "$mode_label" "$opt"
  done
done

echo "✅ Phase A+B finished. stamp=${STAMP}"
