#!/bin/bash
set -euo pipefail

REPO_ROOT="/data31/private/wangziran/eap_auto"
RESULTS_ROOT="/data31/private/wangziran/eap_auto/results/ioi_0209"

TS="$(date +%Y%m%d_%H%M)"

run_one() {
  local variant_script="$1"
  local layer="$2"
  local head="$3"
  local family="$4"
  local task_prefix="$5"
  local repeat_idx="$6"
  local variant_tag="$7"

  local out_dir="${RESULTS_ROOT}/hypothesis/${family}/${layer}.${head}_${TS}_all_${variant_tag}_r${repeat_idx}"
  echo "=== Running ${layer}.${head} (${variant_tag}, repeat ${repeat_idx}) ==="
  echo "Output: ${out_dir}"

  bash "${REPO_ROOT}/tests/experiments/${variant_script}" \
    --layer "${layer}" \
    --head "${head}" \
    --task-prefix "${task_prefix}" \
    --output-family "${family}" \
    --results-root "${RESULTS_ROOT}" \
    --optimize-only dual \
    --output-dir "${out_dir}"
}

for r in 1 2; do
  # variant A: every-epoch validation
  run_one "run_NMH_val_every_epoch.sh" 9 9 "Name_Mover_Head" "NMH" "${r}" "val_every_epoch"
  run_one "run_NMH_val_every_epoch.sh" 10 7 "Negative_Name_Mover_Head" "NNMH" "${r}" "val_every_epoch"

  # variant B: doubled validation sample size
  run_one "run_NMH_val_size_x2.sh" 9 9 "Name_Mover_Head" "NMH" "${r}" "val_size_x2"
  run_one "run_NMH_val_size_x2.sh" 10 7 "Negative_Name_Mover_Head" "NNMH" "${r}" "val_size_x2"
done

echo "✅ All runs finished."

