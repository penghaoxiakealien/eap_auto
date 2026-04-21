#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/home/wangziran/eap_auto/results/ioi_0301}"
COMPARE_PY="${REPO_ROOT}/tests/experiments/compare_hypotheses_to_references.py"
REFERENCE_ROOT="${REFERENCE_ROOT:-${RESULTS_ROOT}/answer}"

MODEL="${MODEL:-claude-sonnet-4-20250514-thinking}"
API_KEY="${OPENROUTER_API_KEY:-}"

if [[ -z "$API_KEY" ]]; then
  # 与 auto_NMH.py / auto_s_att.py 默认保持一致
  API_KEY="sk-99F0IFe53pSHOPQ3phWbEAEx86ZDOqkE58Ov9aYCS9AOQ2C7"
fi

# ioi_0301 通常没有 answer 目录，默认回退到 ioi_0126 的标准答案目录。
if [[ ! -d "$REFERENCE_ROOT" ]]; then
  FALLBACK_REF="/home/wangziran/eap_auto/results/ioi_0126/answer"
  if [[ -d "$FALLBACK_REF" ]]; then
    REFERENCE_ROOT="$FALLBACK_REF"
  fi
fi

latest_test_for() {
  local family="$1"
  local head="$2"
  local mode="$3"
  local pattern="${RESULTS_ROOT}/hypothesis/${family}/${head}_*_${mode}/test_results.json"
  ls -1t $pattern 2>/dev/null | head -n 1 || true
}

run_pair_compare() {
  local family="$1"
  local label="$2"
  local ref_file="$3"
  local head="$4"
  local mode="$5"

  local all_path causal_or_att_path out_path
  all_path="$(latest_test_for "$family" "$head" "all")"
  causal_or_att_path="$(latest_test_for "$family" "$head" "$mode")"

  if [[ -z "$all_path" || -z "$causal_or_att_path" ]]; then
    echo "[skip] ${family} ${head} ${mode}: missing all/mode test_results.json"
    return 0
  fi

  out_path="$(dirname "$all_path")/compare_${mode}_vs_all.json"
  echo "[run] ${family} ${head}: ${mode} vs all"
  python "$COMPARE_PY" \
    --model "$MODEL" \
    --api-key "$API_KEY" \
    --pair-reference-path "${REFERENCE_ROOT}/${ref_file}" \
    --pair-label "${label}_${head}_${mode}_vs_all" \
    --candidate-a "$causal_or_att_path" \
    --candidate-b "$all_path" \
    --output "$out_path"
}

echo "=== Compare causal/att vs all under ${RESULTS_ROOT} ==="
echo "=== Reference answers from ${REFERENCE_ROOT} ==="

# NMH
for head in 9.6 9.9 10.0; do
  run_pair_compare "Name_Mover_Head" "NMH" "NMH.txt" "$head" "causal"
  run_pair_compare "Name_Mover_Head" "NMH" "NMH.txt" "$head" "att"
done

# NNMH
for head in 10.7 11.10; do
  run_pair_compare "Negative_Name_Mover_Head" "NNMH" "NNMH.txt" "$head" "causal"
  run_pair_compare "Negative_Name_Mover_Head" "NNMH" "NNMH.txt" "$head" "att"
done

# SIH
for head in 7.9 8.6 8.10; do
  run_pair_compare "SIH" "SIH" "SIH.txt" "$head" "causal"
  run_pair_compare "SIH" "SIH" "SIH.txt" "$head" "att"
done

# DTH
for head in 0.1 0.10 3.0; do
  run_pair_compare "DTH" "DTH" "DTH.txt" "$head" "causal"
  run_pair_compare "DTH" "DTH" "DTH.txt" "$head" "att"
done

echo "✅ Done."
