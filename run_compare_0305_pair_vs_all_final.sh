#!/bin/bash
set -euo pipefail

REPO_ROOT="/data31/private/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/data31/private/wangziran/eap_auto/results/ioi_0305}"
REF_ROOT="${REF_ROOT:-${RESULTS_ROOT}/answer}"
COMPARE_PY="${REPO_ROOT}/tests/experiments/compare_hypotheses_to_references.py"
MODEL="${MODEL:-claude-sonnet-4-20250514-thinking}"
API_KEY="${OPENROUTER_API_KEY:-sk-99F0IFe53pSHOPQ3phWbEAEx86ZDOqkE58Ov9aYCS9AOQ2C7}"

latest_file () {
  local family="$1" head="$2" mode="$3" file="$4"
  local pattern="${RESULTS_ROOT}/hypothesis/${family}/${head}_*_${mode}/${file}"
  ls -1t $pattern 2>/dev/null | head -n 1 || true
}

run_pair() {
  local family="$1"
  local ref_file="$2"
  local head="$3"
  local a_tag="$4"
  local a_file="$5"
  local b_file="$6"
  local out_dir out_path
  out_dir="$(dirname "$b_file")"
  out_path="${out_dir}/pair_${head}_${a_tag}_vs_all_final.json"
  local skip_existing="${SKIP_EXISTING:-1}"

  if [[ "$skip_existing" == "1" && -f "$out_path" ]]; then
    echo "[skip-exists] ${family} ${head}: ${a_tag}"
    return 0
  fi

  echo "[run] ${family} ${head}: A=${a_tag} vs B=all_final"
  python "$COMPARE_PY" \
    --model "$MODEL" \
    --api-key "$API_KEY" \
    --pair-reference-path "${REF_ROOT}/${ref_file}" \
    --pair-label "${family}_${head}_${a_tag}_vs_all_final" \
    --candidate-a "$a_file" \
    --candidate-b "$b_file" \
    --output "$out_path"
}

run_family() {
  local family="$1"; shift
  local ref_file="$1"; shift
  local heads=("$@")
  for h in "${heads[@]}"; do
    local b_all_final
    b_all_final="$(latest_file "$family" "$h" "all" "test_results.json")"
    if [[ -z "$b_all_final" ]]; then
      echo "[skip] ${family} ${h}: missing B (_all final)"
      continue
    fi

    local a_all_initial a_causal_initial a_causal_final a_att_initial a_att_final
    a_all_initial="$(latest_file "$family" "$h" "all" "initial_test_results.json")"
    a_causal_initial="$(latest_file "$family" "$h" "causal" "initial_test_results.json")"
    a_causal_final="$(latest_file "$family" "$h" "causal" "test_results.json")"
    a_att_initial="$(latest_file "$family" "$h" "att" "initial_test_results.json")"
    a_att_final="$(latest_file "$family" "$h" "att" "test_results.json")"

    if [[ -n "$a_all_initial" ]]; then
      run_pair "$family" "$ref_file" "$h" "all_initial" "$a_all_initial" "$b_all_final"
    else
      echo "[skip] ${family} ${h}: missing A all_initial"
    fi
    if [[ -n "$a_causal_initial" ]]; then
      run_pair "$family" "$ref_file" "$h" "causal_initial" "$a_causal_initial" "$b_all_final"
    else
      echo "[skip] ${family} ${h}: missing A causal_initial"
    fi
    if [[ -n "$a_causal_final" ]]; then
      run_pair "$family" "$ref_file" "$h" "causal_final" "$a_causal_final" "$b_all_final"
    else
      echo "[skip] ${family} ${h}: missing A causal_final"
    fi
    if [[ -n "$a_att_initial" ]]; then
      run_pair "$family" "$ref_file" "$h" "att_initial" "$a_att_initial" "$b_all_final"
    else
      echo "[skip] ${family} ${h}: missing A att_initial"
    fi
    if [[ -n "$a_att_final" ]]; then
      run_pair "$family" "$ref_file" "$h" "att_final" "$a_att_final" "$b_all_final"
    else
      echo "[skip] ${family} ${h}: missing A att_final"
    fi
  done
}

echo "Using RESULTS_ROOT=${RESULTS_ROOT}"
echo "Using REF_ROOT=${REF_ROOT}"
echo "Using MODEL=${MODEL}"

run_family "Name_Mover_Head" "NMH.txt" 9.6 9.9 10.0
run_family "Negative_Name_Mover_Head" "NNMH.txt" 10.7 11.10
run_family "SIH" "SIH.txt" 7.9 8.6 8.10
run_family "DTH" "DTH.txt" 0.1 0.10 3.0

echo "✅ pair comparisons finished."
