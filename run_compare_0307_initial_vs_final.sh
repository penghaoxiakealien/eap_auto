#!/bin/bash
set -euo pipefail

REPO_ROOT="/data31/private/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/data31/private/wangziran/eap_auto/results/ioi_0307}"
REF_ROOT="${REF_ROOT:-/data31/private/wangziran/eap_auto/results/ioi_0126/answer}"
PYTHON_BIN="${PYTHON_BIN:-/home/wangziran/miniconda3/envs/eap-ig/bin/python}"
MODEL="${MODEL:-claude-sonnet-4-20250514-thinking}"
API_KEY="${API_KEY:-sk-cdENMjpwVIpdd1Iv0auFiHizYdgnWM0ZFKhHN3UBYqKIoqpA}"

OUT_ROOT="${RESULTS_ROOT}/compare_initial_vs_final"
mkdir -p "$OUT_ROOT"

run_one() {
  local family="$1"
  local head="$2"
  local ref_file="$3"

  local run_dir
  run_dir=$(ls -1dt "${RESULTS_ROOT}/hypothesis/${family}/${head}_"*_all 2>/dev/null | head -n 1 || true)
  if [[ -z "$run_dir" ]]; then
    echo "[skip] ${family} ${head}: no _all run dir"
    return 0
  fi

  local iter1="${run_dir}/iteration_results/iteration_1.json"
  local test_file="${run_dir}/test_results.json"
  local ref_path="${REF_ROOT}/${ref_file}"
  if [[ ! -f "$iter1" || ! -f "$test_file" || ! -f "$ref_path" ]]; then
    echo "[skip] ${family} ${head}: missing iter1/test/reference"
    return 0
  fi

  local out_dir="${OUT_ROOT}/${family}/${head}"
  mkdir -p "$out_dir"
  local initial_txt="${out_dir}/initial_hypothesis.txt"
  local output_json="${out_dir}/initial_vs_final_pair.json"

  "$PYTHON_BIN" - <<PY
import json, pathlib
src = pathlib.Path("${iter1}")
dst = pathlib.Path("${initial_txt}")
d = json.loads(src.read_text(encoding="utf-8"))
txt = (d.get("hypothesis_before") or d.get("hypothesis") or "").strip()
dst.write_text(txt, encoding="utf-8")
print(f"[ok] initial extracted: {dst} len={len(txt)}")
PY

  echo "[run] ${family} ${head}: A=initial, B=final(test_results)"
  "$PYTHON_BIN" "${REPO_ROOT}/tests/experiments/compare_hypotheses_to_references.py" \
    --pair-reference-path "$ref_path" \
    --pair-label "${family}_${head}" \
    --candidate-a "$initial_txt" \
    --candidate-b "$test_file" \
    --output "$output_json" \
    --model "$MODEL" \
    --api-key "$API_KEY"
}

echo "=== compare initial vs final on ioi_0307 ==="
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "REF_ROOT=${REF_ROOT}"

run_one "Name_Mover_Head" "9.6" "NMH.txt"
run_one "Name_Mover_Head" "9.9" "NMH.txt"
run_one "Name_Mover_Head" "10.0" "NMH.txt"

run_one "Negative_Name_Mover_Head" "10.7" "NNMH.txt"
run_one "Negative_Name_Mover_Head" "11.10" "NNMH.txt"

run_one "SIH" "7.9" "SIH.txt"
run_one "SIH" "8.6" "SIH.txt"
run_one "SIH" "8.10" "SIH.txt"

run_one "DTH" "0.1" "DTH.txt"
run_one "DTH" "0.10" "DTH.txt"
run_one "DTH" "3.0" "DTH.txt"

echo "✅ Done. Outputs under: ${OUT_ROOT}"
