#!/bin/bash
set -euo pipefail

REPO_ROOT="/data31/private/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/data31/private/wangziran/eap_auto/results/ioi_0307}"
REF_ROOT="${REF_ROOT:-/data31/private/wangziran/eap_auto/results/ioi_0126/answer}"
PYTHON_BIN="${PYTHON_BIN:-/home/wangziran/miniconda3/envs/eap-ig/bin/python}"
MODEL="${MODEL:-claude-sonnet-4-20250514-thinking}"
API_KEY="${API_KEY:-sk-cdENMjpwVIpdd1Iv0auFiHizYdgnWM0ZFKhHN3UBYqKIoqpA}"
OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/ab_bias_check_0307_sample}"

DTH_RUN="${RESULTS_ROOT}/hypothesis/DTH/0.1_20260306_1614_all"
SIH_RUN="${RESULTS_ROOT}/hypothesis/SIH/8.6_20260306_1614_all"

mkdir -p "${OUT_ROOT}/DTH_0.1" "${OUT_ROOT}/SIH_8.6"

# 从 iteration_1 抽取 initial hypothesis 文本
"$PYTHON_BIN" - <<'PY'
import json, pathlib
pairs = [
    ("/data31/private/wangziran/eap_auto/results/ioi_0307/hypothesis/DTH/0.1_20260306_1614_all/iteration_results/iteration_1.json",
     "/data31/private/wangziran/eap_auto/results/ioi_0307/ab_bias_check_0307_sample/DTH_0.1/initial_hypothesis.txt"),
    ("/data31/private/wangziran/eap_auto/results/ioi_0307/hypothesis/SIH/8.6_20260306_1614_all/iteration_results/iteration_1.json",
     "/data31/private/wangziran/eap_auto/results/ioi_0307/ab_bias_check_0307_sample/SIH_8.6/initial_hypothesis.txt"),
]
for src, dst in pairs:
    d = json.loads(pathlib.Path(src).read_text(encoding="utf-8"))
    txt = (d.get("hypothesis_before") or d.get("hypothesis") or "").strip()
    pathlib.Path(dst).write_text(txt, encoding="utf-8")
    print(f"[ok] wrote initial: {dst} (len={len(txt)})")
PY

echo "[run] DTH 0.1 bias check (forward=5, reverse=5)"
"$PYTHON_BIN" "${REPO_ROOT}/tests/experiments/check_pair_position_bias.py" \
  --reference "${REF_ROOT}/DTH.txt" \
  --candidate-a "${OUT_ROOT}/DTH_0.1/initial_hypothesis.txt" \
  --candidate-b "${DTH_RUN}/final_hypothesis.json" \
  --label "DTH_0.1" \
  --forward-runs 5 \
  --reverse-runs 5 \
  --output-dir "${OUT_ROOT}/DTH_0.1" \
  --model "${MODEL}" \
  --api-key "${API_KEY}"

echo "[run] SIH 8.6 bias check (forward=5, reverse=5)"
"$PYTHON_BIN" "${REPO_ROOT}/tests/experiments/check_pair_position_bias.py" \
  --reference "${REF_ROOT}/SIH.txt" \
  --candidate-a "${OUT_ROOT}/SIH_8.6/initial_hypothesis.txt" \
  --candidate-b "${SIH_RUN}/final_hypothesis.json" \
  --label "SIH_8.6" \
  --forward-runs 5 \
  --reverse-runs 5 \
  --output-dir "${OUT_ROOT}/SIH_8.6" \
  --model "${MODEL}" \
  --api-key "${API_KEY}"

echo "✅ Done. Outputs under: ${OUT_ROOT}"
