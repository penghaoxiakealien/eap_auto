#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/home/wangziran/eap_auto/results/ioi_0305}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/home/wangziran/eap_auto/results/ioi_0126/answer}"

python "$REPO_ROOT/tests/experiments/eval_initial_hypothesis_on_test.py" \
  --results-root "$RESULTS_ROOT" \
  --families "DTH"

# 导出一次总表（包含 initial 分数）
python "$REPO_ROOT/tests/experiments/export_ioi_comparison_table.py" \
  --results-root "$RESULTS_ROOT" \
  --reference-root "$REFERENCE_ROOT"

echo "✅ Batch3 done (DTH initial test eval + export table)."
