#!/bin/bash
set -euo pipefail

REPO_ROOT="/data31/private/wangziran/eap_auto"
RESULTS_ROOT="${RESULTS_ROOT:-/data31/private/wangziran/eap_auto/results/ioi_0304}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/data31/private/wangziran/eap_auto/results/ioi_0126/answer}"

python "$REPO_ROOT/tests/experiments/eval_initial_hypothesis_on_test.py" \
  --results-root "$RESULTS_ROOT"

python "$REPO_ROOT/tests/experiments/export_ioi_comparison_table.py" \
  --results-root "$RESULTS_ROOT" \
  --reference-root "$REFERENCE_ROOT"

echo "✅ initial test evaluation + table export done."
